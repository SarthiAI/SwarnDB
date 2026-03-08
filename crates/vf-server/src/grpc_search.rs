// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;
use std::time::Instant;

use tonic::{Request, Response, Status};

use vf_core::types::{Metadata, MetadataValue, VectorId};
use vf_graph::RelationshipQueryEngine;
use vf_query::{BatchExecutor, FilterExpression, FilterStrategy, QueryExecutor};

use crate::proto::swarndb::v1::search_service_server::SearchService;
use crate::proto::swarndb::v1::{
    self as proto, BatchSearchRequest, BatchSearchResponse, SearchRequest, SearchResponse,
};
use crate::state::AppState;

pub struct SearchServiceImpl {
    state: AppState,
}

impl SearchServiceImpl {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl SearchService for SearchServiceImpl {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let query_vector = req
            .query
            .as_ref()
            .map(|v| v.values.clone())
            .ok_or_else(|| Status::invalid_argument("query vector is required"))?;

        let filter = req
            .filter
            .as_ref()
            .map(convert_filter)
            .transpose()?;

        let strategy = parse_strategy(&req.strategy);

        let metadata_store = collection.metadata_cache.get_or_rebuild(&collection.store);

        let results = QueryExecutor::search(
            &collection.index as &dyn vf_index::traits::VectorIndex,
            &query_vector,
            req.k as usize,
            filter.as_ref(),
            &strategy,
            Some(&collection.index_manager),
            &metadata_store,
        )
        .map_err(|e| Status::internal(format!("search error: {}", e)))?;

        let max_edges = if req.max_graph_edges == 0 { 10u32 } else { req.max_graph_edges };
        let graph_threshold = if req.graph_threshold == 0.0 { None } else { Some(req.graph_threshold) };

        // Batch lookup all graph edges at once instead of per-result
        let graph_edges_map = if req.include_graph {
            let ids: Vec<VectorId> = results.iter().map(|r| r.id).collect();
            RelationshipQueryEngine::get_related_batch(&collection.graph, &ids, graph_threshold, max_edges)
        } else {
            HashMap::new()
        };

        let scored_results: Vec<proto::ScoredResult> = results
            .into_iter()
            .map(|r| {
                let metadata = if req.include_metadata {
                    metadata_store.get(&r.id).map(|m| convert_metadata_to_proto(m))
                } else {
                    None
                };
                let graph_edges = graph_edges_map
                    .get(&r.id)
                    .map(|edges| {
                        edges.iter().map(|&(target_id, similarity)| proto::RelatedEdge {
                            target_id,
                            similarity,
                        }).collect()
                    })
                    .unwrap_or_default();
                proto::ScoredResult {
                    id: r.id,
                    score: r.score,
                    metadata,
                    graph_edges,
                }
            })
            .collect();

        let search_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(SearchResponse {
            results: scored_results,
            search_time_us,
        }))
    }

    async fn batch_search(
        &self,
        request: Request<BatchSearchRequest>,
    ) -> Result<Response<BatchSearchResponse>, Status> {
        let timer = Instant::now();
        let req = request.into_inner();

        if req.queries.is_empty() {
            return Ok(Response::new(BatchSearchResponse {
                results: vec![],
                total_time_us: 0,
            }));
        }

        // Check if all queries target the same collection for batch optimization
        let first_collection = &req.queries[0].collection;
        let all_same_collection = req.queries.iter().all(|q| q.collection == *first_collection);

        let results = if all_same_collection {
            self.batch_search_same_collection(&req.queries).await?
        } else {
            self.batch_search_mixed_collections(&req.queries).await?
        };

        let total_time_us = timer.elapsed().as_micros() as u64;

        Ok(Response::new(BatchSearchResponse {
            results,
            total_time_us,
        }))
    }
}

impl SearchServiceImpl {
    async fn batch_search_same_collection(
        &self,
        queries: &[SearchRequest],
    ) -> Result<Vec<SearchResponse>, Status> {
        let collection_name = &queries[0].collection;
        let collections = self.state.collections.read();
        let collection = collections
            .get(collection_name)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", collection_name)))?;

        // Check if all queries share the same filter and strategy (uniform batch)
        let first_filter = queries[0]
            .filter
            .as_ref()
            .map(convert_filter)
            .transpose()?;
        let first_strategy = parse_strategy(&queries[0].strategy);
        let first_k = queries[0].k;
        let include_metadata = queries[0].include_metadata;

        let all_uniform = queries.iter().skip(1).all(|q| {
            q.k == first_k && q.strategy == queries[0].strategy && q.filter == queries[0].filter
        });

        let metadata_store = collection.metadata_cache.get_or_rebuild(&collection.store);

        if all_uniform {
            let query_vectors: Vec<Vec<f32>> = queries
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let values = q
                        .query
                        .as_ref()
                        .map(|v| v.values.clone())
                        .unwrap_or_default();
                    if values.is_empty() {
                        return Err(Status::invalid_argument(format!(
                            "query vector at index {} is missing or empty", i
                        )));
                    }
                    Ok(values)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let batch_results = BatchExecutor::search_batch_uniform(
                &collection.index as &dyn vf_index::traits::VectorIndex,
                &query_vectors,
                first_k as usize,
                first_filter.as_ref(),
                &first_strategy,
                Some(&collection.index_manager),
                &metadata_store,
            );

            let include_graph = queries[0].include_graph;
            let max_e = if queries[0].max_graph_edges == 0 { 10u32 } else { queries[0].max_graph_edges };
            let g_thresh = if queries[0].graph_threshold == 0.0 { None } else { Some(queries[0].graph_threshold) };

            batch_results
                .into_iter()
                .map(|r| {
                    let results = r.map_err(|e| Status::internal(format!("search error: {}", e)))?;
                    let graph_edges_map = if include_graph {
                        let ids: Vec<VectorId> = results.iter().map(|r| r.id).collect();
                        RelationshipQueryEngine::get_related_batch(&collection.graph, &ids, g_thresh, max_e)
                    } else {
                        HashMap::new()
                    };
                    Ok(to_search_response(results, &metadata_store, include_metadata, &graph_edges_map))
                })
                .collect()
        } else {
            // Non-uniform: execute each query individually
            let mut responses = Vec::with_capacity(queries.len());
            for q in queries {
                let query_vector = q
                    .query
                    .as_ref()
                    .map(|v| v.values.clone())
                    .ok_or_else(|| Status::invalid_argument("query vector is required"))?;
                let filter = q.filter.as_ref().map(convert_filter).transpose()?;
                let strategy = parse_strategy(&q.strategy);

                let results = QueryExecutor::search(
                    &collection.index as &dyn vf_index::traits::VectorIndex,
                    &query_vector,
                    q.k as usize,
                    filter.as_ref(),
                    &strategy,
                    Some(&collection.index_manager),
                    &metadata_store,
                )
                .map_err(|e| Status::internal(format!("search error: {}", e)))?;

                let graph_edges_map = if q.include_graph {
                    let max_e = if q.max_graph_edges == 0 { 10u32 } else { q.max_graph_edges };
                    let g_thresh = if q.graph_threshold == 0.0 { None } else { Some(q.graph_threshold) };
                    let ids: Vec<VectorId> = results.iter().map(|r| r.id).collect();
                    RelationshipQueryEngine::get_related_batch(&collection.graph, &ids, g_thresh, max_e)
                } else {
                    HashMap::new()
                };
                responses.push(to_search_response(results, &metadata_store, q.include_metadata, &graph_edges_map));
            }
            Ok(responses)
        }
    }

    async fn batch_search_mixed_collections(
        &self,
        queries: &[SearchRequest],
    ) -> Result<Vec<SearchResponse>, Status> {
        let collections = self.state.collections.read();
        let mut responses = Vec::with_capacity(queries.len());

        for q in queries {
            let collection = collections
                .get(&q.collection)
                .ok_or_else(|| Status::not_found(format!("collection '{}' not found", q.collection)))?;

            let query_vector = q
                .query
                .as_ref()
                .map(|v| v.values.clone())
                .ok_or_else(|| Status::invalid_argument("query vector is required"))?;
            let filter = q.filter.as_ref().map(convert_filter).transpose()?;
            let strategy = parse_strategy(&q.strategy);
            let metadata_store = collection.metadata_cache.get_or_rebuild(&collection.store);

            let results = QueryExecutor::search(
                &collection.index as &dyn vf_index::traits::VectorIndex,
                &query_vector,
                q.k as usize,
                filter.as_ref(),
                &strategy,
                Some(&collection.index_manager),
                &metadata_store,
            )
            .map_err(|e| Status::internal(format!("search error: {}", e)))?;

            let graph_edges_map = if q.include_graph {
                let max_e = if q.max_graph_edges == 0 { 10u32 } else { q.max_graph_edges };
                let g_thresh = if q.graph_threshold == 0.0 { None } else { Some(q.graph_threshold) };
                let ids: Vec<VectorId> = results.iter().map(|r| r.id).collect();
                RelationshipQueryEngine::get_related_batch(&collection.graph, &ids, g_thresh, max_e)
            } else {
                HashMap::new()
            };
            responses.push(to_search_response(results, &metadata_store, q.include_metadata, &graph_edges_map));
        }

        Ok(responses)
    }
}

// ── Helper functions ────────────────────────────────────────────────────

fn parse_strategy(s: &str) -> FilterStrategy {
    match s {
        "pre_filter" => FilterStrategy::PreFilter,
        "post_filter" => FilterStrategy::PostFilter { oversample_factor: 3 },
        _ => FilterStrategy::Auto,
    }
}

fn to_search_response(
    results: Vec<vf_core::types::ScoredResult>,
    metadata_store: &HashMap<VectorId, Metadata>,
    include_metadata: bool,
    graph_edges_map: &HashMap<VectorId, Vec<(VectorId, f32)>>,
) -> SearchResponse {
    let scored_results = results
        .into_iter()
        .map(|r| {
            let metadata = if include_metadata {
                metadata_store.get(&r.id).map(|m| convert_metadata_to_proto(m))
            } else {
                None
            };
            let graph_edges = graph_edges_map
                .get(&r.id)
                .map(|edges| {
                    edges.iter().map(|&(target_id, similarity)| proto::RelatedEdge {
                        target_id,
                        similarity,
                    }).collect()
                })
                .unwrap_or_default();
            proto::ScoredResult {
                id: r.id,
                score: r.score,
                metadata,
                graph_edges,
            }
        })
        .collect();

    SearchResponse {
        results: scored_results,
        search_time_us: 0, // individual timing not tracked in batch
    }
}

fn convert_metadata_to_proto(metadata: &Metadata) -> proto::Metadata {
    let fields = metadata
        .iter()
        .map(|(k, v)| {
            let proto_value = match v {
                MetadataValue::String(s) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::StringValue(s.clone())),
                },
                MetadataValue::Int(i) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::IntValue(*i)),
                },
                MetadataValue::Float(f) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::FloatValue(*f)),
                },
                MetadataValue::Bool(b) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::BoolValue(*b)),
                },
                MetadataValue::StringList(list) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::StringListValue(
                        proto::StringList {
                            values: list.clone(),
                        },
                    )),
                },
            };
            (k.clone(), proto_value)
        })
        .collect();

    proto::Metadata { fields }
}

fn convert_proto_metadata_value(value: &proto::MetadataValue) -> Result<MetadataValue, Status> {
    match &value.value {
        Some(proto::metadata_value::Value::StringValue(s)) => Ok(MetadataValue::String(s.clone())),
        Some(proto::metadata_value::Value::IntValue(i)) => Ok(MetadataValue::Int(*i)),
        Some(proto::metadata_value::Value::FloatValue(f)) => Ok(MetadataValue::Float(*f)),
        Some(proto::metadata_value::Value::BoolValue(b)) => Ok(MetadataValue::Bool(*b)),
        Some(proto::metadata_value::Value::StringListValue(list)) => {
            Ok(MetadataValue::StringList(list.values.clone()))
        }
        None => Err(Status::invalid_argument("metadata value is empty")),
    }
}

fn convert_filter(proto_filter: &proto::FilterExpression) -> Result<FilterExpression, Status> {
    let filter = proto_filter
        .filter
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("filter expression is empty"))?;

    match filter {
        proto::filter_expression::Filter::And(and_filter) => {
            let children: Vec<FilterExpression> = and_filter
                .filters
                .iter()
                .map(convert_filter)
                .collect::<Result<_, _>>()?;
            Ok(FilterExpression::And(children))
        }
        proto::filter_expression::Filter::Or(or_filter) => {
            let children: Vec<FilterExpression> = or_filter
                .filters
                .iter()
                .map(convert_filter)
                .collect::<Result<_, _>>()?;
            Ok(FilterExpression::Or(children))
        }
        proto::filter_expression::Filter::Not(not_filter) => {
            let inner = not_filter
                .filter
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("NOT filter has no inner expression"))?;
            let child = convert_filter(inner)?;
            Ok(FilterExpression::Not(Box::new(child)))
        }
        proto::filter_expression::Filter::Field(field_filter) => {
            convert_field_filter(field_filter)
        }
    }
}

fn convert_field_filter(f: &proto::FieldFilter) -> Result<FilterExpression, Status> {
    let field = &f.field;

    match f.op.as_str() {
        "eq" => {
            let value = f
                .value
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("'eq' requires a value"))?;
            Ok(FilterExpression::Eq(
                field.clone(),
                convert_proto_metadata_value(value)?,
            ))
        }
        "ne" => {
            let value = f
                .value
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("'ne' requires a value"))?;
            Ok(FilterExpression::Ne(
                field.clone(),
                convert_proto_metadata_value(value)?,
            ))
        }
        "gt" => {
            let value = f
                .value
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("'gt' requires a value"))?;
            Ok(FilterExpression::Gt(
                field.clone(),
                convert_proto_metadata_value(value)?,
            ))
        }
        "gte" => {
            let value = f
                .value
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("'gte' requires a value"))?;
            Ok(FilterExpression::Gte(
                field.clone(),
                convert_proto_metadata_value(value)?,
            ))
        }
        "lt" => {
            let value = f
                .value
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("'lt' requires a value"))?;
            Ok(FilterExpression::Lt(
                field.clone(),
                convert_proto_metadata_value(value)?,
            ))
        }
        "lte" => {
            let value = f
                .value
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("'lte' requires a value"))?;
            Ok(FilterExpression::Lte(
                field.clone(),
                convert_proto_metadata_value(value)?,
            ))
        }
        "in" => {
            if f.values.is_empty() {
                return Err(Status::invalid_argument("'in' requires at least one value"));
            }
            let values: Vec<MetadataValue> = f
                .values
                .iter()
                .map(convert_proto_metadata_value)
                .collect::<Result<_, _>>()?;
            Ok(FilterExpression::In(field.clone(), values))
        }
        "between" => {
            if f.values.len() != 2 {
                return Err(Status::invalid_argument(
                    "'between' requires exactly 2 values",
                ));
            }
            let low = convert_proto_metadata_value(&f.values[0])?;
            let high = convert_proto_metadata_value(&f.values[1])?;
            Ok(FilterExpression::Between(field.clone(), low, high))
        }
        "exists" => Ok(FilterExpression::Exists(field.clone())),
        "contains" => {
            let value = f
                .value
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("'contains' requires a value"))?;
            match &value.value {
                Some(proto::metadata_value::Value::StringValue(s)) => {
                    Ok(FilterExpression::Contains(field.clone(), s.clone()))
                }
                _ => Err(Status::invalid_argument(
                    "'contains' requires a string value",
                )),
            }
        }
        other => Err(Status::invalid_argument(format!(
            "unsupported filter op: '{}'",
            other
        ))),
    }
}
