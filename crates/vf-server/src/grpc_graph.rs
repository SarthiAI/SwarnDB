// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use tonic::{Request, Response, Status};

use vf_graph::{
    EdgeDirection, EdgeId, EdgePageFilter, GraphStore, GraphTraversal, NodeId, NodeKind,
    NodePageFilter, NodeSource, Provenance, RelationshipQueryEngine, TraversalOrder,
};

use vf_extraction::RejectRule;

use crate::edge_ops;
use crate::proto::swarndb::v1::graph_service_server::GraphService;
use crate::proto::swarndb::v1::{
    BulkImportEdgesRequest, BulkImportEdgesResponse, BulkImportFormat, BulkImportRowError,
    DeleteEdgeRequest, DeleteEdgeResponse, DeleteNodeRequest, DeleteNodeResponse, EdgeAudit,
    EnumerateEdgesRequest, EnumerateEdgesResponse, EnumerateNodesRequest, EnumerateNodesResponse,
    GetEdgeRequest, GetEdgeResponse, GetNodeRequest, GetNodeResponse, GetRelatedRequest,
    GetRelatedResponse, GraphEdge, HybridCompare, HybridDirection, HybridInList, HybridPath,
    HybridPredicate, HybridPropertyRef, HybridQueryPlan, HybridQueryRequest, HybridQueryResponse,
    HybridReturnKind, HybridStep, HybridTemporalFilter, HybridVectorMath, ListEdgesRequest,
    ListEdgesResponse, NodeAudit, PutEdgeRequest,
    PutEdgeResponse, PutNodeRequest, PutNodeResponse, RejectEdgeRequest, RejectEdgeResponse,
    RrfRankSpec, SetThresholdRequest, SetThresholdResponse, TraversalNode, TraverseRequest,
    TraverseResponse, TypedEdge, TypedNode, TypedNodeKind, UpdateEdgeRequest, UpdateEdgeResponse,
    UpdateNodeRequest, UpdateNodeResponse, VerifyEdgeRequest, VerifyEdgeResponse,
};
use vf_query::hybrid::{
    CompareOp as HCompareOp, HybridExecutor, NodeRecord, OnMissingVector as HOnMissingVector,
    Predicate as HPredicate, PropertyRef as HPropertyRef, QueryPlan as HQueryPlan, QueryResult,
    ReturnKind, RrfRankSpec as HRrfRankSpec, Step as HStep, TemporalFilter, VectorMathOp,
};
use crate::state::{metered_read, metered_write, AppState, CollectionAvailability};

fn status_from_availability(avail: CollectionAvailability) -> Status {
    match avail {
        CollectionAvailability::Recovering { .. } => Status::unavailable(avail.user_message()),
        CollectionAvailability::NotFound { .. } => Status::not_found(avail.user_message()),
    }
}

// ── Typed graph helpers ──────────────────────────────────────────────

/// Default page size when a client passes 0 for the enumeration limit.
const ENUM_DEFAULT_LIMIT: u32 = 1_000;
/// Hard server cap on an enumeration page so one call stays memory-bounded.
const ENUM_MAX_LIMIT: u32 = 10_000;

/// Clamp a client-supplied enumeration limit into the allowed page range.
fn clamp_enum_limit(limit: u32) -> usize {
    if limit == 0 {
        ENUM_DEFAULT_LIMIT as usize
    } else {
        limit.min(ENUM_MAX_LIMIT) as usize
    }
}

fn parse_props(s: &str) -> Result<HashMap<String, serde_json::Value>, Status> {
    if s.trim().is_empty() {
        return Ok(HashMap::new());
    }
    serde_json::from_str(s)
        .map_err(|e| Status::invalid_argument(format!("invalid properties_json: {e}")))
}

fn parse_provenance(s: &str) -> Result<Provenance, Status> {
    if s.trim().is_empty() {
        return Ok(Provenance::default());
    }
    serde_json::from_str(s)
        .map_err(|e| Status::invalid_argument(format!("invalid provenance_json: {e}")))
}

fn parse_node_source(s: &str) -> NodeSource {
    match s {
        "ingested" => NodeSource::Ingested,
        "extracted" => NodeSource::Extracted,
        _ => NodeSource::Manual,
    }
}

fn parse_direction(s: &str) -> EdgeDirection {
    match s {
        "incoming" => EdgeDirection::Incoming,
        "both" => EdgeDirection::Both,
        _ => EdgeDirection::Outgoing,
    }
}

fn node_to_proto(n: &vf_graph::model::Node) -> TypedNode {
    let (kind, label) = match &n.kind {
        NodeKind::Content => (TypedNodeKind::TypedNodeContent as i32, String::new()),
        NodeKind::Entity { label } => (TypedNodeKind::TypedNodeEntity as i32, label.clone()),
    };
    TypedNode {
        id: n.id.0,
        kind,
        label,
        properties_json: serde_json::to_string(&n.properties)
            .unwrap_or_else(|_| "{}".to_string()),
        embedding: n.embedding.clone().unwrap_or_default(),
        source: match n.source {
            NodeSource::Manual => "manual",
            NodeSource::Ingested => "ingested",
            NodeSource::Extracted => "extracted",
        }
        .to_string(),
        created_at: n.created_at,
        created_by: n.created_by.clone().unwrap_or_default(),
        history: n.history.iter().map(node_audit_to_proto).collect(),
        updated_at: n.updated_at.unwrap_or_default(),
    }
}

fn node_audit_to_proto(a: &vf_graph::NodeAudit) -> NodeAudit {
    NodeAudit {
        action: a.action.clone(),
        actor: a.actor.clone().unwrap_or_default(),
        at: a.at,
    }
}

fn edge_to_proto(e: &vf_graph::model::Edge) -> TypedEdge {
    TypedEdge {
        id: e.id.0,
        source: e.source.0,
        target: e.target.0,
        edge_type: e.edge_type.as_str().to_string(),
        properties_json: serde_json::to_string(&e.properties)
            .unwrap_or_else(|_| "{}".to_string()),
        provenance_json: serde_json::to_string(&e.provenance)
            .unwrap_or_else(|_| "{}".to_string()),
        confidence: e.confidence,
        verified: e.verified,
        is_manual: e.is_manual,
        created_at: e.created_at,
        history: e.history.iter().map(audit_to_proto).collect(),
        valid_from: e.valid_from,
        valid_until: e.valid_until,
        temporal_context: e.temporal_context.clone(),
    }
}

fn audit_to_proto(a: &vf_graph::EdgeAudit) -> EdgeAudit {
    EdgeAudit {
        action: a.action.clone(),
        actor: a.actor.clone().unwrap_or_default(),
        at: a.at,
    }
}

// Empty actor string means no actor recorded.
fn actor_opt(s: &str) -> Option<String> {
    if s.trim().is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

// Best-effort display name for a node: its "name" property if present.
fn node_name(store: &vf_graph::TypedGraphStore, id: NodeId) -> Option<String> {
    store
        .get_node(id)
        .and_then(|n| n.properties.get("name").and_then(|v| v.as_str()).map(|s| s.to_string()))
}

// ── Hybrid query proto -> domain conversion (P02) ────────────────────────

fn proto_direction_to_domain(d: i32) -> EdgeDirection {
    match HybridDirection::try_from(d).unwrap_or(HybridDirection::HybridDirOutgoing) {
        HybridDirection::HybridDirIncoming => EdgeDirection::Incoming,
        HybridDirection::HybridDirBoth => EdgeDirection::Both,
        HybridDirection::HybridDirOutgoing => EdgeDirection::Outgoing,
    }
}

fn proto_return_kind_to_domain(k: i32) -> ReturnKind {
    match HybridReturnKind::try_from(k).unwrap_or(HybridReturnKind::HybridReturnNodes) {
        HybridReturnKind::HybridReturnEdges => ReturnKind::Edges,
        HybridReturnKind::HybridReturnPaths => ReturnKind::Paths,
        HybridReturnKind::HybridReturnNodes => ReturnKind::Nodes,
    }
}

// Empty edge_type means "no filter".
fn proto_edge_type(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

// proto HybridTemporalFilter -> domain TemporalFilter (P17). Absent message =
// no filter (None), so the executor keeps the byte-identical fast path. When
// present, the proto bool `include_unbounded` maps straight through (a wire
// caller must set it explicitly; see proto note). An empty context string maps
// to None (= context ignored), mirroring the proto_edge_type idiom.
fn proto_temporal_to_domain(t: Option<HybridTemporalFilter>) -> Option<TemporalFilter> {
    t.map(|t| TemporalFilter {
        as_of: t.as_of,
        include_unbounded: t.include_unbounded,
        context: t.context.filter(|c| !c.trim().is_empty()),
    })
}

// proto HybridOnMissingVector -> domain OnMissingVector. Unknown or SKIP value
// defaults to Skip (additive, byte-stable). Shared by VectorRank and VectorMath.
fn proto_on_missing_to_domain(on_missing: i32) -> HOnMissingVector {
    match crate::proto::swarndb::v1::HybridOnMissingVector::try_from(on_missing) {
        Ok(crate::proto::swarndb::v1::HybridOnMissingVector::HybridOnMissingError) => {
            HOnMissingVector::Error
        }
        _ => HOnMissingVector::Skip,
    }
}

// proto HybridVectorMath -> domain Step::VectorMath (P17). Maps the oneof op to
// the matching VectorMathOp variant; HybridIsolation.centroids (Vec<HybridVector>)
// flattens to Vec<Vec<f32>> via each wrapper's `values`.
fn proto_vector_math_to_domain(vm: HybridVectorMath) -> Result<HStep, Status> {
    use crate::proto::swarndb::v1::hybrid_vector_math::Op;
    let op = match vm.op {
        Some(Op::Analogy(a)) => VectorMathOp::Analogy {
            a: a.a,
            b: a.b,
            c: a.c,
        },
        Some(Op::Diversity(d)) => VectorMathOp::Diversity {
            query: d.query,
            lambda: d.lambda,
        },
        Some(Op::Cone(c)) => VectorMathOp::Cone {
            direction: c.direction,
            aperture_radians: c.aperture_radians,
        },
        Some(Op::Isolation(i)) => VectorMathOp::Isolation {
            centroids: i.centroids.into_iter().map(|v| v.values).collect(),
        },
        Some(Op::Centroid(_)) => VectorMathOp::Centroid {},
        Some(Op::Interpolate(i)) => VectorMathOp::Interpolate {
            a: i.a,
            b: i.b,
            t: i.t,
        },
        None => return Err(Status::invalid_argument("empty vector_math op")),
    };
    Ok(HStep::VectorMath {
        op,
        k: vm.k as usize,
        on_missing: proto_on_missing_to_domain(vm.on_missing),
    })
}

fn proto_property_ref_to_domain(p: HybridPropertyRef) -> Result<HPropertyRef, Status> {
    use crate::proto::swarndb::v1::hybrid_property_ref::Ref;
    match p.r#ref {
        Some(Ref::Property(key)) => Ok(HPropertyRef::Property(key)),
        Some(Ref::Label(_)) => Ok(HPropertyRef::Label),
        Some(Ref::Kind(_)) => Ok(HPropertyRef::Kind),
        Some(Ref::IncidentEdgeCount(c)) => Ok(HPropertyRef::IncidentEdgeCount {
            edge_type: proto_edge_type(c.edge_type),
            direction: proto_direction_to_domain(c.direction),
        }),
        None => Err(Status::invalid_argument("empty hybrid property ref")),
    }
}

// Scalar JSON literal carried as a string.
fn parse_value_json(s: &str) -> Result<serde_json::Value, Status> {
    serde_json::from_str::<serde_json::Value>(s)
        .map_err(|e| Status::invalid_argument(format!("invalid value_json '{s}': {e}")))
}

fn parse_values_json(values: Vec<String>) -> Result<Vec<serde_json::Value>, Status> {
    values.iter().map(|v| parse_value_json(v)).collect()
}

fn proto_compare_op_to_domain(op: i32) -> HCompareOp {
    use crate::proto::swarndb::v1::HybridCompareOp;
    match HybridCompareOp::try_from(op).unwrap_or(HybridCompareOp::HybridCmpEq) {
        HybridCompareOp::HybridCmpNe => HCompareOp::Ne,
        HybridCompareOp::HybridCmpLt => HCompareOp::Lt,
        HybridCompareOp::HybridCmpLe => HCompareOp::Le,
        HybridCompareOp::HybridCmpGt => HCompareOp::Gt,
        HybridCompareOp::HybridCmpGe => HCompareOp::Ge,
        HybridCompareOp::HybridCmpEq => HCompareOp::Eq,
    }
}

fn proto_compare_to_domain(c: HybridCompare) -> Result<HPredicate, Status> {
    let field = c
        .field
        .ok_or_else(|| Status::invalid_argument("hybrid compare missing field"))?;
    Ok(HPredicate::Compare {
        field: proto_property_ref_to_domain(field)?,
        op: proto_compare_op_to_domain(c.op),
        value: parse_value_json(&c.value_json)?,
    })
}

fn proto_in_list_to_domain(l: HybridInList) -> Result<(HPropertyRef, Vec<serde_json::Value>), Status> {
    let field = l
        .field
        .ok_or_else(|| Status::invalid_argument("hybrid in-list missing field"))?;
    Ok((
        proto_property_ref_to_domain(field)?,
        parse_values_json(l.values_json)?,
    ))
}

fn proto_predicate_to_domain(p: HybridPredicate) -> Result<HPredicate, Status> {
    use crate::proto::swarndb::v1::hybrid_predicate::Pred;
    match p.pred {
        Some(Pred::Compare(c)) => proto_compare_to_domain(c),
        Some(Pred::InList(l)) => {
            let (field, values) = proto_in_list_to_domain(l)?;
            Ok(HPredicate::In { field, values })
        }
        Some(Pred::NotInList(l)) => {
            let (field, values) = proto_in_list_to_domain(l)?;
            Ok(HPredicate::NotIn { field, values })
        }
        Some(Pred::Exists(r)) => Ok(HPredicate::Exists {
            field: proto_property_ref_to_domain(r)?,
        }),
        Some(Pred::And(list)) => {
            let preds = list
                .preds
                .into_iter()
                .map(proto_predicate_to_domain)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(HPredicate::And(preds))
        }
        Some(Pred::Or(list)) => {
            let preds = list
                .preds
                .into_iter()
                .map(proto_predicate_to_domain)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(HPredicate::Or(preds))
        }
        Some(Pred::Not(inner)) => Ok(HPredicate::Not(Box::new(proto_predicate_to_domain(
            *inner,
        )?))),
        Some(Pred::Always(_)) => Ok(HPredicate::Always),
        None => Err(Status::invalid_argument("empty hybrid predicate")),
    }
}

fn proto_step_to_domain(s: HybridStep) -> Result<HStep, Status> {
    use crate::proto::swarndb::v1::hybrid_step::Step as PStep;
    match s.step {
        Some(PStep::VectorSimilar(v)) => Ok(HStep::VectorSimilar {
            vector: v.vector,
            k: v.k as usize,
            ef_search: v.ef_search.map(|e| e as usize),
        }),
        Some(PStep::FromNodes(f)) => Ok(HStep::FromNodes {
            nodes: f.nodes.into_iter().map(NodeId).collect(),
        }),
        Some(PStep::ScanByFilter(sf)) => Ok(HStep::ScanByFilter {
            is_entity: if sf.filter_by_kind {
                Some(sf.kind == TypedNodeKind::TypedNodeEntity as i32)
            } else {
                None
            },
            label: if sf.label.trim().is_empty() {
                None
            } else {
                Some(sf.label)
            },
            predicate: proto_khop_predicate(sf.predicate)?,
        }),
        Some(PStep::Traverse(t)) => Ok(HStep::Traverse {
            edge_type: proto_edge_type(t.edge_type),
            direction: proto_direction_to_domain(t.direction),
            temporal: proto_temporal_to_domain(t.temporal),
        }),
        Some(PStep::KHop(k)) => Ok(HStep::KHop {
            edge_type: proto_edge_type(k.edge_type),
            max: k.max,
            predicate: proto_khop_predicate(k.predicate)?,
            weight: k.weight.as_ref().map(proto_weight_to_domain),
            order_by_weight: k.order_by_weight,
            temporal: proto_temporal_to_domain(k.temporal),
        }),
        Some(PStep::ShortestPath(sp)) => Ok(HStep::ShortestPath {
            edge_types: sp.edge_types,
            target: NodeId(sp.target),
            weighted: sp.weighted,
            weight: sp.weight.as_ref().map(proto_weight_to_domain),
            temporal: proto_temporal_to_domain(sp.temporal),
        }),
        Some(PStep::MutualNeighbors(p)) => Ok(HStep::MutualNeighbors {
            other: proto_plan_to_domain(p)?,
        }),
        Some(PStep::Intersect(p)) => Ok(HStep::Intersect {
            other: proto_plan_to_domain(p)?,
        }),
        Some(PStep::Union(p)) => Ok(HStep::Union {
            other: proto_plan_to_domain(p)?,
        }),
        Some(PStep::Filter(pred)) => Ok(HStep::Filter {
            predicate: proto_predicate_to_domain(pred)?,
        }),
        Some(PStep::CollectEdges(c)) => Ok(HStep::CollectEdges {
            edge_type: proto_edge_type(c.edge_type),
            direction: proto_direction_to_domain(c.direction),
        }),
        Some(PStep::Limit(n)) => Ok(HStep::Limit { n: n as usize }),
        Some(PStep::VectorRank(v)) => Ok(HStep::VectorRank {
            vector: v.vector,
            k: v.k as usize,
            on_missing: proto_on_missing_to_domain(v.on_missing),
        }),
        Some(PStep::VectorMath(vm)) => proto_vector_math_to_domain(vm),
        None => Err(Status::invalid_argument("empty hybrid step")),
    }
}

// KHop predicate is a non-boxed nested message; absent = None.
fn proto_khop_predicate(pred: Option<HybridPredicate>) -> Result<Option<HPredicate>, Status> {
    match pred {
        Some(p) => Ok(Some(proto_predicate_to_domain(p)?)),
        None => Ok(None),
    }
}

// proto HybridWeightSpec -> domain WeightParams (P13). Empty key falls back to
// the default "weight". Callers map the Option, so absent = no-op weighting.
fn proto_weight_to_domain(
    w: &crate::proto::swarndb::v1::HybridWeightSpec,
) -> vf_query::hybrid::WeightParams {
    vf_query::hybrid::WeightParams {
        use_confidence: w.use_confidence,
        min_confidence: w.min_confidence,
        recency_half_life_ms: w.recency_half_life_ms,
        use_explicit_weight: w.use_explicit_weight,
        explicit_weight_key: if w.explicit_weight_key.is_empty() {
            "weight".to_string()
        } else {
            w.explicit_weight_key.clone()
        },
    }
}

// Recursive for nested sub-query plans.
fn proto_plan_to_domain(p: HybridQueryPlan) -> Result<HQueryPlan, Status> {
    let return_kind = proto_return_kind_to_domain(p.return_kind);
    let steps = p
        .steps
        .into_iter()
        .map(proto_step_to_domain)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(HQueryPlan { steps, return_kind })
}

// True when an RRF spec carries no meaningful field (a wholly-unspecified spec).
// Per ADR-024 the default graph-augmented path is vector_rank, so RRF runs only
// on an EXPLICIT request: a present-but-empty spec resolves to the default path,
// not RRF. The documented rank_rrf builder always sets k, so a real opt-in is
// never mistaken for empty.
fn rrf_spec_is_empty(s: &RrfRankSpec) -> bool {
    s.k <= 0
        && s.rrf_k <= 0
        && s.k_hop_max <= 0
        && s.relation_edge_types.is_empty()
        && s.hub_damping <= 0.0
}

// Optional graph-aware RRF ranking spec (P07). Absent on the request = OFF.
// A non-positive k means "no cut"; rrf_k / k_hop_max defaults are resolved in
// the executor so the canonical 60 / 2 apply when the client sends 0.
fn proto_rrf_to_domain(s: RrfRankSpec) -> HRrfRankSpec {
    HRrfRankSpec {
        k: if s.k > 0 { Some(s.k as usize) } else { None },
        rrf_k: if s.rrf_k > 0 { s.rrf_k as u32 } else { 0 },
        k_hop_max: if s.k_hop_max > 0 { s.k_hop_max as u32 } else { 0 },
        relation_edge_types: s.relation_edge_types,
        // Hub-aware damping (ADR-019). 0.0 = OFF = unchanged route counting; the
        // client sends 0.0 when the field is unset, so the default is byte-stable.
        hub_damping: s.hub_damping.max(0.0),
        // Edge quality weighting (P13). Absent = unweighted route counting.
        edge_weight: s.edge_weight.as_ref().map(proto_weight_to_domain),
    }
}

// Virtual proto node for an unmaterialized vector hit (store had no record).
fn virtual_content_node(id: u64) -> TypedNode {
    TypedNode {
        id,
        kind: TypedNodeKind::TypedNodeContent as i32,
        label: String::new(),
        properties_json: String::new(),
        embedding: Vec::new(),
        source: String::new(),
        created_at: 0,
        created_by: String::new(),
        history: Vec::new(),
        updated_at: 0,
    }
}

pub struct GraphServiceImpl {
    state: AppState,
}

impl GraphServiceImpl {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl GraphService for GraphServiceImpl {
    async fn get_related(
        &self,
        request: Request<GetRelatedRequest>,
    ) -> Result<Response<GetRelatedResponse>, Status> {
        let req = request.into_inner();

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let coll_handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let collection = metered_read(&coll_handle);

        if collection.config.is_vector_only() {
            return Err(Status::failed_precondition(format!(
                "collection '{}' is in vector-only mode; graph queries are not available",
                req.collection
            )));
        }

        let threshold = if req.threshold > 0.0 {
            Some(req.threshold)
        } else {
            None
        };

        let related = RelationshipQueryEngine::get_related(
            &collection.graph,
            req.vector_id,
            threshold,
        )
        .map_err(|e| Status::internal(format!("graph error: {}", e)))?;

        let mut edges: Vec<GraphEdge> = related
            .into_iter()
            .map(|(target_id, similarity)| GraphEdge {
                target_id,
                similarity,
            })
            .collect();

        if req.max_results > 0 {
            edges.truncate(req.max_results as usize);
        }

        Ok(Response::new(GetRelatedResponse { edges }))
    }

    async fn traverse(
        &self,
        request: Request<TraverseRequest>,
    ) -> Result<Response<TraverseResponse>, Status> {
        let req = request.into_inner();

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let coll_handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let collection = metered_read(&coll_handle);

        if collection.config.is_vector_only() {
            return Err(Status::failed_precondition(format!(
                "collection '{}' is in vector-only mode; graph queries are not available",
                req.collection
            )));
        }

        let threshold = if req.threshold > 0.0 {
            Some(req.threshold)
        } else {
            None
        };

        let max_results = if req.max_results > 0 {
            Some(req.max_results as usize)
        } else {
            None
        };

        let traversal_results = GraphTraversal::traverse(
            &collection.graph,
            req.start_id,
            &TraversalOrder::BreadthFirst,
            req.depth as usize,
            threshold,
            max_results,
        )
        .map_err(|e| Status::internal(format!("traversal error: {}", e)))?;

        let nodes: Vec<TraversalNode> = traversal_results
            .into_iter()
            .map(|r| TraversalNode {
                id: r.id,
                depth: r.depth as u32,
                path_similarity: r.path_similarity,
                path: r.path,
            })
            .collect();

        Ok(Response::new(TraverseResponse { nodes }))
    }

    async fn set_threshold(
        &self,
        request: Request<SetThresholdRequest>,
    ) -> Result<Response<SetThresholdResponse>, Status> {
        let req = request.into_inner();

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let coll_handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut collection = metered_write(&coll_handle);

        if collection.config.is_vector_only() {
            return Err(Status::failed_precondition(format!(
                "collection '{}' is in vector-only mode; graph queries are not available",
                req.collection
            )));
        }

        if req.vector_id == 0 {
            // Update collection-level default threshold
            collection.graph.config_mut().default_threshold = req.threshold;
            collection.deferred_graph.store(true, Ordering::Release);
        } else {
            // Set per-vector threshold override
            collection
                .graph
                .set_vector_threshold(req.vector_id, req.threshold);
        }

        Ok(Response::new(SetThresholdResponse { success: true }))
    }

    // ── Typed graph (Hybrid mode), ADR-007 R4 ──

    async fn put_node(
        &self,
        request: Request<PutNodeRequest>,
    ) -> Result<Response<PutNodeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let kind = match req.kind() {
            TypedNodeKind::TypedNodeEntity => NodeKind::Entity {
                label: req.label.clone(),
            },
            _ => NodeKind::Content,
        };
        // Consistency guard: a content node with an embedding via put_node would be
        // stored inline but never indexed into HNSW, so it would not be searchable and
        // would break the NodeId==VectorId bridge. Searchable content vectors must go
        // through vectors.insert. Entity nodes may carry an inline embedding (graph-scoped
        // vector_rank), and content placeholders without an embedding remain allowed.
        if matches!(kind, NodeKind::Content) && !req.embedding.is_empty() {
            return Err(Status::invalid_argument(
                "put_node: a content node with an embedding is not searchable via put_node; \
                 create it with vectors.insert (which assigns the id, indexes the vector, and \
                 links the content node by the NodeId==VectorId bridge). put_node is for entity \
                 nodes (which may carry an inline embedding for graph-scoped ranking) and for \
                 content placeholders without an embedding.",
            ));
        }
        let properties = parse_props(&req.properties_json)?;
        let source = parse_node_source(&req.source);

        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&handle);
        let lsn = {
            let cm = self.state.collection_manager.read();
            cm.get_collection(&req.collection)
                .map(|c| c.current_lsn())
                .unwrap_or(0)
        };
        let coll_ref = &mut *coll;
        // Allocate the node id from the unified per-collection id authority (the
        // vector store's next_id) BEFORE taking the &mut graph_store borrow, so
        // entity ids can never collide with vector/content ids. The id is not
        // inserted into vectors, so the NodeId == VectorId bridge stays
        // content-only. This immutable borrow of `store` is released before the
        // mutable borrow of `graph_store` below.
        let id = vf_graph::NodeId(coll_ref.store.alloc_id());
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        let node = vf_graph::model::Node {
            id,
            kind,
            properties,
            embedding: if req.embedding.is_empty() {
                None
            } else {
                Some(req.embedding)
            },
            source,
            created_at: vf_graph::now_millis(),
            created_by: if req.created_by.is_empty() {
                None
            } else {
                Some(req.created_by)
            },
            updated_at: None,
            history: Vec::new(),
        };
        store
            .put_node(node, lsn)
            .map_err(|e| Status::internal(format!("put_node failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(PutNodeResponse { id: id.0 }))
    }

    async fn get_node(
        &self,
        request: Request<GetNodeRequest>,
    ) -> Result<Response<GetNodeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let coll = metered_read(&handle);
        let store = coll.graph_store.as_ref().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        match store.get_node(NodeId(req.id)) {
            Some(n) => Ok(Response::new(GetNodeResponse {
                found: true,
                node: Some(node_to_proto(&n)),
            })),
            None => Ok(Response::new(GetNodeResponse {
                found: false,
                node: None,
            })),
        }
    }

    async fn delete_node(
        &self,
        request: Request<DeleteNodeRequest>,
    ) -> Result<Response<DeleteNodeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&handle);
        let lsn = {
            let cm = self.state.collection_manager.read();
            cm.get_collection(&req.collection)
                .map(|c| c.current_lsn())
                .unwrap_or(0)
        };
        let coll_ref = &mut *coll;
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        let deleted = store
            .delete_node(NodeId(req.id), lsn)
            .map_err(|e| Status::internal(format!("delete_node failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(DeleteNodeResponse { deleted }))
    }

    async fn put_edge(
        &self,
        request: Request<PutEdgeRequest>,
    ) -> Result<Response<PutEdgeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        if req.edge_type.trim().is_empty() {
            return Err(Status::invalid_argument("edge_type must not be empty"));
        }
        let properties = parse_props(&req.properties_json)?;
        let provenance = parse_provenance(&req.provenance_json)?;
        let confidence = if req.confidence > 0.0 { req.confidence } else { 1.0 };

        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&handle);
        let lsn = {
            let cm = self.state.collection_manager.read();
            cm.get_collection(&req.collection)
                .map(|c| c.current_lsn())
                .unwrap_or(0)
        };
        let coll_ref = &mut *coll;
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        let edge_type = store.intern(&req.edge_type);
        let id = store.alloc_edge_id();
        let mut edge = vf_graph::model::Edge {
            id,
            source: NodeId(req.source),
            target: NodeId(req.target),
            edge_type,
            properties,
            provenance,
            confidence,
            verified: req.verified,
            is_manual: req.is_manual,
            created_at: vf_graph::now_millis(),
            history: Vec::new(),
            valid_from: req.valid_from,
            valid_until: req.valid_until,
            // Empty-string context maps to None to honor the "absent = none" contract.
            temporal_context: req
                .temporal_context
                .filter(|c| !c.trim().is_empty()),
        };
        edge.record_audit("created", None, vf_graph::now_millis());
        store
            .put_edge(edge, lsn)
            .map_err(|e| Status::internal(format!("put_edge failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(PutEdgeResponse { id: id.0 }))
    }

    async fn get_edge(
        &self,
        request: Request<GetEdgeRequest>,
    ) -> Result<Response<GetEdgeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let coll = metered_read(&handle);
        let store = coll.graph_store.as_ref().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        match store.get_edge(vf_graph::EdgeId(req.id)) {
            Some(e) => Ok(Response::new(GetEdgeResponse {
                found: true,
                edge: Some(edge_to_proto(&e)),
            })),
            None => Ok(Response::new(GetEdgeResponse {
                found: false,
                edge: None,
            })),
        }
    }

    async fn delete_edge(
        &self,
        request: Request<DeleteEdgeRequest>,
    ) -> Result<Response<DeleteEdgeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&handle);
        let lsn = {
            let cm = self.state.collection_manager.read();
            cm.get_collection(&req.collection)
                .map(|c| c.current_lsn())
                .unwrap_or(0)
        };
        let coll_ref = &mut *coll;
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        let deleted = store
            .delete_edge(vf_graph::EdgeId(req.id), lsn)
            .map_err(|e| Status::internal(format!("delete_edge failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(DeleteEdgeResponse { deleted }))
    }

    async fn list_edges(
        &self,
        request: Request<ListEdgesRequest>,
    ) -> Result<Response<ListEdgesResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let coll = metered_read(&handle);
        let store = coll.graph_store.as_ref().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        let dir = parse_direction(&req.direction);
        let filter = if req.edge_type.trim().is_empty() {
            None
        } else {
            Some(req.edge_type.as_str())
        };
        let edges: Vec<TypedEdge> = store
            .edges_for_node(NodeId(req.node), dir)
            .into_iter()
            .filter(|e| filter.map(|t| e.edge_type.as_str() == t).unwrap_or(true))
            .map(|e| edge_to_proto(&e))
            .collect();
        Ok(Response::new(ListEdgesResponse { edges }))
    }

    // ── Paginated whole-graph enumeration (ADR-014) ──

    async fn enumerate_nodes(
        &self,
        request: Request<EnumerateNodesRequest>,
    ) -> Result<Response<EnumerateNodesResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let coll = metered_read(&handle);
        let store = coll.graph_store.as_ref().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;

        let limit = clamp_enum_limit(req.limit);
        let mut filter = NodePageFilter::default();
        if req.filter_by_kind {
            // Entity kind maps to is_entity = true; content to false.
            filter.is_entity = Some(req.kind == TypedNodeKind::TypedNodeEntity as i32);
        }
        if !req.label.trim().is_empty() {
            filter.label = Some(req.label.clone());
        }
        // P16. Optional property condition (incl. structural terms).
        if let Some(pred) = req.predicate {
            filter.predicate = Some(proto_predicate_to_domain(pred)?);
        }

        let page = store.nodes_page(NodeId(req.after_id), limit, &filter);
        let nodes: Vec<TypedNode> = page.items.iter().map(node_to_proto).collect();
        Ok(Response::new(EnumerateNodesResponse {
            nodes,
            next_cursor: page.next_cursor,
            has_more: page.has_more,
        }))
    }

    async fn enumerate_edges(
        &self,
        request: Request<EnumerateEdgesRequest>,
    ) -> Result<Response<EnumerateEdgesResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let coll = metered_read(&handle);
        let store = coll.graph_store.as_ref().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;

        let limit = clamp_enum_limit(req.limit);
        let mut filter = EdgePageFilter::default();
        if !req.edge_type.trim().is_empty() {
            filter.edge_type = Some(req.edge_type.clone());
        }
        // P16. Optional edge property condition. Reject node-only structural
        // terms (incident-edge count) here: on an edge frontier they resolve to
        // None and would silently match nothing, so fail loudly instead.
        if let Some(pred) = req.predicate {
            let domain = proto_predicate_to_domain(pred)?;
            if domain.needs_store() {
                return Err(Status::invalid_argument(
                    "incident-edge-count is node-only",
                ));
            }
            filter.predicate = Some(domain);
        }
        // P16. Optional endpoint-node constraints.
        if req.filter_by_endpoint_kind {
            filter.endpoint_is_entity =
                Some(req.endpoint_kind == TypedNodeKind::TypedNodeEntity as i32);
        }
        if !req.endpoint_label.trim().is_empty() {
            filter.endpoint_label = Some(req.endpoint_label.clone());
        }

        let page = store.edges_page(EdgeId(req.after_id), limit, &filter);
        let edges: Vec<TypedEdge> = page.items.iter().map(edge_to_proto).collect();
        Ok(Response::new(EnumerateEdgesResponse {
            edges,
            next_cursor: page.next_cursor,
            has_more: page.has_more,
        }))
    }

    // ── Mutable nodes (P16) ──

    async fn update_node(
        &self,
        request: Request<UpdateNodeRequest>,
    ) -> Result<Response<UpdateNodeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        // Parse optional properties outside the lock so a bad payload fails fast.
        // Only the property bag is mutable: source, kind, embedding, created_at,
        // and created_by are immutable provenance. Embedding stays immutable here
        // so the NodeId==VectorId bridge cannot desync from the indexed vector;
        // searchable content vectors are re-indexed via vectors.upsert instead.
        let new_props = match req.properties_json.as_deref() {
            Some(s) => Some(parse_props(s)?),
            None => None,
        };
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&handle);
        let lsn = {
            let cm = self.state.collection_manager.read();
            cm.get_collection(&req.collection)
                .map(|c| c.current_lsn())
                .unwrap_or(0)
        };
        let coll_ref = &mut *coll;
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        let updated = store
            .update_node(
                NodeId(req.node_id),
                new_props,
                actor_opt(&req.actor),
                vf_graph::now_millis(),
                lsn,
            )
            .map_err(|e| Status::internal(format!("update_node failed: {e}")))?
            .ok_or_else(|| Status::not_found("node not found"))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(UpdateNodeResponse {
            node: Some(node_to_proto(&updated)),
        }))
    }

    // ── Manual-edge lifecycle and bulk import (P04) ──

    async fn update_edge(
        &self,
        request: Request<UpdateEdgeRequest>,
    ) -> Result<Response<UpdateEdgeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        // Parse optional properties outside the lock so a bad payload fails fast.
        let new_props = match req.properties_json.as_deref() {
            Some(s) => Some(parse_props(s)?),
            None => None,
        };
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&handle);
        let lsn = {
            let cm = self.state.collection_manager.read();
            cm.get_collection(&req.collection)
                .map(|c| c.current_lsn())
                .unwrap_or(0)
        };
        let coll_ref = &mut *coll;
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        let mut edge = store
            .get_edge(EdgeId(req.edge_id))
            .ok_or_else(|| Status::not_found("edge not found"))?;
        if !edge.is_manual {
            return Err(Status::failed_precondition(
                "only manual edges may be updated",
            ));
        }
        if let Some(props) = new_props {
            edge.properties = props;
        }
        if let Some(c) = req.confidence {
            edge.confidence = c;
        }
        if let Some(v) = req.verified {
            edge.verified = v;
        }
        edge.record_audit("updated", actor_opt(&req.actor), vf_graph::now_millis());
        store
            .put_edge(edge.clone(), lsn)
            .map_err(|e| Status::internal(format!("put_edge failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(UpdateEdgeResponse {
            edge: Some(edge_to_proto(&edge)),
        }))
    }

    async fn verify_edge(
        &self,
        request: Request<VerifyEdgeRequest>,
    ) -> Result<Response<VerifyEdgeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&handle);
        let lsn = {
            let cm = self.state.collection_manager.read();
            cm.get_collection(&req.collection)
                .map(|c| c.current_lsn())
                .unwrap_or(0)
        };
        let coll_ref = &mut *coll;
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
            Status::failed_precondition(format!(
                "collection '{}' is not in hybrid mode; typed graph is unavailable",
                req.collection
            ))
        })?;
        let mut edge = store
            .get_edge(EdgeId(req.edge_id))
            .ok_or_else(|| Status::not_found("edge not found"))?;
        edge.verified = true;
        edge.record_audit("verified", actor_opt(&req.actor), vf_graph::now_millis());
        store
            .put_edge(edge.clone(), lsn)
            .map_err(|e| Status::internal(format!("put_edge failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(VerifyEdgeResponse {
            edge: Some(edge_to_proto(&edge)),
        }))
    }

    async fn reject_edge(
        &self,
        request: Request<RejectEdgeRequest>,
    ) -> Result<Response<RejectEdgeResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;

        // Delete the edge and build the reject rule while holding the write
        // guard; the guard is dropped before calling the extraction manager.
        let (deleted, rule) = {
            let mut coll = metered_write(&handle);
            let lsn = {
                let cm = self.state.collection_manager.read();
                cm.get_collection(&req.collection)
                    .map(|c| c.current_lsn())
                    .unwrap_or(0)
            };
            let coll_ref = &mut *coll;
            let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
                Status::failed_precondition(format!(
                    "collection '{}' is not in hybrid mode; typed graph is unavailable",
                    req.collection
                ))
            })?;
            let edge = match store.get_edge(EdgeId(req.edge_id)) {
                Some(e) => e,
                None => {
                    return Ok(Response::new(RejectEdgeResponse {
                        deleted: false,
                        rule_added: false,
                    }))
                }
            };
            let src_name = node_name(store, edge.source);
            let tgt_name = node_name(store, edge.target);
            let rule = RejectRule {
                source_doc: edge.provenance.source_doc.clone(),
                source_chunk_id: edge.provenance.source_chunk_id,
                edge_type: edge.edge_type.as_str().to_string(),
                source_name: src_name,
                target_name: tgt_name,
            };
            let deleted = store
                .delete_edge(EdgeId(req.edge_id), lsn)
                .map_err(|e| Status::internal(format!("delete_edge failed: {e}")))?;
            let _ = store.sync_delta();
            coll_ref.dirty.store(true, Ordering::Release);
            coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
            (deleted, rule)
        };

        // Guard dropped: now safe to call into the extraction manager.
        let rule_added = match self.state.extraction.add_reject_rule(&req.collection, rule) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("failed to persist reject rule: {e}");
                false
            }
        };
        Ok(Response::new(RejectEdgeResponse {
            deleted,
            rule_added,
        }))
    }

    async fn bulk_import_edges(
        &self,
        request: Request<BulkImportEdgesRequest>,
    ) -> Result<Response<BulkImportEdgesResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        // Snapshot the known edge types from the merged ontology (no guard held).
        let mut known_edge_types: std::collections::HashSet<String> = self
            .state
            .extraction
            .get_ontology(&req.collection)
            .map(|o| o.edge_types.into_iter().map(|t| t.edge_type).collect())
            .unwrap_or_default();

        // Parse the payload.
        let format = match BulkImportFormat::try_from(req.format)
            .unwrap_or(BulkImportFormat::Csv)
        {
            BulkImportFormat::Jsonl => edge_ops::BulkFormat::Jsonl,
            BulkImportFormat::Csv => edge_ops::BulkFormat::Csv,
        };
        let (parsed_rows, parse_errors) = edge_ops::parse_bulk_edges(format, &req.data);
        let total_rows = (parsed_rows.len() + parse_errors.len()) as u64;

        // Optionally extend the ontology with unknown edge types (no guard held).
        if req.auto_add_edge_types {
            let unknown: Vec<String> = parsed_rows
                .iter()
                .map(|r| r.edge_type.clone())
                .filter(|t| !known_edge_types.contains(t))
                .collect();
            if !unknown.is_empty() {
                let mut seen = std::collections::HashSet::new();
                let extension = vf_extraction::Ontology {
                    entity_labels: Vec::new(),
                    edge_types: unknown
                        .into_iter()
                        .filter(|t| seen.insert(t.clone()))
                        .map(|t| {
                            vf_extraction::EdgeTypeDef::new(
                                t,
                                String::new(),
                                Vec::new(),
                                Vec::new(),
                            )
                        })
                        .collect(),
                    // Auto-edge-type extension carries no user prompt.
                    system_prompt: None,
                    extra_guidance: None,
                    link_passages: false,
                    // Auto-edge-type extension does not change resolution mode.
                    entity_resolution: vf_extraction::EntityResolution::Normalized,
                };
                self.state
                    .extraction
                    .set_ontology(&req.collection, None, extension, false)
                    .map_err(|e| Status::internal(format!("ontology update failed: {e}")))?;
                // Refresh the known-types set after the extension.
                known_edge_types = self
                    .state
                    .extraction
                    .get_ontology(&req.collection)
                    .map(|o| o.edge_types.into_iter().map(|t| t.edge_type).collect())
                    .unwrap_or(known_edge_types);
            }
        }

        // Apply rows under the write guard.
        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let (imported, apply_errors) = {
            let mut coll = metered_write(&handle);
            let lsn = {
                let cm = self.state.collection_manager.read();
                cm.get_collection(&req.collection)
                    .map(|c| c.current_lsn())
                    .unwrap_or(0)
            };
            let coll_ref = &mut *coll;
            // Build the valid-endpoint id set before taking the &mut graph_store
            // borrow. An id is valid if it is an existing plain vector (the
            // NodeId == VectorId bridge treats it as a virtual content node) or
            // a materialized typed node. This owns a HashSet and holds no borrow.
            let valid_node_ids: std::collections::HashSet<u64> = {
                let mut candidates: std::collections::HashSet<u64> =
                    std::collections::HashSet::new();
                for r in &parsed_rows {
                    candidates.insert(r.source);
                    candidates.insert(r.target);
                }
                candidates
                    .into_iter()
                    .filter(|&id| {
                        coll_ref.store.contains(id)
                            || coll_ref
                                .graph_store
                                .as_ref()
                                .map(|g| g.get_node(vf_graph::NodeId(id)).is_some())
                                .unwrap_or(false)
                    })
                    .collect()
            };
            let store = coll_ref.graph_store.as_mut().ok_or_else(|| {
                Status::failed_precondition(format!(
                    "collection '{}' is not in hybrid mode; typed graph is unavailable",
                    req.collection
                ))
            })?;
            let (imported, errs) = edge_ops::apply_bulk_edges(
                store,
                &known_edge_types,
                &valid_node_ids,
                parsed_rows,
                actor_opt(&req.actor),
                lsn,
            );
            let _ = store.sync_delta();
            coll_ref.dirty.store(true, Ordering::Release);
            coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
            (imported, errs)
        };

        let mut errors: Vec<BulkImportRowError> = Vec::with_capacity(parse_errors.len() + apply_errors.len());
        for e in parse_errors.into_iter().chain(apply_errors.into_iter()) {
            errors.push(BulkImportRowError {
                row: e.row,
                message: e.message,
            });
        }
        Ok(Response::new(BulkImportEdgesResponse {
            total_rows,
            imported,
            failed: total_rows.saturating_sub(imported),
            errors,
        }))
    }

    // ── Hybrid query engine (P02) ──

    async fn hybrid_query(
        &self,
        request: Request<HybridQueryRequest>,
    ) -> Result<Response<HybridQueryResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let coll = metered_read(&handle);

        // Hybrid queries need a graph layer; reject vector-only. AutoSimilarity
        // is allowed and runs against an empty typed store.
        if coll.config.effective_mode() == vf_core::types::Mode::VectorOnly {
            return Err(Status::failed_precondition(format!(
                "collection '{}' is in vector-only mode; hybrid queries are not available",
                req.collection
            )));
        }

        // Bind a typed store: the collection's own when present, else a local
        // empty store kept alive for the executor borrow.
        let fallback;
        let store: &dyn vf_graph::GraphStore = match coll.graph_store.as_ref() {
            Some(s) => s,
            None => {
                fallback = vf_graph::TypedGraphStore::with_defaults();
                &fallback
            }
        };

        let proto_plan = req
            .plan
            .ok_or_else(|| Status::invalid_argument("hybrid query plan is required"))?;
        // Optional opt-in ranking spec. Absent OR a wholly-unspecified spec means
        // RRF is OFF, so the default execute path below runs (ADR-024: the default
        // graph-augmented path is vector_rank via the plan's steps). RRF runs only
        // when the caller sends an explicit spec.
        let rrf_spec = req
            .rrf_rank
            .filter(|s| !rrf_spec_is_empty(s))
            .map(proto_rrf_to_domain);
        let plan = proto_plan_to_domain(proto_plan)?;

        let hybrid_timer = Instant::now();
        let exec = HybridExecutor::new(coll.index.as_vector_index(), store);
        // The hybrid executor is CPU-bound and runs under the collection read
        // guard. Offload it with block_in_place so the multi-threaded runtime
        // keeps the /readyz and /livez probes responsive during a large
        // VectorRank or RRF rank (requires the multi_thread runtime, which
        // #[tokio::main] provides by default).
        let result = match &rrf_spec {
            // DEFAULT (no explicit RRF spec): run the plan's steps as composed.
            // The default graph-augmented path is vector_rank, expressed as a
            // vector_rank step in the plan (ADR-024); no added proximity work.
            None => tokio::task::block_in_place(|| exec.execute(&plan))
                .map_err(|e| Status::invalid_argument(e.to_string()))?,
            // OPT-IN: fuse vector and graph-proximity rankings with RRF.
            Some(spec) => tokio::task::block_in_place(|| exec.execute_rrf(&plan, spec))
                .map_err(|e| Status::invalid_argument(e.to_string()))?,
        };
        crate::metrics::record_hybrid_query_latency(hybrid_timer);

        let mut resp = HybridQueryResponse::default();
        match result {
            QueryResult::Nodes(records) => {
                resp.nodes = records
                    .into_iter()
                    .map(|NodeRecord { id, node }| match node {
                        Some(n) => node_to_proto(&n),
                        None => virtual_content_node(id.0),
                    })
                    .collect();
            }
            QueryResult::Edges(edges) => {
                resp.edges = edges.iter().map(edge_to_proto).collect();
            }
            QueryResult::Paths(paths) => {
                resp.paths = paths
                    .into_iter()
                    .map(|p| HybridPath {
                        nodes: p.nodes.iter().map(|n| n.0).collect(),
                    })
                    .collect();
            }
        }
        Ok(Response::new(resp))
    }
}
