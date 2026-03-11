// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use std::sync::atomic::Ordering;

use tonic::{Request, Response, Status};

use vf_graph::{GraphTraversal, RelationshipQueryEngine, TraversalOrder};

use crate::proto::swarndb::v1::graph_service_server::GraphService;
use crate::proto::swarndb::v1::{
    GetRelatedRequest, GetRelatedResponse, GraphEdge, SetThresholdRequest, SetThresholdResponse,
    TraversalNode, TraverseRequest, TraverseResponse,
};
use crate::state::AppState;

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

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| {
                Status::not_found(format!("collection '{}' not found", req.collection))
            })?;

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

        let collections = self.state.collections.read();
        let collection = collections
            .get(&req.collection)
            .ok_or_else(|| {
                Status::not_found(format!("collection '{}' not found", req.collection))
            })?;

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

        let depth = (req.depth as usize).min(100);
        let traversal_results = GraphTraversal::traverse(
            &collection.graph,
            req.start_id,
            &TraversalOrder::BreadthFirst,
            depth,
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

        if req.threshold <= 0.0 || req.threshold > 1.0 {
            return Err(Status::invalid_argument(
                "threshold must be >0.0 and <=1.0",
            ));
        }

        let mut collections = self.state.collections.write();
        let collection = collections
            .get_mut(&req.collection)
            .ok_or_else(|| {
                Status::not_found(format!("collection '{}' not found", req.collection))
            })?;

        if req.vector_id == 0 {
            // Update collection-level default threshold
            collection.graph.config_mut().default_threshold = req.threshold;
            collection.config.default_similarity_threshold = Some(req.threshold);
            collection.deferred_graph.store(true, Ordering::Release);
        } else {
            // Set per-vector threshold override
            collection
                .graph
                .set_vector_threshold(req.vector_id, req.threshold);
        }

        Ok(Response::new(SetThresholdResponse { success: true }))
    }
}
