// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! gRPC CollectionService handler implementation.

use tonic::{Request, Response, Status};

use crate::convert::{distance_metric_to_string, parse_distance_metric};
use crate::proto::swarndb::v1::collection_service_server::CollectionService;
use crate::proto::swarndb::v1::{
    CreateCollectionRequest, CreateCollectionResponse, DeleteCollectionRequest,
    DeleteCollectionResponse, GetCollectionRequest, GetCollectionResponse,
    ListCollectionsRequest, ListCollectionsResponse,
};
use crate::state::{AppState, CollectionState, MetadataCache};
use vf_core::store::InMemoryVectorStore;
use vf_core::types::{CollectionConfig, DataTypeConfig};
use vf_graph::VirtualGraph;
use vf_index::hnsw::HnswIndex;
use vf_query::IndexManager;

pub struct CollectionServiceImpl {
    state: AppState,
}

impl CollectionServiceImpl {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl CollectionService for CollectionServiceImpl {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();

        if req.name.is_empty() {
            return Err(Status::invalid_argument("collection name must not be empty"));
        }
        if req.dimension == 0 {
            return Err(Status::invalid_argument("dimension must be greater than 0"));
        }

        let metric_str = if req.distance_metric.is_empty() {
            "cosine"
        } else {
            &req.distance_metric
        };
        let distance_metric = parse_distance_metric(metric_str).ok_or_else(|| {
            Status::invalid_argument(format!("unknown distance metric: {}", req.distance_metric))
        })?;

        let dimension = req.dimension as usize;
        let threshold = if req.default_threshold > 0.0 {
            Some(req.default_threshold)
        } else {
            None
        };

        let config = CollectionConfig {
            name: req.name.clone(),
            dimension,
            distance_metric,
            default_similarity_threshold: threshold,
            max_vectors: req.max_vectors as usize,
            data_type: DataTypeConfig::F32,
        };

        // Check for duplicate in-memory BEFORE persisting to storage
        {
            let collections = self.state.collections.read();
            if collections.contains_key(&req.name) {
                return Err(Status::already_exists(format!(
                    "collection '{}' already exists",
                    req.name
                )));
            }
        }

        // Persist to storage layer
        {
            let mut cm = self.state.collection_manager.write();
            if let Err(e) = cm.create_collection(config.clone()) {
                return Err(Status::internal(format!("storage create failed: {}", e)));
            }
        }

        let store = InMemoryVectorStore::new(dimension);
        let index = HnswIndex::with_defaults(dimension, distance_metric);
        let index_manager = IndexManager::with_defaults();
        let graph = match threshold {
            Some(t) => VirtualGraph::with_threshold(t, distance_metric),
            None => VirtualGraph::with_threshold(0.7, distance_metric),
        };

        let collection_state = CollectionState {
            config,
            store,
            index,
            index_manager,
            graph,
            metadata_cache: MetadataCache::new(),
            status: std::sync::Arc::new(std::sync::RwLock::new(
                crate::state::CollectionStatus::Ready,
            )),
            deferred_index: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            deferred_graph: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            deferred_metadata: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };

        let mut collections = self.state.collections.write();
        collections.insert(req.name.clone(), collection_state);

        Ok(Response::new(CreateCollectionResponse {
            name: req.name,
            success: true,
        }))
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<DeleteCollectionResponse>, Status> {
        let req = request.into_inner();

        // Remove from storage layer
        {
            let mut cm = self.state.collection_manager.write();
            if let Err(e) = cm.drop_collection(&req.name) {
                tracing::warn!(collection = %req.name, "storage drop failed: {}", e);
            }
        }

        let mut collections = self.state.collections.write();
        if collections.remove(&req.name).is_none() {
            return Err(Status::not_found(format!(
                "collection '{}' not found",
                req.name
            )));
        }

        Ok(Response::new(DeleteCollectionResponse { success: true }))
    }

    async fn get_collection(
        &self,
        request: Request<GetCollectionRequest>,
    ) -> Result<Response<GetCollectionResponse>, Status> {
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let coll = collections.get(&req.name).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.name))
        })?;

        let status_str = coll.status.read().unwrap().as_str().to_string();
        Ok(Response::new(GetCollectionResponse {
            name: coll.config.name.clone(),
            dimension: coll.config.dimension as u32,
            distance_metric: distance_metric_to_string(coll.config.distance_metric),
            vector_count: coll.store.len() as u64,
            default_threshold: coll.config.default_similarity_threshold.unwrap_or(0.0),
            status: status_str,
        }))
    }

    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let collections = self.state.collections.read();

        let list: Vec<GetCollectionResponse> = collections
            .values()
            .map(|coll| {
                let status_str = coll.status.read().unwrap().as_str().to_string();
                GetCollectionResponse {
                    name: coll.config.name.clone(),
                    dimension: coll.config.dimension as u32,
                    distance_metric: distance_metric_to_string(coll.config.distance_metric),
                    vector_count: coll.store.len() as u64,
                    default_threshold: coll.config.default_similarity_threshold.unwrap_or(0.0),
                    status: status_str,
                }
            })
            .collect();

        Ok(Response::new(ListCollectionsResponse {
            collections: list,
        }))
    }
}
