// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! gRPC CollectionService handler implementation.

use tonic::{Request, Response, Status};

use crate::convert::{
    build_hnsw_params, distance_metric_to_string, parse_distance_metric, proto_mode_to_core,
};
use crate::proto::swarndb::v1::collection_service_server::CollectionService;
use crate::proto::swarndb::v1::{
    CollectionRecoveryEntry, CreateCollectionRequest, CreateCollectionResponse,
    DeleteCollectionRequest, DeleteCollectionResponse, GetCollectionMetricsRequest,
    GetCollectionMetricsResponse, GetCollectionRequest, GetCollectionResponse,
    GetPersistenceStatusRequest, GetPersistenceStatusResponse, GetRecoveryStatusRequest,
    GetRecoveryStatusResponse, ListCollectionsRequest, ListCollectionsResponse, RecoveryPath,
    SnapshotCollectionRequest, SnapshotCollectionResponse,
};
use crate::snapshot::force_snapshot_collection;
use crate::state::{
    metered_read, AppState, CollectionAvailability, CollectionState, MetadataCache, RecoveryStatus,
};
use crate::validation::{validate_collection_name, ValidationConfig};

/// Convert a `CollectionAvailability` returned by the readiness guard into a
/// `tonic::Status`. Recovering collections become `Unavailable` (503-equivalent
/// in gRPC) with a retry-after hint embedded in the message; missing
/// collections become `NotFound`.
fn status_from_availability(avail: CollectionAvailability) -> Status {
    match avail {
        CollectionAvailability::Recovering { .. } => Status::unavailable(avail.user_message()),
        CollectionAvailability::NotFound { .. } => Status::not_found(avail.user_message()),
    }
}
use vf_core::store::InMemoryVectorStore;
use vf_core::types::{
    CollectionConfig, DataTypeConfig, QuantizationConfig, ScalarQuantizationConfig,
};
use vf_graph::VirtualGraph;
use vf_index::hnsw::HnswIndex;
use vf_index::quantized_hnsw::QuantizedHnswIndex;
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

        // Full name validation (length, charset, reserved-name rejection) so the
        // gRPC path cannot bypass the rules enforced on the REST path. Covers the
        // empty-name case and the reserved leading-underscore namespace.
        if let Err(e) = validate_collection_name(&req.name, &ValidationConfig::default()) {
            return Err(Status::invalid_argument(e.to_string()));
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

        let quantization_config = if let Some(quant) = &req.quantization {
            match &quant.method {
                Some(crate::proto::swarndb::v1::quantization_config::Method::Scalar(sq)) => {
                    Some(QuantizationConfig::Scalar(ScalarQuantizationConfig {
                        quantile: if sq.quantile > 0.0 { sq.quantile } else { 0.99 },
                        always_ram: sq.always_ram,
                    }))
                }
                None => None,
            }
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
            quantization_config,
            mode: Some(proto_mode_to_core(req.mode())),
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

        // Optional HNSW build parameters from the request; omitted fields fall
        // through to the server defaults via build_hnsw_params.
        let hnsw_params = build_hnsw_params(req.m, req.ef_construction);

        let store = InMemoryVectorStore::new(dimension);
        let index: Box<dyn vf_index::traits::PersistableIndex> = match &config.quantization_config {
            Some(QuantizationConfig::Scalar(sq_config)) => {
                let q_index = QuantizedHnswIndex::new(
                    dimension,
                    distance_metric,
                    hnsw_params.clone(),
                    sq_config.clone(),
                );
                // Set data_dir so post_optimize() can train the quantizer.
                let collection_dir = {
                    let cm = self.state.collection_manager.read();
                    cm.get_collection(&req.name)
                        .map(|c| c.collection_dir().to_path_buf())
                        .ok()
                };
                if let Some(dir) = collection_dir {
                    q_index.set_data_dir(dir);
                }
                Box::new(q_index)
            }
            None => Box::new(HnswIndex::new(dimension, distance_metric, hnsw_params)),
        };

        // Attach a fresh hnsw.delta writer so first inserts are recorded for
        // incremental replay on the next boot.
        let collection_dir_for_delta = {
            let cm = self.state.collection_manager.read();
            cm.get_collection(&req.name)
                .map(|c| c.collection_dir().to_path_buf())
                .ok()
        };
        if let Some(dir) = collection_dir_for_delta {
            let hnsw_delta_path = dir.join("hnsw.delta");
            match vf_index::hnsw_delta::HnswDeltaWriter::create(&hnsw_delta_path) {
                Ok(writer) => index.set_delta_writer(writer),
                Err(e) => tracing::warn!(
                    collection = %req.name,
                    "failed to create initial hnsw delta writer: {e}"
                ),
            }
        }

        let index_manager = IndexManager::with_defaults();
        let graph = match threshold {
            Some(t) => VirtualGraph::with_threshold(t, distance_metric),
            None => VirtualGraph::with_threshold(0.7, distance_metric),
        };

        // Typed graph store for Hybrid collections (ADR-007 R4); None otherwise.
        let graph_store = {
            let dir = {
                let cm = self.state.collection_manager.read();
                cm.get_collection(&req.name)
                    .map(|c| c.collection_dir().to_path_buf())
                    .ok()
            };
            dir.and_then(|d| crate::state::AppState::create_hybrid_graph_store(&d, &config))
        };

        let collection_state = CollectionState {
            config,
            store,
            index,
            index_manager,
            graph,
            graph_store,
            metadata_cache: MetadataCache::new(),
            status: std::sync::Arc::new(std::sync::RwLock::new(
                crate::state::CollectionStatus::Ready,
            )),
            deferred_index: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            deferred_graph: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            deferred_metadata: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            dirty: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            mutation_count: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            collection_read_acquisitions: std::sync::atomic::AtomicU64::new(0),
            collection_write_acquisitions: std::sync::atomic::AtomicU64::new(0),
            total_blocked_microseconds: std::sync::atomic::AtomicU64::new(0),
        };

        // Map write lock is required to insert a new collection entry. The
        // CollectionState is wrapped in a per-collection RwLock so all later
        // mutation paths can run under the per-collection lock alone.
        {
            let mut collections = self.state.collections.write();
            collections.insert(
                req.name.clone(),
                std::sync::Arc::new(parking_lot::RwLock::new(collection_state)),
            );
        }

        // Register the extraction runtime for Hybrid collections now that the
        // handle is published. A no-op for VectorOnly / AutoSimilarity.
        self.state.register_extraction_if_hybrid(&req.name);

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

        tracing::info!(target: "vf_server::audit", collection = %req.name, "audit: collection delete requested via gRPC");

        self.state
            .require_collection_ready(&req.name)
            .map_err(status_from_availability)?;

        // Cancel any in-flight extraction jobs for this collection and drop its
        // extraction runtime, so a dropped collection never leaves a runaway job
        // (ADR-016). Cooperative cancel: workers drain and skip its chunks.
        self.state.extraction.cancel_collection_jobs(&req.name);

        // Remove from storage layer
        {
            let mut cm = self.state.collection_manager.write();
            if let Err(e) = cm.drop_collection(&req.name) {
                tracing::warn!(collection = %req.name, "storage drop failed: {}", e);
            }
        }

        // Map write lock to evict the entry. In-flight handlers that hold an
        // Arc<RwLock<CollectionState>> from a previous lookup complete normally
        // and the inner state is dropped once the last Arc handle is released.
        let mut collections = self.state.collections.write();
        if collections.remove(&req.name).is_none() {
            return Err(Status::not_found(format!(
                "collection '{}' not found",
                req.name
            )));
        }
        // Drop any recovery_paths entry so the recovery_status surface does
        // not retain a stale entry for a deleted collection.
        self.state.recovery_paths.write().remove(&req.name);

        Ok(Response::new(DeleteCollectionResponse { success: true }))
    }

    async fn get_collection(
        &self,
        request: Request<GetCollectionRequest>,
    ) -> Result<Response<GetCollectionResponse>, Status> {
        let req = request.into_inner();

        self.state
            .require_collection_ready(&req.name)
            .map_err(status_from_availability)?;

        let coll_handle = self.state.collection_handle(&req.name).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.name))
        })?;
        let coll = metered_read(&coll_handle);

        let status_str = coll.status.read().unwrap().as_str().to_string();
        let quantization_type = match &coll.config.quantization_config {
            Some(QuantizationConfig::Scalar(_)) => "scalar".to_string(),
            None => "none".to_string(),
        };
        Ok(Response::new(GetCollectionResponse {
            name: coll.config.name.clone(),
            dimension: coll.config.dimension as u32,
            distance_metric: distance_metric_to_string(coll.config.distance_metric),
            vector_count: coll.store.len() as u64,
            default_threshold: coll.config.default_similarity_threshold.unwrap_or(0.0),
            status: status_str,
            quantization_type,
            // Live index node count; may trail vector_count under a deferred build.
            indexed_count: coll.index.len() as u64,
        }))
    }

    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        // Snapshot the handle list under a short map read lock, then release
        // the map lock before per-collection reads to avoid map-level contention
        // with create/delete during listing.
        let handles: Vec<std::sync::Arc<parking_lot::RwLock<crate::state::CollectionState>>> = {
            let collections = self.state.collections.read();
            collections.values().cloned().collect()
        };

        let list: Vec<GetCollectionResponse> = handles
            .iter()
            .map(|h| {
                let coll = metered_read(h);
                let status_str = coll.status.read().unwrap().as_str().to_string();
                let quantization_type = match &coll.config.quantization_config {
                    Some(QuantizationConfig::Scalar(_)) => "scalar".to_string(),
                    None => "none".to_string(),
                };
                GetCollectionResponse {
                    name: coll.config.name.clone(),
                    dimension: coll.config.dimension as u32,
                    distance_metric: distance_metric_to_string(coll.config.distance_metric),
                    vector_count: coll.store.len() as u64,
                    default_threshold: coll.config.default_similarity_threshold.unwrap_or(0.0),
                    status: status_str,
                    quantization_type,
                    // Live index node count; may trail vector_count under a deferred build.
                    indexed_count: coll.index.len() as u64,
                }
            })
            .collect();

        Ok(Response::new(ListCollectionsResponse {
            collections: list,
        }))
    }

    async fn get_recovery_status(
        &self,
        _request: Request<GetRecoveryStatusRequest>,
    ) -> Result<Response<GetRecoveryStatusResponse>, Status> {
        let snap = self.state.recovery_status_snapshot();
        let mut entries: Vec<CollectionRecoveryEntry> = snap
            .paths
            .iter()
            .map(|(name, p)| CollectionRecoveryEntry {
                name: name.clone(),
                path: recovery_status_to_proto(*p) as i32,
            })
            .collect();
        // Stable ordering keeps the response deterministic for clients.
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Response::new(GetRecoveryStatusResponse {
            elapsed_secs: snap.elapsed_secs,
            collections: entries,
            path: recovery_status_to_proto(snap.path) as i32,
        }))
    }

    async fn snapshot_collection(
        &self,
        request: Request<SnapshotCollectionRequest>,
    ) -> Result<Response<SnapshotCollectionResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.name)
            .map_err(status_from_availability)?;
        let lsn = force_snapshot_collection(&self.state, &req.name)
            .map_err(Status::internal)?;
        Ok(Response::new(SnapshotCollectionResponse {
            last_snapshot_lsn: lsn,
        }))
    }

    async fn get_persistence_status(
        &self,
        request: Request<GetPersistenceStatusRequest>,
    ) -> Result<Response<GetPersistenceStatusResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.name)
            .map_err(status_from_availability)?;
        let p = self
            .state
            .persistence_status(&req.name)
            .map_err(Status::internal)?;
        Ok(Response::new(GetPersistenceStatusResponse {
            last_snapshot_lsn: p.last_snapshot_lsn,
            current_lsn: p.current_lsn,
            next_lsn: p.next_lsn,
        }))
    }

    async fn get_collection_metrics(
        &self,
        request: Request<GetCollectionMetricsRequest>,
    ) -> Result<Response<GetCollectionMetricsResponse>, Status> {
        let req = request.into_inner();
        self.state
            .require_collection_ready(&req.name)
            .map_err(status_from_availability)?;
        let m = self
            .state
            .collection_metrics(&req.name)
            .map_err(Status::internal)?;
        Ok(Response::new(GetCollectionMetricsResponse {
            map_lock_acquisitions: m.map_lock_acquisitions,
            collection_read_acquisitions: m.collection_read_acquisitions,
            collection_write_acquisitions: m.collection_write_acquisitions,
            total_blocked_microseconds: m.total_blocked_microseconds,
        }))
    }
}

/// Map the in-memory recovery enum onto the wire enum. Prost generates
/// `RecoveryPath` variants by converting the proto SCREAMING_SNAKE_CASE
/// names (e.g. `RECOVERY_UNKNOWN`) to upper-camel-case (`RecoveryUnknown`)
/// without stripping the shared prefix.
fn recovery_status_to_proto(s: RecoveryStatus) -> RecoveryPath {
    match s {
        RecoveryStatus::Unknown => RecoveryPath::RecoveryUnknown,
        RecoveryStatus::CleanShutdown => RecoveryPath::RecoveryCleanShutdown,
        RecoveryStatus::IncrementalReplay => RecoveryPath::RecoveryIncrementalReplay,
        RecoveryStatus::FullRebuild => RecoveryPath::RecoveryFullRebuild,
    }
}
