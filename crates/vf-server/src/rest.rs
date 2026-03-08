// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::extract::{Path, Query, State, FromRequest};
use axum::http::StatusCode;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use vf_core::store::{InMemoryVectorStore, VectorRecord};
use vf_core::types::{
    CollectionConfig, DataTypeConfig, DistanceMetricType, Metadata, MetadataValue, VectorId,
};
use vf_core::vector::VectorData;
use vf_graph::{GraphTraversal, RelationshipQueryEngine, TraversalOrder};
use vf_index::traits::VectorIndex;
use vf_query::vector_math::*;
use vf_query::{FilterExpression, FilterStrategy, IndexManager, QueryExecutor};

use crate::convert::{distance_metric_to_string, parse_distance_metric};
use crate::metrics;
use crate::state::{AppState, CollectionState, CollectionStatus, MetadataCache};
use crate::validation::{
    validate_batch_lock_size, validate_bulk_insert_options, validate_collection_name,
    validate_ef_construction, validate_ef_search, validate_index_mode, validate_wal_flush_every,
    ValidationConfig,
};

// ── Error handling ──────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
}

fn err(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    let s = status;
    (s, Json(ErrorResponse { error: msg.into(), code: s.as_u16() }))
}

// ── Custom JSON extractor (returns 400 instead of 422) ──────────────────

struct ValidatedJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<ErrorResponse>);

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(ValidatedJson(value)),
            Err(rejection) => Err(err(StatusCode::BAD_REQUEST, format!("invalid request body: {}", rejection))),
        }
    }
}

// ── Collection types ────────────────────────────────────────────────────

fn default_threshold() -> f32 { 0.0 }

#[derive(Deserialize)]
pub struct CreateCollectionReq {
    pub name: String,
    pub dimension: u32,
    #[serde(default = "default_distance")]
    pub distance_metric: String,
    #[serde(default = "default_threshold")]
    pub default_threshold: f32,
    #[serde(default)]
    pub max_vectors: u64,
}

fn default_distance() -> String { "cosine".to_string() }

#[derive(Serialize)]
pub struct CreateCollectionRes {
    pub name: String,
    pub success: bool,
}

#[derive(Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: u32,
    pub distance_metric: String,
    pub vector_count: u64,
    pub default_threshold: f32,
    pub status: String,
}

#[derive(Serialize)]
pub struct ListCollectionsRes {
    pub collections: Vec<CollectionInfo>,
}

#[derive(Serialize)]
pub struct DeleteCollectionRes {
    pub success: bool,
}

// ── Vector types ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct InsertVectorReq {
    #[serde(default)]
    pub id: u64,
    pub values: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct InsertVectorRes {
    pub id: u64,
    pub success: bool,
}

#[derive(Serialize)]
pub struct GetVectorRes {
    pub id: u64,
    pub values: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct UpdateVectorReq {
    #[serde(default)]
    pub values: Option<Vec<f32>>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct UpdateVectorRes {
    pub success: bool,
}

#[derive(Serialize)]
pub struct DeleteVectorRes {
    pub success: bool,
}

#[derive(Deserialize)]
pub struct BulkInsertReq {
    pub vectors: Vec<InsertVectorReq>,
    #[serde(default)]
    pub batch_lock_size: Option<u32>,
    #[serde(default)]
    pub defer_graph: Option<bool>,
    #[serde(default)]
    pub wal_flush_every: Option<u32>,
    #[serde(default)]
    pub ef_construction: Option<u32>,
    #[serde(default)]
    pub index_mode: Option<String>,
    #[serde(default)]
    pub skip_metadata_index: Option<bool>,
    #[serde(default)]
    pub parallel_build: Option<bool>,
}

#[derive(Serialize)]
pub struct BulkInsertRes {
    pub inserted_count: u64,
    pub errors: Vec<String>,
}

// ── Search types ────────────────────────────────────────────────────────

fn default_strategy() -> String { "auto".to_string() }

fn default_max_graph_edges() -> u32 { 10 }

#[derive(Deserialize)]
pub struct SearchReq {
    pub query: Vec<f32>,
    pub k: u32,
    #[serde(default)]
    pub filter: Option<serde_json::Value>,
    #[serde(default = "default_strategy")]
    pub strategy: String,
    #[serde(default)]
    pub include_metadata: bool,
    #[serde(default)]
    pub include_graph: bool,
    #[serde(default)]
    pub graph_threshold: f32,
    #[serde(default = "default_max_graph_edges")]
    pub max_graph_edges: u32,
    #[serde(default)]
    pub ef_search: Option<u32>,
}

#[derive(Serialize)]
pub struct ScoredResult {
    pub id: u64,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub graph_edges: Vec<GraphEdge>,
}

#[derive(Serialize)]
pub struct SearchRes {
    pub results: Vec<ScoredResult>,
    pub search_time_us: u64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub warning: String,
}

#[derive(Deserialize)]
pub struct BatchSearchReq {
    pub queries: Vec<BatchSearchQuery>,
}

#[derive(Deserialize)]
pub struct BatchSearchQuery {
    pub collection: String,
    pub query: Vec<f32>,
    pub k: u32,
    #[serde(default)]
    pub filter: Option<serde_json::Value>,
    #[serde(default = "default_strategy")]
    pub strategy: String,
    #[serde(default)]
    pub include_metadata: bool,
    #[serde(default)]
    pub include_graph: bool,
    #[serde(default)]
    pub graph_threshold: f32,
    #[serde(default = "default_max_graph_edges")]
    pub max_graph_edges: u32,
    #[serde(default)]
    pub ef_search: Option<u32>,
}

#[derive(Serialize)]
pub struct BatchSearchRes {
    pub results: Vec<SearchRes>,
    pub total_time_us: u64,
}

// ── Graph types ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct GraphEdge {
    pub target_id: u64,
    pub similarity: f32,
}

#[derive(Serialize)]
pub struct GetRelatedRes {
    pub edges: Vec<GraphEdge>,
}

#[derive(Deserialize)]
pub struct GetRelatedQuery {
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub max_results: Option<u32>,
}

#[derive(Deserialize)]
pub struct TraverseReq {
    pub start_id: u64,
    #[serde(default = "default_depth")]
    pub depth: u32,
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub max_results: Option<u32>,
}

fn default_depth() -> u32 { 2 }

#[derive(Serialize)]
pub struct TraversalNode {
    pub id: u64,
    pub depth: u32,
    pub path_similarity: f32,
    pub path: Vec<u64>,
}

#[derive(Serialize)]
pub struct TraverseRes {
    pub nodes: Vec<TraversalNode>,
}

#[derive(Deserialize)]
pub struct SetThresholdReq {
    #[serde(default)]
    pub vector_id: u64,
    pub threshold: f32,
}

#[derive(Serialize)]
pub struct SetThresholdRes {
    pub success: bool,
}

// ── Router ──────────────────────────────────────────────────────────────

pub fn rest_router(state: AppState) -> Router {
    Router::new()
        // Collections
        .route("/api/v1/collections", post(create_collection))
        .route("/api/v1/collections", get(list_collections))
        .route("/api/v1/collections/{name}", get(get_collection))
        .route("/api/v1/collections/{name}", delete(delete_collection))
        // Vectors
        .route("/api/v1/collections/{collection}/vectors", post(insert_vector))
        .route("/api/v1/collections/{collection}/vectors/{id}", get(get_vector))
        .route("/api/v1/collections/{collection}/vectors/{id}", put(update_vector))
        .route("/api/v1/collections/{collection}/vectors/{id}", delete(delete_vector))
        .route("/api/v1/collections/{collection}/vectors/bulk", post(bulk_insert))
        .route("/api/v1/collections/{collection}/optimize", post(optimize_collection))
        // Search
        .route("/api/v1/collections/{collection}/search", post(search))
        .route("/api/v1/search/batch", post(batch_search))
        // Graph
        .route("/api/v1/collections/{collection}/graph/related/{id}", get(get_related))
        .route("/api/v1/collections/{collection}/graph/traverse", post(traverse))
        .route("/api/v1/collections/{collection}/graph/threshold", post(set_threshold))
        // Vector Math
        .route("/api/v1/collections/{collection}/math/ghosts", post(detect_ghosts))
        .route("/api/v1/collections/{collection}/math/cone", post(cone_search))
        .route("/api/v1/collections/{collection}/math/centroid", post(compute_centroid))
        .route("/api/v1/math/interpolate", post(interpolate))
        .route("/api/v1/collections/{collection}/math/drift", post(detect_drift))
        .route("/api/v1/collections/{collection}/math/cluster", post(cluster))
        .route("/api/v1/collections/{collection}/math/pca", post(reduce_dimensions))
        .route("/api/v1/math/analogy", post(compute_analogy))
        .route("/api/v1/collections/{collection}/math/diversity", post(diversity_sample))
        .with_state(state)
}

// ── Collection handlers ─────────────────────────────────────────────────

async fn create_collection(
    State(state): State<AppState>,
    ValidatedJson(req): ValidatedJson<CreateCollectionReq>,
) -> Result<Json<CreateCollectionRes>, (StatusCode, Json<ErrorResponse>)> {
    let validation_config = ValidationConfig::default();
    if let Err(e) = validate_collection_name(&req.name, &validation_config) {
        return Err(err(StatusCode::BAD_REQUEST, e.to_string()));
    }
    if req.dimension == 0 {
        return Err(err(StatusCode::BAD_REQUEST, "dimension must be greater than 0"));
    }

    let distance_metric = parse_distance_metric(&req.distance_metric)
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, format!("unknown distance metric: {}", req.distance_metric)))?;

    let dimension = req.dimension as usize;
    let threshold = if req.default_threshold > 0.0 { Some(req.default_threshold) } else { None };

    let config = CollectionConfig {
        name: req.name.clone(),
        dimension,
        distance_metric,
        default_similarity_threshold: threshold,
        max_vectors: req.max_vectors as usize,
        data_type: DataTypeConfig::F32,
    };

    let store = InMemoryVectorStore::new(dimension);
    let index = vf_index::hnsw::HnswIndex::with_defaults(dimension, distance_metric);
    let index_manager = IndexManager::with_defaults();
    let graph = match threshold {
        Some(t) => vf_graph::VirtualGraph::with_threshold(t, distance_metric),
        None => vf_graph::VirtualGraph::with_threshold(0.7, distance_metric),
    };

    // Check for duplicate in-memory BEFORE persisting to storage
    {
        let collections = state.collections.read();
        if collections.contains_key(&req.name) {
            return Err(err(StatusCode::CONFLICT, format!("collection '{}' already exists", req.name)));
        }
    }

    // Persist to storage layer
    {
        let mut cm = state.collection_manager.write();
        if let Err(e) = cm.create_collection(config.clone()) {
            return Err(err(StatusCode::INTERNAL_SERVER_ERROR, format!("storage create failed: {}", e)));
        }
    }

    let collection_state = CollectionState { config, store, index, index_manager, graph, metadata_cache: MetadataCache::new(), status: std::sync::Arc::new(std::sync::RwLock::new(crate::state::CollectionStatus::Ready)), deferred_index: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)), deferred_graph: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)), deferred_metadata: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)) };

    let mut collections = state.collections.write();
    collections.insert(req.name.clone(), collection_state);

    Ok(Json(CreateCollectionRes { name: req.name, success: true }))
}

async fn list_collections(
    State(state): State<AppState>,
) -> Json<ListCollectionsRes> {
    let collections = state.collections.read();
    let list = collections.values().map(|c| {
        let status_str = c.status.read().unwrap().as_str().to_string();
        CollectionInfo {
            name: c.config.name.clone(),
            dimension: c.config.dimension as u32,
            distance_metric: distance_metric_to_string(c.config.distance_metric),
            vector_count: c.store.len() as u64,
            default_threshold: c.config.default_similarity_threshold.unwrap_or(0.0),
            status: status_str,
        }
    }).collect();
    Json(ListCollectionsRes { collections: list })
}

async fn get_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, (StatusCode, Json<ErrorResponse>)> {
    let collections = state.collections.read();
    let c = collections.get(&name)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", name)))?;
    let status_str = c.status.read().unwrap().as_str().to_string();
    Ok(Json(CollectionInfo {
        name: c.config.name.clone(),
        dimension: c.config.dimension as u32,
        distance_metric: distance_metric_to_string(c.config.distance_metric),
        vector_count: c.store.len() as u64,
        default_threshold: c.config.default_similarity_threshold.unwrap_or(0.0),
        status: status_str,
    }))
}

async fn delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<DeleteCollectionRes>, (StatusCode, Json<ErrorResponse>)> {
    // Remove from storage layer
    {
        let mut cm = state.collection_manager.write();
        if let Err(e) = cm.drop_collection(&name) {
            tracing::warn!(collection = %name, "storage drop failed: {}", e);
        }
    }

    let mut collections = state.collections.write();
    if collections.remove(&name).is_none() {
        return Err(err(StatusCode::NOT_FOUND, format!("collection '{}' not found", name)));
    }
    Ok(Json(DeleteCollectionRes { success: true }))
}

// ── Vector handlers ─────────────────────────────────────────────────────

async fn insert_vector(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<InsertVectorReq>,
) -> Result<Json<InsertVectorRes>, (StatusCode, Json<ErrorResponse>)> {
    if req.values.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "vector values must not be empty"));
    }

    let core_metadata = req.metadata.as_ref().map(json_to_metadata)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;

    let mut collections = state.collections.write();
    let coll = collections.get_mut(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    if req.values.len() != coll.config.dimension {
        return Err(err(StatusCode::BAD_REQUEST, format!("vector dimension mismatch: expected {}, got {}", coll.config.dimension, req.values.len())));
    }

    let vector_data = VectorData::F32(req.values.clone());

    let assigned_id = if req.id == 0 {
        coll.store.insert_auto_id(vector_data, core_metadata.clone())
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("store insert failed: {}", e)))?
    } else {
        let record = VectorRecord::new(req.id, vector_data, core_metadata.clone());
        coll.store.insert(record)
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("store insert failed: {}", e)))?;
        req.id
    };

    if let Err(e) = coll.index.add(assigned_id, &req.values) {
        // Rollback: remove from store since index add failed
        let _ = coll.store.delete(assigned_id);
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, format!("index insert failed: {}", e)));
    }

    if let Some(ref meta) = core_metadata {
        coll.index_manager.index_record(assigned_id, meta);
    }

    // Compute virtual graph edges for the newly inserted vector
    if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
        &mut coll.graph, &coll.index, assigned_id, &req.values, 10,
    ) {
        tracing::warn!(collection = %collection, id = assigned_id, "graph compute failed: {}", e);
    }

    // Persist to storage layer (best-effort)
    {
        let mut cm = state.collection_manager.write();
        if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
            if let Err(e) = storage_coll.insert(assigned_id, VectorData::F32(req.values), core_metadata) {
                tracing::warn!(collection = %collection, id = assigned_id, "storage insert failed: {}", e);
            }
        }
    }

    Ok(Json(InsertVectorRes { id: assigned_id, success: true }))
}

async fn get_vector(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, u64)>,
) -> Result<Json<GetVectorRes>, (StatusCode, Json<ErrorResponse>)> {
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    let record = coll.store.get(id)
        .map_err(|e| err(StatusCode::NOT_FOUND, format!("vector not found: {}", e)))?;

    let metadata_json = record.metadata.as_ref().map(metadata_to_json);

    Ok(Json(GetVectorRes {
        id: record.id,
        values: record.data.to_f32_vec(),
        metadata: metadata_json,
    }))
}

async fn update_vector(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, u64)>,
    ValidatedJson(req): ValidatedJson<UpdateVectorReq>,
) -> Result<Json<UpdateVectorRes>, (StatusCode, Json<ErrorResponse>)> {
    if req.values.is_none() && req.metadata.is_none() {
        return Err(err(StatusCode::BAD_REQUEST, "at least one of 'values' or 'metadata' must be provided"));
    }

    if let Some(ref v) = req.values {
        if v.is_empty() {
            return Err(err(StatusCode::BAD_REQUEST, "vector values must not be empty"));
        }
    }

    let core_metadata = req.metadata.as_ref().map(json_to_metadata)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;

    let mut collections = state.collections.write();
    let coll = collections.get_mut(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    if let Some(ref v) = req.values {
        if v.len() != coll.config.dimension {
            return Err(err(StatusCode::BAD_REQUEST, format!("vector dimension mismatch: expected {}, got {}", coll.config.dimension, v.len())));
        }
    }

    let vector_data = req.values.as_ref().map(|v| VectorData::F32(v.clone()));
    coll.store.update(id, vector_data.clone(), core_metadata.clone())
        .map_err(|e| err(StatusCode::NOT_FOUND, format!("update failed: {}", e)))?;

    // Only update vector index if new vector data was provided
    if let Some(ref values) = req.values {
        let _ = coll.index.remove(id);
        coll.index.add(id, values)
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("index update failed: {}", e)))?;
    }

    // Update metadata index if metadata was provided
    if let Some(ref meta) = core_metadata {
        coll.index_manager.remove_record(id);
        coll.index_manager.index_record(id, meta);
    }

    // Persist to storage layer (best-effort)
    {
        let mut cm = state.collection_manager.write();
        if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
            let storage_data = req.values.map(VectorData::F32);
            if let Err(e) = storage_coll.update(id, storage_data, core_metadata) {
                tracing::warn!(collection = %collection, id = id, "storage update failed: {}", e);
            }
        }
    }

    Ok(Json(UpdateVectorRes { success: true }))
}

async fn delete_vector(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, u64)>,
) -> Result<Json<DeleteVectorRes>, (StatusCode, Json<ErrorResponse>)> {
    let mut collections = state.collections.write();
    let coll = collections.get_mut(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    coll.store.delete(id)
        .map_err(|e| err(StatusCode::NOT_FOUND, format!("delete failed: {}", e)))?;

    let _ = coll.index.remove(id);
    coll.index_manager.remove_record(id);
    coll.graph.remove_node(id);

    // Persist to storage layer (best-effort)
    {
        let mut cm = state.collection_manager.write();
        if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
            if let Err(e) = storage_coll.delete(id) {
                tracing::warn!(collection = %collection, id = id, "storage delete failed: {}", e);
            }
        }
    }

    Ok(Json(DeleteVectorRes { success: true }))
}

async fn bulk_insert(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<BulkInsertReq>,
) -> Result<Json<BulkInsertRes>, (StatusCode, Json<ErrorResponse>)> {
    let mut inserted_count: u64 = 0;
    let mut errors: Vec<String> = Vec::new();
    let mut inserted_ids: Vec<u64> = Vec::new();
    let mut inserted_vectors: HashMap<u64, Vec<f32>> = HashMap::new();

    // Validate and extract optimization parameters
    let index_mode = req.index_mode.as_deref().unwrap_or("immediate");
    validate_index_mode(index_mode)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;

    let raw_batch_lock_size = req.batch_lock_size.unwrap_or(1);
    validate_batch_lock_size(raw_batch_lock_size, state.max_batch_lock_size)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;

    let raw_wal_flush_every = req.wal_flush_every.unwrap_or(1);
    validate_wal_flush_every(raw_wal_flush_every, state.max_wal_flush_interval)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;

    if let Some(ef) = req.ef_construction {
        if ef > 0 {
            validate_ef_construction(ef, state.max_ef_construction)
                .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
            tracing::info!(ef_construction = ef, "ef_construction override for bulk insert");
        }
    }

    let parallel_build = req.parallel_build.unwrap_or(false);
    validate_bulk_insert_options(parallel_build, index_mode)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;

    let defer_graph = req.defer_graph.unwrap_or(false);
    if defer_graph {
        tracing::warn!("defer_graph enabled: search may return stale results until optimize() is called");
    }

    // Reject empty vector list early
    if req.vectors.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "vectors list is empty".to_string()));
    }

    let batch_lock_size = raw_batch_lock_size.max(1) as usize;
    let wal_flush_every = raw_wal_flush_every as usize;
    let _ef_construction = req.ef_construction; // TODO: HNSW parameter override
    let deferred_index_mode = index_mode == "deferred";
    let skip_metadata_index = req.skip_metadata_index.unwrap_or(false);
    let _parallel_build = parallel_build; // stored for optimize()

    // Verify collection exists before iterating
    {
        let collections = state.collections.read();
        if !collections.contains_key(&collection) {
            return Err(err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)));
        }
    }

    // Pre-validate and parse all vectors before acquiring locks
    struct PreparedVector {
        index: usize,
        id: u64,
        values: Vec<f32>,
        metadata: Option<Metadata>,
    }

    let mut prepared: Vec<PreparedVector> = Vec::with_capacity(req.vectors.len());
    for (i, v) in req.vectors.into_iter().enumerate() {
        if v.values.is_empty() {
            errors.push(format!("item {}: missing or empty vector", i));
            continue;
        }

        let core_metadata = match v.metadata.as_ref().map(json_to_metadata).transpose() {
            Ok(m) => m,
            Err(e) => {
                errors.push(format!("item {}: {}", i, e));
                continue;
            }
        };

        prepared.push(PreparedVector {
            index: i,
            id: v.id,
            values: v.values,
            metadata: core_metadata,
        });
    }

    // Process vectors in batches (batch_lock_size controls how many vectors
    // are inserted per lock acquisition to reduce lock contention)
    let mut wal_counter: usize = 0;
    let mut pending_wal: Vec<(u64, Vec<f32>, Option<Metadata>)> = Vec::new();

    for chunk in prepared.chunks(batch_lock_size.max(1)) {
        // Acquire write lock once per batch
        let mut batch_assigned: Vec<(u64, Vec<f32>, Option<Metadata>)> = Vec::new();

        {
            let mut collections = state.collections.write();
            let coll = match collections.get_mut(&collection) {
                Some(c) => c,
                None => {
                    for pv in chunk {
                        errors.push(format!("item {}: collection '{}' not found", pv.index, collection));
                    }
                    continue;
                }
            };

            for pv in chunk {
                if pv.values.len() != coll.config.dimension {
                    errors.push(format!(
                        "item {}: vector dimension mismatch: expected {}, got {}",
                        pv.index, coll.config.dimension, pv.values.len()
                    ));
                    continue;
                }

                let vector_data = VectorData::F32(pv.values.clone());

                let assigned_id = if pv.id == 0 {
                    match coll.store.insert_auto_id(vector_data, pv.metadata.clone()) {
                        Ok(id) => id,
                        Err(e) => {
                            errors.push(format!("item {}: store insert failed: {}", pv.index, e));
                            continue;
                        }
                    }
                } else {
                    let record = VectorRecord::new(pv.id, vector_data, pv.metadata.clone());
                    match coll.store.insert(record) {
                        Ok(()) => pv.id,
                        Err(e) => {
                            errors.push(format!("item {}: store insert failed for id {}: {}", pv.index, pv.id, e));
                            continue;
                        }
                    }
                };

                // Index: skip if deferred mode
                if !deferred_index_mode {
                    if let Err(e) = coll.index.add(assigned_id, &pv.values) {
                        let _ = coll.store.delete(assigned_id);
                        errors.push(format!("item {}: index insert failed: {}", pv.index, e));
                        continue;
                    }
                }

                // Metadata index: skip if flag set
                if !skip_metadata_index {
                    if let Some(ref meta) = pv.metadata {
                        coll.index_manager.index_record(assigned_id, meta);
                    }
                }

                // Graph: compute per-vector only if not deferred
                if !defer_graph {
                    if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
                        &mut coll.graph, &coll.index, assigned_id, &pv.values, 10,
                    ) {
                        tracing::warn!(collection = %collection, id = assigned_id, "graph compute failed: {}", e);
                    }
                }

                batch_assigned.push((assigned_id, pv.values.clone(), pv.metadata.clone()));
                inserted_ids.push(assigned_id);
                inserted_vectors.insert(assigned_id, pv.values.clone());
                inserted_count += 1;
            }
        } // write lock released

        // WAL persistence with batched flushing
        for (assigned_id, values, core_metadata) in batch_assigned {
            pending_wal.push((assigned_id, values, core_metadata));
            wal_counter += 1;

            if wal_flush_every <= 1 || wal_counter >= wal_flush_every {
                let mut cm = state.collection_manager.write();
                if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
                    for (id, vals, meta) in pending_wal.drain(..) {
                        if let Err(e) = storage_coll.insert(id, VectorData::F32(vals), meta) {
                            tracing::warn!(collection = %collection, id = id, "storage bulk insert failed: {}", e);
                        }
                    }
                }
                wal_counter = 0;
            }
        }
    }

    // Flush any remaining WAL entries
    if !pending_wal.is_empty() {
        let mut cm = state.collection_manager.write();
        if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
            for (id, vals, meta) in pending_wal.drain(..) {
                if let Err(e) = storage_coll.insert(id, VectorData::F32(vals), meta) {
                    tracing::warn!(collection = %collection, id = id, "storage bulk insert failed: {}", e);
                }
            }
        }
    }

    // Batch graph recomputation (only when graph is NOT deferred)
    if !defer_graph && !inserted_ids.is_empty() {
        let mut collections = state.collections.write();
        if let Some(coll) = collections.get_mut(&collection) {
            if let Err(e) = vf_graph::RelationshipComputer::compute_batch(
                &mut coll.graph, &coll.index, &inserted_ids, &inserted_vectors, 10,
            ) {
                tracing::warn!(collection = %collection, "graph compute_batch after bulk_insert failed: {}", e);
            }
        }
    }

    // Set deferred flags and update collection status if any optimization was deferred
    if inserted_count > 0 && (deferred_index_mode || defer_graph || skip_metadata_index) {
        let collections = state.collections.read();
        if let Some(coll) = collections.get(&collection) {
            if deferred_index_mode {
                coll.deferred_index.store(true, Ordering::Release);
            }
            if defer_graph {
                coll.deferred_graph.store(true, Ordering::Release);
            }
            if skip_metadata_index {
                coll.deferred_metadata.store(true, Ordering::Release);
            }
            // Mark collection as pending optimization
            if let Ok(mut status) = coll.status.write() {
                *status = CollectionStatus::PendingOptimization;
            }
        }
    }

    Ok(Json(BulkInsertRes { inserted_count, errors }))
}

// ── Optimize handler ────────────────────────────────────────────────────

#[derive(Serialize)]
struct OptimizeRes {
    status: String,
    message: String,
    duration_ms: u64,
    vectors_processed: u64,
}

async fn optimize_collection(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<OptimizeRes>, (StatusCode, Json<ErrorResponse>)> {
    match state.optimize_collection(&collection) {
        Ok(result) => Ok(Json(OptimizeRes {
            status: result.status,
            message: result.message,
            duration_ms: result.duration_ms,
            vectors_processed: result.vectors_processed,
        })),
        Err(e) if e.contains("not found") => Err(err(StatusCode::NOT_FOUND, e)),
        Err(e) if e.contains("already being optimized") => {
            Err(err(StatusCode::CONFLICT, e))
        }
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

// ── Search handlers ─────────────────────────────────────────────────────

async fn search(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<SearchReq>,
) -> Result<Json<SearchRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();

    if req.query.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "query vector is required"));
    }

    let filter = req.filter.as_ref().map(convert_json_filter)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;

    let strategy = parse_strategy(&req.strategy);

    let ef_search = validate_ef_search(req.ef_search, state.max_ef_search)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;

    if let Some(ef) = ef_search {
        tracing::debug!(ef_search = ?ef, collection = %collection, "per-query ef_search override");
        metrics::record_ef_search(ef, &collection);
    }

    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    if req.query.len() != coll.config.dimension {
        return Err(err(StatusCode::BAD_REQUEST, format!("query dimension mismatch: expected {}, got {}", coll.config.dimension, req.query.len())));
    }

    let metadata_store = coll.metadata_cache.get_or_rebuild(&coll.store);

    // Check collection status for stale results warning
    let warning = if let Ok(status) = coll.status.read() {
        match *status {
            CollectionStatus::PendingOptimization => {
                "collection has pending optimizations; results may be stale or incomplete. Call optimize() to rebuild indexes.".to_string()
            }
            _ => String::new(),
        }
    } else {
        String::new()
    };

    let results = QueryExecutor::search(
        &coll.index as &dyn VectorIndex,
        &req.query,
        req.k as usize,
        filter.as_ref(),
        &strategy,
        Some(&coll.index_manager),
        &metadata_store,
        ef_search,
    ).map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("search error: {}", e)))?;

    let max_edges = if req.max_graph_edges == 0 { 10u32 } else { req.max_graph_edges };
    let graph_threshold = if req.graph_threshold == 0.0 { None } else { Some(req.graph_threshold) };

    // Batch lookup all graph edges at once instead of per-result
    let graph_edges_map = if req.include_graph {
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        RelationshipQueryEngine::get_related_batch(&coll.graph, &ids, graph_threshold, max_edges)
    } else {
        std::collections::HashMap::new()
    };

    let scored = results.into_iter().map(|r| {
        let metadata = if req.include_metadata {
            metadata_store.get(&r.id).map(metadata_to_json)
        } else {
            None
        };
        let graph_edges = graph_edges_map
            .get(&r.id)
            .map(|edges| {
                edges.iter().map(|&(target_id, similarity)| GraphEdge { target_id, similarity }).collect()
            })
            .unwrap_or_default();
        ScoredResult { id: r.id, score: r.score, metadata, graph_edges }
    }).collect();

    Ok(Json(SearchRes { results: scored, search_time_us: timer.elapsed().as_micros() as u64, warning }))
}

async fn batch_search(
    State(state): State<AppState>,
    ValidatedJson(req): ValidatedJson<BatchSearchReq>,
) -> Result<Json<BatchSearchRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();

    if req.queries.is_empty() {
        return Ok(Json(BatchSearchRes { results: vec![], total_time_us: 0 }));
    }

    let collections = state.collections.read();
    let mut responses = Vec::with_capacity(req.queries.len());

    for q in &req.queries {
        let query_timer = Instant::now();

        let coll = collections.get(&q.collection)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", q.collection)))?;

        if q.query.is_empty() {
            return Err(err(StatusCode::BAD_REQUEST, "query vector is required"));
        }

        if q.query.len() != coll.config.dimension {
            return Err(err(StatusCode::BAD_REQUEST, format!("query dimension mismatch: expected {}, got {}", coll.config.dimension, q.query.len())));
        }

        let filter = q.filter.as_ref().map(convert_json_filter)
            .transpose()
            .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;

        let strategy = parse_strategy(&q.strategy);

        let ef_search = validate_ef_search(q.ef_search, state.max_ef_search)
            .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;

        if let Some(ef) = ef_search {
            tracing::debug!(ef_search = ?ef, collection = %q.collection, "per-query ef_search override");
            metrics::record_ef_search(ef, &q.collection);
        }

        let metadata_store = coll.metadata_cache.get_or_rebuild(&coll.store);

        // Check collection status for stale results warning
        let warning = if let Ok(status) = coll.status.read() {
            match *status {
                CollectionStatus::PendingOptimization => {
                    "collection has pending optimizations; results may be stale or incomplete. Call optimize() to rebuild indexes.".to_string()
                }
                _ => String::new(),
            }
        } else {
            String::new()
        };

        let results = QueryExecutor::search(
            &coll.index as &dyn VectorIndex,
            &q.query,
            q.k as usize,
            filter.as_ref(),
            &strategy,
            Some(&coll.index_manager),
            &metadata_store,
            ef_search,
        ).map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("search error: {}", e)))?;

        let max_edges = if q.max_graph_edges == 0 { 10u32 } else { q.max_graph_edges };
        let graph_threshold = if q.graph_threshold == 0.0 { None } else { Some(q.graph_threshold) };

        // Batch lookup all graph edges at once instead of per-result
        let graph_edges_map = if q.include_graph {
            let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
            RelationshipQueryEngine::get_related_batch(&coll.graph, &ids, graph_threshold, max_edges)
        } else {
            std::collections::HashMap::new()
        };

        let scored = results.into_iter().map(|r| {
            let metadata = if q.include_metadata {
                metadata_store.get(&r.id).map(metadata_to_json)
            } else {
                None
            };
            let graph_edges = graph_edges_map
                .get(&r.id)
                .map(|edges| {
                    edges.iter().map(|&(target_id, similarity)| GraphEdge { target_id, similarity }).collect()
                })
                .unwrap_or_default();
            ScoredResult { id: r.id, score: r.score, metadata, graph_edges }
        }).collect();

        responses.push(SearchRes { results: scored, search_time_us: query_timer.elapsed().as_micros() as u64, warning });
    }

    Ok(Json(BatchSearchRes { results: responses, total_time_us: timer.elapsed().as_micros() as u64 }))
}

// ── Graph handlers ──────────────────────────────────────────────────────

async fn get_related(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, u64)>,
    Query(params): Query<GetRelatedQuery>,
) -> Result<Json<GetRelatedRes>, (StatusCode, Json<ErrorResponse>)> {
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    let threshold = params.threshold.filter(|&t| t > 0.0);

    let related = RelationshipQueryEngine::get_related(&coll.graph, id, threshold)
        .map_err(|e| {
            let msg = format!("graph error: {}", e);
            if msg.contains("not found") {
                err(StatusCode::NOT_FOUND, msg)
            } else {
                err(StatusCode::INTERNAL_SERVER_ERROR, msg)
            }
        })?;

    let mut edges: Vec<GraphEdge> = related.into_iter()
        .map(|(target_id, similarity)| GraphEdge { target_id, similarity })
        .collect();

    if let Some(max) = params.max_results {
        if max > 0 {
            edges.truncate(max as usize);
        }
    }

    Ok(Json(GetRelatedRes { edges }))
}

async fn traverse(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<TraverseReq>,
) -> Result<Json<TraverseRes>, (StatusCode, Json<ErrorResponse>)> {
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    let threshold = req.threshold.filter(|&t| t > 0.0);
    let max_results = req.max_results.filter(|&m| m > 0).map(|m| m as usize);

    let traversal_results = GraphTraversal::traverse(
        &coll.graph,
        req.start_id,
        &TraversalOrder::BreadthFirst,
        req.depth as usize,
        threshold,
        max_results,
    ).map_err(|e| {
        let msg = format!("traversal error: {}", e);
        if msg.contains("not found") {
            err(StatusCode::NOT_FOUND, msg)
        } else {
            err(StatusCode::INTERNAL_SERVER_ERROR, msg)
        }
    })?;

    let nodes = traversal_results.into_iter().map(|r| TraversalNode {
        id: r.id,
        depth: r.depth as u32,
        path_similarity: r.path_similarity,
        path: r.path,
    }).collect();

    Ok(Json(TraverseRes { nodes }))
}

async fn set_threshold(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<SetThresholdReq>,
) -> Result<Json<SetThresholdRes>, (StatusCode, Json<ErrorResponse>)> {
    let mut collections = state.collections.write();
    let coll = collections.get_mut(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    if req.vector_id == 0 {
        coll.graph.config_mut().default_threshold = req.threshold;
    } else {
        coll.graph.set_vector_threshold(req.vector_id, req.threshold);
    }

    Ok(Json(SetThresholdRes { success: true }))
}

// ── Helpers: metadata conversion ────────────────────────────────────────

fn json_to_metadata(value: &serde_json::Value) -> Result<Metadata, String> {
    let obj = value.as_object()
        .ok_or_else(|| "metadata must be a JSON object".to_string())?;
    let mut meta = HashMap::new();
    for (k, v) in obj {
        let mv = json_value_to_metadata_value(v)
            .ok_or_else(|| format!("unsupported metadata type for key '{}'", k))?;
        meta.insert(k.clone(), mv);
    }
    Ok(meta)
}

fn json_value_to_metadata_value(v: &serde_json::Value) -> Option<MetadataValue> {
    match v {
        serde_json::Value::String(s) => Some(MetadataValue::String(s.clone())),
        serde_json::Value::Bool(b) => Some(MetadataValue::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(MetadataValue::Int(i))
            } else {
                n.as_f64().map(|f| MetadataValue::Float(f))
            }
        }
        serde_json::Value::Array(arr) => {
            let strings: Option<Vec<String>> = arr.iter().map(|v| v.as_str().map(|s| s.to_string())).collect();
            strings.map(MetadataValue::StringList)
        }
        _ => None,
    }
}

fn metadata_to_json(meta: &Metadata) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in meta {
        let jv = match v {
            MetadataValue::String(s) => serde_json::Value::String(s.clone()),
            MetadataValue::Int(i) => serde_json::json!(*i),
            MetadataValue::Float(f) => serde_json::json!(*f),
            MetadataValue::Bool(b) => serde_json::Value::Bool(*b),
            MetadataValue::StringList(list) => serde_json::json!(list),
        };
        map.insert(k.clone(), jv);
    }
    serde_json::Value::Object(map)
}

// ── Helpers: filter conversion ──────────────────────────────────────────

fn convert_json_filter(value: &serde_json::Value) -> Result<FilterExpression, String> {
    let obj = value.as_object()
        .ok_or_else(|| "filter must be a JSON object".to_string())?;

    if let Some(and_val) = obj.get("and") {
        let arr = and_val.as_array()
            .ok_or_else(|| "'and' must be an array".to_string())?;
        let children: Result<Vec<_>, _> = arr.iter().map(convert_json_filter).collect();
        return Ok(FilterExpression::And(children?));
    }
    if let Some(or_val) = obj.get("or") {
        let arr = or_val.as_array()
            .ok_or_else(|| "'or' must be an array".to_string())?;
        let children: Result<Vec<_>, _> = arr.iter().map(convert_json_filter).collect();
        return Ok(FilterExpression::Or(children?));
    }
    if let Some(not_val) = obj.get("not") {
        let child = convert_json_filter(not_val)?;
        return Ok(FilterExpression::Not(Box::new(child)));
    }

    let field = obj.get("field")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "filter must have a 'field' string".to_string())?
        .to_string();
    let op = obj.get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "filter must have an 'op' string".to_string())?;

    match op {
        "eq" => {
            let v = obj.get("value").ok_or("'eq' requires a 'value'")?;
            let mv = json_value_to_metadata_value(v)
                .ok_or_else(|| "unsupported value type for 'eq'".to_string())?;
            Ok(FilterExpression::Eq(field, mv))
        }
        "ne" => {
            let v = obj.get("value").ok_or("'ne' requires a 'value'")?;
            let mv = json_value_to_metadata_value(v)
                .ok_or_else(|| "unsupported value type for 'ne'".to_string())?;
            Ok(FilterExpression::Ne(field, mv))
        }
        "gt" => {
            let v = obj.get("value").ok_or("'gt' requires a 'value'")?;
            let mv = json_value_to_metadata_value(v)
                .ok_or_else(|| "unsupported value type for 'gt'".to_string())?;
            Ok(FilterExpression::Gt(field, mv))
        }
        "gte" => {
            let v = obj.get("value").ok_or("'gte' requires a 'value'")?;
            let mv = json_value_to_metadata_value(v)
                .ok_or_else(|| "unsupported value type for 'gte'".to_string())?;
            Ok(FilterExpression::Gte(field, mv))
        }
        "lt" => {
            let v = obj.get("value").ok_or("'lt' requires a 'value'")?;
            let mv = json_value_to_metadata_value(v)
                .ok_or_else(|| "unsupported value type for 'lt'".to_string())?;
            Ok(FilterExpression::Lt(field, mv))
        }
        "lte" => {
            let v = obj.get("value").ok_or("'lte' requires a 'value'")?;
            let mv = json_value_to_metadata_value(v)
                .ok_or_else(|| "unsupported value type for 'lte'".to_string())?;
            Ok(FilterExpression::Lte(field, mv))
        }
        "in" => {
            let arr = obj.get("values")
                .and_then(|v| v.as_array())
                .ok_or("'in' requires a 'values' array")?;
            let values: Result<Vec<_>, _> = arr.iter()
                .map(|v| json_value_to_metadata_value(v).ok_or_else(|| "unsupported value in 'in' array".to_string()))
                .collect();
            Ok(FilterExpression::In(field, values?))
        }
        "between" => {
            let arr = obj.get("values")
                .and_then(|v| v.as_array())
                .ok_or("'between' requires a 'values' array")?;
            if arr.len() != 2 {
                return Err("'between' requires exactly 2 values".to_string());
            }
            let low = json_value_to_metadata_value(&arr[0])
                .ok_or_else(|| "unsupported low value in 'between'".to_string())?;
            let high = json_value_to_metadata_value(&arr[1])
                .ok_or_else(|| "unsupported high value in 'between'".to_string())?;
            Ok(FilterExpression::Between(field, low, high))
        }
        "exists" => Ok(FilterExpression::Exists(field)),
        "contains" => {
            let v = obj.get("value")
                .and_then(|v| v.as_str())
                .ok_or("'contains' requires a string 'value'")?;
            Ok(FilterExpression::Contains(field, v.to_string()))
        }
        other => Err(format!("unsupported filter op: '{}'", other)),
    }
}

// ── Helpers: search ─────────────────────────────────────────────────────

fn parse_strategy(s: &str) -> FilterStrategy {
    match s {
        "pre_filter" => FilterStrategy::PreFilter,
        "post_filter" => FilterStrategy::PostFilter { oversample_factor: 3 },
        _ => FilterStrategy::Auto,
    }
}

// ── Vector Math helpers ─────────────────────────────────────────────────

fn get_vectors_from_collection(store: &InMemoryVectorStore, ids: &[u64]) -> Vec<(VectorId, Vec<f32>)> {
    if ids.is_empty() {
        store.iter_vector_data()
    } else {
        ids.iter().filter_map(|&id| {
            store.get(id).ok().map(|record| (id, record.data.to_f32_vec()))
        }).collect()
    }
}

fn resolve_metric(metric: &str) -> DistanceMetricType {
    if metric.is_empty() {
        DistanceMetricType::Euclidean
    } else {
        parse_distance_metric(metric).unwrap_or(DistanceMetricType::Euclidean)
    }
}

// ── Vector Math types ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct DetectGhostsReq {
    #[serde(default = "default_ghost_threshold")]
    threshold: f32,
    #[serde(default)]
    centroids: Option<Vec<Vec<f32>>>,
    #[serde(default = "default_auto_k")]
    auto_k: u32,
    #[serde(default = "default_euclidean")]
    metric: String,
}
fn default_ghost_threshold() -> f32 { 0.0 }
fn default_auto_k() -> u32 { 8 }
fn default_euclidean() -> String { "euclidean".to_string() }

#[derive(Serialize)]
struct GhostVectorRes { id: u64, isolation_score: f32 }

#[derive(Serialize)]
struct DetectGhostsRes { ghosts: Vec<GhostVectorRes>, compute_time_us: u64 }

#[derive(Deserialize)]
struct ConeSearchReq {
    direction: Vec<f32>,
    aperture_radians: f32,
}

#[derive(Serialize)]
struct ConeSearchResultRes { id: u64, cosine_similarity: f32, angle_radians: f32 }

#[derive(Serialize)]
struct ConeSearchRes { results: Vec<ConeSearchResultRes>, compute_time_us: u64 }

#[derive(Deserialize)]
struct ComputeCentroidReq {
    #[serde(default)]
    vector_ids: Vec<u64>,
    #[serde(default)]
    weights: Vec<f32>,
}

#[derive(Serialize)]
struct ComputeCentroidRes { centroid: Vec<f32>, compute_time_us: u64 }

fn default_lerp() -> String { "lerp".to_string() }

#[derive(Deserialize)]
struct InterpolateReq {
    a: Vec<f32>,
    b: Vec<f32>,
    #[serde(default)]
    t: f32,
    #[serde(default = "default_lerp")]
    method: String,
    #[serde(default)]
    sequence_count: u32,
}

#[derive(Serialize)]
struct InterpolateRes { results: Vec<Vec<f32>>, compute_time_us: u64 }

#[derive(Deserialize)]
struct DetectDriftReq {
    window1_ids: Vec<u64>,
    window2_ids: Vec<u64>,
    #[serde(default = "default_euclidean")]
    metric: String,
    #[serde(default)]
    threshold: Option<f32>,
}

#[derive(Serialize)]
struct DetectDriftRes {
    centroid_shift: f32,
    mean_distance_window1: f32,
    mean_distance_window2: f32,
    spread_change: f32,
    has_drifted: bool,
    compute_time_us: u64,
}

#[derive(Deserialize)]
struct ClusterReq {
    k: u32,
    #[serde(default = "default_max_iterations")]
    max_iterations: u32,
    #[serde(default = "default_tolerance")]
    tolerance: f32,
    #[serde(default = "default_euclidean")]
    metric: String,
}
fn default_max_iterations() -> u32 { 100 }
fn default_tolerance() -> f32 { 1e-4 }

#[derive(Serialize)]
struct ClusterAssignmentRes { id: u64, cluster: u32, distance_to_centroid: f32 }

#[derive(Serialize)]
struct ClusterRes {
    centroids: Vec<Vec<f32>>,
    assignments: Vec<ClusterAssignmentRes>,
    iterations: u32,
    converged: bool,
    compute_time_us: u64,
}

#[derive(Deserialize)]
struct ReduceDimensionsReq {
    #[serde(default = "default_n_components")]
    n_components: u32,
    #[serde(default)]
    vector_ids: Vec<u64>,
}
fn default_n_components() -> u32 { 2 }

#[derive(Serialize)]
struct ReduceDimensionsRes {
    components: Vec<Vec<f32>>,
    explained_variance: Vec<f32>,
    mean: Vec<f32>,
    projected: Vec<Vec<f32>>,
    compute_time_us: u64,
}

#[derive(Deserialize)]
struct AnalogyTerm {
    vector: Vec<f32>,
    weight: f32,
}

#[derive(Deserialize)]
struct ComputeAnalogyReq {
    #[serde(default)]
    a: Option<Vec<f32>>,
    #[serde(default)]
    b: Option<Vec<f32>>,
    #[serde(default)]
    c: Option<Vec<f32>>,
    #[serde(default)]
    normalize: bool,
    #[serde(default)]
    terms: Vec<AnalogyTerm>,
}

#[derive(Serialize)]
struct ComputeAnalogyRes { result: Vec<f32>, compute_time_us: u64 }

#[derive(Deserialize)]
struct DiversitySampleReq {
    query: Vec<f32>,
    k: u32,
    lambda: f32,
    #[serde(default)]
    candidate_ids: Vec<u64>,
}

#[derive(Serialize)]
struct DiversitySampleResultRes { id: u64, relevance_score: f32, mmr_score: f32 }

#[derive(Serialize)]
struct DiversitySampleRes { results: Vec<DiversitySampleResultRes>, compute_time_us: u64 }

// ── Vector Math handlers ────────────────────────────────────────────────

async fn detect_ghosts(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<DetectGhostsReq>,
) -> Result<Json<DetectGhostsRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    let owned_vectors = get_vectors_from_collection(&coll.store, &[]);
    let vectors: Vec<(VectorId, &[f32])> = owned_vectors
        .iter()
        .map(|(id, v)| (*id, v.as_slice()))
        .collect();

    let centroids: Vec<Vec<f32>> = match req.centroids {
        Some(ref c) if !c.is_empty() => c.clone(),
        _ => {
            let auto_k = if req.auto_k == 0 { 8 } else { req.auto_k as usize };
            let metric = resolve_metric(&req.metric);
            let config = KMeansConfig {
                k: auto_k,
                metric,
                ..Default::default()
            };
            let km = KMeans::new(config);
            let result = km.cluster(&vectors);
            result.centroids
        }
    };

    let metric = resolve_metric(&req.metric);
    let detector = GhostDetector::new(req.threshold, metric);
    let ghosts = detector.detect(&vectors, &centroids);

    Ok(Json(DetectGhostsRes {
        ghosts: ghosts.into_iter().map(|g| GhostVectorRes {
            id: g.id,
            isolation_score: g.isolation_score,
        }).collect(),
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}

async fn cone_search(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<ConeSearchReq>,
) -> Result<Json<ConeSearchRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    if req.direction.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "direction vector is required"));
    }

    let owned_vectors = get_vectors_from_collection(&coll.store, &[]);
    let vectors: Vec<(VectorId, &[f32])> = owned_vectors
        .iter()
        .map(|(id, v)| (*id, v.as_slice()))
        .collect();

    let results = ConeSearch::search(&req.direction, req.aperture_radians, &vectors);

    Ok(Json(ConeSearchRes {
        results: results.into_iter().map(|r| ConeSearchResultRes {
            id: r.id,
            cosine_similarity: r.cosine_similarity,
            angle_radians: r.angle_radians,
        }).collect(),
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}

async fn compute_centroid(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<ComputeCentroidReq>,
) -> Result<Json<ComputeCentroidRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    let owned_vectors = get_vectors_from_collection(&coll.store, &req.vector_ids);
    let vec_slices: Vec<&[f32]> = owned_vectors.iter().map(|(_, v)| v.as_slice()).collect();

    if vec_slices.is_empty() {
        return Err(err(StatusCode::NOT_FOUND, "no vectors found"));
    }

    let centroid = if !req.weights.is_empty() {
        CentroidComputer::compute_weighted(&vec_slices, &req.weights)
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, "weighted centroid computation failed (dimension or weight mismatch)"))?
    } else {
        CentroidComputer::compute(&vec_slices)
            .ok_or_else(|| err(StatusCode::INTERNAL_SERVER_ERROR, "centroid computation failed"))?
    };

    Ok(Json(ComputeCentroidRes {
        centroid,
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}

async fn interpolate(
    State(_state): State<AppState>,
    ValidatedJson(req): ValidatedJson<InterpolateReq>,
) -> Result<Json<InterpolateRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();

    if req.a.is_empty() || req.b.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "vectors 'a' and 'b' are required"));
    }

    let method = if req.method.is_empty() { "lerp" } else { &req.method };

    let results = if req.sequence_count > 0 {
        let n = req.sequence_count as usize;
        match method {
            "slerp" => Interpolator::slerp_sequence(&req.a, &req.b, n)
                .ok_or_else(|| err(StatusCode::BAD_REQUEST, "slerp sequence failed (dimension mismatch or invalid input)"))?,
            _ => Interpolator::lerp_sequence(&req.a, &req.b, n)
                .ok_or_else(|| err(StatusCode::BAD_REQUEST, "lerp sequence failed (dimension mismatch or invalid input)"))?,
        }
    } else {
        let result = match method {
            "slerp" => Interpolator::slerp(&req.a, &req.b, req.t)
                .ok_or_else(|| err(StatusCode::BAD_REQUEST, "slerp failed (dimension mismatch or t out of range)"))?,
            _ => Interpolator::lerp(&req.a, &req.b, req.t)
                .ok_or_else(|| err(StatusCode::BAD_REQUEST, "lerp failed (dimension mismatch or t out of range)"))?,
        };
        vec![result]
    };

    Ok(Json(InterpolateRes {
        results,
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}

async fn detect_drift(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<DetectDriftReq>,
) -> Result<Json<DetectDriftRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    let owned_w1 = get_vectors_from_collection(&coll.store, &req.window1_ids);
    let owned_w2 = get_vectors_from_collection(&coll.store, &req.window2_ids);

    let w1_slices: Vec<&[f32]> = owned_w1.iter().map(|(_, v)| v.as_slice()).collect();
    let w2_slices: Vec<&[f32]> = owned_w2.iter().map(|(_, v)| v.as_slice()).collect();

    let metric = resolve_metric(&req.metric);
    let detector = DriftDetector::new(metric);

    let report = detector.detect(&w1_slices, &w2_slices)
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "drift detection failed (empty windows)"))?;

    let has_drifted = match req.threshold {
        Some(t) if t > 0.0 => report.centroid_shift > t,
        _ => false,
    };

    Ok(Json(DetectDriftRes {
        centroid_shift: report.centroid_shift,
        mean_distance_window1: report.mean_distance_window1,
        mean_distance_window2: report.mean_distance_window2,
        spread_change: report.spread_change,
        has_drifted,
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}

async fn cluster(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<ClusterReq>,
) -> Result<Json<ClusterRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    let owned_vectors = get_vectors_from_collection(&coll.store, &[]);
    let vectors: Vec<(VectorId, &[f32])> = owned_vectors
        .iter()
        .map(|(id, v)| (*id, v.as_slice()))
        .collect();

    let metric = resolve_metric(&req.metric);
    let config = KMeansConfig {
        k: req.k as usize,
        max_iterations: if req.max_iterations == 0 { 100 } else { req.max_iterations as usize },
        tolerance: if req.tolerance == 0.0 { 1e-4 } else { req.tolerance },
        metric,
    };

    let km = KMeans::new(config);
    let result = km.cluster(&vectors);

    Ok(Json(ClusterRes {
        centroids: result.centroids,
        assignments: result.assignments.into_iter().map(|a| ClusterAssignmentRes {
            id: a.id,
            cluster: a.cluster as u32,
            distance_to_centroid: a.distance_to_centroid,
        }).collect(),
        iterations: result.iterations as u32,
        converged: result.converged,
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}

async fn reduce_dimensions(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<ReduceDimensionsReq>,
) -> Result<Json<ReduceDimensionsRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    let owned_vectors = get_vectors_from_collection(&coll.store, &req.vector_ids);
    let vec_slices: Vec<&[f32]> = owned_vectors.iter().map(|(_, v)| v.as_slice()).collect();

    let n_components = if req.n_components == 0 { 2 } else { req.n_components as usize };
    let pca = Pca::new(PcaConfig {
        n_components,
        ..Default::default()
    });

    let result = pca.fit_transform(&vec_slices)
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "PCA failed (need at least 2 vectors with matching dimensions)"))?;

    Ok(Json(ReduceDimensionsRes {
        components: result.components,
        explained_variance: result.explained_variance,
        mean: result.mean,
        projected: result.projected,
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}

async fn compute_analogy(
    State(_state): State<AppState>,
    ValidatedJson(req): ValidatedJson<ComputeAnalogyReq>,
) -> Result<Json<ComputeAnalogyRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();

    let mut result = if !req.terms.is_empty() {
        let terms: Vec<(Vec<f32>, f32)> = req.terms.iter()
            .map(|t| (t.vector.clone(), t.weight))
            .collect();
        let term_refs: Vec<(&[f32], f32)> = terms.iter()
            .map(|(v, w)| (v.as_slice(), *w))
            .collect();
        AnalogyComputer::arithmetic(&term_refs)
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, "arithmetic computation failed (dimension mismatch or empty terms)"))?
    } else {
        let a = req.a.as_ref()
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, "vector 'a' is required"))?;
        let b = req.b.as_ref()
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, "vector 'b' is required"))?;
        let c = req.c.as_ref()
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, "vector 'c' is required"))?;

        AnalogyComputer::analogy(a, b, c)
            .ok_or_else(|| err(StatusCode::BAD_REQUEST, "analogy computation failed (dimension mismatch)"))?
    };

    if req.normalize {
        AnalogyComputer::normalize(&mut result);
    }

    Ok(Json(ComputeAnalogyRes {
        result,
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}

async fn diversity_sample(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<DiversitySampleReq>,
) -> Result<Json<DiversitySampleRes>, (StatusCode, Json<ErrorResponse>)> {
    let timer = Instant::now();
    let collections = state.collections.read();
    let coll = collections.get(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    if req.query.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "query vector is required"));
    }

    let owned_candidates = get_vectors_from_collection(&coll.store, &req.candidate_ids);
    let candidates: Vec<(VectorId, &[f32])> = owned_candidates
        .iter()
        .map(|(id, v)| (*id, v.as_slice()))
        .collect();

    let results = DiversitySampler::mmr(&req.query, &candidates, req.k as usize, req.lambda);

    Ok(Json(DiversitySampleRes {
        results: results.into_iter().map(|r| DiversitySampleResultRes {
            id: r.id,
            relevance_score: r.relevance_score,
            mmr_score: r.mmr_score,
        }).collect(),
        compute_time_us: timer.elapsed().as_micros() as u64,
    }))
}
