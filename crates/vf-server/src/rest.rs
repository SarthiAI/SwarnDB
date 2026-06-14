// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::extract::{Path, Query, State, FromRequest};
use axum::http::StatusCode;
use axum::routing::{delete, get, patch, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use vf_core::store::InMemoryVectorStore;
use vf_core::types::{
    CollectionConfig, DataTypeConfig, DistanceMetricType, Metadata, MetadataValue, Mode,
    SearchQuantizationParams, VectorId,
};
use vf_core::vector::VectorData;
use vf_graph::{GraphStore, GraphTraversal, RelationshipQueryEngine, TraversalOrder};
use vf_query::vector_math::*;
use vf_query::{FilterExpression, FilterStrategy, IndexManager, QueryExecutor};

use crate::bulk_checkpoint_token::{
    encode_resume_token, hash_collection_name, resolve_bulk_checkpoint_path,
};
use crate::convert::{build_hnsw_params, distance_metric_to_string, parse_distance_metric};
use crate::metrics;
use crate::state::{
    metered_read, metered_write, AppState, CollectionAvailability, CollectionState,
    CollectionStatus, MetadataCache,
};
use crate::validation::{
    request_body_limit_bytes, request_size_limit_layer, validate_batch_lock_size,
    validate_bulk_insert_options, validate_collection_name, validate_ef_construction,
    validate_ef_search, validate_index_mode, validate_wal_flush_every, ValidationConfig,
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

/// Convert a `CollectionAvailability` returned by the per-endpoint readiness
/// guard into the `(status, body)` shape used by every REST handler. A
/// recovering collection becomes 503; a missing collection becomes 404. The
/// 503 body intentionally surfaces the load progress so clients can choose a
/// sensible backoff instead of treating the response as a hard failure.
fn err_from_availability(
    avail: CollectionAvailability,
) -> (StatusCode, Json<ErrorResponse>) {
    match avail {
        CollectionAvailability::Recovering { .. } => {
            err(StatusCode::SERVICE_UNAVAILABLE, avail.user_message())
        }
        CollectionAvailability::NotFound { .. } => {
            err(StatusCode::NOT_FOUND, avail.user_message())
        }
    }
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
pub struct QuantizationReq {
    pub r#type: String,  // "scalar"
    pub quantile: Option<f32>,
    pub always_ram: Option<bool>,
}

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
    pub quantization: Option<QuantizationReq>,
    // Optional HNSW build parameters; when set, override server defaults.
    #[serde(default)]
    pub m: Option<u32>,
    #[serde(default)]
    pub ef_construction: Option<u32>,
    #[serde(default)]
    pub mode: Option<String>,
}

fn default_distance() -> String { "cosine".to_string() }

/// Parse the optional REST mode string into the core Mode. None defaults to
/// VectorOnly (the new-collection default); an unknown value is a 400.
fn parse_mode(s: Option<&str>) -> Result<Mode, (StatusCode, Json<ErrorResponse>)> {
    match s {
        None => Ok(Mode::VectorOnly),
        Some("vector_only") => Ok(Mode::VectorOnly),
        Some("auto_similarity") => Ok(Mode::AutoSimilarity),
        Some("hybrid") => Ok(Mode::Hybrid),
        Some(other) => Err(err(StatusCode::BAD_REQUEST, format!("unknown mode: {}", other))),
    }
}

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
    // Live HNSW index node count. May trail vector_count under a deferred /
    // pending-optimization build; vector_count stays the stored-row count.
    pub indexed_count: u64,
    pub default_threshold: f32,
    pub status: String,
    pub quantization_type: String,
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
    // Write a checkpoint after every N completed chunks. 0 / None = never.
    #[serde(default)]
    pub checkpoint_every: Option<u32>,
    // Opaque resume token returned by a previous partial bulk insert response.
    #[serde(default)]
    pub resume_token: Option<String>,
}

#[derive(Serialize)]
pub struct BulkInsertRes {
    pub inserted_count: u64,
    pub errors: Vec<String>,
    #[serde(default)]
    pub last_completed_batch_idx: u64,
    #[serde(default)]
    pub last_committed_lsn: u64,
    #[serde(default)]
    pub resume_token: String,
    // Server-assigned ids in input order, parallel to the Ok arms of the chunk loop.
    #[serde(default)]
    pub assigned_ids: Vec<u64>,
}

#[derive(Deserialize)]
pub struct BulkInsertFromPathReq {
    pub path: String,
    #[serde(default)]
    pub dim: u32,
    #[serde(default)]
    pub expected_count: u64,
    #[serde(default)]
    pub total_count_hint: u64,
    #[serde(default)]
    pub id_start: u64,
    #[serde(default)]
    pub ids_path: String,
    #[serde(default)]
    pub skip_metadata_index: bool,
    #[serde(default)]
    pub index_mode: String,
    #[serde(default)]
    pub ef_construction: u32,
    #[serde(default)]
    pub chunk_size: u32,
}

// ── Search types ────────────────────────────────────────────────────────

fn default_strategy() -> String { "auto".to_string() }

fn default_max_graph_edges() -> u32 { 10 }

#[derive(Deserialize)]
pub struct SearchQuantizationReq {
    #[serde(default)]
    pub rescore: Option<bool>,
    #[serde(default)]
    pub oversampling: Option<f32>,
    #[serde(default)]
    pub ignore: Option<bool>,
}

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
    #[serde(default)]
    pub quantization: Option<SearchQuantizationReq>,
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
    #[serde(default)]
    pub quantization: Option<SearchQuantizationReq>,
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

// ── Hybrid query types (P02) ────────────────────────────────────────────

/// A node row in a hybrid query response. `node` is present when the typed
/// store resolved the id; absent for an unmaterialized vector hit.
#[derive(Serialize)]
pub struct HybridNodeRes {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<vf_graph::Node>,
}

#[derive(Serialize)]
pub struct HybridQueryRes {
    pub nodes: Vec<HybridNodeRes>,
    pub edges: Vec<vf_graph::TypedEdge>,
    pub paths: Vec<Vec<u64>>,
}

// ── Typed graph CRUD types (P04) ─────────────────────────────────────────
//
// Node/edge JSON shape is the same one hybrid_query emits: we serialize
// vf_graph::Node and vf_graph::TypedEdge directly, so edges carry id, source,
// target, edge_type, properties, provenance, confidence, verified, is_manual,
// created_at, and history (action/actor/at). No separate DTO is introduced for
// the node/edge body to keep the shape identical across the API.

#[derive(Deserialize)]
pub struct PutNodeReq {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub properties: Option<serde_json::Value>,
    #[serde(default)]
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub created_by: Option<String>,
}

#[derive(Serialize)]
pub struct PutNodeRes {
    pub id: u64,
}

#[derive(Serialize)]
pub struct DeletedRes {
    pub deleted: bool,
}

#[derive(Deserialize)]
pub struct ListEdgesQuery {
    #[serde(default)]
    pub direction: Option<String>,
    #[serde(default)]
    pub edge_type: Option<String>,
}

#[derive(Serialize)]
pub struct ListEdgesRes {
    pub edges: Vec<vf_graph::TypedEdge>,
}

#[derive(Deserialize)]
pub struct PutEdgeReq {
    pub source: u64,
    pub target: u64,
    pub edge_type: String,
    #[serde(default)]
    pub properties: Option<serde_json::Value>,
    #[serde(default)]
    pub provenance: Option<serde_json::Value>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub verified: bool,
    #[serde(default)]
    pub is_manual: bool,
    // P17. Optional temporal validity window and context; absent = None.
    #[serde(default)]
    pub valid_from: Option<u64>,
    #[serde(default)]
    pub valid_until: Option<u64>,
    #[serde(default)]
    pub temporal_context: Option<String>,
}

#[derive(Serialize)]
pub struct PutEdgeRes {
    pub id: u64,
}

// Absent field = unchanged (proto3-optional parity).
#[derive(Deserialize)]
pub struct UpdateEdgeReq {
    #[serde(default)]
    pub properties: Option<serde_json::Value>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub verified: Option<bool>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Serialize)]
pub struct EdgeRes {
    pub edge: vf_graph::TypedEdge,
}

// Absent field = unchanged (proto3-optional parity). Only properties are
// mutable; provenance and the embedding are immutable.
#[derive(Deserialize)]
pub struct UpdateNodeReq {
    #[serde(default)]
    pub properties: Option<serde_json::Value>,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Serialize)]
pub struct NodeRes {
    pub node: vf_graph::Node,
}

// Body for verify/reject: an optional actor, or an empty body.
#[derive(Deserialize, Default)]
pub struct ActorReq {
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Serialize)]
pub struct RejectEdgeRes {
    pub deleted: bool,
    pub rule_added: bool,
}

#[derive(Deserialize)]
pub struct BulkImportEdgesReq {
    pub format: String,
    pub data: String,
    #[serde(default)]
    pub auto_add_edge_types: bool,
    #[serde(default)]
    pub actor: Option<String>,
}

#[derive(Serialize)]
pub struct BulkImportRowErrorRes {
    pub row: u64,
    pub message: String,
}

#[derive(Serialize)]
pub struct BulkImportEdgesRes {
    pub total_rows: u64,
    pub imported: u64,
    pub failed: u64,
    pub errors: Vec<BulkImportRowErrorRes>,
}

// ── Document diff / re-extraction types (P04) ────────────────────────────

#[derive(Deserialize)]
pub struct DiffDocumentReq {
    pub doc_id: String,
    #[serde(default)]
    pub chunks: Vec<ChunkReq>,
}

#[derive(Serialize)]
pub struct ChunkDiffRes {
    pub chunk_id: u64,
    pub action: String,
}

#[derive(Serialize)]
pub struct DiffDocumentRes {
    pub diffs: Vec<ChunkDiffRes>,
}

#[derive(Serialize)]
pub struct ReextractDocumentRes {
    pub job_id: String,
    pub unchanged: u64,
    pub changed: u64,
    pub added: u64,
    pub deleted: u64,
    pub edges_deleted: u64,
    pub nodes_deleted: u64,
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
        .route("/api/v1/collections/{collection}/vectors/bulk-from-path", post(bulk_insert_from_path))
        .route("/api/v1/collections/{collection}/optimize", post(optimize_collection))
        .route("/api/v1/collections/{collection}/prune-wal", post(prune_wal_collection))
        .route("/api/v1/collections/{collection}/compact", post(compact_collection))
        // Recovery, persistence, snapshot, and per-collection lock metrics.
        // Operational endpoints (recovery_status, metrics) sit at the top
        // level next to /health and /readyz. Collection-scoped endpoints
        // (snapshot, persistence_status) join the existing /api/v1 surface.
        .route("/recovery_status", get(recovery_status))
        .route("/api/v1/collections/{collection}/snapshot", post(snapshot_collection))
        .route("/api/v1/collections/{collection}/persistence_status", get(persistence_status))
        .route("/metrics/collection/{collection}", get(collection_metrics))
        // Search
        .route("/api/v1/collections/{collection}/search", post(search))
        .route("/api/v1/search/batch", post(batch_search))
        // Graph
        .route("/api/v1/collections/{collection}/graph/related/{id}", get(get_related))
        .route("/api/v1/collections/{collection}/graph/traverse", post(traverse))
        .route("/api/v1/collections/{collection}/graph/threshold", post(set_threshold))
        .route("/api/v1/collections/{collection}/hybrid_query", post(hybrid_query))
        // Typed graph CRUD + manual-edge lifecycle (Hybrid mode only)
        .route("/api/v1/collections/{collection}/graph/nodes", post(rest_put_node))
        .route("/api/v1/collections/{collection}/graph/nodes/{node_id}", get(rest_get_node))
        .route("/api/v1/collections/{collection}/graph/nodes/{node_id}", patch(rest_update_node))
        .route("/api/v1/collections/{collection}/graph/nodes/{node_id}", delete(rest_delete_node))
        .route("/api/v1/collections/{collection}/graph/nodes/{node_id}/edges", get(rest_list_edges))
        .route("/api/v1/collections/{collection}/graph/edges", post(rest_put_edge))
        // Static path declared before the {edge_id} param routes to avoid any
        // matchit static-vs-param ambiguity.
        .route("/api/v1/collections/{collection}/graph/bulk-import-edges", post(rest_bulk_import_edges))
        .route("/api/v1/collections/{collection}/graph/edges/{edge_id}", get(rest_get_edge))
        .route("/api/v1/collections/{collection}/graph/edges/{edge_id}", patch(rest_update_edge))
        .route("/api/v1/collections/{collection}/graph/edges/{edge_id}", delete(rest_delete_edge))
        .route("/api/v1/collections/{collection}/graph/edges/{edge_id}/verify", post(rest_verify_edge))
        .route("/api/v1/collections/{collection}/graph/edges/{edge_id}/reject", post(rest_reject_edge))
        // Extraction (Hybrid mode only)
        .route("/api/v1/collections/{collection}/llm-config", put(set_llm_config))
        .route("/api/v1/collections/{collection}/llm-config", get(get_llm_config))
        .route("/api/v1/collections/{collection}/llm-config/rotate", post(rotate_llm_config))
        .route("/api/v1/collections/{collection}/ontology", put(set_ontology))
        .route("/api/v1/collections/{collection}/ontology", get(get_ontology))
        .route("/api/v1/collections/{collection}/extraction/cost-preview", post(extraction_cost_preview))
        .route("/api/v1/collections/{collection}/extraction", post(start_extraction))
        .route("/api/v1/collections/{collection}/extraction/{job_id}", get(extraction_status))
        .route("/api/v1/collections/{collection}/extraction/{job_id}/cancel", post(cancel_extraction))
        .route("/api/v1/collections/{collection}/proposals", get(list_proposals))
        .route("/api/v1/collections/{collection}/proposals/{id}/approve", post(approve_proposal))
        .route("/api/v1/collections/{collection}/proposals/{id}/reject", post(reject_proposal))
        // Document-update diff and re-extraction (Hybrid mode only, P04)
        .route("/api/v1/collections/{collection}/extraction/diff", post(rest_diff_document))
        .route("/api/v1/collections/{collection}/extraction/reextract", post(rest_reextract_document))
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
        // Body-size cap protects every POST/PUT route from oversized payloads.
        .layer(request_size_limit_layer(request_body_limit_bytes()))
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

    let quantization_config = match &req.quantization {
        Some(qr) if qr.r#type == "scalar" => {
            Some(vf_core::types::QuantizationConfig::Scalar(
                vf_core::types::ScalarQuantizationConfig {
                    quantile: qr.quantile.unwrap_or(0.99),
                    always_ram: qr.always_ram.unwrap_or(true),
                },
            ))
        }
        Some(qr) => {
            return Err(err(
                StatusCode::BAD_REQUEST,
                format!("unsupported quantization type: {}", qr.r#type),
            ));
        }
        None => None,
    };

    let parsed_mode = parse_mode(req.mode.as_deref())?;

    let config = CollectionConfig {
        name: req.name.clone(),
        dimension,
        distance_metric,
        default_similarity_threshold: threshold,
        max_vectors: req.max_vectors as usize,
        data_type: DataTypeConfig::F32,
        quantization_config,
        mode: Some(parsed_mode),
    };

    // Check for duplicate in-memory BEFORE persisting to storage
    {
        let collections = state.collections.read();
        if collections.contains_key(&req.name) {
            return Err(err(StatusCode::CONFLICT, format!("collection '{}' already exists", req.name)));
        }
    }

    // Persist to storage layer first so collection_dir is available for set_data_dir.
    {
        let mut cm = state.collection_manager.write();
        if let Err(e) = cm.create_collection(config.clone()) {
            let msg = e.to_string();
            // Storage layer may report duplicate via existing config.json.
            let lower = msg.to_lowercase();
            if lower.contains("already exists") || lower.contains("exists") {
                return Err(err(StatusCode::CONFLICT, format!("collection '{}' already exists", req.name)));
            }
            return Err(err(StatusCode::INTERNAL_SERVER_ERROR, format!("storage create failed: {}", msg)));
        }
    }

    // Optional HNSW build parameters from the request; omitted fields fall
    // through to the server defaults via build_hnsw_params.
    let hnsw_params = build_hnsw_params(req.m, req.ef_construction);

    let store = InMemoryVectorStore::new(dimension);
    let index: Box<dyn vf_index::traits::PersistableIndex> = match &config.quantization_config {
        Some(vf_core::types::QuantizationConfig::Scalar(sq_config)) => {
            let q_index = vf_index::quantized_hnsw::QuantizedHnswIndex::new(
                dimension,
                distance_metric,
                hnsw_params.clone(),
                sq_config.clone(),
            );
            // Set data_dir unconditionally so post_optimize() can train the quantizer.
            let collection_dir = {
                let cm = state.collection_manager.read();
                cm.get_collection(&req.name)
                    .map(|c| c.collection_dir().to_path_buf())
                    .ok()
            };
            match collection_dir {
                Some(dir) => q_index.set_data_dir(dir),
                None => {
                    return Err(err(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("collection_dir missing after create for '{}'", req.name),
                    ));
                }
            }
            Box::new(q_index)
        }
        None => Box::new(vf_index::hnsw::HnswIndex::new(dimension, distance_metric, hnsw_params)),
    };

    // Attach a fresh hnsw.delta writer so first inserts are recorded for
    // incremental replay on the next boot.
    let collection_dir_for_delta = {
        let cm = state.collection_manager.read();
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
        Some(t) => vf_graph::VirtualGraph::with_threshold(t, distance_metric),
        None => vf_graph::VirtualGraph::with_threshold(0.7, distance_metric),
    };

    // Typed graph store for Hybrid collections (ADR-007 R4); None otherwise.
    let graph_store = {
        let dir = {
            let cm = state.collection_manager.read();
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
        status: std::sync::Arc::new(std::sync::RwLock::new(crate::state::CollectionStatus::Ready)),
        deferred_index: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        deferred_graph: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        deferred_metadata: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        dirty: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        mutation_count: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        collection_read_acquisitions: std::sync::atomic::AtomicU64::new(0),
        collection_write_acquisitions: std::sync::atomic::AtomicU64::new(0),
        total_blocked_microseconds: std::sync::atomic::AtomicU64::new(0),
    };

    // Map write lock just long enough to insert the per-collection RwLock handle.
    {
        let mut collections = state.collections.write();
        collections.insert(
            req.name.clone(),
            std::sync::Arc::new(parking_lot::RwLock::new(collection_state)),
        );
    }

    // Register the extraction runtime for Hybrid collections now that the
    // handle is published. A no-op for VectorOnly / AutoSimilarity.
    state.register_extraction_if_hybrid(&req.name);

    Ok(Json(CreateCollectionRes { name: req.name, success: true }))
}

async fn list_collections(
    State(state): State<AppState>,
) -> Json<ListCollectionsRes> {
    // Snapshot handles under a short map read lock so per-collection reads can
    // run independently after the map lock is released.
    let handles: Vec<std::sync::Arc<parking_lot::RwLock<CollectionState>>> = {
        let collections = state.collections.read();
        collections.values().cloned().collect()
    };
    let list = handles.iter().map(|h| {
        let c = h.read();
        let status_str = c.status.read().unwrap().as_str().to_string();
        let quantization_type = match &c.config.quantization_config {
            Some(vf_core::types::QuantizationConfig::Scalar(_)) => "scalar".to_string(),
            None => "none".to_string(),
        };
        CollectionInfo {
            name: c.config.name.clone(),
            dimension: c.config.dimension as u32,
            distance_metric: distance_metric_to_string(c.config.distance_metric),
            vector_count: c.store.len() as u64,
            indexed_count: c.index.len() as u64,
            default_threshold: c.config.default_similarity_threshold.unwrap_or(0.0),
            status: status_str,
            quantization_type,
        }
    }).collect();
    Json(ListCollectionsRes { collections: list })
}

async fn get_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&name).map_err(err_from_availability)?;
    let coll_handle = state.collection_handle(&name)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", name)))?;
    let c = metered_read(&coll_handle);
    let status_str = c.status.read().unwrap().as_str().to_string();
    let quantization_type = match &c.config.quantization_config {
        Some(vf_core::types::QuantizationConfig::Scalar(_)) => "scalar".to_string(),
        None => "none".to_string(),
    };
    Ok(Json(CollectionInfo {
        name: c.config.name.clone(),
        dimension: c.config.dimension as u32,
        distance_metric: distance_metric_to_string(c.config.distance_metric),
        vector_count: c.store.len() as u64,
        indexed_count: c.index.len() as u64,
        default_threshold: c.config.default_similarity_threshold.unwrap_or(0.0),
        status: status_str,
        quantization_type,
    }))
}

async fn delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<DeleteCollectionRes>, (StatusCode, Json<ErrorResponse>)> {
    tracing::info!(target: "vf_server::audit", collection = %name, "audit: collection delete requested via REST");
    state.require_collection_ready(&name).map_err(err_from_availability)?;
    // Cancel any in-flight extraction jobs for this collection and drop its
    // extraction runtime, so a dropped collection never leaves a runaway job
    // (ADR-016). Cooperative cancel: workers drain and skip its chunks.
    state.extraction.cancel_collection_jobs(&name);
    // Remove from storage layer
    {
        let mut cm = state.collection_manager.write();
        if let Err(e) = cm.drop_collection(&name) {
            tracing::warn!(collection = %name, "storage drop failed: {}", e);
        }
    }

    // Map write lock to evict the entry; any in-flight handler still holding
    // an Arc<RwLock<CollectionState>> completes naturally and the inner state
    // is dropped when the last Arc handle is released.
    let mut collections = state.collections.write();
    if collections.remove(&name).is_none() {
        return Err(err(StatusCode::NOT_FOUND, format!("collection '{}' not found", name)));
    }
    // Drop any recovery_paths entry so /recovery_status does not surface a
    // stale entry for a deleted collection.
    state.recovery_paths.write().remove(&name);
    Ok(Json(DeleteCollectionRes { success: true }))
}

// ── Vector handlers ─────────────────────────────────────────────────────

async fn insert_vector(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<InsertVectorReq>,
) -> Result<Json<InsertVectorRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    if req.values.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "vector values must not be empty"));
    }

    let core_metadata = req.metadata.as_ref().map(json_to_metadata)
        .transpose()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;

    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&coll_handle);

    if req.values.len() != coll.config.dimension {
        return Err(err(StatusCode::BAD_REQUEST, format!("vector dimension mismatch: expected {}, got {}", coll.config.dimension, req.values.len())));
    }

    let assigned_id = if req.id == 0 {
        coll.store.insert_metadata_auto_id(core_metadata.clone())
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("store insert failed: {}", e)))?
    } else {
        coll.store.insert_metadata(req.id, core_metadata.clone())
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("store insert failed: {}", e)))?;
        req.id
    };

    // Persist to storage layer first to obtain the WAL LSN.
    let lsn = {
        let mut cm = state.collection_manager.write();
        if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
            if let Err(e) = storage_coll.insert(assigned_id, VectorData::F32(req.values.clone()), core_metadata.clone()) {
                tracing::warn!(collection = %collection, id = assigned_id, "storage insert failed: {}", e);
            }
            storage_coll.current_lsn().saturating_sub(1)
        } else {
            0
        }
    };

    if let Err(e) = coll.index.add_with_lsn(assigned_id, &req.values, lsn) {
        let _ = coll.store.delete(assigned_id);
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, format!("index insert failed: {}", e)));
    }

    if let Some(ref meta) = core_metadata {
        coll.index_manager.index_record(assigned_id, meta);
    }

    // Only populate the similarity graph when the resolved mode enables it.
    if coll.config.graph_enabled() {
        // Reborrow the lock guard to a plain &mut CollectionState so the
        // compiler can split the disjoint field borrows of graph and index.
        let coll: &mut CollectionState = &mut *coll;
        if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
            &mut coll.graph, coll.index.as_vector_index(), assigned_id, &req.values, 10,
        ) {
            tracing::warn!(collection = %collection, id = assigned_id, "graph compute failed: {}", e);
        }
    }

    coll.dirty.store(true, Ordering::Release);
    coll.mutation_count.fetch_add(1, Ordering::Relaxed);

    Ok(Json(InsertVectorRes { id: assigned_id, success: true }))
}

async fn get_vector(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, u64)>,
) -> Result<Json<GetVectorRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    let meta_record = coll.store.get(id)
        .map_err(|e| err(StatusCode::NOT_FOUND, format!("vector not found: {}", e)))?;
    let vector_data = coll.index.get_vector(id)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("vector retrieval failed: {}", e)))?;

    let metadata_json = meta_record.metadata.as_ref().map(metadata_to_json);

    Ok(Json(GetVectorRes {
        id: meta_record.id,
        values: vector_data,
        metadata: metadata_json,
    }))
}

async fn update_vector(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, u64)>,
    ValidatedJson(req): ValidatedJson<UpdateVectorReq>,
) -> Result<Json<UpdateVectorRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
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

    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&coll_handle);

    if let Some(ref v) = req.values {
        if v.len() != coll.config.dimension {
            return Err(err(StatusCode::BAD_REQUEST, format!("vector dimension mismatch: expected {}, got {}", coll.config.dimension, v.len())));
        }
    }

    coll.store.update_metadata(id, core_metadata.clone())
        .map_err(|e| err(StatusCode::NOT_FOUND, format!("update failed: {}", e)))?;

    // Persist to storage layer first to obtain the WAL LSN.
    let lsn = {
        let mut cm = state.collection_manager.write();
        if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
            let storage_data = req.values.clone().map(VectorData::F32);
            if let Err(e) = storage_coll.update(id, storage_data, core_metadata.clone()) {
                tracing::warn!(collection = %collection, id = id, "storage update failed: {}", e);
            }
            storage_coll.current_lsn().saturating_sub(1)
        } else {
            0
        }
    };

    if let Some(ref values) = req.values {
        let _ = coll.index.remove_with_lsn(id, lsn);
        coll.index.add_with_lsn(id, values, lsn)
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("index update failed: {}", e)))?;
    }

    if let Some(ref meta) = core_metadata {
        coll.index_manager.remove_record(id);
        coll.index_manager.index_record(id, meta);
    }

    coll.dirty.store(true, Ordering::Release);
    coll.mutation_count.fetch_add(1, Ordering::Relaxed);

    Ok(Json(UpdateVectorRes { success: true }))
}

async fn delete_vector(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, u64)>,
) -> Result<Json<DeleteVectorRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&coll_handle);

    coll.store.delete(id)
        .map_err(|e| err(StatusCode::NOT_FOUND, format!("delete failed: {}", e)))?;

    // Persist to storage layer first to obtain the WAL LSN.
    let lsn = {
        let mut cm = state.collection_manager.write();
        if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
            if let Err(e) = storage_coll.delete(id) {
                tracing::warn!(collection = %collection, id = id, "storage delete failed: {}", e);
            }
            storage_coll.current_lsn().saturating_sub(1)
        } else {
            0
        }
    };

    let _ = coll.index.remove_with_lsn(id, lsn);
    coll.index_manager.remove_record(id);
    coll.graph.remove_node_with_lsn(id, lsn);

    coll.dirty.store(true, Ordering::Release);
    coll.mutation_count.fetch_add(1, Ordering::Relaxed);

    Ok(Json(DeleteVectorRes { success: true }))
}

async fn bulk_insert(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<BulkInsertReq>,
) -> Result<Json<BulkInsertRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let mut inserted_count: u64 = 0;
    // Pending errors carry an optional row id so the final reconciliation pass
    // can drop entries for rows that did land in the store.
    let mut pending_errors: Vec<(Option<u64>, String)> = Vec::new();
    // Track ids successfully committed by THIS call only; reconciliation uses
    // this set instead of a global store lookup to avoid false positives under
    // concurrent overlapping bulk_inserts.
    let mut committed_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    // Server-assigned ids in input order, parallel to committed_ids; pushed on
    // each Ok store-insert arm and popped if the corresponding chunk rolls back
    // so the final response only carries durably committed rows.
    let mut assigned_ids: Vec<u64> = Vec::new();

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
    // wal_flush_every is retained for API compatibility but no longer batches
    // storage writes. Each insert now writes the WAL inline so the resulting
    // LSN can be threaded into the HNSW delta writer (add_with_lsn).
    let _wal_flush_every = raw_wal_flush_every as usize;
    let _ef_construction = req.ef_construction; // TODO: HNSW parameter override
    let deferred_index_mode = index_mode == "deferred";
    let skip_metadata_index = req.skip_metadata_index.unwrap_or(false);
    let _parallel_build = parallel_build; // stored for optimize()
    let checkpoint_every = req.checkpoint_every.unwrap_or(0);
    let resume_token = req.resume_token.clone().unwrap_or_default();

    // Verify collection exists before iterating. Map read lock dropped at the
    // end of the block; per-chunk lookups below use the per-collection handle.
    {
        let collections = state.collections.read();
        if !collections.contains_key(&collection) {
            return Err(err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)));
        }
    }

    // Resolve checkpoint path under the per-collection data dir.
    let checkpoint_path = resolve_bulk_checkpoint_path(&state, &collection);

    // Stable u64 id for the collection. Hashed from the collection name with
    // DefaultHasher; collisions are tolerated because the id is only used to
    // detect on-disk checkpoint drift, not for cross-collection identity.
    let collection_id_hash = hash_collection_name(&collection);

    // Resume handling: if the client supplied a token, validate against the
    // on-disk checkpoint and compute the start chunk index. Otherwise start
    // from the first chunk.
    let mut start_chunk_idx: usize = 0;
    let mut last_completed_batch_idx: u64 = 0;
    let mut last_committed_lsn: u64 = 0;
    let mut any_chunk_completed: bool = false;
    if !resume_token.is_empty() {
        let cp_path = match &checkpoint_path {
            Some(p) => p.clone(),
            None => {
                return Err(err(
                    StatusCode::BAD_REQUEST,
                    "resume_token provided but server cannot resolve a data directory for this collection"
                        .to_string(),
                ));
            }
        };
        let cp = match vf_storage::bulk_checkpoint::BulkCheckpoint::read(&cp_path) {
            Ok(cp) => cp,
            Err(e) => {
                return Err(err(
                    StatusCode::BAD_REQUEST,
                    format!("resume_token provided but no checkpoint on disk: {}", e),
                ));
            }
        };
        let expected = encode_resume_token(&collection, cp.last_committed_lsn);
        if expected != resume_token {
            return Err(err(
                StatusCode::BAD_REQUEST,
                "resume_token does not match on-disk checkpoint (client view diverged from server state)"
                    .to_string(),
            ));
        }
        start_chunk_idx = (cp.last_completed_batch_idx as usize).saturating_add(1);
        last_completed_batch_idx = cp.last_completed_batch_idx;
        last_committed_lsn = cp.last_committed_lsn;
        any_chunk_completed = true;
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
            let tag = if v.id != 0 { Some(v.id) } else { None };
            pending_errors.push((tag, format!("item {}: missing or empty vector", i)));
            continue;
        }

        let core_metadata = match v.metadata.as_ref().map(json_to_metadata).transpose() {
            Ok(m) => m,
            Err(e) => {
                let tag = if v.id != 0 { Some(v.id) } else { None };
                pending_errors.push((tag, format!("item {}: {}", i, e)));
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

    // Process vectors in batches. Per chunk we write WAL first per item to
    // obtain the LSN, then issue a single parallel bulk_add_with_lsn for the
    // whole chunk so the hnsw.delta writer records inserts in input order and
    // restart replay stays incremental.
    // Owned chunks so per-row vector bytes can be moved (not cloned) into the
    // Arc that the index path and the graph compute step both share.
    let chunk_size = batch_lock_size.max(1);
    let mut chunks: Vec<Vec<PreparedVector>> = Vec::new();
    while !prepared.is_empty() {
        let take = chunk_size.min(prepared.len());
        chunks.push(prepared.drain(..take).collect());
    }
    let total_chunks = chunks.len();
    let mut last_resume_token: String = String::new();

    for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
        // Skip chunks already covered by the prior checkpoint.
        if chunk_idx < start_chunk_idx {
            continue;
        }

        // Per-collection write lock for the chunk; searches on OTHER collections
        // continue to run because the map RwLock is no longer held in write mode.
        let coll_handle = match state.collection_handle(&collection) {
            Some(h) => h,
            None => {
                for pv in chunk {
                    let tag = if pv.id != 0 { Some(pv.id) } else { None };
                    pending_errors.push((tag, format!("item {}: collection '{}' not found", pv.index, collection)));
                }
                continue;
            }
        };
        // Phase 1: acquire the per-collection write guard and run the per-item
        // validate + store + WAL inserts inside its own lexical scope so the
        // non-Send guard is dropped at the block's closing brace, before any
        // await. The block returns the owned values the post-build phase needs.
        //
        // Each vector is wrapped once in Arc; chunk_items and the graph compute
        // map share Arc clones so the bulk path never duplicates the vector bytes.
        let (mut chunk_items, chunk_metas, chunk_max_lsn): (
            Vec<(VectorId, Arc<Vec<f32>>, u64)>,
            Vec<Option<Metadata>>,
            u64,
        ) = {
            let coll = metered_write(&coll_handle);

            let mut chunk_items: Vec<(VectorId, Arc<Vec<f32>>, u64)> = Vec::with_capacity(chunk.len());
            let mut chunk_metas: Vec<Option<Metadata>> = Vec::with_capacity(chunk.len());
            let mut chunk_max_lsn: u64 = last_committed_lsn;

            for pv in chunk {
                if pv.values.len() != coll.config.dimension {
                    let tag = if pv.id != 0 { Some(pv.id) } else { None };
                    pending_errors.push((tag, format!(
                        "item {}: vector dimension mismatch: expected {}, got {}",
                        pv.index, coll.config.dimension, pv.values.len()
                    )));
                    continue;
                }

                let assigned_id = if pv.id == 0 {
                    match coll.store.insert_metadata_auto_id(pv.metadata.clone()) {
                        Ok(id) => {
                            committed_ids.insert(id);
                            assigned_ids.push(id);
                            id
                        }
                        Err(e) => {
                            pending_errors.push((None, format!("item {}: store insert failed: {}", pv.index, e)));
                            continue;
                        }
                    }
                } else {
                    match coll.store.insert_metadata(pv.id, pv.metadata.clone()) {
                        Ok(()) => {
                            committed_ids.insert(pv.id);
                            assigned_ids.push(pv.id);
                            pv.id
                        }
                        Err(e) => {
                            pending_errors.push((Some(pv.id), format!("item {}: store insert failed for id {}: {}", pv.index, pv.id, e)));
                            continue;
                        }
                    }
                };

                // Persist to storage layer first so we can capture the WAL LSN.
                let lsn = {
                    let mut cm = state.collection_manager.write();
                    if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
                        if let Err(e) = storage_coll.insert(
                            assigned_id,
                            VectorData::F32(pv.values.clone()),
                            pv.metadata.clone(),
                        ) {
                            tracing::warn!(collection = %collection, id = assigned_id, "storage bulk insert failed: {}", e);
                        }
                        storage_coll.current_lsn().saturating_sub(1)
                    } else {
                        0
                    }
                };

                if lsn > chunk_max_lsn {
                    chunk_max_lsn = lsn;
                }

                // Single Arc per row; the index path and the graph compute share it.
                let vec_arc = Arc::new(pv.values);
                chunk_items.push((assigned_id, vec_arc, lsn));
                chunk_metas.push(pv.metadata);
            }

            (chunk_items, chunk_metas, chunk_max_lsn)
        };
        // <-- per-collection write guard dropped here, before the await.

        // Single parallel bulk add for the chunk. Whole-batch-fails on any
        // late dim or id collision; emit a per-row error so the response
        // contract errors_count + inserted_count == rows_seen holds.
        if !deferred_index_mode && !chunk_items.is_empty() {
            // Capture ids for D-1 rollback and run the CPU-bound build in
            // spawn_blocking (which acquires its own fresh guard). No guard is
            // held across the await; the post-build phase acquires a separate
            // fresh guard binding.
            let chunk_ids_for_rollback: Vec<u64> =
                chunk_items.iter().map(|(id, _, _)| *id).collect();

            let build_handle = coll_handle.clone();
            let items = std::mem::take(&mut chunk_items);
            let build_res = tokio::task::spawn_blocking(move || {
                let coll = metered_write(&build_handle);
                match coll.index.bulk_add_with_lsn(&items) {
                    Ok(()) => Ok(items),
                    Err(e) => Err(e),
                }
            })
            .await;

            let build_err = match build_res {
                Ok(Ok(returned_items)) => {
                    chunk_items = returned_items;
                    None
                }
                Ok(Err(e)) => Some(format!("{}", e)),
                Err(join_err) => Some(format!("bulk index build task join error: {join_err}")),
            };
            if let Some(e) = build_err {
                // D-1 rollback under a fresh post-await guard binding.
                let coll_post = metered_write(&coll_handle);
                let rolled_back: std::collections::HashSet<u64> =
                    chunk_ids_for_rollback.iter().copied().collect();
                // Rollback metadata-store inserts for this chunk.
                for id in &chunk_ids_for_rollback {
                    let _ = coll_post.store.delete(*id);
                    // Drop rolled-back ids so reconciliation does not
                    // reclassify them as successful inserts.
                    committed_ids.remove(id);
                }
                // Symmetric pop from assigned_ids so the response excludes
                // rows that did not make it past the index step.
                assigned_ids.retain(|id| !rolled_back.contains(id));
                let err_str = format!(
                    "chunk {}: bulk index insert failed: {}",
                    chunk_idx, e
                );
                for id in &chunk_ids_for_rollback {
                    pending_errors.push((Some(*id), err_str.clone()));
                }
                drop(coll_post);
                drop(chunk_metas);
                continue;
            }
        }

        // Post-build phase: the metadata + graph work runs inside an inner block
        // so the !Send write guard's region ends before the await below. An
        // explicit drop() is not enough to shrink the async state machine's
        // captured region; only a lexical block reliably scopes the guard out so
        // the handler future stays Send (axum Handler requires Send).
        {
            let mut coll = metered_write(&coll_handle);

            // Single pass: index metadata, count, and seed the graph compute map.
            let mut chunk_ids: Vec<u64> = Vec::with_capacity(chunk_items.len());
            let mut chunk_vectors_map: HashMap<u64, Arc<Vec<f32>>> =
                HashMap::with_capacity(chunk_items.len());
            for ((assigned_id, vec_arc, _), meta_opt) in chunk_items.iter().zip(chunk_metas.into_iter()) {
                if !skip_metadata_index {
                    if let Some(ref m) = meta_opt {
                        coll.index_manager.index_record(*assigned_id, m);
                    }
                }
                inserted_count += 1;
                chunk_ids.push(*assigned_id);
                chunk_vectors_map.insert(*assigned_id, Arc::clone(vec_arc));
            }
            drop(chunk_items);

            // Graph recompute for just this chunk while we still hold the write lock.
            // Skipped when the resolved mode disables the graph (vector-only).
            if !defer_graph && coll.config.graph_enabled() && !chunk_ids.is_empty() {
                // Reborrow the lock guard to a plain &mut CollectionState so the
                // compiler can split the disjoint field borrows of graph and index.
                let coll: &mut CollectionState = &mut *coll;
                if let Err(e) = vf_graph::RelationshipComputer::compute_batch_parallel(
                    &mut coll.graph,
                    coll.index.as_vector_index(),
                    &chunk_ids,
                    &chunk_vectors_map,
                    10,
                ) {
                    tracing::warn!(collection = %collection, "graph compute_batch_parallel after bulk_insert failed: {}", e);
                }
            }
            drop(chunk_vectors_map);
            drop(chunk_ids);
            // The per-collection write lock is released at the end of this block,
            // before any checkpoint IO or await, so the checkpoint write does not
            // block concurrent searches.
        }

        // Mark the chunk as the most recently completed.
        last_completed_batch_idx = chunk_idx as u64;
        last_committed_lsn = chunk_max_lsn;
        any_chunk_completed = true;

        // Persist a checkpoint every N completed chunks.
        if checkpoint_every > 0 && (chunk_idx as u64 + 1) % (checkpoint_every as u64) == 0 {
            if let Some(ref cp_path) = checkpoint_path {
                let cp = vf_storage::bulk_checkpoint::BulkCheckpoint::new(
                    collection_id_hash,
                    last_completed_batch_idx,
                    last_committed_lsn,
                );
                if let Err(e) = cp.write_atomic(cp_path) {
                    tracing::warn!(
                        collection = %collection,
                        chunk_idx,
                        "bulk_insert checkpoint write failed: {}",
                        e
                    );
                } else {
                    last_resume_token = encode_resume_token(&collection, last_committed_lsn);
                }
            }
        }

        // Yield between chunks so the runtime can schedule probes and reads.
        tokio::task::yield_now().await;
    }

    // Return capacity ratchet from the metadata-index manager to the allocator
    // now that the bulk write burst is done. Per-collection write lock; map lock
    // stays free.
    if inserted_count > 0 && !skip_metadata_index {
        if let Some(handle) = state.collection_handle(&collection) {
            let mut coll = metered_write(&handle);
            coll.index_manager.compact();
        }
    }

    // Set deferred flags and update collection status if any optimization was deferred.
    // Per-collection read lock is enough; atomics and the inner status RwLock handle their own sync.
    if inserted_count > 0 && (deferred_index_mode || defer_graph || skip_metadata_index) {
        if let Some(handle) = state.collection_handle(&collection) {
            let coll = metered_read(&handle);
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

    // Successful end-to-end completion: scrub the checkpoint so a future call
    // does not see a stale resume state. Anything other than full completion
    // (errors with chunks remaining unprocessed) leaves the file in place.
    let fully_complete = any_chunk_completed
        && total_chunks > 0
        && last_completed_batch_idx as usize + 1 == total_chunks;
    if fully_complete {
        if let Some(ref cp_path) = checkpoint_path {
            let _ = vf_storage::bulk_checkpoint::BulkCheckpoint::delete(cp_path);
        }
        last_resume_token = String::new();
    }

    // Reconcile pending errors against the live store. A row that landed in the
    // collection must not appear in the user-facing errors list; reclaimed
    // entries are counted as inserted so errors_count + inserted_count stays
    // equal to rows_seen.
    let mut errors: Vec<String> = Vec::with_capacity(pending_errors.len());
    {
        // Reconcile against this call's own committed set so a concurrent
        // overlapping bulk_insert by another client cannot cause us to
        // silently claim credit for rows we did not commit.
        for (id_tag, msg) in pending_errors {
            let committed = match id_tag {
                Some(id) => committed_ids.contains(&id),
                None => false,
            };
            if committed {
                inserted_count += 1;
            } else {
                errors.push(msg);
            }
        }
    }

    // Snapshot HNSW base + graph base at end of a successful bulk_insert so
    // the next restart picks IncrementalReplay (seconds) over FullRebuild
    // (minutes). Mirrors the gRPC handler.
    if inserted_count > 0 {
        if let Err(e) = crate::snapshot::force_snapshot_collection(&state, &collection) {
            tracing::warn!(
                collection = %collection,
                "post-bulk_insert snapshot failed: {}",
                e
            );
        }
    }

    Ok(Json(BulkInsertRes {
        inserted_count,
        errors,
        last_completed_batch_idx,
        last_committed_lsn,
        resume_token: last_resume_token,
        assigned_ids,
    }))
}

// ── Bulk insert from path handler ───────────────────────────────────────

async fn bulk_insert_from_path(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<BulkInsertFromPathReq>,
) -> Result<Json<BulkInsertRes>, (StatusCode, Json<ErrorResponse>)> {
    use crate::bulk_insert_from_path::{self as bifp, DatasetView, IdsSource};
    use std::os::fd::AsRawFd;

    state
        .require_collection_ready(&collection)
        .map_err(err_from_availability)?;

    if req.path.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "path must not be empty"));
    }

    let index_mode = if req.index_mode.is_empty() {
        "immediate"
    } else {
        req.index_mode.as_str()
    };
    validate_index_mode(index_mode)
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
    let index_mode_deferred = index_mode == "deferred";
    let skip_metadata_index = req.skip_metadata_index;

    if req.ef_construction > 0 {
        validate_ef_construction(req.ef_construction, state.max_ef_construction)
            .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
    }

    let allowed_roots = &state.config.bulk_insert_allowed_roots;
    if allowed_roots.is_empty() {
        return Err(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "bulk_insert_allowed_roots is empty; server misconfigured",
        ));
    }

    let mut root_files: Vec<std::fs::File> = Vec::with_capacity(allowed_roots.len());
    for root in allowed_roots {
        match std::fs::File::open(root) {
            Ok(f) => root_files.push(f),
            Err(e) => {
                return Err(err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to open allow-list root {}: {}", root.display(), e),
                ));
            }
        }
    }
    let root_fds: Vec<std::os::fd::RawFd> =
        root_files.iter().map(|f| f.as_raw_fd()).collect();

    let (data_file, mmap, view): (std::fs::File, memmap2::Mmap, DatasetView) =
        match bifp::open_validated(&req.path, &root_fds, req.dim as usize, req.expected_count) {
            Ok(t) => t,
            Err(e) => return Err(map_bifp_error_rest(e)),
        };
    let _data_file = data_file;
    // F14: own the mmap behind an Arc so the CPU-bound HNSW build can be
    // offloaded to spawn_blocking. Each build closure takes an Arc clone
    // (refcount bump only; the mapped pages stay shared, zero copy) and
    // reconstructs its row slices inside the blocking thread, so no mmap borrow
    // is ever held across an .await.
    let mmap = std::sync::Arc::new(mmap);

    let coll_handle = state.collection_handle(&collection).ok_or_else(|| {
        err(
            StatusCode::NOT_FOUND,
            format!("collection '{}' not found", collection),
        )
    })?;

    {
        let coll = metered_read(&coll_handle);
        if coll.config.dimension != view.dim {
            return Err(err(
                StatusCode::BAD_REQUEST,
                format!(
                    "vector dimension mismatch: collection expects {}, file has {}",
                    coll.config.dimension, view.dim
                ),
            ));
        }
    }

    let row_bytes: &[u8] = &mmap[view.header_offset..];
    if row_bytes.len() < view.count.saturating_mul(view.dim).saturating_mul(4) {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "file is shorter than count * dim * 4 bytes",
        ));
    }
    let flat: &[f32] = bytemuck::cast_slice(row_bytes);
    let rows: Vec<&[f32]> = flat
        .chunks_exact(view.dim)
        .take(view.count)
        .collect();

    let ids_holder: IdsSource;
    let ids: Vec<u64> = if req.ids_path.is_empty() {
        let start = if req.id_start == 0 { 1 } else { req.id_start };
        ids_holder = IdsSource::Sequential;
        (0..view.count as u64).map(|i| start + i).collect()
    } else {
        match bifp::open_ids_validated(&req.ids_path, &root_fds, view.count) {
            Ok((file, mmap_ids, ids_vec)) => {
                ids_holder = IdsSource::Mmap { _file: file, _mmap: mmap_ids };
                ids_vec
            }
            Err(e) => return Err(map_bifp_error_rest(e)),
        }
    };
    let _ids_holder = ids_holder;

    if ids.len() != rows.len() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            format!(
                "ids count {} does not match vectors count {}",
                ids.len(),
                rows.len()
            ),
        ));
    }

    let mut committed_ids: std::collections::HashSet<u64> =
        std::collections::HashSet::with_capacity(rows.len());
    let mut assigned_ids: Vec<u64> = Vec::with_capacity(rows.len());
    let mut pending_errors: Vec<(Option<u64>, String)> = Vec::new();
    let mut inserted_count: u64 = 0;
    let mut last_committed_lsn: u64 = 0;

    // `row_orig_idx` records each survivor's original dense row index so the
    // F14 offload can reconstruct its mmap slice inside spawn_blocking without
    // holding a borrow across the await.
    let mut row_ids: Vec<u64> = Vec::with_capacity(rows.len());
    let mut row_vecs: Vec<&[f32]> = Vec::with_capacity(rows.len());
    let mut row_orig_idx: Vec<usize> = Vec::with_capacity(rows.len());
    // Per-collection write lock for the metadata-store inserts. The guard lives
    // only inside this block so the non-Send parking_lot guard is dropped at the
    // closing brace, well before any spawn_blocking().await below.
    {
        let coll = metered_write(&coll_handle);
        for (i, vec_slice) in rows.iter().enumerate() {
            let id_in = ids[i];
            match coll.store.insert_metadata(id_in, None) {
                Ok(()) => {
                    committed_ids.insert(id_in);
                    assigned_ids.push(id_in);
                    row_ids.push(id_in);
                    row_vecs.push(*vec_slice);
                    row_orig_idx.push(i);
                }
                Err(e) => {
                    pending_errors.push((
                        Some(id_in),
                        format!("row {}: store insert failed for id {}: {}", i, id_in, e),
                    ));
                }
            }
        }
    }

    let mut lsns: Vec<u64> = Vec::with_capacity(row_ids.len());
    {
        let mut cm = state.collection_manager.write();
        if let Ok(storage_coll) = cm.get_collection_mut(&collection) {
            for (idx, id_in) in row_ids.iter().enumerate() {
                if let Err(e) = storage_coll.insert(
                    *id_in,
                    VectorData::F32(row_vecs[idx].to_vec()),
                    None,
                ) {
                    tracing::warn!(
                        collection = %collection,
                        id = *id_in,
                        "storage bulk_insert_from_path insert failed: {}",
                        e
                    );
                }
                let lsn = storage_coll.current_lsn().saturating_sub(1);
                if lsn > last_committed_lsn {
                    last_committed_lsn = lsn;
                }
                lsns.push(lsn);
            }
        } else {
            for _ in &row_ids {
                lsns.push(0);
            }
        }
    }

    let items: Vec<(VectorId, &[f32], u64)> = row_ids
        .iter()
        .zip(row_vecs.iter())
        .zip(lsns.iter())
        .map(|((id, v), lsn)| (*id, *v, *lsn))
        .collect();

    let total_hint = if req.total_count_hint == 0 {
        items.len()
    } else {
        req.total_count_hint as usize
    };

    // F14: capture the row layout as owned, mmap-independent data so the
    // CPU-bound HNSW build can move into spawn_blocking. `items` only borrows
    // the mmap; the offloaded build instead carries (id, orig_idx, lsn) triples
    // plus an Arc<Mmap> clone and rebuilds the &[f32] slices inside the blocking
    // thread. Dimension and header offset are plain usize copies. We keep
    // bulk_add_from_slice_iter (NOT bulk_add_with_lsn) so the chunked path does
    // not re-snapshot all prior nodes per chunk.
    let build_rows: Vec<(VectorId, usize, u64)> = row_ids
        .iter()
        .zip(row_orig_idx.iter())
        .zip(lsns.iter())
        .map(|((id, orig), lsn)| (*id, *orig, *lsn))
        .collect();
    // Stable count for the post-build deferred / metadata / F3 bookkeeping;
    // build_rows itself is moved into the (non-chunked) build closure.
    let item_count = build_rows.len();
    let view_dim = view.dim;
    let view_header_offset = view.header_offset;
    // Drop the mmap-borrowing bindings before the build so only the Arc keeps
    // the mapping alive; the closures reconstruct slices on demand.
    drop(items);
    drop(row_vecs);
    drop(rows);

    if !index_mode_deferred && !build_rows.is_empty() {
        if req.chunk_size == 0 {
            // Non-chunked path: offload the full single bulk_add to
            // spawn_blocking. The metadata-insert guard was already dropped at
            // its block above, so no parking_lot guard crosses the await; the
            // closure takes a fresh guard on the same handle and rebuilds the
            // borrowed slices from the Arc<Mmap> clone.
            let build_handle = coll_handle.clone();
            let mmap_for_build = mmap.clone();
            let rows_for_build = build_rows;
            let build_res = tokio::task::spawn_blocking(move || {
                let flat: &[f32] =
                    bytemuck::cast_slice(&mmap_for_build[view_header_offset..]);
                let items: Vec<(VectorId, &[f32], u64)> = rows_for_build
                    .iter()
                    .map(|(id, orig, lsn)| {
                        let start = orig * view_dim;
                        (*id, &flat[start..start + view_dim], *lsn)
                    })
                    .collect();
                let coll = metered_write(&build_handle);
                match coll.index.bulk_add_from_slice_iter(&items, total_hint) {
                    Ok(()) => Ok(rows_for_build),
                    Err(e) => Err((format!("{}", e), rows_for_build)),
                }
            })
            .await;

            match build_res {
                Ok(Ok(built_rows)) => {
                    inserted_count = built_rows.len() as u64;
                }
                Ok(Err((e, failed_rows))) => {
                    // Rollback under a fresh post-await guard binding.
                    let coll_post = metered_write(&coll_handle);
                    let rolled_back: std::collections::HashSet<u64> =
                        failed_rows.iter().map(|(id, _, _)| *id).collect();
                    for (id, _, _) in &failed_rows {
                        let _ = coll_post.store.delete(*id);
                        committed_ids.remove(id);
                    }
                    drop(coll_post);
                    assigned_ids.retain(|id| !rolled_back.contains(id));
                    let err_str = format!("bulk_insert_from_path: index insert failed: {}", e);
                    for (id, _, _) in &failed_rows {
                        pending_errors.push((Some(*id), err_str.clone()));
                    }
                }
                Err(join_err) => {
                    let err_str = format!(
                        "bulk_insert_from_path: index build task join error: {join_err}"
                    );
                    let coll_post = metered_write(&coll_handle);
                    for id in &assigned_ids {
                        let _ = coll_post.store.delete(*id);
                        committed_ids.remove(id);
                    }
                    drop(coll_post);
                    for id in &assigned_ids {
                        pending_errors.push((Some(*id), err_str.clone()));
                    }
                    assigned_ids.clear();
                }
            }
        } else {
            // Chunked path: no parking_lot guard is live here (the
            // metadata-insert guard was dropped at its block above); for each
            // chunk, offload the bulk_add to spawn_blocking (fresh guard inside,
            // slices rebuilt from an Arc<Mmap> clone), then snapshot, prune WAL,
            // and release memory back to the OS on the async side.
            let chunk_sz = req.chunk_size as usize;
            for chunk in build_rows.chunks(chunk_sz) {
                let build_handle = coll_handle.clone();
                let mmap_for_build = mmap.clone();
                let chunk_rows: Vec<(VectorId, usize, u64)> = chunk.to_vec();
                let build_res = tokio::task::spawn_blocking(move || {
                    let flat: &[f32] =
                        bytemuck::cast_slice(&mmap_for_build[view_header_offset..]);
                    let items: Vec<(VectorId, &[f32], u64)> = chunk_rows
                        .iter()
                        .map(|(id, orig, lsn)| {
                            let start = orig * view_dim;
                            (*id, &flat[start..start + view_dim], *lsn)
                        })
                        .collect();
                    let coll_inner = metered_write(&build_handle);
                    match coll_inner.index.bulk_add_from_slice_iter(&items, total_hint) {
                        Ok(()) => Ok(chunk_rows),
                        Err(e) => Err((format!("{}", e), chunk_rows)),
                    }
                })
                .await;

                match build_res {
                    Ok(Ok(built_rows)) => {
                        inserted_count += built_rows.len() as u64;
                    }
                    Ok(Err((e, failed_rows))) => {
                        let coll_inner = metered_write(&coll_handle);
                        let rolled_back: std::collections::HashSet<u64> =
                            failed_rows.iter().map(|(id, _, _)| *id).collect();
                        for (id, _, _) in &failed_rows {
                            let _ = coll_inner.store.delete(*id);
                            committed_ids.remove(id);
                        }
                        drop(coll_inner);
                        assigned_ids.retain(|id| !rolled_back.contains(id));
                        let err_str =
                            format!("bulk_insert_from_path: index insert failed: {}", e);
                        for (id, _, _) in &failed_rows {
                            pending_errors.push((Some(*id), err_str.clone()));
                        }
                        break;
                    }
                    Err(join_err) => {
                        let err_str = format!(
                            "bulk_insert_from_path: index build task join error: {join_err}"
                        );
                        let coll_inner = metered_write(&coll_handle);
                        let rolled_back: std::collections::HashSet<u64> =
                            chunk.iter().map(|(id, _, _)| *id).collect();
                        for (id, _, _) in chunk {
                            let _ = coll_inner.store.delete(*id);
                            committed_ids.remove(id);
                        }
                        drop(coll_inner);
                        assigned_ids.retain(|id| !rolled_back.contains(id));
                        for (id, _, _) in chunk {
                            pending_errors.push((Some(*id), err_str.clone()));
                        }
                        break;
                    }
                }

                if let Err(e) = crate::snapshot::force_snapshot_collection(&state, &collection)
                {
                    tracing::warn!(
                        collection = %collection,
                        "chunked bulk_insert_from_path: snapshot failed: {}",
                        e
                    );
                }
                if let Err(e) = state.prune_wal_for_collection(&collection) {
                    tracing::warn!(
                        collection = %collection,
                        "chunked bulk_insert_from_path: prune_wal failed: {}",
                        e
                    );
                }
                vf_index::release_to_os();
            }
        }
    } else if index_mode_deferred {
        inserted_count = item_count as u64;
    }

    // I12 (ADR-025): the file mapping is no longer needed once the build is done.
    // Drop our Arc and purge the allocator arenas so the build-time heap freed
    // inside bulk_add_from_slice_iter returns to the OS now.
    drop(mmap);
    vf_index::purge_allocator_arenas();

    // Post-build bookkeeping under one fresh write guard, acquired AFTER all
    // spawn_blocking().await points so no parking_lot guard ever crosses an
    // await. Mirrors the post-build fresh-guard binding in
    // bulk_insert_with_options. `mut` is required because
    // IndexManager::compact takes &mut self.
    let mut coll = metered_write(&coll_handle);

    if index_mode_deferred && item_count > 0 {
        coll.deferred_index.store(true, Ordering::Release);
    }

    if !skip_metadata_index {
        coll.index_manager.compact();
    } else if item_count > 0 {
        coll.deferred_metadata.store(true, Ordering::Release);
    }

    // F3: when anything was deferred, flip the collection to
    // PendingOptimization so searches surface the stale-results warning and do
    // not silently return 0 over an unbuilt index. Mirrors the REST bulk_insert
    // deferred bookkeeping (~rest.rs:1547-1562).
    let any_deferred = (index_mode_deferred && item_count > 0)
        || (skip_metadata_index && item_count > 0);
    if any_deferred {
        if let Ok(mut status) = coll.status.write() {
            *status = CollectionStatus::PendingOptimization;
        }
    }

    drop(coll);

    let mut errors: Vec<String> = Vec::with_capacity(pending_errors.len());
    for (id_tag, msg) in pending_errors {
        let committed = match id_tag {
            Some(id) => committed_ids.contains(&id),
            None => false,
        };
        if committed {
            inserted_count += 1;
        } else {
            errors.push(msg);
        }
    }

    // Always snapshot the HNSW base + graph base at the end of a successful
    // bulk_insert_from_path call (single-pass branch). The chunked path
    // already snapshots per chunk. Without this, the single-pass path
    // leaves a fat hnsw.delta and no hnsw.base, forcing FullRebuild on
    // any restart. Snapshot failures are logged but not surfaced because
    // the WAL holds the durable record; the next snapshot covers it.
    if inserted_count > 0 {
        if let Err(e) = crate::snapshot::force_snapshot_collection(&state, &collection) {
            tracing::warn!(
                collection = %collection,
                "post-bulk_insert_from_path snapshot failed: {}",
                e
            );
        }
    }

    Ok(Json(BulkInsertRes {
        inserted_count,
        errors,
        last_completed_batch_idx: 0,
        last_committed_lsn,
        resume_token: String::new(),
        assigned_ids,
    }))
}

fn map_bifp_error_rest(
    e: crate::bulk_insert_from_path::BulkFromPathError,
) -> (StatusCode, Json<ErrorResponse>) {
    use crate::bulk_insert_from_path::BulkFromPathError as E;
    let msg = e.to_string();
    match e {
        E::PathDenied { .. } => err(StatusCode::FORBIDDEN, msg),
        E::RelativePath { .. }
        | E::TraversalAttempt { .. }
        | E::NullByte { .. }
        | E::BadMagic { .. }
        | E::DimensionMismatch { .. }
        | E::CountMismatch { .. } => err(StatusCode::BAD_REQUEST, msg),
        E::MmapFailed { .. } | E::Io { .. } => err(StatusCode::INTERNAL_SERVER_ERROR, msg),
    }
}

// ── Optimize handler ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct OptimizeReq {
    #[serde(default)]
    rebuild_graph: bool,
}

fn default_true() -> bool { true }

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
    body: Option<Json<OptimizeReq>>,
) -> Result<Json<OptimizeRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let rebuild_graph = body.map_or(false, |b| b.rebuild_graph);
    match state.optimize_collection(&collection, rebuild_graph).await {
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

// ── WAL Prune handler ──────────────────────────────────────────────────

async fn prune_wal_collection(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let start = std::time::Instant::now();

    match state.prune_wal_for_collection(&collection) {
        Ok((files_deleted, bytes_freed)) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            Ok(Json(serde_json::json!({
                "status": "completed",
                "files_deleted": files_deleted,
                "bytes_freed": bytes_freed,
                "duration_ms": duration_ms
            })))
        }
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

// ── Compact handler ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CompactReq {
    #[serde(default = "default_zero_u32")]
    min_segments: u32,
    #[serde(default = "default_true")]
    remove_deleted: bool,
}

fn default_zero_u32() -> u32 { 0 }

async fn compact_collection(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    body: Option<Json<CompactReq>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let start = std::time::Instant::now();
    let req = body.map(|b| b.0).unwrap_or(CompactReq { min_segments: 0, remove_deleted: true });

    let min_segments = if req.min_segments == 0 { 4 } else { req.min_segments as usize };

    // F7: the on-disk segment merge inside compact_collection holds the global
    // collection_manager write lock for its whole duration. Running it inline
    // would pin a runtime worker (and stall /readyz) for the full merge. Offload
    // to spawn_blocking on a cheap AppState clone (Arc bumps); the global write
    // lock is acquired inside the closure and released the moment the merge
    // returns. No parking_lot guard crosses the await.
    let remove_deleted = req.remove_deleted;
    let compact_state = state.clone();
    let compact_collection_name = collection.clone();
    let result = tokio::task::spawn_blocking(move || {
        compact_state.compact_collection(&compact_collection_name, min_segments, remove_deleted)
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("compact task join error: {e}")))?;

    match result {
        Ok(result) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            Ok(Json(serde_json::json!({
                "status": "completed",
                "segments_merged": result.segments_merged,
                "vectors_written": result.vectors_written,
                "vectors_removed": result.vectors_removed,
                "duration_ms": duration_ms
            })))
        }
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

// ── Recovery / persistence / snapshot / metrics handlers ────────────────

#[derive(Serialize)]
struct RecoveryStatusEntry {
    name: String,
    path: String,
}

#[derive(Serialize)]
struct RecoveryStatusRes {
    path: String,
    elapsed_secs: u64,
    collections: Vec<RecoveryStatusEntry>,
}

async fn recovery_status(
    State(state): State<AppState>,
) -> Json<RecoveryStatusRes> {
    let snap = state.recovery_status_snapshot();
    let mut collections: Vec<RecoveryStatusEntry> = snap
        .paths
        .iter()
        .map(|(name, p)| RecoveryStatusEntry {
            name: name.clone(),
            path: p.as_str().to_string(),
        })
        .collect();
    collections.sort_by(|a, b| a.name.cmp(&b.name));
    Json(RecoveryStatusRes {
        path: snap.path.as_str().to_string(),
        elapsed_secs: snap.elapsed_secs,
        collections,
    })
}

#[derive(Serialize)]
struct SnapshotRes {
    last_snapshot_lsn: u64,
}

async fn snapshot_collection(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<SnapshotRes>, (StatusCode, Json<ErrorResponse>)> {
    state
        .require_collection_ready(&collection)
        .map_err(err_from_availability)?;
    match crate::snapshot::force_snapshot_collection(&state, &collection) {
        Ok(lsn) => Ok(Json(SnapshotRes { last_snapshot_lsn: lsn })),
        Err(e) if e.contains("not found") => Err(err(StatusCode::NOT_FOUND, e)),
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

#[derive(Serialize)]
struct PersistenceStatusRes {
    last_snapshot_lsn: u64,
    current_lsn: u64,
    next_lsn: u64,
}

async fn persistence_status(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<PersistenceStatusRes>, (StatusCode, Json<ErrorResponse>)> {
    state
        .require_collection_ready(&collection)
        .map_err(err_from_availability)?;
    match state.persistence_status(&collection) {
        Ok(p) => Ok(Json(PersistenceStatusRes {
            last_snapshot_lsn: p.last_snapshot_lsn,
            current_lsn: p.current_lsn,
            next_lsn: p.next_lsn,
        })),
        Err(e) if e.contains("not found") => Err(err(StatusCode::NOT_FOUND, e)),
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

#[derive(Serialize)]
struct CollectionMetricsRes {
    map_lock_acquisitions: u64,
    collection_read_acquisitions: u64,
    collection_write_acquisitions: u64,
    total_blocked_microseconds: u64,
}

async fn collection_metrics(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<CollectionMetricsRes>, (StatusCode, Json<ErrorResponse>)> {
    state
        .require_collection_ready(&collection)
        .map_err(err_from_availability)?;
    match state.collection_metrics(&collection) {
        Ok(m) => Ok(Json(CollectionMetricsRes {
            map_lock_acquisitions: m.map_lock_acquisitions,
            collection_read_acquisitions: m.collection_read_acquisitions,
            collection_write_acquisitions: m.collection_write_acquisitions,
            total_blocked_microseconds: m.total_blocked_microseconds,
        })),
        Err(e) if e.contains("not found") => Err(err(StatusCode::NOT_FOUND, e)),
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

// ── Search handlers ─────────────────────────────────────────────────────

async fn search(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<SearchReq>,
) -> Result<Json<SearchRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
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

    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

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

    let quantization_params = req.quantization.as_ref().map(|q| SearchQuantizationParams {
        rescore: q.rescore.unwrap_or(true),
        oversampling: q.oversampling.unwrap_or(3.0),
        ignore: q.ignore.unwrap_or(false),
    });

    let results = QueryExecutor::search_quantized(
        coll.index.as_vector_index(),
        &req.query,
        req.k as usize,
        filter.as_ref(),
        &strategy,
        Some(&coll.index_manager),
        &metadata_store,
        ef_search,
        quantization_params.as_ref(),
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

    metrics::record_search_latency_rest(timer, &collection);
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

    // Per-query availability guard: each query may target a different
    // collection. The guard must run before the read-lock acquisition so a
    // recovering collection produces a 503 (with retry-after info) rather
    // than a 404, matching the gRPC counterpart.
    for q in &req.queries {
        state.require_collection_ready(&q.collection).map_err(err_from_availability)?;
    }

    let mut responses = Vec::with_capacity(req.queries.len());

    for q in &req.queries {
        let query_timer = Instant::now();

        let coll_handle = state.collection_handle(&q.collection)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", q.collection)))?;
        let coll = metered_read(&coll_handle);

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

        let quantization_params = q.quantization.as_ref().map(|qp| SearchQuantizationParams {
            rescore: qp.rescore.unwrap_or(true),
            oversampling: qp.oversampling.unwrap_or(3.0),
            ignore: qp.ignore.unwrap_or(false),
        });

        let results = QueryExecutor::search_quantized(
            coll.index.as_vector_index(),
            &q.query,
            q.k as usize,
            filter.as_ref(),
            &strategy,
            Some(&coll.index_manager),
            &metadata_store,
            ef_search,
            quantization_params.as_ref(),
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

        metrics::record_search_latency_rest(query_timer, &q.collection);
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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    if coll.config.is_vector_only() {
        return Err(err(StatusCode::PRECONDITION_FAILED, format!(
            "collection '{}' is in vector-only mode; graph queries are not available",
            collection
        )));
    }

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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    if coll.config.is_vector_only() {
        return Err(err(StatusCode::PRECONDITION_FAILED, format!(
            "collection '{}' is in vector-only mode; graph queries are not available",
            collection
        )));
    }

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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&coll_handle);

    if coll.config.is_vector_only() {
        return Err(err(StatusCode::PRECONDITION_FAILED, format!(
            "collection '{}' is in vector-only mode; graph queries are not available",
            collection
        )));
    }

    if req.vector_id == 0 {
        coll.graph.config_mut().default_threshold = req.threshold;
        coll.deferred_graph.store(true, Ordering::Release);
    } else {
        coll.graph.set_vector_threshold(req.vector_id, req.threshold);
    }

    Ok(Json(SetThresholdRes { success: true }))
}

// ── Hybrid query handler (P02) ──────────────────────────────────────────

// The body is a serde QueryPlan directly; no proto translation for REST. The
// plan's steps run as composed: the default graph-augmented ranking is the
// vector_rank step (graph-first scope-then-rank, ADR-024). RRF is the opt-in
// fusion path on the gRPC surface (an explicit RrfRankSpec on the request).
async fn hybrid_query(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(plan): ValidatedJson<vf_query::hybrid::QueryPlan>,
) -> Result<Json<HybridQueryRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    // Hybrid queries need a graph layer; reject vector-only. AutoSimilarity is
    // allowed and runs against an empty typed store.
    if coll.config.effective_mode() == Mode::VectorOnly {
        return Err(err(StatusCode::PRECONDITION_FAILED, format!(
            "collection '{}' is in vector-only mode; hybrid queries are not available",
            collection
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

    let hybrid_timer = Instant::now();
    let exec = vf_query::hybrid::HybridExecutor::new(coll.index.as_vector_index(), store);
    // The executor is CPU-bound and synchronous, and runs while holding the
    // collection read guard. Offload it onto the blocking pool via
    // block_in_place so the async runtime (and the /readyz /livez probes)
    // stays responsive during a large VectorRank. block_in_place runs the
    // closure on the current worker thread, so borrowing exec and plan is
    // fine (no 'static bound, unlike spawn_blocking). Requires the
    // multi-threaded runtime, which main.rs uses (#[tokio::main] default).
    let result = tokio::task::block_in_place(|| exec.execute(&plan))
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
    metrics::record_hybrid_query_latency(hybrid_timer);

    let mut res = HybridQueryRes { nodes: Vec::new(), edges: Vec::new(), paths: Vec::new() };
    match result {
        vf_query::hybrid::QueryResult::Nodes(records) => {
            res.nodes = records.into_iter()
                .map(|r| HybridNodeRes { id: r.id.0, node: r.node })
                .collect();
        }
        vf_query::hybrid::QueryResult::Edges(edges) => {
            res.edges = edges;
        }
        vf_query::hybrid::QueryResult::Paths(paths) => {
            res.paths = paths.into_iter()
                .map(|p| p.nodes.iter().map(|n| n.0).collect())
                .collect();
        }
    }
    Ok(Json(res))
}

// ── Typed graph helpers (P04) ────────────────────────────────────────────

// Parse an optional JSON value into a property map. None -> empty map.
fn parse_props_value(
    v: Option<serde_json::Value>,
) -> Result<HashMap<String, serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(HashMap::new()),
        Some(serde_json::Value::Object(m)) => Ok(m.into_iter().collect()),
        Some(_) => Err(err(StatusCode::BAD_REQUEST, "properties must be a JSON object")),
    }
}

// Parse an optional JSON provenance value. None -> default provenance.
fn parse_provenance_value(
    v: Option<serde_json::Value>,
) -> Result<vf_graph::Provenance, (StatusCode, Json<ErrorResponse>)> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(vf_graph::Provenance::default()),
        Some(value) => serde_json::from_value(value)
            .map_err(|e| err(StatusCode::BAD_REQUEST, format!("invalid provenance: {e}"))),
    }
}

fn parse_node_source(s: Option<&str>) -> vf_graph::NodeSource {
    match s {
        Some("ingested") => vf_graph::NodeSource::Ingested,
        Some("extracted") => vf_graph::NodeSource::Extracted,
        _ => vf_graph::NodeSource::Manual,
    }
}

fn parse_edge_direction(s: Option<&str>) -> vf_graph::EdgeDirection {
    match s {
        Some("incoming") => vf_graph::EdgeDirection::Incoming,
        Some("both") => vf_graph::EdgeDirection::Both,
        _ => vf_graph::EdgeDirection::Outgoing,
    }
}

// Empty/blank actor means no actor recorded.
fn actor_opt(s: Option<String>) -> Option<String> {
    s.filter(|v| !v.trim().is_empty())
}

// Best-effort display name for a node: its "name" property if present.
fn node_display_name(store: &vf_graph::TypedGraphStore, id: vf_graph::NodeId) -> Option<String> {
    store
        .get_node(id)
        .and_then(|n| n.properties.get("name").and_then(|v| v.as_str()).map(|s| s.to_string()))
}

// Current LSN for the collection, defaulting to 0 when unknown.
fn current_lsn(state: &AppState, collection: &str) -> u64 {
    let cm = state.collection_manager.read();
    cm.get_collection(collection).map(|c| c.current_lsn()).unwrap_or(0)
}

// 412 returned when a collection lacks a typed graph store (not hybrid mode).
fn not_hybrid_err(collection: &str) -> (StatusCode, Json<ErrorResponse>) {
    err(
        StatusCode::PRECONDITION_FAILED,
        format!("collection '{}' is not in hybrid mode; typed graph is unavailable", collection),
    )
}

// ── Typed graph handlers (P04) ───────────────────────────────────────────

async fn rest_put_node(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<PutNodeReq>,
) -> Result<Json<PutNodeRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let kind = match req.kind.as_deref() {
        Some("entity") => vf_graph::NodeKind::Entity {
            label: req.label.clone().unwrap_or_default(),
        },
        _ => vf_graph::NodeKind::Content,
    };
    // Consistency guard: a content node with an embedding via put_node would be stored
    // inline but never indexed into HNSW, so it would not be searchable and would break the
    // NodeId==VectorId bridge. Searchable content vectors must go through vectors.insert.
    // Entity nodes may carry an inline embedding (graph-scoped vector_rank), and content
    // placeholders without an embedding remain allowed.
    if matches!(kind, vf_graph::NodeKind::Content) && !req.embedding.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "put_node: a content node with an embedding is not searchable via put_node; \
             create it with vectors.insert (which assigns the id, indexes the vector, and \
             links the content node by the NodeId==VectorId bridge). put_node is for entity \
             nodes (which may carry an inline embedding for graph-scoped ranking) and for \
             content placeholders without an embedding.",
        ));
    }
    let properties = parse_props_value(req.properties)?;
    let source = parse_node_source(req.source.as_deref());

    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&handle);
    let lsn = current_lsn(&state, &collection);
    let coll_ref = &mut *coll;
    // Allocate the node id from the unified per-collection id authority (the
    // vector store's next_id) BEFORE taking the &mut graph_store borrow, so
    // entity ids can never collide with vector/content ids. The id is not
    // inserted into vectors, so the NodeId == VectorId bridge stays content-only.
    let id = vf_graph::NodeId(coll_ref.store.alloc_id());
    let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
    let node = vf_graph::Node {
        id,
        kind,
        properties,
        embedding: if req.embedding.is_empty() { None } else { Some(req.embedding) },
        source,
        created_at: vf_graph::now_millis(),
        created_by: actor_opt(req.created_by),
        updated_at: None,
        history: Vec::new(),
    };
    store
        .put_node(node, lsn)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("put_node failed: {e}")))?;
    let _ = store.sync_delta();
    coll_ref.dirty.store(true, Ordering::Release);
    coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
    Ok(Json(PutNodeRes { id: id.0 }))
}

async fn rest_get_node(
    State(state): State<AppState>,
    Path((collection, node_id)): Path<(String, u64)>,
) -> Result<Json<vf_graph::Node>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&handle);
    let store = coll.graph_store.as_ref().ok_or_else(|| not_hybrid_err(&collection))?;
    match store.get_node(vf_graph::NodeId(node_id)) {
        Some(n) => Ok(Json(n)),
        None => Err(err(StatusCode::NOT_FOUND, "node not found")),
    }
}

async fn rest_delete_node(
    State(state): State<AppState>,
    Path((collection, node_id)): Path<(String, u64)>,
) -> Result<Json<DeletedRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&handle);
    let lsn = current_lsn(&state, &collection);
    let coll_ref = &mut *coll;
    let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
    let deleted = store
        .delete_node(vf_graph::NodeId(node_id), lsn)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("delete_node failed: {e}")))?;
    let _ = store.sync_delta();
    coll_ref.dirty.store(true, Ordering::Release);
    coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
    Ok(Json(DeletedRes { deleted }))
}

async fn rest_list_edges(
    State(state): State<AppState>,
    Path((collection, node_id)): Path<(String, u64)>,
    Query(q): Query<ListEdgesQuery>,
) -> Result<Json<ListEdgesRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&handle);
    let store = coll.graph_store.as_ref().ok_or_else(|| not_hybrid_err(&collection))?;
    let dir = parse_edge_direction(q.direction.as_deref());
    let filter = q.edge_type.as_deref().filter(|t| !t.trim().is_empty());
    let edges: Vec<vf_graph::TypedEdge> = store
        .edges_for_node(vf_graph::NodeId(node_id), dir)
        .into_iter()
        .filter(|e| filter.map(|t| e.edge_type.as_str() == t).unwrap_or(true))
        .collect();
    Ok(Json(ListEdgesRes { edges }))
}

async fn rest_put_edge(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<PutEdgeReq>,
) -> Result<Json<PutEdgeRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    if req.edge_type.trim().is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "edge_type must not be empty"));
    }
    let properties = parse_props_value(req.properties)?;
    let provenance = parse_provenance_value(req.provenance)?;
    let confidence = match req.confidence {
        Some(c) if c > 0.0 => c,
        _ => 1.0,
    };

    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&handle);
    let lsn = current_lsn(&state, &collection);
    let coll_ref = &mut *coll;
    let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
    let edge_type = store.intern(&req.edge_type);
    let id = store.alloc_edge_id();
    let mut edge = vf_graph::TypedEdge {
        id,
        source: vf_graph::NodeId(req.source),
        target: vf_graph::NodeId(req.target),
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
        temporal_context: req.temporal_context.filter(|c| !c.trim().is_empty()),
    };
    edge.record_audit("created", None, vf_graph::now_millis());
    store
        .put_edge(edge, lsn)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("put_edge failed: {e}")))?;
    let _ = store.sync_delta();
    coll_ref.dirty.store(true, Ordering::Release);
    coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
    Ok(Json(PutEdgeRes { id: id.0 }))
}

async fn rest_get_edge(
    State(state): State<AppState>,
    Path((collection, edge_id)): Path<(String, u64)>,
) -> Result<Json<vf_graph::TypedEdge>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&handle);
    let store = coll.graph_store.as_ref().ok_or_else(|| not_hybrid_err(&collection))?;
    match store.get_edge(vf_graph::EdgeId(edge_id)) {
        Some(e) => Ok(Json(e)),
        None => Err(err(StatusCode::NOT_FOUND, "edge not found")),
    }
}

async fn rest_update_node(
    State(state): State<AppState>,
    Path((collection, node_id)): Path<(String, u64)>,
    ValidatedJson(req): ValidatedJson<UpdateNodeReq>,
) -> Result<Json<NodeRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    // Parse optional properties outside the lock so a bad payload fails fast.
    // Only the property bag is mutable; provenance and embedding are immutable
    // so the NodeId==VectorId bridge cannot desync.
    let new_props = match req.properties {
        Some(serde_json::Value::Null) | None => None,
        Some(v) => Some(parse_props_value(Some(v))?),
    };
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&handle);
    let lsn = current_lsn(&state, &collection);
    let coll_ref = &mut *coll;
    let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
    let node = store
        .update_node(
            vf_graph::NodeId(node_id),
            new_props,
            actor_opt(req.actor),
            vf_graph::now_millis(),
            lsn,
        )
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("update_node failed: {e}")))?
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "node not found"))?;
    let _ = store.sync_delta();
    coll_ref.dirty.store(true, Ordering::Release);
    coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
    Ok(Json(NodeRes { node }))
}

async fn rest_update_edge(
    State(state): State<AppState>,
    Path((collection, edge_id)): Path<(String, u64)>,
    ValidatedJson(req): ValidatedJson<UpdateEdgeReq>,
) -> Result<Json<EdgeRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    // Parse optional properties outside the lock so a bad payload fails fast.
    let new_props = match req.properties {
        Some(serde_json::Value::Null) | None => None,
        Some(v) => Some(parse_props_value(Some(v))?),
    };
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&handle);
    let lsn = current_lsn(&state, &collection);
    let coll_ref = &mut *coll;
    let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
    let mut edge = store
        .get_edge(vf_graph::EdgeId(edge_id))
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "edge not found"))?;
    if !edge.is_manual {
        return Err(err(StatusCode::PRECONDITION_FAILED, "only manual edges may be updated"));
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
    edge.record_audit("updated", actor_opt(req.actor), vf_graph::now_millis());
    store
        .put_edge(edge.clone(), lsn)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("put_edge failed: {e}")))?;
    let _ = store.sync_delta();
    coll_ref.dirty.store(true, Ordering::Release);
    coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
    Ok(Json(EdgeRes { edge }))
}

async fn rest_delete_edge(
    State(state): State<AppState>,
    Path((collection, edge_id)): Path<(String, u64)>,
) -> Result<Json<DeletedRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&handle);
    let lsn = current_lsn(&state, &collection);
    let coll_ref = &mut *coll;
    let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
    let deleted = store
        .delete_edge(vf_graph::EdgeId(edge_id), lsn)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("delete_edge failed: {e}")))?;
    let _ = store.sync_delta();
    coll_ref.dirty.store(true, Ordering::Release);
    coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
    Ok(Json(DeletedRes { deleted }))
}

async fn rest_verify_edge(
    State(state): State<AppState>,
    Path((collection, edge_id)): Path<(String, u64)>,
    ValidatedJson(req): ValidatedJson<ActorReq>,
) -> Result<Json<EdgeRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let mut coll = metered_write(&handle);
    let lsn = current_lsn(&state, &collection);
    let coll_ref = &mut *coll;
    let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
    let mut edge = store
        .get_edge(vf_graph::EdgeId(edge_id))
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "edge not found"))?;
    edge.verified = true;
    edge.record_audit("verified", actor_opt(req.actor), vf_graph::now_millis());
    store
        .put_edge(edge.clone(), lsn)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("put_edge failed: {e}")))?;
    let _ = store.sync_delta();
    coll_ref.dirty.store(true, Ordering::Release);
    coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
    Ok(Json(EdgeRes { edge }))
}

async fn rest_reject_edge(
    State(state): State<AppState>,
    Path((collection, edge_id)): Path<(String, u64)>,
    ValidatedJson(req): ValidatedJson<ActorReq>,
) -> Result<Json<RejectEdgeRes>, (StatusCode, Json<ErrorResponse>)> {
    let _ = req; // body accepted for parity; actor is unused on reject.
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;

    // Delete the edge and build the reject rule under the write guard; the
    // guard is dropped before calling into the extraction manager.
    let (deleted, rule) = {
        let mut coll = metered_write(&handle);
        let lsn = current_lsn(&state, &collection);
        let coll_ref = &mut *coll;
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
        let edge = match store.get_edge(vf_graph::EdgeId(edge_id)) {
            Some(e) => e,
            None => return Ok(Json(RejectEdgeRes { deleted: false, rule_added: false })),
        };
        let src_name = node_display_name(store, edge.source);
        let tgt_name = node_display_name(store, edge.target);
        let rule = vf_extraction::RejectRule {
            source_doc: edge.provenance.source_doc.clone(),
            source_chunk_id: edge.provenance.source_chunk_id,
            edge_type: edge.edge_type.as_str().to_string(),
            source_name: src_name,
            target_name: tgt_name,
        };
        let deleted = store
            .delete_edge(vf_graph::EdgeId(edge_id), lsn)
            .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, format!("delete_edge failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        (deleted, rule)
    };

    // Guard dropped: now safe to call into the extraction manager.
    let rule_added = match state.extraction.add_reject_rule(&collection, rule) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!("failed to persist reject rule: {e}");
            false
        }
    };
    Ok(Json(RejectEdgeRes { deleted, rule_added }))
}

async fn rest_bulk_import_edges(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<BulkImportEdgesReq>,
) -> Result<Json<BulkImportEdgesRes>, (StatusCode, Json<ErrorResponse>)> {
    state.require_collection_ready(&collection).map_err(err_from_availability)?;

    let format = match req.format.to_ascii_lowercase().as_str() {
        "csv" => crate::edge_ops::BulkFormat::Csv,
        "jsonl" => crate::edge_ops::BulkFormat::Jsonl,
        other => {
            return Err(err(
                StatusCode::BAD_REQUEST,
                format!("unsupported format '{other}'; expected 'csv' or 'jsonl'"),
            ))
        }
    };

    // Snapshot the known edge types from the merged ontology (no guard held).
    let mut known_edge_types: std::collections::HashSet<String> = state
        .extraction
        .get_ontology(&collection)
        .map(|o| o.edge_types.into_iter().map(|t| t.edge_type).collect())
        .unwrap_or_default();

    // Parse the payload.
    let (parsed_rows, parse_errors) = crate::edge_ops::parse_bulk_edges(format, &req.data);
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
                    .map(|t| vf_extraction::EdgeTypeDef::new(t, String::new(), Vec::new(), Vec::new()))
                    .collect(),
                // Auto-edge-type extension carries no user prompt.
                system_prompt: None,
                extra_guidance: None,
                link_passages: false,
                // Auto-edge-type extension does not change resolution mode.
                entity_resolution: vf_extraction::EntityResolution::Normalized,
            };
            state
                .extraction
                .set_ontology(&collection, None, extension, false)
                .map_err(err_from_extraction)?;
            known_edge_types = state
                .extraction
                .get_ontology(&collection)
                .map(|o| o.edge_types.into_iter().map(|t| t.edge_type).collect())
                .unwrap_or(known_edge_types);
        }
    }

    // Apply rows under the write guard.
    let handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let (imported, apply_errors) = {
        let mut coll = metered_write(&handle);
        let lsn = current_lsn(&state, &collection);
        let coll_ref = &mut *coll;
        // Build the valid-endpoint id set before taking the &mut graph_store
        // borrow. An id is valid if it is an existing plain vector (the
        // NodeId == VectorId bridge treats it as a virtual content node) or a
        // materialized typed node. This owns a HashSet and holds no borrow.
        let valid_node_ids: std::collections::HashSet<u64> = {
            let mut candidates: std::collections::HashSet<u64> = std::collections::HashSet::new();
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
        let store = coll_ref.graph_store.as_mut().ok_or_else(|| not_hybrid_err(&collection))?;
        let (imported, errs) = crate::edge_ops::apply_bulk_edges(
            store,
            &known_edge_types,
            &valid_node_ids,
            parsed_rows,
            actor_opt(req.actor),
            lsn,
        );
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        (imported, errs)
    };

    let mut errors: Vec<BulkImportRowErrorRes> =
        Vec::with_capacity(parse_errors.len() + apply_errors.len());
    for e in parse_errors.into_iter().chain(apply_errors.into_iter()) {
        errors.push(BulkImportRowErrorRes { row: e.row, message: e.message });
    }
    Ok(Json(BulkImportEdgesRes {
        total_rows,
        imported,
        failed: total_rows.saturating_sub(imported),
        errors,
    }))
}

// ── Document diff / re-extraction handlers (P04) ─────────────────────────

fn chunk_diff_action_str(a: vf_extraction::ChunkDiffAction) -> &'static str {
    match a {
        vf_extraction::ChunkDiffAction::Unchanged => "unchanged",
        vf_extraction::ChunkDiffAction::Changed => "changed",
        vf_extraction::ChunkDiffAction::New => "new",
        vf_extraction::ChunkDiffAction::Deleted => "deleted",
    }
}

async fn rest_diff_document(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<DiffDocumentReq>,
) -> Result<Json<DiffDocumentRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(ChunkReq::into_content).collect();
    let diffs = state
        .extraction
        .diff_document(&collection, &req.doc_id, &chunks)
        .map_err(err_from_extraction)?;
    Ok(Json(DiffDocumentRes {
        diffs: diffs
            .into_iter()
            .map(|d| ChunkDiffRes {
                chunk_id: d.chunk_id,
                action: chunk_diff_action_str(d.action).to_string(),
            })
            .collect(),
    }))
}

async fn rest_reextract_document(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<DiffDocumentReq>,
) -> Result<Json<ReextractDocumentRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(ChunkReq::into_content).collect();
    let s = state
        .extraction
        .reextract_document(&collection, &req.doc_id, chunks)
        .map_err(err_from_extraction)?;
    Ok(Json(ReextractDocumentRes {
        job_id: s.job_id,
        unchanged: s.unchanged,
        changed: s.changed,
        added: s.added,
        deleted: s.deleted,
        edges_deleted: s.edges_deleted,
        nodes_deleted: s.nodes_deleted,
    }))
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

fn get_vectors_from_collection(index: &dyn vf_index::traits::PersistableIndex, ids: &[u64]) -> Vec<(VectorId, Vec<f32>)> {
    if ids.is_empty() {
        index.iter_vectors_owned()
    } else {
        ids.iter()
            .filter_map(|&id| index.get_vector(id).ok().map(|data| (id, data)))
            .collect()
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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let timer = Instant::now();
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    let owned_vectors = get_vectors_from_collection(&*coll.index, &[]);
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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let timer = Instant::now();
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    if req.direction.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "direction vector is required"));
    }

    let owned_vectors = get_vectors_from_collection(&*coll.index, &[]);
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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let timer = Instant::now();
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    let owned_vectors = get_vectors_from_collection(&*coll.index, &req.vector_ids);
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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let timer = Instant::now();
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    let owned_w1 = get_vectors_from_collection(&*coll.index, &req.window1_ids);
    let owned_w2 = get_vectors_from_collection(&*coll.index, &req.window2_ids);

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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let timer = Instant::now();
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    let owned_vectors = get_vectors_from_collection(&*coll.index, &[]);
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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let timer = Instant::now();
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    let owned_vectors = get_vectors_from_collection(&*coll.index, &req.vector_ids);
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
    state.require_collection_ready(&collection).map_err(err_from_availability)?;
    let timer = Instant::now();
    let coll_handle = state.collection_handle(&collection)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection)))?;
    let coll = metered_read(&coll_handle);

    if req.query.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "query vector is required"));
    }

    let owned_candidates = get_vectors_from_collection(&*coll.index, &req.candidate_ids);
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

// ── Extraction handlers (Hybrid mode only, P03) ──────────────────────────

use vf_extraction::{
    ChunkContent, EdgeTypeDef, EntityLabelDef, EntityResolution, ExtractionError, LlmConfig,
    Ontology,
};

/// Map an `ExtractionError` onto a REST `(status, body)`.
fn err_from_extraction(e: ExtractionError) -> (StatusCode, Json<ErrorResponse>) {
    match e {
        ExtractionError::Config(m) => err(StatusCode::BAD_REQUEST, m),
        ExtractionError::Ontology(m) => err(StatusCode::BAD_REQUEST, m),
        ExtractionError::Parse(m) => err(StatusCode::BAD_REQUEST, m),
        ExtractionError::JobNotFound(m) => err(StatusCode::NOT_FOUND, format!("job not found: {m}")),
        ExtractionError::Crypto(m) => err(StatusCode::PRECONDITION_FAILED, m),
        ExtractionError::Cancelled => {
            err(StatusCode::CONFLICT, "extraction cancelled".to_string())
        }
        ExtractionError::Llm(m) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("llm error: {m}")),
        ExtractionError::Io(m) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("io error: {m}")),
        ExtractionError::Graph(m) => {
            err(StatusCode::INTERNAL_SERVER_ERROR, format!("graph write error: {m}"))
        }
    }
}

/// Readiness + Hybrid-mode gate for the extraction REST handlers. Non-Hybrid
/// collections return 412 Precondition Failed.
fn extraction_gate(
    state: &AppState,
    collection: &str,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    state
        .require_collection_ready(collection)
        .map_err(err_from_availability)?;
    let handle = state.collection_handle(collection).ok_or_else(|| {
        err(StatusCode::NOT_FOUND, format!("collection '{}' not found", collection))
    })?;
    let mode = {
        let coll = metered_read(&handle);
        coll.config.effective_mode()
    };
    if mode != Mode::Hybrid {
        return Err(err(
            StatusCode::PRECONDITION_FAILED,
            format!("collection '{}' is not in hybrid mode; extraction is not available", collection),
        ));
    }
    Ok(())
}

// ── Extraction DTOs ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChunkReq {
    pub doc_id: String,
    pub chunk_id: u64,
    pub text: String,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

impl ChunkReq {
    fn into_content(self) -> ChunkContent {
        ChunkContent {
            doc_id: self.doc_id,
            chunk_id: self.chunk_id,
            text: self.text,
            embedding: self.embedding,
        }
    }
}

#[derive(Deserialize)]
struct SetLlmConfigReq {
    base_url: String,
    api_key: String,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
    timeout_seconds: u64,
}

#[derive(Serialize)]
struct GetLlmConfigRes {
    base_url: String,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
    timeout_seconds: u64,
    api_key_set: bool,
}

#[derive(Deserialize)]
struct RotateLlmConfigReq {
    new_api_key: String,
}

#[derive(Serialize)]
struct SuccessRes {
    success: bool,
}

#[derive(Serialize, Deserialize)]
struct EntityLabelDto {
    label: String,
    description: String,
}

#[derive(Serialize, Deserialize)]
struct EdgeTypeDto {
    edge_type: String,
    description: String,
    #[serde(default)]
    source_labels: Vec<String>,
    #[serde(default)]
    target_labels: Vec<String>,
}

#[derive(Serialize, Deserialize, Default)]
struct OntologyDto {
    #[serde(default)]
    entity_labels: Vec<EntityLabelDto>,
    #[serde(default)]
    edge_types: Vec<EdgeTypeDto>,
    // Full override of the generic extraction task framing. Empty means use the default prompt.
    #[serde(default)]
    system_prompt: Option<String>,
    // Domain guidance appended on top of the framing. Empty means use the default prompt.
    #[serde(default)]
    extra_guidance: Option<String>,
    // ADR-012. Opt-in passage-to-entity linking for GraphRAG.
    #[serde(default)]
    link_passages: bool,
    // ADR-020. Entity-resolution mode; "normalized" (default) or "fuzzy".
    #[serde(default)]
    entity_resolution: EntityResolution,
}

impl OntologyDto {
    fn from_ontology(o: &Ontology) -> Self {
        Self {
            entity_labels: o
                .entity_labels
                .iter()
                .map(|l| EntityLabelDto {
                    label: l.label.clone(),
                    description: l.description.clone(),
                })
                .collect(),
            edge_types: o
                .edge_types
                .iter()
                .map(|t| EdgeTypeDto {
                    edge_type: t.edge_type.clone(),
                    description: t.description.clone(),
                    source_labels: t.source_labels.clone(),
                    target_labels: t.target_labels.clone(),
                })
                .collect(),
            system_prompt: o.system_prompt.clone(),
            extra_guidance: o.extra_guidance.clone(),
            link_passages: o.link_passages,
            entity_resolution: o.entity_resolution,
        }
    }

    fn into_ontology(self) -> Ontology {
        Ontology {
            entity_labels: self
                .entity_labels
                .into_iter()
                .map(|l| EntityLabelDef::new(l.label, l.description))
                .collect(),
            edge_types: self
                .edge_types
                .into_iter()
                .map(|t| EdgeTypeDef::new(t.edge_type, t.description, t.source_labels, t.target_labels))
                .collect(),
            // Empty/whitespace -> None so the engine falls back to the default prompt.
            system_prompt: self
                .system_prompt
                .filter(|s| !s.trim().is_empty()),
            extra_guidance: self
                .extra_guidance
                .filter(|s| !s.trim().is_empty()),
            link_passages: self.link_passages,
            entity_resolution: self.entity_resolution,
        }
    }
}

#[derive(Deserialize)]
struct SetOntologyReq {
    #[serde(default)]
    base_template: Option<String>,
    #[serde(default)]
    extension: OntologyDto,
    #[serde(default)]
    replace: bool,
}

#[derive(Deserialize)]
struct ChunksReq {
    #[serde(default)]
    chunks: Vec<ChunkReq>,
}

#[derive(Serialize)]
struct CostPreviewRes {
    chunks: u64,
    estimated_input_tokens: u64,
    estimated_output_tokens: u64,
    estimated_cost_usd: f64,
    model: String,
    pricing_known: bool,
}

#[derive(Serialize)]
struct StartExtractionRes {
    job_id: String,
}

#[derive(Serialize)]
struct ChunkErrorRes {
    doc_id: String,
    chunk_id: u64,
    error: String,
}

#[derive(Serialize)]
struct JobStatusRes {
    job_id: String,
    collection: String,
    state: String,
    total_chunks: u64,
    processed_chunks: u64,
    entities_written: u64,
    edges_written: u64,
    cache_hits: u64,
    cache_misses: u64,
    error: Option<String>,
    failed_chunks: u64,
    chunk_errors: Vec<ChunkErrorRes>,
}

#[derive(Serialize)]
struct ProposalRes {
    id: String,
    kind: String,
    name: String,
    description: String,
    examples: Vec<String>,
    status: String,
    source_doc: Option<String>,
    source_chunk_id: Option<u64>,
}

#[derive(Serialize)]
struct ListProposalsRes {
    proposals: Vec<ProposalRes>,
}

fn job_state_str(state: vf_extraction::JobState) -> &'static str {
    match state {
        vf_extraction::JobState::Queued => "queued",
        vf_extraction::JobState::Running => "running",
        vf_extraction::JobState::Completed => "completed",
        vf_extraction::JobState::CompletedWithErrors => "completed_with_errors",
        vf_extraction::JobState::Failed => "failed",
        vf_extraction::JobState::Cancelled => "cancelled",
    }
}

fn job_status_to_res(s: vf_extraction::JobStatus) -> JobStatusRes {
    JobStatusRes {
        job_id: s.job_id,
        collection: s.collection,
        state: job_state_str(s.state).to_string(),
        total_chunks: s.total_chunks as u64,
        processed_chunks: s.processed_chunks as u64,
        entities_written: s.entities_written as u64,
        edges_written: s.edges_written as u64,
        cache_hits: s.cache_hits as u64,
        cache_misses: s.cache_misses as u64,
        error: s.error,
        failed_chunks: s.failed_chunks as u64,
        chunk_errors: s
            .chunk_errors
            .iter()
            .map(|c| ChunkErrorRes {
                doc_id: c.doc_id.clone(),
                chunk_id: c.chunk_id,
                error: c.error.clone(),
            })
            .collect(),
    }
}

fn proposal_to_res(p: &vf_extraction::OntologyProposal) -> ProposalRes {
    let kind = match p.kind {
        vf_extraction::ProposalKind::EntityLabel => "entity_label",
        vf_extraction::ProposalKind::EdgeType => "edge_type",
    };
    let status = match p.status {
        vf_extraction::ProposalStatus::Pending => "pending",
        vf_extraction::ProposalStatus::Approved => "approved",
        vf_extraction::ProposalStatus::Rejected => "rejected",
    };
    ProposalRes {
        id: p.id.clone(),
        kind: kind.to_string(),
        name: p.name.clone(),
        description: p.description.clone(),
        examples: p.examples.clone(),
        status: status.to_string(),
        source_doc: p.source_doc.clone(),
        source_chunk_id: p.source_chunk_id,
    }
}

// ── Extraction handlers ──────────────────────────────────────────────

async fn set_llm_config(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<SetLlmConfigReq>,
) -> Result<Json<SuccessRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let config = LlmConfig::new(
        req.base_url,
        req.api_key,
        req.model_name,
        req.temperature,
        req.max_tokens,
        req.timeout_seconds,
    );
    state
        .extraction
        .set_llm_config(&collection, config)
        .map_err(err_from_extraction)?;
    Ok(Json(SuccessRes { success: true }))
}

async fn get_llm_config(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<GetLlmConfigRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let r = state
        .extraction
        .get_llm_config(&collection)
        .map_err(err_from_extraction)?;
    Ok(Json(GetLlmConfigRes {
        base_url: r.base_url,
        model_name: r.model_name,
        temperature: r.temperature,
        max_tokens: r.max_tokens,
        timeout_seconds: r.timeout_seconds,
        api_key_set: r.api_key_set,
    }))
}

async fn rotate_llm_config(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<RotateLlmConfigReq>,
) -> Result<Json<SuccessRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    state
        .extraction
        .rotate_llm_config(&collection, &req.new_api_key)
        .map_err(err_from_extraction)?;
    Ok(Json(SuccessRes { success: true }))
}

async fn set_ontology(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<SetOntologyReq>,
) -> Result<Json<SuccessRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let base_template = req
        .base_template
        .filter(|s| !s.trim().is_empty());
    state
        .extraction
        .set_ontology(&collection, base_template, req.extension.into_ontology(), req.replace)
        .map_err(err_from_extraction)?;
    Ok(Json(SuccessRes { success: true }))
}

async fn get_ontology(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<OntologyDto>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let ontology = state
        .extraction
        .get_ontology(&collection)
        .map_err(err_from_extraction)?;
    Ok(Json(OntologyDto::from_ontology(&ontology)))
}

async fn extraction_cost_preview(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<ChunksReq>,
) -> Result<Json<CostPreviewRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(ChunkReq::into_content).collect();
    let estimate = state
        .extraction
        .cost_preview(&collection, &chunks)
        .await
        .map_err(err_from_extraction)?;
    Ok(Json(CostPreviewRes {
        chunks: estimate.chunks as u64,
        estimated_input_tokens: estimate.estimated_input_tokens,
        estimated_output_tokens: estimate.estimated_output_tokens,
        estimated_cost_usd: estimate.estimated_cost_usd,
        model: estimate.model,
        pricing_known: estimate.pricing_known,
    }))
}

async fn start_extraction(
    State(state): State<AppState>,
    Path(collection): Path<String>,
    ValidatedJson(req): ValidatedJson<ChunksReq>,
) -> Result<Json<StartExtractionRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(ChunkReq::into_content).collect();
    let job_id = state
        .extraction
        .start_extraction(&collection, chunks)
        .map_err(err_from_extraction)?;
    metrics::record_extraction_job("started");
    Ok(Json(StartExtractionRes { job_id }))
}

async fn extraction_status(
    State(state): State<AppState>,
    Path((collection, job_id)): Path<(String, String)>,
) -> Result<Json<JobStatusRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let status = state
        .extraction
        .job_status(&collection, &job_id)
        .map_err(err_from_extraction)?;
    let total = status.cache_hits + status.cache_misses;
    if total > 0 {
        metrics::set_extraction_cache_hit_rate(
            &collection,
            status.cache_hits as f64 / total as f64,
        );
    }
    Ok(Json(job_status_to_res(status)))
}

async fn cancel_extraction(
    State(state): State<AppState>,
    Path((collection, job_id)): Path<(String, String)>,
) -> Result<Json<SuccessRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    state
        .extraction
        .cancel_extraction(&collection, &job_id)
        .map_err(err_from_extraction)?;
    Ok(Json(SuccessRes { success: true }))
}

async fn list_proposals(
    State(state): State<AppState>,
    Path(collection): Path<String>,
) -> Result<Json<ListProposalsRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    let proposals = state
        .extraction
        .list_proposals(&collection)
        .map_err(err_from_extraction)?;
    Ok(Json(ListProposalsRes {
        proposals: proposals.iter().map(proposal_to_res).collect(),
    }))
}

async fn approve_proposal(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<Json<SuccessRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    state
        .extraction
        .approve_proposal(&collection, &id)
        .map_err(err_from_extraction)?;
    Ok(Json(SuccessRes { success: true }))
}

async fn reject_proposal(
    State(state): State<AppState>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<Json<SuccessRes>, (StatusCode, Json<ErrorResponse>)> {
    extraction_gate(&state, &collection)?;
    state
        .extraction
        .reject_proposal(&collection, &id)
        .map_err(err_from_extraction)?;
    Ok(Json(SuccessRes { success: true }))
}
