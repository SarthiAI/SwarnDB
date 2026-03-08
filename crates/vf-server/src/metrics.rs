// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Prometheus metrics module for SwarnDB vf-server.
//!
//! Provides metric recording helpers and a `/metrics` HTTP handler
//! that renders all collected metrics in Prometheus exposition format.

use std::time::Instant;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

// ── Metric name constants ───────────────────────────────────────────────

/// Search latency histogram (seconds) — REST endpoint.
pub const SEARCH_LATENCY_REST: &str = "swarndb_search_latency_seconds_rest";

/// Search latency histogram (seconds) — gRPC endpoint.
pub const SEARCH_LATENCY_GRPC: &str = "swarndb_search_latency_seconds_grpc";

/// Total request counter, labelled by endpoint and status.
pub const REQUEST_TOTAL: &str = "swarndb_requests_total";

/// Total error counter, labelled by endpoint and error kind.
pub const ERROR_TOTAL: &str = "swarndb_errors_total";

/// Gauge: number of vectors currently stored across all collections.
pub const VECTOR_COUNT: &str = "swarndb_vectors_total";

/// Gauge: number of active collections.
pub const COLLECTION_COUNT: &str = "swarndb_collections_total";

/// Gauge: WAL size in bytes.
pub const WAL_SIZE_BYTES: &str = "swarndb_wal_size_bytes";

/// Gauge: number of storage segments.
pub const SEGMENT_COUNT: &str = "swarndb_segments_total";

/// Histogram: per-query ef_search override values.
pub const SEARCH_EF_USED: &str = "swarndb_search_ef_used";

// ── Setup ───────────────────────────────────────────────────────────────

/// Installs the Prometheus metrics recorder and returns a handle
/// that can later render the collected metrics.
///
/// Call this **once** at server startup before any metrics are recorded.
pub fn setup_metrics() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus metrics recorder")
}

// ── Axum handler ────────────────────────────────────────────────────────

/// Axum handler that renders all Prometheus metrics as plain text.
///
/// Mount this on the REST router as `GET /metrics`.
pub async fn metrics_handler(
    axum::extract::State(handle): axum::extract::State<PrometheusHandle>,
) -> impl IntoResponse {
    let body = handle.render();
    (StatusCode::OK, [("content-type", "text/plain; version=0.0.4; charset=utf-8")], body)
}

// ── Recording helpers ───────────────────────────────────────────────────

/// Records search latency for a REST request.
pub fn record_search_latency_rest(start: Instant, collection: &str) {
    let elapsed = start.elapsed().as_secs_f64();
    histogram!(SEARCH_LATENCY_REST, "collection" => collection.to_owned())
        .record(elapsed);
}

/// Records search latency for a gRPC request.
pub fn record_search_latency_grpc(start: Instant, collection: &str) {
    let elapsed = start.elapsed().as_secs_f64();
    histogram!(SEARCH_LATENCY_GRPC, "collection" => collection.to_owned())
        .record(elapsed);
}

/// Increments the request counter for the given endpoint and status.
pub fn record_request(endpoint: &str, status: &str) {
    counter!(REQUEST_TOTAL, "endpoint" => endpoint.to_owned(), "status" => status.to_owned())
        .increment(1);
}

/// Increments the error counter for the given endpoint and error kind.
pub fn record_error(endpoint: &str, kind: &str) {
    counter!(ERROR_TOTAL, "endpoint" => endpoint.to_owned(), "kind" => kind.to_owned())
        .increment(1);
}

/// Updates the vector-count gauge.
pub fn set_vector_count(count: u64) {
    gauge!(VECTOR_COUNT).set(count as f64);
}

/// Updates the collection-count gauge.
pub fn set_collection_count(count: u64) {
    gauge!(COLLECTION_COUNT).set(count as f64);
}

/// Updates the WAL size gauge (bytes).
pub fn set_wal_size_bytes(bytes: u64) {
    gauge!(WAL_SIZE_BYTES).set(bytes as f64);
}

/// Updates the segment count gauge.
pub fn set_segment_count(count: u64) {
    gauge!(SEGMENT_COUNT).set(count as f64);
}

/// Records a per-query ef_search override value.
pub fn record_ef_search(ef_value: usize, collection: &str) {
    histogram!(SEARCH_EF_USED, "collection" => collection.to_owned())
        .record(ef_value as f64);
}
