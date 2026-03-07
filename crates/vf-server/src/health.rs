// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Existing response types (unchanged)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Serialize)]
pub struct ReadyResponse {
    pub ready: bool,
    pub collections: usize,
    pub total_vectors: u64,
}

// ---------------------------------------------------------------------------
// K8s probe types
// ---------------------------------------------------------------------------

/// Tracks whether the server has completed its initialization sequence.
/// Shared via `Arc` so it can be set from the startup path and read by probes.
#[derive(Clone)]
pub struct ServerStatus {
    pub initialized: Arc<AtomicBool>,
}

impl ServerStatus {
    pub fn new() -> Self {
        Self {
            initialized: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Mark the server as fully initialized (call once startup is complete).
    pub fn mark_initialized(&self) {
        self.initialized.store(true, Ordering::Release);
    }

    /// Check whether the server has finished initializing.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }
}

impl Default for ServerStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic response body for K8s probe endpoints.
#[derive(Serialize)]
pub struct ProbeResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checks: Option<HashMap<String, String>>,
}

/// Combined state passed to K8s probe handlers.
#[derive(Clone)]
pub struct ProbeState {
    pub app: AppState,
    pub server_status: ServerStatus,
}

// ---------------------------------------------------------------------------
// Existing handlers (unchanged)
// ---------------------------------------------------------------------------

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn ready(State(state): State<AppState>) -> Json<ReadyResponse> {
    let collections = state.collections.read();
    let total_vectors: u64 = collections
        .values()
        .map(|c| c.store.len() as u64)
        .sum();
    Json(ReadyResponse {
        ready: true,
        collections: collections.len(),
        total_vectors,
    })
}

// ---------------------------------------------------------------------------
// K8s probe handlers
// ---------------------------------------------------------------------------

/// `GET /healthz` – Kubernetes liveness probe.
/// Returns 200 as long as the server process is running (simple ping).
async fn healthz() -> Json<ProbeResponse> {
    Json(ProbeResponse {
        status: "alive".to_string(),
        checks: None,
    })
}

/// `GET /readyz` – Kubernetes readiness probe.
/// Returns 200 when all checks pass, 503 otherwise.
///
/// Checks:
///   - `collections_accessible`: can acquire a read lock on the collections map
///   - `collections_loaded`: at least one collection exists OR server just started
async fn readyz(State(state): State<ProbeState>) -> impl IntoResponse {
    let mut checks: HashMap<String, String> = HashMap::new();
    let mut all_ok = true;

    // Check 1: collections are accessible (can acquire read lock without blocking forever).
    // `parking_lot::RwLock::try_read` returns None if the lock cannot be acquired.
    match state.app.collections.try_read() {
        Some(guard) => {
            checks.insert("collections_accessible".to_string(), "ok".to_string());

            // Check 2: at least one collection exists, or server just started (not yet initialized).
            let has_collections = !guard.is_empty();
            let just_started = !state.server_status.is_initialized();
            if has_collections || just_started {
                checks.insert("collections_loaded".to_string(), "ok".to_string());
            } else {
                checks.insert(
                    "collections_loaded".to_string(),
                    "no collections loaded".to_string(),
                );
                all_ok = false;
            }
        }
        None => {
            checks.insert(
                "collections_accessible".to_string(),
                "lock unavailable".to_string(),
            );
            all_ok = false;
        }
    }

    let response = ProbeResponse {
        status: if all_ok {
            "ready".to_string()
        } else {
            "not_ready".to_string()
        },
        checks: Some(checks),
    };

    if all_ok {
        (StatusCode::OK, Json(response))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(response))
    }
}

/// `GET /startupz` – Kubernetes startup probe.
/// Returns 200 once server initialization is complete, 503 otherwise.
async fn startupz(State(state): State<ProbeState>) -> impl IntoResponse {
    if state.server_status.is_initialized() {
        (
            StatusCode::OK,
            Json(ProbeResponse {
                status: "started".to_string(),
                checks: None,
            }),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ProbeResponse {
                status: "starting".to_string(),
                checks: None,
            }),
        )
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the health/probe router.
///
/// The `ServerStatus` is accepted separately so callers can hold a reference
/// to call `mark_initialized()` once startup completes.
pub fn health_router(state: AppState, server_status: ServerStatus) -> Router {
    let probe_state = ProbeState {
        app: state.clone(),
        server_status,
    };

    // Existing endpoints (AppState)
    let existing = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .with_state(state);

    // K8s probe endpoints (ProbeState)
    let probes = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/startupz", get(startupz))
        .with_state(probe_state);

    existing.merge(probes)
}
