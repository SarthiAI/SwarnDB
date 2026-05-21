// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

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
    /// Filled while recovery is still running so operators can see the boot
    /// progress. Omitted once `status == "ok"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collections_loaded: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collections_total: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_progress: Option<Vec<String>>,
}

/// `/health` returns 503 only if recovery has been running longer than this
/// grace window. Liveness probes typically run on a short interval; we want a
/// long window so /health stays 200 during a normal cold restart and only
/// flips to 503 if the box is genuinely stuck.
const HEALTH_RECOVERY_GRACE: Duration = Duration::from_secs(900);

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

async fn health(State(state): State<ProbeState>) -> impl IntoResponse {
    let version = env!("CARGO_PKG_VERSION").to_string();

    if state.server_status.is_initialized() {
        return (
            StatusCode::OK,
            Json(HealthResponse {
                status: "ok".to_string(),
                version,
                collections_loaded: None,
                collections_total: None,
                in_progress: None,
            }),
        );
    }

    // Recovery is still in flight. Surface the progress so operators can
    // distinguish a slow boot from a wedged box.
    let ls = state.app.loading_state.read();
    let loaded = ls.loaded();
    let total = ls.total;
    let elapsed = ls.started_at.elapsed();
    let mut in_progress: Vec<String> = ls.in_progress.iter().cloned().collect();
    in_progress.sort();
    drop(ls);

    let body = HealthResponse {
        status: "recovering".to_string(),
        version,
        collections_loaded: Some(loaded),
        collections_total: Some(total),
        in_progress: Some(in_progress),
    };

    // Stay 200 OK during a normal recovery so the Docker / runc liveness
    // probe does not start killing the container before the work is done.
    // Only flip to 503 if recovery has been stuck past the grace window;
    // orchestration-level readiness gating belongs on /readyz.
    let status = if elapsed > HEALTH_RECOVERY_GRACE {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };
    (status, Json(body))
}

async fn ready(State(state): State<AppState>) -> Json<ReadyResponse> {
    // Snapshot handles under a short map read lock, then sum vector counts via
    // per-collection read locks so /ready never blocks behind a bulk insert.
    let (handles, count): (Vec<_>, usize) = {
        let collections = state.collections.read();
        let v: Vec<_> = collections.values().cloned().collect();
        let n = v.len();
        (v, n)
    };
    let total_vectors: u64 = handles
        .iter()
        .map(|h| h.read().store.len() as u64)
        .sum();
    Json(ReadyResponse {
        ready: true,
        collections: count,
        total_vectors,
    })
}

// ---------------------------------------------------------------------------
// K8s probe handlers
// ---------------------------------------------------------------------------

/// `GET /healthz` - Kubernetes liveness probe.
/// Returns 200 as long as the server process is running (simple ping).
async fn healthz() -> Json<ProbeResponse> {
    Json(ProbeResponse {
        status: "alive".to_string(),
        checks: None,
    })
}

/// `GET /readyz` - Kubernetes readiness probe.
/// Returns 200 when the server is ready to receive production traffic and
/// 503 otherwise. The contract here is intentionally stricter than `/health`:
/// liveness (`/health`) stays 200 throughout a slow boot so the container is
/// not killed mid-recovery, while readiness (`/readyz`) only flips to 200
/// once every persisted collection has finished loading.
///
/// Checks:
///   - `server_initialized`: the boot path has marked recovery complete
///   - `collections_accessible`: a read lock on the collections map is
///      obtainable (proves the runtime is not deadlocked)
async fn readyz(State(state): State<ProbeState>) -> impl IntoResponse {
    let mut checks: HashMap<String, String> = HashMap::new();
    let mut all_ok = true;

    // Check 1: recovery must be complete before /readyz returns 200. This is
    // the gate the docker / k8s orchestration uses to route traffic.
    if state.server_status.is_initialized() {
        checks.insert("server_initialized".to_string(), "ok".to_string());
    } else {
        // Surface the recovery view so orchestrators can show progress.
        let ls = state.app.loading_state.read();
        checks.insert(
            "server_initialized".to_string(),
            format!(
                "recovering {} of {} collections",
                ls.loaded(),
                ls.total
            ),
        );
        drop(ls);
        all_ok = false;
    }

    // Check 2: collections map is accessible (no global lock deadlock).
    // `parking_lot::RwLock::try_read` returns None if the lock cannot be acquired.
    match state.app.collections.try_read() {
        Some(_guard) => {
            checks.insert("collections_accessible".to_string(), "ok".to_string());
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

/// `GET /startupz` - Kubernetes startup probe.
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

    // `/ready` still uses AppState; it predates the K8s probe surface.
    let app_only = Router::new()
        .route("/ready", get(ready))
        .with_state(state);

    // `/health`, plus the K8s probe endpoints, all share the same probe
    // state so that /health can read both `server_status` and the AppState
    // loading view.
    let probes = Router::new()
        .route("/health", get(health))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/startupz", get(startupz))
        .with_state(probe_state);

    app_only.merge(probes)
}
