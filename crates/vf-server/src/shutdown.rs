// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Graceful shutdown handling for the SwarnDB server.
//!
//! Listens for OS termination signals (SIGTERM, SIGINT) and coordinates
//! a clean shutdown: draining connections, flushing WAL, and syncing segments.

use crate::state::AppState;

/// Waits for a shutdown signal (SIGTERM or SIGINT).
///
/// Returns once any termination signal is received. On Unix systems,
/// both SIGTERM and SIGINT are handled. On other platforms, only
/// Ctrl+C (SIGINT equivalent) is supported.
pub async fn wait_for_shutdown() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("received SIGINT (Ctrl+C), initiating graceful shutdown");
        }
        _ = terminate => {
            tracing::info!("received SIGTERM, initiating graceful shutdown");
        }
    }
}

/// Performs graceful shutdown of all server resources.
///
/// This function:
/// 1. Logs shutdown initiation
/// 2. Iterates all loaded collections and flushes pending data
///    (WAL flush, segment sync)
/// 3. Logs shutdown completion
///
/// Since `Collection` lives in `vf-storage`, the actual flush/sync
/// calls are conceptual here and will be wired up once the storage
/// layer exposes the necessary APIs.
pub async fn graceful_shutdown(state: AppState) {
    tracing::info!("graceful shutdown initiated, draining resources...");

    // Flush all storage collections via CollectionManager.
    {
        let mut cm = state.collection_manager.write();
        let names: Vec<String> = cm.list_collections().iter().map(|s| s.to_string()).collect();
        let collection_count = names.len();

        if collection_count == 0 {
            tracing::info!("no collections loaded, nothing to flush");
        } else {
            tracing::info!(
                "flushing {} collection(s) before shutdown",
                collection_count
            );

            for name in &names {
                tracing::info!(collection = %name, "flushing WAL and syncing segments");
                match cm.get_collection_mut(name) {
                    Ok(coll) => {
                        if let Err(e) = coll.flush_memtable() {
                            tracing::error!(collection = %name, "flush failed: {}", e);
                        } else {
                            tracing::info!(collection = %name, "collection flush complete");
                        }
                    }
                    Err(e) => {
                        tracing::error!(collection = %name, "failed to get collection for flush: {}", e);
                    }
                }
            }
        }
    }

    tracing::info!("graceful shutdown complete, all resources released");
}

/// A signal wrapper that can be shared across server components.
///
/// Wraps `tokio::sync::watch` to broadcast shutdown notification
/// to all interested parties (REST server, gRPC server, background tasks).
pub struct ShutdownSignal {
    sender: tokio::sync::watch::Sender<bool>,
    receiver: tokio::sync::watch::Receiver<bool>,
}

impl ShutdownSignal {
    /// Creates a new shutdown signal in the non-signaled state.
    pub fn new() -> Self {
        let (sender, receiver) = tokio::sync::watch::channel(false);
        Self { sender, receiver }
    }

    /// Returns a receiver that can be cloned and shared with server components.
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<bool> {
        self.receiver.clone()
    }

    /// Triggers the shutdown signal, notifying all subscribers.
    pub fn trigger(&self) {
        let _ = self.sender.send(true);
        tracing::info!("shutdown signal broadcast to all subscribers");
    }

    /// Returns true if shutdown has been signaled.
    pub fn is_shutdown(&self) -> bool {
        *self.receiver.borrow()
    }
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// Guard that performs cleanup when dropped.
///
/// Attach to scopes where resources must be released on exit,
/// even in the case of panics or early returns.
pub struct ShutdownGuard {
    state: Option<AppState>,
    signal: Option<ShutdownSignal>,
}

impl ShutdownGuard {
    /// Creates a new shutdown guard that will flush state and trigger
    /// the signal on drop.
    pub fn new(state: AppState, signal: ShutdownSignal) -> Self {
        Self {
            state: Some(state),
            signal: Some(signal),
        }
    }
}

impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        if let Some(signal) = self.signal.take() {
            signal.trigger();
        }

        if let Some(state) = self.state.take() {
            let mut cm = state.collection_manager.write();
            let names: Vec<String> = cm.list_collections().iter().map(|s| s.to_string()).collect();
            let count = names.len();
            if count > 0 {
                tracing::warn!(
                    "ShutdownGuard dropped with {} collection(s) still loaded, \
                     performing synchronous cleanup",
                    count
                );
                for name in &names {
                    tracing::info!(collection = %name, "emergency flush on guard drop");
                    if let Ok(coll) = cm.get_collection_mut(name) {
                        if let Err(e) = coll.flush_memtable() {
                            tracing::error!(collection = %name, "emergency flush failed: {}", e);
                        }
                    }
                }
            }
            drop(cm);
            tracing::info!("ShutdownGuard cleanup complete");
        }
    }
}
