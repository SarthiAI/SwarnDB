// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

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
/// 3. Persists HNSW topology snapshots (via atomic_write)
/// 4. Persists virtual graph base snapshots (via atomic_write)
/// 5. Updates wal_meta.json with final LSN
/// 6. Writes a `shutdown_clean` marker per collection
/// 7. Logs shutdown completion
pub async fn graceful_shutdown(state: AppState) {
    tracing::info!("graceful shutdown initiated, draining resources...");

    // Phase 1: Flush all memtables via CollectionManager.
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

    // Phase 2: Persist HNSW topology, graph snapshots, WAL meta, and
    // write clean shutdown marker for each collection.
    {
        let collections = state.collections.read();
        let cm = state.collection_manager.read();

        for (name, coll_state) in collections.iter() {
            // Resolve collection directory and current LSN from CollectionManager.
            let (collection_dir, current_lsn) = match cm.get_collection(name) {
                Ok(storage_coll) => (
                    storage_coll.collection_dir().to_path_buf(),
                    storage_coll.current_lsn(),
                ),
                Err(e) => {
                    tracing::error!(
                        collection = %name,
                        "failed to get storage collection for snapshot: {}",
                        e
                    );
                    continue;
                }
            };

            // 1. Snapshot and persist HNSW topology.
            {
                let snapshot = coll_state.index.snapshot_topology(current_lsn);
                let hnsw_path = collection_dir.join("hnsw.base");
                let res = vf_storage::atomic_write::atomic_write_with_callback(
                    &hnsw_path,
                    |file| {
                        vf_index::hnsw_persistence::serialize_topology(&snapshot, file)
                            .map_err(|e| {
                                std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    format!("HNSW serialize error: {}", e),
                                )
                            })?;
                        Ok(())
                    },
                );
                match res {
                    Ok(()) => {
                        tracing::info!(collection = %name, "HNSW topology snapshot persisted");
                    }
                    Err(e) => {
                        tracing::error!(
                            collection = %name,
                            "failed to persist HNSW topology: {}",
                            e
                        );
                        continue;
                    }
                }
            }

            // Persist any index-specific sidecars via the PersistableIndex trait.
            // On plain HNSW this rewrites hnsw.base (redundant but cheap).
            // On SQ8 this writes quantizer.json + codes.bin + vectors.mmap.
            if let Err(e) = coll_state.index.serialize_state_to_dir(&collection_dir) {
                tracing::error!(
                    collection = %name,
                    "failed to persist index sidecars via trait: {}", e
                );
                // Continue: hnsw.base was already written by the manual path above, so
                // the collection is at least partially persisted on a sidecar error.
            }

            // Sync pending HNSW delta writer buffers so the BufWriter contents
            // survive shutdown and feed the next boot's IncrementalReplay path.
            if let Some(mut delta_writer) = coll_state.index.take_delta_writer() {
                if let Err(e) = delta_writer.sync() {
                    tracing::warn!(
                        collection = %name,
                        "failed to sync hnsw delta on graceful shutdown: {}", e
                    );
                }
                drop(delta_writer);
            }

            // 2. Serialize and persist virtual graph base snapshot (only if not deferred).
            {
                let graph_deferred = coll_state.deferred_graph.load(std::sync::atomic::Ordering::Acquire);
                if !graph_deferred {
                    let graph_path = collection_dir.join("graph.base");
                    let res = vf_storage::atomic_write::atomic_write_with_callback(
                        &graph_path,
                        |file| {
                            vf_graph::serialize_base(&coll_state.graph, current_lsn, file)
                                .map_err(|e| {
                                    std::io::Error::new(
                                        std::io::ErrorKind::Other,
                                        format!("graph serialize error: {}", e),
                                    )
                                })?;
                            Ok(())
                        },
                    );
                    match res {
                        Ok(()) => {
                            tracing::info!(collection = %name, "virtual graph snapshot persisted");
                        }
                        Err(e) => {
                            tracing::error!(
                                collection = %name,
                                "failed to persist virtual graph: {}",
                                e
                            );
                            continue;
                        }
                    }
                } else {
                    tracing::info!(collection = %name, "skipping graph snapshot (deferred)");
                }
            }

            // 3. Update wal_meta.json with the final LSN.
            {
                let meta = vf_storage::wal::WalMeta::new(current_lsn);
                if let Err(e) = vf_storage::wal::save_wal_meta(&collection_dir, &meta) {
                    tracing::error!(
                        collection = %name,
                        "failed to update wal_meta.json: {}",
                        e
                    );
                    continue;
                }
                tracing::info!(
                    collection = %name,
                    lsn = current_lsn,
                    "wal_meta.json updated"
                );
            }

            // 4. Write clean shutdown marker (LAST, after all snapshots).
            if let Err(e) = vf_storage::collection::write_shutdown_marker(&collection_dir) {
                tracing::error!(
                    collection = %name,
                    "failed to write shutdown marker: {}",
                    e
                );
            } else {
                tracing::info!(collection = %name, "clean shutdown marker written");
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
