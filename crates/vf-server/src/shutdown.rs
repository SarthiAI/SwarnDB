// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! Graceful shutdown handling for the SwarnDB server.
//!
//! Listens for OS termination signals (SIGTERM, SIGINT) and coordinates
//! a clean shutdown: draining connections, flushing WAL, and syncing segments.

use crate::state::{metered_read, AppState};

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
    // write clean shutdown marker for each collection. Map read lock is
    // released after the handle snapshot so per-collection reads do not
    // contend with the map RwLock.
    {
        let handles: Vec<(String, std::sync::Arc<parking_lot::RwLock<crate::state::CollectionState>>)> = {
            let collections = state.collections.read();
            collections
                .iter()
                .map(|(k, v)| (k.clone(), std::sync::Arc::clone(v)))
                .collect()
        };
        let cm = state.collection_manager.read();

        for (name, handle) in handles.iter() {
            let coll_state = metered_read(handle);
            let name = name.as_str();
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

            // Orphan guard (must run before any base/marker write): a deferred or
            // never-optimized collection can hold an empty live index over
            // non-empty storage. Writing an empty hnsw.base plus a clean-shutdown
            // marker would steer the next boot down the CleanShutdown path, which
            // trusts the empty base as good and never rebuilds. When that shape is
            // seen, skip the empty-base write, the sidecar serialize, and the
            // clean-shutdown marker so the next boot replays/rebuilds from storage
            // (where the recovery orphan guard backstops it). Mirrors the
            // index.len()==0 over non-empty storage condition in state.rs recovery.
            let index_empty_over_storage =
                coll_state.store.len() > 0 && coll_state.index.len() == 0;
            if index_empty_over_storage {
                tracing::warn!(
                    collection = %name,
                    stored = coll_state.store.len(),
                    "shutdown: empty index over non-empty storage; skipping empty base \
                     and clean-shutdown marker so next boot rebuilds from storage"
                );
            }

            // 1. Snapshot and persist HNSW topology (skip if the index is empty
            //    over non-empty storage; see the orphan guard above).
            if !index_empty_over_storage {
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
                // Release the snapshot's per-node neighbor clones before the
                // shutdown loop continues to the next collection.
                drop(snapshot);
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
            }

            // G1 ordering invariant: hnsw.base MUST be fsynced before
            // hnsw.delta so recovery (which replays delta only for LSN >
            // base.embedded_lsn) can safely discard any orphaned delta tail
            // after a torn shutdown. Do not reorder these two persist steps.
            //
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

            // 2b. Serialize the typed graph base for Hybrid collections (ADR-007 R4).
            if let Some(ref gs) = coll_state.graph_store {
                let typed_path = collection_dir.join("graph_typed.base");
                let res = vf_storage::atomic_write::atomic_write_with_callback(
                    &typed_path,
                    |file| {
                        vf_graph::serialize_typed_base(gs, current_lsn, file).map_err(|e| {
                            std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("typed graph serialize error: {}", e),
                            )
                        })?;
                        Ok(())
                    },
                );
                match res {
                    Ok(()) => {
                        tracing::info!(collection = %name, "typed graph snapshot persisted")
                    }
                    Err(e) => {
                        tracing::error!(collection = %name, "failed to persist typed graph: {}", e)
                    }
                }
                // Sync the typed delta tail so post-snapshot edge writes survive a crash.
                if let Err(e) = gs.sync_delta() {
                    tracing::warn!(
                        collection = %name,
                        "failed to sync typed graph delta on shutdown: {}", e
                    );
                }
            }

            // 3. Update wal_meta.json with the final LSN.
            //    Both next_lsn and last_snapshot_lsn advance to current_lsn:
            //    the hnsw.base and graph.base writes above embed current_lsn,
            //    so last_snapshot_lsn must mirror that. Loading the existing
            //    meta first preserves the wal_format_version field.
            //
            //    Orphan guard: when the index is empty over non-empty storage,
            //    no fresh hnsw.base was written above, so advancing
            //    last_snapshot_lsn to current_lsn would understate the work the
            //    next boot has to redo and could mask a stale base as current.
            //    Leave last_snapshot_lsn untouched in that case (only next_lsn
            //    advances) so recovery treats the collection as needing a
            //    rebuild from storage rather than trusting a snapshot envelope
            //    that does not exist.
            {
                let mut meta = match vf_storage::wal::load_wal_meta(&collection_dir) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(
                            collection = %name,
                            "failed to load wal_meta.json before shutdown update: {}", e
                        );
                        vf_storage::wal::WalMeta::new(current_lsn)
                    }
                };
                meta.next_lsn = current_lsn;
                if !index_empty_over_storage {
                    meta.last_snapshot_lsn = current_lsn;
                }
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
            //    Orphan guard: skip the marker when the index is empty over
            //    non-empty storage. plan_recovery() picks CleanShutdown only
            //    when both the marker and hnsw.base are present; without the
            //    marker the next boot falls through to IncrementalReplay or
            //    FullRebuild, which rebuilds the index from storage (and the
            //    recovery orphan guard in state.rs backstops the same shape).
            //    Writing the marker here would steer the next boot down the
            //    CleanShutdown path and trust a stale/empty base as good.
            if index_empty_over_storage {
                // Remove any stale marker from a prior clean shutdown so a
                // leftover marker plus a stale hnsw.base cannot still trip the
                // CleanShutdown path on the next boot.
                vf_storage::collection::remove_shutdown_marker(&collection_dir);
                tracing::warn!(
                    collection = %name,
                    "shutdown: empty index over non-empty storage; clean-shutdown \
                     marker skipped so next boot rebuilds from storage"
                );
            } else if let Err(e) = vf_storage::collection::write_shutdown_marker(&collection_dir) {
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
