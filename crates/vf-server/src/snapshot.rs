// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Background snapshot scheduler for periodic HNSW and graph persistence.
//!
//! Spawns a tokio task that periodically checks each collection and snapshots
//! HNSW topology + virtual graph to base files when mutation thresholds or
//! time intervals are exceeded. Also prunes old WAL files after each snapshot.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::watch;

use std::sync::atomic::Ordering;

use crate::state::{metered_read, metered_write, AppState};

/// Configuration for the background snapshot scheduler.
pub struct SnapshotConfig {
    /// How often to check snapshot triggers (default 30s).
    pub check_interval: Duration,
    /// Snapshot after this many mutations (default 25_000, sized for 60s recovery SLA at 1M).
    pub mutation_threshold: u64,
    /// Max time between snapshots (default 120s, sized for 60s recovery SLA at 1M).
    pub time_interval: Duration,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            mutation_threshold: 25_000,
            time_interval: Duration::from_secs(120),
        }
    }
}

impl SnapshotConfig {
    /// Create a SnapshotConfig from environment variable values.
    pub fn from_env(
        check_interval_secs: u64,
        mutation_threshold: u64,
        interval_secs: u64,
    ) -> Self {
        Self {
            check_interval: Duration::from_secs(check_interval_secs),
            mutation_threshold,
            time_interval: Duration::from_secs(interval_secs),
        }
    }
}

/// Start the background snapshot scheduler.
///
/// Spawns a loop that periodically checks each collection and snapshots
/// HNSW topology + virtual graph to base files when triggered.
/// Responds to the shutdown signal for clean cancellation.
pub async fn start_snapshot_scheduler(
    state: Arc<AppState>,
    mut shutdown_rx: watch::Receiver<bool>,
    config: SnapshotConfig,
) {
    let mut last_snapshot_times: HashMap<String, Instant> = HashMap::new();

    tracing::info!(
        check_interval_secs = config.check_interval.as_secs(),
        mutation_threshold = config.mutation_threshold,
        time_interval_secs = config.time_interval.as_secs(),
        "snapshot scheduler started"
    );

    loop {
        tokio::select! {
            _ = tokio::time::sleep(config.check_interval) => {},
            _ = shutdown_rx.changed() => {
                tracing::info!("snapshot scheduler shutting down");
                return;
            }
        }

        // Collect the list of collection names under a short read lock.
        let collection_names: Vec<String> = {
            let collections = state.collections.read();
            collections.keys().cloned().collect()
        };

        for name in &collection_names {
            let now = Instant::now();
            let last_time = last_snapshot_times
                .get(name)
                .copied()
                .unwrap_or(now);

            // On first encounter, record the time and skip (avoids immediate snapshot on startup).
            if !last_snapshot_times.contains_key(name) {
                last_snapshot_times.insert(name.clone(), now);
                continue;
            }

            let time_since_last = now.duration_since(last_time);
            let time_triggered = time_since_last >= config.time_interval;

            // Check dirty flag and mutation count from CollectionState.
            // Brief per-collection read lock; atomics are cheap.
            let (is_dirty, mutations) = match state.collection_handle(name) {
                Some(handle) => {
                    let coll = metered_read(&handle);
                    (
                        coll.dirty.load(Ordering::Acquire),
                        coll.mutation_count.load(Ordering::Acquire),
                    )
                }
                None => continue,
            };

            let mutation_triggered = mutations >= config.mutation_threshold;

            // Trigger if dirty AND (time interval exceeded OR mutation threshold exceeded).
            if !is_dirty || (!time_triggered && !mutation_triggered) {
                continue;
            }

            // Resolve the collection directory from CollectionManager.
            // current_lsn is read inside snapshot_collection so the LSN
            // and the snapshot share the same lifetime window.
            let collection_dir = {
                let cm = state.collection_manager.read();
                match cm.get_collection(name) {
                    Ok(coll) => coll.collection_dir().to_path_buf(),
                    Err(e) => {
                        tracing::warn!(
                            collection = %name,
                            "snapshot scheduler: cannot resolve collection dir: {}",
                            e
                        );
                        continue;
                    }
                }
            };

            // Perform the snapshot under a read lock on collections.
            match snapshot_collection(name, &state, &collection_dir) {
                Ok(snapshot_lsn) => {
                    last_snapshot_times.insert(name.clone(), Instant::now());
                    tracing::info!(
                        collection = %name,
                        lsn = snapshot_lsn,
                        "background snapshot completed"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        collection = %name,
                        "background snapshot failed: {}",
                        e
                    );
                }
            }
        }
    }
}

/// Force a synchronous snapshot of a single collection. Resolves the
/// collection directory from the storage manager and delegates to
/// `snapshot_collection`. Returns the embedded LSN on success.
///
/// This is the entry point for the `POST /collections/{name}/snapshot`
/// REST endpoint and its matching gRPC RPC. Callers should not hold any
/// per-collection lock when invoking it; the snapshot path takes its own
/// per-collection read and write locks in sequence.
pub fn force_snapshot_collection(state: &AppState, name: &str) -> Result<u64, String> {
    let collection_dir = {
        let cm = state.collection_manager.read();
        let coll = cm
            .get_collection(name)
            .map_err(|e| format!("collection '{}' not found: {}", name, e))?;
        coll.collection_dir().to_path_buf()
    };
    snapshot_collection(name, state, &collection_dir)
}

/// Snapshot a single collection's HNSW topology and virtual graph to base files.
///
/// Steps:
/// 1. Snapshot HNSW topology and atomic-write to hnsw.base
/// 2. Serialize virtual graph and atomic-write to graph.base
/// 3. Take and reset delta writers (HNSW + graph)
/// 4. Update wal_meta.json with last_snapshot_lsn
/// 5. Prune old WAL files
///
/// Returns the LSN that was embedded into the snapshot.
fn snapshot_collection(
    name: &str,
    state: &AppState,
    collection_dir: &Path,
) -> Result<u64, String> {
    // 1. Read current_lsn BEFORE taking the snapshot. This is the "safe
    //    understatement" direction: the embedded LSN may be slightly lower
    //    than the highest LSN actually captured in the snapshot (any insert
    //    that races between the LSN read and the snapshot's internal
    //    inner.read() acquisition lands in both the snapshot AND the delta
    //    tail). On replay, those entries get re-applied from the delta;
    //    making AddNode idempotent at the index layer is the correctness
    //    closer and lives outside this initiative. Reversing the order
    //    would OVERSTATE LSN coverage and silently lose inserts that
    //    committed after the snapshot, which is the worse failure mode.
    let current_lsn = {
        let cm = state.collection_manager.read();
        match cm.get_collection(name) {
            Ok(coll) => coll.current_lsn(),
            Err(e) => return Err(format!("cannot resolve current_lsn for '{}': {}", name, e)),
        }
    };

    // atomic_write_with_callback writes to .tmp, fsyncs, then renames.
    // A crash mid-write leaves the previous hnsw.base intact; recovery
    // never observes a torn snapshot. Per-collection read lock keeps the
    // scheduler concurrent with searches on the same collection and with
    // mutation work on other collections.
    {
        let handle = state
            .collection_handle(name)
            .ok_or_else(|| format!("collection '{}' not found", name))?;
        let coll_state = metered_read(&handle);

        let snapshot = coll_state.index.snapshot_topology(current_lsn);
        let hnsw_path = collection_dir.join("hnsw.base");

        vf_storage::atomic_write::atomic_write_with_callback(&hnsw_path, |file| {
            vf_index::hnsw_persistence::serialize_topology(&snapshot, file).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("HNSW serialize error: {}", e),
                )
            })?;
            Ok(())
        })
        .map_err(|e| format!("HNSW snapshot write failed: {}", e))?;

        // Release the per-node neighbor clones the snapshot held before the
        // next scheduler tick allocates its own.
        drop(snapshot);
    }

    // 2. Serialize virtual graph under a per-collection read lock (only if not deferred).
    {
        let handle = state
            .collection_handle(name)
            .ok_or_else(|| format!("collection '{}' not found", name))?;
        let coll_state = metered_read(&handle);

        let graph_deferred = coll_state.deferred_graph.load(Ordering::Acquire);
        if !graph_deferred {
            let graph_path = collection_dir.join("graph.base");

            vf_storage::atomic_write::atomic_write_with_callback(&graph_path, |file| {
                vf_graph::serialize_base(&coll_state.graph, current_lsn, file).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("graph serialize error: {}", e),
                    )
                })?;
                Ok(())
            })
            .map_err(|e| format!("graph snapshot write failed: {}", e))?;
        }
    }

    // 3. Reset delta writers under a per-collection write lock.
    //    Take existing writers, create fresh ones, and set them back.
    //    Per-collection (not map) write lock keeps the rest of the server
    //    responsive to mutations on other collections.
    {
        let handle = state
            .collection_handle(name)
            .ok_or_else(|| format!("collection '{}' not found", name))?;
        let mut coll_state = metered_write(&handle);

        // Reset HNSW delta writer.
        let _old_hnsw_delta = coll_state.index.take_delta_writer();
        let hnsw_delta_path = collection_dir.join("hnsw.delta");
        match vf_index::hnsw_delta::HnswDeltaWriter::create(&hnsw_delta_path) {
            Ok(writer) => {
                coll_state.index.set_delta_writer(writer);
            }
            Err(e) => {
                tracing::warn!(
                    collection = %name,
                    "failed to create fresh HNSW delta writer: {}",
                    e
                );
            }
        }

        // Reset graph delta writer.
        let _old_graph_delta = coll_state.graph.take_delta_writer();
        let graph_delta_path = collection_dir.join("graph.delta");
        match vf_graph::GraphDeltaWriter::create(&graph_delta_path) {
            Ok(writer) => {
                coll_state.graph.set_delta_writer(writer);
            }
            Err(e) => {
                tracing::warn!(
                    collection = %name,
                    "failed to create fresh graph delta writer: {}",
                    e
                );
            }
        }
    }

    // 4. Update wal_meta.json with last_snapshot_lsn.
    //    next_lsn is bumped to current_lsn so a crash right after this point
    //    leaves wal_meta in sync with the snapshot envelope. save_wal_meta
    //    uses atomic_write under the hood, so a torn write is impossible.
    {
        let mut meta = vf_storage::wal::load_wal_meta(collection_dir)
            .map_err(|e| format!("failed to load wal_meta: {}", e))?;
        meta.last_snapshot_lsn = current_lsn;
        if meta.next_lsn < current_lsn {
            meta.next_lsn = current_lsn;
        }
        vf_storage::wal::save_wal_meta(collection_dir, &meta)
            .map_err(|e| format!("failed to save wal_meta: {}", e))?;
    }

    // 5. Prune old WAL files.
    let pruned = prune_old_wal_files(collection_dir);
    if pruned > 0 {
        tracing::info!(
            collection = %name,
            deleted = pruned,
            "pruned old WAL files"
        );
    }

    // 6. Reset dirty flag and mutation count.
    if let Some(handle) = state.collection_handle(name) {
        let coll_state = metered_read(&handle);
        coll_state.dirty.store(false, Ordering::Release);
        coll_state.mutation_count.store(0, Ordering::Release);
    }

    Ok(current_lsn)
}

/// Prune old WAL files after a successful snapshot.
///
/// Deletes all `wal_*.log.old` files in the collection directory.
/// These are rotated WAL segments that are no longer needed because
/// the snapshot captures all state up to the current LSN.
///
/// Returns the number of files deleted.
pub(crate) fn prune_old_wal_files(collection_dir: &Path) -> usize {
    let mut deleted = 0;
    if let Ok(entries) = std::fs::read_dir(collection_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("wal_") && name.ends_with(".log.old") {
                    if std::fs::remove_file(&path).is_ok() {
                        deleted += 1;
                    } else {
                        tracing::warn!(
                            path = %path.display(),
                            "failed to delete old WAL file"
                        );
                    }
                }
            }
        }
    }
    deleted
}
