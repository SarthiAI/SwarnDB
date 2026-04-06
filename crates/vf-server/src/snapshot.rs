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

use crate::state::AppState;

/// Configuration for the background snapshot scheduler.
pub struct SnapshotConfig {
    /// How often to check snapshot triggers (default 30s).
    pub check_interval: Duration,
    /// Snapshot after this many mutations (default 50_000).
    pub mutation_threshold: u64,
    /// Max time between snapshots (default 15 min).
    pub time_interval: Duration,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            mutation_threshold: 50_000,
            time_interval: Duration::from_secs(900),
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
            let (is_dirty, mutations) = {
                let collections = state.collections.read();
                match collections.get(name) {
                    Some(coll) => (
                        coll.dirty.load(Ordering::Acquire),
                        coll.mutation_count.load(Ordering::Acquire),
                    ),
                    None => continue,
                }
            };

            let mutation_triggered = mutations >= config.mutation_threshold;

            // Trigger if dirty AND (time interval exceeded OR mutation threshold exceeded).
            if !is_dirty || (!time_triggered && !mutation_triggered) {
                continue;
            }

            // Resolve the collection directory from CollectionManager.
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

            // Get the current LSN from the storage collection.
            let current_lsn = {
                let cm = state.collection_manager.read();
                match cm.get_collection(name) {
                    Ok(coll) => coll.current_lsn(),
                    Err(_) => continue,
                }
            };

            // Perform the snapshot under a read lock on collections.
            match snapshot_collection(name, &state, &collection_dir, current_lsn) {
                Ok(()) => {
                    last_snapshot_times.insert(name.clone(), Instant::now());
                    tracing::info!(
                        collection = %name,
                        lsn = current_lsn,
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

/// Snapshot a single collection's HNSW topology and virtual graph to base files.
///
/// Steps:
/// 1. Snapshot HNSW topology and atomic-write to hnsw.base
/// 2. Serialize virtual graph and atomic-write to graph.base
/// 3. Take and reset delta writers (HNSW + graph)
/// 4. Update wal_meta.json with last_snapshot_lsn
/// 5. Prune old WAL files
fn snapshot_collection(
    name: &str,
    state: &AppState,
    collection_dir: &Path,
    current_lsn: u64,
) -> Result<(), String> {
    // 1. Snapshot HNSW topology under a read lock.
    {
        let collections = state.collections.read();
        let coll_state = collections
            .get(name)
            .ok_or_else(|| format!("collection '{}' not found", name))?;

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
    }

    // 2. Serialize virtual graph under a read lock (only if not deferred).
    {
        let collections = state.collections.read();
        let coll_state = collections
            .get(name)
            .ok_or_else(|| format!("collection '{}' not found", name))?;

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

    // 3. Reset delta writers under a write lock.
    //    Take existing writers, create fresh ones, and set them back.
    {
        let mut collections = state.collections.write();
        let coll_state = collections
            .get_mut(name)
            .ok_or_else(|| format!("collection '{}' not found", name))?;

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
    {
        let mut meta = vf_storage::wal::load_wal_meta(collection_dir)
            .map_err(|e| format!("failed to load wal_meta: {}", e))?;
        meta.last_snapshot_lsn = current_lsn;
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
    {
        let collections = state.collections.read();
        if let Some(coll_state) = collections.get(name) {
            coll_state.dirty.store(false, Ordering::Release);
            coll_state.mutation_count.store(0, Ordering::Release);
        }
    }

    Ok(())
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
