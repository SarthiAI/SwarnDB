// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! Crash recovery utilities for the SwarnDB storage engine.
//!
//! [`RecoveryManager`] replays the WAL into a memtable, verifies segment
//! integrity, and orchestrates full collection recovery after an unclean
//! shutdown.

use std::path::{Path, PathBuf};

use log::{info, warn};

use crate::error::{StorageError, StorageResult};
use crate::format::WalOp;
use crate::segment::Segment;
use crate::wal::WalReader;
use vf_core::store::{InMemoryVectorStore, VectorRecord};
use vf_core::types::{Metadata, VectorId};
use vf_core::vector::VectorData;

// ── Supporting structs ──────────────────────────────────────────────────────

/// Statistics gathered during WAL replay.
#[derive(Debug, Default)]
pub struct RecoveryStats {
    pub entries_replayed: u64,
    pub inserts: u64,
    pub updates: u64,
    pub deletes: u64,
    pub errors_skipped: u64,
}

/// Result of verifying a single segment file.
#[derive(Debug)]
pub struct SegmentVerification {
    pub path: PathBuf,
    pub valid: bool,
    pub vector_count: u64,
    pub error: Option<String>,
}

/// Full report returned by [`RecoveryManager::recover_collection`].
#[derive(Debug)]
pub struct RecoveryReport {
    pub segments_verified: Vec<SegmentVerification>,
    pub wal_stats: RecoveryStats,
    pub memtable_size: usize,
}

// ── RecoveryManager ─────────────────────────────────────────────────────────

/// Stateless utility for crash recovery operations.
pub struct RecoveryManager;

impl RecoveryManager {
    /// Replay a WAL file into the given in-memory vector store.
    ///
    /// If the WAL file does not exist, returns empty [`RecoveryStats`] (not an
    /// error). On a CRC mismatch the replay stops immediately - entries after
    /// corruption are considered unreliable.
    pub fn replay_wal(
        wal_path: &Path,
        memtable: &InMemoryVectorStore,
    ) -> StorageResult<RecoveryStats> {
        let mut stats = RecoveryStats::default();

        // If the WAL file does not exist, return empty stats.
        if !wal_path.exists() {
            info!("WAL file does not exist at {:?}, nothing to replay", wal_path);
            return Ok(stats);
        }

        let reader = WalReader::open(wal_path)?;
        info!("Replaying WAL from {:?}", wal_path);

        for result in reader {
            match result {
                Ok(entry) => {
                    match entry.op {
                        WalOp::Insert => {
                            match Self::apply_insert(&entry.payload, memtable) {
                                Ok(()) => stats.inserts += 1,
                                Err(e) => {
                                    warn!("WAL replay: skipping insert entry: {}", e);
                                    stats.errors_skipped += 1;
                                }
                            }
                        }
                        WalOp::Update => {
                            match Self::apply_update(&entry.payload, memtable) {
                                Ok(()) => stats.updates += 1,
                                Err(e) => {
                                    warn!("WAL replay: skipping update entry: {}", e);
                                    stats.errors_skipped += 1;
                                }
                            }
                        }
                        WalOp::Delete => {
                            match Self::apply_delete(&entry.payload, memtable) {
                                Ok(()) => stats.deletes += 1,
                                Err(e) => {
                                    warn!("WAL replay: skipping delete entry: {}", e);
                                    stats.errors_skipped += 1;
                                }
                            }
                        }
                        WalOp::CreateCollection | WalOp::DropCollection => {
                            // Handled at CollectionManager level; skip here.
                        }
                    }
                    stats.entries_replayed += 1;
                }
                Err(StorageError::ChecksumMismatch { expected, computed }) => {
                    warn!(
                        "WAL CRC mismatch at {:?}: expected {:#010x}, computed {:#010x}. \
                         Stopping replay - subsequent entries are unreliable.",
                        wal_path, expected, computed
                    );
                    break;
                }
                Err(e) => {
                    warn!("WAL replay: error reading entry: {}. Stopping replay.", e);
                    break;
                }
            }
        }

        info!(
            "WAL replay complete: {} entries, {} inserts, {} updates, {} deletes, {} errors skipped",
            stats.entries_replayed, stats.inserts, stats.updates, stats.deletes, stats.errors_skipped
        );

        Ok(stats)
    }

    /// Verify the integrity of a single segment file.
    ///
    /// Opens the segment, validates its header (magic + checksum), and
    /// iterates all vectors to confirm readability.
    pub fn verify_segment(segment_path: &Path) -> StorageResult<SegmentVerification> {
        let segment = match Segment::open(segment_path) {
            Ok(seg) => seg,
            Err(e) => {
                return Ok(SegmentVerification {
                    path: segment_path.to_path_buf(),
                    valid: false,
                    vector_count: 0,
                    error: Some(format!("failed to open segment: {}", e)),
                });
            }
        };

        // Header is already validated by Segment::open (magic + checksum).
        // Now iterate all vectors to verify readability.
        let count = segment.vector_count();
        for result in segment.iter_vectors() {
            if let Err(e) = result {
                return Ok(SegmentVerification {
                    path: segment_path.to_path_buf(),
                    valid: false,
                    vector_count: count,
                    error: Some(format!("vector read error: {}", e)),
                });
            }
        }

        Ok(SegmentVerification {
            path: segment_path.to_path_buf(),
            valid: true,
            vector_count: count,
            error: None,
        })
    }

    /// Recover a collection directory after an unclean shutdown.
    ///
    /// 1. Scans for `*.vfs` segment files and verifies each.
    /// 2. Replays the WAL (`wal.bin`) into a fresh [`InMemoryVectorStore`].
    /// 3. Returns a [`RecoveryReport`] summarising what was found.
    pub fn recover_collection(
        collection_dir: &Path,
        dimension: usize,
    ) -> StorageResult<RecoveryReport> {
        info!("Recovering collection from {:?}", collection_dir);

        // ── 1. Scan and verify segments ─────────────────────────────────
        let mut segments_verified = Vec::new();

        if collection_dir.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(collection_dir)
                .map_err(StorageError::Io)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map_or(false, |ext| ext == "vfs")
                })
                .collect();

            // Sort by filename for deterministic order.
            entries.sort_by_key(|e| e.file_name());

            for entry in entries {
                let path = entry.path();
                info!("Verifying segment {:?}", path);
                let verification = Self::verify_segment(&path)?;
                segments_verified.push(verification);
            }
        }

        // ── 2. Replay WAL ───────────────────────────────────────────────
        let wal_path = collection_dir.join("wal.log");
        let memtable = InMemoryVectorStore::new(dimension);
        let wal_stats = Self::replay_wal(&wal_path, &memtable)?;

        let memtable_size = memtable.len();

        info!(
            "Collection recovery complete: {} segments verified, {} WAL entries replayed, {} vectors in memtable",
            segments_verified.len(),
            wal_stats.entries_replayed,
            memtable_size
        );

        Ok(RecoveryReport {
            segments_verified,
            wal_stats,
            memtable_size,
        })
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Deserialize an insert payload and apply it to the memtable.
    ///
    /// Payload format: `(VectorId, VectorData, Option<Metadata>)` serialized
    /// with bincode. If the id already exists, treat as an update (delete +
    /// re-insert).
    fn apply_insert(
        payload: &[u8],
        memtable: &InMemoryVectorStore,
    ) -> StorageResult<()> {
        let (id, data, metadata): (VectorId, VectorData, Option<Metadata>) =
            bincode::deserialize(payload)?;

        let record = VectorRecord::new(id, Some(data), metadata);

        match memtable.insert(record.clone()) {
            Ok(()) => Ok(()),
            Err(vf_core::store::StoreError::AlreadyExists(_)) => {
                // Treat as update: delete old record, then re-insert.
                let _ = memtable.delete(id);
                memtable
                    .insert(record)
                    .map_err(|e| StorageError::Serialization(format!("re-insert failed: {}", e)))
            }
            Err(e) => Err(StorageError::Serialization(format!(
                "insert failed: {}",
                e
            ))),
        }
    }

    /// Deserialize an update payload and apply it to the memtable.
    ///
    /// Same payload format as insert. If the vector does not yet exist in
    /// the memtable, insert it instead.
    fn apply_update(
        payload: &[u8],
        memtable: &InMemoryVectorStore,
    ) -> StorageResult<()> {
        let (id, data, metadata): (VectorId, Option<VectorData>, Option<Metadata>) =
            bincode::deserialize(payload)?;

        match memtable.update(id, data.clone(), metadata.clone()) {
            Ok(()) => Ok(()),
            Err(vf_core::store::StoreError::NotFound(_)) => {
                // Vector not present yet - insert it if data is available.
                let Some(d) = data else {
                    return Ok(());
                };
                let record = VectorRecord::new(id, Some(d), metadata);
                memtable
                    .insert(record)
                    .map_err(|e| StorageError::Serialization(format!("insert-on-update failed: {}", e)))
            }
            Err(e) => Err(StorageError::Serialization(format!(
                "update failed: {}",
                e
            ))),
        }
    }

    /// Deserialize a delete payload and apply it to the memtable.
    ///
    /// Payload is a bincode-serialized `VectorId`. Not-found is silently
    /// ignored (the vector may have already been deleted or never inserted
    /// into this memtable).
    fn apply_delete(
        payload: &[u8],
        memtable: &InMemoryVectorStore,
    ) -> StorageResult<()> {
        let id: VectorId = bincode::deserialize(payload)?;

        match memtable.delete(id) {
            Ok(_) => Ok(()),
            Err(vf_core::store::StoreError::NotFound(_)) => {
                // Silently ignore - vector was not present.
                Ok(())
            }
            Err(e) => Err(StorageError::Serialization(format!(
                "delete failed: {}",
                e
            ))),
        }
    }
}

// ── Snapshot-based recovery planning ───────────────────────────────────────

/// Describes the recovery strategy for a collection's indexes.
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Clean shutdown: load base snapshots directly, no WAL replay needed.
    CleanShutdown,
    /// Base exists: load base + replay delta + replay WAL tail.
    IncrementalReplay {
        hnsw_base_lsn: u64,
        graph_base_lsn: u64,
    },
    /// No base snapshots: full rebuild from vectors (legacy path).
    FullRebuild,
}

/// Recovery plan for a single collection.
#[derive(Debug)]
pub struct CollectionRecoveryPlan {
    pub collection_name: String,
    pub collection_dir: PathBuf,
    pub strategy: RecoveryStrategy,
    pub has_shutdown_clean: bool,
    pub has_hnsw_base: bool,
    pub has_graph_base: bool,
    pub has_hnsw_delta: bool,
    pub has_graph_delta: bool,
}

/// Plan recovery for a collection by inspecting available files.
///
/// This is a file-existence-based planner only. It does NOT validate file
/// contents (magic bytes, CRC, etc.) because vf-storage cannot depend on
/// vf-index or vf-graph. The caller (AppState in vf-server) is responsible
/// for validating base snapshots and falling back to FullRebuild if they
/// turn out to be corrupt.
pub fn plan_recovery(collection_name: &str, collection_dir: &Path) -> CollectionRecoveryPlan {
    let has_shutdown_clean = collection_dir.join("shutdown_clean").exists();
    let has_hnsw_base = collection_dir.join("hnsw.base").exists();
    let has_graph_base = collection_dir.join("graph.base").exists();
    let has_hnsw_delta = collection_dir.join("hnsw.delta").exists();
    let has_graph_delta = collection_dir.join("graph.delta").exists();

    let strategy = if has_shutdown_clean && has_hnsw_base {
        RecoveryStrategy::CleanShutdown
    } else if has_hnsw_base {
        RecoveryStrategy::IncrementalReplay {
            hnsw_base_lsn: 0, // caller will read actual LSN from files
            graph_base_lsn: 0,
        }
    } else {
        RecoveryStrategy::FullRebuild
    };

    CollectionRecoveryPlan {
        collection_name: collection_name.to_string(),
        collection_dir: collection_dir.to_path_buf(),
        strategy,
        has_shutdown_clean,
        has_hnsw_base,
        has_graph_base,
        has_hnsw_delta,
        has_graph_delta,
    }
}

/// Recovery statistics for snapshot-based recovery logging.
#[derive(Debug, Default)]
pub struct SnapshotRecoveryStats {
    pub strategy_used: String,
    pub hnsw_nodes_loaded: usize,
    pub graph_nodes_loaded: usize,
    pub delta_entries_replayed: usize,
    pub recovery_time_ms: u64,
}
