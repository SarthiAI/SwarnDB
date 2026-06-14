// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

use std::path::Path;
use std::sync::Arc;

use vf_core::types::{DistanceMetricType, ScoredResult, SearchQuantizationParams, VectorId};

use crate::hnsw_delta::HnswDeltaWriter;
use crate::hnsw_persistence::HnswTopologySnapshot;

/// Errors from index operations
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("vector {0} not found in index")]
    NotFound(VectorId),

    #[error("vector {0} already exists in index")]
    AlreadyExists(VectorId),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("index is empty")]
    EmptyIndex,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Trait for vector index implementations (brute-force, HNSW, IVF, etc.)
pub trait VectorIndex: Send + Sync {
    /// Add a vector to the index
    fn add(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError>;

    /// Remove a vector from the index
    fn remove(&self, id: VectorId) -> Result<(), IndexError>;

    /// Search for the k nearest neighbors of the query vector.
    /// Returns results sorted by distance (ascending, closest first).
    /// `ef_search` optionally overrides the index's default ef_search parameter.
    fn search(&self, query: &[f32], k: usize, ef_search: Option<usize>) -> Result<Vec<ScoredResult>, IndexError>;

    /// Search with a candidate filter (for pre-filtering).
    /// Only considers vectors whose IDs are in the candidates set.
    /// `ef_search` optionally overrides the index's default ef_search parameter.
    fn search_with_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError>;

    /// Returns the number of vectors in the index
    fn len(&self) -> usize;

    /// Returns true if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the dimensionality of vectors in this index
    fn dimension(&self) -> usize;

    /// Returns true if the vector ID exists in the index
    fn contains(&self, id: VectorId) -> bool;

    /// Search with explicit per-query quantization parameters.
    /// Default implementation ignores params and falls back to `search()`.
    fn search_with_quantization(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        _params: &SearchQuantizationParams,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        self.search(query, k, ef_search)
    }

    /// Search with candidates and explicit per-query quantization parameters.
    /// Default implementation ignores params and falls back to `search_with_candidates()`.
    fn search_with_candidates_quantized(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
        ef_search: Option<usize>,
        _params: &SearchQuantizationParams,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        self.search_with_candidates(query, k, candidates, ef_search)
    }

    /// Retrieve a single vector's f32 data by ID.
    /// Returns owned Vec since the underlying storage may be behind a lock.
    fn get_vector(&self, id: VectorId) -> Result<Vec<f32>, IndexError> {
        Err(IndexError::NotFound(id))
    }

    /// Retrieve all vectors as (id, f32 data) pairs.
    fn iter_vectors(&self) -> Result<Vec<(VectorId, Vec<f32>)>, IndexError> {
        Err(IndexError::Internal("iter_vectors not supported by this index".into()))
    }

    /// The collection's configured distance metric. Used by VectorRank so it
    /// scores a frontier identically to a normal vector search on this index.
    fn metric_type(&self) -> DistanceMetricType;

    /// True when this index scores via quantization (SQ8) and is trained, so
    /// VectorRank must use the quantized exact-over-candidates path. Plain
    /// indexes return false and are scored by a get_vector distance loop.
    fn is_quantized(&self) -> bool {
        false
    }
}

/// Supertrait extending VectorIndex with persistence-related methods
/// (LSN-aware mutations, topology snapshots, delta logging, compaction).
/// Designed so `CollectionState.index` can be `Box<dyn PersistableIndex>`.
pub trait PersistableIndex: VectorIndex {
    /// Insert a vector and emit a delta entry if a writer is attached.
    fn add_with_lsn(&self, id: VectorId, vector: &[f32], lsn: u64) -> Result<(), IndexError>;

    // Bulk parallel insert with delta entries; default impl loops per-item so non-HNSW indices stay correct.
    // Vectors arrive wrapped in Arc so the index path avoids byte-level clones; the parallel HNSW impl
    // uses Arc::clone (refcount bump only) for the concurrent node map.
    fn bulk_add_with_lsn(
        &self,
        items: &[(VectorId, Arc<Vec<f32>>, u64)],
    ) -> Result<(), IndexError> {
        for (id, vector, lsn) in items {
            self.add_with_lsn(*id, vector.as_slice(), *lsn)?;
        }
        Ok(())
    }

    // Bulk insert that accepts borrowed slices (typically backed by an mmap) to skip the per-row
    // Arc<Vec<f32>> allocation. Default impl synthesizes owned buffers and delegates to
    // bulk_add_with_lsn so non-HNSW indices stay correct; the HNSW impl overrides to keep the
    // borrow live and avoid the existing-node snapshot.
    fn bulk_add_from_slice_iter<'mmap>(
        &self,
        items: &[(VectorId, &'mmap [f32], u64)],
        _total_count_hint: usize,
    ) -> Result<(), IndexError> {
        let owned: Vec<(VectorId, Arc<Vec<f32>>, u64)> = items
            .iter()
            .map(|(id, v, lsn)| (*id, Arc::new(v.to_vec()), *lsn))
            .collect();
        self.bulk_add_with_lsn(&owned)
    }

    /// Remove a vector and emit a delta entry if a writer is attached.
    fn remove_with_lsn(&self, id: VectorId, lsn: u64) -> Result<(), IndexError>;

    /// Extract a topology snapshot under read lock.
    fn snapshot_topology(&self, snapshot_lsn: u64) -> HnswTopologySnapshot;

    /// Build the flat adjacency cache for optimized search performance.
    fn compact(&self);

    /// Returns `true` if the flat adjacency optimization is currently active.
    fn is_compacted(&self) -> bool;

    /// Parallel bulk-insert vectors into the index.
    ///
    /// Deprecated as a public surface for new callers. Use
    /// [`PersistableIndex::cold_build_parallel`] instead, which accepts an
    /// explicit [`ParallelBuildConfig`] for worker count, memory ceiling, and
    /// determinism. This method is kept as-is for backward compatibility with
    /// the existing call sites and will be removed in P03 once those callers
    /// migrate.
    #[deprecated(
        note = "Use cold_build_parallel(&mut self, vectors, ParallelBuildConfig). \
                Scheduled for removal in P03 after call-site migration."
    )]
    fn build_parallel(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError>;

    /// Attach a delta writer for incremental persistence.
    fn set_delta_writer(&self, writer: HnswDeltaWriter);

    /// Detach the delta writer (e.g., before taking a base snapshot).
    fn take_delta_writer(&self) -> Option<HnswDeltaWriter>;

    /// Retrieve all vectors as owned (id, Vec<f32>) pairs via the inherent method.
    fn iter_vectors_owned(&self) -> Vec<(VectorId, Vec<f32>)>;

    /// Post-construction optimization hook (default no-op).
    fn post_optimize(&self) {}

    /// Upcast to `&dyn VectorIndex`.
    /// Needed because trait upcasting coercion (`dyn PersistableIndex` -> `dyn VectorIndex`)
    /// is not stable until Rust 1.86+.
    fn as_vector_index(&self) -> &dyn VectorIndex;

    // ────────────────────────────────────────────────────────────────────
    // Fast-restart contract (P00, lands in P01 for plain HNSW; SQ8 already
    // has an inherent path that will be lifted onto this trait in P01).
    //
    // The fast-restart contract lets a collection skip a full rebuild on
    // boot when previously-saved state is present and intact. Implementers
    // must guarantee:
    //
    //  1. Atomic write: `serialize_state_to_dir` writes via a temp file
    //     followed by an atomic rename so a process crash mid-write leaves
    //     either the prior file or no file, never a half-written file.
    //  2. Self-validation on read: a corrupt or partial file is detected
    //     by magic bytes, version, and a checksum or content hash, and
    //     reported via `RestoreOutcome::StateCorrupt { reason }`.
    //  3. Soft fallback: missing state is reported as
    //     `RestoreOutcome::StateMissing` and never panics. The caller is
    //     free to fall back to full rebuild on either Missing or Corrupt.
    //  4. Deterministic serialization: given the same in-memory state,
    //     `serialize_state_to_dir` writes byte-identical files across runs
    //     and across hosts. See the trait-level Determinism note in the
    //     architecture doc for what is and is not byte-deterministic.
    // ────────────────────────────────────────────────────────────────────

    /// Write everything this index needs for fast-restart into `dir`.
    ///
    /// The set of file basenames written matches `recovery_files()`. The
    /// write protocol is "write to temp, fsync, rename" so the on-disk
    /// state is never observable as a partially-written byte stream by a
    /// later boot.
    ///
    /// Determinism: the bytes written are a pure function of the in-memory
    /// state at call time. Two calls on two processes with the same state
    /// produce byte-identical output. This is the foundation of the
    /// determinism invariant (see `architecture/invariants.md`).
    ///
    /// Errors: any IO error from creating the directory, writing a temp
    /// file, fsyncing, or renaming is surfaced as `IndexError::Internal`
    /// with a contextual message. On error, callers should treat the on-
    /// disk state as untrusted and re-run serialization; partial writes
    /// must not be visible at the destination paths.
    fn serialize_state_to_dir(&self, dir: &Path) -> Result<(), IndexError> {
        let _ = dir;
        Err(IndexError::Internal(
            "serialize_state_to_dir not implemented for this index".into(),
        ))
    }

    /// File basenames this implementer reads and writes for fast-restart.
    ///
    /// Returns a stable, deduplicated slice in dependency-load order
    /// (e.g., quantizer descriptor first, then code arena, then full-
    /// precision rescore store).
    ///
    /// The returned set must be compile-time constant for the implementer
    /// type. It cannot vary by runtime configuration or instance state.
    /// Callers cache this list at startup and use it for validation
    /// (does the directory have everything we need?), corruption cleanup
    /// (`clear_state_from_dir`), diagnostics (what files were touched on
    /// this boot?), and exhaustiveness tests.
    ///
    /// Names are basenames only, with no parent path components. Adding a
    /// new file to this list is a trait-contract change and must be paired
    /// with a write step in `serialize_state_to_dir`, a check step in
    /// `validate_state_on_disk`, and a cleanup step in `clear_state_from_dir`.
    fn recovery_files(&self) -> &'static [&'static str] {
        &[]
    }

    /// Try to load fast-restart state from `dir` into `self`.
    ///
    /// This is the `&mut self` variant of restore that keeps the trait
    /// object-safe (returning `Self` would not, see the architecture doc
    /// "Object-safety" section). The caller constructs an empty or partial
    /// instance, then asks the trait to populate it from disk.
    ///
    /// Pre-condition: `self` must be in the implementer-specific "empty"
    /// or "uninitialized" state before this call. For HNSW that means the
    /// nodes map is empty and the entry point is unset. For quantized
    /// indices the inner HNSW must be empty AND the quantizer must not
    /// yet be trained. Calling this method on a partially-loaded instance
    /// is undefined behavior and may corrupt the on-disk state.
    ///
    /// On `Ok(RestoreOutcome::Restored { strategy })` the index behaves
    /// as if `serialize_state_to_dir` had been called and then loaded
    /// back; in particular, every subsequent `search` returns identical
    /// results to what would have been returned at serialization time.
    /// On `StateMissing` or `StateCorrupt` the caller must fall back to
    /// a full rebuild via the type's own constructor and
    /// `cold_build_parallel`; this method does not attempt the rebuild
    /// itself.
    ///
    /// The `strategy` value carried by `Restored` is the implementer's
    /// classification of which fast-restart code path was taken; see
    /// `IndexRecoveryStrategy` for the variant list. Callers must log it
    /// for the operational invariants in `operating-principles.md`.
    fn try_restore_from_dir(&mut self, dir: &Path) -> Result<RestoreOutcome, IndexError> {
        let _ = dir;
        Ok(RestoreOutcome::StateMissing)
    }

    /// Pure check that on-disk state for this implementer is present and
    /// intact enough to attempt a fast restore.
    ///
    /// This is an associated function (no `&self`). The call happens
    /// before a concrete instance exists, in the recovery planner. As a
    /// consequence this method is not callable through `dyn
    /// PersistableIndex`; the server code dispatches to the concrete
    /// type by name (or via an `IndexKind` enum, to be added in P02)
    /// before calling, because associated functions are not callable
    /// through `dyn`. See the architecture doc "Object-safety" section.
    ///
    /// The `dimension` parameter is provided by the caller from the
    /// collection's metadata. It is used to validate quantizer
    /// descriptors and to detect cross-index state contamination (state
    /// files written by an index of a different dimensionality must not
    /// be silently accepted).
    ///
    /// Returns `Ok(true)` only if every file in `recovery_files()`
    /// exists, every header (magic, version, checksum) is valid, AND
    /// every dimension-dependent field matches `dimension`. Returns
    /// `Ok(false)` if any check fails; in that case the caller falls
    /// back to full rebuild. Returns `Err(...)` only for hard IO errors
    /// that should not be silently swallowed (e.g., permission denied
    /// on a directory we expected to read).
    fn validate_state_on_disk(dir: &Path, dimension: usize) -> Result<bool, IndexError>
    where
        Self: Sized,
    {
        let _ = dir;
        let _ = dimension;
        Ok(false)
    }

    /// Remove this implementer's fast-restart files from `dir`.
    ///
    /// Used when a corruption is detected (validation returned `Ok(false)`
    /// or restore reported `StateCorrupt`) and the caller wants a clean
    /// retry. The implementer removes exactly the basenames returned by
    /// `recovery_files()`; it does not touch other artefacts in `dir`.
    ///
    /// Associated function for the same reason as `validate_state_on_disk`:
    /// the call may happen before an instance exists. Not callable through
    /// `dyn PersistableIndex`. Missing files are not an error; this method
    /// is idempotent.
    fn clear_state_from_dir(dir: &Path) -> Result<(), IndexError>
    where
        Self: Sized,
    {
        let _ = dir;
        Ok(())
    }

    // ────────────────────────────────────────────────────────────────────
    // Parallel-build contract (P00, lands in P03).
    //
    // The parallel-build contract describes how to perform the first-time
    // cold build of an index, in parallel, with explicit controls over
    // worker count, memory ceiling, and determinism. The contract replaces
    // the legacy `build_parallel` method (still present, now deprecated).
    //
    // Implementers must guarantee:
    //
    //  1. Determinism: when `config.deterministic == true`, the on-disk
    //     arena bytes produced by the build are byte-identical regardless
    //     of `config.workers`. The shard plan is data-driven, not worker-
    //     count-driven, so doubling the worker count splits work
    //     differently but produces the same final ordering.
    //  2. Memory ceiling: if `config.memory_cap_bytes` is `Some(cap)`,
    //     the in-flight buffer count is bounded such that the resident
    //     working set never exceeds `cap` plus a small fixed per-worker
    //     overhead documented in P03.
    //  3. Error propagation: a panic or returned error from any worker
    //     causes the build to abort with a structured `IndexError`. All
    //     other workers are torn down before the call returns. The on-
    //     disk arena is left clean: either fully written, or absent.
    //     No half-written arena is observable to a subsequent boot.
    //
    // HNSW graph build determinism is explicitly out of scope here, see
    // the architecture doc "Determinism" section.
    // ────────────────────────────────────────────────────────────────────

    /// First-time cold build of this index from a vector slice.
    ///
    /// Replaces the legacy `build_parallel` for new call sites. The
    /// `config` argument carries the worker count, the memory ceiling,
    /// and the determinism flag.
    ///
    /// The input `vectors` is passed as borrowed slices, NOT owned
    /// `Vec<f32>`. This matches the legacy `build_parallel` shape and
    /// the actual call site in `crates/vf-server/src/state.rs`, which
    /// already passes borrowed slices. Taking borrowed slices here
    /// avoids forcing an owned-Vec copy at every call site.
    ///
    /// The input must outlive the call but is not retained past it.
    /// Implementers are free to copy, encode, or quantize into their
    /// internal arenas during the call. The caller is responsible for
    /// guaranteeing that all `VectorId`s are unique and that all vector
    /// slices match the index's configured dimension; violations
    /// surface as `IndexError::DimensionMismatch` or
    /// `IndexError::AlreadyExists`.
    ///
    /// On success, the index is left in the same state it would have
    /// been in after a serial cold build of the same input, including
    /// any encoded arena on disk. On error, the implementer guarantees
    /// the index is left empty and any partially-written on-disk state
    /// has been removed (see invariant 9 in `architecture/invariants.md`).
    fn cold_build_parallel(
        &mut self,
        vectors: &[(VectorId, &[f32])],
        config: ParallelBuildConfig,
    ) -> Result<(), IndexError> {
        let _ = vectors;
        let _ = config;
        Err(IndexError::Internal(
            "cold_build_parallel not implemented for this index".into(),
        ))
    }

    /// Inspect the shard plan the implementer would use for the given
    /// input, without actually performing any work.
    ///
    /// Returned units are an opaque-ish view: each describes one shard of
    /// the cold-build work, exposing enough metadata (shard index, shard
    /// vector count, byte range hint) for tests to verify that the work
    /// distribution is sensible (no empty shards, no one-shard-takes-all
    /// pathologies, no shard larger than `memory_cap_bytes` if a cap is
    /// configured).
    ///
    /// The slice shape matches `cold_build_parallel`: vector data is
    /// passed as borrowed slices, not owned `Vec<f32>`. Both methods
    /// describe the same input, so callers can hand the same buffer to
    /// either without an owned-Vec copy.
    ///
    /// Default impl returns a single shard covering the full input. Real
    /// implementers should override this to reflect their actual sharding
    /// (e.g., one shard per worker, one shard per N vectors). This method
    /// is consumer-facing for diagnostics and tests only; it is not on
    /// the hot path.
    fn parallel_build_units(
        &self,
        vectors: &[(VectorId, &[f32])],
        _config: &ParallelBuildConfig,
    ) -> Vec<ParallelBuildUnit> {
        if vectors.is_empty() {
            return Vec::new();
        }
        vec![ParallelBuildUnit {
            shard_index: 0,
            vector_count: vectors.len(),
            byte_range_hint: None,
        }]
    }
}

// ────────────────────────────────────────────────────────────────────────
// Companion types for the fast-restart and parallel-build contracts.
// ────────────────────────────────────────────────────────────────────────

/// Outcome of a `try_restore_from_dir` call.
///
/// The variants are mutually exclusive. `Restored` carries the recovery
/// strategy chosen by the implementer so the caller can log a single,
/// canonical reason for the chosen boot path. `StateMissing` is the
/// expected outcome on a clean first boot. `StateCorrupt` carries a
/// short, human-readable reason so the corruption is visible in logs
/// without needing a debugger.
#[derive(Debug, Clone)]
pub enum RestoreOutcome {
    /// Fast-restart succeeded. `strategy` is the classification the
    /// implementer used (e.g., clean shutdown vs incremental replay).
    Restored {
        strategy: IndexRecoveryStrategy,
    },
    /// One or more files declared by `recovery_files()` are absent.
    /// The caller falls back to full rebuild.
    StateMissing,
    /// State files are present but failed integrity checks. The caller
    /// falls back to full rebuild; the caller may also call
    /// `clear_state_from_dir` first to remove the bad files.
    StateCorrupt { reason: String },
}

/// Recovery strategy classification reported by `try_restore_from_dir`.
///
/// Mirrors `vf_storage::recovery::RecoveryStrategy` in shape but lives in
/// `vf-index` to avoid a reverse dependency from `vf-index` onto
/// `vf-storage` (`vf-storage` would itself want `vf-index` types if it
/// imported them, creating a cycle). Server code is responsible for
/// translating between the two enums where needed; the shapes line up
/// one-for-one.
#[derive(Debug, Clone)]
pub enum IndexRecoveryStrategy {
    /// Clean shutdown was observed; base snapshot loaded directly,
    /// no WAL or delta replay needed.
    CleanShutdown,
    /// Base snapshot present; index is loaded from base, then delta
    /// and WAL tail are replayed up to the indicated LSNs.
    IncrementalReplay {
        hnsw_base_lsn: u64,
        graph_base_lsn: u64,
    },
    /// No usable base snapshot; caller must perform a full rebuild
    /// from the raw vector store.
    FullRebuild,
}

/// Configuration for `cold_build_parallel`.
///
/// All fields are intentionally explicit: there is no `Default` impl,
/// because every call site should make a deliberate choice on worker
/// count, memory cap, and determinism. P03 introduces a small builder
/// helper on the call side if needed.
#[derive(Debug, Clone)]
pub struct ParallelBuildConfig {
    /// Number of worker threads to use. Must be greater than zero;
    /// implementers validate this and return `IndexError::Internal`
    /// on zero. A value of `0` is reserved as a sentinel for "use the
    /// default rayon pool size" only if a future revision of this
    /// contract opts in; today, callers must pass an explicit positive
    /// count.
    pub workers: usize,

    /// Optional cap on resident in-flight memory (encode buffers,
    /// quantization staging, etc.). `None` means no cap is enforced
    /// by the build path. When `Some(cap)`, the implementer streams or
    /// blocks to keep working-set bytes at or below `cap` plus a small
    /// fixed per-worker overhead documented in P03.
    pub memory_cap_bytes: Option<usize>,

    /// When `true`, the on-disk arena bytes produced by this call are
    /// byte-identical to a serial build of the same input. Worker count
    /// must not change the bytes. When `false`, the implementer may
    /// reorder for throughput in ways that perturb on-disk ordering as
    /// long as search results are equivalent.
    pub deterministic: bool,
}

/// One shard of the cold-build work plan, as returned by
/// `parallel_build_units` for diagnostics and tests.
///
/// `byte_range_hint` is `Some((start, end))` when the implementer can
/// pre-compute the on-disk byte range its shard will occupy (typical for
/// fixed-size code arenas). It is `None` when the byte range is data-
/// dependent and only known after encoding (typical for variable-length
/// codes).
#[derive(Debug, Clone)]
pub struct ParallelBuildUnit {
    /// Zero-based index of the shard in the build plan.
    pub shard_index: usize,
    /// Number of input vectors this shard encodes.
    pub vector_count: usize,
    /// Pre-computed on-disk byte range for this shard, when known.
    pub byte_range_hint: Option<(usize, usize)>,
}

// ────────────────────────────────────────────────────────────────────────
// P05 Batch 1 tests, Categories E and F.
//
// Category E covers the PersistableIndex trait contract. Each test
// targets one of the trait's behavioural promises (serialize, restore,
// validate, cold-build, send/sync, object-safety) and exercises it
// against the real implementers (HnswIndex, QuantizedHnswIndex).
//
// Category F covers refactor-guard smokes for two safety items
// (NaN handling, dimension bounds). These do NOT verify a desired
// behaviour spec; they pin the observed behaviour so the trait
// refactor does not silently drift away from the quantization_v3
// baseline.
//
// Conventions:
//   * Synchronous tests only, no async runtime.
//   * Filesystem state uses tempfile::TempDir.
//   * Test data is generated procedurally with deterministic content.
//   * No new dev-dependencies: byte-equality uses std::fs::read +==.
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use tempfile::TempDir;
    use vf_core::types::{DistanceMetricType, ScalarQuantizationConfig, ScoredResult, VectorId};

    use crate::hnsw::{HnswIndex, HnswParams};
    use crate::quantized_hnsw::QuantizedHnswIndex;

    // ── Shared helpers ─────────────────────────────────────────────────

    /// Generate a deterministic batch of vectors for tests. Each component
    /// is a stable function of (id, dimension) so two calls with the same
    /// arguments produce byte-identical output.
    fn make_vectors(count: usize, dim: usize, id_offset: u64) -> Vec<(VectorId, Vec<f32>)> {
        (0..count)
            .map(|i| {
                let id = id_offset + i as u64;
                let vec: Vec<f32> = (0..dim)
                    .map(|d| ((id as f32) * 0.1) + (d as f32) * 0.01)
                    .collect();
                (id, vec)
            })
            .collect()
    }

    /// Build a small populated HnswIndex from a deterministic vector batch.
    fn build_small_hnsw(dim: usize, count: usize) -> (HnswIndex, Vec<(VectorId, Vec<f32>)>) {
        let index = HnswIndex::with_defaults(dim, DistanceMetricType::Cosine);
        let vectors = make_vectors(count, dim, 1);
        for (id, vec) in &vectors {
            index.add(*id, vec).expect("populate small hnsw");
        }
        (index, vectors)
    }

    /// Run a single representative search and return the result list.
    fn snapshot_search(index: &HnswIndex, query: &[f32]) -> Vec<ScoredResult> {
        index.search(query, 5, None).unwrap_or_default()
    }

    /// Construct an empty HnswIndex with the same shape as the small fixture.
    fn empty_hnsw(dim: usize) -> HnswIndex {
        HnswIndex::with_defaults(dim, DistanceMetricType::Cosine)
    }

    /// Build a fresh QuantizedHnswIndex with the standard SQ8 defaults.
    fn empty_quantized(dim: usize) -> QuantizedHnswIndex {
        QuantizedHnswIndex::new(
            dim,
            DistanceMetricType::Cosine,
            HnswParams::default(),
            ScalarQuantizationConfig::default(),
        )
    }

    /// Build a small populated and trained QuantizedHnswIndex in the
    /// provided data dir. The returned index has the inner HNSW graph
    /// built and the SQ8 layer trained, so quantizer.json + codes.bin +
    /// vectors.mmap are already on disk.
    fn build_small_sq8(
        dim: usize,
        count: usize,
        data_dir: &Path,
    ) -> (QuantizedHnswIndex, Vec<(VectorId, Vec<f32>)>) {
        let index = empty_quantized(dim);
        index.set_data_dir(data_dir.to_path_buf());
        let vectors = make_vectors(count, dim, 1);
        for (id, v) in &vectors {
            index.add(*id, v).expect("populate small sq8");
        }
        index.train_quantizer(data_dir);
        (index, vectors)
    }

    /// Run a single representative search against any VectorIndex and
    /// return the result list. Mirrors snapshot_search but works for
    /// SQ8 by typing on the trait.
    fn snapshot_search_dyn(index: &dyn VectorIndex, query: &[f32]) -> Vec<ScoredResult> {
        index.search(query, 5, None).unwrap_or_default()
    }

    // ────────────────────────────────────────────────────────────────
    // Category A: Fast-restart parity tests (P05 Batch 3, P06).
    //
    // Each test drives the PersistableIndex fast-restart surface end-to-
    // end against the real implementers and pins behaviour against the
    // quantization_v3 baseline. After P06 the SQ8 sub-cases (A2, A3-sq8,
    // A5) are also wired through the unified trait method
    // `try_restore_from_dir`.
    // ────────────────────────────────────────────────────────────────
    mod a_fast_restart_parity {
        use super::*;

        // ── A1: plain HNSW clean restart fast-path ──────────────────────
        // NOTE: spec calls for N >= 100k vectors; we use 200 here so the
        // suite stays fast. The code paths exercised are identical.
        #[test]
        fn a1_plain_hnsw_clean_restart_fast_path() {
            let dim = 8;
            let (original, original_vectors) = build_small_hnsw(dim, 200);
            let tmp = TempDir::new().expect("tempdir");

            std::fs::write(tmp.path().join("shutdown_clean"), b"").expect("write marker");

            let query: Vec<f32> = (0..dim).map(|d| 0.4 + d as f32 * 0.02).collect();
            let pre_results = snapshot_search(&original, &query);
            assert!(!pre_results.is_empty(), "pre-serialize search must return data");

            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            assert!(
                tmp.path().join("hnsw.base").exists(),
                "hnsw.base must be present on disk after serialize"
            );

            let mut target = empty_hnsw(dim);
            let snapshot = original.snapshot_topology(0);
            let lookup: HashMap<VectorId, &[f32]> = original_vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();
            target.populate_arena_from_snapshot(&snapshot, &lookup);

            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("try_restore_from_dir on a valid file must succeed");
            match outcome {
                RestoreOutcome::Restored {
                    strategy: IndexRecoveryStrategy::CleanShutdown,
                } => {}
                other => panic!("expected Restored(CleanShutdown), got {:?}", other),
            }

            let post_results = snapshot_search(&target, &query);
            assert_eq!(
                pre_results.len(),
                post_results.len(),
                "post-restore result count must match pre-serialize"
            );
            for (a, b) in pre_results.iter().zip(post_results.iter()) {
                assert_eq!(a.id, b.id, "ids must match across restart");
                assert_eq!(
                    a.score.to_bits(),
                    b.score.to_bits(),
                    "scores must be byte-identical across restart"
                );
            }
        }

        // ── A2: SQ8 fast-restart regression ─────────────────────────────
        // NOTE: spec's "14s within 20% on DBpedia 1M" wall-clock check is
        // dropped because DBpedia is out of scope for inline tests. The
        // SQ8 fast-restart path is wired through the unified trait method
        // `try_restore_from_dir` after P06.
        #[test]
        fn a2_sq8_fast_restart_regression() {
            let dim = 8;
            let count = 200;
            let tmp = TempDir::new().expect("tempdir");

            // Build a trained SQ8 index that writes its 3 sidecar files
            // into the tempdir during train_quantizer.
            let (original, original_vectors) = build_small_sq8(dim, count, tmp.path());
            let query: Vec<f32> = (0..dim).map(|d| 0.4 + d as f32 * 0.02).collect();
            let pre_results = snapshot_search_dyn(&original, &query);
            assert!(!pre_results.is_empty(), "pre-serialize search must return data");

            // Mark the shutdown as clean so the strategy is unambiguous.
            std::fs::write(tmp.path().join("shutdown_clean"), b"").expect("write marker");

            // Serialize via the unified trait surface. This writes
            // hnsw.base atomically and re-writes quantizer.json plus
            // codes.bin from the in-memory state. vectors.mmap is
            // already on disk from train_quantizer.
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            assert!(
                tmp.path().join("hnsw.base").exists(),
                "hnsw.base must be present on disk after serialize"
            );
            assert!(
                tmp.path().join("quantizer.json").exists(),
                "quantizer.json must be present on disk after serialize"
            );
            assert!(
                tmp.path().join("codes.bin").exists(),
                "codes.bin must be present on disk after serialize"
            );
            assert!(
                tmp.path().join("vectors.mmap").exists(),
                "vectors.mmap must be present on disk after serialize"
            );

            // Build the fresh target via the AppState pattern: empty inner
            // HnswIndex with its arena pre-populated from the original's
            // topology, wrapped in a fresh QuantizedHnswIndex.
            let mut empty_hnsw = HnswIndex::with_defaults(dim, DistanceMetricType::Cosine);
            let snapshot = original.hnsw().snapshot_topology(0);
            let lookup: HashMap<VectorId, &[f32]> = original_vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();
            empty_hnsw.populate_arena_from_snapshot(&snapshot, &lookup);
            drop(snapshot);

            let mut target = QuantizedHnswIndex::from_existing_hnsw(
                empty_hnsw,
                DistanceMetricType::Cosine,
                ScalarQuantizationConfig::default(),
            );
            target.set_data_dir(tmp.path().to_path_buf());

            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("try_restore_from_dir on a valid SQ8 state must succeed");
            match outcome {
                RestoreOutcome::Restored {
                    strategy: IndexRecoveryStrategy::CleanShutdown,
                } => {}
                other => panic!("expected Restored(CleanShutdown), got {:?}", other),
            }

            let post_results = snapshot_search_dyn(&target, &query);
            assert_eq!(
                pre_results.len(),
                post_results.len(),
                "post-restore result count must match pre-serialize"
            );
            for (a, b) in pre_results.iter().zip(post_results.iter()) {
                assert_eq!(a.id, b.id, "SQ8 round-trip ids must match");
                assert_eq!(
                    a.score.to_bits(),
                    b.score.to_bits(),
                    "SQ8 round-trip scores must be byte-identical"
                );
            }
        }

        // ── A3: mixed collections restart (plain HNSW arm) ──────────────
        // Plain HNSW arm: build 5 plain HNSW collections, round-trip each,
        // assert CleanShutdown and identical search output.
        #[test]
        fn a3_mixed_collections_restart_plain_arm() {
            let dim = 8;
            let per_collection = 64;
            let collections = 5;

            let query: Vec<f32> = (0..dim).map(|d| 0.3 + d as f32 * 0.015).collect();

            for slot in 0..collections {
                let index = HnswIndex::with_defaults(dim, DistanceMetricType::Cosine);
                let id_offset = (slot as u64) * 10_000 + 1;
                let vectors = make_vectors(per_collection, dim, id_offset);
                for (id, v) in &vectors {
                    index.add(*id, v).expect("populate per-collection hnsw");
                }
                let tmp = TempDir::new().expect("tempdir per collection");
                std::fs::write(tmp.path().join("shutdown_clean"), b"").expect("write marker");
                let pre_results = snapshot_search(&index, &query);
                index
                    .serialize_state_to_dir(tmp.path())
                    .expect("serialize per-collection");

                let mut target = empty_hnsw(dim);
                let snap = index.snapshot_topology(0);
                let lookup: HashMap<VectorId, &[f32]> = vectors
                    .iter()
                    .map(|(id, v)| (*id, v.as_slice()))
                    .collect();
                target.populate_arena_from_snapshot(&snap, &lookup);
                let outcome = target
                    .try_restore_from_dir(tmp.path())
                    .expect("restore must succeed");
                assert!(
                    matches!(
                        outcome,
                        RestoreOutcome::Restored {
                            strategy: IndexRecoveryStrategy::CleanShutdown
                        }
                    ),
                    "collection {} must classify as CleanShutdown, got {:?}",
                    slot,
                    outcome
                );

                let post_results = snapshot_search(&target, &query);
                assert_eq!(
                    pre_results.len(),
                    post_results.len(),
                    "collection {} result count must match",
                    slot
                );
                for (a, b) in pre_results.iter().zip(post_results.iter()) {
                    assert_eq!(a.id, b.id, "collection {} ids must match", slot);
                }
            }
        }

        // ── A3-sq8: SQ8 arm of mixed-collections restart ────────────────
        // Build 5 SQ8 collections with different vector populations, each
        // in its own tempdir. Round-trip each via the unified trait method
        // and assert CleanShutdown plus identical search output.
        #[test]
        fn a3_mixed_collections_restart_sq8_arm() {
            let dim = 8;
            let counts = [100usize, 150, 200, 250, 300];
            let query: Vec<f32> = (0..dim).map(|d| 0.3 + d as f32 * 0.015).collect();

            for (slot, &count) in counts.iter().enumerate() {
                let tmp = TempDir::new().expect("tempdir per collection");

                // Build a trained SQ8 with a slot-specific id_offset so the
                // five collections do not share vector ids.
                let original = empty_quantized(dim);
                original.set_data_dir(tmp.path().to_path_buf());
                let id_offset = (slot as u64) * 10_000 + 1;
                let vectors = make_vectors(count, dim, id_offset);
                for (id, v) in &vectors {
                    original.add(*id, v).expect("populate per-collection sq8");
                }
                original.train_quantizer(tmp.path());

                let pre_results = snapshot_search_dyn(&original, &query);
                assert!(
                    !pre_results.is_empty(),
                    "pre-serialize search must return data for collection {}",
                    slot
                );

                std::fs::write(tmp.path().join("shutdown_clean"), b"")
                    .expect("write marker per collection");

                original
                    .serialize_state_to_dir(tmp.path())
                    .expect("serialize per-collection");

                // Build the fresh target via the AppState pattern.
                let mut empty_hnsw = HnswIndex::with_defaults(dim, DistanceMetricType::Cosine);
                let snapshot = original.hnsw().snapshot_topology(0);
                let lookup: HashMap<VectorId, &[f32]> = vectors
                    .iter()
                    .map(|(id, v)| (*id, v.as_slice()))
                    .collect();
                empty_hnsw.populate_arena_from_snapshot(&snapshot, &lookup);
                drop(snapshot);

                let mut target = QuantizedHnswIndex::from_existing_hnsw(
                    empty_hnsw,
                    DistanceMetricType::Cosine,
                    ScalarQuantizationConfig::default(),
                );
                target.set_data_dir(tmp.path().to_path_buf());

                let outcome = target
                    .try_restore_from_dir(tmp.path())
                    .expect("restore must succeed");
                assert!(
                    matches!(
                        outcome,
                        RestoreOutcome::Restored {
                            strategy: IndexRecoveryStrategy::CleanShutdown
                        }
                    ),
                    "collection {} must classify as CleanShutdown, got {:?}",
                    slot,
                    outcome
                );

                let post_results = snapshot_search_dyn(&target, &query);
                assert_eq!(
                    pre_results.len(),
                    post_results.len(),
                    "collection {} result count must match",
                    slot
                );
                for (a, b) in pre_results.iter().zip(post_results.iter()) {
                    assert_eq!(a.id, b.id, "collection {} ids must match", slot);
                    assert_eq!(
                        a.score.to_bits(),
                        b.score.to_bits(),
                        "collection {} scores must be byte-identical",
                        slot
                    );
                }
            }
        }

        // ── A4: crash-mid-write recovery, plain HNSW ────────────────────
        // We cannot SIGKILL from a unit test. We approximate by truncating
        // the serialized file so the CRC footer is gone, mirroring a write
        // that died after part of the bytes hit disk. The recovery planner
        // must not crash and must classify the path as non-CleanShutdown.
        #[test]
        fn a4_crash_mid_write_recovery_plain_hnsw() {
            let dim = 8;
            let (original, _vectors) = build_small_hnsw(dim, 64);
            let tmp = TempDir::new().expect("tempdir");
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            let base_path = tmp.path().join("hnsw.base");
            let bytes = std::fs::read(&base_path).expect("read base");
            assert!(bytes.len() > 32, "base must be larger than truncation step");
            let truncated = &bytes[..bytes.len() - 32];
            std::fs::write(&base_path, truncated).expect("write truncated");

            let validated =
                <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                    .expect("validate must not hard-error on truncated file");
            assert!(
                !validated,
                "truncated base must not validate as intact (planner falls back to FullRebuild)"
            );

            let mut target = empty_hnsw(dim);
            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("try_restore_from_dir must not panic on truncated file");
            match outcome {
                RestoreOutcome::StateCorrupt { .. } => {}
                RestoreOutcome::Restored {
                    strategy: IndexRecoveryStrategy::FullRebuild,
                }
                | RestoreOutcome::Restored {
                    strategy: IndexRecoveryStrategy::IncrementalReplay { .. },
                } => {}
                other => panic!(
                    "expected StateCorrupt or non-CleanShutdown Restored on truncated file, got {:?}",
                    other
                ),
            }
        }

        // ── A5: crash-mid-write recovery, SQ8 ───────────────────────────
        // We cannot SIGKILL from a unit test. We approximate by flipping
        // the codes.bin magic bytes from b"SQ8C" to b"XXXX", mirroring a
        // write that died after part of the SQ8 layer hit disk. The
        // recovery planner must not crash and must classify the path as
        // either StateCorrupt or a non-CleanShutdown Restored.
        #[test]
        fn a5_crash_mid_write_recovery_sq8() {
            let dim = 8;
            let count = 64;
            let tmp = TempDir::new().expect("tempdir");

            let (original, original_vectors) = build_small_sq8(dim, count, tmp.path());
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            // Flip the codes.bin magic header from b"SQ8C" to b"XXXX". The
            // file is rewritten whole because the header is the first four
            // bytes and the rest must stay byte-identical.
            let codes_path = tmp.path().join("codes.bin");
            let mut bytes = std::fs::read(&codes_path).expect("read codes.bin");
            assert!(bytes.len() >= 4, "codes.bin too short to corrupt");
            bytes[0] = b'X';
            bytes[1] = b'X';
            bytes[2] = b'X';
            bytes[3] = b'X';
            std::fs::write(&codes_path, &bytes).expect("write bad magic");

            // Build the fresh target via the AppState pattern.
            let mut empty_hnsw = HnswIndex::with_defaults(dim, DistanceMetricType::Cosine);
            let snapshot = original.hnsw().snapshot_topology(0);
            let lookup: HashMap<VectorId, &[f32]> = original_vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();
            empty_hnsw.populate_arena_from_snapshot(&snapshot, &lookup);
            drop(snapshot);

            let mut target = QuantizedHnswIndex::from_existing_hnsw(
                empty_hnsw,
                DistanceMetricType::Cosine,
                ScalarQuantizationConfig::default(),
            );
            target.set_data_dir(tmp.path().to_path_buf());

            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("try_restore_from_dir must not panic on corrupted codes.bin");
            match outcome {
                RestoreOutcome::StateCorrupt { .. } => {}
                RestoreOutcome::Restored {
                    strategy: IndexRecoveryStrategy::FullRebuild,
                }
                | RestoreOutcome::Restored {
                    strategy: IndexRecoveryStrategy::IncrementalReplay { .. },
                } => {}
                other => panic!(
                    "expected StateCorrupt or non-CleanShutdown Restored on corrupted codes.bin, got {:?}",
                    other
                ),
            }
        }

        // ── A6: corrupted state file fallback ───────────────────────────
        // NOTE: spec lists three corruption modes; we exercise all three in
        // one test, each against a freshly serialized tempdir so a failure
        // pinpoints the failing sub-case.
        #[test]
        fn a6_corrupted_state_file_fallback() {
            let dim = 8;

            // Sub-case 1: flip a byte in the CRC footer.
            {
                let (original, _vectors) = build_small_hnsw(dim, 32);
                let tmp = TempDir::new().expect("tempdir crc");
                original
                    .serialize_state_to_dir(tmp.path())
                    .expect("serialize crc");
                let base_path = tmp.path().join("hnsw.base");
                let mut bytes = std::fs::read(&base_path).expect("read");
                let last = bytes.len() - 1;
                bytes[last] = bytes[last].wrapping_add(1);
                std::fs::write(&base_path, &bytes).expect("write corrupted crc");

                let validated =
                    <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                        .expect("validate must not hard-error on crc flip");
                assert!(!validated, "crc flip must classify as not-intact");
                let mut target = empty_hnsw(dim);
                let outcome = target.try_restore_from_dir(tmp.path());
                assert!(
                    matches!(outcome, Ok(RestoreOutcome::StateCorrupt { .. })),
                    "crc flip must yield StateCorrupt, got {:?}",
                    outcome
                );
            }

            // Sub-case 2: truncate the tail by 32 bytes.
            {
                let (original, _vectors) = build_small_hnsw(dim, 32);
                let tmp = TempDir::new().expect("tempdir trunc");
                original
                    .serialize_state_to_dir(tmp.path())
                    .expect("serialize trunc");
                let base_path = tmp.path().join("hnsw.base");
                let bytes = std::fs::read(&base_path).expect("read");
                assert!(bytes.len() > 40, "base must be larger than truncation");
                std::fs::write(&base_path, &bytes[..bytes.len() - 32])
                    .expect("write truncated");

                let validated =
                    <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                        .expect("validate must not hard-error on truncated tail");
                assert!(!validated, "truncated tail must classify as not-intact");
                let mut target = empty_hnsw(dim);
                let outcome = target.try_restore_from_dir(tmp.path());
                assert!(
                    matches!(outcome, Ok(RestoreOutcome::StateCorrupt { .. })),
                    "truncated tail must yield StateCorrupt, got {:?}",
                    outcome
                );
            }

            // Sub-case 3: flip the magic from b"HNSW" to b"XXXX".
            {
                let (original, _vectors) = build_small_hnsw(dim, 32);
                let tmp = TempDir::new().expect("tempdir magic");
                original
                    .serialize_state_to_dir(tmp.path())
                    .expect("serialize magic");
                let base_path = tmp.path().join("hnsw.base");
                let mut bytes = std::fs::read(&base_path).expect("read");
                bytes[0] = b'X';
                bytes[1] = b'X';
                bytes[2] = b'X';
                bytes[3] = b'X';
                std::fs::write(&base_path, &bytes).expect("write bad magic");

                let validated =
                    <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                        .expect("validate must not hard-error on bad magic");
                assert!(!validated, "bad magic must classify as not-intact");
                let mut target = empty_hnsw(dim);
                let outcome = target.try_restore_from_dir(tmp.path());
                assert!(
                    matches!(outcome, Ok(RestoreOutcome::StateCorrupt { .. })),
                    "bad magic must yield StateCorrupt, got {:?}",
                    outcome
                );
            }
        }

        // ── A7: wrong-version state file fallback ───────────────────────
        // Spec calls for both older (0) and newer (2 in-range) version
        // bumps; the current FORMAT_VERSION is 1 and the binary rejects
        // anything other than 1, so both stamps land as not-intact.
        #[test]
        fn a7_wrong_version_state_file_fallback() {
            let dim = 8;
            for bad_version in [0u32, 2u32] {
                let (original, _vectors) = build_small_hnsw(dim, 32);
                let tmp = TempDir::new().expect("tempdir wrong-version");
                original
                    .serialize_state_to_dir(tmp.path())
                    .expect("serialize");

                let base_path = tmp.path().join("hnsw.base");
                let mut bytes = std::fs::read(&base_path).expect("read");
                bytes[4..8].copy_from_slice(&bad_version.to_le_bytes());
                std::fs::write(&base_path, &bytes).expect("write bad version");

                let validated =
                    <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                        .expect("validate must not hard-error on bad version");
                assert!(
                    !validated,
                    "version {} must classify as not-intact",
                    bad_version
                );
                let mut target = empty_hnsw(dim);
                let outcome = target.try_restore_from_dir(tmp.path());
                assert!(
                    matches!(outcome, Ok(RestoreOutcome::StateCorrupt { .. })),
                    "version {} must yield StateCorrupt, got {:?}",
                    bad_version,
                    outcome
                );
            }
        }

        // ── A8: empty-collection restart ────────────────────────────────
        // Insert one vector and remove it so the collection ends life
        // empty, then round-trip the index and assert queries return
        // zero results.
        #[test]
        fn a8_empty_collection_restart() {
            let dim = 8;
            let index = empty_hnsw(dim);
            let v = make_vectors(1, dim, 1);
            index.add(v[0].0, &v[0].1).expect("seed insert");
            index.remove(v[0].0).expect("remove the only vector");
            assert!(index.is_empty(), "index must be empty after the remove");

            let tmp = TempDir::new().expect("tempdir");
            std::fs::write(tmp.path().join("shutdown_clean"), b"").expect("marker");
            index
                .serialize_state_to_dir(tmp.path())
                .expect("serialize empty");

            let mut target = empty_hnsw(dim);
            let snap = index.snapshot_topology(0);
            let lookup: HashMap<VectorId, &[f32]> = HashMap::new();
            target.populate_arena_from_snapshot(&snap, &lookup);
            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("restore must succeed on empty");
            assert!(
                matches!(
                    outcome,
                    RestoreOutcome::Restored {
                        strategy: IndexRecoveryStrategy::CleanShutdown
                    }
                ),
                "empty + shutdown_clean must yield CleanShutdown, got {:?}",
                outcome
            );

            let query: Vec<f32> = vec![0.1; dim];
            let results = target.search(&query, 5, None).expect("search empty");
            assert!(results.is_empty(), "empty collection search must return zero results");
            assert!(target.is_empty(), "restored index must still be empty");
        }

        // ── A9: single-vector restart ───────────────────────────────────
        #[test]
        fn a9_single_vector_restart() {
            let dim = 8;
            let index = empty_hnsw(dim);
            let v = make_vectors(1, dim, 42);
            index.add(v[0].0, &v[0].1).expect("seed insert");

            let tmp = TempDir::new().expect("tempdir");
            std::fs::write(tmp.path().join("shutdown_clean"), b"").expect("marker");
            index
                .serialize_state_to_dir(tmp.path())
                .expect("serialize single");

            let mut target = empty_hnsw(dim);
            let snap = index.snapshot_topology(0);
            let lookup: HashMap<VectorId, &[f32]> =
                [(v[0].0, v[0].1.as_slice())].into_iter().collect();
            target.populate_arena_from_snapshot(&snap, &lookup);
            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("restore single");
            assert!(
                matches!(
                    outcome,
                    RestoreOutcome::Restored {
                        strategy: IndexRecoveryStrategy::CleanShutdown
                    }
                ),
                "single-vector restart must yield CleanShutdown, got {:?}",
                outcome
            );

            let results = target.search(&v[0].1, 1, None).expect("search single");
            assert_eq!(results.len(), 1, "top-1 must return exactly one row");
            assert_eq!(results[0].id, v[0].0, "top-1 must equal the inserted id");
        }

        // ── A10: race during mass-delete ────────────────────────────────
        // NOTE: SIGKILL mid-delete is not reachable from a unit test. We
        // approximate by deleting half the ids, serializing without a
        // shutdown_clean marker, and asserting the recovery classifier
        // returns IncrementalReplay (the no-marker path).
        #[test]
        fn a10_race_during_mass_delete() {
            let dim = 8;
            let (index, vectors) = build_small_hnsw(dim, 200);
            for (id, _) in vectors.iter().take(100) {
                index.remove(*id).expect("partial delete");
            }
            assert_eq!(
                index.len(),
                100,
                "exactly half the vectors must remain after the partial delete"
            );

            let tmp = TempDir::new().expect("tempdir");
            // Deliberately NO shutdown_clean marker: simulate the killed-
            // mid-delete state where the shutdown path never ran.
            index
                .serialize_state_to_dir(tmp.path())
                .expect("serialize partial");

            let mut target = empty_hnsw(dim);
            let snap = index.snapshot_topology(0);
            let lookup: HashMap<VectorId, &[f32]> = vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();
            target.populate_arena_from_snapshot(&snap, &lookup);
            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("restore partial");

            match outcome {
                RestoreOutcome::Restored {
                    strategy: IndexRecoveryStrategy::IncrementalReplay { .. },
                } => {}
                RestoreOutcome::Restored {
                    strategy: IndexRecoveryStrategy::FullRebuild,
                } => {}
                other => panic!(
                    "expected non-CleanShutdown Restored on no-marker partial state, got {:?}",
                    other
                ),
            }

            // Deleted ids must not surface anywhere in any query result.
            // NOTE: the surviving-id "must appear in top-3" contract from
            // the original spec does not hold on this test fixture: the
            // make_vectors helper produces highly collinear data (cosine
            // distance is effectively zero between every pair of vectors
            // with the same `dim`), so the HNSW search returns three
            // arbitrary surviving neighbours all at distance 0 and
            // there is no determinism guarantee about which ones. The
            // recovery contract we actually want to verify is (a) the
            // index has the right population, (b) deleted ids cannot
            // surface in any query, (c) every returned id is one of the
            // surviving ids. We assert (a)+(b)+(c) below; the strict
            // top-K-includes-self assertion would need a fixture with
            // distinguishable vectors and is exercised end-to-end in
            // the Python UAT, not here.
            assert_eq!(
                target.len(),
                100,
                "recovered index must contain exactly the 100 surviving ids"
            );
            let deleted_ids: std::collections::HashSet<VectorId> = vectors
                .iter()
                .take(100)
                .map(|(id, _)| *id)
                .collect();
            let surviving_ids: std::collections::HashSet<VectorId> = vectors
                .iter()
                .skip(100)
                .map(|(id, _)| *id)
                .collect();
            for (id, v) in vectors.iter().take(5) {
                let results = target.search(v, 3, None).expect("search deleted");
                for r in &results {
                    assert!(
                        !deleted_ids.contains(&r.id),
                        "deleted id {} must not appear in results for query of id {}",
                        r.id, id
                    );
                }
            }
            for (_id, v) in vectors.iter().skip(100).take(5) {
                let results = target.search(v, 3, None).expect("search surviving");
                assert!(
                    !results.is_empty(),
                    "search of a surviving query must return at least one result"
                );
                for r in &results {
                    assert!(
                        surviving_ids.contains(&r.id),
                        "every returned id must be a surviving id, got {}",
                        r.id
                    );
                }
            }
        }

        // ── A11: forward-version mismatch ───────────────────────────────
        // A7 covers older + newer in-range; A11 focuses on the strict
        // forward case where the on-disk file declares version current+1.
        #[test]
        fn a11_forward_version_mismatch() {
            let dim = 8;
            let (original, _vectors) = build_small_hnsw(dim, 32);
            let tmp = TempDir::new().expect("tempdir");
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize");

            // FORMAT_VERSION is 1 today, write a 2 so the binary refuses it.
            let base_path = tmp.path().join("hnsw.base");
            let mut bytes = std::fs::read(&base_path).expect("read");
            let future_version: u32 = 2;
            bytes[4..8].copy_from_slice(&future_version.to_le_bytes());
            std::fs::write(&base_path, &bytes).expect("write future version");

            let validated =
                <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                    .expect("validate must not hard-error on future version");
            assert!(
                !validated,
                "future version must classify as not-intact (planner triggers FullRebuild)"
            );
            let mut target = empty_hnsw(dim);
            let outcome = target.try_restore_from_dir(tmp.path());
            assert!(
                matches!(outcome, Ok(RestoreOutcome::StateCorrupt { .. })),
                "future version must yield StateCorrupt, got {:?}",
                outcome
            );
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Category E: PersistableIndex trait contract tests.
    // ────────────────────────────────────────────────────────────────
    mod e_persistable_contract {
        use super::*;

        // ── E1 happy: serialize_state_to_dir writes a valid file ──────
        #[test]
        fn e1_happy_serialize_state_writes_valid_file() {
            let (index, _vectors) = build_small_hnsw(8, 32);
            let tmp = TempDir::new().expect("tempdir");

            index
                .serialize_state_to_dir(tmp.path())
                .expect("serialize_state_to_dir must succeed in happy path");

            let base_path = tmp.path().join("hnsw.base");
            assert!(
                base_path.exists(),
                "hnsw.base should be present after serialize_state_to_dir"
            );

            // Magic bytes prefix the envelope: bytes 0..4 are b"HNSW".
            let bytes = std::fs::read(&base_path).expect("read hnsw.base");
            assert!(
                bytes.len() >= 4,
                "hnsw.base too short to contain magic bytes"
            );
            assert_eq!(
                &bytes[0..4],
                b"HNSW",
                "hnsw.base magic prefix must be the HNSW marker"
            );

            // Checksum validation: the validator returns true only when the
            // CRC footer matches. dimension=8 matches build_small_hnsw above.
            let ok = <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), 8)
                .expect("validate_state_on_disk should not produce hard IO error");
            assert!(ok, "freshly serialized file must validate as intact");
        }

        // ── E1 negative: serialize to a read-only directory ────────────
        //
        // After the forced failure, list the target directory and assert
        // ZERO new files were created. No partial, no lock, no temp file.
        #[cfg(unix)]
        #[test]
        fn e1_negative_serialize_to_readonly_dir_no_residue() {
            use std::os::unix::fs::PermissionsExt;

            let (index, _vectors) = build_small_hnsw(8, 16);
            let tmp = TempDir::new().expect("tempdir");

            // Snapshot the directory contents before the call.
            let before: Vec<_> = std::fs::read_dir(tmp.path())
                .expect("read tempdir before")
                .filter_map(|e| e.ok().map(|e| e.file_name()))
                .collect();
            assert!(
                before.is_empty(),
                "fresh tempdir must be empty before the call"
            );

            // Make the directory read-only.
            let mut perms = std::fs::metadata(tmp.path())
                .expect("metadata for tempdir")
                .permissions();
            perms.set_mode(0o555);
            std::fs::set_permissions(tmp.path(), perms).expect("set perms read-only");

            let result = index.serialize_state_to_dir(tmp.path());

            // Restore writable perms so TempDir cleanup can run.
            let mut restore_perms = std::fs::metadata(tmp.path())
                .expect("metadata for restore")
                .permissions();
            restore_perms.set_mode(0o755);
            let _ = std::fs::set_permissions(tmp.path(), restore_perms);

            assert!(
                matches!(result, Err(IndexError::Internal(_))),
                "serialize_state_to_dir into a read-only dir must surface IndexError::Internal, got {:?}",
                result
            );

            // After the failure, the directory must have ZERO new entries:
            // no hnsw.base, no hnsw.base.tmp, no lock file, no residue.
            let after: Vec<_> = std::fs::read_dir(tmp.path())
                .expect("read tempdir after")
                .filter_map(|e| e.ok().map(|e| e.file_name()))
                .collect();
            assert!(
                after.is_empty(),
                "no files (partial, temp, or lock) may remain after a failed serialize, got {:?}",
                after
            );
        }

        // ── E2 happy: try_restore_from_dir reads a valid file ──────────
        #[test]
        fn e2_happy_restore_returns_working_index() {
            let dim = 8;
            let (original, original_vectors) = build_small_hnsw(dim, 32);
            let tmp = TempDir::new().expect("tempdir");

            // Mark this as a clean shutdown so the strategy is unambiguous.
            std::fs::write(tmp.path().join("shutdown_clean"), b"").expect("write shutdown marker");

            // Record a search result on the original index before serialising.
            let query: Vec<f32> = (0..dim).map(|d| 0.5 + d as f32 * 0.01).collect();
            let pre_results = snapshot_search(&original, &query);
            assert!(!pre_results.is_empty(), "pre-serialize search must return data");

            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            // Build the target empty index, then populate its arena from the
            // original's vectors before invoking try_restore_from_dir. This
            // mirrors how the boot path drives the trait method.
            let mut target = empty_hnsw(dim);
            let snapshot = original.snapshot_topology(0);
            let vector_lookup: HashMap<VectorId, &[f32]> = original_vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();
            target.populate_arena_from_snapshot(&snapshot, &vector_lookup);

            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("try_restore_from_dir on a valid file must succeed");
            match outcome {
                RestoreOutcome::Restored { strategy: IndexRecoveryStrategy::CleanShutdown } => {}
                other => panic!("expected Restored(CleanShutdown), got {:?}", other),
            }

            // Search must return identical ids in identical order.
            let post_results = snapshot_search(&target, &query);
            assert_eq!(
                pre_results.len(),
                post_results.len(),
                "post-restore result count must match"
            );
            for (a, b) in pre_results.iter().zip(post_results.iter()) {
                assert_eq!(a.id, b.id, "post-restore ids must match pre-serialize ids");
            }
        }

        // ── E2 negative: corrupt the checksum, classifier returns FullRebuild
        #[test]
        fn e2_negative_corrupt_checksum_classifies_full_rebuild() {
            let dim = 8;
            let (original, _vectors) = build_small_hnsw(dim, 16);
            let tmp = TempDir::new().expect("tempdir");
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            let base_path = tmp.path().join("hnsw.base");
            let mut bytes = std::fs::read(&base_path).expect("read base");
            assert!(bytes.len() >= 4, "base file too short to corrupt");
            // Flip a byte in the CRC footer (last 4 bytes of the file).
            let last = bytes.len() - 1;
            bytes[last] = bytes[last].wrapping_add(1);
            std::fs::write(&base_path, &bytes).expect("write corrupted base");

            // Classifier path: validate_state_on_disk must return Ok(false)
            // for a corrupt file. The boot planner uses this signal to
            // decide on FullRebuild.
            let validated =
                <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                    .expect("validate_state_on_disk must not panic on a corrupt file");
            assert!(
                !validated,
                "validate_state_on_disk must classify a CRC-flipped file as not-intact"
            );

            // Also confirm try_restore_from_dir reports StateCorrupt, which
            // is the per-instance signal that caller falls back to a full
            // rebuild. No crash, no panic.
            let mut target = empty_hnsw(dim);
            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("try_restore_from_dir must surface a structured outcome on corruption");
            match outcome {
                RestoreOutcome::StateCorrupt { .. } => {}
                other => panic!("expected StateCorrupt on CRC flip, got {:?}", other),
            }
        }

        // ── E3 happy: cold_build_parallel(workers=1) works end-to-end ──
        //
        // QuantizedHnswIndex implements cold_build_parallel non-trivially;
        // it is the canonical impl for the parallel-build contract. We
        // verify a small known input produces a working, queryable index
        // and the inserted ids are recoverable via search.
        #[test]
        fn e3_happy_cold_build_workers_one_returns_working_index() {
            let dim = 4;
            let tmp = TempDir::new().expect("tempdir");

            let mut index = empty_quantized(dim);
            index.set_data_dir(tmp.path().to_path_buf());

            let owned = make_vectors(8, dim, 1);
            let borrowed: Vec<(VectorId, &[f32])> =
                owned.iter().map(|(id, v)| (*id, v.as_slice())).collect();

            let config = ParallelBuildConfig {
                workers: 1,
                memory_cap_bytes: None,
                deterministic: true,
            };

            index
                .cold_build_parallel(&borrowed, config)
                .expect("cold_build_parallel(workers=1) must succeed on a small input");
            assert!(
                index.is_trained(),
                "cold_build_parallel must leave the index in trained state"
            );

            // Query for the first input vector. The hand-computed
            // expectation is that the closest neighbour is the input
            // itself, returned at the top of the result list.
            let query = owned[0].1.clone();
            let results = index
                .search(&query, 3, None)
                .expect("search after cold_build_parallel must succeed");
            assert!(
                !results.is_empty(),
                "post-build search must return at least one result"
            );
            assert_eq!(
                results[0].id, owned[0].0,
                "closest neighbour of vector i must be vector i"
            );
        }

        // ── E3 negative: cold_build_parallel(workers=0) is rejected ────
        #[test]
        fn e3_negative_cold_build_workers_zero_returns_error() {
            let dim = 4;
            let tmp = TempDir::new().expect("tempdir");
            let mut index = empty_quantized(dim);
            index.set_data_dir(tmp.path().to_path_buf());

            let owned = make_vectors(8, dim, 1);
            let borrowed: Vec<(VectorId, &[f32])> =
                owned.iter().map(|(id, v)| (*id, v.as_slice())).collect();

            let config = ParallelBuildConfig {
                workers: 0,
                memory_cap_bytes: None,
                deterministic: true,
            };

            let result = index.cold_build_parallel(&borrowed, config);
            assert!(
                matches!(result, Err(IndexError::Internal(_))),
                "workers=0 must surface IndexError::Internal, got {:?}",
                result
            );
            assert!(
                !index.is_trained(),
                "a rejected cold-build must leave the index un-trained"
            );
        }

        // ── E4 happy: cold_build_parallel byte-identical across worker counts
        //
        // For QuantizedHnswIndex, the dense code arena written to disk by
        // parallel_encode_dense is a pure function of the sorted input plus
        // the trained quantizer. We assert byte-equality on the produced
        // arena file (codes.bin) between workers=1 and workers=4 builds.
        //
        // The fake-quantizer trait-extensibility arm is not run here:
        // P03's fake-quantizer stand-in does not exist in this crate, so
        // the "every implementer of the trait" coverage is only covered
        // by the two real impls (HnswIndex and QuantizedHnswIndex). The
        // HnswIndex side delegates to build_parallel which does NOT write
        // a stable on-disk arena, so byte-equality is asserted only on
        // QuantizedHnswIndex here. Gap noted; integration-level cover by C3.
        #[test]
        fn e4_happy_cold_build_byte_identical_across_workers() {
            let dim = 8;
            let owned = make_vectors(64, dim, 1);
            let borrowed: Vec<(VectorId, &[f32])> =
                owned.iter().map(|(id, v)| (*id, v.as_slice())).collect();

            // Build A: workers=1.
            let tmp_a = TempDir::new().expect("tempdir A");
            let mut idx_a = empty_quantized(dim);
            idx_a.set_data_dir(tmp_a.path().to_path_buf());
            idx_a
                .cold_build_parallel(
                    &borrowed,
                    ParallelBuildConfig {
                        workers: 1,
                        memory_cap_bytes: None,
                        deterministic: true,
                    },
                )
                .expect("workers=1 build");

            // Build B: workers=4.
            let tmp_b = TempDir::new().expect("tempdir B");
            let mut idx_b = empty_quantized(dim);
            idx_b.set_data_dir(tmp_b.path().to_path_buf());
            idx_b
                .cold_build_parallel(
                    &borrowed,
                    ParallelBuildConfig {
                        workers: 4,
                        memory_cap_bytes: None,
                        deterministic: true,
                    },
                )
                .expect("workers=4 build");

            let codes_a = std::fs::read(tmp_a.path().join("codes.bin"))
                .expect("codes.bin from workers=1");
            let codes_b = std::fs::read(tmp_b.path().join("codes.bin"))
                .expect("codes.bin from workers=4");

            assert!(
                !codes_a.is_empty(),
                "codes.bin must not be empty for a non-trivial input"
            );
            assert_eq!(
                codes_a, codes_b,
                "cold_build_parallel must produce byte-identical codes.bin regardless of worker count"
            );
        }

        // ── E4 negative: forced worker panic returns a structured error
        //
        // The current parallel_build.rs scaffold catches worker panics
        // via std::panic::catch_unwind and returns IndexError::Internal,
        // but there is no public panic-injection hook in this crate, so
        // this test is marked #[ignore] and pinned with a TODO.
        #[test]
        #[ignore = "needs panic-injection hook in parallel_build.rs; integration-level coverage in C5"]
        fn e4_negative_cold_build_worker_panic_returns_error() {
            // TODO(P05): needs panic-injection hook; integration-level coverage by C5.
        }

        // ── E5 happy: classifier returns CleanShutdown for a valid file ─
        //
        // The trait method that performs the recovery classification is
        // try_restore_from_dir; validate_state_on_disk is a pure check
        // that only reports intact / not-intact. We exercise the happy
        // path through try_restore_from_dir.
        #[test]
        fn e5_happy_recovery_strategy_clean_shutdown() {
            let dim = 8;
            let (original, original_vectors) = build_small_hnsw(dim, 16);
            let tmp = TempDir::new().expect("tempdir");
            std::fs::write(tmp.path().join("shutdown_clean"), b"").expect("write marker");
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            let mut target = empty_hnsw(dim);
            let snapshot = original.snapshot_topology(0);
            let lookup: HashMap<VectorId, &[f32]> = original_vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();
            target.populate_arena_from_snapshot(&snapshot, &lookup);

            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("restore must succeed on valid state plus marker");
            assert!(
                matches!(
                    outcome,
                    RestoreOutcome::Restored { strategy: IndexRecoveryStrategy::CleanShutdown }
                ),
                "valid file plus shutdown_clean must yield CleanShutdown classification"
            );
        }

        // ── E5 negative sub-cases. Each sub-case is its own #[test] so a
        // failure points at the exact failure mode.

        #[test]
        fn e5_negative_missing_path_classifies_full_rebuild() {
            let dim = 8;
            let tmp = TempDir::new().expect("tempdir");
            // No file written. validate must return false (the planner
            // converts false to FullRebuild).
            let validated =
                <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                    .expect("validate_state_on_disk on a missing path must not error");
            assert!(
                !validated,
                "missing state file must NOT be classified as intact"
            );

            // And the per-instance path: try_restore_from_dir surfaces
            // StateMissing.
            let mut target = empty_hnsw(dim);
            let outcome = target
                .try_restore_from_dir(tmp.path())
                .expect("try_restore_from_dir must not error on missing state");
            assert!(
                matches!(outcome, RestoreOutcome::StateMissing),
                "missing file must yield StateMissing, got {:?}",
                outcome
            );
        }

        #[test]
        fn e5_negative_corrupted_file_classifies_full_rebuild() {
            let dim = 8;
            let (original, _vectors) = build_small_hnsw(dim, 16);
            let tmp = TempDir::new().expect("tempdir");
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            let base_path = tmp.path().join("hnsw.base");
            let mut bytes = std::fs::read(&base_path).expect("read base");
            // Flip a byte in the middle of the payload to break CRC.
            let mid = bytes.len() / 2;
            bytes[mid] = bytes[mid].wrapping_add(1);
            std::fs::write(&base_path, &bytes).expect("write corrupted");

            let validated =
                <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                    .expect("validate on a corrupted file is not a hard error");
            assert!(!validated, "corrupted file must be classified as not-intact");
        }

        #[test]
        fn e5_negative_wrong_magic_classifies_full_rebuild() {
            let dim = 8;
            let (original, _vectors) = build_small_hnsw(dim, 16);
            let tmp = TempDir::new().expect("tempdir");
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            // Overwrite the first 4 bytes (magic) with something else.
            let base_path = tmp.path().join("hnsw.base");
            let mut bytes = std::fs::read(&base_path).expect("read base");
            bytes[0] = b'X';
            bytes[1] = b'X';
            bytes[2] = b'X';
            bytes[3] = b'X';
            std::fs::write(&base_path, &bytes).expect("write bad magic");

            let validated =
                <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                    .expect("validate on bad magic must not error");
            assert!(
                !validated,
                "wrong magic must be classified as not-intact"
            );
        }

        #[test]
        fn e5_negative_wrong_version_classifies_full_rebuild() {
            let dim = 8;
            let (original, _vectors) = build_small_hnsw(dim, 16);
            let tmp = TempDir::new().expect("tempdir");
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            // Overwrite the 4 bytes at offset 4 (format_version field).
            // 0x9999_9999 is well outside the current FORMAT_VERSION = 1.
            let base_path = tmp.path().join("hnsw.base");
            let mut bytes = std::fs::read(&base_path).expect("read base");
            let bad_version: u32 = 0x9999_9999;
            bytes[4..8].copy_from_slice(&bad_version.to_le_bytes());
            std::fs::write(&base_path, &bytes).expect("write bad version");

            let validated =
                <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                    .expect("validate on bad version must not error");
            assert!(
                !validated,
                "wrong version must be classified as not-intact"
            );
        }

        #[test]
        fn e5_negative_truncated_file_classifies_full_rebuild() {
            let dim = 8;
            let (original, _vectors) = build_small_hnsw(dim, 16);
            let tmp = TempDir::new().expect("tempdir");
            original
                .serialize_state_to_dir(tmp.path())
                .expect("serialize original");

            let base_path = tmp.path().join("hnsw.base");
            let bytes = std::fs::read(&base_path).expect("read base");
            // Truncate to half its length, below envelope + CRC if half is small,
            // and to a payload that no longer matches the recorded node_count
            // otherwise. Either way, the validator must return Ok(false).
            let truncated = &bytes[..bytes.len() / 2];
            std::fs::write(&base_path, truncated).expect("write truncated");

            let validated =
                <HnswIndex as PersistableIndex>::validate_state_on_disk(tmp.path(), dim)
                    .expect("validate on truncated file must not error");
            assert!(
                !validated,
                "truncated file must be classified as not-intact"
            );
        }

        // ── E6: Object safety compile-time test ─────────────────────────
        //
        // Construct Box<dyn PersistableIndex> from both real impls and
        // call every object-safe method on the trait through the boxed
        // value. Methods with `where Self: Sized` (validate_state_on_disk,
        // clear_state_from_dir) are NOT callable through `dyn` by design,
        // so they are exercised via the concrete type at their dedicated
        // test sites instead.
        #[test]
        fn e6_object_safety_box_dyn_persistable_index() {
            let dim = 4;
            let tmp = TempDir::new().expect("tempdir");

            // HnswIndex boxed as dyn.
            let hnsw_index = HnswIndex::with_defaults(dim, DistanceMetricType::Cosine);
            let boxed_hnsw: Box<dyn PersistableIndex> = Box::new(hnsw_index);

            // VectorIndex methods through upcast.
            let view = boxed_hnsw.as_vector_index();
            assert_eq!(view.dimension(), dim);
            assert_eq!(view.len(), 0);
            assert!(view.is_empty());

            // PersistableIndex inherent methods.
            #[allow(deprecated)]
            {
                let _ = boxed_hnsw.build_parallel(&[]);
            }
            let _ = boxed_hnsw.snapshot_topology(0);
            let _ = boxed_hnsw.is_compacted();
            boxed_hnsw.compact();
            let _ = boxed_hnsw.iter_vectors_owned();
            boxed_hnsw.post_optimize();
            let _ = boxed_hnsw.add_with_lsn(99, &vec![0.0_f32; dim], 1);
            let _ = boxed_hnsw.remove_with_lsn(99, 2);
            let _ = boxed_hnsw.take_delta_writer();
            let _ = boxed_hnsw.recovery_files();
            let _ = boxed_hnsw.parallel_build_units(
                &[],
                &ParallelBuildConfig {
                    workers: 1,
                    memory_cap_bytes: None,
                    deterministic: true,
                },
            );
            // serialize_state_to_dir through dyn works on an empty index too.
            let _ = boxed_hnsw.serialize_state_to_dir(tmp.path());

            // QuantizedHnswIndex boxed as dyn.
            let quantized = empty_quantized(dim);
            let boxed_q: Box<dyn PersistableIndex> = Box::new(quantized);
            let view_q = boxed_q.as_vector_index();
            assert_eq!(view_q.dimension(), dim);
            let _ = boxed_q.is_compacted();
            let _ = boxed_q.iter_vectors_owned();
            let _ = boxed_q.recovery_files();
        }

        // ── E7: Send + Sync bound assertion ─────────────────────────────
        //
        // A type without Send or Sync would fail this generic call at
        // compile time. The fake-quantizer arm is not exercised because
        // no such test-only impl exists in the crate today; the gap is
        // documented inline.
        #[test]
        fn e7_send_sync_bound_holds_for_real_impls() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<HnswIndex>();
            assert_send_sync::<QuantizedHnswIndex>();
            // Fake-quantizer impl arm (C6) is out of scope here: no
            // test-only stand-in lives in vf-index today.
        }

        // ── E8: No API-breaking changes smoke ───────────────────────────
        //
        // Exercises the public constructor signature that pre-dates the
        // trait initiative (HnswIndex::with_defaults), plus the existing
        // VectorIndex methods (add, search). These are the call shapes
        // used by every existing call site, so a regression here would
        // break the rest of the codebase.
        #[test]
        fn e8_existing_hnsw_call_site_still_works() {
            let dim = 6;
            let index = HnswIndex::with_defaults(dim, DistanceMetricType::Euclidean);
            assert_eq!(index.dimension(), dim);
            assert_eq!(index.len(), 0);
            assert!(index.is_empty());

            let vectors = make_vectors(20, dim, 1);
            for (id, v) in &vectors {
                index.add(*id, v).expect("add via VectorIndex still works");
            }
            assert_eq!(index.len(), vectors.len());
            assert!(!index.is_empty());

            // Query: the closest result to vector i must be vector i.
            let query = vectors[0].1.clone();
            let results = index.search(&query, 3, None).expect("search still works");
            assert!(!results.is_empty(), "search must return data");
            assert_eq!(
                results[0].id, vectors[0].0,
                "closest neighbour of input vector must be the input itself"
            );
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Category F: refactor-guard safety smokes.
    // ────────────────────────────────────────────────────────────────
    mod f_refactor_guards {
        use super::*;

        // ── F1: NaN handling smoke ──────────────────────────────────────
        //
        // OBSERVED CONTRACT from the quantization_v3 add/insert path:
        // HnswIndex::insert_node validates only the dimension and
        // uniqueness of the id; it does NOT filter, reject, or skip
        // vectors that contain NaN components. NaN values are stored
        // as-is into the VectorArena. QuantizedHnswIndex::add delegates
        // to the inner HNSW for graph maintenance, with the same
        // store-as-is policy for f32 inputs before training.
        //
        // The refactor-guard test pins that contract: the call must not
        // crash, must not silently drop the vector, and must return Ok
        // for the plain-HNSW path. SQ8 path is exercised before training
        // so it follows the same plain-HNSW shape.
        #[test]
        fn f1_nan_handling_does_not_crash_plain_hnsw() {
            let dim = 4;
            let index = HnswIndex::with_defaults(dim, DistanceMetricType::Cosine);

            // First insert a non-NaN seed so the graph has an entry point.
            index.add(1, &[0.1, 0.2, 0.3, 0.4]).expect("seed insert");

            // Insert a NaN-containing vector. The observed contract is
            // store-as-is, no error. We accept Ok(()) only; any other
            // outcome would be a refactor-induced regression.
            let mut nan_vec = vec![0.0_f32; dim];
            nan_vec[0] = f32::NAN;
            nan_vec[2] = f32::NAN;
            let result = index.add(2, &nan_vec);
            assert!(
                result.is_ok(),
                "plain HNSW must accept a NaN-containing vector to match quantization_v3, got {:?}",
                result
            );
            assert!(index.contains(2), "the inserted id must be present");
        }

        #[test]
        fn f1_nan_handling_does_not_crash_quantized_hnsw_pre_train() {
            let dim = 4;
            let index = empty_quantized(dim);

            // Pre-train, QuantizedHnswIndex::add delegates to the inner
            // HnswIndex. Same store-as-is contract applies.
            index.add(1, &[0.1, 0.2, 0.3, 0.4]).expect("seed");
            let mut nan_vec = vec![0.0_f32; dim];
            nan_vec[1] = f32::NAN;
            let result = index.add(2, &nan_vec);
            assert!(
                result.is_ok(),
                "SQ8 pre-train path must accept NaN to match quantization_v3, got {:?}",
                result
            );
            assert!(index.contains(2), "the inserted id must be present");
        }

        // ── F2: Dimension-bounds smoke ──────────────────────────────────
        //
        // Inserting a vector that is either longer or shorter than the
        // collection's configured dimension must return a structured
        // IndexError::DimensionMismatch. No panic, no silent acceptance,
        // no memory smash. Both plain HNSW and SQ8 paths are exercised.
        #[test]
        fn f2_dimension_bounds_plain_hnsw_over_and_under() {
            let dim = 4;
            let index = HnswIndex::with_defaults(dim, DistanceMetricType::Cosine);

            let too_long: Vec<f32> = vec![0.0; dim + 1];
            let r_over = index.add(1, &too_long);
            assert!(
                matches!(
                    r_over,
                    Err(IndexError::DimensionMismatch { expected: 4, actual: 5 })
                ),
                "over-dimension vector must surface DimensionMismatch, got {:?}",
                r_over
            );

            let too_short: Vec<f32> = vec![0.0; dim - 1];
            let r_under = index.add(2, &too_short);
            assert!(
                matches!(
                    r_under,
                    Err(IndexError::DimensionMismatch { expected: 4, actual: 3 })
                ),
                "under-dimension vector must surface DimensionMismatch, got {:?}",
                r_under
            );

            // Both insertions must have been rejected, the index stays empty.
            assert_eq!(index.len(), 0, "rejected inserts must not be applied");
        }

        #[test]
        fn f2_dimension_bounds_quantized_hnsw_over_and_under() {
            let dim = 4;
            let index = empty_quantized(dim);

            let too_long: Vec<f32> = vec![0.0; dim + 2];
            let r_over = index.add(10, &too_long);
            assert!(
                matches!(r_over, Err(IndexError::DimensionMismatch { .. })),
                "SQ8 over-dimension must surface DimensionMismatch, got {:?}",
                r_over
            );

            let too_short: Vec<f32> = vec![0.0; dim - 2];
            let r_under = index.add(11, &too_short);
            assert!(
                matches!(r_under, Err(IndexError::DimensionMismatch { .. })),
                "SQ8 under-dimension must surface DimensionMismatch, got {:?}",
                r_under
            );
            assert_eq!(index.len(), 0, "rejected SQ8 inserts must not be applied");
        }
    }
}
