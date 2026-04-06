// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use vf_core::types::{ScoredResult, SearchQuantizationParams, VectorId};

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
    /// Returns results sorted by distance (ascending — closest first).
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
}

/// Supertrait extending VectorIndex with persistence-related methods
/// (LSN-aware mutations, topology snapshots, delta logging, compaction).
/// Designed so `CollectionState.index` can be `Box<dyn PersistableIndex>`.
pub trait PersistableIndex: VectorIndex {
    /// Insert a vector and emit a delta entry if a writer is attached.
    fn add_with_lsn(&self, id: VectorId, vector: &[f32], lsn: u64) -> Result<(), IndexError>;

    /// Remove a vector and emit a delta entry if a writer is attached.
    fn remove_with_lsn(&self, id: VectorId, lsn: u64) -> Result<(), IndexError>;

    /// Extract a topology snapshot under read lock.
    fn snapshot_topology(&self, snapshot_lsn: u64) -> HnswTopologySnapshot;

    /// Build the flat adjacency cache for optimized search performance.
    fn compact(&self);

    /// Returns `true` if the flat adjacency optimization is currently active.
    fn is_compacted(&self) -> bool;

    /// Parallel bulk-insert vectors into the index.
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
}
