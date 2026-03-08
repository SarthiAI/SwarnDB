// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use vf_core::types::{ScoredResult, VectorId};

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
}
