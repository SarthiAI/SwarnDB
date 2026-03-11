// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use std::collections::HashMap;

use rayon::prelude::*;
use vf_core::types::{Metadata, ScoredResult, VectorId};
use vf_index::traits::VectorIndex;

use crate::filter::{FilterExpression, QueryError};
use crate::index_manager::IndexManager;
use crate::strategy::{FilterStrategy, QueryExecutor};

pub struct BatchQuery {
    pub query_vector: Vec<f32>,
    pub k: usize,
    pub filter: Option<FilterExpression>,
    pub strategy: FilterStrategy,
    pub ef_search: Option<usize>,
}

pub struct BatchResult {
    pub results: Vec<ScoredResult>,
    pub query_index: usize,
}

pub struct BatchExecutor;

/// Default maximum batch size. Override via `search_batch_with_limit`.
const DEFAULT_MAX_BATCH_SIZE: usize = usize::MAX;

impl BatchExecutor {
    pub fn search_batch(
        index: &dyn VectorIndex,
        queries: &[BatchQuery],
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
    ) -> Vec<Result<Vec<ScoredResult>, QueryError>> {
        Self::search_batch_with_limit(index, queries, index_manager, metadata_store, DEFAULT_MAX_BATCH_SIZE)
    }

    pub fn search_batch_with_limit(
        index: &dyn VectorIndex,
        queries: &[BatchQuery],
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
        max_batch_size: usize,
    ) -> Vec<Result<Vec<ScoredResult>, QueryError>> {
        if queries.len() > max_batch_size {
            let err_msg = format!(
                "batch size {} exceeds configured maximum of {max_batch_size}",
                queries.len()
            );
            return (0..queries.len())
                .map(|_| Err(QueryError::Internal(err_msg.clone())))
                .collect();
        }
        queries
            .par_iter()
            .map(|q| {
                QueryExecutor::search(
                    index,
                    &q.query_vector,
                    q.k,
                    q.filter.as_ref(),
                    &q.strategy,
                    index_manager,
                    metadata_store,
                    q.ef_search,
                )
            })
            .collect()
    }

    pub fn search_batch_uniform(
        index: &dyn VectorIndex,
        query_vectors: &[Vec<f32>],
        k: usize,
        filter: Option<&FilterExpression>,
        strategy: &FilterStrategy,
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
        ef_search: Option<usize>,
    ) -> Vec<Result<Vec<ScoredResult>, QueryError>> {
        Self::search_batch_uniform_with_limit(index, query_vectors, k, filter, strategy, index_manager, metadata_store, ef_search, DEFAULT_MAX_BATCH_SIZE)
    }

    pub fn search_batch_uniform_with_limit(
        index: &dyn VectorIndex,
        query_vectors: &[Vec<f32>],
        k: usize,
        filter: Option<&FilterExpression>,
        strategy: &FilterStrategy,
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
        ef_search: Option<usize>,
        max_batch_size: usize,
    ) -> Vec<Result<Vec<ScoredResult>, QueryError>> {
        if query_vectors.len() > max_batch_size {
            let err_msg = format!(
                "batch size {} exceeds configured maximum of {max_batch_size}",
                query_vectors.len()
            );
            return (0..query_vectors.len())
                .map(|_| Err(QueryError::Internal(err_msg.clone())))
                .collect();
        }
        query_vectors
            .par_iter()
            .map(|query| {
                QueryExecutor::search(
                    index,
                    query,
                    k,
                    filter,
                    strategy,
                    index_manager,
                    metadata_store,
                    ef_search,
                )
            })
            .collect()
    }
}
