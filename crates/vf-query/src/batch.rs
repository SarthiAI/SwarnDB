// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

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

impl BatchExecutor {
    pub fn search_batch(
        index: &dyn VectorIndex,
        queries: &[BatchQuery],
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
    ) -> Vec<Result<Vec<ScoredResult>, QueryError>> {
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
