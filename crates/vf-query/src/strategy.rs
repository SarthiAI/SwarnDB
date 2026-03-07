// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;

use roaring::RoaringBitmap;
use vf_core::types::{Metadata, ScoredResult, VectorId};
use vf_index::traits::VectorIndex;

use crate::eval::FilterEvaluator;
use crate::filter::{FilterExpression, QueryError};
use crate::index_manager::IndexManager;

#[derive(Clone, Debug)]
pub enum FilterStrategy {
    PreFilter,
    PostFilter { oversample_factor: usize },
    Auto,
}

impl Default for FilterStrategy {
    fn default() -> Self {
        FilterStrategy::Auto
    }
}

pub struct QueryExecutor;

impl QueryExecutor {
    pub fn search(
        index: &dyn VectorIndex,
        query: &[f32],
        k: usize,
        filter: Option<&FilterExpression>,
        strategy: &FilterStrategy,
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        let filter = match filter {
            Some(f) => f,
            None => return Ok(index.search(query, k)?),
        };

        match strategy {
            FilterStrategy::PreFilter => {
                Self::execute_pre_filter(index, query, k, filter, index_manager, metadata_store)
            }
            FilterStrategy::PostFilter { oversample_factor } => {
                Self::execute_post_filter(index, query, k, filter, *oversample_factor, metadata_store)
            }
            FilterStrategy::Auto => {
                Self::execute_auto(index, query, k, filter, index_manager, metadata_store)
            }
        }
    }

    pub fn auto_search(
        index: &dyn VectorIndex,
        query: &[f32],
        k: usize,
        filter: Option<&FilterExpression>,
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        Self::search(index, query, k, filter, &FilterStrategy::Auto, index_manager, metadata_store)
    }

    fn execute_pre_filter(
        index: &dyn VectorIndex,
        query: &[f32],
        k: usize,
        filter: &FilterExpression,
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        let candidates = Self::resolve_candidates(filter, index_manager, metadata_store);
        Ok(index.search_with_candidates(query, k, &candidates)?)
    }

    fn execute_post_filter(
        index: &dyn VectorIndex,
        query: &[f32],
        k: usize,
        filter: &FilterExpression,
        oversample_factor: usize,
        metadata_store: &HashMap<VectorId, Metadata>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        let expanded_k = k * oversample_factor;
        let results = index.search(query, expanded_k)?;

        let filtered: Vec<ScoredResult> = results
            .into_iter()
            .filter(|r| {
                metadata_store
                    .get(&r.id)
                    .map_or(false, |meta| FilterEvaluator::evaluate(filter, meta))
            })
            .take(k)
            .collect();

        Ok(filtered)
    }

    fn execute_auto(
        index: &dyn VectorIndex,
        query: &[f32],
        k: usize,
        filter: &FilterExpression,
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        let index_len = index.len();
        if index_len == 0 {
            return Ok(Vec::new());
        }

        if let Some(im) = index_manager {
            if let Some(bitmap) = im.evaluate_filter(filter) {
                let selectivity = bitmap.len() as f64 / index_len as f64;

                if selectivity > 0.01 {
                    let candidates = bitmap_to_candidates(&bitmap);
                    return Ok(index.search_with_candidates(query, k, &candidates)?);
                } else {
                    let oversample = compute_oversample(selectivity);
                    return Self::execute_post_filter(
                        index, query, k, filter, oversample, metadata_store,
                    );
                }
            }
        }

        Self::execute_post_filter(index, query, k, filter, 3, metadata_store)
    }

    fn resolve_candidates(
        filter: &FilterExpression,
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
    ) -> Vec<VectorId> {
        if let Some(im) = index_manager {
            if let Some(bitmap) = im.evaluate_filter(filter) {
                return bitmap_to_candidates(&bitmap);
            }
        }
        FilterEvaluator::evaluate_batch(filter, metadata_store)
    }
}

fn bitmap_to_candidates(bitmap: &RoaringBitmap) -> Vec<VectorId> {
    bitmap.iter().map(|id| id as u64).collect()
}

fn compute_oversample(selectivity: f64) -> usize {
    if selectivity <= 0.0 {
        return 20;
    }
    let raw = (1.0 / selectivity).ceil() as usize;
    raw.max(3).min(20)
}
