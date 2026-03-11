// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use std::collections::HashMap;

use roaring::RoaringBitmap;
use vf_core::types::{Metadata, ScoredResult, VectorId};
use vf_index::traits::VectorIndex;

use crate::eval::CompiledFilter;
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
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        let filter = match filter {
            Some(f) => f,
            None => return Ok(index.search(query, k, ef_search)?),
        };

        match strategy {
            FilterStrategy::PreFilter => {
                Self::execute_pre_filter(index, query, k, filter, index_manager, metadata_store, ef_search)
            }
            FilterStrategy::PostFilter { oversample_factor } => {
                // Use adaptive oversampling: if an IndexManager is available,
                // estimate selectivity and override the fixed factor.
                let adaptive_factor = if let Some(im) = index_manager {
                    let selectivity = im.estimate_selectivity(filter);
                    let estimated = compute_oversample(selectivity);
                    // Use the larger of user-specified and estimated factors
                    estimated.max(*oversample_factor)
                } else {
                    *oversample_factor
                };
                Self::execute_post_filter(index, query, k, filter, adaptive_factor, metadata_store, ef_search)
            }
            FilterStrategy::Auto => {
                Self::execute_auto(index, query, k, filter, index_manager, metadata_store, ef_search)
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
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        Self::search(
            index,
            query,
            k,
            filter,
            &FilterStrategy::Auto,
            index_manager,
            metadata_store,
            ef_search,
        )
    }

    fn execute_pre_filter(
        index: &dyn VectorIndex,
        query: &[f32],
        k: usize,
        filter: &FilterExpression,
        index_manager: Option<&IndexManager>,
        metadata_store: &HashMap<VectorId, Metadata>,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        let candidates = Self::resolve_candidates(filter, index_manager, metadata_store);
        Ok(index.search_with_candidates(query, k, &candidates, ef_search)?)
    }

    fn execute_post_filter(
        index: &dyn VectorIndex,
        query: &[f32],
        k: usize,
        filter: &FilterExpression,
        oversample_factor: usize,
        metadata_store: &HashMap<VectorId, Metadata>,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        let expanded_k = k.saturating_mul(oversample_factor);
        let results = index.search(query, expanded_k, ef_search)?;

        // Compile filter once, evaluate many times without AST traversal
        let compiled = CompiledFilter::compile(filter);

        let filtered: Vec<ScoredResult> = results
            .into_iter()
            .filter(|r| {
                metadata_store
                    .get(&r.id)
                    .map_or(false, |meta| compiled.evaluate(meta))
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
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, QueryError> {
        let index_len = index.len();
        if index_len == 0 {
            return Ok(Vec::new());
        }

        if let Some(im) = index_manager {
            // Try exact bitmap evaluation first
            if let Some(bitmap) = im.evaluate_filter(filter) {
                let selectivity = bitmap.len() as f64 / index_len as f64;

                if selectivity > 0.01 {
                    let candidates = bitmap_to_candidates(&bitmap);
                    return Ok(index.search_with_candidates(query, k, &candidates, ef_search)?);
                } else {
                    let oversample = compute_oversample(selectivity);
                    return Self::execute_post_filter(
                        index,
                        query,
                        k,
                        filter,
                        oversample,
                        metadata_store,
                        ef_search,
                    );
                }
            }

            // Bitmap evaluation not possible (e.g., Exists, Not, or missing
            // index) -- fall back to estimated selectivity for adaptive
            // oversampling instead of a fixed factor.
            let estimated_selectivity = im.estimate_selectivity(filter);
            let oversample = compute_oversample(estimated_selectivity);
            return Self::execute_post_filter(index, query, k, filter, oversample, metadata_store, ef_search);
        }

        // No IndexManager available at all -- use a conservative default
        Self::execute_post_filter(index, query, k, filter, 3, metadata_store, ef_search)
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
        CompiledFilter::compile(filter).evaluate_batch(metadata_store)
    }
}

/// Convert a `RoaringBitmap` (u32 IDs) to a `Vec<VectorId>` (u64).
///
/// **Note:** `RoaringBitmap` only stores u32 values, so any `VectorId`
/// above `u32::MAX` (4,294,967,295) will never appear in bitmap-filtered
/// results. This is acceptable for current workloads but should be
/// revisited if VectorId space exceeds 4B.
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
