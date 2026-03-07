// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Shared HNSW types extracted to break circular dependencies between modules.

use vf_core::types::VectorId;

pub(crate) struct HnswNode {
    pub(crate) vector: Vec<f32>,
    pub(crate) neighbors: Vec<Vec<VectorId>>,
}

impl HnswNode {
    pub(crate) fn new(vector: Vec<f32>, level: usize) -> Self {
        let neighbors = (0..=level).map(|_| Vec::new()).collect();
        Self { vector, neighbors }
    }

    pub(crate) fn max_level(&self) -> usize {
        self.neighbors.len().saturating_sub(1)
    }
}
