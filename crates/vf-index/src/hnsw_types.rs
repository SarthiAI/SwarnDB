// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Shared HNSW types extracted to break circular dependencies between modules.

use vf_core::types::VectorId;

pub(crate) struct HnswNode {
    /// Slot index into the `VectorArena` that owns this node's vector data.
    pub(crate) vector_slot: usize,
    pub(crate) neighbors: Vec<Vec<VectorId>>,
}

impl HnswNode {
    pub(crate) fn new(vector_slot: usize, level: usize) -> Self {
        let neighbors = (0..=level).map(|_| Vec::new()).collect();
        Self { vector_slot, neighbors }
    }

    pub(crate) fn max_level(&self) -> usize {
        self.neighbors.len().saturating_sub(1)
    }
}
