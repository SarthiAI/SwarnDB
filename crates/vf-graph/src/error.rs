// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use vf_core::types::VectorId;

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("node not found: {0}")]
    NodeNotFound(VectorId),

    #[error("edge not found: {from} -> {to}")]
    EdgeNotFound { from: VectorId, to: VectorId },

    #[error("max traversal depth exceeded: {0}")]
    MaxDepthExceeded(usize),

    #[error("index error: {0}")]
    IndexError(#[from] vf_index::traits::IndexError),

    #[error("internal error: {0}")]
    Internal(String),
}
