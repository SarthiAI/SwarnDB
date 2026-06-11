// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Errors surfaced by the hybrid query executor.

/// Failure modes of hybrid query planning and execution.
#[derive(Debug, thiserror::Error)]
pub enum HybridQueryError {
    /// The underlying vector index search failed.
    #[error("vector index error: {0}")]
    Index(#[from] vf_index::traits::IndexError),

    /// The plan is malformed: a step ran against the wrong frontier kind,
    /// a sub-plan returned an unexpected kind, or a result kind mismatch.
    #[error("invalid query plan: {0}")]
    InvalidPlan(String),

    /// The plan carried no steps.
    #[error("empty query plan")]
    EmptyPlan,
}
