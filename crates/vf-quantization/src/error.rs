// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantizationError {
    #[error("not trained: call train() first")]
    NotTrained,

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("empty training data")]
    EmptyTrainingData,

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}
