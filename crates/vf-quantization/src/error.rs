// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

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
