// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

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

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("corrupt persisted state: {0}")]
    Corrupt(String),
}
