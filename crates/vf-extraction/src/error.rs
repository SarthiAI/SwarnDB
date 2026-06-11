// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Error type for the extraction layer.

/// Errors raised across the extraction layer.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    /// The upstream LLM request failed (network, HTTP status, or transport).
    #[error("llm request failed: {0}")]
    Llm(String),
    /// The LLM response could not be parsed into the expected schema.
    #[error("llm response parse failed: {0}")]
    Parse(String),
    /// An entity or edge did not validate against the active ontology.
    #[error("ontology validation failed: {0}")]
    Ontology(String),
    /// Sealing or opening an api key failed.
    #[error("encryption error: {0}")]
    Crypto(String),
    /// A configuration value was missing or invalid.
    #[error("configuration error: {0}")]
    Config(String),
    /// A filesystem operation failed.
    #[error("io error: {0}")]
    Io(String),
    /// Writing nodes or edges to the graph failed.
    #[error("graph write error: {0}")]
    Graph(String),
    /// The referenced job id is unknown.
    #[error("job not found: {0}")]
    JobNotFound(String),
    /// The operation was cancelled.
    #[error("cancelled")]
    Cancelled,
}
