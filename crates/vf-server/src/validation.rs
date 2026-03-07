// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Input validation and security middleware for request handling.
//!
//! Provides configurable limits for vector dimensions, batch sizes,
//! request body sizes, metadata sizes, and collection name validation
//! to protect against OOM, abuse, and malformed input.

use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

/// Configuration for input validation limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum allowed vector dimension.
    #[serde(default = "default_max_vector_dimension")]
    pub max_vector_dimension: usize,

    /// Maximum number of vectors in a single batch request.
    #[serde(default = "default_max_vectors_per_request")]
    pub max_vectors_per_request: usize,

    /// Maximum request body size in bytes.
    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,

    /// Maximum metadata size in bytes per vector.
    #[serde(default = "default_max_metadata_size_bytes")]
    pub max_metadata_size_bytes: usize,

    /// Maximum length of a collection name.
    #[serde(default = "default_max_collection_name_length")]
    pub max_collection_name_length: usize,

    /// Allowed pattern for collection names (for documentation; enforced via char check).
    #[serde(default = "default_allowed_collection_name_pattern")]
    pub allowed_collection_name_pattern: String,
}

fn default_max_vector_dimension() -> usize {
    65536
}

fn default_max_vectors_per_request() -> usize {
    10000
}

fn default_max_request_body_bytes() -> usize {
    64 * 1024 * 1024 // 64 MB
}

fn default_max_metadata_size_bytes() -> usize {
    1024 * 1024 // 1 MB
}

fn default_max_collection_name_length() -> usize {
    256
}

fn default_allowed_collection_name_pattern() -> String {
    "^[a-zA-Z0-9_-]+$".to_string()
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_vector_dimension: default_max_vector_dimension(),
            max_vectors_per_request: default_max_vectors_per_request(),
            max_request_body_bytes: default_max_request_body_bytes(),
            max_metadata_size_bytes: default_max_metadata_size_bytes(),
            max_collection_name_length: default_max_collection_name_length(),
            allowed_collection_name_pattern: default_allowed_collection_name_pattern(),
        }
    }
}

/// Errors returned by input validation functions.
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("collection name is empty")]
    CollectionNameEmpty,

    #[error("collection name too long: {length} characters (max {max})")]
    CollectionNameTooLong { length: usize, max: usize },

    #[error("collection name '{name}' contains invalid characters (allowed pattern: {pattern})")]
    CollectionNameInvalidChars { name: String, pattern: String },

    #[error("vector dimension is zero")]
    DimensionZero,

    #[error("vector dimension {dimension} exceeds maximum {max}")]
    DimensionTooLarge { dimension: usize, max: usize },

    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },

    #[error("vector contains NaN value")]
    VectorContainsNaN,

    #[error("vector contains infinity value")]
    VectorContainsInfinity,

    #[error("metadata size {size} bytes exceeds maximum {max} bytes")]
    MetadataTooLarge { size: usize, max: usize },

    #[error("batch size {count} exceeds maximum {max}")]
    BatchTooLarge { count: usize, max: usize },

    #[error("request body size {size} bytes exceeds maximum {max} bytes")]
    RequestBodyTooLarge { size: usize, max: usize },
}

impl ValidationError {
    /// Maps the validation error to an appropriate HTTP status code.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::RequestBodyTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::BAD_REQUEST,
        }
    }
}

/// Validates a collection name against configured constraints.
///
/// Checks that the name is non-empty, within length limits, and contains
/// only alphanumeric characters, underscores, or hyphens.
pub fn validate_collection_name(
    name: &str,
    config: &ValidationConfig,
) -> Result<(), ValidationError> {
    if name.is_empty() {
        return Err(ValidationError::CollectionNameEmpty);
    }

    if name.len() > config.max_collection_name_length {
        return Err(ValidationError::CollectionNameTooLong {
            length: name.len(),
            max: config.max_collection_name_length,
        });
    }

    let valid = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');

    if !valid {
        return Err(ValidationError::CollectionNameInvalidChars {
            name: name.to_string(),
            pattern: config.allowed_collection_name_pattern.clone(),
        });
    }

    Ok(())
}

/// Validates that a vector dimension is within configured limits.
pub fn validate_vector_dimension(
    dim: usize,
    config: &ValidationConfig,
) -> Result<(), ValidationError> {
    if dim == 0 {
        return Err(ValidationError::DimensionZero);
    }

    if dim > config.max_vector_dimension {
        return Err(ValidationError::DimensionTooLarge {
            dimension: dim,
            max: config.max_vector_dimension,
        });
    }

    Ok(())
}

/// Validates vector data for dimension match, NaN, and infinity values.
pub fn validate_vector_data(
    data: &[f32],
    expected_dim: usize,
) -> Result<(), ValidationError> {
    if data.len() != expected_dim {
        return Err(ValidationError::VectorDimensionMismatch {
            expected: expected_dim,
            actual: data.len(),
        });
    }

    for &val in data {
        if val.is_nan() {
            return Err(ValidationError::VectorContainsNaN);
        }
        if val.is_infinite() {
            return Err(ValidationError::VectorContainsInfinity);
        }
    }

    Ok(())
}

/// Validates that metadata size is within configured limits.
pub fn validate_metadata_size(
    metadata: &[u8],
    config: &ValidationConfig,
) -> Result<(), ValidationError> {
    if metadata.len() > config.max_metadata_size_bytes {
        return Err(ValidationError::MetadataTooLarge {
            size: metadata.len(),
            max: config.max_metadata_size_bytes,
        });
    }

    Ok(())
}

/// Validates that a batch size is within configured limits.
pub fn validate_batch_size(
    count: usize,
    config: &ValidationConfig,
) -> Result<(), ValidationError> {
    if count > config.max_vectors_per_request {
        return Err(ValidationError::BatchTooLarge {
            count,
            max: config.max_vectors_per_request,
        });
    }

    Ok(())
}

/// Returns an Axum body size limit layer configured to the given maximum bytes.
///
/// Apply this to your router to enforce request body size limits and protect
/// against OOM from oversized payloads.
pub fn request_size_limit_layer(max_bytes: usize) -> DefaultBodyLimit {
    DefaultBodyLimit::max(max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ValidationConfig {
        ValidationConfig {
            max_vector_dimension: 1024,
            max_vectors_per_request: 100,
            max_request_body_bytes: 1024,
            max_metadata_size_bytes: 512,
            max_collection_name_length: 32,
            ..Default::default()
        }
    }

    #[test]
    fn valid_collection_name() {
        let cfg = test_config();
        assert!(validate_collection_name("my_collection-1", &cfg).is_ok());
    }

    #[test]
    fn empty_collection_name() {
        let cfg = test_config();
        assert!(matches!(
            validate_collection_name("", &cfg),
            Err(ValidationError::CollectionNameEmpty)
        ));
    }

    #[test]
    fn collection_name_too_long() {
        let cfg = test_config();
        let long = "a".repeat(33);
        assert!(matches!(
            validate_collection_name(&long, &cfg),
            Err(ValidationError::CollectionNameTooLong { .. })
        ));
    }

    #[test]
    fn collection_name_invalid_chars() {
        let cfg = test_config();
        assert!(matches!(
            validate_collection_name("my collection!", &cfg),
            Err(ValidationError::CollectionNameInvalidChars { .. })
        ));
    }

    #[test]
    fn dimension_zero() {
        let cfg = test_config();
        assert!(matches!(
            validate_vector_dimension(0, &cfg),
            Err(ValidationError::DimensionZero)
        ));
    }

    #[test]
    fn dimension_too_large() {
        let cfg = test_config();
        assert!(matches!(
            validate_vector_dimension(2048, &cfg),
            Err(ValidationError::DimensionTooLarge { .. })
        ));
    }

    #[test]
    fn valid_dimension() {
        let cfg = test_config();
        assert!(validate_vector_dimension(512, &cfg).is_ok());
    }

    #[test]
    fn vector_data_dimension_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            validate_vector_data(&data, 4),
            Err(ValidationError::VectorDimensionMismatch { .. })
        ));
    }

    #[test]
    fn vector_data_contains_nan() {
        let data = vec![1.0, f32::NAN, 3.0];
        assert!(matches!(
            validate_vector_data(&data, 3),
            Err(ValidationError::VectorContainsNaN)
        ));
    }

    #[test]
    fn vector_data_contains_infinity() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        assert!(matches!(
            validate_vector_data(&data, 3),
            Err(ValidationError::VectorContainsInfinity)
        ));
    }

    #[test]
    fn valid_vector_data() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(validate_vector_data(&data, 3).is_ok());
    }

    #[test]
    fn metadata_too_large() {
        let cfg = test_config();
        let meta = vec![0u8; 1024];
        assert!(matches!(
            validate_metadata_size(&meta, &cfg),
            Err(ValidationError::MetadataTooLarge { .. })
        ));
    }

    #[test]
    fn valid_metadata() {
        let cfg = test_config();
        let meta = vec![0u8; 256];
        assert!(validate_metadata_size(&meta, &cfg).is_ok());
    }

    #[test]
    fn batch_too_large() {
        let cfg = test_config();
        assert!(matches!(
            validate_batch_size(200, &cfg),
            Err(ValidationError::BatchTooLarge { .. })
        ));
    }

    #[test]
    fn valid_batch_size() {
        let cfg = test_config();
        assert!(validate_batch_size(50, &cfg).is_ok());
    }

    #[test]
    fn status_codes() {
        let err = ValidationError::RequestBodyTooLarge { size: 100, max: 50 };
        assert_eq!(err.status_code(), StatusCode::PAYLOAD_TOO_LARGE);

        let err = ValidationError::DimensionZero;
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }
}
