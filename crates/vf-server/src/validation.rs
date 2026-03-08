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

    #[error("ef_search value {value} exceeds maximum {max}")]
    EfSearchTooLarge { value: usize, max: usize },

    #[error("batch_lock_size {value} exceeds maximum {max}")]
    BatchLockSizeTooLarge { value: u32, max: u32 },

    #[error("batch_lock_size must be at least 1")]
    BatchLockSizeZero,

    #[error("wal_flush_every {value} exceeds maximum {max}")]
    WalFlushEveryTooLarge { value: u32, max: u32 },

    #[error("ef_construction {value} is below minimum {min}")]
    EfConstructionTooSmall { value: u32, min: u32 },

    #[error("ef_construction {value} exceeds maximum {max}")]
    EfConstructionTooLarge { value: u32, max: u32 },

    #[error("invalid index_mode '{mode}': must be 'immediate' or 'deferred'")]
    InvalidIndexMode { mode: String },

    #[error("parallel_build requires index_mode='deferred'")]
    ParallelBuildRequiresDeferred,
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

/// Validates the ef_search parameter for HNSW queries.
///
/// Returns `Ok(None)` if ef_search is `None` or `Some(0)` (use default).
/// Returns an error if the value exceeds `max_ef_search`.
/// Otherwise returns `Ok(Some(value as usize))`.
pub fn validate_ef_search(
    ef_search: Option<u32>,
    max_ef_search: usize,
) -> Result<Option<usize>, ValidationError> {
    match ef_search {
        None => Ok(None),
        Some(0) => Ok(None),
        Some(val) => {
            let val_usize = val as usize;
            if val_usize > max_ef_search {
                return Err(ValidationError::EfSearchTooLarge {
                    value: val_usize,
                    max: max_ef_search,
                });
            }
            Ok(Some(val_usize))
        }
    }
}

/// Validates the batch_lock_size parameter for bulk insert operations.
///
/// Returns an error if the value is zero or exceeds the configured maximum.
/// Logs a warning when the value exceeds 5000 (high lock contention risk).
pub fn validate_batch_lock_size(
    size: u32,
    max_batch_lock_size: u32,
) -> Result<(), ValidationError> {
    if size == 0 {
        return Err(ValidationError::BatchLockSizeZero);
    }
    if size > max_batch_lock_size {
        return Err(ValidationError::BatchLockSizeTooLarge {
            value: size,
            max: max_batch_lock_size,
        });
    }
    if size > 5000 {
        tracing::warn!(
            batch_lock_size = size,
            "large batch_lock_size may hold write lock too long, blocking readers"
        );
    }
    Ok(())
}

/// Validates the wal_flush_every parameter for bulk insert operations.
///
/// Logs a WARN when value is 0 (WAL disabled) and an INFO when value > 1 (batched).
pub fn validate_wal_flush_every(
    n: u32,
    max_wal_flush_interval: u32,
) -> Result<(), ValidationError> {
    if n > max_wal_flush_interval {
        return Err(ValidationError::WalFlushEveryTooLarge {
            value: n,
            max: max_wal_flush_interval,
        });
    }
    if n == 0 {
        tracing::warn!("WAL disabled (wal_flush_every=0), data loss possible on crash");
    } else if n > 1 {
        tracing::info!(wal_flush_every = n, "WAL batched every {} vectors", n);
    }
    Ok(())
}

/// Validates the ef_construction override for bulk insert HNSW parameter.
///
/// Returns an error if the value is below 8 or exceeds the configured maximum.
pub fn validate_ef_construction(
    ef: u32,
    max_ef_construction: u32,
) -> Result<(), ValidationError> {
    if ef < 8 {
        return Err(ValidationError::EfConstructionTooSmall {
            value: ef,
            min: 8,
        });
    }
    if ef > max_ef_construction {
        return Err(ValidationError::EfConstructionTooLarge {
            value: ef,
            max: max_ef_construction,
        });
    }
    if ef > 500 {
        tracing::warn!(
            ef_construction = ef,
            "high ef_construction override may significantly slow insert"
        );
    }
    Ok(())
}

/// Validates the index_mode parameter for bulk insert operations.
///
/// Valid values: "immediate", "deferred".
pub fn validate_index_mode(mode: &str) -> Result<(), ValidationError> {
    match mode {
        "immediate" | "deferred" => Ok(()),
        _ => Err(ValidationError::InvalidIndexMode {
            mode: mode.to_string(),
        }),
    }
}

/// Validates bulk insert option combinations.
///
/// Returns an error if parallel_build is true but index_mode is not "deferred".
pub fn validate_bulk_insert_options(
    parallel_build: bool,
    index_mode: &str,
) -> Result<(), ValidationError> {
    if parallel_build && index_mode != "deferred" {
        return Err(ValidationError::ParallelBuildRequiresDeferred);
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
    fn ef_search_none_returns_ok_none() {
        assert_eq!(validate_ef_search(None, 10000).unwrap(), None);
    }

    #[test]
    fn ef_search_zero_returns_ok_none() {
        assert_eq!(validate_ef_search(Some(0), 10000).unwrap(), None);
    }

    #[test]
    fn ef_search_valid_value() {
        assert_eq!(validate_ef_search(Some(500), 10000).unwrap(), Some(500));
    }

    #[test]
    fn ef_search_over_limit() {
        assert!(matches!(
            validate_ef_search(Some(20000), 10000),
            Err(ValidationError::EfSearchTooLarge { value: 20000, max: 10000 })
        ));
    }

    #[test]
    fn status_codes() {
        let err = ValidationError::RequestBodyTooLarge { size: 100, max: 50 };
        assert_eq!(err.status_code(), StatusCode::PAYLOAD_TOO_LARGE);

        let err = ValidationError::DimensionZero;
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }
}
