// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Internal vector identifier — 64-bit for efficiency.
/// External APIs expose UUIDs, mapped to VectorId internally.
pub type VectorId = u64;

/// Internal collection identifier
pub type CollectionId = u64;

/// Metadata value types supported by SwarnDB
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    /// UTF-8 string value
    String(String),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating point
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// List of strings (for multi-label/tagging use cases)
    StringList(Vec<String>),
}

impl MetadataValue {
    /// Returns the type name as a string
    pub fn type_name(&self) -> &'static str {
        match self {
            MetadataValue::String(_) => "string",
            MetadataValue::Int(_) => "int",
            MetadataValue::Float(_) => "float",
            MetadataValue::Bool(_) => "bool",
            MetadataValue::StringList(_) => "string_list",
        }
    }

    /// Try to extract as i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            MetadataValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as f64
    pub fn as_float(&self) -> Option<f64> {
        match self {
            MetadataValue::Float(v) => Some(*v),
            MetadataValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to extract as string reference
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Try to extract as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataValue::String(v) => write!(f, "\"{}\"", v),
            MetadataValue::Int(v) => write!(f, "{}", v),
            MetadataValue::Float(v) => write!(f, "{}", v),
            MetadataValue::Bool(v) => write!(f, "{}", v),
            MetadataValue::StringList(v) => write!(f, "{:?}", v),
        }
    }
}

/// Metadata is a map of string keys to typed values
pub type Metadata = HashMap<String, MetadataValue>;

/// Configuration for a collection
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Name of the collection
    pub name: String,
    /// Vector dimensionality (all vectors in this collection must have this dimension)
    pub dimension: usize,
    /// Distance metric to use for this collection
    pub distance_metric: DistanceMetricType,
    /// Default similarity threshold for virtual graph relationships
    pub default_similarity_threshold: Option<f32>,
    /// Maximum number of vectors in this collection (0 = unlimited)
    pub max_vectors: usize,
    /// Data type for vector storage
    pub data_type: DataTypeConfig,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            dimension: 0,
            distance_metric: DistanceMetricType::Cosine,
            default_similarity_threshold: None,
            max_vectors: 0,
            data_type: DataTypeConfig::F32,
        }
    }
}

/// Distance metric type identifier (serializable, used in config)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetricType {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

impl fmt::Display for DistanceMetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceMetricType::Cosine => write!(f, "cosine"),
            DistanceMetricType::Euclidean => write!(f, "euclidean"),
            DistanceMetricType::DotProduct => write!(f, "dot_product"),
            DistanceMetricType::Manhattan => write!(f, "manhattan"),
        }
    }
}

/// Data type configuration (serializable, used in config)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataTypeConfig {
    F32,
    F16,
    U8,
}

/// A scored search result
#[derive(Clone, Debug)]
pub struct ScoredResult {
    /// The vector ID
    pub id: VectorId,
    /// The distance score (lower = more similar)
    pub score: f32,
    /// Optional metadata (populated if requested)
    pub metadata: Option<Metadata>,
}

impl ScoredResult {
    pub fn new(id: VectorId, score: f32) -> Self {
        Self {
            id,
            score,
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Similarity threshold configuration with precedence rules.
/// Precedence: per-vector > per-query > per-collection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityThreshold {
    /// The threshold value (0.0 to 1.0 for cosine, varies for other metrics)
    pub value: f32,
    /// Source of this threshold (for debugging/audit)
    pub source: ThresholdSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdSource {
    Collection,
    Query,
    Vector,
}

impl SimilarityThreshold {
    pub fn collection(value: f32) -> Self {
        Self { value, source: ThresholdSource::Collection }
    }

    pub fn query(value: f32) -> Self {
        Self { value, source: ThresholdSource::Query }
    }

    pub fn vector(value: f32) -> Self {
        Self { value, source: ThresholdSource::Vector }
    }

    /// Resolve the effective threshold given multiple levels.
    /// Precedence: per-vector > per-query > per-collection
    pub fn resolve(
        collection: Option<&SimilarityThreshold>,
        query: Option<&SimilarityThreshold>,
        vector: Option<&SimilarityThreshold>,
    ) -> Option<f32> {
        vector
            .map(|t| t.value)
            .or_else(|| query.map(|t| t.value))
            .or_else(|| collection.map(|t| t.value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_value_types() {
        let s = MetadataValue::String("hello".to_string());
        assert_eq!(s.type_name(), "string");
        assert_eq!(s.as_str(), Some("hello"));

        let i = MetadataValue::Int(42);
        assert_eq!(i.as_int(), Some(42));
        assert_eq!(i.as_float(), Some(42.0));

        let f = MetadataValue::Float(3.14);
        assert_eq!(f.as_float(), Some(3.14));
        assert_eq!(f.as_int(), None);

        let b = MetadataValue::Bool(true);
        assert_eq!(b.as_bool(), Some(true));
    }

    #[test]
    fn test_metadata_display() {
        let s = MetadataValue::String("test".to_string());
        assert_eq!(format!("{}", s), "\"test\"");

        let i = MetadataValue::Int(42);
        assert_eq!(format!("{}", i), "42");
    }

    #[test]
    fn test_similarity_threshold_precedence() {
        let collection = SimilarityThreshold::collection(0.8);
        let query = SimilarityThreshold::query(0.9);
        let vector = SimilarityThreshold::vector(0.95);

        // Vector supersedes all
        assert_eq!(
            SimilarityThreshold::resolve(Some(&collection), Some(&query), Some(&vector)),
            Some(0.95)
        );

        // Query supersedes collection
        assert_eq!(
            SimilarityThreshold::resolve(Some(&collection), Some(&query), None),
            Some(0.9)
        );

        // Collection is fallback
        assert_eq!(
            SimilarityThreshold::resolve(Some(&collection), None, None),
            Some(0.8)
        );

        // None if nothing set
        assert_eq!(
            SimilarityThreshold::resolve(None, None, None),
            None
        );
    }

    #[test]
    fn test_scored_result() {
        let result = ScoredResult::new(42, 0.95);
        assert_eq!(result.id, 42);
        assert_eq!(result.score, 0.95);
        assert!(result.metadata.is_none());

        let mut meta = HashMap::new();
        meta.insert("key".to_string(), MetadataValue::String("val".to_string()));
        let result = result.with_metadata(meta);
        assert!(result.metadata.is_some());
    }

    #[test]
    fn test_collection_config_default() {
        let config = CollectionConfig::default();
        assert_eq!(config.dimension, 0);
        assert_eq!(config.distance_metric, DistanceMetricType::Cosine);
        assert_eq!(config.max_vectors, 0);
    }
}
