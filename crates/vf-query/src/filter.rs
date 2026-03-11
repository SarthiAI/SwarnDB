// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use serde::{Deserialize, Serialize};
use vf_core::types::MetadataValue;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FilterExpression {
    And(Vec<FilterExpression>),
    Or(Vec<FilterExpression>),
    Not(Box<FilterExpression>),

    Eq(String, MetadataValue),
    Ne(String, MetadataValue),
    Gt(String, MetadataValue),
    Gte(String, MetadataValue),
    Lt(String, MetadataValue),
    Lte(String, MetadataValue),

    In(String, Vec<MetadataValue>),
    Between(String, MetadataValue, MetadataValue),

    Exists(String),
    Contains(String, String),
}

impl FilterExpression {
    pub fn and(filters: Vec<FilterExpression>) -> Self {
        FilterExpression::And(filters)
    }

    pub fn or(filters: Vec<FilterExpression>) -> Self {
        FilterExpression::Or(filters)
    }

    pub fn not(filter: FilterExpression) -> Self {
        FilterExpression::Not(Box::new(filter))
    }

    pub fn eq(field: impl Into<String>, value: MetadataValue) -> Self {
        FilterExpression::Eq(field.into(), value)
    }

    pub fn ne(field: impl Into<String>, value: MetadataValue) -> Self {
        FilterExpression::Ne(field.into(), value)
    }

    pub fn gt(field: impl Into<String>, value: MetadataValue) -> Self {
        FilterExpression::Gt(field.into(), value)
    }

    pub fn gte(field: impl Into<String>, value: MetadataValue) -> Self {
        FilterExpression::Gte(field.into(), value)
    }

    pub fn lt(field: impl Into<String>, value: MetadataValue) -> Self {
        FilterExpression::Lt(field.into(), value)
    }

    pub fn lte(field: impl Into<String>, value: MetadataValue) -> Self {
        FilterExpression::Lte(field.into(), value)
    }

    /// Maximum number of values allowed in an In filter to prevent unbounded memory usage.
    /// Raised to 1M to support tag-based filtering with large category sets.
    pub const MAX_IN_VALUES: usize = 1_000_000;

    pub fn r#in(field: impl Into<String>, values: Vec<MetadataValue>) -> Self {
        FilterExpression::In(field.into(), values)
    }

    /// Validate the filter expression, returning an error if limits are exceeded.
    pub fn validate(&self) -> Result<(), QueryError> {
        self.validate_inner(0)
    }

    /// Maximum nesting depth for filter expressions.
    const MAX_NESTING_DEPTH: usize = 32;

    fn validate_inner(&self, depth: usize) -> Result<(), QueryError> {
        if depth > Self::MAX_NESTING_DEPTH {
            return Err(QueryError::Internal(format!(
                "filter nesting depth exceeds maximum of {}",
                Self::MAX_NESTING_DEPTH
            )));
        }
        match self {
            FilterExpression::And(children) | FilterExpression::Or(children) => {
                for child in children {
                    child.validate_inner(depth + 1)?;
                }
            }
            FilterExpression::Not(child) => {
                child.validate_inner(depth + 1)?;
            }
            FilterExpression::In(_, values) => {
                if values.len() > Self::MAX_IN_VALUES {
                    return Err(QueryError::Internal(format!(
                        "In filter has {} values, maximum is {}",
                        values.len(),
                        Self::MAX_IN_VALUES
                    )));
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn between(field: impl Into<String>, low: MetadataValue, high: MetadataValue) -> Self {
        FilterExpression::Between(field.into(), low, high)
    }

    pub fn exists(field: impl Into<String>) -> Self {
        FilterExpression::Exists(field.into())
    }

    pub fn contains(field: impl Into<String>, value: impl Into<String>) -> Self {
        FilterExpression::Contains(field.into(), value.into())
    }

    /// Compile this filter expression into a flat instruction list for
    /// efficient repeated evaluation. Avoids recursive AST traversal.
    pub fn compile(&self) -> crate::eval::CompiledFilter {
        crate::eval::CompiledFilter::compile(self)
    }

    pub fn referenced_fields(&self) -> Vec<&str> {
        let mut fields = Vec::new();
        self.collect_fields(&mut fields);
        fields.sort_unstable();
        fields.dedup();
        fields
    }

    fn collect_fields<'a>(&'a self, fields: &mut Vec<&'a str>) {
        match self {
            FilterExpression::And(children) | FilterExpression::Or(children) => {
                for child in children {
                    child.collect_fields(fields);
                }
            }
            FilterExpression::Not(child) => child.collect_fields(fields),
            FilterExpression::Eq(f, _)
            | FilterExpression::Ne(f, _)
            | FilterExpression::Gt(f, _)
            | FilterExpression::Gte(f, _)
            | FilterExpression::Lt(f, _)
            | FilterExpression::Lte(f, _) => fields.push(f.as_str()),
            FilterExpression::In(f, _) | FilterExpression::Between(f, _, _) => {
                fields.push(f.as_str())
            }
            FilterExpression::Exists(f) | FilterExpression::Contains(f, _) => {
                fields.push(f.as_str())
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("field not found: {0}")]
    FieldNotFound(String),

    #[error("type mismatch for field {field}: expected {expected}, got {got}")]
    TypeMismatch {
        field: String,
        expected: String,
        got: String,
    },

    #[error("index error: {0}")]
    IndexError(#[from] vf_index::traits::IndexError),

    #[error("internal error: {0}")]
    Internal(String),
}
