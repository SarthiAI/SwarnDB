// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use crate::filter::FilterExpression;
use std::cmp::Ordering;
use std::collections::HashMap;
use vf_core::types::{Metadata, MetadataValue, VectorId};

/// A single flat operation in the compiled filter instruction list.
/// The stack-based evaluator processes these sequentially, pushing/popping
/// boolean results on an evaluation stack.
#[derive(Clone, Debug)]
pub enum FilterOp {
    /// Push result of field == value
    Eq(String, MetadataValue),
    /// Push result of field != value
    Ne(String, MetadataValue),
    /// Push result of field > value
    Gt(String, MetadataValue),
    /// Push result of field >= value
    Gte(String, MetadataValue),
    /// Push result of field < value
    Lt(String, MetadataValue),
    /// Push result of field <= value
    Lte(String, MetadataValue),
    /// Push result of field IN values
    In(String, Vec<MetadataValue>),
    /// Push result of low <= field <= high
    Between(String, MetadataValue, MetadataValue),
    /// Push result of field exists in metadata
    Exists(String),
    /// Push result of field contains substring/element
    Contains(String, String),
    /// Pop top, push !top
    Not,
    /// Pop N values, push AND of all
    And(usize),
    /// Pop N values, push OR of all
    Or(usize),
}

/// Pre-compiled filter that avoids repeated AST tree traversal.
/// Uses a flat instruction list evaluated with a boolean stack.
#[derive(Clone, Debug)]
pub struct CompiledFilter {
    ops: Vec<FilterOp>,
}

impl CompiledFilter {
    /// Compile a FilterExpression into a flat instruction list.
    pub fn compile(expr: &FilterExpression) -> Self {
        let mut ops = Vec::new();
        Self::emit(expr, &mut ops);
        CompiledFilter { ops }
    }

    /// Recursively emit ops for an expression (post-order: children first, then combinator).
    fn emit(expr: &FilterExpression, ops: &mut Vec<FilterOp>) {
        match expr {
            FilterExpression::And(children) => {
                let count = children.len();
                for child in children {
                    Self::emit(child, ops);
                }
                ops.push(FilterOp::And(count));
            }
            FilterExpression::Or(children) => {
                let count = children.len();
                for child in children {
                    Self::emit(child, ops);
                }
                ops.push(FilterOp::Or(count));
            }
            FilterExpression::Not(child) => {
                Self::emit(child, ops);
                ops.push(FilterOp::Not);
            }
            FilterExpression::Eq(f, v) => ops.push(FilterOp::Eq(f.clone(), v.clone())),
            FilterExpression::Ne(f, v) => ops.push(FilterOp::Ne(f.clone(), v.clone())),
            FilterExpression::Gt(f, v) => ops.push(FilterOp::Gt(f.clone(), v.clone())),
            FilterExpression::Gte(f, v) => ops.push(FilterOp::Gte(f.clone(), v.clone())),
            FilterExpression::Lt(f, v) => ops.push(FilterOp::Lt(f.clone(), v.clone())),
            FilterExpression::Lte(f, v) => ops.push(FilterOp::Lte(f.clone(), v.clone())),
            FilterExpression::In(f, vs) => ops.push(FilterOp::In(f.clone(), vs.clone())),
            FilterExpression::Between(f, lo, hi) => {
                ops.push(FilterOp::Between(f.clone(), lo.clone(), hi.clone()))
            }
            FilterExpression::Exists(f) => ops.push(FilterOp::Exists(f.clone())),
            FilterExpression::Contains(f, s) => ops.push(FilterOp::Contains(f.clone(), s.clone())),
        }
    }

    /// Evaluate the compiled filter against a single metadata map.
    /// Uses a stack-based evaluator - no recursion during evaluation.
    #[inline]
    pub fn evaluate(&self, metadata: &Metadata) -> bool {
        let mut stack: Vec<bool> = Vec::with_capacity(self.ops.len());

        for op in &self.ops {
            match op {
                FilterOp::Eq(field, value) => {
                    let result = metadata
                        .get(field.as_str())
                        .map_or(false, |v| values_equal(v, value));
                    stack.push(result);
                }
                FilterOp::Ne(field, value) => {
                    let result = metadata
                        .get(field.as_str())
                        .map_or(false, |v| !values_equal(v, value));
                    stack.push(result);
                }
                FilterOp::Gt(field, value) => {
                    let result = metadata
                        .get(field.as_str())
                        .and_then(|v| compare_values(v, value))
                        .map_or(false, |ord| ord == Ordering::Greater);
                    stack.push(result);
                }
                FilterOp::Gte(field, value) => {
                    let result = metadata
                        .get(field.as_str())
                        .and_then(|v| compare_values(v, value))
                        .map_or(false, |ord| ord != Ordering::Less);
                    stack.push(result);
                }
                FilterOp::Lt(field, value) => {
                    let result = metadata
                        .get(field.as_str())
                        .and_then(|v| compare_values(v, value))
                        .map_or(false, |ord| ord == Ordering::Less);
                    stack.push(result);
                }
                FilterOp::Lte(field, value) => {
                    let result = metadata
                        .get(field.as_str())
                        .and_then(|v| compare_values(v, value))
                        .map_or(false, |ord| ord != Ordering::Greater);
                    stack.push(result);
                }
                FilterOp::In(field, values) => {
                    let result = metadata.get(field.as_str()).map_or(false, |v| {
                        values.iter().any(|candidate| values_equal(v, candidate))
                    });
                    stack.push(result);
                }
                FilterOp::Between(field, low, high) => {
                    let result = metadata.get(field.as_str()).map_or(false, |v| {
                        let ge_low = compare_values(v, low).map_or(false, |o| o != Ordering::Less);
                        let le_high =
                            compare_values(v, high).map_or(false, |o| o != Ordering::Greater);
                        ge_low && le_high
                    });
                    stack.push(result);
                }
                FilterOp::Exists(field) => {
                    stack.push(metadata.contains_key(field.as_str()));
                }
                FilterOp::Contains(field, target) => {
                    let result = match metadata.get(field.as_str()) {
                        Some(MetadataValue::StringList(list)) => list.iter().any(|s| s == target),
                        Some(MetadataValue::String(s)) => s.contains(target.as_str()),
                        _ => false,
                    };
                    stack.push(result);
                }
                FilterOp::Not => {
                    if let Some(top) = stack.last_mut() {
                        *top = !*top;
                    }
                }
                FilterOp::And(count) => {
                    let len = stack.len();
                    let start = len.saturating_sub(*count);
                    let result = stack[start..].iter().all(|&v| v);
                    stack.truncate(start);
                    stack.push(result);
                }
                FilterOp::Or(count) => {
                    let len = stack.len();
                    let start = len.saturating_sub(*count);
                    let result = stack[start..].iter().any(|&v| v);
                    stack.truncate(start);
                    stack.push(result);
                }
            }
        }

        stack.last().copied().unwrap_or(false)
    }

    /// Batch evaluate compiled filter against a metadata store.
    pub fn evaluate_batch(&self, records: &HashMap<VectorId, Metadata>) -> Vec<VectorId> {
        records
            .iter()
            .filter(|(_, meta)| self.evaluate(meta))
            .map(|(&id, _)| id)
            .collect()
    }
}

pub struct FilterEvaluator;

impl FilterEvaluator {
    /// Evaluate a filter expression against metadata (uncompiled, tree-walking path).
    pub fn evaluate(filter: &FilterExpression, metadata: &Metadata) -> bool {
        match filter {
            FilterExpression::And(children) => children.iter().all(|c| Self::evaluate(c, metadata)),
            FilterExpression::Or(children) => children.iter().any(|c| Self::evaluate(c, metadata)),
            FilterExpression::Not(child) => !Self::evaluate(child, metadata),

            FilterExpression::Eq(field, value) => metadata
                .get(field)
                .map_or(false, |v| values_equal(v, value)),

            FilterExpression::Ne(field, value) => metadata
                .get(field)
                .map_or(false, |v| !values_equal(v, value)),

            FilterExpression::Gt(field, value) => metadata
                .get(field)
                .and_then(|v| compare_values(v, value))
                .map_or(false, |ord| ord == Ordering::Greater),

            FilterExpression::Gte(field, value) => metadata
                .get(field)
                .and_then(|v| compare_values(v, value))
                .map_or(false, |ord| ord != Ordering::Less),

            FilterExpression::Lt(field, value) => metadata
                .get(field)
                .and_then(|v| compare_values(v, value))
                .map_or(false, |ord| ord == Ordering::Less),

            FilterExpression::Lte(field, value) => metadata
                .get(field)
                .and_then(|v| compare_values(v, value))
                .map_or(false, |ord| ord != Ordering::Greater),

            FilterExpression::In(field, values) => metadata.get(field).map_or(false, |v| {
                values.iter().any(|candidate| values_equal(v, candidate))
            }),

            FilterExpression::Between(field, low, high) => metadata.get(field).map_or(false, |v| {
                let ge_low = compare_values(v, low).map_or(false, |o| o != Ordering::Less);
                let le_high = compare_values(v, high).map_or(false, |o| o != Ordering::Greater);
                ge_low && le_high
            }),

            FilterExpression::Exists(field) => metadata.contains_key(field),

            FilterExpression::Contains(field, target) => match metadata.get(field) {
                Some(MetadataValue::StringList(list)) => list.iter().any(|s| s == target),
                Some(MetadataValue::String(s)) => s.contains(target.as_str()),
                _ => false,
            },
        }
    }

    /// Batch evaluate filter against a metadata store (uncompiled path).
    pub fn evaluate_batch(
        filter: &FilterExpression,
        records: &HashMap<VectorId, Metadata>,
    ) -> Vec<VectorId> {
        records
            .iter()
            .filter(|(_, meta)| Self::evaluate(filter, meta))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Compile a filter expression for repeated evaluation.
    pub fn compile(filter: &FilterExpression) -> CompiledFilter {
        CompiledFilter::compile(filter)
    }

    /// Evaluate using a pre-compiled filter (avoids AST traversal).
    pub fn evaluate_compiled(compiled: &CompiledFilter, metadata: &Metadata) -> bool {
        compiled.evaluate(metadata)
    }

    /// Batch evaluate using a pre-compiled filter.
    pub fn evaluate_batch_compiled(
        compiled: &CompiledFilter,
        records: &HashMap<VectorId, Metadata>,
    ) -> Vec<VectorId> {
        compiled.evaluate_batch(records)
    }
}

fn values_equal(a: &MetadataValue, b: &MetadataValue) -> bool {
    if a == b {
        return true;
    }
    match (a, b) {
        (MetadataValue::Int(ai), MetadataValue::Float(bf)) => (*ai as f64) == *bf,
        (MetadataValue::Float(af), MetadataValue::Int(bi)) => *af == (*bi as f64),
        _ => false,
    }
}

fn compare_values(a: &MetadataValue, b: &MetadataValue) -> Option<Ordering> {
    match (a, b) {
        (MetadataValue::Int(ai), MetadataValue::Int(bi)) => ai.partial_cmp(bi),
        (MetadataValue::Float(af), MetadataValue::Float(bf)) => af.partial_cmp(bf),
        (MetadataValue::Int(ai), MetadataValue::Float(bf)) => (*ai as f64).partial_cmp(bf),
        (MetadataValue::Float(af), MetadataValue::Int(bi)) => af.partial_cmp(&(*bi as f64)),
        (MetadataValue::String(sa), MetadataValue::String(sb)) => Some(sa.cmp(sb)),
        _ => None,
    }
}
