// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use crate::filter::FilterExpression;
use std::cmp::Ordering;
use std::collections::HashMap;
use vf_core::types::{Metadata, MetadataValue, VectorId};

pub struct FilterEvaluator;

impl FilterEvaluator {
    pub fn evaluate(filter: &FilterExpression, metadata: &Metadata) -> bool {
        match filter {
            FilterExpression::And(children) => {
                children.iter().all(|c| Self::evaluate(c, metadata))
            }
            FilterExpression::Or(children) => {
                children.iter().any(|c| Self::evaluate(c, metadata))
            }
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

            FilterExpression::In(field, values) => metadata
                .get(field)
                .map_or(false, |v| values.iter().any(|candidate| values_equal(v, candidate))),

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
