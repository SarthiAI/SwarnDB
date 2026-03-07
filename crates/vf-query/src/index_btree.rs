// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;
use std::collections::BTreeMap;
use std::ops::Bound;

use ordered_float::OrderedFloat;
use roaring::RoaringBitmap;
use vf_core::types::{MetadataValue, VectorId};

/// Wrapper for MetadataValue that implements Ord (needed for BTreeMap keys).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IndexKey {
    Bool(bool),
    Int(i64),
    Float(OrderedFloat<f64>),
    String(String),
}

impl IndexKey {
    /// Convert a MetadataValue to an IndexKey.
    /// Returns None for StringList (not indexable as a single key).
    pub fn from_metadata(value: &MetadataValue) -> Option<Self> {
        match value {
            MetadataValue::Int(v) => Some(IndexKey::Int(*v)),
            MetadataValue::Float(v) => Some(IndexKey::Float(OrderedFloat(*v))),
            MetadataValue::String(v) => Some(IndexKey::String(v.clone())),
            MetadataValue::Bool(v) => Some(IndexKey::Bool(*v)),
            MetadataValue::StringList(_) => None,
        }
    }
}

/// Sorted B-tree index for numeric and string fields supporting range queries.
pub struct BTreeIndex {
    field_name: String,
    tree: BTreeMap<IndexKey, RoaringBitmap>,
    // VectorId is u64 but RoaringBitmap uses u32. We truncate via `id as u32`,
    // limiting this index to ~4 billion vectors (acceptable for v1).
    id_to_key: HashMap<u32, IndexKey>,
}

impl BTreeIndex {
    pub fn new(field_name: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
            tree: BTreeMap::new(),
            id_to_key: HashMap::new(),
        }
    }

    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Insert a vector ID with its metadata value for this field.
    pub fn insert(&mut self, id: VectorId, value: &MetadataValue) {
        let Some(key) = IndexKey::from_metadata(value) else {
            return;
        };
        let id32 = id as u32;

        // Remove old mapping if present
        if let Some(old_key) = self.id_to_key.remove(&id32) {
            if let Some(bitmap) = self.tree.get_mut(&old_key) {
                bitmap.remove(id32);
                if bitmap.is_empty() {
                    self.tree.remove(&old_key);
                }
            }
        }

        self.tree
            .entry(key.clone())
            .or_insert_with(RoaringBitmap::new)
            .insert(id32);
        self.id_to_key.insert(id32, key);
    }

    /// Remove a vector ID from the index.
    pub fn remove(&mut self, id: VectorId) {
        let id32 = id as u32;
        if let Some(key) = self.id_to_key.remove(&id32) {
            if let Some(bitmap) = self.tree.get_mut(&key) {
                bitmap.remove(id32);
                if bitmap.is_empty() {
                    self.tree.remove(&key);
                }
            }
        }
    }

    /// Exact equality lookup.
    pub fn eq_lookup(&self, value: &MetadataValue) -> RoaringBitmap {
        let Some(key) = IndexKey::from_metadata(value) else {
            return RoaringBitmap::new();
        };
        self.tree
            .get(&key)
            .cloned()
            .unwrap_or_default()
    }

    /// Range query: all IDs where field value is in [low, high] (inclusive).
    pub fn range(&self, low: &MetadataValue, high: &MetadataValue) -> RoaringBitmap {
        let (Some(low_key), Some(high_key)) = (
            IndexKey::from_metadata(low),
            IndexKey::from_metadata(high),
        ) else {
            return RoaringBitmap::new();
        };
        let mut result = RoaringBitmap::new();
        for (_, bitmap) in self.tree.range(low_key..=high_key) {
            result |= bitmap;
        }
        result
    }

    /// Greater than.
    pub fn gt(&self, value: &MetadataValue) -> RoaringBitmap {
        let Some(key) = IndexKey::from_metadata(value) else {
            return RoaringBitmap::new();
        };
        let mut result = RoaringBitmap::new();
        for (_, bitmap) in self.tree.range((Bound::Excluded(key), Bound::Unbounded)) {
            result |= bitmap;
        }
        result
    }

    /// Greater than or equal.
    pub fn gte(&self, value: &MetadataValue) -> RoaringBitmap {
        let Some(key) = IndexKey::from_metadata(value) else {
            return RoaringBitmap::new();
        };
        let mut result = RoaringBitmap::new();
        for (_, bitmap) in self.tree.range(key..) {
            result |= bitmap;
        }
        result
    }

    /// Less than.
    pub fn lt(&self, value: &MetadataValue) -> RoaringBitmap {
        let Some(key) = IndexKey::from_metadata(value) else {
            return RoaringBitmap::new();
        };
        let mut result = RoaringBitmap::new();
        for (_, bitmap) in self.tree.range(..key) {
            result |= bitmap;
        }
        result
    }

    /// Less than or equal.
    pub fn lte(&self, value: &MetadataValue) -> RoaringBitmap {
        let Some(key) = IndexKey::from_metadata(value) else {
            return RoaringBitmap::new();
        };
        let mut result = RoaringBitmap::new();
        for (_, bitmap) in self.tree.range(..=key) {
            result |= bitmap;
        }
        result
    }

    /// Number of unique values indexed.
    pub fn cardinality(&self) -> usize {
        self.tree.len()
    }

    /// Total number of indexed vector IDs.
    pub fn len(&self) -> usize {
        self.id_to_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_key.is_empty()
    }
}
