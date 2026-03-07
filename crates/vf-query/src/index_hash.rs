// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;

use roaring::RoaringBitmap;
use vf_core::types::{MetadataValue, VectorId};

/// Hash key for O(1) equality lookups.
/// Float is excluded because f64 does not implement Hash/Eq.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HashKey {
    Int(i64),
    String(String),
    Bool(bool),
}

impl HashKey {
    /// Convert a MetadataValue to a HashKey.
    /// Returns None for Float and StringList (not supported by hash index).
    pub fn from_metadata(value: &MetadataValue) -> Option<Self> {
        match value {
            MetadataValue::Int(v) => Some(HashKey::Int(*v)),
            MetadataValue::String(v) => Some(HashKey::String(v.clone())),
            MetadataValue::Bool(v) => Some(HashKey::Bool(*v)),
            MetadataValue::Float(_) | MetadataValue::StringList(_) => None,
        }
    }
}

/// Hash-based metadata index for O(1) equality lookups on string and categorical fields.
pub struct HashIndex {
    field_name: String,
    map: HashMap<HashKey, RoaringBitmap>,
    // VectorId is u64 but RoaringBitmap uses u32. We truncate via `id as u32`,
    // limiting this index to ~4 billion vectors (acceptable for v1).
    id_to_key: HashMap<u32, HashKey>,
}

impl HashIndex {
    pub fn new(field_name: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
            map: HashMap::new(),
            id_to_key: HashMap::new(),
        }
    }

    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Insert a vector ID with its metadata value for this field.
    pub fn insert(&mut self, id: VectorId, value: &MetadataValue) {
        let Some(key) = HashKey::from_metadata(value) else {
            return;
        };
        let id32 = id as u32;

        // Remove old mapping if present
        if let Some(old_key) = self.id_to_key.remove(&id32) {
            if let Some(bitmap) = self.map.get_mut(&old_key) {
                bitmap.remove(id32);
                if bitmap.is_empty() {
                    self.map.remove(&old_key);
                }
            }
        }

        self.map
            .entry(key.clone())
            .or_insert_with(RoaringBitmap::new)
            .insert(id32);
        self.id_to_key.insert(id32, key);
    }

    /// Remove a vector ID from the index.
    pub fn remove(&mut self, id: VectorId) {
        let id32 = id as u32;
        if let Some(key) = self.id_to_key.remove(&id32) {
            if let Some(bitmap) = self.map.get_mut(&key) {
                bitmap.remove(id32);
                if bitmap.is_empty() {
                    self.map.remove(&key);
                }
            }
        }
    }

    /// O(1) exact match lookup.
    pub fn eq_lookup(&self, value: &MetadataValue) -> RoaringBitmap {
        let Some(key) = HashKey::from_metadata(value) else {
            return RoaringBitmap::new();
        };
        self.map
            .get(&key)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if a value exists in the index (O(1)).
    pub fn contains_value(&self, value: &MetadataValue) -> bool {
        let Some(key) = HashKey::from_metadata(value) else {
            return false;
        };
        self.map.contains_key(&key)
    }

    /// Number of unique values indexed.
    pub fn cardinality(&self) -> usize {
        self.map.len()
    }

    /// Total number of indexed vector IDs.
    pub fn len(&self) -> usize {
        self.id_to_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_key.is_empty()
    }

    /// IN query: union of multiple eq_lookups.
    pub fn in_lookup(&self, values: &[MetadataValue]) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for value in values {
            if let Some(key) = HashKey::from_metadata(value) {
                if let Some(bitmap) = self.map.get(&key) {
                    result |= bitmap;
                }
            }
        }
        result
    }
}
