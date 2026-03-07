// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;

use roaring::RoaringBitmap;
use vf_core::types::{MetadataValue, VectorId};

/// Key type for bitmap index — discrete, hashable values only.
/// Float values are excluded (f64 is not hashable); use BTreeIndex for floats.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BitmapKey {
    Int(i64),
    String(String),
    Bool(bool),
}

/// RoaringBitmap-based metadata index for low-cardinality fields.
///
/// Each unique value gets its own bitmap. Set operations (AND/OR/NOT)
/// enable efficient multi-filter evaluation.
///
/// Note: RoaringBitmap uses u32 internally. VectorId (u64) is truncated
/// via `id as u32`. This limits the index to ~4 billion IDs, which is
/// acceptable for v1.
pub struct BitmapIndex {
    field_name: String,
    bitmaps: HashMap<BitmapKey, RoaringBitmap>,
    all_ids: RoaringBitmap,
    /// Reverse mapping: ID -> keys it was inserted under, for clean re-insert and targeted removal.
    id_to_keys: HashMap<u32, Vec<BitmapKey>>,
}

impl BitmapIndex {
    pub fn new(field_name: impl Into<String>) -> Self {
        Self {
            field_name: field_name.into(),
            bitmaps: HashMap::new(),
            all_ids: RoaringBitmap::new(),
            id_to_keys: HashMap::new(),
        }
    }

    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Insert a vector ID under the given metadata value.
    /// For `StringList` values, the ID is inserted into a separate bitmap for each element.
    /// Re-inserting an ID with a different value cleans up stale entries first.
    pub fn insert(&mut self, id: VectorId, value: &MetadataValue) {
        let id32 = id as u32;

        // Remove old keys if this ID was previously indexed (Bug 1: reverse mapping)
        if let Some(old_keys) = self.id_to_keys.remove(&id32) {
            for key in &old_keys {
                if let Some(bm) = self.bitmaps.get_mut(key) {
                    bm.remove(id32);
                    if bm.is_empty() {
                        self.bitmaps.remove(key);
                    }
                }
            }
            self.all_ids.remove(id32);
        }

        match value {
            MetadataValue::StringList(list) => {
                if !list.is_empty() {
                    let mut keys = Vec::with_capacity(list.len());
                    for s in list {
                        let key = BitmapKey::String(s.clone());
                        self.bitmaps.entry(key.clone()).or_default().insert(id32);
                        keys.push(key);
                    }
                    self.all_ids.insert(id32);
                    self.id_to_keys.insert(id32, keys);
                }
            }
            _ => {
                if let Some(key) = Self::to_key(value) {
                    self.bitmaps.entry(key.clone()).or_default().insert(id32);
                    self.all_ids.insert(id32);
                    self.id_to_keys.insert(id32, vec![key]);
                }
            }
        }
    }

    /// Remove a vector ID from all bitmaps.
    pub fn remove(&mut self, id: VectorId) {
        let id32 = id as u32;
        self.all_ids.remove(id32);

        // Use reverse map to only touch relevant bitmaps (Bug 1)
        if let Some(old_keys) = self.id_to_keys.remove(&id32) {
            for key in &old_keys {
                if let Some(bm) = self.bitmaps.get_mut(key) {
                    bm.remove(id32);
                }
            }
        } else {
            // Fallback: ID was inserted before reverse map existed
            for bitmap in self.bitmaps.values_mut() {
                bitmap.remove(id32);
            }
        }

        // Prune empty bitmaps to prevent memory leaks (Bug 3)
        self.bitmaps.retain(|_, bm| !bm.is_empty());
    }

    /// Exact equality lookup — returns the bitmap for a single value.
    pub fn eq_lookup(&self, value: &MetadataValue) -> RoaringBitmap {
        match Self::to_key(value) {
            Some(key) => self.bitmaps.get(&key).cloned().unwrap_or_default(),
            None => RoaringBitmap::new(),
        }
    }

    /// Not-equal lookup — all indexed IDs minus those matching `value`.
    pub fn ne_lookup(&self, value: &MetadataValue) -> RoaringBitmap {
        let eq = self.eq_lookup(value);
        &self.all_ids - &eq
    }

    /// IN query — union of bitmaps for each value in the list.
    pub fn in_lookup(&self, values: &[MetadataValue]) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for value in values {
            result |= self.eq_lookup(value);
        }
        result
    }

    /// Contains lookup for StringList fields.
    /// Since each string element is indexed individually, this is equivalent
    /// to an eq_lookup with a String value.
    pub fn contains_lookup(&self, value: &str) -> RoaringBitmap {
        let key = BitmapKey::String(value.to_owned());
        self.bitmaps.get(&key).cloned().unwrap_or_default()
    }

    /// Returns a reference to the bitmap of all indexed IDs.
    pub fn all(&self) -> &RoaringBitmap {
        &self.all_ids
    }

    /// Number of unique values tracked by this index.
    pub fn cardinality(&self) -> usize {
        self.bitmaps.len()
    }

    /// Total number of indexed vector IDs.
    pub fn len(&self) -> usize {
        self.all_ids.len() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.all_ids.is_empty()
    }

    /// Convert a `MetadataValue` to a `BitmapKey`.
    /// Returns `None` for Float and StringList (StringList is handled separately).
    fn to_key(value: &MetadataValue) -> Option<BitmapKey> {
        match value {
            MetadataValue::Int(v) => Some(BitmapKey::Int(*v)),
            MetadataValue::String(v) => Some(BitmapKey::String(v.clone())),
            MetadataValue::Bool(v) => Some(BitmapKey::Bool(*v)),
            MetadataValue::Float(_) | MetadataValue::StringList(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_eq_lookup() {
        let mut idx = BitmapIndex::new("category");
        idx.insert(1, &MetadataValue::String("sports".into()));
        idx.insert(2, &MetadataValue::String("sports".into()));
        idx.insert(3, &MetadataValue::String("news".into()));

        let result = idx.eq_lookup(&MetadataValue::String("sports".into()));
        assert_eq!(result.len(), 2);
        assert!(result.contains(1));
        assert!(result.contains(2));
    }

    #[test]
    fn test_ne_lookup() {
        let mut idx = BitmapIndex::new("status");
        idx.insert(1, &MetadataValue::Bool(true));
        idx.insert(2, &MetadataValue::Bool(false));
        idx.insert(3, &MetadataValue::Bool(true));

        let result = idx.ne_lookup(&MetadataValue::Bool(true));
        assert_eq!(result.len(), 1);
        assert!(result.contains(2));
    }

    #[test]
    fn test_in_lookup() {
        let mut idx = BitmapIndex::new("priority");
        idx.insert(1, &MetadataValue::Int(1));
        idx.insert(2, &MetadataValue::Int(2));
        idx.insert(3, &MetadataValue::Int(3));

        let result = idx.in_lookup(&[MetadataValue::Int(1), MetadataValue::Int(3)]);
        assert_eq!(result.len(), 2);
        assert!(result.contains(1));
        assert!(result.contains(3));
    }

    #[test]
    fn test_string_list_insert_and_contains() {
        let mut idx = BitmapIndex::new("tags");
        idx.insert(1, &MetadataValue::StringList(vec!["a".into(), "b".into()]));
        idx.insert(2, &MetadataValue::StringList(vec!["b".into(), "c".into()]));

        assert_eq!(idx.contains_lookup("a").len(), 1);
        assert_eq!(idx.contains_lookup("b").len(), 2);
        assert_eq!(idx.contains_lookup("c").len(), 1);
        assert_eq!(idx.contains_lookup("d").len(), 0);
    }

    #[test]
    fn test_remove() {
        let mut idx = BitmapIndex::new("field");
        idx.insert(1, &MetadataValue::String("x".into()));
        idx.insert(2, &MetadataValue::String("x".into()));
        assert_eq!(idx.len(), 2);

        idx.remove(1);
        assert_eq!(idx.len(), 1);
        assert!(!idx.eq_lookup(&MetadataValue::String("x".into())).contains(1));
        assert!(idx.eq_lookup(&MetadataValue::String("x".into())).contains(2));
    }

    #[test]
    fn test_float_ignored() {
        let mut idx = BitmapIndex::new("score");
        idx.insert(1, &MetadataValue::Float(3.14));

        assert_eq!(idx.len(), 0);
        assert_eq!(idx.cardinality(), 0);
        assert!(idx.eq_lookup(&MetadataValue::Float(3.14)).is_empty());
    }

    #[test]
    fn test_cardinality_and_len() {
        let mut idx = BitmapIndex::new("cat");
        assert!(idx.is_empty());
        assert_eq!(idx.cardinality(), 0);

        idx.insert(1, &MetadataValue::String("a".into()));
        idx.insert(2, &MetadataValue::String("b".into()));
        idx.insert(3, &MetadataValue::String("a".into()));

        assert_eq!(idx.len(), 3);
        assert_eq!(idx.cardinality(), 2);
        assert!(!idx.is_empty());
    }
}
