// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;

use roaring::RoaringBitmap;
use vf_core::types::{Metadata, MetadataValue, VectorId};

use crate::filter::FilterExpression;
use crate::index_bitmap::BitmapIndex;
use crate::index_btree::BTreeIndex;
use crate::index_hash::HashIndex;

pub enum MetadataIndex {
    BTree(BTreeIndex),
    Hash(HashIndex),
    Bitmap(BitmapIndex),
}

pub struct IndexConfig {
    pub bitmap_cardinality_threshold: usize,
    pub btree_cardinality_threshold: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            bitmap_cardinality_threshold: 100,
            btree_cardinality_threshold: 10_000,
        }
    }
}

struct FieldStats {
    total_values: usize,
    unique_values: usize,
}

pub struct IndexManager {
    indexes: HashMap<String, MetadataIndex>,
    config: IndexConfig,
    field_stats: HashMap<String, FieldStats>,
    records: HashMap<u32, HashMap<String, MetadataValue>>,
}

impl IndexManager {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            indexes: HashMap::new(),
            config,
            field_stats: HashMap::new(),
            records: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(IndexConfig::default())
    }

    pub fn index_record(&mut self, id: VectorId, metadata: &Metadata) {
        let id32 = id as u32;
        for (field, value) in metadata {
            if !self.indexes.contains_key(field) {
                self.indexes.insert(
                    field.clone(),
                    MetadataIndex::Bitmap(BitmapIndex::new(field.clone())),
                );
                self.field_stats.insert(
                    field.clone(),
                    FieldStats {
                        total_values: 0,
                        unique_values: 0,
                    },
                );
            }

            match self.indexes.get_mut(field).unwrap() {
                MetadataIndex::BTree(idx) => idx.insert(id, value),
                MetadataIndex::Hash(idx) => idx.insert(id, value),
                MetadataIndex::Bitmap(idx) => idx.insert(id, value),
            }

            self.records
                .entry(id32)
                .or_default()
                .insert(field.clone(), value.clone());

            self.update_field_stats(field);
        }
    }

    pub fn remove_record(&mut self, id: VectorId) {
        let id32 = id as u32;
        for index in self.indexes.values_mut() {
            match index {
                MetadataIndex::BTree(idx) => idx.remove(id),
                MetadataIndex::Hash(idx) => idx.remove(id),
                MetadataIndex::Bitmap(idx) => idx.remove(id),
            }
        }
        if let Some(fields) = self.records.remove(&id32) {
            for field in fields.keys() {
                self.update_field_stats(field);
            }
        }
    }

    pub fn get_index(&self, field: &str) -> Option<&MetadataIndex> {
        self.indexes.get(field)
    }

    pub fn evaluate_filter(&self, filter: &FilterExpression) -> Option<RoaringBitmap> {
        match filter {
            FilterExpression::Eq(field, value) => {
                let idx = self.indexes.get(field)?;
                Some(match idx {
                    MetadataIndex::BTree(i) => i.eq_lookup(value),
                    MetadataIndex::Hash(i) => i.eq_lookup(value),
                    MetadataIndex::Bitmap(i) => i.eq_lookup(value),
                })
            }
            FilterExpression::Ne(field, value) => {
                let idx = self.indexes.get(field)?;
                match idx {
                    MetadataIndex::Bitmap(i) => Some(i.ne_lookup(value)),
                    _ => None,
                }
            }
            FilterExpression::Gt(field, value) => {
                let idx = self.indexes.get(field)?;
                match idx {
                    MetadataIndex::BTree(i) => Some(i.gt(value)),
                    _ => None,
                }
            }
            FilterExpression::Gte(field, value) => {
                let idx = self.indexes.get(field)?;
                match idx {
                    MetadataIndex::BTree(i) => Some(i.gte(value)),
                    _ => None,
                }
            }
            FilterExpression::Lt(field, value) => {
                let idx = self.indexes.get(field)?;
                match idx {
                    MetadataIndex::BTree(i) => Some(i.lt(value)),
                    _ => None,
                }
            }
            FilterExpression::Lte(field, value) => {
                let idx = self.indexes.get(field)?;
                match idx {
                    MetadataIndex::BTree(i) => Some(i.lte(value)),
                    _ => None,
                }
            }
            FilterExpression::Between(field, low, high) => {
                let idx = self.indexes.get(field)?;
                match idx {
                    MetadataIndex::BTree(i) => Some(i.range(low, high)),
                    _ => None,
                }
            }
            FilterExpression::In(field, values) => {
                let idx = self.indexes.get(field)?;
                match idx {
                    MetadataIndex::Hash(i) => Some(i.in_lookup(values)),
                    MetadataIndex::Bitmap(i) => Some(i.in_lookup(values)),
                    MetadataIndex::BTree(i) => {
                        let mut result = RoaringBitmap::new();
                        for v in values {
                            result |= i.eq_lookup(v);
                        }
                        Some(result)
                    }
                }
            }
            FilterExpression::Exists(_) => None,
            FilterExpression::Contains(field, value) => {
                let idx = self.indexes.get(field)?;
                match idx {
                    MetadataIndex::Bitmap(i) => Some(i.contains_lookup(value)),
                    _ => None,
                }
            }
            FilterExpression::And(children) => {
                // Empty AND is vacuously true (match all); return None since
                // IndexManager doesn't track the full ID universe.
                if children.is_empty() {
                    return None;
                }
                let mut result: Option<RoaringBitmap> = None;
                for child in children {
                    let bitmap = self.evaluate_filter(child)?;
                    result = Some(match result {
                        Some(acc) => acc & bitmap,
                        None => bitmap,
                    });
                }
                result.or_else(|| Some(RoaringBitmap::new()))
            }
            FilterExpression::Or(children) => {
                let mut result = RoaringBitmap::new();
                for child in children {
                    let bitmap = self.evaluate_filter(child)?;
                    result |= bitmap;
                }
                Some(result)
            }
            FilterExpression::Not(_) => None,
        }
    }

    pub fn rebuild_field_index(&mut self, field: &str) {
        let stats = match self.field_stats.get(field) {
            Some(s) => s,
            None => return,
        };

        let target = self.select_index_type(stats.unique_values);
        let current = match self.indexes.get(field) {
            Some(idx) => idx,
            None => return,
        };

        let needs_rebuild = !matches!(
            (&target, current),
            (IndexType::Bitmap, MetadataIndex::Bitmap(_))
                | (IndexType::Hash, MetadataIndex::Hash(_))
                | (IndexType::BTree, MetadataIndex::BTree(_))
        );

        if !needs_rebuild {
            return;
        }

        let new_index: MetadataIndex = match target {
            IndexType::Bitmap => MetadataIndex::Bitmap(BitmapIndex::new(field)),
            IndexType::Hash => MetadataIndex::Hash(HashIndex::new(field)),
            IndexType::BTree => MetadataIndex::BTree(BTreeIndex::new(field)),
        };
        self.indexes.insert(field.to_owned(), new_index);

        for (id32, fields) in &self.records {
            if let Some(value) = fields.get(field) {
                let id = *id32 as VectorId;
                match self.indexes.get_mut(field).unwrap() {
                    MetadataIndex::BTree(idx) => idx.insert(id, value),
                    MetadataIndex::Hash(idx) => idx.insert(id, value),
                    MetadataIndex::Bitmap(idx) => idx.insert(id, value),
                }
            }
        }
    }

    pub fn maybe_rebuild_indexes(&mut self) {
        let fields_to_rebuild: Vec<String> = self
            .field_stats
            .iter()
            .filter_map(|(field, stats)| {
                let target = self.select_index_type(stats.unique_values);
                let current = self.indexes.get(field)?;
                let matches = matches!(
                    (&target, current),
                    (IndexType::Bitmap, MetadataIndex::Bitmap(_))
                        | (IndexType::Hash, MetadataIndex::Hash(_))
                        | (IndexType::BTree, MetadataIndex::BTree(_))
                );
                if matches {
                    None
                } else {
                    Some(field.clone())
                }
            })
            .collect();

        for field in &fields_to_rebuild {
            self.rebuild_field_index(field);
        }
    }

    pub fn field_count(&self) -> usize {
        self.indexes.len()
    }

    pub fn fields(&self) -> Vec<&str> {
        self.indexes.keys().map(|s| s.as_str()).collect()
    }

    fn update_field_stats(&mut self, field: &str) {
        let unique = match self.indexes.get(field) {
            Some(MetadataIndex::BTree(idx)) => idx.cardinality(),
            Some(MetadataIndex::Hash(idx)) => idx.cardinality(),
            Some(MetadataIndex::Bitmap(idx)) => idx.cardinality(),
            None => return,
        };
        let total = match self.indexes.get(field) {
            Some(MetadataIndex::BTree(idx)) => idx.len(),
            Some(MetadataIndex::Hash(idx)) => idx.len(),
            Some(MetadataIndex::Bitmap(idx)) => idx.len(),
            None => return,
        };
        if let Some(stats) = self.field_stats.get_mut(field) {
            stats.unique_values = unique;
            stats.total_values = total;
        }
    }

    /// Returns the total number of indexed records.
    pub fn total_records(&self) -> usize {
        self.records.len()
    }

    /// Estimates filter selectivity (fraction of records that pass) using index
    /// cardinalities, without fully evaluating bitmaps.  Returns a value in
    /// (0.0, 1.0].  Falls back to 1.0 (no filtering) when estimation is not
    /// possible.
    pub fn estimate_selectivity(&self, filter: &FilterExpression) -> f64 {
        let total = self.records.len();
        if total == 0 {
            return 1.0;
        }
        self.estimate_selectivity_inner(filter, total as f64)
    }

    fn estimate_selectivity_inner(&self, filter: &FilterExpression, total: f64) -> f64 {
        match filter {
            FilterExpression::Eq(field, _) => {
                // Assuming uniform distribution: 1 / unique_values
                if let Some(stats) = self.field_stats.get(field) {
                    if stats.unique_values > 0 {
                        return (1.0 / stats.unique_values as f64).clamp(0.001, 1.0);
                    }
                }
                1.0
            }
            FilterExpression::Ne(field, _) => {
                // 1 - (1 / unique_values)
                if let Some(stats) = self.field_stats.get(field) {
                    if stats.unique_values > 0 {
                        return (1.0 - 1.0 / stats.unique_values as f64).clamp(0.001, 1.0);
                    }
                }
                1.0
            }
            FilterExpression::In(field, values) => {
                // |values| / unique_values
                if let Some(stats) = self.field_stats.get(field) {
                    if stats.unique_values > 0 {
                        return (values.len() as f64 / stats.unique_values as f64)
                            .clamp(0.001, 1.0);
                    }
                }
                1.0
            }
            FilterExpression::Gt(_, _)
            | FilterExpression::Gte(_, _)
            | FilterExpression::Lt(_, _)
            | FilterExpression::Lte(_, _) => {
                // Assume range predicates keep ~33% of data (heuristic)
                0.33
            }
            FilterExpression::Between(_, _, _) => {
                // Assume between keeps ~10% of data (heuristic)
                0.10
            }
            FilterExpression::Exists(field) => {
                // fraction of records that have this field
                if let Some(stats) = self.field_stats.get(field) {
                    return (stats.total_values as f64 / total).clamp(0.001, 1.0);
                }
                1.0
            }
            FilterExpression::Contains(_, _) => {
                // Substring match: conservative estimate
                0.10
            }
            FilterExpression::And(children) => {
                // Assume independence: multiply selectivities
                children
                    .iter()
                    .map(|c| self.estimate_selectivity_inner(c, total))
                    .fold(1.0, |acc, s| acc * s)
                    .clamp(0.001, 1.0)
            }
            FilterExpression::Or(children) => {
                // P(A|B) = P(A) + P(B) - P(A)*P(B) for independent events
                if children.is_empty() {
                    return 0.001;
                }
                let result = children
                    .iter()
                    .map(|c| self.estimate_selectivity_inner(c, total))
                    .fold(0.0, |acc, s| acc + s - acc * s);
                result.clamp(0.001, 1.0)
            }
            FilterExpression::Not(child) => {
                let inner = self.estimate_selectivity_inner(child, total);
                (1.0 - inner).clamp(0.001, 1.0)
            }
        }
    }

    fn select_index_type(&self, unique_values: usize) -> IndexType {
        if unique_values <= self.config.bitmap_cardinality_threshold {
            IndexType::Bitmap
        } else if unique_values >= self.config.btree_cardinality_threshold {
            IndexType::BTree
        } else {
            IndexType::Hash
        }
    }
}

enum IndexType {
    Bitmap,
    Hash,
    BTree,
}
