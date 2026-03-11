// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use crate::types::{Metadata, VectorId};
use crate::vector::VectorData;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Errors from store operations
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("vector {0} not found")]
    NotFound(VectorId),

    #[error("vector {0} already exists")]
    AlreadyExists(VectorId),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("store is full (capacity: {0})")]
    CapacityExceeded(usize),

    #[error("vector error: {0}")]
    VectorError(#[from] crate::vector::VectorError),
}

/// A record stored in the vector store
#[derive(Clone, Debug)]
pub struct VectorRecord {
    pub id: VectorId,
    pub data: VectorData,
    pub metadata: Option<Metadata>,
    pub version: u64,
}

impl VectorRecord {
    pub fn new(id: VectorId, data: VectorData, metadata: Option<Metadata>) -> Self {
        Self {
            id,
            data,
            metadata,
            version: 1,
        }
    }
}

/// In-memory vector store. DashMap-based lock-free concurrent storage.
/// Used as the memtable layer before flushing to segments.
///
/// All mutation methods take `&self` instead of `&mut self`, enabling
/// concurrent reads and writes without external locking.
pub struct InMemoryVectorStore {
    vectors: DashMap<VectorId, VectorRecord>,
    dimension: usize,
    next_id: AtomicU64,
    /// Monotonically increasing counter bumped on every mutation (insert/update/delete/clear).
    /// Consumers can compare this value to detect store changes and invalidate caches.
    generation: AtomicU64,
}

// DashMap and AtomicU64 are both Send + Sync, so InMemoryVectorStore is too.
// These static assertions ensure the compiler enforces it.
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}
    fn _assertions() {
        _assert_send::<InMemoryVectorStore>();
        _assert_sync::<InMemoryVectorStore>();
    }
};

impl InMemoryVectorStore {
    /// Create a new store with the given vector dimensionality
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: DashMap::new(),
            dimension,
            next_id: AtomicU64::new(1),
            generation: AtomicU64::new(0),
        }
    }

    /// Returns the number of vectors in the store
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Returns true if the store is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Returns the dimension all vectors must have
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the current generation counter. Incremented on every mutation.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Generate the next available vector ID (thread-safe)
    pub fn next_id(&self) -> VectorId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Insert a vector with an explicit ID.
    /// Returns error if ID already exists or dimension doesn't match.
    pub fn insert(&self, record: VectorRecord) -> Result<(), StoreError> {
        // Validate dimension
        let dim = record.data.dimension();
        if dim != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                actual: dim,
            });
        }

        // Atomic check-and-insert via entry API to avoid TOCTOU race
        let record_id = record.id;
        match self.vectors.entry(record.id) {
            Entry::Occupied(_) => return Err(StoreError::AlreadyExists(record.id)),
            Entry::Vacant(v) => { v.insert(record); }
        }

        self.generation.fetch_add(1, Ordering::Release);

        // CAS loop to update next_id if needed
        let new_next = record_id.saturating_add(1);
        let mut current = self.next_id.load(Ordering::Relaxed);
        while new_next > current {
            match self.next_id.compare_exchange_weak(
                current,
                new_next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }

        Ok(())
    }

    /// Insert a vector with auto-generated ID.
    /// Returns the assigned VectorId.
    pub fn insert_auto_id(
        &self,
        data: VectorData,
        metadata: Option<Metadata>,
    ) -> Result<VectorId, StoreError> {
        let dim = data.dimension();
        if dim != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                actual: dim,
            });
        }

        let id = self.next_id();
        let record = VectorRecord::new(id, data, metadata);
        self.vectors.insert(id, record);
        self.generation.fetch_add(1, Ordering::Release);
        Ok(id)
    }

    /// Get a vector by ID (returns a clone)
    pub fn get(&self, id: VectorId) -> Result<VectorRecord, StoreError> {
        self.vectors
            .get(&id)
            .map(|r| r.value().clone())
            .ok_or(StoreError::NotFound(id))
    }

    /// Update a vector's data. Increments the version.
    pub fn update(
        &self,
        id: VectorId,
        data: Option<VectorData>,
        metadata: Option<Metadata>,
    ) -> Result<(), StoreError> {
        if let Some(ref d) = data {
            let dim = d.dimension();
            if dim != self.dimension {
                return Err(StoreError::DimensionMismatch {
                    expected: self.dimension,
                    actual: dim,
                });
            }
        }

        let mut entry = self.vectors.get_mut(&id).ok_or(StoreError::NotFound(id))?;
        if let Some(d) = data {
            entry.data = d;
        }
        if metadata.is_some() {
            entry.metadata = metadata;
        }
        entry.version += 1;
        self.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Delete a vector by ID. Returns the removed record.
    pub fn delete(&self, id: VectorId) -> Result<VectorRecord, StoreError> {
        let result = self.vectors
            .remove(&id)
            .map(|(_, record)| record)
            .ok_or(StoreError::NotFound(id));
        if result.is_ok() {
            self.generation.fetch_add(1, Ordering::Release);
        }
        result
    }

    /// Check if a vector exists
    pub fn contains(&self, id: VectorId) -> bool {
        self.vectors.contains_key(&id)
    }

    /// Collect all vector records as (id, record) pairs.
    ///
    /// Returns owned copies since DashMap references cannot escape the iterator.
    /// NOTE: This clones both vector data AND metadata. Prefer `iter_vector_data`
    /// or `iter_metadata` when only one component is needed.
    pub fn iter_cloned(&self) -> Vec<(VectorId, VectorRecord)> {
        self.vectors
            .iter()
            .map(|r| (*r.key(), r.value().clone()))
            .collect()
    }

    /// Lightweight iteration returning only (id, f32 vector data) pairs.
    ///
    /// Avoids cloning metadata, making this ~30-50% faster than `iter_cloned`
    /// for search-oriented workloads that only need vector data for distance
    /// computation and ranking.
    pub fn iter_vector_data(&self) -> Vec<(VectorId, Vec<f32>)> {
        self.vectors
            .iter()
            .map(|r| (*r.key(), r.value().data.to_f32_vec()))
            .collect()
    }

    /// Iteration returning only (id, metadata) pairs, skipping entries with no metadata.
    ///
    /// Avoids cloning vector data, useful for rebuilding metadata caches
    /// without the cost of copying potentially large f32 vectors.
    pub fn iter_metadata(&self) -> Vec<(VectorId, Metadata)> {
        self.vectors
            .iter()
            .filter_map(|r| {
                r.value()
                    .metadata
                    .as_ref()
                    .map(|m| (*r.key(), m.clone()))
            })
            .collect()
    }

    /// Fetch metadata for a single vector by ID.
    ///
    /// Returns `None` if the vector does not exist or has no metadata.
    /// Use this for lazy post-search metadata retrieval instead of
    /// pre-cloning all metadata via `iter_cloned`.
    pub fn get_metadata_by_id(&self, id: VectorId) -> Option<Metadata> {
        self.vectors
            .get(&id)
            .and_then(|r| r.value().metadata.clone())
    }

    /// Get all vector IDs
    pub fn ids(&self) -> Vec<VectorId> {
        self.vectors.iter().map(|r| *r.key()).collect()
    }

    /// Clear all vectors from the store
    pub fn clear(&self) {
        self.vectors.clear();
        self.next_id.store(1, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Get the f32 data for a vector (converting if needed)
    pub fn get_f32_data(&self, id: VectorId) -> Result<Vec<f32>, StoreError> {
        let record = self.get(id)?;
        Ok(record.data.to_f32_vec())
    }

    /// Bulk insert multiple vectors. Stops on first error.
    ///
    /// **Non-transactional:** If insertion fails mid-batch (e.g., due to a
    /// duplicate ID or dimension mismatch), records successfully inserted
    /// before the error remain in the store and are NOT rolled back.
    ///
    /// This is acceptable for in-memory usage because:
    /// 1. The store is a memtable layer flushed to durable segments separately.
    /// 2. Callers can validate inputs (dimensions, uniqueness) before batching.
    /// 3. Partial insertion is recoverable — the caller knows which ID failed
    ///    and can retry or compensate.
    pub fn insert_batch(
        &self,
        records: Vec<VectorRecord>,
    ) -> Result<usize, StoreError> {
        let count = records.len();
        for record in records {
            self.insert(record)?;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MetadataValue;
    use std::collections::HashMap;

    fn make_metadata() -> Metadata {
        let mut m = HashMap::new();
        m.insert("category".to_string(), MetadataValue::String("test".to_string()));
        m.insert("score".to_string(), MetadataValue::Int(42));
        m
    }

    #[test]
    fn test_new_store() {
        let store = InMemoryVectorStore::new(128);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.dimension(), 128);
    }

    #[test]
    fn test_insert_and_get() {
        let store = InMemoryVectorStore::new(3);
        let record = VectorRecord::new(0, VectorData::F32(vec![1.0, 2.0, 3.0]), None);
        store.insert(record).unwrap();

        assert_eq!(store.len(), 1);
        let retrieved = store.get(0).unwrap();
        assert_eq!(retrieved.id, 0);
        assert_eq!(retrieved.version, 1);
    }

    #[test]
    fn test_insert_auto_id() {
        let store = InMemoryVectorStore::new(3);
        let id1 = store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();
        let id2 = store.insert_auto_id(VectorData::F32(vec![4.0, 5.0, 6.0]), None).unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_insert_with_metadata() {
        let store = InMemoryVectorStore::new(3);
        let meta = make_metadata();
        let id = store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), Some(meta)).unwrap();
        let record = store.get(id).unwrap();
        let meta = record.metadata.as_ref().unwrap();
        assert_eq!(meta.get("category").unwrap().as_str(), Some("test"));
    }

    #[test]
    fn test_insert_dimension_mismatch() {
        let store = InMemoryVectorStore::new(3);
        let result = store.insert_auto_id(VectorData::F32(vec![1.0, 2.0]), None);
        assert!(result.is_err());
        match result.unwrap_err() {
            StoreError::DimensionMismatch { expected: 3, actual: 2 } => {}
            e => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_insert_duplicate() {
        let store = InMemoryVectorStore::new(3);
        let record = VectorRecord::new(0, VectorData::F32(vec![1.0, 2.0, 3.0]), None);
        store.insert(record.clone()).unwrap();
        let result = store.insert(record);
        assert!(matches!(result, Err(StoreError::AlreadyExists(0))));
    }

    #[test]
    fn test_update() {
        let store = InMemoryVectorStore::new(3);
        let id = store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();

        store.update(id, Some(VectorData::F32(vec![4.0, 5.0, 6.0])), None).unwrap();
        let record = store.get(id).unwrap();
        assert_eq!(record.data.to_f32_vec(), vec![4.0, 5.0, 6.0]);
        assert_eq!(record.version, 2);
    }

    #[test]
    fn test_update_nonexistent() {
        let store = InMemoryVectorStore::new(3);
        let result = store.update(999, Some(VectorData::F32(vec![1.0, 2.0, 3.0])), None);
        assert!(matches!(result, Err(StoreError::NotFound(999))));
    }

    #[test]
    fn test_delete() {
        let store = InMemoryVectorStore::new(3);
        let id = store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();
        assert_eq!(store.len(), 1);

        let removed = store.delete(id).unwrap();
        assert_eq!(removed.id, id);
        assert_eq!(store.len(), 0);
        assert!(!store.contains(id));
    }

    #[test]
    fn test_delete_nonexistent() {
        let store = InMemoryVectorStore::new(3);
        let result = store.delete(999);
        assert!(matches!(result, Err(StoreError::NotFound(999))));
    }

    #[test]
    fn test_contains() {
        let store = InMemoryVectorStore::new(3);
        assert!(!store.contains(1));
        let id = store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();
        assert!(store.contains(id));
    }

    #[test]
    fn test_clear() {
        let store = InMemoryVectorStore::new(3);
        store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();
        store.insert_auto_id(VectorData::F32(vec![4.0, 5.0, 6.0]), None).unwrap();
        assert_eq!(store.len(), 2);

        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_ids() {
        let store = InMemoryVectorStore::new(3);
        store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();
        store.insert_auto_id(VectorData::F32(vec![4.0, 5.0, 6.0]), None).unwrap();
        let mut ids = store.ids();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_get_f32_data() {
        let store = InMemoryVectorStore::new(3);
        let id = store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();
        let data = store.get_f32_data(id).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch_insert() {
        let store = InMemoryVectorStore::new(3);
        let records = vec![
            VectorRecord::new(0, VectorData::F32(vec![1.0, 2.0, 3.0]), None),
            VectorRecord::new(1, VectorData::F32(vec![4.0, 5.0, 6.0]), None),
            VectorRecord::new(2, VectorData::F32(vec![7.0, 8.0, 9.0]), None),
        ];
        let count = store.insert_batch(records).unwrap();
        assert_eq!(count, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_iter_cloned() {
        let store = InMemoryVectorStore::new(3);
        store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();
        store.insert_auto_id(VectorData::F32(vec![4.0, 5.0, 6.0]), None).unwrap();

        let count = store.iter_cloned().len();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_version_tracking() {
        let store = InMemoryVectorStore::new(3);
        let id = store.insert_auto_id(VectorData::F32(vec![1.0, 2.0, 3.0]), None).unwrap();
        assert_eq!(store.get(id).unwrap().version, 1);

        store.update(id, Some(VectorData::F32(vec![4.0, 5.0, 6.0])), None).unwrap();
        assert_eq!(store.get(id).unwrap().version, 2);

        store.update(id, Some(VectorData::F32(vec![7.0, 8.0, 9.0])), None).unwrap();
        assert_eq!(store.get(id).unwrap().version, 3);
    }
}
