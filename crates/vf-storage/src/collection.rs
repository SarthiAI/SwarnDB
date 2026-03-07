// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Collection and CollectionManager for organizing vectors into named groups.
//!
//! A [`Collection`] owns a WAL, an in-memory memtable, and zero or more
//! immutable on-disk segments. The [`CollectionManager`] manages the lifecycle
//! of multiple collections under a shared base directory.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{StorageError, StorageResult};
use crate::format::{DataTypeFlag, WalOp};
use crate::segment::{Segment, SegmentWriter};
use crate::wal::WalWriter;
use vf_core::store::{InMemoryVectorStore, VectorRecord};
use vf_core::types::{CollectionConfig, DataTypeConfig, Metadata, VectorId};
use vf_core::vector::VectorData;

/// Default WAL maximum size before rotation (64 MB).
const DEFAULT_WAL_MAX_SIZE: u64 = 64 * 1024 * 1024;

/// Default number of vectors in the memtable before flushing to a segment.
const DEFAULT_MEMTABLE_FLUSH_THRESHOLD: usize = 10_000;

// ── Collection ──────────────────────────────────────────────────────────────

/// A named collection of vectors backed by a WAL, memtable, and segments.
pub struct Collection {
    name: String,
    config: CollectionConfig,
    wal: WalWriter,
    memtable: InMemoryVectorStore,
    segments: Vec<Segment>,
    next_segment_id: u64,
    collection_dir: PathBuf,
}

impl Collection {
    /// Create a brand-new collection on disk.
    ///
    /// Creates `base_dir/<name>/` and initialises a fresh WAL and empty memtable.
    pub fn create(base_dir: &Path, config: CollectionConfig) -> StorageResult<Self> {
        let collection_dir = base_dir.join(&config.name);
        fs::create_dir_all(&collection_dir).map_err(StorageError::Io)?;

        let wal_path = collection_dir.join("wal.log");
        let wal = WalWriter::create(&wal_path, DEFAULT_WAL_MAX_SIZE)?;

        let memtable = InMemoryVectorStore::new(config.dimension);

        Ok(Collection {
            name: config.name.clone(),
            config,
            wal,
            memtable,
            segments: Vec::new(),
            next_segment_id: 0,
            collection_dir,
        })
    }

    /// Open an existing collection from `collection_dir`.
    ///
    /// Re-opens the WAL for append, creates a fresh memtable, and scans for
    /// existing segment files (sorted by id ascending).
    pub fn open(collection_dir: &Path, config: CollectionConfig) -> StorageResult<Self> {
        let wal_path = collection_dir.join("wal.log");
        let wal = if wal_path.exists() {
            WalWriter::open(&wal_path, DEFAULT_WAL_MAX_SIZE)?
        } else {
            WalWriter::create(&wal_path, DEFAULT_WAL_MAX_SIZE)?
        };

        let memtable = InMemoryVectorStore::new(config.dimension);

        // Scan for segment files.
        let mut segments = Vec::new();
        if collection_dir.is_dir() {
            for entry in fs::read_dir(collection_dir).map_err(StorageError::Io)? {
                let entry = entry.map_err(StorageError::Io)?;
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "vfs" {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            if stem.starts_with("segment_") {
                                let seg = Segment::open(&path)?;
                                segments.push(seg);
                            }
                        }
                    }
                }
            }
        }

        // Sort by segment id ascending.
        segments.sort_by_key(|s| s.id());

        let next_segment_id = segments.last().map_or(0, |s| s.id() + 1);

        Ok(Collection {
            name: config.name.clone(),
            config,
            wal,
            memtable,
            segments,
            next_segment_id,
            collection_dir: collection_dir.to_path_buf(),
        })
    }

    // ── Mutations ────────────────────────────────────────────────────────

    /// Insert a new vector into the collection.
    ///
    /// The operation is first written to the WAL, then applied to the memtable.
    /// If the memtable exceeds the flush threshold it is flushed to a segment.
    pub fn insert(
        &mut self,
        id: VectorId,
        data: VectorData,
        metadata: Option<Metadata>,
    ) -> StorageResult<()> {
        let payload = bincode::serialize(&(id, &data, &metadata))?;
        self.wal.append(WalOp::Insert, 0, &payload)?;

        let record = VectorRecord::new(id, data, metadata);
        // Ignore AlreadyExists from memtable — WAL is the source of truth.
        let _ = self.memtable.insert(record);

        if self.memtable.len() >= DEFAULT_MEMTABLE_FLUSH_THRESHOLD {
            self.flush_memtable()?;
        }

        Ok(())
    }

    /// Update an existing vector.
    ///
    /// If the vector is not present in the memtable it is inserted there;
    /// segments are not modified (the memtable shadows older segment data).
    pub fn update(
        &mut self,
        id: VectorId,
        data: Option<VectorData>,
        metadata: Option<Metadata>,
    ) -> StorageResult<()> {
        let payload = bincode::serialize(&(id, &data, &metadata))?;
        self.wal.append(WalOp::Update, 0, &payload)?;

        // Try updating in memtable; if not found, insert as new record.
        if self.memtable.update(id, data.clone(), metadata.clone()).is_err() {
            if let Some(d) = data {
                let record = VectorRecord::new(id, d, metadata);
                let _ = self.memtable.insert(record);
            }
        }

        Ok(())
    }

    /// Delete a vector by id.
    ///
    /// The delete is recorded in the WAL and removed from the memtable if
    /// present. Vectors that only exist in segments will be handled during
    /// compaction (not yet implemented).
    pub fn delete(&mut self, id: VectorId) -> StorageResult<()> {
        let payload = bincode::serialize(&id)?;
        self.wal.append(WalOp::Delete, 0, &payload)?;

        // Ignore not-found — the vector may only be in segments.
        let _ = self.memtable.delete(id);

        Ok(())
    }

    // ── Reads ────────────────────────────────────────────────────────────

    /// Look up a vector by id, checking the memtable first, then segments
    /// from newest to oldest.
    ///
    /// Returns `None` if the vector is not found anywhere.
    pub fn get(&self, id: VectorId) -> StorageResult<Option<(Vec<f32>, Option<Metadata>)>> {
        // Check memtable first.
        if let Ok(record) = self.memtable.get(id) {
            return Ok(Some((record.data.to_f32_vec(), record.metadata.clone())));
        }

        // Search segments newest-first.
        for seg in self.segments.iter().rev() {
            if let Some(_idx) = seg.find_vector(id)? {
                let (_vid, data) = seg.get_vector_data(_idx)?;
                let meta = seg.get_metadata(id)?;
                return Ok(Some((data, meta)));
            }
        }

        Ok(None)
    }

    // ── Flush ────────────────────────────────────────────────────────────

    /// Flush the current memtable to a new on-disk segment.
    ///
    /// After flushing:
    /// - The new segment is opened and added to the segment list.
    /// - The memtable is cleared.
    /// - The WAL is rotated if it exceeds the maximum size.
    /// - `next_segment_id` is incremented.
    pub fn flush_memtable(&mut self) -> StorageResult<()> {
        if self.memtable.is_empty() {
            return Ok(());
        }

        let data_type_flag = data_type_config_to_flag(self.config.data_type);

        let seg_path = SegmentWriter::flush_memtable(
            &self.collection_dir,
            self.next_segment_id,
            &self.memtable,
            data_type_flag,
        )?;

        let segment = Segment::open(&seg_path)?;
        self.segments.push(segment);

        self.memtable.clear();

        if self.wal.should_rotate() {
            let archive_name = format!("wal_{}.log.old", self.next_segment_id);
            let archive_path = self.collection_dir.join(archive_name);
            self.wal.rotate(&archive_path)?;
        }

        self.next_segment_id += 1;

        Ok(())
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Returns the collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the collection configuration.
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Returns the number of on-disk segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the number of vectors currently in the memtable.
    pub fn memtable_size(&self) -> usize {
        self.memtable.len()
    }

    /// Load all vectors from segments and memtable for index rebuilding.
    ///
    /// Returns `(VectorId, Vec<f32>, Option<Metadata>)` tuples. Segments are
    /// read oldest-first so that newer segment data overwrites older entries.
    /// Memtable entries (if any) take highest precedence.
    pub fn load_all_vectors(
        &self,
    ) -> StorageResult<Vec<(VectorId, Vec<f32>, Option<Metadata>)>> {
        let mut map: HashMap<VectorId, (Vec<f32>, Option<Metadata>)> = HashMap::new();

        // Read segments oldest-first (already sorted by id ascending).
        for seg in &self.segments {
            let count = seg.vector_count() as usize;
            for i in 0..count {
                let (id, data) = seg.get_vector_data(i)?;
                let meta = seg.get_metadata(id)?;
                map.insert(id, (data, meta));
            }
        }

        // Memtable entries override segment data.
        for (id, record) in self.memtable.iter_cloned() {
            map.insert(id, (record.data.to_f32_vec(), record.metadata));
        }

        let result: Vec<(VectorId, Vec<f32>, Option<Metadata>)> = map
            .into_iter()
            .map(|(id, (data, meta))| (id, data, meta))
            .collect();

        Ok(result)
    }
}

// ── CollectionManager ───────────────────────────────────────────────────────

/// Manages the lifecycle of multiple [`Collection`]s under a shared base
/// directory. Each collection is stored in its own subdirectory.
pub struct CollectionManager {
    collections: HashMap<String, Collection>,
    base_path: PathBuf,
}

impl CollectionManager {
    /// Create a new manager, scanning `base_path` for existing collections.
    ///
    /// Each immediate subdirectory that contains a `config.json` is treated as
    /// a collection and opened automatically.
    pub fn new(base_path: &Path) -> StorageResult<Self> {
        fs::create_dir_all(base_path).map_err(StorageError::Io)?;

        let mut collections = HashMap::new();

        if base_path.is_dir() {
            for entry in fs::read_dir(base_path).map_err(StorageError::Io)? {
                let entry = entry.map_err(StorageError::Io)?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }

                let config_path = path.join("config.json");
                if !config_path.exists() {
                    continue;
                }

                let config_bytes = fs::read(&config_path).map_err(StorageError::Io)?;
                let config: CollectionConfig = serde_json::from_slice(&config_bytes)
                    .map_err(|e| StorageError::Serialization(e.to_string()))?;

                let collection = Collection::open(&path, config)?;
                let name = collection.name().to_string();
                collections.insert(name, collection);
            }
        }

        Ok(CollectionManager {
            collections,
            base_path: base_path.to_path_buf(),
        })
    }

    /// Create a new collection with the given configuration.
    ///
    /// Writes `config.json` into the collection directory and initialises
    /// the WAL and memtable.
    pub fn create_collection(&mut self, config: CollectionConfig) -> StorageResult<()> {
        if self.collections.contains_key(&config.name) {
            return Err(StorageError::CollectionAlreadyExists(config.name.clone()));
        }

        let collection_dir = self.base_path.join(&config.name);
        fs::create_dir_all(&collection_dir).map_err(StorageError::Io)?;

        // Persist config as JSON.
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        fs::write(collection_dir.join("config.json"), config_json).map_err(StorageError::Io)?;

        let collection = Collection::create(&self.base_path, config)?;
        let name = collection.name().to_string();
        self.collections.insert(name, collection);

        Ok(())
    }

    /// Drop (delete) a collection by name.
    ///
    /// Removes the collection from memory and deletes the entire collection
    /// directory from disk.
    pub fn drop_collection(&mut self, name: &str) -> StorageResult<()> {
        self.collections.remove(name).ok_or_else(|| {
            StorageError::CollectionNotFound(name.to_string())
        })?;

        let collection_dir = self.base_path.join(name);
        if collection_dir.exists() {
            fs::remove_dir_all(&collection_dir).map_err(|e| {
                StorageError::CollectionDropFailed(format!(
                    "failed to remove {}: {e}",
                    collection_dir.display()
                ))
            })?;
        }

        Ok(())
    }

    /// Get a shared reference to a collection by name.
    pub fn get_collection(&self, name: &str) -> StorageResult<&Collection> {
        self.collections
            .get(name)
            .ok_or_else(|| StorageError::CollectionNotFound(name.to_string()))
    }

    /// Get a mutable reference to a collection by name.
    pub fn get_collection_mut(&mut self, name: &str) -> StorageResult<&mut Collection> {
        self.collections
            .get_mut(name)
            .ok_or_else(|| StorageError::CollectionNotFound(name.to_string()))
    }

    /// List the names of all loaded collections.
    pub fn list_collections(&self) -> Vec<&str> {
        self.collections.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the number of loaded collections.
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a [`DataTypeConfig`] (from vf-core) to a [`DataTypeFlag`] (on-disk).
fn data_type_config_to_flag(dt: DataTypeConfig) -> DataTypeFlag {
    match dt {
        DataTypeConfig::F32 => DataTypeFlag::F32,
        DataTypeConfig::F16 => DataTypeFlag::F16,
        DataTypeConfig::U8 => DataTypeFlag::U8,
    }
}
