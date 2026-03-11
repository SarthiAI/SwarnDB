// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Collection and CollectionManager for organizing vectors into named groups.
//!
//! A [`Collection`] owns a WAL, an in-memory memtable, and zero or more
//! immutable on-disk segments. The [`CollectionManager`] manages the lifecycle
//! of multiple collections under a shared base directory.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{StorageError, StorageResult};
use crate::format::{DataTypeFlag, WalOp};
use crate::recovery::RecoveryManager;
use crate::segment::{Segment, SegmentWriter};
use crate::wal::WalWriter;
use crate::FilePermissionConfig;
use vf_core::store::{InMemoryVectorStore, VectorRecord};
use vf_core::types::{CollectionConfig, DataTypeConfig, Metadata, VectorId};
use vf_core::vector::VectorData;

/// Default WAL maximum size before rotation (256 MB).
const DEFAULT_WAL_MAX_SIZE: u64 = 256 * 1024 * 1024;

/// Read WAL max size from `SWARNDB_WAL_MAX_SIZE` env var (in bytes), falling
/// back to [`DEFAULT_WAL_MAX_SIZE`] when the variable is absent or invalid.
fn wal_max_size() -> u64 {
    match std::env::var("SWARNDB_WAL_MAX_SIZE") {
        Ok(val) => match val.parse::<u64>() {
            Ok(0) => DEFAULT_WAL_MAX_SIZE,
            Ok(n) => n,
            Err(_) => DEFAULT_WAL_MAX_SIZE,
        },
        Err(_) => DEFAULT_WAL_MAX_SIZE,
    }
}


/// Validate that a collection name is safe for use in file paths.
///
/// Rejects names containing "..", "/", "\", null bytes, or any character
/// that is not alphanumeric, dash, or underscore. Also rejects empty names.
fn sanitize_collection_name(name: &str) -> StorageResult<()> {
    if name.is_empty() {
        return Err(StorageError::InvalidCollectionName(
            "collection name must not be empty".into(),
        ));
    }
    if name.contains("..") || name.contains('/') || name.contains('\\') || name.contains('\0') {
        return Err(StorageError::InvalidCollectionName(format!(
            "collection name contains forbidden characters: {name:?}"
        )));
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(StorageError::InvalidCollectionName(format!(
            "collection name contains non-alphanumeric/dash/underscore characters: {name:?}"
        )));
    }
    Ok(())
}

/// Verify that `target` is strictly within `base_dir` after canonicalization.
fn verify_path_within(target: &Path, base_dir: &Path) -> StorageResult<()> {
    let canonical_base = fs::canonicalize(base_dir).map_err(StorageError::Io)?;
    let canonical_target = fs::canonicalize(target).map_err(StorageError::Io)?;
    if !canonical_target.starts_with(&canonical_base) {
        return Err(StorageError::PathTraversal(format!(
            "resolved path {:?} is outside data directory {:?}",
            canonical_target, canonical_base
        )));
    }
    Ok(())
}

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
    #[allow(dead_code)]
    perm_config: FilePermissionConfig,
    /// Tombstone set: vector IDs that have been deleted but may still exist in
    /// on-disk segments. Populated from WAL replay on open and from `delete()`
    /// calls at runtime. Consulted in `get()` and `load_all_vectors()` to
    /// filter out stale segment data.
    deleted_ids: HashSet<VectorId>,
}

impl Collection {
    /// Create a brand-new collection on disk.
    ///
    /// Creates `base_dir/<name>/` and initialises a fresh WAL and empty memtable.
    pub fn create(base_dir: &Path, config: CollectionConfig) -> StorageResult<Self> {
        Self::create_with_perms(base_dir, config, FilePermissionConfig::default())
    }

    /// Create a new collection with custom file/directory permission settings.
    pub fn create_with_perms(
        base_dir: &Path,
        mut config: CollectionConfig,
        perm_config: FilePermissionConfig,
    ) -> StorageResult<Self> {
        // Allow env var override for memtable flush threshold
        if let Ok(val) = std::env::var("SWARNDB_MEMTABLE_FLUSH_THRESHOLD") {
            if let Ok(threshold) = val.parse::<usize>() {
                config.memtable_flush_threshold = threshold;
            }
        }
        sanitize_collection_name(&config.name)?;
        let collection_dir = base_dir.join(&config.name);
        fs::create_dir_all(&collection_dir).map_err(StorageError::Io)?;

        // Set directory permissions using configurable mode.
        perm_config.apply_dir_permissions(&collection_dir)?;

        let wal_path = collection_dir.join("wal.log");
        let wal = WalWriter::create(&wal_path, wal_max_size())?;

        let memtable = InMemoryVectorStore::new(config.dimension);

        Ok(Collection {
            name: config.name.clone(),
            config,
            wal,
            memtable,
            segments: Vec::new(),
            next_segment_id: 0,
            collection_dir,
            perm_config,
            deleted_ids: HashSet::new(),
        })
    }

    /// Open an existing collection from `collection_dir`.
    ///
    /// Re-opens the WAL for append, creates a fresh memtable, and scans for
    /// existing segment files (sorted by id ascending).
    pub fn open(collection_dir: &Path, config: CollectionConfig) -> StorageResult<Self> {
        Self::open_with_perms(collection_dir, config, FilePermissionConfig::default())
    }

    /// Open an existing collection with custom file/directory permission settings.
    pub fn open_with_perms(
        collection_dir: &Path,
        mut config: CollectionConfig,
        perm_config: FilePermissionConfig,
    ) -> StorageResult<Self> {
        // Allow env var override for memtable flush threshold
        if let Ok(val) = std::env::var("SWARNDB_MEMTABLE_FLUSH_THRESHOLD") {
            if let Ok(threshold) = val.parse::<usize>() {
                config.memtable_flush_threshold = threshold;
            }
        }
        let wal_path = collection_dir.join("wal.log");
        let wal_size = wal_max_size();
        let wal = if wal_path.exists() {
            WalWriter::open(&wal_path, wal_size)?
        } else {
            WalWriter::create(&wal_path, wal_size)?
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

        // Replay WAL to rebuild memtable state and collect delete tombstones.
        let wal_path = collection_dir.join("wal.log");
        let deleted_ids = if wal_path.exists() {
            RecoveryManager::replay_wal_collecting_deletes(
                &wal_path,
                &memtable,
            )?
        } else {
            HashSet::new()
        };

        Ok(Collection {
            name: config.name.clone(),
            config,
            wal,
            memtable,
            segments,
            next_segment_id,
            collection_dir: collection_dir.to_path_buf(),
            perm_config,
            deleted_ids,
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
        // AlreadyExists is expected (overwrite scenario) — WAL is the source of truth.
        // DimensionMismatch after WAL write indicates an inconsistent state; log it.
        if let Err(e) = self.memtable.insert(record) {
            match &e {
                vf_core::store::StoreError::AlreadyExists(_) => { /* expected for overwrites */ }
                other => {
                    log::warn!("memtable insert failed after WAL write for id {id}: {other}");
                }
            }
        }

        // Clear any prior tombstone — vector is alive again.
        self.deleted_ids.remove(&id);

        if self.memtable.len() >= self.config.memtable_flush_threshold {
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
                // AlreadyExists is expected; other errors indicate inconsistency.
                if let Err(e) = self.memtable.insert(record) {
                    match &e {
                        vf_core::store::StoreError::AlreadyExists(_) => {}
                        other => {
                            log::warn!("memtable insert (update fallback) failed after WAL write for id {id}: {other}");
                        }
                    }
                }
            }
        }

        // Clear any prior tombstone — vector is alive again.
        self.deleted_ids.remove(&id);

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

        // Record tombstone so segment reads are filtered.
        self.deleted_ids.insert(id);

        Ok(())
    }

    // ── Reads ────────────────────────────────────────────────────────────

    /// Look up a vector by id, checking the memtable first, then segments
    /// from newest to oldest.
    ///
    /// Returns `None` if the vector is not found anywhere.
    pub fn get(&self, id: VectorId) -> StorageResult<Option<(Vec<f32>, Option<Metadata>)>> {
        // Tombstone check: if the id has been deleted, it should not be visible
        // even if it still exists in segments.
        if self.deleted_ids.contains(&id) {
            return Ok(None);
        }

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

    /// Returns a reference to the set of deleted vector IDs (tombstones).
    pub fn deleted_ids(&self) -> &HashSet<VectorId> {
        &self.deleted_ids
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

        // Filter out tombstoned vectors before returning.
        let result: Vec<(VectorId, Vec<f32>, Option<Metadata>)> = map
            .into_iter()
            .filter(|(id, _)| !self.deleted_ids.contains(id))
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
    perm_config: FilePermissionConfig,
}

impl CollectionManager {
    /// Create a new manager, scanning `base_path` for existing collections.
    ///
    /// Each immediate subdirectory that contains a `config.json` is treated as
    /// a collection and opened automatically.
    pub fn new(base_path: &Path) -> StorageResult<Self> {
        Self::new_with_perms(base_path, FilePermissionConfig::default())
    }

    /// Create a new manager with custom file/directory permission settings.
    pub fn new_with_perms(base_path: &Path, perm_config: FilePermissionConfig) -> StorageResult<Self> {
        fs::create_dir_all(base_path).map_err(StorageError::Io)?;

        // Set directory permissions using configurable mode.
        perm_config.apply_dir_permissions(base_path)?;

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

                let collection = Collection::open_with_perms(&path, config, perm_config.clone())?;
                let name = collection.name().to_string();
                collections.insert(name, collection);
            }
        }

        Ok(CollectionManager {
            collections,
            base_path: base_path.to_path_buf(),
            perm_config,
        })
    }

    /// Create a new collection with the given configuration.
    ///
    /// Writes `config.json` into the collection directory and initialises
    /// the WAL and memtable.
    pub fn create_collection(&mut self, config: CollectionConfig) -> StorageResult<()> {
        sanitize_collection_name(&config.name)?;
        if self.collections.contains_key(&config.name) {
            return Err(StorageError::CollectionAlreadyExists(config.name.clone()));
        }

        let collection_dir = self.base_path.join(&config.name);
        fs::create_dir_all(&collection_dir).map_err(StorageError::Io)?;

        // Set directory permissions using configurable mode.
        self.perm_config.apply_dir_permissions(&collection_dir)?;

        // Persist config as JSON (atomic write: tmp file then rename).
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        let config_path = collection_dir.join("config.json");
        let tmp_path = collection_dir.join("config.json.tmp");
        {
            let mut f = fs::File::create(&tmp_path).map_err(StorageError::Io)?;
            f.write_all(config_json.as_bytes()).map_err(StorageError::Io)?;
            f.sync_all().map_err(StorageError::Io)?;
        }
        fs::rename(&tmp_path, &config_path).map_err(StorageError::Io)?;
        // Set file permissions using configurable mode.
        self.perm_config.apply_file_permissions(&config_path)?;

        let collection = Collection::create_with_perms(&self.base_path, config, self.perm_config.clone())?;
        let name = collection.name().to_string();
        self.collections.insert(name, collection);

        Ok(())
    }

    /// Drop (delete) a collection by name.
    ///
    /// Removes the collection from memory and deletes the entire collection
    /// directory from disk.
    pub fn drop_collection(&mut self, name: &str) -> StorageResult<()> {
        sanitize_collection_name(name)?;
        self.collections.remove(name).ok_or_else(|| {
            StorageError::CollectionNotFound(name.to_string())
        })?;

        let collection_dir = self.base_path.join(name);
        if collection_dir.exists() {
            // Verify resolved path is within our data directory to prevent
            // symlink-based deletion outside the expected directory.
            verify_path_within(&collection_dir, &self.base_path)?;

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
