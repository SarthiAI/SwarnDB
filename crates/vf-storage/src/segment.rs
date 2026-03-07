// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Segment file reader (`Segment`) and writer (`SegmentWriter`).
//!
//! A segment is an immutable, memory-mapped file that stores vectors and their
//! optional metadata. See [`crate::format`] for the on-disk layout.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{StorageError, StorageResult};
use crate::format::{DataTypeFlag, SegmentHeader, SEGMENT_HEADER_SIZE};
use vf_core::store::InMemoryVectorStore;
use vf_core::types::{Metadata, VectorId};
use vf_core::vector::VectorData;

// ── Segment (read-only) ─────────────────────────────────────────────────────

/// A sealed, immutable, read-only segment backed by a memory-mapped file.
pub struct Segment {
    id: u64,
    #[allow(dead_code)]
    path: PathBuf,
    mmap: Mmap,
    header: SegmentHeader,
    /// Size of one vector entry in the data region: 8 (id) + dim * element_size.
    vector_entry_size: usize,
}

impl Segment {
    /// Open an existing segment file as a read-only memory map.
    ///
    /// Validates the header and extracts the segment id from the filename
    /// (e.g. `segment_00000005.vfs` → 5).
    pub fn open(path: &Path) -> StorageResult<Self> {
        let mmap = crate::mmap::open_read(path)?;

        if mmap.len() < SEGMENT_HEADER_SIZE {
            return Err(StorageError::TruncatedData {
                expected: SEGMENT_HEADER_SIZE,
                actual: mmap.len(),
            });
        }

        // Parse header from the first 64 bytes.
        let mut cursor = Cursor::new(&mmap[..SEGMENT_HEADER_SIZE]);
        let header = SegmentHeader::read_from(&mut cursor)?;
        header.validate()?;

        // Compute entry size. Vectors are always stored as f32 on disk (4 bytes per element),
        // regardless of the header's data_type field.
        let vector_entry_size = 8 + (header.dimension as usize) * 4;

        // Extract segment id from filename stem.
        let id = Self::parse_segment_id(path)?;

        Ok(Segment {
            id,
            path: path.to_path_buf(),
            mmap,
            header,
            vector_entry_size,
        })
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Returns the segment id (derived from filename).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns a reference to the segment header.
    pub fn header(&self) -> &SegmentHeader {
        &self.header
    }

    /// Returns the number of vectors stored in this segment.
    pub fn vector_count(&self) -> u64 {
        self.header.vector_count
    }

    /// Returns the vector dimensionality.
    pub fn dimension(&self) -> u32 {
        self.header.dimension
    }

    // ── Vector data access ───────────────────────────────────────────────

    /// Read the vector at the given 0-based `index` from the data region.
    ///
    /// Returns `(VectorId, Vec<f32>)`. Vectors are always stored as f32 on disk,
    /// so the element size used here is always 4 bytes regardless of the header's
    /// `data_type` field.
    pub fn get_vector_data(&self, index: usize) -> StorageResult<(VectorId, Vec<f32>)> {
        self.check_index(index)?;

        let offset = self.header.data_offset as usize + index * self.vector_entry_size;
        let end = offset + self.vector_entry_size;

        if end > self.mmap.len() {
            return Err(StorageError::TruncatedData {
                expected: end,
                actual: self.mmap.len(),
            });
        }

        let slice = &self.mmap[offset..end];

        // Read id (u64 LE).
        let id = u64::from_le_bytes(slice[0..8].try_into().unwrap());

        // Read dim f32 values (LE). On-disk storage is always f32 (4 bytes).
        let dim = self.header.dimension as usize;
        let mut data = Vec::with_capacity(dim);
        for i in 0..dim {
            let start = 8 + i * 4;
            let val = f32::from_le_bytes(slice[start..start + 4].try_into().unwrap());
            data.push(val);
        }

        Ok((id, data))
    }

    /// Read only the vector id at the given 0-based `index`.
    pub fn get_vector_id(&self, index: usize) -> StorageResult<VectorId> {
        self.check_index(index)?;

        let offset = self.header.data_offset as usize + index * self.vector_entry_size;
        let end = offset + 8;

        if end > self.mmap.len() {
            return Err(StorageError::TruncatedData {
                expected: end,
                actual: self.mmap.len(),
            });
        }

        let id = u64::from_le_bytes(self.mmap[offset..end].try_into().unwrap());
        Ok(id)
    }

    /// Linear scan to find the index of a vector with the given id.
    ///
    /// Returns `None` if the id is not present.
    pub fn find_vector(&self, target_id: VectorId) -> StorageResult<Option<usize>> {
        let count = self.header.vector_count as usize;
        for i in 0..count {
            let id = self.get_vector_id(i)?;
            if id == target_id {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    // ── Metadata access ──────────────────────────────────────────────────

    /// Look up metadata for the given vector id in the metadata region.
    ///
    /// Returns `None` if no metadata entry exists for this id.
    pub fn get_metadata(&self, target_id: VectorId) -> StorageResult<Option<Metadata>> {
        let meta_start = self.header.meta_offset as usize;
        let file_len = self.mmap.len();

        if meta_start >= file_len {
            // No metadata region at all.
            return Ok(None);
        }

        let mut pos = meta_start;
        while pos + 12 <= file_len {
            // Read id (u64 LE).
            let id = u64::from_le_bytes(self.mmap[pos..pos + 8].try_into().unwrap());
            pos += 8;

            // Read meta_len (u32 LE).
            let meta_len =
                u32::from_le_bytes(self.mmap[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            if pos + meta_len > file_len {
                return Err(StorageError::TruncatedData {
                    expected: pos + meta_len,
                    actual: file_len,
                });
            }

            if id == target_id {
                let meta: Metadata = bincode::deserialize(&self.mmap[pos..pos + meta_len])?;
                return Ok(Some(meta));
            }

            pos += meta_len;
        }

        Ok(None)
    }

    // ── Iteration ────────────────────────────────────────────────────────

    /// Iterate over all vectors in index order.
    pub fn iter_vectors(&self) -> impl Iterator<Item = StorageResult<(VectorId, Vec<f32>)>> + '_ {
        let count = self.header.vector_count as usize;
        (0..count).map(move |i| self.get_vector_data(i))
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Validate that `index` is within bounds.
    fn check_index(&self, index: usize) -> StorageResult<()> {
        if index >= self.header.vector_count as usize {
            return Err(StorageError::SegmentInvalidHeader(format!(
                "vector index {index} out of range (count={})",
                self.header.vector_count
            )));
        }
        Ok(())
    }

    /// Extract the numeric segment id from a path like `segment_00000005.vfs`.
    fn parse_segment_id(path: &Path) -> StorageResult<u64> {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                StorageError::SegmentInvalidHeader("cannot extract filename stem".into())
            })?;

        // Expected format: "segment_XXXXXXXX"
        let id_str = stem.strip_prefix("segment_").ok_or_else(|| {
            StorageError::SegmentInvalidHeader(format!(
                "filename does not match segment_XXXXXXXX pattern: {stem}"
            ))
        })?;

        id_str.parse::<u64>().map_err(|_| {
            StorageError::SegmentInvalidHeader(format!("cannot parse segment id from: {id_str}"))
        })
    }
}

// ── SegmentWriter ────────────────────────────────────────────────────────────

/// Stateless utility for writing segment files to disk.
pub struct SegmentWriter;

impl SegmentWriter {
    /// Write a new segment file at `path`.
    ///
    /// All vectors are stored as f32 on disk (converted via `to_f32_vec()`).
    pub fn write_segment(
        path: &Path,
        _segment_id: u64,
        records: &[(VectorId, &VectorData, Option<&Metadata>)],
        dimension: u32,
        data_type: DataTypeFlag,
    ) -> StorageResult<()> {
        // Vectors are always stored as f32 on disk (4 bytes per element),
        // regardless of the data_type parameter (which is recorded in the header
        // for informational/provenance purposes).
        let vector_entry_size = 8 + (dimension as usize) * 4;
        let vector_data_size = records.len() * vector_entry_size;

        // Pre-serialize all metadata entries so we know the total size.
        let mut meta_entries: Vec<(VectorId, Vec<u8>)> = Vec::new();
        for &(id, _, ref meta_opt) in records {
            if let Some(meta) = meta_opt {
                let bytes = bincode::serialize(meta)?;
                meta_entries.push((id, bytes));
            }
        }

        let metadata_size: usize = meta_entries
            .iter()
            .map(|(_, bytes)| 8 + 4 + bytes.len()) // id + meta_len + payload
            .sum();

        let total_size = SEGMENT_HEADER_SIZE + vector_data_size + metadata_size;

        // Create the file with a mutable mmap.
        let mut mmap = crate::mmap::create(path, total_size)?;

        // ── Write header ─────────────────────────────────────────────────
        let mut header = SegmentHeader::new(dimension, data_type);
        header.vector_count = records.len() as u64;
        header.data_offset = SEGMENT_HEADER_SIZE as u64;
        header.meta_offset = (SEGMENT_HEADER_SIZE + vector_data_size) as u64;
        header.checksum = header.compute_checksum();

        let mut cursor = Cursor::new(&mut mmap[..SEGMENT_HEADER_SIZE]);
        header.write_to(&mut cursor)?;

        // ── Write vector data region ─────────────────────────────────────
        let mut offset = SEGMENT_HEADER_SIZE;
        for &(id, ref vdata, _) in records {
            // Write id (u64 LE).
            mmap[offset..offset + 8].copy_from_slice(&id.to_le_bytes());
            offset += 8;

            // Convert to f32 and write each element as LE bytes.
            let floats = vdata.to_f32_vec();
            for &val in &floats {
                mmap[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                offset += 4;
            }
        }

        // ── Write metadata region ────────────────────────────────────────
        for (id, bytes) in &meta_entries {
            mmap[offset..offset + 8].copy_from_slice(&id.to_le_bytes());
            offset += 8;

            let meta_len = bytes.len() as u32;
            mmap[offset..offset + 4].copy_from_slice(&meta_len.to_le_bytes());
            offset += 4;

            mmap[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        }

        // Flush to disk.
        crate::mmap::sync(&mmap)?;

        Ok(())
    }

    /// Convenience method: flush an `InMemoryVectorStore` to a new segment file.
    ///
    /// The segment file is created at `dir/segment_{segment_id:08}.vfs`.
    /// Returns the path of the newly created segment file.
    pub fn flush_memtable(
        dir: &Path,
        segment_id: u64,
        store: &InMemoryVectorStore,
        data_type: DataTypeFlag,
    ) -> StorageResult<PathBuf> {
        let filename = format!("segment_{segment_id:08}.vfs");
        let path = dir.join(&filename);

        let dimension = store.dimension() as u32;

        // Collect records from the store (cloned, since DashMap doesn't allow borrowed iteration).
        let cloned = store.iter_cloned();
        let mut records: Vec<(VectorId, VectorData, Option<Metadata>)> = cloned
            .into_iter()
            .map(|(id, rec)| (id, rec.data, rec.metadata))
            .collect();

        // Sort by id for deterministic output.
        records.sort_by_key(|&(id, _, _)| id);

        let refs: Vec<(VectorId, &VectorData, Option<&Metadata>)> = records
            .iter()
            .map(|(id, data, meta)| (*id, data, meta.as_ref()))
            .collect();

        Self::write_segment(&path, segment_id, &refs, dimension, data_type)?;

        Ok(path)
    }
}
