// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use memmap2::Mmap;
use vf_core::types::VectorId;

const MAGIC: &[u8; 4] = b"VFMM";
const HEADER_SIZE: usize = 16; // magic(4) + dimension(4) + count(8)

/// Memory-mapped store for original f32 vectors on disk.
/// Provides zero-copy read access via mmap for exact rescoring during quantized search.
/// Supports both bulk build and incremental append.
pub struct MmapVectorStore {
    mmap: Option<Mmap>,
    write_file: Option<File>,
    path: PathBuf,
    dimension: usize,
    vector_count: usize,
    slot_to_offset: HashMap<VectorId, usize>, // id -> byte offset of f32 data in file
}

impl MmapVectorStore {
    /// Size in bytes of one entry: 8 (id) + dimension * 4 (f32 data).
    #[inline]
    fn entry_size(&self) -> usize {
        8 + self.dimension * 4
    }

    /// Write the 16-byte header to the given file at position 0.
    fn write_header(file: &mut File, dimension: usize, count: usize) -> io::Result<()> {
        file.seek(SeekFrom::Start(0))?;
        file.write_all(MAGIC)?;
        file.write_all(&(dimension as u32).to_le_bytes())?;
        file.write_all(&(count as u64).to_le_bytes())?;
        Ok(())
    }

    /// Write a single vector entry (id + f32 data) at the current file position.
    fn write_entry(file: &mut File, id: VectorId, vector: &[f32]) -> io::Result<()> {
        file.write_all(&id.to_le_bytes())?;
        for &val in vector {
            file.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Update only the count field in the header (bytes 8..16).
    fn update_header_count(file: &mut File, count: usize) -> io::Result<()> {
        file.seek(SeekFrom::Start(8))?;
        file.write_all(&(count as u64).to_le_bytes())?;
        Ok(())
    }

    /// Create an Mmap from the file at the given path.
    fn create_mmap(path: &Path) -> io::Result<Mmap> {
        let file = File::open(path)?;
        // Safety: the file is fully written and flushed before we mmap it.
        // We only read from the mmap; writes go through the separate write_file handle.
        unsafe { Mmap::map(&file) }
    }

    /// Open a write handle for appending.
    fn open_write_handle(path: &Path) -> io::Result<File> {
        OpenOptions::new().read(true).write(true).open(path)
    }

    /// Scan entries in the mmap to build slot_to_offset.
    fn build_index(mmap: &[u8], dimension: usize, count: usize) -> HashMap<VectorId, usize> {
        let entry_size = 8 + dimension * 4;
        let mut map = HashMap::with_capacity(count);
        for i in 0..count {
            let entry_start = HEADER_SIZE + i * entry_size;
            if entry_start + entry_size > mmap.len() {
                break;
            }
            let id_bytes: [u8; 8] = mmap[entry_start..entry_start + 8]
                .try_into()
                .expect("slice length mismatch for id");
            let id = u64::from_le_bytes(id_bytes);
            let data_offset = entry_start + 8; // offset to f32 data
            map.insert(id, data_offset);
        }
        map
    }

    /// Build a new store from a slice of vectors. Writes all to file, then mmaps.
    pub fn build(
        path: &Path,
        vectors: &[(VectorId, Vec<f32>)],
        dimension: usize,
    ) -> io::Result<Self> {
        let mut file = File::create(path)?;

        // Write header
        Self::write_header(&mut file, dimension, vectors.len())?;

        // Write each entry
        for (id, vector) in vectors.iter() {
            debug_assert_eq!(
                vector.len(),
                dimension,
                "vector dimension mismatch: expected {}, got {}",
                dimension,
                vector.len()
            );
            Self::write_entry(&mut file, *id, vector)?;
        }

        file.flush()?;
        file.sync_all()?;
        drop(file);

        // Mmap the file
        let mmap = Self::create_mmap(path)?;
        let slot_to_offset = Self::build_index(&mmap, dimension, vectors.len());
        let write_file = Self::open_write_handle(path)?;

        Ok(Self {
            mmap: Some(mmap),
            write_file: Some(write_file),
            path: path.to_path_buf(),
            dimension,
            vector_count: vectors.len(),
            slot_to_offset,
        })
    }

    /// Open an existing store file and mmap it (for recovery / restart).
    pub fn from_file(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;

        // Read and validate header
        let mut header = [0u8; HEADER_SIZE];
        file.read_exact(&mut header)?;

        if &header[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic bytes: not a VFMM file",
            ));
        }

        let dimension = u32::from_le_bytes(
            header[4..8].try_into().expect("slice length mismatch"),
        ) as usize;
        let count = u64::from_le_bytes(
            header[8..16].try_into().expect("slice length mismatch"),
        ) as usize;

        drop(file);

        // Mmap the file
        let mmap = Self::create_mmap(path)?;
        let slot_to_offset = Self::build_index(&mmap, dimension, count);
        let write_file = Self::open_write_handle(path)?;

        Ok(Self {
            mmap: Some(mmap),
            write_file: Some(write_file),
            path: path.to_path_buf(),
            dimension,
            vector_count: count,
            slot_to_offset,
        })
    }

    /// Get a vector by ID. Returns an f32 slice from the mmap region (zero-copy).
    pub fn get_vector(&self, id: VectorId) -> Option<&[f32]> {
        let &offset = self.slot_to_offset.get(&id)?;
        let mmap = self.mmap.as_ref()?;
        let byte_len = self.dimension * 4;

        // Bounds check before pointer cast
        if offset + byte_len > mmap.len() {
            return None;
        }

        let ptr = mmap[offset..].as_ptr();

        // Safety: offset is always aligned to 4 bytes (header=16, entry id=8, dim*4
        // are all multiples of 4). We verified bounds above. The mmap region is valid
        // for the lifetime of &self.
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, self.dimension) };
        Some(slice)
    }

    /// Batch get vectors by IDs.
    pub fn get_vectors_batch(&self, ids: &[VectorId]) -> Vec<(VectorId, &[f32])> {
        ids.iter()
            .filter_map(|&id| self.get_vector(id).map(|v| (id, v)))
            .collect()
    }

    /// Append a single vector after initial build. Writes to file, remaps.
    pub fn append_vector(&mut self, id: VectorId, vector: &[f32]) -> io::Result<()> {
        assert_eq!(
            vector.len(),
            self.dimension,
            "vector dimension mismatch: expected {}, got {}",
            self.dimension,
            vector.len()
        );

        // Drop mmap before writing
        self.mmap = None;

        let file = self
            .write_file
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "write file not available"))?;

        // Seek to end and write the entry
        file.seek(SeekFrom::End(0))?;
        Self::write_entry(file, id, vector)?;

        // Update count in header
        self.vector_count += 1;
        Self::update_header_count(file, self.vector_count)?;

        file.flush()?;
        file.sync_all()?;

        // Re-mmap
        let mmap = Self::create_mmap(&self.path)?;

        // Compute offset for the new entry
        let entry_index = self.vector_count - 1;
        let data_offset = HEADER_SIZE + entry_index * self.entry_size() + 8;
        self.slot_to_offset.insert(id, data_offset);

        self.mmap = Some(mmap);
        Ok(())
    }

    /// Append multiple vectors at once. Single remap at the end.
    pub fn append_vectors_batch(
        &mut self,
        vectors: &[(VectorId, Vec<f32>)],
    ) -> io::Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Drop mmap before writing
        self.mmap = None;

        let file = self
            .write_file
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "write file not available"))?;

        // Seek to end and write all entries
        file.seek(SeekFrom::End(0))?;
        for (id, vector) in vectors.iter() {
            debug_assert_eq!(vector.len(), self.dimension);
            Self::write_entry(file, *id, vector)?;
        }

        // Update count
        let old_count = self.vector_count;
        self.vector_count += vectors.len();
        Self::update_header_count(file, self.vector_count)?;

        file.flush()?;
        file.sync_all()?;

        // Re-mmap
        let mmap = Self::create_mmap(&self.path)?;

        // Build offsets for new entries
        let entry_size = self.entry_size();
        for (i, (id, _)) in vectors.iter().enumerate() {
            let entry_index = old_count + i;
            let data_offset = HEADER_SIZE + entry_index * entry_size + 8;
            self.slot_to_offset.insert(*id, data_offset);
        }

        self.mmap = Some(mmap);
        Ok(())
    }

    /// Mark a vector as removed (lazy deletion).
    /// Space is reclaimed only on rebuild.
    pub fn remove_vector(&mut self, id: VectorId) {
        self.slot_to_offset.remove(&id);
    }

    /// Rebuild the store from scratch, compacting removed entries.
    /// Writes to a temp file, then atomically renames over the original.
    pub fn rebuild(&mut self, vectors: &[(VectorId, Vec<f32>)]) -> io::Result<()> {
        let tmp_path = self.path.with_extension("tmp");

        // Write to temp file
        {
            let mut file = File::create(&tmp_path)?;
            Self::write_header(&mut file, self.dimension, vectors.len())?;
            for (id, vector) in vectors.iter() {
                debug_assert_eq!(vector.len(), self.dimension);
                Self::write_entry(&mut file, *id, vector)?;
            }
            file.flush()?;
            file.sync_all()?;
        }

        // Drop mmap and write handle before rename
        self.mmap = None;
        self.write_file = None;

        // Atomic rename
        fs::rename(&tmp_path, &self.path)?;

        // Re-open and re-mmap
        let mmap = Self::create_mmap(&self.path)?;
        self.slot_to_offset = Self::build_index(&mmap, self.dimension, vectors.len());
        self.vector_count = vectors.len();
        self.write_file = Some(Self::open_write_handle(&self.path)?);
        self.mmap = Some(mmap);

        Ok(())
    }

    /// Check if a vector ID exists in the store (not removed).
    pub fn contains(&self, id: VectorId) -> bool {
        self.slot_to_offset.contains_key(&id)
    }

    /// Number of accessible vectors (excludes lazily-removed entries).
    pub fn len(&self) -> usize {
        self.slot_to_offset.len()
    }

    /// Whether the store has no accessible vectors.
    pub fn is_empty(&self) -> bool {
        self.slot_to_offset.is_empty()
    }

    /// Path to the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_vectors(count: usize, dim: usize, id_offset: u64) -> Vec<(VectorId, Vec<f32>)> {
        (0..count)
            .map(|i| {
                let id = id_offset + i as u64;
                let vec: Vec<f32> = (0..dim).map(|d| (id as f32) * 100.0 + d as f32).collect();
                (id, vec)
            })
            .collect()
    }

    #[test]
    fn test_build_and_get() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.vfmm");
        let dim = 8;
        let vectors = make_vectors(50, dim, 1);

        let store = MmapVectorStore::build(&path, &vectors, dim).unwrap();
        assert_eq!(store.len(), 50);
        assert!(!store.is_empty());

        // Verify each vector
        for (id, expected) in &vectors {
            let got = store.get_vector(*id).expect("vector not found");
            assert_eq!(got.len(), dim);
            for (a, b) in got.iter().zip(expected.iter()) {
                assert_eq!(*a, *b);
            }
        }

        // Non-existent id
        assert!(store.get_vector(9999).is_none());
    }

    #[test]
    fn test_batch_get() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.vfmm");
        let dim = 4;
        let vectors = make_vectors(10, dim, 100);

        let store = MmapVectorStore::build(&path, &vectors, dim).unwrap();

        let ids = vec![100, 103, 107, 9999];
        let results = store.get_vectors_batch(&ids);
        assert_eq!(results.len(), 3); // 9999 doesn't exist
        assert_eq!(results[0].0, 100);
        assert_eq!(results[1].0, 103);
        assert_eq!(results[2].0, 107);
    }

    #[test]
    fn test_append_and_verify() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.vfmm");
        let dim = 8;
        let vectors = make_vectors(50, dim, 1);

        let mut store = MmapVectorStore::build(&path, &vectors, dim).unwrap();
        assert_eq!(store.len(), 50);

        // Append 10 more one at a time
        let extra = make_vectors(10, dim, 1000);
        for (id, vec) in &extra {
            store.append_vector(*id, vec).unwrap();
        }
        assert_eq!(store.len(), 60);

        // Verify all 60
        for (id, expected) in vectors.iter().chain(extra.iter()) {
            let got = store.get_vector(*id).expect("vector not found after append");
            for (a, b) in got.iter().zip(expected.iter()) {
                assert_eq!(*a, *b);
            }
        }
    }

    #[test]
    fn test_append_batch() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.vfmm");
        let dim = 4;
        let vectors = make_vectors(20, dim, 1);

        let mut store = MmapVectorStore::build(&path, &vectors, dim).unwrap();

        let extra = make_vectors(15, dim, 500);
        store.append_vectors_batch(&extra).unwrap();
        assert_eq!(store.len(), 35);

        for (id, expected) in vectors.iter().chain(extra.iter()) {
            let got = store.get_vector(*id).unwrap();
            assert_eq!(got, expected.as_slice());
        }
    }

    #[test]
    fn test_from_file_recovery() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.vfmm");
        let dim = 8;
        let vectors = make_vectors(50, dim, 1);

        {
            let _store = MmapVectorStore::build(&path, &vectors, dim).unwrap();
            // store dropped here
        }

        // Reopen from file
        let store = MmapVectorStore::from_file(&path).unwrap();
        assert_eq!(store.len(), 50);

        for (id, expected) in &vectors {
            let got = store.get_vector(*id).unwrap();
            assert_eq!(got, expected.as_slice());
        }
    }

    #[test]
    fn test_remove_and_rebuild() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.vfmm");
        let dim = 8;
        let vectors = make_vectors(50, dim, 1);

        let mut store = MmapVectorStore::build(&path, &vectors, dim).unwrap();

        // Append 10 more
        let extra = make_vectors(10, dim, 1000);
        store.append_vectors_batch(&extra).unwrap();
        assert_eq!(store.len(), 60);

        // Remove some vectors via lazy deletion
        store.remove_vector(1);
        store.remove_vector(25);
        store.remove_vector(1005);
        assert_eq!(store.len(), 57);
        assert!(!store.contains(1));
        assert!(!store.contains(25));
        assert!(!store.contains(1005));
        assert!(store.contains(2));

        // Rebuild with 40 vectors
        let rebuild_vecs = make_vectors(40, dim, 2000);
        store.rebuild(&rebuild_vecs).unwrap();
        assert_eq!(store.len(), 40);

        // Old vectors should not exist
        assert!(!store.contains(1));
        assert!(!store.contains(50));
        assert!(!store.contains(1000));

        // New vectors should exist with correct data
        for (id, expected) in &rebuild_vecs {
            assert!(store.contains(*id));
            let got = store.get_vector(*id).unwrap();
            assert_eq!(got, expected.as_slice());
        }
    }

    #[test]
    fn test_empty_store() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.vfmm");
        let dim = 4;

        let store = MmapVectorStore::build(&path, &[], dim).unwrap();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.get_vector(1).is_none());
    }

    #[test]
    fn test_invalid_magic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.vfmm");

        // Write garbage
        let mut f = File::create(&path).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
        f.flush().unwrap();
        drop(f);

        let result = MmapVectorStore::from_file(&path);
        assert!(result.is_err());
        let err = match result {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn test_contains_and_path() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.vfmm");
        let dim = 4;
        let vectors = make_vectors(5, dim, 10);

        let store = MmapVectorStore::build(&path, &vectors, dim).unwrap();
        assert!(store.contains(10));
        assert!(store.contains(14));
        assert!(!store.contains(0));
        assert!(!store.contains(15));
        assert_eq!(store.path(), path);
    }
}
