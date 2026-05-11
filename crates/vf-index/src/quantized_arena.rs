// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Contiguous u8 storage for scalar-quantized vector codes.
//!
//! Mirrors [`crate::arena::VectorArena`] but stores `u8` quantized codes
//! instead of `f32` vectors. Each slot holds exactly `dimension` bytes
//! (one byte per original vector dimension after SQ8 quantization).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use vf_core::types::VectorId;

const SQ8C_MAGIC: &[u8; 4] = b"SQ8C";
const SQ8C_VERSION: u32 = 1;
const SQ8C_HEADER_SIZE: usize = 16;

/// Contiguous u8 storage for scalar-quantized vector codes.
///
/// Each slot stores `dimension` bytes. Slot `i` occupies
/// `[i * dim .. (i + 1) * dim]` in the flat buffer.
///
/// Supports push, get, and free with slot reuse, same pattern as
/// [`crate::arena::VectorArena`].
pub struct QuantizedArena {
    /// Flat buffer: slot `i` occupies `[i*dim .. (i+1)*dim]`.
    data: Vec<u8>,
    /// Number of u8 elements per quantized code.
    dimension: usize,
    /// Total number of allocated slots (including freed ones still in buffer).
    slot_count: usize,
    /// Free-list of reusable slot indices from prior deletions.
    free_slots: Vec<usize>,
}

impl QuantizedArena {
    /// Creates a new quantized arena for the given dimension.
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "QuantizedArena dimension must be > 0");
        Self {
            data: Vec::new(),
            dimension,
            slot_count: 0,
            free_slots: Vec::new(),
        }
    }

    /// Creates a new quantized arena pre-allocated for `capacity` codes.
    pub fn with_capacity(dimension: usize, capacity: usize) -> Self {
        assert!(dimension > 0, "QuantizedArena dimension must be > 0");
        Self {
            data: Vec::with_capacity(dimension * capacity),
            dimension,
            slot_count: 0,
            free_slots: Vec::new(),
        }
    }

    /// Build an arena from a dense byte buffer where slot `i` occupies
    /// `data[i * dimension .. (i+1) * dimension]`. Used by parallel cold-build
    /// to skip the per-slot push overhead. The caller is responsible for
    /// ensuring `data.len() == slot_count * dimension`.
    ///
    /// # Panics
    /// Panics if `data.len() != slot_count * dimension`. This is a programmer
    /// error in the cold-build path, not a runtime fallback condition.
    pub fn from_dense(data: Vec<u8>, dimension: usize, slot_count: usize) -> Self {
        assert!(dimension > 0, "QuantizedArena dimension must be > 0");
        assert_eq!(
            data.len(),
            slot_count * dimension,
            "QuantizedArena::from_dense: data.len() ({}) does not match slot_count ({}) * dimension ({})",
            data.len(),
            slot_count,
            dimension
        );
        Self {
            data,
            dimension,
            slot_count,
            free_slots: Vec::new(),
        }
    }

    /// Stores a quantized code in the arena and returns its slot index.
    ///
    /// Reuses a previously freed slot if available, otherwise appends.
    ///
    /// # Panics
    /// Panics if `code.len() != self.dimension`.
    pub fn push(&mut self, code: &[u8]) -> usize {
        assert_eq!(
            code.len(),
            self.dimension,
            "QuantizedArena::push: expected dimension {}, got {}",
            self.dimension,
            code.len()
        );

        if let Some(slot) = self.free_slots.pop() {
            // Reuse a freed slot, overwrite in place.
            let start = slot * self.dimension;
            self.data[start..start + self.dimension].copy_from_slice(code);
            slot
        } else {
            // Append a new slot.
            let slot = self.slot_count;
            self.data.extend_from_slice(code);
            self.slot_count += 1;
            slot
        }
    }

    /// Returns an immutable slice to the quantized code at the given slot.
    ///
    /// # Panics
    /// Panics if `slot >= slot_count`.
    #[inline]
    pub fn get(&self, slot: usize) -> &[u8] {
        let start = slot * self.dimension;
        &self.data[start..start + self.dimension]
    }

    /// Marks a slot as free for future reuse.
    ///
    /// The underlying memory is not reclaimed, it will be overwritten by the
    /// next `push` that reuses this slot. Callers must not access the slot
    /// after freeing it.
    pub fn free(&mut self, slot: usize) {
        debug_assert!(slot < self.slot_count, "QuantizedArena::free: slot out of bounds");
        self.free_slots.push(slot);
    }

    /// Returns the number of active (non-freed) codes.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.slot_count - self.free_slots.len()
    }

    /// Drops all data and resets the arena.
    pub fn clear(&mut self) {
        self.data.clear();
        self.free_slots.clear();
        self.slot_count = 0;
    }

    /// Returns the code dimension (bytes per quantized vector).
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns an estimate of the total memory consumed by this arena in bytes.
    ///
    /// Includes the data buffer plus overhead from the free-list and struct fields.
    pub fn memory_bytes(&self) -> usize {
        // data buffer (capacity, not len, to reflect actual allocation)
        let data_bytes = self.data.capacity();
        // free_slots Vec overhead
        let free_list_bytes = self.free_slots.capacity() * std::mem::size_of::<usize>();
        // struct fields (dimension, slot_count, 3 Vec headers)
        let struct_overhead = std::mem::size_of::<Self>();
        data_bytes + free_list_bytes + struct_overhead
    }

    /// Total number of slots (including freed ones still in the buffer).
    #[inline]
    pub fn len(&self) -> usize {
        self.slot_count
    }

    /// Whether the arena holds zero slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slot_count == 0
    }

    /// Atomically persist the arena's u8 codes plus a slot→VectorId mapping
    /// to disk.
    ///
    /// Format: 16-byte header `[magic "SQ8C", version u32 LE, count u32 LE,
    /// dim u32 LE]` followed by `count * 8` bytes of `VectorId` (one id per
    /// slot) followed by `count * dim` raw u8 code bytes.
    ///
    /// `slot_ids[i]` is the VectorId stored in slot `i`. The arena must
    /// currently be free-list-empty (i.e. no holes); recovery does not
    /// support sparse slots, call `compact()` first if needed.
    pub fn save_to_path(
        &self,
        path: &Path,
        slot_ids: &[VectorId],
    ) -> Result<(), io::Error> {
        if slot_ids.len() != self.slot_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "slot_ids length {} does not match slot_count {}",
                    slot_ids.len(),
                    self.slot_count
                ),
            ));
        }
        if !self.free_slots.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot persist QuantizedArena with non-empty free list",
            ));
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let tmp_path = path.with_extension("tmp");
        let _ = fs::remove_file(&tmp_path);

        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;

            file.write_all(SQ8C_MAGIC)?;
            file.write_all(&SQ8C_VERSION.to_le_bytes())?;
            file.write_all(&(self.slot_count as u32).to_le_bytes())?;
            file.write_all(&(self.dimension as u32).to_le_bytes())?;

            for id in slot_ids {
                file.write_all(&id.to_le_bytes())?;
            }
            file.write_all(&self.data[..self.slot_count * self.dimension])?;

            file.sync_data()?;
        }

        fs::rename(&tmp_path, path)?;

        if let Some(parent) = path.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }

        Ok(())
    }

    /// Load a previously persisted arena (and its slot→VectorId mapping)
    /// from disk.
    ///
    /// Validates magic, version, and dimension. Returns `InvalidData` on
    /// any mismatch or short read. The returned `Vec<VectorId>` is indexed
    /// by slot number.
    pub fn load_from_path(
        path: &Path,
        dimension: usize,
    ) -> Result<(Self, Vec<VectorId>), io::Error> {
        let mut file = File::open(path)?;
        let mut header = [0u8; SQ8C_HEADER_SIZE];
        file.read_exact(&mut header)?;

        if &header[0..4] != SQ8C_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic bytes: not an SQ8C file",
            ));
        }

        let version = u32::from_le_bytes(header[4..8].try_into().expect("slice len"));
        if version != SQ8C_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported SQ8C version {version}"),
            ));
        }

        let count = u32::from_le_bytes(header[8..12].try_into().expect("slice len")) as usize;
        let stored_dim =
            u32::from_le_bytes(header[12..16].try_into().expect("slice len")) as usize;

        if stored_dim != dimension {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "SQ8C dimension mismatch: file has {stored_dim}, expected {dimension}"
                ),
            ));
        }

        // Read slot→id table.
        let id_bytes = count * 8;
        let mut id_buf = vec![0u8; id_bytes];
        file.read_exact(&mut id_buf)?;
        let mut slot_ids: Vec<VectorId> = Vec::with_capacity(count);
        for i in 0..count {
            let off = i * 8;
            let id = u64::from_le_bytes(id_buf[off..off + 8].try_into().expect("slice len"));
            slot_ids.push(id);
        }

        // Read codes body.
        let expected_bytes = count * dimension;
        let mut data = Vec::with_capacity(expected_bytes);
        file.read_to_end(&mut data)?;

        if data.len() != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "SQ8C body length mismatch: expected {expected_bytes} bytes, got {}",
                    data.len()
                ),
            ));
        }

        Ok((
            Self {
                data,
                dimension,
                slot_count: count,
                free_slots: Vec::new(),
            },
            slot_ids,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_get() {
        let mut qa = QuantizedArena::new(4);
        let s0 = qa.push(&[10, 20, 30, 40]);
        let s1 = qa.push(&[50, 60, 70, 80]);
        assert_eq!(qa.get(s0), &[10, 20, 30, 40]);
        assert_eq!(qa.get(s1), &[50, 60, 70, 80]);
        assert_eq!(qa.active_count(), 2);
    }

    #[test]
    fn free_and_reuse() {
        let mut qa = QuantizedArena::new(3);
        let s0 = qa.push(&[1, 2, 3]);
        let s1 = qa.push(&[4, 5, 6]);
        qa.free(s0);
        assert_eq!(qa.active_count(), 1);

        // Next push should reuse the freed slot.
        let s2 = qa.push(&[7, 8, 9]);
        assert_eq!(s2, s0);
        assert_eq!(qa.get(s2), &[7, 8, 9]);
        // Other slot is unaffected.
        assert_eq!(qa.get(s1), &[4, 5, 6]);
    }

    #[test]
    fn clear_resets_everything() {
        let mut qa = QuantizedArena::new(2);
        qa.push(&[1, 2]);
        qa.push(&[3, 4]);
        qa.free(0);
        qa.clear();
        assert_eq!(qa.active_count(), 0);
        assert_eq!(qa.dimension(), 2);
    }

    #[test]
    fn active_count_after_push_and_free() {
        let mut qa = QuantizedArena::new(2);
        assert_eq!(qa.active_count(), 0);

        qa.push(&[1, 2]);
        assert_eq!(qa.active_count(), 1);

        qa.push(&[3, 4]);
        assert_eq!(qa.active_count(), 2);

        qa.free(0);
        assert_eq!(qa.active_count(), 1);

        qa.free(1);
        assert_eq!(qa.active_count(), 0);

        // Reuse both freed slots.
        qa.push(&[5, 6]);
        assert_eq!(qa.active_count(), 1);

        qa.push(&[7, 8]);
        assert_eq!(qa.active_count(), 2);
    }

    #[test]
    #[should_panic(expected = "expected dimension 4, got 2")]
    fn wrong_dimension_panics() {
        let mut qa = QuantizedArena::new(4);
        qa.push(&[1, 2]);
    }

    #[test]
    #[should_panic(expected = "dimension must be > 0")]
    fn zero_dimension_panics() {
        QuantizedArena::new(0);
    }

    #[test]
    fn with_capacity_works() {
        let qa = QuantizedArena::with_capacity(8, 100);
        assert_eq!(qa.dimension(), 8);
        assert_eq!(qa.active_count(), 0);
        // Pre-allocated capacity should be at least 8 * 100 = 800 bytes.
        assert!(qa.data.capacity() >= 800);
    }

    #[test]
    fn memory_bytes_is_reasonable() {
        let mut qa = QuantizedArena::new(4);
        let baseline = qa.memory_bytes();
        qa.push(&[1, 2, 3, 4]);
        // After pushing, memory should be at least as large as before.
        assert!(qa.memory_bytes() >= baseline);
    }
}
