// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Contiguous u8 storage for scalar-quantized vector codes.
//!
//! Mirrors [`crate::arena::VectorArena`] but stores `u8` quantized codes
//! instead of `f32` vectors. Each slot holds exactly `dimension` bytes
//! (one byte per original vector dimension after SQ8 quantization).

/// Contiguous u8 storage for scalar-quantized vector codes.
///
/// Each slot stores `dimension` bytes. Slot `i` occupies
/// `[i * dim .. (i + 1) * dim]` in the flat buffer.
///
/// Supports push, get, and free with slot reuse — same pattern as
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
            // Reuse a freed slot — overwrite in place.
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
    /// The underlying memory is not reclaimed — it will be overwritten by the
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
