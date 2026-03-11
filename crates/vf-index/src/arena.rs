// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Arena allocator for HNSW graph nodes.
//!
//! Provides cache-friendly, contiguous storage for graph nodes to reduce heap
//! fragmentation. Nodes are allocated in fixed-size chunks; existing chunks are
//! never reallocated so references obtained via index remain stable.

use std::collections::HashSet;
use vf_core::types::VectorId;

/// Compact handle into the arena. Upper bits encode the chunk index, lower bits
/// encode the offset within that chunk. The split point is determined by the
/// chunk size (which must be a power of two).
pub type ArenaNodeId = u32;

/// A single node stored inside the arena.
#[derive(Debug, Clone)]
pub struct ArenaNode {
    pub vector: Vec<f32>,
    pub neighbors: Vec<Vec<VectorId>>,
}

/// Typed arena that allocates [`ArenaNode`]s in power-of-two sized chunks.
///
/// # Design
///
/// * Chunks are `Vec<ArenaNode>` pre-allocated to `chunk_size` capacity.
/// * New chunks are appended when the current one is full — existing chunks are
///   never moved or reallocated.
/// * [`ArenaNodeId`] encodes `(chunk_index, offset)` in a single `u32` using
///   bit-shifting so that `get` / `get_mut` are O(1).
///
/// The arena is `Send` (each `Vec` owns its data) but not necessarily `Sync`;
/// callers are expected to protect it with an external lock (e.g. `RwLock`).
pub struct NodeArena {
    chunks: Vec<Vec<ArenaNode>>,
    chunk_size: usize,
    /// log2(chunk_size) — used for bit-shift encoding of ArenaNodeId.
    shift: u32,
    /// Bitmask for extracting offset from an ArenaNodeId (`chunk_size - 1`).
    mask: u32,
    /// Total number of nodes allocated so far.
    len: usize,
}

impl NodeArena {
    /// Default chunk size (number of nodes per chunk).
    pub const DEFAULT_CHUNK_SIZE: usize = 4096;

    /// Creates a new arena.
    ///
    /// `chunk_size` **must** be a power of two so that the id-encoding scheme
    /// works via bit-shifts. Panics otherwise.
    pub fn new(chunk_size: usize) -> Self {
        assert!(
            chunk_size.is_power_of_two(),
            "NodeArena chunk_size must be a power of two, got {chunk_size}"
        );
        assert!(
            chunk_size <= (u32::MAX as usize + 1),
            "chunk_size is too large"
        );

        let shift = chunk_size.trailing_zeros();
        let mask = (chunk_size - 1) as u32;

        Self {
            chunks: Vec::new(),
            chunk_size,
            shift,
            mask,
            len: 0,
        }
    }

    /// Allocates a node in the arena and returns its compact id.
    ///
    /// * `vector` — the embedding vector for this node.
    /// * `max_level` — the HNSW level assigned to this node (0-based). Neighbor
    ///   vecs are pre-allocated for levels `0..=max_level`.
    /// * `max_neighbors` — capacity hint for each level's neighbor list.
    pub fn alloc(
        &mut self,
        vector: Vec<f32>,
        max_level: usize,
        max_neighbors: usize,
    ) -> ArenaNodeId {
        // Ensure there is room in the current chunk.
        if self.chunks.is_empty() || self.chunks.last().unwrap().len() == self.chunk_size {
            let chunk = Vec::with_capacity(self.chunk_size);
            // Reserve the full capacity up-front so the chunk never reallocates.
            debug_assert_eq!(chunk.capacity(), self.chunk_size);
            self.chunks.push(chunk);
        }

        let chunk_idx = self.chunks.len() - 1;
        debug_assert!(chunk_idx < (1usize << (32 - self.shift)), "arena chunk index overflow");
        let offset = self.chunks[chunk_idx].len();

        // Pre-allocate neighbor vecs for each level.
        let neighbors: Vec<Vec<VectorId>> = (0..=max_level)
            .map(|_| Vec::with_capacity(max_neighbors))
            .collect();

        let node = ArenaNode { vector, neighbors };
        self.chunks[chunk_idx].push(node);
        self.len += 1;

        Self::encode_id(chunk_idx, offset, self.shift)
    }

    /// O(1) immutable access by arena id.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    pub fn get(&self, id: ArenaNodeId) -> &ArenaNode {
        let (chunk_idx, offset) = self.decode_id(id);
        &self.chunks[chunk_idx][offset]
    }

    /// O(1) mutable access by arena id.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, id: ArenaNodeId) -> &mut ArenaNode {
        let (chunk_idx, offset) = self.decode_id(id);
        &mut self.chunks[chunk_idx][offset]
    }

    /// Returns the total number of allocated nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the arena contains no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drops all nodes and reclaims memory.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.len = 0;
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Encode a `(chunk_index, offset)` pair into a single `ArenaNodeId`.
    ///
    /// Uses checked casts to prevent silent truncation on 64-bit systems.
    ///
    /// # Panics
    /// Panics if `chunk_idx` or `offset` exceed `u32::MAX`.
    #[inline]
    fn encode_id(chunk_idx: usize, offset: usize, shift: u32) -> ArenaNodeId {
        let chunk_u32: u32 = chunk_idx
            .try_into()
            .expect("NodeArena: chunk_idx exceeds u32::MAX");
        let offset_u32: u32 = offset
            .try_into()
            .expect("NodeArena: offset exceeds u32::MAX");
        (chunk_u32 << shift) | offset_u32
    }

    /// Decode an `ArenaNodeId` back into `(chunk_index, offset)`.
    #[inline]
    fn decode_id(&self, id: ArenaNodeId) -> (usize, usize) {
        let chunk_idx = (id >> self.shift) as usize;
        let offset = (id & self.mask) as usize;
        (chunk_idx, offset)
    }
}

// ── VectorArena: contiguous f32 storage for HNSW node vectors ──────────

/// Contiguous arena for storing fixed-dimension f32 vectors.
///
/// All vectors are packed end-to-end in a single `Vec<f32>` buffer so that
/// sequential access patterns (e.g., during HNSW search) benefit from CPU
/// cache prefetching. Each vector occupies exactly `dimension` f32 slots,
/// and is addressed by a zero-based slot index.
///
/// The arena supports:
/// - O(1) allocation (append to the end of the buffer)
/// - O(1) access by slot index
/// - Slot reuse via a free-list to handle deletions without compaction
pub struct VectorArena {
    /// Flat buffer: slot `i` occupies `[i*dim .. (i+1)*dim]`.
    data: Vec<f32>,
    /// Number of f32 elements per vector.
    dimension: usize,
    /// Total number of allocated slots (including freed ones still in buffer).
    slot_count: usize,
    /// Free-list of reusable slot indices from prior deletions.
    free_slots: Vec<usize>,
    /// Set of currently freed slot indices for double-free and use-after-free protection.
    freed_set: HashSet<usize>,
}

impl VectorArena {
    /// Creates a new vector arena for the given dimension.
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "VectorArena dimension must be > 0");
        Self {
            data: Vec::new(),
            dimension,
            slot_count: 0,
            free_slots: Vec::new(),
            freed_set: HashSet::new(),
        }
    }

    /// Creates a new vector arena pre-allocated for `capacity` vectors.
    ///
    /// # Panics
    /// Panics if `dimension * capacity` overflows `usize`.
    pub fn with_capacity(dimension: usize, capacity: usize) -> Self {
        assert!(dimension > 0, "VectorArena dimension must be > 0");
        let total_capacity = dimension.checked_mul(capacity).unwrap_or_else(|| {
            panic!(
                "VectorArena::with_capacity: overflow computing dimension ({}) * capacity ({})",
                dimension, capacity
            )
        });
        Self {
            data: Vec::with_capacity(total_capacity),
            dimension,
            slot_count: 0,
            free_slots: Vec::new(),
            freed_set: HashSet::new(),
        }
    }

    /// Stores a vector in the arena and returns its slot index.
    ///
    /// Reuses a previously freed slot if available, otherwise appends.
    ///
    /// # Panics
    /// Panics if `vector.len() != self.dimension`.
    pub fn push(&mut self, vector: &[f32]) -> usize {
        assert_eq!(
            vector.len(),
            self.dimension,
            "VectorArena::push: expected dimension {}, got {}",
            self.dimension,
            vector.len()
        );

        if let Some(slot) = self.free_slots.pop() {
            // Reuse a freed slot — overwrite in place.
            self.freed_set.remove(&slot);
            let start = slot * self.dimension;
            self.data[start..start + self.dimension].copy_from_slice(vector);
            slot
        } else {
            // Append a new slot.
            let slot = self.slot_count;
            self.data.extend_from_slice(vector);
            self.slot_count += 1;
            slot
        }
    }

    /// Returns an immutable slice to the vector at the given slot.
    ///
    /// # Panics
    /// Panics if `slot >= slot_count`.
    #[inline]
    pub fn get(&self, slot: usize) -> &[f32] {
        let start = slot * self.dimension;
        &self.data[start..start + self.dimension]
    }

    /// Marks a slot as free for future reuse.
    ///
    /// The underlying memory is not reclaimed — it will be overwritten by the
    /// next `push` that reuses this slot. Callers must not access the slot
    /// after freeing it.
    pub fn free(&mut self, slot: usize) {
        assert!(slot < self.slot_count, "VectorArena::free: slot {} out of bounds (slot_count={})", slot, self.slot_count);
        assert!(
            !self.freed_set.contains(&slot),
            "VectorArena::free: double-free detected on slot {}",
            slot
        );
        self.freed_set.insert(slot);
        self.free_slots.push(slot);
    }

    /// Returns the vector dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of active (non-freed) vectors.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.slot_count - self.free_slots.len()
    }

    /// Drops all data and resets the arena.
    pub fn clear(&mut self) {
        self.data.clear();
        self.free_slots.clear();
        self.freed_set.clear();
        self.slot_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_get() {
        let mut arena = NodeArena::new(4); // small chunk for testing
        let id0 = arena.alloc(vec![1.0, 2.0, 3.0], 2, 16);
        let id1 = arena.alloc(vec![4.0, 5.0, 6.0], 0, 32);

        assert_eq!(arena.len(), 2);
        assert!(!arena.is_empty());

        let node0 = arena.get(id0);
        assert_eq!(node0.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(node0.neighbors.len(), 3); // levels 0, 1, 2

        let node1 = arena.get(id1);
        assert_eq!(node1.vector, vec![4.0, 5.0, 6.0]);
        assert_eq!(node1.neighbors.len(), 1); // level 0 only
    }

    #[test]
    fn get_mut_modifies_node() {
        let mut arena = NodeArena::new(4);
        let id = arena.alloc(vec![1.0], 1, 8);

        arena.get_mut(id).neighbors[0].push(42);
        assert_eq!(arena.get(id).neighbors[0], vec![42]);
    }

    #[test]
    fn grows_across_chunks() {
        let mut arena = NodeArena::new(4);
        let mut ids = Vec::new();
        for i in 0..10 {
            ids.push(arena.alloc(vec![i as f32], 0, 4));
        }
        assert_eq!(arena.len(), 10);
        // 10 nodes / 4 per chunk = 3 chunks (4 + 4 + 2)
        assert_eq!(arena.chunks.len(), 3);

        // Verify each node is accessible and correct.
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(arena.get(id).vector, vec![i as f32]);
        }
    }

    #[test]
    fn clear_resets_arena() {
        let mut arena = NodeArena::new(4);
        arena.alloc(vec![1.0], 0, 4);
        arena.alloc(vec![2.0], 0, 4);
        arena.clear();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn non_power_of_two_panics() {
        NodeArena::new(3);
    }

    #[test]
    fn default_chunk_size_is_power_of_two() {
        assert!(NodeArena::DEFAULT_CHUNK_SIZE.is_power_of_two());
    }

    #[test]
    fn id_encoding_roundtrip() {
        let arena = NodeArena::new(8);
        // chunk 5, offset 3
        let id = NodeArena::encode_id(5, 3, arena.shift);
        let (c, o) = arena.decode_id(id);
        assert_eq!(c, 5);
        assert_eq!(o, 3);
    }

    #[test]
    fn pre_allocated_neighbor_capacity() {
        let mut arena = NodeArena::new(4);
        let id = arena.alloc(vec![0.0], 3, 16);
        let node = arena.get(id);
        assert_eq!(node.neighbors.len(), 4); // levels 0..=3
        for level_neighbors in &node.neighbors {
            assert!(level_neighbors.capacity() >= 16);
        }
    }

    // ── VectorArena tests ──────────────────────────────────────────────

    #[test]
    fn vector_arena_push_and_get() {
        let mut va = VectorArena::new(3);
        let s0 = va.push(&[1.0, 2.0, 3.0]);
        let s1 = va.push(&[4.0, 5.0, 6.0]);
        assert_eq!(va.get(s0), &[1.0, 2.0, 3.0]);
        assert_eq!(va.get(s1), &[4.0, 5.0, 6.0]);
        assert_eq!(va.active_count(), 2);
    }

    #[test]
    fn vector_arena_free_and_reuse() {
        let mut va = VectorArena::new(2);
        let s0 = va.push(&[1.0, 2.0]);
        let s1 = va.push(&[3.0, 4.0]);
        va.free(s0);
        assert_eq!(va.active_count(), 1);
        let s2 = va.push(&[5.0, 6.0]);
        assert_eq!(s2, s0); // reuses freed slot
        assert_eq!(va.get(s2), &[5.0, 6.0]);
        assert_eq!(va.get(s1), &[3.0, 4.0]);
    }

    #[test]
    fn vector_arena_clear() {
        let mut va = VectorArena::new(2);
        va.push(&[1.0, 2.0]);
        va.clear();
        assert_eq!(va.active_count(), 0);
    }

    #[test]
    #[should_panic(expected = "dimension must be > 0")]
    fn vector_arena_zero_dim_panics() {
        VectorArena::new(0);
    }

    #[test]
    #[should_panic(expected = "expected dimension 3, got 2")]
    fn vector_arena_wrong_dim_panics() {
        let mut va = VectorArena::new(3);
        va.push(&[1.0, 2.0]);
    }
}
