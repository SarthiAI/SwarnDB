// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Arena allocator for HNSW graph nodes.
//!
//! Provides cache-friendly, contiguous storage for graph nodes to reduce heap
//! fragmentation. Nodes are allocated in fixed-size chunks; existing chunks are
//! never reallocated so references obtained via index remain stable.

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
    #[inline]
    fn encode_id(chunk_idx: usize, offset: usize, shift: u32) -> ArenaNodeId {
        ((chunk_idx as u32) << shift) | (offset as u32)
    }

    /// Decode an `ArenaNodeId` back into `(chunk_index, offset)`.
    #[inline]
    fn decode_id(&self, id: ArenaNodeId) -> (usize, usize) {
        let chunk_idx = (id >> self.shift) as usize;
        let offset = (id & self.mask) as usize;
        (chunk_idx, offset)
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
}
