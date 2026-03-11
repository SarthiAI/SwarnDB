// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Flat (contiguous) adjacency list for cache-friendly HNSW graph traversal.
//!
//! Instead of `Vec<Vec<VectorId>>` per node (which causes pointer chasing),
//! all neighbor IDs are stored in a single contiguous `Vec<VectorId>` buffer.
//! Accessing neighbors for a node at a given layer is a simple slice operation.

use std::collections::HashMap;
use std::convert::TryFrom;
use log;
use vf_core::types::VectorId;

use crate::hnsw_types::HnswNode;

/// Controls when compaction is performed after fragmentation exceeds the threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionMode {
    /// Compact immediately inside `maybe_compact()` (blocks the caller).
    Inline,
    /// Only flag that compaction is needed; the caller must later invoke
    /// `run_deferred_compaction()` from a background task or maintenance window.
    Deferred,
}

/// Per-node metadata describing where its neighbors live in the flat buffer.
#[derive(Debug, Clone)]
struct NodeLayout {
    /// Start position in the data buffer.
    offset: u32,
    /// Number of layers this node spans (0..num_layers-1).
    num_layers: u8,
    /// Number of neighbors stored per layer.
    layer_sizes: Vec<u16>,
    /// Whether this node has been logically deleted.
    deleted: bool,
}

impl NodeLayout {
    /// Total number of neighbor slots occupied by this node in the buffer.
    fn total_size(&self) -> u32 {
        self.layer_sizes.iter().map(|&s| s as u32).sum()
    }
}

/// Default fragmentation ratio threshold above which `maybe_compact()` triggers
/// a full compaction. 0.3 means compact when 30% of the buffer is wasted space.
const DEFAULT_FRAGMENTATION_THRESHOLD: f64 = 0.3;

/// A cache-friendly flat adjacency list that stores all HNSW neighbor lists
/// in a single contiguous `Vec<VectorId>` buffer.
///
/// # Layout
///
/// For a node with layers 0..L, its neighbors are laid out contiguously:
/// ```text
/// [layer_0_neighbors...][layer_1_neighbors...][...][layer_L_neighbors...]
/// ```
///
/// Retrieving neighbors for a given layer is an O(1) slice into this buffer.
///
/// # Fragmentation tracking
///
/// When nodes are deleted or neighbor lists change size, the old space becomes
/// dead (orphaned) slots in the buffer. The `wasted_slots` counter tracks this
/// and `maybe_compact()` automatically defragments when the fragmentation ratio
/// exceeds the configured threshold.
#[derive(Debug, Clone)]
pub struct FlatAdjacencyList {
    /// Contiguous storage for all neighbor IDs.
    data: Vec<VectorId>,
    /// Per-node layout metadata.
    index: HashMap<VectorId, NodeLayout>,
    /// Number of active (non-deleted) nodes.
    active_count: usize,
    /// Number of wasted (orphaned/dead) slots in the data buffer.
    wasted_slots: usize,
    /// Fragmentation ratio threshold for auto-compaction (0.0 .. 1.0).
    fragmentation_threshold: f64,
    /// Whether compaction runs inline or is deferred to a background pass.
    compaction_mode: CompactionMode,
    /// Flag set when deferred compaction is needed but hasn't run yet.
    needs_compaction: bool,
}

impl FlatAdjacencyList {
    /// Creates a new empty flat adjacency list with the default fragmentation threshold.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            index: HashMap::new(),
            active_count: 0,
            wasted_slots: 0,
            fragmentation_threshold: DEFAULT_FRAGMENTATION_THRESHOLD,
            compaction_mode: CompactionMode::Inline,
            needs_compaction: false,
        }
    }

    /// Creates a new empty flat adjacency list with a custom fragmentation threshold.
    ///
    /// `threshold` is clamped to the range `[0.01, 1.0]`.
    pub fn with_fragmentation_threshold(threshold: f64) -> Self {
        Self {
            fragmentation_threshold: threshold.clamp(0.01, 1.0),
            ..Self::new()
        }
    }

    /// Sets the compaction mode (inline vs deferred).
    pub fn set_compaction_mode(&mut self, mode: CompactionMode) {
        self.compaction_mode = mode;
    }

    /// Returns the current compaction mode.
    pub fn compaction_mode(&self) -> CompactionMode {
        self.compaction_mode
    }

    /// Sets the fragmentation threshold at runtime.
    ///
    /// `threshold` is clamped to the range `[0.01, 1.0]`.
    pub fn set_fragmentation_threshold(&mut self, threshold: f64) {
        self.fragmentation_threshold = threshold.clamp(0.01, 1.0);
    }

    /// Returns the current fragmentation threshold.
    pub fn fragmentation_threshold(&self) -> f64 {
        self.fragmentation_threshold
    }

    /// Returns `true` if deferred compaction has been flagged but not yet run.
    pub fn needs_compaction(&self) -> bool {
        self.needs_compaction
    }

    /// Inserts a node with its neighbors organized by layer.
    ///
    /// `neighbors_per_layer` is a slice of slices: one entry per layer,
    /// each containing the neighbor IDs for that layer.
    ///
    /// The neighbors are appended to the end of the contiguous buffer.
    pub fn insert_node(&mut self, id: VectorId, neighbors_per_layer: &[&[VectorId]]) {
        let offset = u32::try_from(self.data.len()).expect(
            "FlatAdjacencyList: data buffer offset exceeds u32::MAX"
        );
        let num_layers = u8::try_from(neighbors_per_layer.len()).expect(
            "FlatAdjacencyList: number of layers exceeds u8::MAX (255)"
        );
        let mut layer_sizes = Vec::with_capacity(neighbors_per_layer.len());

        for layer_neighbors in neighbors_per_layer {
            let size = u16::try_from(layer_neighbors.len()).expect(
                "FlatAdjacencyList: layer neighbor count exceeds u16::MAX (65535)"
            );
            layer_sizes.push(size);
            self.data.extend_from_slice(layer_neighbors);
        }

        // If the node already exists, its old entry becomes an orphaned gap.
        // Track the wasted space and adjust active_count.
        if let Some(old) = self.index.get(&id) {
            if !old.deleted {
                self.active_count -= 1;
                // Only count as newly wasted if not already deleted.
                // Deleted nodes had their space counted in remove_node().
                self.wasted_slots += old.total_size() as usize;
            }
        }

        self.index.insert(
            id,
            NodeLayout {
                offset,
                num_layers,
                layer_sizes,
                deleted: false,
            },
        );
        self.active_count += 1;
    }

    /// Returns the neighbors of `id` at the given `layer` as a contiguous slice.
    ///
    /// This is an O(1) operation — just pointer arithmetic into the flat buffer.
    /// Returns `None` if the node doesn't exist, is deleted, or the layer is out of range.
    pub fn get_neighbors(&self, id: VectorId, layer: usize) -> Option<&[VectorId]> {
        let layout = self.index.get(&id)?;
        if layout.deleted || layer >= layout.num_layers as usize {
            return None;
        }

        let start = layout.offset as usize
            + layout.layer_sizes[..layer]
                .iter()
                .map(|&s| s as usize)
                .sum::<usize>();
        let count = layout.layer_sizes[layer] as usize;

        Some(&self.data[start..start + count])
    }

    /// Updates the neighbors of `id` at the given `layer`.
    ///
    /// If the new neighbor count matches the old count, this is an in-place update
    /// (no reallocation). If the count differs, the node's entire allocation is
    /// invalidated and re-appended at the end of the buffer (the old space becomes
    /// a gap that can be reclaimed by `compact()`).
    ///
    /// Returns `false` if the node doesn't exist or is deleted.
    pub fn set_neighbors(
        &mut self,
        id: VectorId,
        layer: usize,
        neighbors: &[VectorId],
    ) -> bool {
        let layout = match self.index.get(&id) {
            Some(l) if !l.deleted && layer < l.num_layers as usize => l,
            _ => return false,
        };

        let old_count = layout.layer_sizes[layer] as usize;

        if neighbors.len() == old_count {
            // In-place update — same size, just overwrite the slice.
            let start = layout.offset as usize
                + layout.layer_sizes[..layer]
                    .iter()
                    .map(|&s| s as usize)
                    .sum::<usize>();
            self.data[start..start + old_count].copy_from_slice(neighbors);
            return true;
        }

        // Size changed — collect all layer data, mark old space as gap, re-append.
        let num_layers = layout.num_layers as usize;
        let old_total = layout.total_size() as usize;
        let mut all_layers: Vec<Vec<VectorId>> = Vec::with_capacity(num_layers);

        let mut cursor = layout.offset as usize;
        for l in 0..num_layers {
            let count = layout.layer_sizes[l] as usize;
            if l == layer {
                all_layers.push(neighbors.to_vec());
            } else {
                all_layers.push(self.data[cursor..cursor + count].to_vec());
            }
            cursor += count;
        }

        // Track the old allocation as wasted space.
        self.wasted_slots += old_total;

        // Re-append at end of buffer.
        let new_offset = u32::try_from(self.data.len()).expect(
            "FlatAdjacencyList: data buffer offset exceeds u32::MAX"
        );
        let mut new_layer_sizes = Vec::with_capacity(num_layers);
        for layer_data in &all_layers {
            let size = u16::try_from(layer_data.len()).expect(
                "FlatAdjacencyList: layer neighbor count exceeds u16::MAX (65535)"
            );
            new_layer_sizes.push(size);
            self.data.extend_from_slice(layer_data);
        }

        let layout = self.index.get_mut(&id).unwrap();
        layout.offset = new_offset;
        layout.layer_sizes = new_layer_sizes;

        self.maybe_compact();

        true
    }

    /// Marks a node as deleted (lazy deletion).
    ///
    /// The space in the buffer is not reclaimed until `compact()` is called.
    /// Returns `false` if the node doesn't exist or is already deleted.
    pub fn remove_node(&mut self, id: VectorId) -> bool {
        match self.index.get_mut(&id) {
            Some(layout) if !layout.deleted => {
                self.wasted_slots += layout.total_size() as usize;
                layout.deleted = true;
                self.active_count -= 1;
                true
            }
            _ => false,
        }
    }

    /// Returns the number of active (non-deleted) nodes.
    pub fn len(&self) -> usize {
        self.active_count
    }

    /// Returns `true` if there are no active nodes.
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// Returns the current fragmentation ratio (wasted slots / total buffer length).
    ///
    /// A value of 0.0 means no fragmentation; 1.0 means entirely wasted.
    /// Returns 0.0 if the buffer is empty.
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.wasted_slots as f64 / self.data.len() as f64
    }

    /// Returns the number of wasted (dead/orphaned) slots in the buffer.
    pub fn wasted_slot_count(&self) -> usize {
        self.wasted_slots
    }

    /// Compacts the buffer if the fragmentation ratio exceeds the configured threshold.
    ///
    /// In `Inline` mode, compaction runs immediately and `true` is returned.
    /// In `Deferred` mode, the `needs_compaction` flag is set and `false` is
    /// returned — the caller should later invoke `run_deferred_compaction()`.
    pub fn maybe_compact(&mut self) -> bool {
        if self.fragmentation_ratio() > self.fragmentation_threshold {
            match self.compaction_mode {
                CompactionMode::Inline => {
                    self.compact();
                    return true;
                }
                CompactionMode::Deferred => {
                    if !self.needs_compaction {
                        log::debug!(
                            "flat_adj: deferred compaction flagged (fragmentation {:.1}% > threshold {:.1}%)",
                            self.fragmentation_ratio() * 100.0,
                            self.fragmentation_threshold * 100.0,
                        );
                    }
                    self.needs_compaction = true;
                }
            }
        }
        false
    }

    /// Runs compaction if it was previously deferred by `maybe_compact()`.
    ///
    /// Returns `true` if compaction was performed.
    pub fn run_deferred_compaction(&mut self) -> bool {
        if self.needs_compaction {
            self.compact();
            return true;
        }
        false
    }

    /// Defragments the buffer by removing gaps from deleted nodes and
    /// re-appending relocations from `set_neighbors` size changes.
    ///
    /// After compaction, the buffer contains only live data with no gaps,
    /// and all offsets are updated accordingly.
    pub fn compact(&mut self) {
        let old_len = self.data.len();
        let old_ratio = self.fragmentation_ratio();
        let reclaimed = self.wasted_slots;

        let mut new_data: Vec<VectorId> = Vec::with_capacity(self.data.len());

        // Remove deleted nodes from the index entirely.
        self.index.retain(|_, layout| !layout.deleted);

        // Sort active nodes by their current offset so we preserve relative order
        // (helps with cache locality if nodes were inserted in a useful order).
        let mut entries: Vec<(VectorId, u32)> = self
            .index
            .iter()
            .map(|(&id, layout)| (id, layout.offset))
            .collect();
        entries.sort_by_key(|&(_, offset)| offset);

        for (id, _) in entries {
            let layout = self.index.get_mut(&id).unwrap();
            let old_start = layout.offset as usize;
            let total = layout.total_size() as usize;

            let new_offset = u32::try_from(new_data.len()).expect(
                "FlatAdjacencyList::compact: data buffer offset exceeds u32::MAX"
            );
            new_data.extend_from_slice(&self.data[old_start..old_start + total]);
            layout.offset = new_offset;
        }

        self.data = new_data;
        self.wasted_slots = 0;
        self.needs_compaction = false;

        log::info!(
            "flat_adj compaction: buffer {} -> {}, reclaimed {} wasted slots, fragmentation {:.1}% -> 0%",
            old_len,
            self.data.len(),
            reclaimed,
            old_ratio * 100.0,
        );
    }

    /// Builds a `FlatAdjacencyList` from existing HNSW node data.
    ///
    /// Iterates all nodes and flattens their `Vec<Vec<VectorId>>` neighbor lists
    /// into the contiguous buffer.
    pub(crate) fn from_hnsw_nodes(nodes: &HashMap<VectorId, HnswNode>) -> Self {
        let mut flat = Self::new();
        // Pre-allocate a reasonable buffer size.
        flat.data.reserve(nodes.len() * 32);

        for (&id, node) in nodes {
            let neighbors = &node.neighbors;
            let layer_refs: Vec<&[VectorId]> =
                neighbors.iter().map(|layer| layer.as_slice()).collect();
            flat.insert_node(id, &layer_refs);
        }

        flat
    }

    /// Returns the total capacity of the underlying data buffer.
    pub fn buffer_capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns the current length of the data buffer (including gaps from deletions).
    pub fn buffer_len(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of layers for a given node, or `None` if not found/deleted.
    pub fn num_layers(&self, id: VectorId) -> Option<u8> {
        self.index
            .get(&id)
            .filter(|l| !l.deleted)
            .map(|l| l.num_layers)
    }
}

impl Default for FlatAdjacencyList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get_neighbors() {
        let mut adj = FlatAdjacencyList::new();
        let layer0: &[VectorId] = &[10, 20, 30];
        let layer1: &[VectorId] = &[40, 50];
        adj.insert_node(1, &[layer0, layer1]);

        assert_eq!(adj.len(), 1);
        assert_eq!(adj.get_neighbors(1, 0), Some(&[10u64, 20, 30][..]));
        assert_eq!(adj.get_neighbors(1, 1), Some(&[40u64, 50][..]));
        assert_eq!(adj.get_neighbors(1, 2), None);
    }

    #[test]
    fn test_set_neighbors_same_size() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10, 20, 30]]);

        assert!(adj.set_neighbors(1, 0, &[100, 200, 300]));
        assert_eq!(adj.get_neighbors(1, 0), Some(&[100u64, 200, 300][..]));
        // Buffer should not have grown (in-place update).
        assert_eq!(adj.buffer_len(), 3);
    }

    #[test]
    fn test_set_neighbors_different_size() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10, 20, 30], &[40]]);

        assert!(adj.set_neighbors(1, 0, &[100, 200]));
        assert_eq!(adj.get_neighbors(1, 0), Some(&[100u64, 200][..]));
        // Layer 1 should be preserved.
        assert_eq!(adj.get_neighbors(1, 1), Some(&[40u64][..]));
    }

    #[test]
    fn test_remove_and_compact() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10, 20]]);
        adj.insert_node(2, &[&[30, 40]]);
        adj.insert_node(3, &[&[50, 60]]);

        assert_eq!(adj.len(), 3);
        assert!(adj.remove_node(2));
        assert_eq!(adj.len(), 2);
        assert_eq!(adj.get_neighbors(2, 0), None);

        // Buffer still has 6 entries (gap from node 2).
        assert_eq!(adj.buffer_len(), 6);

        adj.compact();
        // After compaction, buffer should only have 4 entries.
        assert_eq!(adj.buffer_len(), 4);
        assert_eq!(adj.get_neighbors(1, 0), Some(&[10u64, 20][..]));
        assert_eq!(adj.get_neighbors(3, 0), Some(&[50u64, 60][..]));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut adj = FlatAdjacencyList::new();
        assert!(!adj.remove_node(999));
    }

    #[test]
    fn test_double_remove() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10]]);
        assert!(adj.remove_node(1));
        assert!(!adj.remove_node(1));
    }

    #[test]
    fn test_is_empty() {
        let mut adj = FlatAdjacencyList::new();
        assert!(adj.is_empty());
        adj.insert_node(1, &[&[10]]);
        assert!(!adj.is_empty());
    }

    #[test]
    fn test_num_layers() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10], &[20], &[30]]);
        assert_eq!(adj.num_layers(1), Some(3));
        assert_eq!(adj.num_layers(999), None);
    }

    #[test]
    fn test_fragmentation_tracking_on_remove() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10, 20]]);
        adj.insert_node(2, &[&[30, 40, 50]]);
        assert_eq!(adj.wasted_slot_count(), 0);
        assert_eq!(adj.fragmentation_ratio(), 0.0);

        adj.remove_node(1);
        // Node 1 had 2 slots, total buffer is 5 slots.
        assert_eq!(adj.wasted_slot_count(), 2);
        assert!((adj.fragmentation_ratio() - 2.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fragmentation_tracking_on_set_neighbors_resize() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10, 20, 30]]); // 3 slots
        assert_eq!(adj.wasted_slot_count(), 0);

        // Resize from 3 to 2 — old 3 slots become wasted.
        adj.set_neighbors(1, 0, &[100, 200]);
        assert_eq!(adj.wasted_slot_count(), 3);
        // Buffer: 3 (old dead) + 2 (new) = 5 total.
        assert!((adj.fragmentation_ratio() - 3.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fragmentation_tracking_on_overwrite_insert() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10, 20]]); // 2 slots
        adj.insert_node(1, &[&[30, 40, 50]]); // overwrite: old 2 become wasted
        assert_eq!(adj.wasted_slot_count(), 2);
        assert_eq!(adj.len(), 1);
    }

    #[test]
    fn test_compact_resets_wasted_slots() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10, 20]]);
        adj.insert_node(2, &[&[30, 40]]);
        adj.remove_node(1);
        assert_eq!(adj.wasted_slot_count(), 2);

        adj.compact();
        assert_eq!(adj.wasted_slot_count(), 0);
        assert_eq!(adj.fragmentation_ratio(), 0.0);
        assert_eq!(adj.buffer_len(), 2); // only node 2's data
    }

    #[test]
    fn test_maybe_compact_threshold() {
        let mut adj = FlatAdjacencyList::with_fragmentation_threshold(0.5);
        adj.insert_node(1, &[&[10, 20]]);
        adj.insert_node(2, &[&[30, 40]]);
        adj.insert_node(3, &[&[50, 60]]);

        // Remove node 1: wasted=2, total=6, ratio=0.33 < 0.5 threshold.
        adj.remove_node(1);
        assert!(!adj.maybe_compact()); // should NOT compact

        // Remove node 2: wasted=4, total=6, ratio=0.67 > 0.5 threshold.
        adj.remove_node(2);
        assert!(adj.maybe_compact()); // SHOULD compact
        assert_eq!(adj.wasted_slot_count(), 0);
        assert_eq!(adj.buffer_len(), 2); // only node 3
    }

    #[test]
    fn test_set_neighbors_same_size_no_fragmentation() {
        let mut adj = FlatAdjacencyList::new();
        adj.insert_node(1, &[&[10, 20, 30]]);
        adj.set_neighbors(1, 0, &[100, 200, 300]);
        // In-place update should NOT increase wasted slots.
        assert_eq!(adj.wasted_slot_count(), 0);
    }
}
