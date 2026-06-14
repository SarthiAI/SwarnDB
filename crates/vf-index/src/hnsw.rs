// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use dashmap::DashMap;
use ordered_float::OrderedFloat;
use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use vf_core::distance::DistanceMetric;
use vf_core::types::{DistanceMetricType, ScoredResult, VectorId};

use crate::arena::VectorArena;
use crate::flat_adj::FlatAdjacencyList;
use crate::hnsw_delta::{HnswDeltaEntry, HnswDeltaOp, HnswDeltaWriter};
use crate::hnsw_persistence::{
    deserialize_topology_mmap, serialize_topology, validate_envelope_at_path,
    HnswTopologySnapshot, TopologyNode,
};
use crate::hnsw_types::HnswNode;
use crate::prefetch::{prefetch_neighbors, prefetch_vector};
use crate::traits::{IndexError, IndexRecoveryStrategy, PersistableIndex, RestoreOutcome, VectorIndex};

#[derive(Debug, Clone)]
pub struct HnswParams {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub m_l: f64,
    pub max_level_cap: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: 2 * m,
            ef_construction: 200,
            ef_search: 50,
            m_l: 1.0 / (m as f64).ln(),
            max_level_cap: 16,
        }
    }
}

impl HnswParams {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            m0: 2 * m,
            ef_construction,
            ef_search,
            m_l: 1.0 / (m as f64).ln(),
            max_level_cap: 16,
        }
    }
}

/// Mutable inner state protected by RwLock.
struct HnswInner {
    nodes: HashMap<VectorId, HnswNode>,
    /// Contiguous arena storing all node vectors for cache-friendly access.
    vectors: VectorArena,
    entry_point: Option<VectorId>,
    max_level: usize,
    rng: StdRng,
    /// Optional flat adjacency list for cache-friendly neighbor access during search.
    /// Populated by calling `HnswIndex::compact()` after the graph is built.
    /// Automatically invalidated (set to `None`) on any graph mutation (add/remove).
    flat_adj: Option<FlatAdjacencyList>,
}

// In-flight new row used by bulk_add_from_slice_iter. The `vector` field
// borrows directly from caller-owned memory (typically an mmap'd file)
// for the duration of the call; no per-row Vec<f32> allocation is made.
// See ADR-001 Decision 3.b.
struct NewRowState<'mmap> {
    vector: &'mmap [f32],
    level: usize,
    neighbors: Vec<Mutex<Vec<VectorId>>>,
}

// Per-worker scratch buffers reused across searches in the bulk-insert parallel phase.
// Each rayon worker allocates these once and clears them between calls, eliminating
// per-call HashSet and BinaryHeap allocations.
thread_local! {
    static SCRATCH_VISITED: RefCell<HashSet<VectorId>> = RefCell::new(HashSet::new());
    static SCRATCH_CANDIDATES: RefCell<BinaryHeap<Reverse<(OrderedFloat<f32>, VectorId)>>> =
        RefCell::new(BinaryHeap::new());
    static SCRATCH_RESULTS: RefCell<BinaryHeap<(OrderedFloat<f32>, VectorId)>> =
        RefCell::new(BinaryHeap::new());
}

// Dedicated scratch buffers for the runtime search path, isolated from the bulk-insert slots
// above to prevent any re-entrant RefCell borrow across concurrent search and bulk insert.
thread_local! {
    static SCRATCH_SEARCH_VISITED: RefCell<HashSet<VectorId>> = RefCell::new(HashSet::new());
    static SCRATCH_SEARCH_CANDIDATES: RefCell<BinaryHeap<Reverse<(OrderedFloat<f32>, VectorId)>>> =
        RefCell::new(BinaryHeap::new());
    static SCRATCH_SEARCH_RESULTS: RefCell<BinaryHeap<(OrderedFloat<f32>, VectorId)>> =
        RefCell::new(BinaryHeap::new());
}

// Return freed per-batch scratch to the OS. Under jemalloc (ADR-025) this is an
// explicit arena purge; on every other target it is a no-op (background decay or
// the system allocator handles release). Re-exported at the crate root as
// `vf_index::release_to_os` so chunked ingest paths outside this crate can call
// it between batches. The intent of the former glibc-only malloc_trim is now
// carried by purge_allocator_arenas below.
#[inline]
pub fn release_to_os() {
    purge_allocator_arenas();
}

// Purge all jemalloc arenas so freed pages return to the OS immediately rather
// than waiting for the decay timer. Invoked right after the optimize step on the
// bulk paths. A no-op on non-jemalloc targets.
//
// 4096 == MALLCTL_ARENAS_ALL (the special index meaning "every arena").
#[inline]
#[cfg(all(not(target_env = "msvc"), any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn purge_allocator_arenas() {
    // SAFETY: the mallctl reads only allocator-internal state and never touches
    // any live allocation owned by us. Errors are ignored; a failed purge just
    // leaves the pages for the background decay thread.
    unsafe {
        tikv_jemalloc_ctl::raw::write(b"arena.4096.purge\0", ()).ok();
    }
}

#[inline]
#[cfg(not(all(not(target_env = "msvc"), any(target_arch = "x86_64", target_arch = "aarch64"))))]
pub fn purge_allocator_arenas() {}

pub struct HnswIndex {
    inner: RwLock<HnswInner>,
    params: HnswParams,
    metric: DistanceMetricType,
    distance_fn: DistanceMetric,
    dimension: usize,
    /// Optional delta writer for incremental persistence between base snapshots.
    /// Kept outside the inner RwLock to avoid holding it during I/O.
    delta_writer: Mutex<Option<HnswDeltaWriter>>,
}

const _: () = { fn _assert_send_sync<T: Send + Sync>() {} fn _check() { _assert_send_sync::<HnswIndex>(); } };

impl HnswIndex {
    pub fn new(dimension: usize, metric: DistanceMetricType, params: HnswParams) -> Self {
        Self {
            inner: RwLock::new(HnswInner {
                nodes: HashMap::new(),
                vectors: VectorArena::new(dimension),
                entry_point: None,
                max_level: 0,
                rng: StdRng::from_entropy(),
                flat_adj: None,
            }),
            params,
            metric,
            distance_fn: DistanceMetric::from_metric_type(metric),
            dimension,
            delta_writer: Mutex::new(None),
        }
    }

    pub fn with_defaults(dimension: usize, metric: DistanceMetricType) -> Self {
        Self::new(dimension, metric, HnswParams::default())
    }

    /// Magic bytes identifying the HNSW binary format.
    const MAGIC: &'static [u8; 4] = b"HNSW";
    /// Current serialization format version.
    const VERSION: u32 = 1;

    fn metric_to_u8(metric: DistanceMetricType) -> u8 {
        match metric {
            DistanceMetricType::Cosine => 0,
            DistanceMetricType::Euclidean => 1,
            DistanceMetricType::DotProduct => 2,
            DistanceMetricType::Manhattan => 3,
        }
    }

    fn metric_from_u8(byte: u8) -> Result<DistanceMetricType, IndexError> {
        match byte {
            0 => Ok(DistanceMetricType::Cosine),
            1 => Ok(DistanceMetricType::Euclidean),
            2 => Ok(DistanceMetricType::DotProduct),
            3 => Ok(DistanceMetricType::Manhattan),
            _ => Err(IndexError::Internal(format!(
                "serialization error: unknown distance metric byte {}",
                byte
            ))),
        }
    }

    /// Serialize the HNSW graph to a binary format.
    pub fn serialize(&self, writer: &mut impl IoWrite) -> Result<(), IndexError> {
        let wrap = |e: std::io::Error| IndexError::Internal(format!("serialization error: {}", e));
        let inner = self.inner.read();

        // --- Header (65 bytes) ---
        writer.write_all(Self::MAGIC).map_err(wrap)?;
        writer.write_all(&Self::VERSION.to_le_bytes()).map_err(wrap)?;
        writer
            .write_all(&(self.dimension as u32).to_le_bytes())
            .map_err(wrap)?;
        writer
            .write_all(&[Self::metric_to_u8(self.metric)])
            .map_err(wrap)?;
        writer
            .write_all(&(inner.nodes.len() as u64).to_le_bytes())
            .map_err(wrap)?;
        let ep = inner.entry_point.unwrap_or(u64::MAX);
        writer.write_all(&ep.to_le_bytes()).map_err(wrap)?;
        writer
            .write_all(&(inner.max_level as u32).to_le_bytes())
            .map_err(wrap)?;
        writer
            .write_all(&(self.params.m as u32).to_le_bytes())
            .map_err(wrap)?;
        writer
            .write_all(&(self.params.m0 as u32).to_le_bytes())
            .map_err(wrap)?;
        writer
            .write_all(&(self.params.ef_construction as u32).to_le_bytes())
            .map_err(wrap)?;
        writer
            .write_all(&(self.params.ef_search as u32).to_le_bytes())
            .map_err(wrap)?;
        writer.write_all(&[0u8; 16]).map_err(wrap)?;

        // --- Per node ---
        for (&id, node) in &inner.nodes {
            writer.write_all(&id.to_le_bytes()).map_err(wrap)?;
            let level = node.max_level() as u32;
            writer.write_all(&level.to_le_bytes()).map_err(wrap)?;
            let vec_data = inner.vectors.get(node.vector_slot);
            for &val in vec_data {
                writer.write_all(&val.to_le_bytes()).map_err(wrap)?;
            }
            for layer_neighbors in &node.neighbors {
                let count = layer_neighbors.len() as u32;
                writer.write_all(&count.to_le_bytes()).map_err(wrap)?;
                for &neighbor_id in layer_neighbors {
                    writer
                        .write_all(&neighbor_id.to_le_bytes())
                        .map_err(wrap)?;
                }
            }
        }

        Ok(())
    }

    /// Deserialize an HNSW graph from a binary format.
    pub fn deserialize(reader: &mut impl IoRead) -> Result<Self, IndexError> {
        let wrap = |e: std::io::Error| IndexError::Internal(format!("serialization error: {}", e));

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(wrap)?;
        if &magic != Self::MAGIC {
            return Err(IndexError::Internal(
                "serialization error: invalid magic bytes".to_string(),
            ));
        }

        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4).map_err(wrap)?;
        let version = u32::from_le_bytes(buf4);
        if version != Self::VERSION {
            return Err(IndexError::Internal(format!(
                "serialization error: unsupported version {}",
                version
            )));
        }

        reader.read_exact(&mut buf4).map_err(wrap)?;
        let dimension = u32::from_le_bytes(buf4) as usize;

        let mut metric_byte = [0u8; 1];
        reader.read_exact(&mut metric_byte).map_err(wrap)?;
        let metric = Self::metric_from_u8(metric_byte[0])?;

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8).map_err(wrap)?;
        let node_count = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8).map_err(wrap)?;
        let ep_raw = u64::from_le_bytes(buf8);
        let entry_point = if ep_raw == u64::MAX { None } else { Some(ep_raw) };

        reader.read_exact(&mut buf4).map_err(wrap)?;
        let max_level = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4).map_err(wrap)?;
        let m = u32::from_le_bytes(buf4) as usize;
        reader.read_exact(&mut buf4).map_err(wrap)?;
        let m0 = u32::from_le_bytes(buf4) as usize;
        reader.read_exact(&mut buf4).map_err(wrap)?;
        let ef_construction = u32::from_le_bytes(buf4) as usize;
        reader.read_exact(&mut buf4).map_err(wrap)?;
        let ef_search = u32::from_le_bytes(buf4) as usize;

        let mut reserved = [0u8; 16];
        reader.read_exact(&mut reserved).map_err(wrap)?;

        let params = HnswParams {
            m,
            m0,
            ef_construction,
            ef_search,
            m_l: 1.0 / (m as f64).ln(),
            max_level_cap: 16,
        };

        let mut nodes = HashMap::with_capacity(node_count);
        let mut vectors = VectorArena::with_capacity(dimension, node_count);
        for _ in 0..node_count {
            reader.read_exact(&mut buf8).map_err(wrap)?;
            let id = u64::from_le_bytes(buf8);

            reader.read_exact(&mut buf4).map_err(wrap)?;
            let level = u32::from_le_bytes(buf4) as usize;

            let mut vector = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                reader.read_exact(&mut buf4).map_err(wrap)?;
                vector.push(f32::from_le_bytes(buf4));
            }

            let mut neighbors = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                reader.read_exact(&mut buf4).map_err(wrap)?;
                let count = u32::from_le_bytes(buf4) as usize;
                let mut layer_neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    reader.read_exact(&mut buf8).map_err(wrap)?;
                    layer_neighbors.push(u64::from_le_bytes(buf8));
                }
                neighbors.push(layer_neighbors);
            }

            let slot = vectors.push(&vector);
            let node = HnswNode { vector_slot: slot, neighbors };
            nodes.insert(id, node);
        }

        Ok(Self {
            inner: RwLock::new(HnswInner {
                nodes,
                vectors,
                entry_point,
                max_level,
                rng: StdRng::seed_from_u64(42),
                flat_adj: None,
            }),
            params,
            metric,
            distance_fn: DistanceMetric::from_metric_type(metric),
            dimension,
            delta_writer: Mutex::new(None),
        })
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance_fn.compute(a, b)
    }

    fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m0
        } else {
            self.params.m
        }
    }

    // ── Internal helpers that operate on HnswInner directly ──────────────

    fn random_level(inner: &mut HnswInner, params: &HnswParams) -> usize {
        let uniform: f64 = inner.rng.gen_range(f64::MIN_POSITIVE..1.0);
        let level = (-uniform.ln() * params.m_l).floor() as usize;
        level.min(params.max_level_cap)
    }

    /// Greedy beam search within a single layer.
    ///
    /// When a `FlatAdjacencyList` is available (populated via `compact()`), neighbor
    /// lookups use the flat contiguous buffer for better cache locality. Otherwise,
    /// the standard per-node `Vec<Vec<VectorId>>` adjacency is used.
    ///
    /// Prefetch hints are issued for the next candidate's neighbor list and vector
    /// data so that the CPU memory subsystem can overlap fetches with distance
    /// computation.
    fn search_layer(
        &self,
        inner: &HnswInner,
        query: &[f32],
        entry_points: &[VectorId],
        ef: usize,
        layer: usize,
    ) -> Vec<(OrderedFloat<f32>, VectorId)> {
        // Reuse per-worker scratch buffers; clear and refill on each call.
        SCRATCH_SEARCH_VISITED.with(|v_cell| {
            SCRATCH_SEARCH_CANDIDATES.with(|c_cell| {
                SCRATCH_SEARCH_RESULTS.with(|r_cell| {
                    let mut visited = v_cell.borrow_mut();
                    let mut candidates = c_cell.borrow_mut();
                    let mut results = r_cell.borrow_mut();
                    visited.clear();
                    candidates.clear();
                    results.clear();

                    for &ep in entry_points {
                        if let Some(node) = inner.nodes.get(&ep) {
                            visited.insert(ep);
                            let dist = OrderedFloat(self.distance(query, inner.vectors.get(node.vector_slot)));
                            candidates.push(Reverse((dist, ep)));
                            results.push((dist, ep));
                        }
                    }

                    while let Some(Reverse((c_dist, c_id))) = candidates.pop() {
                        let furthest_result = results.peek().map(|r| r.0).unwrap_or(OrderedFloat(f32::MAX));
                        if c_dist > furthest_result {
                            break;
                        }

                        // Prefetch the next candidate's data if available, so the CPU can
                        // start fetching while we process the current candidate.
                        if let Some(&Reverse((_, next_id))) = candidates.peek() {
                            if let Some(next_node) = inner.nodes.get(&next_id) {
                                prefetch_vector(inner.vectors.get(next_node.vector_slot));
                                if layer < next_node.neighbors.len() {
                                    prefetch_neighbors(&next_node.neighbors[layer]);
                                }
                            }
                        }

                        // Retrieve neighbors by reference: prefer flat adjacency list if available.
                        // We borrow the neighbor slice directly to avoid allocating a Vec on every iteration.
                        let flat_neighbors: Option<&[VectorId]> = inner.flat_adj.as_ref()
                            .and_then(|flat| flat.get_neighbors(c_id, layer));
                        let node_ref = if flat_neighbors.is_none() {
                            inner.nodes.get(&c_id)
                        } else {
                            None
                        };
                        let neighbors: Option<&[VectorId]> = flat_neighbors.or_else(|| {
                            node_ref.and_then(|node| {
                                if layer < node.neighbors.len() {
                                    Some(node.neighbors[layer].as_slice())
                                } else {
                                    None
                                }
                            })
                        });

                        if let Some(neighbors) = neighbors {
                            // Collect unvisited neighbors into a batch for cache-friendly
                            // distance computation: the query vector stays in L1 while we
                            // iterate over all target vectors.
                            const BATCH_CAP: usize = 64;
                            let mut batch_ids: [VectorId; BATCH_CAP] = [0u64; BATCH_CAP];
                            let mut batch_vecs: [&[f32]; BATCH_CAP] = [&[]; BATCH_CAP];
                            let mut batch_len: usize = 0;

                            for &neighbor_id in neighbors.iter() {
                                if visited.insert(neighbor_id) {
                                    if let Some(neighbor_node) = inner.nodes.get(&neighbor_id) {
                                        batch_ids[batch_len] = neighbor_id;
                                        batch_vecs[batch_len] = inner.vectors.get(neighbor_node.vector_slot);
                                        batch_len += 1;

                                        // Flush when batch is full
                                        if batch_len == BATCH_CAP {
                                            let mut dists = [0.0f32; BATCH_CAP];
                                            self.distance_fn.compute_batch(
                                                query,
                                                &batch_vecs[..batch_len],
                                                &mut dists[..batch_len],
                                            );
                                            for j in 0..batch_len {
                                                let dist = OrderedFloat(dists[j]);
                                                let furthest = results
                                                    .peek()
                                                    .map(|r| r.0)
                                                    .unwrap_or(OrderedFloat(f32::MAX));
                                                if results.len() < ef || dist < furthest {
                                                    candidates.push(Reverse((dist, batch_ids[j])));
                                                    results.push((dist, batch_ids[j]));
                                                    if results.len() > ef {
                                                        results.pop();
                                                    }
                                                }
                                            }
                                            batch_len = 0;
                                        }
                                    }
                                }
                            }

                            // Flush remaining batch
                            if batch_len > 0 {
                                let mut dists = [0.0f32; BATCH_CAP];
                                self.distance_fn.compute_batch(
                                    query,
                                    &batch_vecs[..batch_len],
                                    &mut dists[..batch_len],
                                );
                                for j in 0..batch_len {
                                    let dist = OrderedFloat(dists[j]);
                                    let furthest = results
                                        .peek()
                                        .map(|r| r.0)
                                        .unwrap_or(OrderedFloat(f32::MAX));
                                    if results.len() < ef || dist < furthest {
                                        candidates.push(Reverse((dist, batch_ids[j])));
                                        results.push((dist, batch_ids[j]));
                                        if results.len() > ef {
                                            results.pop();
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Drain results into an owned Vec so the scratch heap can be reused on the next call.
                    results.drain().collect::<Vec<_>>()
                })
            })
        })
    }

    /// Algorithm 4: select neighbors with diversity heuristic.
    fn select_neighbors_heuristic(
        &self,
        inner: &HnswInner,
        _query: &[f32],
        candidates: &[(OrderedFloat<f32>, VectorId)],
        m: usize,
    ) -> Vec<VectorId> {
        if candidates.len() <= m {
            return candidates.iter().map(|&(_, id)| id).collect();
        }

        let mut sorted: Vec<(OrderedFloat<f32>, VectorId)> = candidates.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut result: Vec<(OrderedFloat<f32>, VectorId)> = Vec::with_capacity(m);

        for &(dist_to_query, candidate_id) in &sorted {
            if result.len() >= m {
                break;
            }

            let candidate_vec = match inner.nodes.get(&candidate_id) {
                Some(node) => inner.vectors.get(node.vector_slot),
                None => continue,
            };

            let is_diverse = result.iter().all(|&(_, existing_id)| {
                let existing_vec = match inner.nodes.get(&existing_id) {
                    Some(node) => inner.vectors.get(node.vector_slot),
                    None => return true,
                };
                let dist_between =
                    OrderedFloat(self.distance(candidate_vec, existing_vec));
                dist_to_query < dist_between
            });

            if is_diverse {
                result.push((dist_to_query, candidate_id));
            }
        }

        // If heuristic didn't fill up to m, add remaining candidates by distance
        if result.len() < m {
            let selected: HashSet<VectorId> = result.iter().map(|&(_, id)| id).collect();
            for &(dist, id) in &sorted {
                if result.len() >= m {
                    break;
                }
                if !selected.contains(&id) {
                    result.push((dist, id));
                }
            }
        }

        result.iter().map(|&(_, id)| id).collect()
    }

    fn prune_neighbors(&self, inner: &mut HnswInner, node_id: VectorId, layer: usize, max_conn: usize) {
        let (node_slot, neighbor_ids) = match inner.nodes.get(&node_id) {
            Some(n) if layer < n.neighbors.len() => {
                (n.vector_slot, n.neighbors[layer].clone())
            }
            _ => return,
        };
        let node_vec = inner.vectors.get(node_slot);

        let neighbor_list: Vec<(OrderedFloat<f32>, VectorId)> = neighbor_ids
            .iter()
            .filter_map(|&nid| {
                inner.nodes
                    .get(&nid)
                    .map(|n| (OrderedFloat(self.distance_fn.compute(node_vec, inner.vectors.get(n.vector_slot))), nid))
            })
            .collect();

        let pruned = self.select_neighbors_heuristic(inner, &node_vec, &neighbor_list, max_conn);
        if let Some(node) = inner.nodes.get_mut(&node_id) {
            if layer < node.neighbors.len() {
                node.neighbors[layer] = pruned;
            }
        }
    }

    fn insert_node(&self, inner: &mut HnswInner, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        if inner.nodes.contains_key(&id) {
            return Err(IndexError::AlreadyExists(id));
        }

        let new_level = Self::random_level(inner, &self.params);
        let slot = inner.vectors.push(vector);
        let node = HnswNode::new(slot, new_level);
        inner.nodes.insert(id, node);

        if inner.entry_point.is_none() {
            inner.entry_point = Some(id);
            inner.max_level = new_level;
            // Incrementally add the new (isolated) node to flat_adj if present.
            if let Some(ref mut flat) = inner.flat_adj {
                let empty_layers: Vec<&[VectorId]> = (0..=new_level).map(|_| &[][..]).collect();
                flat.insert_node(id, &empty_layers);
                flat.maybe_compact();
            }
            return Ok(());
        }

        let entry_point = inner.entry_point.unwrap();
        let mut current_ep = entry_point;

        // Track which nodes had their neighbor lists modified so we can
        // incrementally update the flat adjacency cache afterwards.
        let mut modified_neighbors: Vec<VectorId> = Vec::new();

        // Phase 1: greedy descent from top to new_level + 1
        let top = inner.max_level;
        if top > new_level {
            for layer in (new_level + 1..=top).rev() {
                let results = self.search_layer(inner, vector, &[current_ep], 1, layer);
                if let Some(&(_, closest)) = results
                    .iter()
                    .min_by_key(|&&(d, _)| d)
                {
                    current_ep = closest;
                }
            }
        }

        // Phase 2: search and connect from min(new_level, max_level) down to 0
        let start_layer = new_level.min(inner.max_level);
        for layer in (0..=start_layer).rev() {
            let results =
                self.search_layer(inner, vector, &[current_ep], self.params.ef_construction, layer);

            if let Some(&(_, closest)) = results.iter().min_by_key(|&&(d, _)| d) {
                current_ep = closest;
            }

            let m = self.max_connections(layer);
            let neighbors = self.select_neighbors_heuristic(inner, vector, &results, m);

            // Set neighbors for the new node at this layer
            if let Some(node) = inner.nodes.get_mut(&id) {
                if layer < node.neighbors.len() {
                    node.neighbors[layer] = neighbors.clone();
                }
            }

            // Add bidirectional edges and prune if needed
            let max_conn = self.max_connections(layer);
            for &neighbor_id in &neighbors {
                let needs_pruning = if let Some(neighbor_node) = inner.nodes.get_mut(&neighbor_id) {
                    if layer < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[layer].push(id);
                        neighbor_node.neighbors[layer].len() > max_conn
                    } else {
                        false
                    }
                } else {
                    false
                };

                if needs_pruning {
                    self.prune_neighbors(inner, neighbor_id, layer, max_conn);
                }

                modified_neighbors.push(neighbor_id);
            }
        }

        if new_level > inner.max_level {
            inner.max_level = new_level;
            inner.entry_point = Some(id);
        }

        // Incrementally update flat adjacency cache instead of full invalidation.
        if let Some(ref mut flat) = inner.flat_adj {
            // Add the newly inserted node with its final neighbor lists.
            if let Some(node) = inner.nodes.get(&id) {
                let layer_refs: Vec<&[VectorId]> =
                    node.neighbors.iter().map(|l| l.as_slice()).collect();
                flat.insert_node(id, &layer_refs);
            }

            // Update each neighbor whose list was modified.
            modified_neighbors.sort_unstable();
            modified_neighbors.dedup();
            for &nid in &modified_neighbors {
                if let Some(neighbor_node) = inner.nodes.get(&nid) {
                    let layer_refs: Vec<&[VectorId]> =
                        neighbor_node.neighbors.iter().map(|l| l.as_slice()).collect();
                    flat.insert_node(nid, &layer_refs);
                }
            }

            flat.maybe_compact();
        }

        Ok(())
    }

    fn search_knn(&self, inner: &HnswInner, query: &[f32], k: usize, ef_search: Option<usize>) -> Result<Vec<ScoredResult>, IndexError> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let entry_point = match inner.entry_point {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let mut current_ep = entry_point;

        // Phase 1: greedy descent from top layer to layer 1
        for layer in (1..=inner.max_level).rev() {
            let results = self.search_layer(inner, query, &[current_ep], 1, layer);
            if let Some(&(_, closest)) = results.iter().min_by_key(|&&(d, _)| d) {
                current_ep = closest;
            }
        }

        // Phase 2: search at layer 0 with ef_search
        let ef = ef_search.unwrap_or(self.params.ef_search).max(k);
        let results = self.search_layer(inner, query, &[current_ep], ef, 0);

        let mut scored: Vec<ScoredResult> = results
            .iter()
            .map(|&(dist, id)| ScoredResult::new(id, dist.into_inner()))
            .collect();
        scored.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        scored.truncate(k);

        Ok(scored)
    }

    fn delete_node(&self, inner: &mut HnswInner, id: VectorId) -> Result<(), IndexError> {
        let node = inner.nodes.remove(&id).ok_or(IndexError::NotFound(id))?;
        // Free the vector slot in the arena for reuse.
        inner.vectors.free(node.vector_slot);
        let node_level = node.max_level();

        // Track which nodes had their neighbor lists modified for incremental
        // flat adjacency updates.
        let mut modified_neighbors: Vec<VectorId> = Vec::new();

        // Reconnect orphaned neighbors at each layer
        for layer in 0..=node_level {
            let orphaned_neighbors: Vec<VectorId> = node.neighbors[layer].clone();

            // Remove the deleted node from each neighbor's list
            for &neighbor_id in &orphaned_neighbors {
                if let Some(neighbor_node) = inner.nodes.get_mut(&neighbor_id) {
                    if layer < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[layer].retain(|&nid| nid != id);
                        modified_neighbors.push(neighbor_id);
                    }
                }
            }

            // Attempt to reconnect orphaned neighbors with each other
            for &neighbor_id in &orphaned_neighbors {
                let neighbor_slot = match inner.nodes.get(&neighbor_id) {
                    Some(n) => n.vector_slot,
                    None => continue,
                };

                let current_neighbors: Vec<VectorId> = match inner.nodes.get(&neighbor_id) {
                    Some(n) if layer < n.neighbors.len() => n.neighbors[layer].clone(),
                    _ => continue,
                };

                let max_conn = self.max_connections(layer);
                if current_neighbors.len() >= max_conn {
                    continue;
                }

                let neighbor_vec = inner.vectors.get(neighbor_slot);
                let current_set: HashSet<VectorId> = current_neighbors.iter().copied().collect();
                let mut candidates: Vec<(OrderedFloat<f32>, VectorId)> = current_neighbors
                    .iter()
                    .filter_map(|&nid| {
                        inner.nodes
                            .get(&nid)
                            .map(|n| (OrderedFloat(self.distance(neighbor_vec, inner.vectors.get(n.vector_slot))), nid))
                    })
                    .collect();

                // Add other orphaned neighbors as potential new connections
                for &other_id in &orphaned_neighbors {
                    if other_id != neighbor_id && !current_set.contains(&other_id) {
                        if let Some(other_node) = inner.nodes.get(&other_id) {
                            if layer <= other_node.max_level() {
                                let dist = OrderedFloat(
                                    self.distance(neighbor_vec, inner.vectors.get(other_node.vector_slot)),
                                );
                                candidates.push((dist, other_id));
                            }
                        }
                    }
                }

                let selected =
                    self.select_neighbors_heuristic(inner, neighbor_vec, &candidates, max_conn);

                if let Some(neighbor_node) = inner.nodes.get_mut(&neighbor_id) {
                    if layer < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[layer] = selected.clone();
                        modified_neighbors.push(neighbor_id);
                    }
                }

                // Ensure bidirectional edges
                let max_c = self.max_connections(layer);
                for &sel_id in &selected {
                    let needs_pruning = if let Some(sel_node) = inner.nodes.get_mut(&sel_id) {
                        if layer < sel_node.neighbors.len()
                            && !sel_node.neighbors[layer].contains(&neighbor_id)
                        {
                            sel_node.neighbors[layer].push(neighbor_id);
                            modified_neighbors.push(sel_id);
                            sel_node.neighbors[layer].len() > max_c
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if needs_pruning {
                        self.prune_neighbors(inner, sel_id, layer, max_c);
                        modified_neighbors.push(sel_id);
                    }
                }
            }
        }

        // Handle entry point update
        if inner.entry_point == Some(id) {
            if inner.nodes.is_empty() {
                inner.entry_point = None;
                inner.max_level = 0;
            } else {
                // Find the node with the highest level as new entry point
                let (new_ep, new_max) = inner
                    .nodes
                    .iter()
                    .map(|(&nid, node)| (nid, node.max_level()))
                    .max_by_key(|&(_, level)| level)
                    .unwrap();
                inner.entry_point = Some(new_ep);
                inner.max_level = new_max;
            }
        } else if inner.nodes.is_empty() {
            inner.entry_point = None;
            inner.max_level = 0;
        } else if node_level >= inner.max_level {
            // Only recompute max_level if the deleted node could have been the highest
            let actual_max = inner
                .nodes
                .values()
                .map(|n| n.max_level())
                .max()
                .unwrap_or(0);
            inner.max_level = actual_max;
        }

        // Incrementally update flat adjacency cache instead of full invalidation.
        if let Some(ref mut flat) = inner.flat_adj {
            // Remove the deleted node.
            flat.remove_node(id);

            // Update each neighbor whose list was modified during reconnection.
            modified_neighbors.sort_unstable();
            modified_neighbors.dedup();
            for &nid in &modified_neighbors {
                if let Some(neighbor_node) = inner.nodes.get(&nid) {
                    let layer_refs: Vec<&[VectorId]> =
                        neighbor_node.neighbors.iter().map(|l| l.as_slice()).collect();
                    flat.insert_node(nid, &layer_refs);
                }
            }

            flat.maybe_compact();
        }

        Ok(())
    }

    // F1: repair connectivity for one already-present node against the live graph.
    //
    // The parallel bulk builders skip the very first item when the index starts
    // empty (it seeds the entry point), and the next few items search against a
    // still-empty graph, so the earliest nodes can end up with empty neighbor
    // lists and become unreachable. This pass re-derives a node's neighbors from
    // the fully built graph and installs bidirectional edges, mirroring the
    // reconnection logic in `delete_node` (above) and the connect phase of
    // `insert_node`. It runs under the same write lock as Phase 3 and is invoked
    // only for the handful of nodes that need it, so it does not affect recall or
    // build time materially. Modified neighbor ids are appended to `modified` so
    // the caller can refresh `flat_adj` for them.
    fn repair_node_connectivity(
        &self,
        inner: &mut HnswInner,
        id: VectorId,
        modified: &mut Vec<VectorId>,
    ) {
        // The node must exist and the graph must have at least one other node to
        // connect to. A query vector copy is taken so no borrow of the arena is
        // held across the mutations below.
        let (node_level, query_vec): (usize, Vec<f32>) = match inner.nodes.get(&id) {
            Some(node) => (node.max_level(), inner.vectors.get(node.vector_slot).to_vec()),
            None => return,
        };
        let entry_point = match inner.entry_point {
            Some(ep) => ep,
            None => return,
        };

        // Greedy descent from the top layer down to node_level + 1 (ef=1), exactly
        // as insert_node does, so the search enters the node's own layers near its
        // true neighborhood rather than at an arbitrary point.
        let mut current_ep = entry_point;
        let top = inner.max_level;
        if top > node_level {
            for layer in (node_level + 1..=top).rev() {
                let results = self.search_layer(inner, &query_vec, &[current_ep], 1, layer);
                if let Some(&(_, closest)) = results.iter().filter(|&&(_, c)| c != id).min_by_key(|&&(d, _)| d) {
                    current_ep = closest;
                }
            }
        }

        // Connect from min(node_level, max_level) down to 0.
        let start_layer = node_level.min(inner.max_level);
        for layer in (0..=start_layer).rev() {
            let results =
                self.search_layer(inner, &query_vec, &[current_ep], self.params.ef_construction, layer);
            // Exclude self from both the descent hint and the candidate set.
            let candidates: Vec<(OrderedFloat<f32>, VectorId)> =
                results.iter().copied().filter(|&(_, c)| c != id).collect();
            if let Some(&(_, closest)) = candidates.iter().min_by_key(|&&(d, _)| d) {
                current_ep = closest;
            }
            if candidates.is_empty() {
                continue;
            }

            let m = self.max_connections(layer);
            let neighbors = self.select_neighbors_heuristic(inner, &query_vec, &candidates, m);

            // Merge into the node's existing list at this layer (do not drop links
            // it may already hold), dedup, then set.
            if let Some(node) = inner.nodes.get_mut(&id) {
                if layer < node.neighbors.len() {
                    for &nid in &neighbors {
                        if nid != id && !node.neighbors[layer].contains(&nid) {
                            node.neighbors[layer].push(nid);
                        }
                    }
                }
            }

            // Add the reciprocal edge on each chosen neighbor and prune on overflow.
            let max_conn = self.max_connections(layer);
            for &neighbor_id in &neighbors {
                if neighbor_id == id {
                    continue;
                }
                let needs_pruning = if let Some(neighbor_node) = inner.nodes.get_mut(&neighbor_id) {
                    if layer < neighbor_node.neighbors.len()
                        && !neighbor_node.neighbors[layer].contains(&id)
                    {
                        neighbor_node.neighbors[layer].push(id);
                        neighbor_node.neighbors[layer].len() > max_conn
                    } else {
                        false
                    }
                } else {
                    false
                };
                if needs_pruning {
                    self.prune_neighbors(inner, neighbor_id, layer, max_conn);
                }
                modified.push(neighbor_id);
            }
        }
        modified.push(id);
    }

    /// Build the HNSW index in parallel using rayon with fine-grained locking.
    ///
    /// Uses a two-phase approach for true parallelism:
    /// 1. **Sequential setup**: Pre-compute levels for all vectors and insert empty
    ///    nodes into a concurrent node map (fast, no distance computations).
    /// 2. **Parallel connection**: Build graph connections in parallel using per-node
    ///    `Mutex` locks. Each thread performs search (read-only vector access)
    ///    and only locks individual nodes when updating neighbor lists.
    ///
    /// This achieves 2-4x speedup over the old approach which held a global write
    /// lock for every insertion, serializing all graph mutations.
    pub fn build_parallel(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Validate dimensions up front.
        for &(_id, vector) in vectors {
            if vector.len() != self.dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: self.dimension,
                    actual: vector.len(),
                });
            }
        }

        // Check for duplicates within the batch and against existing index.
        {
            let inner = self.inner.read();
            let mut seen = HashSet::with_capacity(vectors.len());
            for &(id, _) in vectors {
                if inner.nodes.contains_key(&id) || !seen.insert(id) {
                    return Err(IndexError::AlreadyExists(id));
                }
            }
        }

        // Phase 1: Pre-compute levels sequentially (requires RNG, very fast).
        let levels: Vec<usize> = {
            let mut inner = self.inner.write();
            vectors
                .iter()
                .map(|_| Self::random_level(&mut inner, &self.params))
                .collect()
        };

        // Build a concurrent node map: each node's neighbors are protected by
        // a per-node Mutex for fine-grained locking during parallel connection.
        // The vector data is immutable after creation, so no lock needed for reads.
        struct ConcurrentNode {
            vector: Vec<f32>,
            neighbors: Vec<Mutex<Vec<VectorId>>>,
        }

        let mut all_nodes: HashMap<VectorId, ConcurrentNode> = HashMap::with_capacity(
            vectors.len() + self.inner.read().nodes.len(),
        );

        // Include existing nodes from the index so search can find them.
        {
            let inner = self.inner.read();
            for (&id, node) in &inner.nodes {
                let neighbors = node
                    .neighbors
                    .iter()
                    .map(|layer| Mutex::new(layer.clone()))
                    .collect();
                all_nodes.insert(
                    id,
                    ConcurrentNode {
                        vector: inner.vectors.get(node.vector_slot).to_vec(),
                        neighbors,
                    },
                );
            }
        }

        // Add new nodes to the concurrent map.
        for (&(id, vector), &level) in vectors.iter().zip(levels.iter()) {
            let neighbors = (0..=level).map(|_| Mutex::new(Vec::new())).collect();
            all_nodes.insert(
                id,
                ConcurrentNode {
                    vector: vector.to_vec(),
                    neighbors,
                },
            );
        }

        // Determine entry point and max level.
        let (initial_ep, initial_max_level) = {
            let inner = self.inner.read();
            (inner.entry_point, inner.max_level)
        };

        let entry_point_atomic = AtomicU64::new(initial_ep.unwrap_or(vectors[0].0));
        let max_level_atomic = AtomicUsize::new(if initial_ep.is_none() {
            levels[0]
        } else {
            initial_max_level
        });

        // Helper: search a layer using the concurrent node map (lock-free reads
        // of vectors, snapshot reads of neighbor lists via brief per-node lock).
        let search_layer_concurrent = |query: &[f32],
                                        entry_points: &[VectorId],
                                        ef: usize,
                                        layer: usize|
         -> Vec<(OrderedFloat<f32>, VectorId)> {
            let mut visited: HashSet<VectorId> = HashSet::new();
            let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, VectorId)>> =
                BinaryHeap::new();
            let mut results: BinaryHeap<(OrderedFloat<f32>, VectorId)> = BinaryHeap::new();

            for &ep in entry_points {
                if let Some(node) = all_nodes.get(&ep) {
                    visited.insert(ep);
                    let dist = OrderedFloat(self.distance(query, &node.vector));
                    candidates.push(Reverse((dist, ep)));
                    results.push((dist, ep));
                }
            }

            while let Some(Reverse((c_dist, c_id))) = candidates.pop() {
                let furthest_result =
                    results.peek().map(|r| r.0).unwrap_or(OrderedFloat(f32::MAX));
                if c_dist > furthest_result {
                    break;
                }

                // Read neighbors with a brief lock (just clone the vec).
                let neighbor_ids: Option<Vec<VectorId>> =
                    all_nodes.get(&c_id).and_then(|node| {
                        if layer < node.neighbors.len() {
                            Some(node.neighbors[layer].lock().clone())
                        } else {
                            None
                        }
                    });

                if let Some(neighbors) = neighbor_ids {
                    for &neighbor_id in &neighbors {
                        if visited.insert(neighbor_id) {
                            if let Some(neighbor_node) = all_nodes.get(&neighbor_id) {
                                let dist = OrderedFloat(
                                    self.distance(query, &neighbor_node.vector),
                                );
                                let furthest = results
                                    .peek()
                                    .map(|r| r.0)
                                    .unwrap_or(OrderedFloat(f32::MAX));

                                if results.len() < ef || dist < furthest {
                                    candidates.push(Reverse((dist, neighbor_id)));
                                    results.push((dist, neighbor_id));
                                    if results.len() > ef {
                                        results.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            results.into_iter().collect()
        };

        // Helper: select neighbors heuristic using concurrent node map.
        let select_neighbors_concurrent =
            |_query: &[f32],
             candidates: &[(OrderedFloat<f32>, VectorId)],
             m: usize|
             -> Vec<VectorId> {
                if candidates.len() <= m {
                    return candidates.iter().map(|&(_, id)| id).collect();
                }

                let mut sorted: Vec<(OrderedFloat<f32>, VectorId)> = candidates.to_vec();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));

                let mut result: Vec<(OrderedFloat<f32>, VectorId)> = Vec::with_capacity(m);

                for &(dist_to_query, candidate_id) in &sorted {
                    if result.len() >= m {
                        break;
                    }

                    let candidate_vec = match all_nodes.get(&candidate_id) {
                        Some(node) => &node.vector,
                        None => continue,
                    };

                    let is_diverse = result.iter().all(|&(_, existing_id)| {
                        let existing_vec = match all_nodes.get(&existing_id) {
                            Some(node) => &node.vector,
                            None => return true,
                        };
                        let dist_between =
                            OrderedFloat(self.distance(candidate_vec, existing_vec));
                        dist_to_query < dist_between
                    });

                    if is_diverse {
                        result.push((dist_to_query, candidate_id));
                    }
                }

                if result.len() < m {
                    let selected: HashSet<VectorId> =
                        result.iter().map(|&(_, id)| id).collect();
                    for &(dist, id) in &sorted {
                        if result.len() >= m {
                            break;
                        }
                        if !selected.contains(&id) {
                            result.push((dist, id));
                        }
                    }
                }

                result.iter().map(|&(_, id)| id).collect()
            };

        // Phase 2: Connect nodes in parallel. Each vector finds its neighbors
        // via concurrent search, then updates neighbor lists with per-node
        // Mutex locks. Only individual nodes are locked during neighbor updates.
        let start_idx = if initial_ep.is_none() { 1 } else { 0 };

        vectors[start_idx..]
            .par_iter()
            .zip(levels[start_idx..].par_iter())
            .for_each(|(&(id, vector), &new_level)| {
                let cur_ep = entry_point_atomic.load(Ordering::Acquire);
                let cur_max = max_level_atomic.load(Ordering::Acquire);
                let mut current_ep = cur_ep;

                // Greedy descent from top to new_level + 1.
                if cur_max > new_level {
                    for layer in (new_level + 1..=cur_max).rev() {
                        let results =
                            search_layer_concurrent(vector, &[current_ep], 1, layer);
                        if let Some(&(_, closest)) =
                            results.iter().min_by_key(|&&(d, _)| d)
                        {
                            current_ep = closest;
                        }
                    }
                }

                // Search and connect from min(new_level, cur_max) down to 0.
                let start_layer = new_level.min(cur_max);
                for layer in (0..=start_layer).rev() {
                    let results = search_layer_concurrent(
                        vector,
                        &[current_ep],
                        self.params.ef_construction,
                        layer,
                    );

                    if let Some(&(_, closest)) =
                        results.iter().min_by_key(|&&(d, _)| d)
                    {
                        current_ep = closest;
                    }

                    let m = self.max_connections(layer);
                    let neighbors = select_neighbors_concurrent(vector, &results, m);

                    // Set neighbors for the new node (lock only this node).
                    if let Some(node) = all_nodes.get(&id) {
                        if layer < node.neighbors.len() {
                            *node.neighbors[layer].lock() = neighbors.clone();
                        }
                    }

                    // Add bidirectional edges with per-node locking and inline pruning.
                    let max_conn = self.max_connections(layer);
                    for &neighbor_id in &neighbors {
                        if let Some(neighbor_node) = all_nodes.get(&neighbor_id) {
                            if layer < neighbor_node.neighbors.len() {
                                let mut neighbor_list =
                                    neighbor_node.neighbors[layer].lock();
                                neighbor_list.push(id);

                                // Prune if over capacity while lock is held.
                                if neighbor_list.len() > max_conn {
                                    let neighbor_vec = &neighbor_node.vector;
                                    let scored: Vec<(OrderedFloat<f32>, VectorId)> =
                                        neighbor_list
                                            .iter()
                                            .filter_map(|&nid| {
                                                all_nodes.get(&nid).map(|n| {
                                                    (
                                                        OrderedFloat(self.distance(
                                                            neighbor_vec,
                                                            &n.vector,
                                                        )),
                                                        nid,
                                                    )
                                                })
                                            })
                                            .collect();
                                    let pruned = select_neighbors_concurrent(
                                        neighbor_vec,
                                        &scored,
                                        max_conn,
                                    );
                                    *neighbor_list = pruned;
                                }
                            }
                        }
                    }
                }

                // Update max_level and entry_point atomically (best-effort hint
                // for other threads' search; definitive values computed in phase 3).
                loop {
                    let old_max = max_level_atomic.load(Ordering::Relaxed);
                    if new_level <= old_max {
                        break;
                    }
                    if max_level_atomic
                        .compare_exchange_weak(
                            old_max,
                            new_level,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        entry_point_atomic.store(id, Ordering::Release);
                        break;
                    }
                }
            });

        // Phase 3: Transfer the concurrent node map back into HnswInner.
        let mut inner = self.inner.write();
        inner.flat_adj = None;
        // Clear existing arena and rebuild from the concurrent nodes.
        inner.vectors.clear();

        for (id, cnode) in all_nodes {
            let neighbors: Vec<Vec<VectorId>> = cnode
                .neighbors
                .into_iter()
                .map(|m| m.into_inner())
                .collect();
            let slot = inner.vectors.push(&cnode.vector);
            inner.nodes.insert(id, HnswNode {
                vector_slot: slot,
                neighbors,
            });
        }

        // Compute definitive entry_point and max_level by scanning all nodes.
        // This avoids subtle races between the atomic updates during parallel build.
        let (final_ep, final_max) = inner
            .nodes
            .iter()
            .map(|(&id, node)| (id, node.max_level()))
            .max_by_key(|&(_, level)| level)
            .unwrap_or((entry_point_atomic.load(Ordering::Acquire), 0));
        inner.entry_point = Some(final_ep);
        inner.max_level = final_max;

        // F1: same seed-skip orphan as the bulk_add paths. On the empty-index
        // case the first vector connected nothing and the next few searched an
        // empty graph; repair the seed, the entry point, and any node left with an
        // empty layer-0 list against the now-complete graph. flat_adj is None here
        // and no delta is emitted, so the repaired neighbor ids need no further
        // bookkeeping.
        if initial_ep.is_none() {
            let mut to_repair: Vec<VectorId> = Vec::new();
            let mut seen: HashSet<VectorId> = HashSet::new();
            if seen.insert(vectors[0].0) {
                to_repair.push(vectors[0].0);
            }
            if let Some(ep) = inner.entry_point {
                if seen.insert(ep) {
                    to_repair.push(ep);
                }
            }
            for &(id, _) in vectors {
                let empty_l0 = inner
                    .nodes
                    .get(&id)
                    .map(|n| n.neighbors.first().map(|l| l.is_empty()).unwrap_or(true))
                    .unwrap_or(false);
                if empty_l0 && seen.insert(id) {
                    to_repair.push(id);
                }
            }
            let mut scratch: Vec<VectorId> = Vec::new();
            for id in to_repair {
                self.repair_node_connectivity(&mut inner, id, &mut scratch);
            }
        }

        Ok(())
    }

    /// Sequentially insert multiple vectors into the index.
    ///
    /// Convenience method for bulk loading without parallelism.
    pub fn bulk_add(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        for (id, vector) in vectors {
            self.add(*id, vector)?;
        }
        Ok(())
    }

    /// Parallel bulk insert that also emits one delta entry per item, in input order.
    //
    // Result is deterministic per (initial graph state, input slice). Not serial-equivalent:
    // within one batch, items do not see each other as candidate neighbors. Outside a batch
    // (across batches or across single-item adds), the existing serial-equivalence contract
    // still holds.
    pub fn bulk_add_with_lsn(
        &self,
        items: &[(VectorId, Arc<Vec<f32>>, u64)],
    ) -> Result<(), IndexError> {
        if items.is_empty() {
            return Ok(());
        }

        // Validate dimensions up front.
        for (_id, vector, _lsn) in items {
            if vector.len() != self.dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: self.dimension,
                    actual: vector.len(),
                });
            }
        }

        // Reject duplicate ids within the batch and ids that already exist in the index.
        {
            let inner = self.inner.read();
            let mut seen = HashSet::with_capacity(items.len());
            for (id, _, _) in items {
                if inner.nodes.contains_key(id) || !seen.insert(*id) {
                    return Err(IndexError::AlreadyExists(*id));
                }
            }
        }

        // Pre-compute levels sequentially under the write lock. RNG access only.
        let levels: Vec<usize> = {
            let mut inner = self.inner.write();
            items
                .iter()
                .map(|_| Self::random_level(&mut inner, &self.params))
                .collect()
        };

        // Concurrent node map: vectors are wrapped in Arc so cloning during the
        // parallel phase costs an atomic increment rather than a full Vec copy.
        // After the rayon phase ends, dropping all_nodes drops the Arcs and the
        // memory is reclaimed in one shot instead of lingering for the lifetime
        // of the bulk_insert call.
        struct ConcurrentNode {
            vector: Arc<Vec<f32>>,
            neighbors: Vec<Mutex<Vec<VectorId>>>,
            // True if this node was modified during the batch (bidirectional edge added or pruned).
            modified: std::sync::atomic::AtomicBool,
        }

        let mut all_nodes: HashMap<VectorId, ConcurrentNode> = HashMap::with_capacity(
            items.len() + self.inner.read().nodes.len(),
        );

        // Snapshot existing nodes into the concurrent map. The vector data is
        // copied out of the arena exactly once into a heap allocation owned by
        // an Arc; the rayon phase reads through the Arc without further copies.
        {
            let inner = self.inner.read();
            for (&id, node) in &inner.nodes {
                let neighbors = node
                    .neighbors
                    .iter()
                    .map(|layer| Mutex::new(layer.clone()))
                    .collect();
                let vec_owned: Vec<f32> = inner.vectors.get(node.vector_slot).to_vec();
                all_nodes.insert(
                    id,
                    ConcurrentNode {
                        vector: Arc::new(vec_owned),
                        neighbors,
                        modified: std::sync::atomic::AtomicBool::new(false),
                    },
                );
            }
        }

        // Add the new items into the concurrent map. Input vectors arrive as
        // Arc<Vec<f32>> from the handler; clone the Arc (refcount bump only).
        for ((id, vector, _lsn), &level) in items.iter().zip(levels.iter()) {
            let neighbors = (0..=level).map(|_| Mutex::new(Vec::new())).collect();
            all_nodes.insert(
                *id,
                ConcurrentNode {
                    vector: Arc::clone(vector),
                    neighbors,
                    modified: std::sync::atomic::AtomicBool::new(false),
                },
            );
        }

        // Determine the starting entry point and max level once, before the parallel phase.
        let (initial_ep, initial_max_level) = {
            let inner = self.inner.read();
            (inner.entry_point, inner.max_level)
        };

        // If the index is empty, seed the entry point with the first new item.
        let seed_ep = initial_ep.unwrap_or(items[0].0);
        let seed_max = if initial_ep.is_none() { levels[0] } else { initial_max_level };
        let entry_point_atomic = AtomicU64::new(seed_ep);
        let max_level_atomic = AtomicUsize::new(seed_max);

        // Concurrent search inside the local node map: lock-free vector reads, brief per-layer lock for neighbor clone.
        let search_layer_concurrent = |query: &[f32],
                                        entry_points: &[VectorId],
                                        ef: usize,
                                        layer: usize|
         -> Vec<(OrderedFloat<f32>, VectorId)> {
            // Reuse per-worker scratch buffers; clear and refill on each call.
            SCRATCH_VISITED.with(|v_cell| {
                SCRATCH_CANDIDATES.with(|c_cell| {
                    SCRATCH_RESULTS.with(|r_cell| {
                        let mut visited = v_cell.borrow_mut();
                        let mut candidates = c_cell.borrow_mut();
                        let mut results = r_cell.borrow_mut();
                        visited.clear();
                        candidates.clear();
                        results.clear();

                        for &ep in entry_points {
                            if let Some(node) = all_nodes.get(&ep) {
                                visited.insert(ep);
                                let dist = OrderedFloat(self.distance(query, node.vector.as_slice()));
                                candidates.push(Reverse((dist, ep)));
                                results.push((dist, ep));
                            }
                        }

                        while let Some(Reverse((c_dist, c_id))) = candidates.pop() {
                            let furthest_result = results.peek().map(|r| r.0).unwrap_or(OrderedFloat(f32::MAX));
                            if c_dist > furthest_result {
                                break;
                            }

                            let neighbor_ids: Option<Vec<VectorId>> =
                                all_nodes.get(&c_id).and_then(|node| {
                                    if layer < node.neighbors.len() {
                                        Some(node.neighbors[layer].lock().clone())
                                    } else {
                                        None
                                    }
                                });

                            if let Some(neighbors) = neighbor_ids {
                                for &neighbor_id in &neighbors {
                                    if visited.insert(neighbor_id) {
                                        if let Some(neighbor_node) = all_nodes.get(&neighbor_id) {
                                            let dist = OrderedFloat(self.distance(query, neighbor_node.vector.as_slice()));
                                            let furthest = results.peek().map(|r| r.0).unwrap_or(OrderedFloat(f32::MAX));
                                            if results.len() < ef || dist < furthest {
                                                candidates.push(Reverse((dist, neighbor_id)));
                                                results.push((dist, neighbor_id));
                                                if results.len() > ef {
                                                    results.pop();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Drain results into an owned Vec so the scratch heap can be reused on the next call.
                        results.drain().collect()
                    })
                })
            })
        };

        // Diversity heuristic over the concurrent map.
        let select_neighbors_concurrent =
            |_query: &[f32], candidates: &[(OrderedFloat<f32>, VectorId)], m: usize| -> Vec<VectorId> {
                if candidates.len() <= m {
                    return candidates.iter().map(|&(_, id)| id).collect();
                }

                let mut sorted: Vec<(OrderedFloat<f32>, VectorId)> = candidates.to_vec();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));

                let mut result: Vec<(OrderedFloat<f32>, VectorId)> = Vec::with_capacity(m);
                for &(dist_to_query, candidate_id) in &sorted {
                    if result.len() >= m {
                        break;
                    }
                    let candidate_vec: &[f32] = match all_nodes.get(&candidate_id) {
                        Some(node) => node.vector.as_slice(),
                        None => continue,
                    };
                    let is_diverse = result.iter().all(|&(_, existing_id)| {
                        let existing_vec: &[f32] = match all_nodes.get(&existing_id) {
                            Some(node) => node.vector.as_slice(),
                            None => return true,
                        };
                        let dist_between = OrderedFloat(self.distance(candidate_vec, existing_vec));
                        dist_to_query < dist_between
                    });
                    if is_diverse {
                        result.push((dist_to_query, candidate_id));
                    }
                }

                if result.len() < m {
                    let selected: HashSet<VectorId> = result.iter().map(|&(_, id)| id).collect();
                    for &(dist, id) in &sorted {
                        if result.len() >= m {
                            break;
                        }
                        if !selected.contains(&id) {
                            result.push((dist, id));
                        }
                    }
                }

                result.iter().map(|&(_, id)| id).collect()
            };

        // Parallel insert. If the index was empty, the first item seeds the entry point and skips connection work.
        let start_idx = if initial_ep.is_none() { 1 } else { 0 };

        items[start_idx..]
            .par_iter()
            .zip(levels[start_idx..].par_iter())
            .for_each(|((id, vector, _lsn), &new_level)| {
                let cur_ep = entry_point_atomic.load(Ordering::Acquire);
                let cur_max = max_level_atomic.load(Ordering::Acquire);
                let mut current_ep = cur_ep;
                let query: &[f32] = vector.as_slice();

                if cur_max > new_level {
                    for layer in (new_level + 1..=cur_max).rev() {
                        let results = search_layer_concurrent(query, &[current_ep], 1, layer);
                        if let Some(&(_, closest)) = results.iter().min_by_key(|&&(d, _)| d) {
                            current_ep = closest;
                        }
                    }
                }

                let start_layer = new_level.min(cur_max);
                for layer in (0..=start_layer).rev() {
                    let results = search_layer_concurrent(
                        query,
                        &[current_ep],
                        self.params.ef_construction,
                        layer,
                    );

                    if let Some(&(_, closest)) = results.iter().min_by_key(|&&(d, _)| d) {
                        current_ep = closest;
                    }

                    let m = self.max_connections(layer);
                    let neighbors = select_neighbors_concurrent(query, &results, m);

                    if let Some(node) = all_nodes.get(id) {
                        if layer < node.neighbors.len() {
                            *node.neighbors[layer].lock() = neighbors.clone();
                        }
                    }

                    let max_conn = self.max_connections(layer);
                    for &neighbor_id in &neighbors {
                        if let Some(neighbor_node) = all_nodes.get(&neighbor_id) {
                            if layer < neighbor_node.neighbors.len() {
                                let mut neighbor_list = neighbor_node.neighbors[layer].lock();
                                neighbor_list.push(*id);
                                neighbor_node.modified.store(true, Ordering::Relaxed);

                                if neighbor_list.len() > max_conn {
                                    let neighbor_vec: &[f32] = neighbor_node.vector.as_slice();
                                    let scored: Vec<(OrderedFloat<f32>, VectorId)> = neighbor_list
                                        .iter()
                                        .filter_map(|&nid| {
                                            all_nodes.get(&nid).map(|n| {
                                                (OrderedFloat(self.distance(neighbor_vec, n.vector.as_slice())), nid)
                                            })
                                        })
                                        .collect();
                                    let pruned = select_neighbors_concurrent(neighbor_vec, &scored, max_conn);
                                    *neighbor_list = pruned;
                                }
                            }
                        }
                    }
                }

                loop {
                    let old_max = max_level_atomic.load(Ordering::Relaxed);
                    if new_level <= old_max {
                        break;
                    }
                    if max_level_atomic
                        .compare_exchange_weak(old_max, new_level, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                    {
                        entry_point_atomic.store(*id, Ordering::Release);
                        break;
                    }
                }
            });

        // Phase 3: transfer back into HnswInner. Preserve existing arena slots; append new vectors.
        // Track final post-batch state for delta emission and flat_adj maintenance.
        let mut new_node_slots: HashMap<VectorId, usize> = HashMap::with_capacity(items.len());
        let mut modified_existing: Vec<VectorId> = Vec::new();

        {
            let mut inner = self.inner.write();

            // Drain the concurrent map. To keep vector_slot assignment deterministic,
            // push new vectors into the arena in input order; existing nodes are processed afterwards
            // since their slot is already stable.
            let mut drained: HashMap<VectorId, ConcurrentNode> = all_nodes;

            // Pass 1: new nodes in input order. Read the vector through the Arc
            // (deref to &[f32]); the arena copies it into its flat buffer.
            for (id, _vec, _lsn) in items {
                if let Some(cnode) = drained.remove(id) {
                    // Reclaim Vec slack from parallel-phase push doublings.
                    let neighbors: Vec<Vec<VectorId>> = cnode
                        .neighbors
                        .into_iter()
                        .map(|m| {
                            let mut v = m.into_inner();
                            v.shrink_to_fit();
                            v
                        })
                        .collect();
                    let slot = inner.vectors.push(cnode.vector.as_slice());
                    new_node_slots.insert(*id, slot);
                    inner.nodes.insert(
                        *id,
                        HnswNode {
                            vector_slot: slot,
                            neighbors,
                        },
                    );
                }
            }

            // Pass 2: existing nodes. Only touch nodes flagged modified; iteration order is irrelevant
            // since slots are preserved and we only mutate the neighbor lists.
            for (id, cnode) in drained.into_iter() {
                if !cnode.modified.load(Ordering::Relaxed) {
                    continue;
                }
                // Reclaim Vec slack from parallel-phase push doublings.
                let neighbors: Vec<Vec<VectorId>> = cnode
                    .neighbors
                    .into_iter()
                    .map(|m| {
                        let mut v = m.into_inner();
                        v.shrink_to_fit();
                        v
                    })
                    .collect();
                if let Some(node) = inner.nodes.get_mut(&id) {
                    node.neighbors = neighbors;
                    modified_existing.push(id);
                }
            }

            // Recompute definitive entry_point and max_level from the final node set.
            // This mirrors build_parallel and avoids races from the atomic hints.
            let (final_ep, final_max) = inner
                .nodes
                .iter()
                .map(|(&id, node)| (id, node.max_level()))
                .max_by_key(|&(_, level)| level)
                .unwrap_or((entry_point_atomic.load(Ordering::Acquire), 0));
            inner.entry_point = Some(final_ep);
            inner.max_level = final_max;

            // The concurrent map is fully consumed by the for loop above; iterator drop releases it.

            // Return capacity ratchet from the arena to the allocator now that
            // the bulk write burst is done.
            inner.vectors.compact();

            // F1: repair the earliest nodes. When the index started empty the
            // seed item connected nothing and the next few items searched an empty
            // graph, so those nodes can be unreachable. Repair the seed, the entry
            // point, and any new node left with an empty layer-0 list, against the
            // now-complete graph. Only runs on the empty-index case; the few
            // repaired nodes are folded into modified_existing for flat_adj refresh
            // and into the delta below (every node here is a new node, so each gets
            // an AddNode delta with its repaired list).
            if initial_ep.is_none() {
                let mut to_repair: Vec<VectorId> = Vec::new();
                let mut seen: HashSet<VectorId> = HashSet::new();
                // Always repair the seed and the final entry point.
                if seen.insert(items[0].0) {
                    to_repair.push(items[0].0);
                }
                if let Some(ep) = inner.entry_point {
                    if seen.insert(ep) {
                        to_repair.push(ep);
                    }
                }
                // Plus any new node left with an empty layer-0 list.
                for (id, _, _) in items {
                    let empty_l0 = inner
                        .nodes
                        .get(id)
                        .map(|n| n.neighbors.first().map(|l| l.is_empty()).unwrap_or(true))
                        .unwrap_or(false);
                    if empty_l0 && seen.insert(*id) {
                        to_repair.push(*id);
                    }
                }
                for id in to_repair {
                    self.repair_node_connectivity(&mut inner, id, &mut modified_existing);
                }
            }

            // Incrementally update flat_adj for every touched node, mirroring insert_node.
            if inner.flat_adj.is_some() {
                // Two passes: re-insert new nodes, then re-insert modified existing nodes.
                let touched: Vec<VectorId> = items
                    .iter()
                    .map(|(id, _, _)| *id)
                    .chain(modified_existing.iter().copied())
                    .collect();
                let mut dedup: HashSet<VectorId> = HashSet::with_capacity(touched.len());
                let mut touched_unique: Vec<VectorId> = Vec::with_capacity(touched.len());
                for id in touched {
                    if dedup.insert(id) {
                        touched_unique.push(id);
                    }
                }
                // Snapshot per-node neighbor layers before borrowing flat_adj mutably.
                let snapshots: Vec<(VectorId, Vec<Vec<VectorId>>)> = touched_unique
                    .iter()
                    .filter_map(|id| inner.nodes.get(id).map(|n| (*id, n.neighbors.clone())))
                    .collect();
                if let Some(ref mut flat) = inner.flat_adj {
                    for (id, layers) in &snapshots {
                        let layer_refs: Vec<&[VectorId]> = layers.iter().map(|l| l.as_slice()).collect();
                        flat.insert_node(*id, &layer_refs);
                    }
                    flat.maybe_compact();
                }
            }
        }

        // Emit deltas in input order, after the inner write lock is released.
        let mut dw = self.delta_writer.lock();
        if let Some(ref mut writer) = *dw {
            let inner = self.inner.read();
            for (id, _vector, lsn) in items {
                if let Some(node) = inner.nodes.get(id) {
                    let neighbors_per_layer: Vec<Vec<VectorId>> = node.neighbors.clone();
                    let entry = HnswDeltaEntry {
                        lsn: *lsn,
                        op: HnswDeltaOp::AddNode {
                            id: *id,
                            level: node.max_level() as u32,
                            vector_slot: node.vector_slot as u64,
                            neighbors_per_layer,
                        },
                    };
                    if let Err(e) = writer.append(&entry) {
                        log::warn!("delta write failed for AddNode id={}: {}", id, e);
                    }
                }
            }

            // Emit SetEntryPoint once at the highest LSN if the entry point moved.
            if inner.entry_point != initial_ep {
                if let Some(ep_id) = inner.entry_point {
                    let max_lsn = items.iter().map(|(_, _, lsn)| *lsn).max().unwrap_or(0);
                    let entry = HnswDeltaEntry {
                        lsn: max_lsn,
                        op: HnswDeltaOp::SetEntryPoint {
                            id: ep_id,
                            level: inner.max_level as u32,
                        },
                    };
                    if let Err(e) = writer.append(&entry) {
                        log::warn!("delta write failed for SetEntryPoint: {}", e);
                    }
                }
            }

            // Flush and rebuild the BufWriter so the internal buffer capacity
            // that grew during the burst is returned to the allocator.
            if let Err(e) = writer.sync() {
                log::warn!("delta sync after bulk_add_with_lsn failed: {}", e);
            }
            if let Err(e) = writer.reset_buffer() {
                log::warn!("delta reset_buffer after bulk_add_with_lsn failed: {}", e);
            }
        }

        // Return freed per-batch scratch to the OS so peak RSS tracks live data.
        release_to_os();

        // Silence unused warning if the field is only consulted on the slow path.
        let _ = new_node_slots;
        Ok(())
    }

    /// Bulk insert path that borrows vector slices directly from caller-owned
    /// memory (typically an mmap'd file) and avoids snapshotting existing
    /// nodes. Phase 2 reads existing nodes through `self.inner.read()` and
    /// keeps in-flight new rows in a DashMap; Phase 3 takes the write lock
    /// once and applies the drained results. See ADR-001 (memory-peak
    /// reduction) Decision 3 for the full algorithmic contract.
    pub fn bulk_add_from_slice_iter<'mmap>(
        &self,
        items: &[(VectorId, &'mmap [f32], u64)],
        total_count_hint: usize,
    ) -> Result<(), IndexError> {
        if items.is_empty() {
            return Ok(());
        }

        // Decision 6.a + P01: pre-reserve every N-sized container in the HNSW
        // path from the caller-supplied hint. reserve takes &mut self, so the
        // write lock is required here. We expand the arena's flat vector
        // buffer, the per-id nodes HashMap, and the free-slot tracker so the
        // bulk path never trips a doubling reallocation.
        //
        // FREE_SLOTS_RESERVE_DIVISOR: empirical 10 percent anticipated
        // deletion rate during a large load. Underrun is harmless (slack
        // returned via VectorArena::compact at end of Phase 3); overrun
        // falls back to the normal doubling for the small remainder.
        const FREE_SLOTS_RESERVE_DIVISOR: usize = 10;

        if total_count_hint > 0 {
            let mut w = self.inner.write();
            let current_vectors = w.vectors.active_count();
            if total_count_hint > current_vectors {
                w.vectors.reserve(total_count_hint - current_vectors);
            }
            let current_nodes = w.nodes.len();
            if total_count_hint > current_nodes {
                w.nodes.reserve(total_count_hint - current_nodes);
            }
            w.vectors
                .reserve_free_slots(total_count_hint / FREE_SLOTS_RESERVE_DIVISOR);
        }

        // Validate dimensions and detect duplicate / already-present ids
        // before any write-lock work. Mirrors bulk_add_with_lsn taxonomy.
        for (_id, vector, _lsn) in items {
            if vector.len() != self.dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: self.dimension,
                    actual: vector.len(),
                });
            }
        }
        {
            let inner = self.inner.read();
            let mut seen = HashSet::with_capacity(items.len());
            for (id, _, _) in items {
                if inner.nodes.contains_key(id) || !seen.insert(*id) {
                    return Err(IndexError::AlreadyExists(*id));
                }
            }
        }

        // Decision 3.a: Phase 1. Brief write lock to draw deterministic
        // random levels via the RwLock-protected RNG.
        let levels: Vec<usize> = {
            let mut inner = self.inner.write();
            items
                .iter()
                .map(|_| Self::random_level(&mut inner, &self.params))
                .collect()
        };

        // Decision 3.b: in-flight new rows live in a DashMap keyed by id.
        // Each NewRowState borrows its vector slice from the caller-held mmap
        // for the duration of this call; no per-row Vec<f32> allocation.
        let new_rows: Arc<DashMap<VectorId, NewRowState<'mmap>>> =
            Arc::new(DashMap::with_capacity(items.len()));
        for ((id, vector_slice, _lsn), &level) in items.iter().zip(levels.iter()) {
            let neighbors: Vec<Mutex<Vec<VectorId>>> =
                (0..=level).map(|_| Mutex::new(Vec::new())).collect();
            new_rows.insert(
                *id,
                NewRowState {
                    vector: *vector_slice,
                    level,
                    neighbors,
                },
            );
        }

        // Decision 3.c: deferred bidirectional edges for existing nodes.
        // Phase 2 cannot mutate existing nodes (read-lock only), so neighbor
        // appends targeting existing nodes are queued here and drained in
        // Phase 3 under the write lock.
        let existing_edge_updates: Arc<
            DashMap<VectorId, Mutex<Vec<(usize, VectorId)>>>,
        > = Arc::new(DashMap::new());

        // Take the read lock and hold it across the entire par_iter. Existing
        // node vector data and neighbor lists are read directly through this
        // guard with zero copy.
        let inner_guard = self.inner.read();

        let (initial_ep, initial_max_level) =
            (inner_guard.entry_point, inner_guard.max_level);

        // If the index is empty, seed the entry point with the first new
        // item. The first item connects no neighbors; subsequent items see
        // it through the unified candidate lookup.
        let seed_ep = initial_ep.unwrap_or(items[0].0);
        let seed_max = if initial_ep.is_none() { levels[0] } else { initial_max_level };
        let entry_point_atomic = AtomicU64::new(seed_ep);
        let max_level_atomic = AtomicUsize::new(seed_max);

        let start_idx = if initial_ep.is_none() { 1 } else { 0 };

        // Decision 3.e: Phase 2 parallel insert.
        items[start_idx..]
            .par_iter()
            .zip(levels[start_idx..].par_iter())
            .for_each(|((id, vector_slice, _lsn), &new_level)| {
                let cur_max = max_level_atomic.load(Ordering::Acquire);
                let mut current_ep = entry_point_atomic.load(Ordering::Acquire);
                let query: &[f32] = *vector_slice;

                // Top-down greedy descent for layers above new_level, ef=1.
                if cur_max > new_level {
                    for layer in (new_level + 1..=cur_max).rev() {
                        let results = self.search_layer_unified(
                            &inner_guard,
                            &new_rows,
                            query,
                            &[current_ep],
                            1,
                            layer,
                        );
                        if let Some(&(_, closest)) =
                            results.iter().min_by_key(|&&(d, _)| d)
                        {
                            current_ep = closest;
                        }
                    }
                }

                // Layer-by-layer connect with ef=ef_construction.
                let start_layer = new_level.min(cur_max);
                for layer in (0..=start_layer).rev() {
                    let results = self.search_layer_unified(
                        &inner_guard,
                        &new_rows,
                        query,
                        &[current_ep],
                        self.params.ef_construction,
                        layer,
                    );
                    if let Some(&(_, closest)) =
                        results.iter().min_by_key(|&&(d, _)| d)
                    {
                        current_ep = closest;
                    }

                    let m = self.max_connections(layer);
                    let neighbors = self.select_neighbors_unified(
                        &inner_guard,
                        &new_rows,
                        query,
                        &results,
                        m,
                    );

                    // Outbound edges on the new row itself. Decision 3.h:
                    // snapshot length only; the DashMap guard is dropped
                    // before the neighbor mutex is taken below.
                    let self_neighbor_layer_count: Option<usize> = {
                        let guard = new_rows.get(id);
                        guard.as_ref().map(|state| state.neighbors.len())
                    };
                    if let Some(layer_count) = self_neighbor_layer_count {
                        if layer < layer_count {
                            if let Some(state) = new_rows.get(id) {
                                *state.neighbors[layer].lock() = neighbors.clone();
                            }
                        }
                    }

                    // Bidirectional edges. For each chosen neighbor: if it
                    // is itself an in-flight new row, mutate its per-layer
                    // mutex; otherwise defer the edge for Phase 3.
                    for &neighbor_id in &neighbors {
                        // Probe whether the neighbor is in-flight without
                        // holding the DashMap guard across the distance
                        // work that may follow.
                        let neighbor_layer_count: Option<usize> = {
                            let guard = new_rows.get(&neighbor_id);
                            guard.as_ref().map(|s| s.neighbors.len())
                        };
                        if let Some(layer_count) = neighbor_layer_count {
                            if layer >= layer_count {
                                continue;
                            }
                            // Re-fetch and lock. The DashMap guard is held
                            // only across the neighbor Mutex acquisition,
                            // not across any distance computation outside
                            // of prune_neighbors_unified (which itself
                            // does NOT acquire neighbor mutexes; see
                            // Decision 3.f.1).
                            if let Some(neighbor_state) = new_rows.get(&neighbor_id) {
                                let mut nlist = neighbor_state.neighbors[layer].lock();
                                nlist.push(*id);
                                let cap = self.max_connections(layer);
                                if nlist.len() > cap {
                                    let neighbor_vec: &[f32] = neighbor_state.vector;
                                    let pruned = self.prune_neighbors_unified(
                                        &inner_guard,
                                        &new_rows,
                                        neighbor_vec,
                                        &nlist,
                                        cap,
                                    );
                                    *nlist = pruned;
                                }
                            }
                        } else {
                            // Existing node: queue the edge for Phase 3.
                            existing_edge_updates
                                .entry(neighbor_id)
                                .or_insert_with(|| Mutex::new(Vec::new()))
                                .lock()
                                .push((layer, *id));
                        }
                    }
                }

                // Promote entry point if this row's level exceeds current
                // max. Standard CAS loop matching bulk_add_with_lsn.
                loop {
                    let old_max = max_level_atomic.load(Ordering::Relaxed);
                    if new_level <= old_max {
                        break;
                    }
                    if max_level_atomic
                        .compare_exchange_weak(
                            old_max,
                            new_level,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        entry_point_atomic.store(*id, Ordering::Release);
                        break;
                    }
                }
            });

        // Release the read guard before taking the write lock for Phase 3.
        drop(inner_guard);

        // Decision 3.i: Phase 3 drain under a single write lock.
        let mut new_node_slots: HashMap<VectorId, usize> = HashMap::with_capacity(items.len());
        let mut modified_existing: Vec<VectorId> = Vec::new();

        {
            let mut inner = self.inner.write();

            // Pass 1: new rows in input order so vector_slot assignment is
            // deterministic. extend the arena from the borrowed mmap slice
            // exactly once per row.
            for (id, _vec, _lsn) in items {
                if let Some((_k, state)) = new_rows.remove(id) {
                    debug_assert_eq!(
                        state.neighbors.len(),
                        state.level + 1,
                        "NewRowState level / neighbors-layer-count invariant broken for VectorId={}",
                        id
                    );
                    let neighbors: Vec<Vec<VectorId>> = state
                        .neighbors
                        .into_iter()
                        .map(|m| {
                            let mut v = m.into_inner();
                            v.shrink_to_fit();
                            v
                        })
                        .collect();
                    let slot = inner.vectors.push(state.vector);
                    new_node_slots.insert(*id, slot);
                    inner.nodes.insert(
                        *id,
                        HnswNode {
                            vector_slot: slot,
                            neighbors,
                        },
                    );
                }
            }

            // Pass 2: drain deferred existing-node edges. For each affected
            // node, append the queued (layer, new_id) edges; prune any
            // layer that overflows max_connections using the same diversity
            // heuristic the legacy path uses.
            let pending_updates: Vec<(VectorId, Vec<(usize, VectorId)>)> = {
                let mut collected: Vec<(VectorId, Vec<(usize, VectorId)>)> =
                    Vec::with_capacity(existing_edge_updates.len());
                for entry in existing_edge_updates.iter() {
                    let key = *entry.key();
                    let updates = entry.value().lock().clone();
                    collected.push((key, updates));
                }
                collected
            };

            for (existing_id, updates) in pending_updates {
                if !inner.nodes.contains_key(&existing_id) {
                    continue;
                }
                // Append edges.
                if let Some(node) = inner.nodes.get_mut(&existing_id) {
                    for (layer, new_id) in &updates {
                        if *layer < node.neighbors.len() {
                            node.neighbors[*layer].push(*new_id);
                        }
                    }
                }
                // Per-layer prune. Score candidates against the existing
                // node's own vector, then run the diversity heuristic.
                // Done in a separate scope so we can borrow inner.vectors
                // immutably while node.neighbors is consulted.
                let layer_count = inner
                    .nodes
                    .get(&existing_id)
                    .map(|n| n.neighbors.len())
                    .unwrap_or(0);
                for layer in 0..layer_count {
                    let cap = self.max_connections(layer);
                    let current_len = inner
                        .nodes
                        .get(&existing_id)
                        .map(|n| n.neighbors[layer].len())
                        .unwrap_or(0);
                    if current_len <= cap {
                        continue;
                    }
                    // Snapshot candidates and the anchor vector.
                    let (anchor_vec_owned, candidates): (Vec<f32>, Vec<VectorId>) = {
                        let node = inner.nodes.get(&existing_id).expect("node present");
                        let anchor: Vec<f32> = inner.vectors.get(node.vector_slot).to_vec();
                        let cands = node.neighbors[layer].clone();
                        (anchor, cands)
                    };
                    // Pass 1 drained every new row into inner.nodes, so the
                    // new_rows DashMap is empty by the time Pass 2 runs. The
                    // sole lookup we need here is inner.nodes; an unresolved
                    // candidate is a stale edge against a removed node and
                    // is dropped from the scored set.
                    let scored: Vec<(OrderedFloat<f32>, VectorId)> = candidates
                        .iter()
                        .filter_map(|&nid| {
                            inner.nodes.get(&nid).map(|node| {
                                let v = inner.vectors.get(node.vector_slot);
                                (OrderedFloat(self.distance(&anchor_vec_owned, v)), nid)
                            })
                        })
                        .collect();
                    let pruned = self.select_neighbors_unified(
                        &inner,
                        &new_rows,
                        &anchor_vec_owned,
                        &scored,
                        cap,
                    );
                    if let Some(node) = inner.nodes.get_mut(&existing_id) {
                        node.neighbors[layer] = pruned;
                    }
                }
                modified_existing.push(existing_id);
            }

            // Recompute definitive entry_point / max_level from the final
            // node set, matching bulk_add_with_lsn.
            let (final_ep, final_max) = inner
                .nodes
                .iter()
                .map(|(&id, node)| (id, node.max_level()))
                .max_by_key(|&(_, level)| level)
                .unwrap_or((entry_point_atomic.load(Ordering::Acquire), 0));
            inner.entry_point = Some(final_ep);
            inner.max_level = final_max;

            // Return arena capacity slack to the allocator.
            inner.vectors.compact();

            // F1: repair the earliest nodes against the now-complete graph. Same
            // rationale and gate as bulk_add_with_lsn: only on the empty-index
            // case, where the seed connected nothing and the next items searched
            // an empty graph. Repaired ids fold into modified_existing for flat_adj
            // and into the delta (every node here is new, so each gets an AddNode).
            if initial_ep.is_none() {
                let mut to_repair: Vec<VectorId> = Vec::new();
                let mut seen: HashSet<VectorId> = HashSet::new();
                // Always repair the seed and the final entry point.
                if seen.insert(items[0].0) {
                    to_repair.push(items[0].0);
                }
                if let Some(ep) = inner.entry_point {
                    if seen.insert(ep) {
                        to_repair.push(ep);
                    }
                }
                // Plus any new node left with an empty layer-0 list.
                for (id, _, _) in items {
                    let empty_l0 = inner
                        .nodes
                        .get(id)
                        .map(|n| n.neighbors.first().map(|l| l.is_empty()).unwrap_or(true))
                        .unwrap_or(false);
                    if empty_l0 && seen.insert(*id) {
                        to_repair.push(*id);
                    }
                }
                for id in to_repair {
                    self.repair_node_connectivity(&mut inner, id, &mut modified_existing);
                }
            }

            // Incrementally update flat_adj for every touched node.
            if inner.flat_adj.is_some() {
                let touched: Vec<VectorId> = items
                    .iter()
                    .map(|(id, _, _)| *id)
                    .chain(modified_existing.iter().copied())
                    .collect();
                let mut dedup: HashSet<VectorId> = HashSet::with_capacity(touched.len());
                let mut touched_unique: Vec<VectorId> = Vec::with_capacity(touched.len());
                for id in touched {
                    if dedup.insert(id) {
                        touched_unique.push(id);
                    }
                }
                let snapshots: Vec<(VectorId, Vec<Vec<VectorId>>)> = touched_unique
                    .iter()
                    .filter_map(|id| inner.nodes.get(id).map(|n| (*id, n.neighbors.clone())))
                    .collect();
                if let Some(ref mut flat) = inner.flat_adj {
                    for (id, layers) in &snapshots {
                        let layer_refs: Vec<&[VectorId]> =
                            layers.iter().map(|l| l.as_slice()).collect();
                        flat.insert_node(*id, &layer_refs);
                    }
                    flat.maybe_compact();
                }
            }
        }

        // Emit deltas in input order, mirroring bulk_add_with_lsn.
        let mut dw = self.delta_writer.lock();
        if let Some(ref mut writer) = *dw {
            let inner = self.inner.read();
            for (id, _vector, lsn) in items {
                if let Some(node) = inner.nodes.get(id) {
                    let neighbors_per_layer: Vec<Vec<VectorId>> = node.neighbors.clone();
                    let entry = HnswDeltaEntry {
                        lsn: *lsn,
                        op: HnswDeltaOp::AddNode {
                            id: *id,
                            level: node.max_level() as u32,
                            vector_slot: node.vector_slot as u64,
                            neighbors_per_layer,
                        },
                    };
                    if let Err(e) = writer.append(&entry) {
                        log::warn!("delta write failed for AddNode id={}: {}", id, e);
                    }
                }
            }

            if inner.entry_point != initial_ep {
                if let Some(ep_id) = inner.entry_point {
                    let max_lsn = items.iter().map(|(_, _, lsn)| *lsn).max().unwrap_or(0);
                    let entry = HnswDeltaEntry {
                        lsn: max_lsn,
                        op: HnswDeltaOp::SetEntryPoint {
                            id: ep_id,
                            level: inner.max_level as u32,
                        },
                    };
                    if let Err(e) = writer.append(&entry) {
                        log::warn!("delta write failed for SetEntryPoint: {}", e);
                    }
                }
            }

            if let Err(e) = writer.sync() {
                log::warn!("delta sync after bulk_add_from_slice_iter failed: {}", e);
            }
            if let Err(e) = writer.reset_buffer() {
                log::warn!(
                    "delta reset_buffer after bulk_add_from_slice_iter failed: {}",
                    e
                );
            }
        }

        // Return freed per-batch scratch to the OS so peak RSS tracks live data.
        release_to_os();

        let _ = new_node_slots;
        Ok(())
    }

    // ── Unified search/select/prune helpers (no-snapshot bulk insert) ──
    //
    // These three helpers back bulk_add_from_slice_iter. They consult both
    // the existing index (via the HnswInner read guard) and the in-flight
    // new rows (via the DashMap of NewRowState). Decision 3.h: DashMap
    // shard guards are released before any distance computation; cloned
    // neighbor lists and the &'mmap [f32] slice headers carry the data out.

    /// Greedy beam search in a single layer that consults both existing
    /// nodes and in-flight new rows. Mirrors the legacy
    /// `search_layer_concurrent` closure body in `bulk_add_with_lsn`.
    fn search_layer_unified<'inner, 'mmap>(
        &self,
        inner: &'inner HnswInner,
        new_rows: &DashMap<VectorId, NewRowState<'mmap>>,
        query: &[f32],
        entry_points: &[VectorId],
        ef: usize,
        layer: usize,
    ) -> Vec<(OrderedFloat<f32>, VectorId)>
    where
        'mmap: 'inner,
    {
        SCRATCH_VISITED.with(|v_cell| {
            SCRATCH_CANDIDATES.with(|c_cell| {
                SCRATCH_RESULTS.with(|r_cell| {
                    let mut visited = v_cell.borrow_mut();
                    let mut candidates = c_cell.borrow_mut();
                    let mut results = r_cell.borrow_mut();
                    visited.clear();
                    candidates.clear();
                    results.clear();

                    for &ep in entry_points {
                        if visited.contains(&ep) {
                            continue;
                        }
                        // Look up the entry-point vector. Decision 3.h: if
                        // it lives in new_rows, copy the slice header into
                        // a local and drop the DashMap guard before the
                        // distance call.
                        let ep_vec_opt: Option<&[f32]> = if let Some(node) = inner.nodes.get(&ep) {
                            Some(inner.vectors.get(node.vector_slot))
                        } else if let Some(state) = new_rows.get(&ep) {
                            let v: &[f32] = state.vector;
                            drop(state);
                            Some(v)
                        } else {
                            None
                        };
                        if let Some(ep_vec) = ep_vec_opt {
                            visited.insert(ep);
                            let dist = OrderedFloat(self.distance(query, ep_vec));
                            candidates.push(Reverse((dist, ep)));
                            results.push((dist, ep));
                        }
                    }

                    while let Some(Reverse((c_dist, c_id))) = candidates.pop() {
                        let furthest_result =
                            results.peek().map(|r| r.0).unwrap_or(OrderedFloat(f32::MAX));
                        if c_dist > furthest_result {
                            break;
                        }

                        // Fetch the neighbor list for c_id at `layer`.
                        // Drop the DashMap guard immediately after cloning.
                        let neighbor_ids: Option<Vec<VectorId>> =
                            if let Some(node) = inner.nodes.get(&c_id) {
                                if layer < node.neighbors.len() {
                                    Some(node.neighbors[layer].clone())
                                } else {
                                    None
                                }
                            } else if let Some(state) = new_rows.get(&c_id) {
                                if layer < state.neighbors.len() {
                                    let cloned = state.neighbors[layer].lock().clone();
                                    drop(state);
                                    Some(cloned)
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                        if let Some(neighbors) = neighbor_ids {
                            for &neighbor_id in &neighbors {
                                if !visited.insert(neighbor_id) {
                                    continue;
                                }
                                // Look up the neighbor's vector slice;
                                // again, drop the DashMap guard before the
                                // distance call.
                                let n_vec_opt: Option<&[f32]> =
                                    if let Some(node) = inner.nodes.get(&neighbor_id) {
                                        Some(inner.vectors.get(node.vector_slot))
                                    } else if let Some(state) = new_rows.get(&neighbor_id) {
                                        let v: &[f32] = state.vector;
                                        drop(state);
                                        Some(v)
                                    } else {
                                        None
                                    };
                                if let Some(n_vec) = n_vec_opt {
                                    let dist = OrderedFloat(self.distance(query, n_vec));
                                    let furthest = results
                                        .peek()
                                        .map(|r| r.0)
                                        .unwrap_or(OrderedFloat(f32::MAX));
                                    if results.len() < ef || dist < furthest {
                                        candidates.push(Reverse((dist, neighbor_id)));
                                        results.push((dist, neighbor_id));
                                        if results.len() > ef {
                                            results.pop();
                                        }
                                    }
                                }
                            }
                        }
                    }

                    results.drain().collect()
                })
            })
        })
    }

    /// Diversity heuristic over a candidate set. Mirrors the legacy
    /// `select_neighbors_concurrent` closure body. Unified across existing
    /// nodes and in-flight new rows.
    fn select_neighbors_unified<'inner, 'mmap>(
        &self,
        inner: &'inner HnswInner,
        new_rows: &DashMap<VectorId, NewRowState<'mmap>>,
        _query: &[f32],
        candidates: &[(OrderedFloat<f32>, VectorId)],
        m: usize,
    ) -> Vec<VectorId>
    where
        'mmap: 'inner,
    {
        if candidates.len() <= m {
            return candidates.iter().map(|&(_, id)| id).collect();
        }

        let mut sorted: Vec<(OrderedFloat<f32>, VectorId)> = candidates.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut result: Vec<(OrderedFloat<f32>, VectorId)> = Vec::with_capacity(m);
        for &(dist_to_query, candidate_id) in &sorted {
            if result.len() >= m {
                break;
            }
            // Capture the candidate vector by copying the slice header.
            // Decision 3.h: drop the DashMap guard before the inner loop
            // below computes pairwise distances.
            let candidate_vec: &[f32] = if let Some(node) = inner.nodes.get(&candidate_id) {
                inner.vectors.get(node.vector_slot)
            } else if let Some(state) = new_rows.get(&candidate_id) {
                let v: &[f32] = state.vector;
                drop(state);
                v
            } else {
                continue;
            };
            let is_diverse = result.iter().all(|&(_, existing_id)| {
                let existing_vec: &[f32] = if let Some(node) = inner.nodes.get(&existing_id) {
                    inner.vectors.get(node.vector_slot)
                } else if let Some(state) = new_rows.get(&existing_id) {
                    let v: &[f32] = state.vector;
                    drop(state);
                    v
                } else {
                    return true;
                };
                let dist_between = OrderedFloat(self.distance(candidate_vec, existing_vec));
                dist_to_query < dist_between
            });
            if is_diverse {
                result.push((dist_to_query, candidate_id));
            }
        }

        if result.len() < m {
            let selected: HashSet<VectorId> = result.iter().map(|&(_, id)| id).collect();
            for &(dist, id) in &sorted {
                if result.len() >= m {
                    break;
                }
                if !selected.contains(&id) {
                    result.push((dist, id));
                }
            }
        }

        result.iter().map(|&(_, id)| id).collect()
    }

    // Decision 3.f.1: prune_neighbors_unified reads ONLY candidate vectors,
    // NEVER candidate neighbor lists. This invariant eliminates the AB-BA
    // deadlock risk when pruning is invoked under one new row's per-layer
    // neighbor Mutex. Do NOT add neighbor-list reads here without first
    // establishing a lock-ordering protocol and a superseding ADR.
    fn prune_neighbors_unified<'inner, 'mmap>(
        &self,
        inner: &'inner HnswInner,
        new_rows: &DashMap<VectorId, NewRowState<'mmap>>,
        query_vec: &[f32],
        candidates: &[VectorId],
        max_keep: usize,
    ) -> Vec<VectorId>
    where
        'mmap: 'inner,
    {
        // Score each candidate against query_vec. Vector lookup ONLY; no
        // neighbor-list reads. DashMap guards are dropped before any
        // distance call.
        let mut scored: Vec<(OrderedFloat<f32>, VectorId)> = candidates
            .iter()
            .filter_map(|&id| {
                let v: &[f32] = if let Some(node) = inner.nodes.get(&id) {
                    inner.vectors.get(node.vector_slot)
                } else if let Some(state) = new_rows.get(&id) {
                    let v: &[f32] = state.vector;
                    drop(state);
                    v
                } else {
                    return None;
                };
                Some((OrderedFloat(self.distance(query_vec, v)), id))
            })
            .collect();

        scored.sort_by(|a, b| a.0.cmp(&b.0));

        let mut kept: Vec<(OrderedFloat<f32>, VectorId)> = Vec::with_capacity(max_keep);
        for &(c_dist, c_id) in &scored {
            if kept.len() >= max_keep {
                break;
            }
            let c_vec: &[f32] = if let Some(node) = inner.nodes.get(&c_id) {
                inner.vectors.get(node.vector_slot)
            } else if let Some(state) = new_rows.get(&c_id) {
                let v: &[f32] = state.vector;
                drop(state);
                v
            } else {
                continue;
            };
            let diverse = kept.iter().all(|&(_, k_id)| {
                let k_vec: &[f32] = if let Some(node) = inner.nodes.get(&k_id) {
                    inner.vectors.get(node.vector_slot)
                } else if let Some(state) = new_rows.get(&k_id) {
                    let v: &[f32] = state.vector;
                    drop(state);
                    v
                } else {
                    return true;
                };
                c_dist < OrderedFloat(self.distance(c_vec, k_vec))
            });
            if diverse {
                kept.push((c_dist, c_id));
            }
        }

        kept.into_iter().map(|(_, id)| id).collect()
    }

    // ── Optimization: Flat Adjacency Compaction ────────────────────────

    /// Compacts the HNSW graph's adjacency data into a flat, contiguous buffer
    /// for cache-friendly neighbor access during search.
    ///
    /// Call this after the graph is fully built (or after a batch of insertions)
    /// to optimize search performance. Once active, the flat adjacency cache is
    /// incrementally maintained on subsequent `add()` and `remove()` calls,
    /// with automatic defragmentation when wasted space exceeds 30%.
    /// Only `build_parallel()` invalidates it entirely (requiring a fresh `compact()`).
    ///
    /// This is an optional optimization; the index works correctly without it.
    pub fn compact(&self) {
        let mut inner = self.inner.write();
        // Pre-size the flat adjacency buffer and per-node index from the known
        // node count to avoid Vec/HashMap doubling reallocations during the
        // post-insert compact step at 1M-node scale.
        let node_count = inner.nodes.len();
        let max_neighbors_per_node = self.params.m0;
        let mut flat = FlatAdjacencyList::with_capacity(node_count, max_neighbors_per_node);
        for (&id, node) in &inner.nodes {
            let layer_refs: Vec<&[VectorId]> =
                node.neighbors.iter().map(|layer| layer.as_slice()).collect();
            flat.insert_node(id, &layer_refs);
        }
        inner.flat_adj = Some(flat);
    }

    /// Returns `true` if the flat adjacency optimization is currently active.
    pub fn is_compacted(&self) -> bool {
        self.inner.read().flat_adj.is_some()
    }

    // ── Backward-compatible alias ────────────────────────────────────────

    /// Alias for `bulk_add`; the arena is now the default storage backend,
    /// so there is no separate arena-backed construction path.
    #[deprecated(note = "Arena is now the default storage. Use `bulk_add` or `build_parallel` instead.")]
    #[allow(deprecated)]
    pub fn build_with_arena(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        self.bulk_add(vectors)
    }

    /// Retrieve a vector by ID from the internal VectorArena.
    /// Acquires a read lock on the HNSW inner state.
    pub fn get_vector(&self, id: VectorId) -> Result<Vec<f32>, IndexError> {
        let inner = self.inner.read();
        let node = inner.nodes.get(&id).ok_or(IndexError::NotFound(id))?;
        Ok(inner.vectors.get(node.vector_slot).to_vec())
    }

    /// Retrieve all vectors from the internal VectorArena.
    /// Returns owned (id, Vec<f32>) pairs.
    pub fn iter_vectors(&self) -> Vec<(VectorId, Vec<f32>)> {
        let inner = self.inner.read();
        inner.nodes.iter()
            .map(|(&id, node)| (id, inner.vectors.get(node.vector_slot).to_vec()))
            .collect()
    }

    // ── Topology snapshot / restore ────────────────────────────────────

    /// Extract a topology snapshot under read lock.
    /// Captures the complete graph structure without vector data.
    pub fn snapshot_topology(&self, snapshot_lsn: u64) -> HnswTopologySnapshot {
        let inner = self.inner.read();

        let nodes: Vec<TopologyNode> = inner
            .nodes
            .iter()
            .map(|(&id, node)| TopologyNode {
                id,
                level: node.max_level(),
                vector_slot: node.vector_slot,
                neighbors: node.neighbors.clone(),
            })
            .collect();

        HnswTopologySnapshot {
            snapshot_lsn,
            timestamp: HnswTopologySnapshot::now_millis(),
            dimension: self.dimension as u32,
            metric: Self::metric_to_u8(self.metric),
            m: self.params.m as u32,
            m0: self.params.m0 as u32,
            ef_construction: self.params.ef_construction as u32,
            ef_search: self.params.ef_search as u32,
            entry_point: inner.entry_point,
            max_level: inner.max_level as u32,
            nodes,
        }
    }

    /// Reconstruct an HnswIndex from a topology snapshot and an externally-provided VectorArena.
    pub fn restore_from_topology(
        snapshot: HnswTopologySnapshot,
        arena: VectorArena,
    ) -> Result<Self, IndexError> {
        let metric = Self::metric_from_u8(snapshot.metric)?;
        let dimension = snapshot.dimension as usize;

        let params = HnswParams {
            m: snapshot.m as usize,
            m0: snapshot.m0 as usize,
            ef_construction: snapshot.ef_construction as usize,
            ef_search: snapshot.ef_search as usize,
            m_l: 1.0 / (snapshot.m as f64).ln(),
            max_level_cap: 16,
        };

        let mut nodes = HashMap::with_capacity(snapshot.nodes.len());
        for topo_node in snapshot.nodes {
            let node = HnswNode {
                vector_slot: topo_node.vector_slot,
                neighbors: topo_node.neighbors,
            };
            nodes.insert(topo_node.id, node);
        }

        Ok(Self {
            inner: RwLock::new(HnswInner {
                nodes,
                vectors: arena,
                entry_point: snapshot.entry_point,
                max_level: snapshot.max_level as usize,
                rng: StdRng::from_entropy(),
                flat_adj: None,
            }),
            params,
            metric,
            distance_fn: DistanceMetric::from_metric_type(metric),
            dimension,
            delta_writer: Mutex::new(None),
        })
    }

    // ── Delta writer management ───────────────────────────────────────

    /// Attach a delta writer for incremental persistence.
    pub fn set_delta_writer(&self, writer: HnswDeltaWriter) {
        *self.delta_writer.lock() = Some(writer);
    }

    /// Detach the delta writer (e.g., before taking a base snapshot).
    pub fn take_delta_writer(&self) -> Option<HnswDeltaWriter> {
        self.delta_writer.lock().take()
    }

    // ── LSN-aware mutation methods ──────────────────────────────────────

    /// Insert a vector and emit a delta entry if a writer is attached.
    pub fn add_with_lsn(&self, id: VectorId, vector: &[f32], lsn: u64) -> Result<(), IndexError> {
        // Capture the entry point before the mutation to detect changes.
        let ep_before = self.inner.read().entry_point;

        // Perform the actual insertion via the existing add() path.
        {
            let mut inner = self.inner.write();
            self.insert_node(&mut inner, id, vector)?;
        }

        // Emit delta entries after successful mutation.
        let mut dw = self.delta_writer.lock();
        if let Some(ref mut writer) = *dw {
            let inner = self.inner.read();

            // Emit AddNode with the new node's full topology.
            if let Some(node) = inner.nodes.get(&id) {
                let neighbors_per_layer: Vec<Vec<VectorId>> = node.neighbors.clone();
                let entry = HnswDeltaEntry {
                    lsn,
                    op: HnswDeltaOp::AddNode {
                        id,
                        level: node.max_level() as u32,
                        vector_slot: node.vector_slot as u64,
                        neighbors_per_layer,
                    },
                };
                if let Err(e) = writer.append(&entry) {
                    log::warn!("delta write failed for AddNode id={}: {}", id, e);
                }
            }

            // Emit SetEntryPoint if the entry point changed.
            if inner.entry_point != ep_before {
                if let Some(ep_id) = inner.entry_point {
                    let entry = HnswDeltaEntry {
                        lsn,
                        op: HnswDeltaOp::SetEntryPoint {
                            id: ep_id,
                            level: inner.max_level as u32,
                        },
                    };
                    if let Err(e) = writer.append(&entry) {
                        log::warn!("delta write failed for SetEntryPoint: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Remove a vector and emit a delta entry if a writer is attached.
    pub fn remove_with_lsn(&self, id: VectorId, lsn: u64) -> Result<(), IndexError> {
        // Capture the entry point before the mutation to detect changes.
        let ep_before = self.inner.read().entry_point;

        // Perform the actual removal via the existing delete path.
        {
            let mut inner = self.inner.write();
            self.delete_node(&mut inner, id)?;
        }

        // Emit delta entries after successful mutation.
        let mut dw = self.delta_writer.lock();
        if let Some(ref mut writer) = *dw {
            let entry = HnswDeltaEntry {
                lsn,
                op: HnswDeltaOp::RemoveNode { id },
            };
            if let Err(e) = writer.append(&entry) {
                log::warn!("delta write failed for RemoveNode id={}: {}", id, e);
            }

            // Emit SetEntryPoint if the entry point changed after removal.
            let inner = self.inner.read();
            if inner.entry_point != ep_before {
                if let Some(ep_id) = inner.entry_point {
                    let ep_entry = HnswDeltaEntry {
                        lsn,
                        op: HnswDeltaOp::SetEntryPoint {
                            id: ep_id,
                            level: inner.max_level as u32,
                        },
                    };
                    if let Err(e) = writer.append(&ep_entry) {
                        log::warn!("delta write failed for SetEntryPoint: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    // ── Read-only graph accessors (for QuantizedHnswIndex) ─────────────

    /// Returns the current entry point of the graph.
    pub fn entry_point(&self) -> Option<VectorId> {
        let inner = self.inner.read();
        inner.entry_point
    }

    /// Returns the maximum level in the graph.
    pub fn max_level(&self) -> usize {
        let inner = self.inner.read();
        inner.max_level
    }

    /// Returns the level of a specific node, or None if not found.
    pub fn node_level(&self, id: VectorId) -> Option<usize> {
        let inner = self.inner.read();
        inner.nodes.get(&id).map(|n| n.max_level())
    }

    /// Returns the neighbors of a node at a specific level.
    pub fn neighbors(&self, id: VectorId, level: usize) -> Option<Vec<VectorId>> {
        let inner = self.inner.read();
        inner.nodes.get(&id).and_then(|n| {
            n.neighbors.get(level).cloned()
        })
    }

    /// Returns the VectorArena slot for a node, or None if not found.
    pub fn node_vector_slot(&self, id: VectorId) -> Option<usize> {
        let inner = self.inner.read();
        inner.nodes.get(&id).map(|n| n.vector_slot)
    }

    /// Clears the VectorArena data (frees f32 vectors from RAM).
    /// Graph topology (nodes, neighbors) is preserved.
    /// After calling this, get_vector() will fail.
    /// Only call this after quantization when u8 codes are ready.
    pub fn clear_arena(&self) {
        let mut inner = self.inner.write();
        inner.vectors.clear();
    }

    /// Returns the number of nodes in the graph.
    pub fn vector_count(&self) -> usize {
        let inner = self.inner.read();
        inner.nodes.len()
    }

    /// Returns a reference to the HNSW configuration parameters.
    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    /// Populate the index's VectorArena from a base-topology snapshot and a
    /// borrowed (id, vector) lookup map.
    ///
    /// Pre-condition: `self` must be empty (no nodes, empty arena), as
    /// produced by `HnswIndex::with_defaults`. The arena is grown by
    /// pushing vectors in the slot order recorded in `snapshot.nodes`,
    /// so the final layout matches the topology's `vector_slot`
    /// references one-for-one.
    ///
    /// Missing entries in `vectors` (vectors that the snapshot mentions
    /// but the storage layer did not return) are filled with zeros to
    /// keep slot indices aligned; this matches the behaviour of the
    /// existing inline boot path in `vf-server`.
    ///
    /// The lookup is taken by borrowed slices to avoid forcing the caller
    /// to clone every vector into an owned `Vec<f32>`. The boot path can
    /// build the map directly from the segment-loaded `Vec<(VectorId,
    /// Vec<f32>, Option<Metadata>)>` slice without an intermediate clone.
    ///
    /// This is the helper the boot path uses before
    /// `<HnswIndex as PersistableIndex>::try_restore_from_dir` so that the
    /// trait method only has to wire the topology onto an already-loaded
    /// arena.
    pub fn populate_arena_from_snapshot(
        &mut self,
        snapshot: &HnswTopologySnapshot,
        vectors: &HashMap<VectorId, &[f32]>,
    ) {
        // Preserve the original vector_slot indices from the snapshot.
        // After a partial-delete sequence, the surviving nodes keep their
        // original slot numbers (with gaps where deletes happened); naively
        // re-appending in slot order would compact the layout and break
        // every vector_slot reference in `inner.nodes`. Instead, size the
        // arena to (max_slot + 1) and write each vector at its original
        // offset, padding gaps with zeros.
        let mut nodes_by_slot: Vec<(usize, VectorId)> = snapshot
            .nodes
            .iter()
            .map(|n| (n.vector_slot, n.id))
            .collect();
        nodes_by_slot.sort_by_key(|(slot, _)| *slot);

        let mut inner = self.inner.write();
        inner.vectors.clear();
        if nodes_by_slot.is_empty() {
            return;
        }

        let max_slot = nodes_by_slot.last().map(|(s, _)| *s).unwrap_or(0);
        let total_slots = max_slot + 1;
        let zero_vec = vec![0.0f32; self.dimension];

        // Pre-grow the buffer in one shot with zero fill, then overwrite
        // each live slot at its original index. This preserves the
        // snapshot's vector_slot references one-for-one even when the
        // snapshot has gaps from deleted vectors.
        inner.vectors.resize_to_slots(total_slots, &zero_vec);
        let mut live: std::collections::HashSet<usize> =
            std::collections::HashSet::with_capacity(nodes_by_slot.len());
        for (slot, id) in &nodes_by_slot {
            let data = vectors.get(id).copied().unwrap_or(zero_vec.as_slice());
            inner.vectors.write_slot(*slot, data);
            live.insert(*slot);
        }
        // Mark non-live slots as free so active_count is accurate and
        // future pushes reuse the gaps rather than appending past the
        // existing range.
        for slot in 0..total_slots {
            if !live.contains(&slot) {
                inner.vectors.free(slot);
            }
        }
    }
}

impl VectorIndex for HnswIndex {
    fn add(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        let mut inner = self.inner.write();
        self.insert_node(&mut inner, id, vector)
    }

    fn remove(&self, id: VectorId) -> Result<(), IndexError> {
        let mut inner = self.inner.write();
        self.delete_node(&mut inner, id)
    }

    fn metric_type(&self) -> DistanceMetricType {
        self.metric
    }

    fn search(&self, query: &[f32], k: usize, ef_search: Option<usize>) -> Result<Vec<ScoredResult>, IndexError> {
        let inner = self.inner.read();
        self.search_knn(&inner, query, k, ef_search)
    }

    fn search_with_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let inner = self.inner.read();

        let entry_point = match inner.entry_point {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let candidate_set: HashSet<VectorId> = candidates.iter().copied().collect();

        // Use a larger ef to get more candidates for post-filtering
        let ef = ef_search.unwrap_or(self.params.ef_search).max(k * 10);

        let mut current_ep = entry_point;

        // Phase 1: greedy descent from top layer to layer 1
        for layer in (1..=inner.max_level).rev() {
            let results = self.search_layer(&inner, query, &[current_ep], 1, layer);
            if let Some(&(_, closest)) = results.iter().min_by_key(|&&(d, _)| d) {
                current_ep = closest;
            }
        }

        // Phase 2: search at layer 0 with enlarged ef
        let results = self.search_layer(&inner, query, &[current_ep], ef, 0);

        // Post-filter to only include IDs in the candidate set
        let mut filtered: Vec<ScoredResult> = results
            .iter()
            .filter(|&&(_, id)| candidate_set.contains(&id))
            .map(|&(dist, id)| ScoredResult::new(id, dist.into_inner()))
            .collect();
        filtered.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        filtered.truncate(k);

        Ok(filtered)
    }

    fn len(&self) -> usize {
        self.inner.read().nodes.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn contains(&self, id: VectorId) -> bool {
        self.inner.read().nodes.contains_key(&id)
    }

    fn get_vector(&self, id: VectorId) -> Result<Vec<f32>, IndexError> {
        self.get_vector(id)
    }

    fn iter_vectors(&self) -> Result<Vec<(VectorId, Vec<f32>)>, IndexError> {
        Ok(self.iter_vectors())
    }
}

impl PersistableIndex for HnswIndex {
    fn add_with_lsn(&self, id: VectorId, vector: &[f32], lsn: u64) -> Result<(), IndexError> {
        self.add_with_lsn(id, vector, lsn)
    }

    // Route trait dispatch to the parallel concrete implementation.
    fn bulk_add_with_lsn(
        &self,
        items: &[(VectorId, Arc<Vec<f32>>, u64)],
    ) -> Result<(), IndexError> {
        self.bulk_add_with_lsn(items)
    }

    fn bulk_add_from_slice_iter<'mmap>(
        &self,
        items: &[(VectorId, &'mmap [f32], u64)],
        total_count_hint: usize,
    ) -> Result<(), IndexError> {
        self.bulk_add_from_slice_iter(items, total_count_hint)
    }

    fn remove_with_lsn(&self, id: VectorId, lsn: u64) -> Result<(), IndexError> {
        self.remove_with_lsn(id, lsn)
    }

    fn snapshot_topology(&self, snapshot_lsn: u64) -> HnswTopologySnapshot {
        self.snapshot_topology(snapshot_lsn)
    }

    fn compact(&self) {
        self.compact()
    }

    fn is_compacted(&self) -> bool {
        self.is_compacted()
    }

    fn build_parallel(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        // Route the trait entry point through the Arc/bulk_add_with_lsn builder
        // so no caller reaches the deprecated double-copy inherent build_parallel
        // via the trait. Each vector is wrapped in an Arc once; LSN=0 is
        // unobservable here (no delta writer attached at build time).
        let items: Vec<(VectorId, Arc<Vec<f32>>, u64)> = vectors
            .iter()
            .map(|(id, v)| (*id, Arc::new(v.to_vec()), 0u64))
            .collect();
        self.bulk_add_with_lsn(&items)
    }

    fn cold_build_parallel(
        &mut self,
        vectors: &[(VectorId, &[f32])],
        _config: crate::traits::ParallelBuildConfig,
    ) -> Result<(), IndexError> {
        // Route through the Arc/bulk_add_with_lsn builder, same as the trait
        // build_parallel above, so the deprecated double-copy path is never
        // reached through this trait method.
        let items: Vec<(VectorId, Arc<Vec<f32>>, u64)> = vectors
            .iter()
            .map(|(id, v)| (*id, Arc::new(v.to_vec()), 0u64))
            .collect();
        self.bulk_add_with_lsn(&items)
    }

    fn set_delta_writer(&self, writer: HnswDeltaWriter) {
        self.set_delta_writer(writer)
    }

    fn take_delta_writer(&self) -> Option<HnswDeltaWriter> {
        self.take_delta_writer()
    }

    fn iter_vectors_owned(&self) -> Vec<(VectorId, Vec<f32>)> {
        self.iter_vectors()
    }

    fn as_vector_index(&self) -> &dyn VectorIndex {
        self
    }

    // ── Fast-restart contract (P01) ────────────────────────────────────
    //
    // Writes and reads `hnsw.base` (the topology snapshot) in `dir`. The
    // delta log file `hnsw.delta` is appended live by the delta writer,
    // not serialised here. The `shutdown_clean` marker is owned by the
    // shutdown path in vf-storage, not by this trait method.
    //
    // The vectors themselves are not serialised by `serialize_state_to_dir`
    // and not loaded by `try_restore_from_dir`. Vectors live in the
    // collection's segment files and the caller is responsible for
    // populating the index's VectorArena before calling restore. See the
    // architecture doc `persistable-index-trait.md`.

    fn serialize_state_to_dir(&self, dir: &Path) -> Result<(), IndexError> {
        // Topology snapshot does not carry a "live" LSN at this level; the
        // delta writer owns the live LSN tail. P01 uses 0 here because the
        // delta log is the authoritative tail. When the project adds a
        // proper snapshot-LSN bookkeeping path the value plumbed in here
        // can be the real one without changing the on-disk format.
        let snapshot = self.snapshot_topology(0);

        let base_path = dir.join("hnsw.base");
        let tmp_path = dir.join("hnsw.base.tmp");

        // Write to a temp file, fsync, rename. Atomic publish so a crash
        // mid-write leaves either the prior file or nothing at the target.
        {
            let mut tmp_file = std::fs::File::create(&tmp_path).map_err(|e| {
                IndexError::Internal(format!(
                    "serialize_state_to_dir: failed to create {}: {}",
                    tmp_path.display(),
                    e
                ))
            })?;

            serialize_topology(&snapshot, &mut tmp_file).map_err(|e| {
                // Best-effort cleanup of the temp file on a write failure.
                let _ = std::fs::remove_file(&tmp_path);
                IndexError::Internal(format!(
                    "serialize_state_to_dir: failed to serialise topology into {}: {}",
                    tmp_path.display(),
                    e
                ))
            })?;

            tmp_file.sync_all().map_err(|e| {
                let _ = std::fs::remove_file(&tmp_path);
                IndexError::Internal(format!(
                    "serialize_state_to_dir: fsync failed on {}: {}",
                    tmp_path.display(),
                    e
                ))
            })?;
        }

        std::fs::rename(&tmp_path, &base_path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_path);
            IndexError::Internal(format!(
                "serialize_state_to_dir: rename {} to {} failed: {}",
                tmp_path.display(),
                base_path.display(),
                e
            ))
        })?;

        // fsync the parent directory so the rename is durable.
        if let Ok(dir_handle) = std::fs::File::open(dir) {
            let _ = dir_handle.sync_all();
        }

        Ok(())
    }

    fn recovery_files(&self) -> &'static [&'static str] {
        &["hnsw.base", "hnsw.delta", "shutdown_clean"]
    }

    /// Pre-condition: `self` must be in the empty state produced by
    /// `HnswIndex::with_defaults` (no nodes, no entry point), and the
    /// caller must have populated `self.inner.vectors` (the VectorArena)
    /// with the collection's vectors prior to this call. The topology
    /// snapshot references vectors by arena slot, not by value, so the
    /// arena must line up slot-for-slot with the snapshot. See
    /// `architecture/persistable-index-trait.md` for the full contract.
    ///
    /// Delta replay carve-out (P01): this method loads only the base
    /// topology from `hnsw.base`. If `shutdown_clean` is absent and the
    /// base is present, the strategy is reported as `IncrementalReplay`
    /// and the caller is responsible for replaying `hnsw.delta` on top.
    /// Lifting delta replay into the trait may happen in a later phase.
    fn try_restore_from_dir(&mut self, dir: &Path) -> Result<RestoreOutcome, IndexError> {
        // Enforce the empty-state precondition. We do NOT silently
        // overwrite a populated index, which would corrupt on-disk state.
        {
            let inner = self.inner.read();
            if !inner.nodes.is_empty() || inner.entry_point.is_some() {
                return Err(IndexError::Internal(
                    "try_restore_from_dir called on non-empty HnswIndex".into(),
                ));
            }
        }

        let base_path = dir.join("hnsw.base");
        if !base_path.exists() {
            return Ok(RestoreOutcome::StateMissing);
        }

        // Soft-load the topology. CRC, magic, or version failures are
        // recoverable; we report them as StateCorrupt for the caller to
        // fall back to a full rebuild.
        let snapshot = match deserialize_topology_mmap(&base_path) {
            Ok(s) => s,
            Err(IndexError::Internal(reason)) => {
                return Ok(RestoreOutcome::StateCorrupt { reason });
            }
            Err(other) => {
                return Ok(RestoreOutcome::StateCorrupt {
                    reason: format!("{}", other),
                });
            }
        };

        // Dimension guard: a snapshot from a different-dimensioned index
        // must not be silently loaded into this one.
        if snapshot.dimension as usize != self.dimension {
            return Ok(RestoreOutcome::StateCorrupt {
                reason: format!(
                    "dimension mismatch: snapshot={}, configured={}",
                    snapshot.dimension, self.dimension
                ),
            });
        }

        // Decide the recovery strategy based on the shutdown marker.
        // CleanShutdown means a successful base plus a shutdown_clean
        // marker (caller already validated the base exists). The base
        // existence is implied here because we just loaded it.
        let shutdown_marker = dir.join("shutdown_clean");
        let strategy = if shutdown_marker.exists() {
            IndexRecoveryStrategy::CleanShutdown
        } else {
            IndexRecoveryStrategy::IncrementalReplay {
                hnsw_base_lsn: snapshot.snapshot_lsn,
                graph_base_lsn: 0,
            }
        };

        // Apply the snapshot in place. Lock the inner state for writing,
        // walk the topology, and populate the nodes map plus the entry-
        // point and max-level fields. The VectorArena is left alone; it
        // was populated by the caller before this method was invoked.
        {
            let mut inner = self.inner.write();
            inner.nodes.reserve(snapshot.nodes.len());
            for topo_node in snapshot.nodes {
                let node = HnswNode {
                    vector_slot: topo_node.vector_slot,
                    neighbors: topo_node.neighbors,
                };
                inner.nodes.insert(topo_node.id, node);
            }
            inner.entry_point = snapshot.entry_point;
            inner.max_level = snapshot.max_level as usize;
            // The flat adjacency cache is invalidated by any mutation; a
            // freshly-restored index has no cache.
            inner.flat_adj = None;
        }

        Ok(RestoreOutcome::Restored { strategy })
    }

    fn validate_state_on_disk(dir: &Path, dimension: usize) -> Result<bool, IndexError>
    where
        Self: Sized,
    {
        let base_path = dir.join("hnsw.base");
        validate_envelope_at_path(&base_path, dimension)
    }

    fn clear_state_from_dir(dir: &Path) -> Result<(), IndexError>
    where
        Self: Sized,
    {
        let remove_if_present = |name: &str| -> Result<(), IndexError> {
            let path = dir.join(name);
            match std::fs::remove_file(&path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(IndexError::Internal(format!(
                    "clear_state_from_dir: failed to remove {}: {}",
                    path.display(),
                    e
                ))),
            }
        };

        // Order matches recovery_files(): base, delta, shutdown marker.
        remove_if_present("hnsw.base")?;
        remove_if_present("hnsw.delta")?;
        remove_if_present("shutdown_clean")?;
        Ok(())
    }
}
