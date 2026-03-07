// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{Read as IoRead, Write as IoWrite};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use vf_core::distance::{get_distance_fn, DistanceFunction};
use vf_core::types::{DistanceMetricType, ScoredResult, VectorId};

use crate::arena::{ArenaNodeId, NodeArena};
use crate::flat_adj::FlatAdjacencyList;
use crate::hnsw_types::HnswNode;
use crate::prefetch::{prefetch_neighbors, prefetch_vector};
use crate::traits::{IndexError, VectorIndex};

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
    entry_point: Option<VectorId>,
    max_level: usize,
    rng: StdRng,
    /// Optional flat adjacency list for cache-friendly neighbor access during search.
    /// Populated by calling `HnswIndex::compact()` after the graph is built.
    /// Automatically invalidated (set to `None`) on any graph mutation (add/remove).
    flat_adj: Option<FlatAdjacencyList>,
}

pub struct HnswIndex {
    inner: RwLock<HnswInner>,
    params: HnswParams,
    metric: DistanceMetricType,
    distance_fn: Box<dyn DistanceFunction>,
    dimension: usize,
}

const _: () = { fn _assert_send_sync<T: Send + Sync>() {} fn _check() { _assert_send_sync::<HnswIndex>(); } };

impl HnswIndex {
    pub fn new(dimension: usize, metric: DistanceMetricType, params: HnswParams) -> Self {
        Self {
            inner: RwLock::new(HnswInner {
                nodes: HashMap::new(),
                entry_point: None,
                max_level: 0,
                rng: StdRng::from_entropy(),
                flat_adj: None,
            }),
            params,
            metric,
            distance_fn: get_distance_fn(metric),
            dimension,
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
            for &val in &node.vector {
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

            let node = HnswNode { vector, neighbors };
            nodes.insert(id, node);
        }

        Ok(Self {
            inner: RwLock::new(HnswInner {
                nodes,
                entry_point,
                max_level,
                rng: StdRng::seed_from_u64(42),
                flat_adj: None,
            }),
            params,
            metric,
            distance_fn: get_distance_fn(metric),
            dimension,
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
        let mut visited: HashSet<VectorId> = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, VectorId)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<f32>, VectorId)> = BinaryHeap::new();

        for &ep in entry_points {
            if let Some(node) = inner.nodes.get(&ep) {
                visited.insert(ep);
                let dist = OrderedFloat(self.distance(query, &node.vector));
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
                    prefetch_vector(&next_node.vector);
                    if layer < next_node.neighbors.len() {
                        prefetch_neighbors(&next_node.neighbors[layer]);
                    }
                }
            }

            // Retrieve neighbors: prefer flat adjacency list if available.
            let neighbor_ids: Option<Vec<VectorId>> = if let Some(ref flat) = inner.flat_adj {
                flat.get_neighbors(c_id, layer).map(|s| s.to_vec())
            } else {
                inner.nodes.get(&c_id).and_then(|node| {
                    if layer < node.neighbors.len() {
                        Some(node.neighbors[layer].clone())
                    } else {
                        None
                    }
                })
            };

            if let Some(neighbors) = neighbor_ids {
                for (i, &neighbor_id) in neighbors.iter().enumerate() {
                    // Prefetch the vector of the neighbor we will process a couple
                    // of iterations ahead, overlapping the fetch with current work.
                    if i + 2 < neighbors.len() {
                        let ahead_id = neighbors[i + 2];
                        if let Some(ahead_node) = inner.nodes.get(&ahead_id) {
                            prefetch_vector(&ahead_node.vector);
                        }
                    }

                    if visited.insert(neighbor_id) {
                        if let Some(neighbor_node) = inner.nodes.get(&neighbor_id) {
                            let dist =
                                OrderedFloat(self.distance(query, &neighbor_node.vector));
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
                Some(node) => &node.vector,
                None => continue,
            };

            let is_diverse = result.iter().all(|&(_, existing_id)| {
                let existing_vec = match inner.nodes.get(&existing_id) {
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
        let (node_vec, neighbor_ids) = match inner.nodes.get(&node_id) {
            Some(n) if layer < n.neighbors.len() => {
                (n.vector.clone(), n.neighbors[layer].clone())
            }
            _ => return,
        };

        let neighbor_list: Vec<(OrderedFloat<f32>, VectorId)> = neighbor_ids
            .iter()
            .filter_map(|&nid| {
                inner.nodes
                    .get(&nid)
                    .map(|n| (OrderedFloat(self.distance_fn.compute(&node_vec, &n.vector)), nid))
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

        // Invalidate the flat adjacency cache — the graph structure is changing.
        inner.flat_adj = None;

        let new_level = Self::random_level(inner, &self.params);
        let node = HnswNode::new(vector.to_vec(), new_level);
        inner.nodes.insert(id, node);

        if inner.entry_point.is_none() {
            inner.entry_point = Some(id);
            inner.max_level = new_level;
            return Ok(());
        }

        let entry_point = inner.entry_point.unwrap();
        let mut current_ep = entry_point;

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
            }
        }

        if new_level > inner.max_level {
            inner.max_level = new_level;
            inner.entry_point = Some(id);
        }

        Ok(())
    }

    fn search_knn(&self, inner: &HnswInner, query: &[f32], k: usize) -> Result<Vec<ScoredResult>, IndexError> {
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
        let ef = self.params.ef_search.max(k);
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
        // Invalidate the flat adjacency cache — the graph structure is changing.
        inner.flat_adj = None;

        let node = inner.nodes.remove(&id).ok_or(IndexError::NotFound(id))?;
        let node_level = node.max_level();

        // Reconnect orphaned neighbors at each layer
        for layer in 0..=node_level {
            let orphaned_neighbors: Vec<VectorId> = node.neighbors[layer].clone();

            // Remove the deleted node from each neighbor's list
            for &neighbor_id in &orphaned_neighbors {
                if let Some(neighbor_node) = inner.nodes.get_mut(&neighbor_id) {
                    if layer < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[layer].retain(|&nid| nid != id);
                    }
                }
            }

            // Attempt to reconnect orphaned neighbors with each other
            for &neighbor_id in &orphaned_neighbors {
                let neighbor_vec = match inner.nodes.get(&neighbor_id) {
                    Some(n) => n.vector.clone(),
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

                let current_set: HashSet<VectorId> = current_neighbors.iter().copied().collect();
                let mut candidates: Vec<(OrderedFloat<f32>, VectorId)> = current_neighbors
                    .iter()
                    .filter_map(|&nid| {
                        inner.nodes
                            .get(&nid)
                            .map(|n| (OrderedFloat(self.distance(&neighbor_vec, &n.vector)), nid))
                    })
                    .collect();

                // Add other orphaned neighbors as potential new connections
                for &other_id in &orphaned_neighbors {
                    if other_id != neighbor_id && !current_set.contains(&other_id) {
                        if let Some(other_node) = inner.nodes.get(&other_id) {
                            if layer <= other_node.max_level() {
                                let dist = OrderedFloat(
                                    self.distance(&neighbor_vec, &other_node.vector),
                                );
                                candidates.push((dist, other_id));
                            }
                        }
                    }
                }

                let selected =
                    self.select_neighbors_heuristic(inner, &neighbor_vec, &candidates, max_conn);

                if let Some(neighbor_node) = inner.nodes.get_mut(&neighbor_id) {
                    if layer < neighbor_node.neighbors.len() {
                        neighbor_node.neighbors[layer] = selected.clone();
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
                            sel_node.neighbors[layer].len() > max_c
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if needs_pruning {
                        self.prune_neighbors(inner, sel_id, layer, max_c);
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

        Ok(())
    }

    /// Build the HNSW index in parallel using rayon.
    ///
    /// Uses parallel iteration over input vectors, with each thread calling
    /// `self.add()` which acquires the internal write lock. The RwLock serializes
    /// graph mutations while allowing rayon to parallelize distance computations
    /// and thread scheduling overhead.
    ///
    /// **Caveats:**
    /// - The write lock serializes graph mutations, so true parallelism is
    ///   limited to distance computations and rayon scheduling; insertions
    ///   themselves are sequential.
    /// - Insertion order is non-deterministic across runs, which affects the
    ///   resulting graph structure (and therefore recall/performance).
    /// - On error, vectors already inserted remain in the index — partial
    ///   insertion is possible.
    pub fn build_parallel(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        vectors
            .par_iter()
            .try_for_each(|(id, vector)| self.add(*id, vector))
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

    // ── Optimization: Flat Adjacency Compaction ────────────────────────

    /// Compacts the HNSW graph's adjacency data into a flat, contiguous buffer
    /// for cache-friendly neighbor access during search.
    ///
    /// Call this after the graph is fully built (or after a batch of insertions)
    /// to optimize search performance. The compacted adjacency is automatically
    /// invalidated on any subsequent `add()` or `remove()` call, so you should
    /// call `compact()` again after further mutations if you want the benefit.
    ///
    /// This is an optional optimization — the index works correctly without it.
    pub fn compact(&self) {
        let mut inner = self.inner.write();
        let flat = FlatAdjacencyList::from_hnsw_nodes(&inner.nodes);
        inner.flat_adj = Some(flat);
    }

    /// Returns `true` if the flat adjacency optimization is currently active.
    pub fn is_compacted(&self) -> bool {
        self.inner.read().flat_adj.is_some()
    }

    // ── Optimization: Arena-backed Construction ────────────────────────

    /// Builds the HNSW index using an arena allocator for node storage during
    /// construction, then transfers the result into the standard `HashMap`-based
    /// storage.
    ///
    /// # Design note
    ///
    /// The default HNSW implementation uses `HashMap<VectorId, HnswNode>` which
    /// stores nodes in heap-allocated slots managed by the hash map. For most
    /// workloads this is efficient because:
    /// - `HashMap` provides O(1) lookup by VectorId (essential for graph traversal).
    /// - Modern allocators (jemalloc, mimalloc) already batch small allocations.
    /// - The flat adjacency optimization (`compact()`) addresses the main cache
    ///   locality concern (neighbor list access patterns).
    ///
    /// This arena-backed path is provided for workloads where the construction
    /// phase allocates millions of nodes and heap fragmentation becomes a concern.
    /// It bulk-allocates nodes in contiguous chunks via `NodeArena`, then transfers
    /// them into the final `HashMap` for search compatibility.
    ///
    /// After calling this method, the arena is consumed and all data lives in the
    /// standard `HashMap` storage. Optionally call `compact()` afterwards for
    /// further search optimization.
    pub fn build_with_arena(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        // Phase 1: Allocate all nodes in the arena for cache-friendly bulk allocation.
        let chunk_size = NodeArena::DEFAULT_CHUNK_SIZE;
        let mut arena = NodeArena::new(chunk_size);
        let mut id_to_arena: HashMap<VectorId, ArenaNodeId> =
            HashMap::with_capacity(vectors.len());

        for &(id, vector) in vectors {
            if vector.len() != self.dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: self.dimension,
                    actual: vector.len(),
                });
            }
            let arena_id = arena.alloc(vector.to_vec(), 0, self.params.m0);
            id_to_arena.insert(id, arena_id);
        }

        // Phase 2: Insert into the HNSW graph using the standard path.
        // The arena ensured contiguous allocation during the bulk alloc phase;
        // now we transfer into the HashMap-based graph for full HNSW connectivity.
        for &(id, vector) in vectors {
            self.add(id, vector)?;
        }

        // Arena is dropped here — its job was to provide cache-friendly allocation
        // during the bulk construction phase. The data now lives in the HashMap.
        drop(arena);
        drop(id_to_arena);

        Ok(())
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

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredResult>, IndexError> {
        let inner = self.inner.read();
        self.search_knn(&inner, query, k)
    }

    fn search_with_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
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
        let ef = self.params.ef_search.max(k * 10);

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
}
