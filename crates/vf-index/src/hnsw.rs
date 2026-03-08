// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{Read as IoRead, Write as IoWrite};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use ordered_float::OrderedFloat;
use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use vf_core::distance::DistanceMetric;
use vf_core::types::{DistanceMetricType, ScoredResult, VectorId};

use crate::arena::VectorArena;
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

pub struct HnswIndex {
    inner: RwLock<HnswInner>,
    params: HnswParams,
    metric: DistanceMetricType,
    distance_fn: DistanceMetric,
    dimension: usize,
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

    // ── Backward-compatible alias ────────────────────────────────────────

    /// Alias for `bulk_add` — the arena is now the default storage backend,
    /// so there is no separate arena-backed construction path.
    #[deprecated(note = "Arena is now the default storage. Use `bulk_add` or `build_parallel` instead.")]
    #[allow(deprecated)]
    pub fn build_with_arena(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        self.bulk_add(vectors)
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
}
