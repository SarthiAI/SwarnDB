// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! SQ8 Quantized HNSW index.
//!
//! Wraps [`HnswIndex`] with scalar-quantized u8 codes in RAM
//! ([`QuantizedArena`]) for fast HNSW traversal via asymmetric distance
//! (f32 query x u8 code), and original f32 vectors on disk
//! ([`MmapVectorStore`]) for exact rescore.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;

use vf_core::distance::DistanceMetric;
use vf_core::types::{
    DistanceMetricType, ScalarQuantizationConfig, ScoredResult, SearchQuantizationParams, VectorId,
};
use vf_quantization::scalar::ScalarQuantizer;
use vf_quantization::sq_simd::{
    sq_asymmetric_cosine_simd, sq_asymmetric_dot_simd, sq_asymmetric_l2_simd,
};

use crate::hnsw::{HnswIndex, HnswParams};
use crate::hnsw_delta::HnswDeltaWriter;
use crate::hnsw_persistence::HnswTopologySnapshot;
use crate::mmap_store::MmapVectorStore;
use crate::quantized_arena::QuantizedArena;
use crate::traits::{IndexError, PersistableIndex, VectorIndex};

// ── Main struct ─────────────────────────────────────────────────────────

/// HNSW index with SQ8 quantized codes for memory-efficient search.
///
/// After training, the graph topology lives in `hnsw`, quantized u8 codes
/// live in `codes` (RAM), and original f32 vectors live in `mmap_store`
/// (disk). Search traverses the HNSW graph using asymmetric SIMD distance
/// (f32 query x u8 code), then optionally rescores the top candidates with
/// exact f32 vectors from disk.
pub struct QuantizedHnswIndex {
    /// Inner HNSW graph (topology + arena before training).
    hnsw: HnswIndex,
    /// Trained scalar quantizer (None until `train_quantizer` is called).
    quantizer: RwLock<Option<ScalarQuantizer>>,
    /// Contiguous u8 code storage (one slot per vector).
    codes: RwLock<QuantizedArena>,
    /// Maps VectorId -> slot index in `codes`.
    code_slots: RwLock<HashMap<VectorId, usize>>,
    /// Memory-mapped f32 vector store on disk (for exact rescore).
    mmap_store: RwLock<Option<MmapVectorStore>>,
    /// SQ8 configuration (quantile, always_ram).
    config: ScalarQuantizationConfig,
    /// Distance metric type.
    metric: DistanceMetricType,
    /// Exact f32 distance function for rescoring.
    distance_fn: DistanceMetric,
    /// Vector dimensionality.
    dimension: usize,
    /// Whether quantizer has been trained and codes are populated.
    trained: AtomicBool,
    /// Cached precomputed scales (ranges / 255.0) for SIMD distance.
    cached_scales: RwLock<Vec<f32>>,
    /// Cached per-dimension minimum values for SIMD distance.
    cached_min_vals: RwLock<Vec<f32>>,
    /// Collection data directory for mmap file creation during post_optimize.
    data_dir: RwLock<Option<PathBuf>>,
}

// Compile-time Send+Sync assertion.
const _: () = {
    fn _assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        _assert_send_sync::<QuantizedHnswIndex>();
    }
};

// ── Constructor ─────────────────────────────────────────────────────────

impl QuantizedHnswIndex {
    /// Create a new quantized HNSW index (untrained).
    ///
    /// Vectors added before training go into the inner HnswIndex arena.
    /// Call `train_quantizer` after bulk-loading to switch to quantized mode.
    pub fn new(
        dimension: usize,
        metric: DistanceMetricType,
        params: HnswParams,
        config: ScalarQuantizationConfig,
    ) -> Self {
        Self {
            hnsw: HnswIndex::new(dimension, metric, params),
            quantizer: RwLock::new(None),
            codes: RwLock::new(QuantizedArena::new(dimension)),
            code_slots: RwLock::new(HashMap::new()),
            mmap_store: RwLock::new(None),
            config,
            metric,
            distance_fn: DistanceMetric::from_metric_type(metric),
            dimension,
            trained: AtomicBool::new(false),
            cached_scales: RwLock::new(Vec::new()),
            cached_min_vals: RwLock::new(Vec::new()),
            data_dir: RwLock::new(None),
        }
    }

    /// Create a QuantizedHnswIndex wrapping an existing HnswIndex.
    ///
    /// Used during recovery: the HNSW graph (topology + vectors) has already
    /// been restored; this constructor wraps it so that `train_quantizer()`
    /// can be called to switch to quantized mode.
    pub fn from_existing_hnsw(
        hnsw: HnswIndex,
        metric: DistanceMetricType,
        config: ScalarQuantizationConfig,
    ) -> Self {
        let dimension = hnsw.dimension();
        Self {
            hnsw,
            quantizer: RwLock::new(None),
            codes: RwLock::new(QuantizedArena::new(dimension)),
            code_slots: RwLock::new(HashMap::new()),
            mmap_store: RwLock::new(None),
            config,
            metric,
            distance_fn: DistanceMetric::from_metric_type(metric),
            dimension,
            trained: AtomicBool::new(false),
            cached_scales: RwLock::new(Vec::new()),
            cached_min_vals: RwLock::new(Vec::new()),
            data_dir: RwLock::new(None),
        }
    }

    /// Returns whether the quantizer has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained.load(Ordering::Acquire)
    }

    /// Returns a reference to the underlying HNSW index.
    pub fn hnsw(&self) -> &HnswIndex {
        &self.hnsw
    }

    /// Store the collection data directory so `post_optimize()` can train
    /// the quantizer without an explicit path argument.
    pub fn set_data_dir(&self, dir: PathBuf) {
        *self.data_dir.write() = Some(dir);
    }

    // ── Training ────────────────────────────────────────────────────────

    /// Train the scalar quantizer, build mmap store, populate quantized codes,
    /// and free the f32 arena from RAM.
    ///
    /// Collects all vectors from the inner HNSW, writes them to an mmap file,
    /// trains the quantizer with the configured quantile, quantizes every
    /// vector into the QuantizedArena, caches SIMD parameters, and clears
    /// the HNSW f32 arena.
    pub fn train_quantizer(&self, data_dir: &Path) {
        // 1. Collect all f32 vectors from the inner HNSW arena.
        let vectors = self.hnsw.iter_vectors();
        if vectors.is_empty() {
            return;
        }

        // 2. Write f32 vectors to disk via MmapVectorStore.
        let mmap_path = data_dir.join("vectors.mmap");
        let mmap_store = MmapVectorStore::build(&mmap_path, &vectors, self.dimension)
            .expect("failed to build mmap vector store");

        // 3. Train ScalarQuantizer on the training data.
        let mut quantizer = ScalarQuantizer::new(self.dimension);
        let vec_refs: Vec<&[f32]> = vectors.iter().map(|(_, v)| v.as_slice()).collect();
        quantizer
            .train_with_quantile(&vec_refs, self.config.quantile)
            .expect("failed to train scalar quantizer");

        // 4. Cache scales and min_vals for SIMD distance kernels.
        let scales = quantizer.scales();
        let min_vals = quantizer.min_vals().to_vec();

        // 5. Batch quantize all vectors into the QuantizedArena.
        let mut codes = QuantizedArena::with_capacity(self.dimension, vectors.len());
        let mut code_slots = HashMap::with_capacity(vectors.len());
        for (id, vec) in &vectors {
            let code = quantizer
                .quantize(vec)
                .expect("quantize failed for trained quantizer");
            let slot = codes.push(&code);
            code_slots.insert(*id, slot);
        }

        // 6. Store everything under write locks.
        *self.quantizer.write() = Some(quantizer);
        *self.codes.write() = codes;
        *self.code_slots.write() = code_slots;
        *self.mmap_store.write() = Some(mmap_store);
        *self.cached_scales.write() = scales;
        *self.cached_min_vals.write() = min_vals;

        // 7. Free f32 vectors from RAM — graph topology is preserved.
        self.hnsw.clear_arena();
        self.trained.store(true, Ordering::Release);
    }

    // ── Asymmetric distance ─────────────────────────────────────────────

    /// Compute asymmetric distance: f32 query vs u8 code.
    /// Uses SIMD-accelerated kernels for L2, dot product, and cosine.
    /// Falls back to scalar computation for Manhattan.
    #[inline]
    fn asymmetric_distance(
        &self,
        query: &[f32],
        code: &[u8],
        scales: &[f32],
        min_vals: &[f32],
    ) -> f32 {
        match self.metric {
            DistanceMetricType::Euclidean => {
                sq_asymmetric_l2_simd(query, code, min_vals, scales)
            }
            DistanceMetricType::DotProduct => {
                sq_asymmetric_dot_simd(query, code, min_vals, scales)
            }
            DistanceMetricType::Cosine => {
                sq_asymmetric_cosine_simd(query, code, min_vals, scales)
            }
            DistanceMetricType::Manhattan => {
                // No SIMD kernel for Manhattan; dequantize and compute inline.
                let mut sum = 0.0f32;
                for d in 0..query.len() {
                    let dequant = code[d] as f32 * scales[d] + min_vals[d];
                    sum += (query[d] - dequant).abs();
                }
                sum
            }
        }
    }

    // ── Quantized beam search ───────────────────────────────────────────

    /// Custom HNSW beam search using asymmetric distance (f32 query x u8 code).
    ///
    /// Traverses the HNSW graph from the top layer down to layer 0 using
    /// quantized distances. Returns up to `expanded_k` candidates sorted
    /// by ascending quantized distance (closest first).
    fn quantized_search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        oversampling: f32,
    ) -> Vec<ScoredResult> {
        let entry = match self.hnsw.entry_point() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let max_level = self.hnsw.max_level();
        let codes = self.codes.read();
        let code_slots = self.code_slots.read();
        let scales = self.cached_scales.read();
        let min_vals = self.cached_min_vals.read();

        // Closure: compute quantized distance for a node.
        let node_distance = |id: VectorId| -> f32 {
            if let Some(&slot) = code_slots.get(&id) {
                let code = codes.get(slot);
                self.asymmetric_distance(query, code, &scales, &min_vals)
            } else {
                f32::MAX
            }
        };

        // Phase 1: Greedy descent from top layer to layer 1.
        let mut current_ep = entry;
        let mut current_dist = node_distance(current_ep);

        for level in (1..=max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = self.hnsw.neighbors(current_ep, level) {
                    for &neighbor in &neighbors {
                        let dist = node_distance(neighbor);
                        if dist < current_dist {
                            current_ep = neighbor;
                            current_dist = dist;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Phase 2: Beam search at layer 0.
        let expanded_k = ((k as f32 * oversampling) as usize).max(k).max(ef_search);

        // candidates: min-heap (closest first for expansion)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, VectorId)>> =
            BinaryHeap::new();
        // results: max-heap (farthest first for eviction)
        let mut results: BinaryHeap<(OrderedFloat<f32>, VectorId)> = BinaryHeap::new();
        let mut visited: HashSet<VectorId> = HashSet::new();

        visited.insert(current_ep);
        let ep_dist = node_distance(current_ep);
        candidates.push(Reverse((OrderedFloat(ep_dist), current_ep)));
        results.push((OrderedFloat(ep_dist), current_ep));

        while let Some(Reverse((c_dist, c_id))) = candidates.pop() {
            // If closest candidate is farther than farthest result, stop.
            if let Some(&(f_dist, _)) = results.peek() {
                if c_dist > f_dist && results.len() >= expanded_k {
                    break;
                }
            }

            if let Some(neighbors) = self.hnsw.neighbors(c_id, 0) {
                for &neighbor in &neighbors {
                    if visited.insert(neighbor) {
                        let dist = node_distance(neighbor);

                        let should_add = results.len() < expanded_k || {
                            if let Some(&(f_dist, _)) = results.peek() {
                                OrderedFloat(dist) < f_dist
                            } else {
                                true
                            }
                        };

                        if should_add {
                            candidates.push(Reverse((OrderedFloat(dist), neighbor)));
                            results.push((OrderedFloat(dist), neighbor));
                            if results.len() > expanded_k {
                                results.pop(); // evict farthest
                            }
                        }
                    }
                }
            }
        }

        // Drain results into a vec sorted by ascending distance (closest first).
        let mut result_vec: Vec<ScoredResult> = results
            .into_vec()
            .into_iter()
            .map(|(dist, id)| ScoredResult::new(id, dist.into_inner()))
            .collect();
        result_vec.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        result_vec
    }

    // ── Rescore with exact f32 ──────────────────────────────────────────

    /// Re-rank candidates using exact f32 vectors from the mmap store.
    fn rescore(
        &self,
        query: &[f32],
        candidates: Vec<ScoredResult>,
        k: usize,
    ) -> Vec<ScoredResult> {
        let mmap = self.mmap_store.read();
        let store = match mmap.as_ref() {
            Some(s) => s,
            None => return candidates.into_iter().take(k).collect(),
        };

        let mut rescored: Vec<ScoredResult> = candidates
            .iter()
            .filter_map(|sr| {
                store.get_vector(sr.id).map(|vec| {
                    ScoredResult::new(sr.id, self.distance_fn.compute(query, vec))
                })
            })
            .collect();

        rescored.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        rescored.truncate(k);
        rescored
    }

    // ── Public search with explicit quantization params ──────────────────

    /// Search with explicit quantization parameters (internal implementation).
    ///
    /// Called from the trait method and directly from the server layer when
    /// the client provides per-query quantization overrides (rescore,
    /// oversampling, ignore).
    pub fn search_with_quantization_impl(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        params: &SearchQuantizationParams,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        if !self.trained.load(Ordering::Acquire) || params.ignore {
            return self.hnsw.search(query, k, ef_search);
        }

        let ef = ef_search.unwrap_or(50);
        let candidates = self.quantized_search(query, k, ef, params.oversampling);

        if params.rescore {
            Ok(self.rescore(query, candidates, k))
        } else {
            Ok(candidates.into_iter().take(k).collect())
        }
    }
}

// ── VectorIndex implementation ──────────────────────────────────────────

impl VectorIndex for QuantizedHnswIndex {
    fn add(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        // Always insert into HNSW graph (topology + arena before training).
        self.hnsw.add(id, vector)?;

        if self.trained.load(Ordering::Acquire) {
            // Quantize and store the code.
            let quantizer = self.quantizer.read();
            if let Some(q) = quantizer.as_ref() {
                let code = q.quantize(vector).map_err(|e| {
                    IndexError::Internal(format!("quantize failed: {}", e))
                })?;
                let slot = self.codes.write().push(&code);
                self.code_slots.write().insert(id, slot);
            }

            // Append exact f32 vector to the mmap store.
            let mut mmap = self.mmap_store.write();
            if let Some(store) = mmap.as_mut() {
                let _ = store.append_vector(id, vector);
            }
        }

        Ok(())
    }

    fn remove(&self, id: VectorId) -> Result<(), IndexError> {
        self.hnsw.remove(id)?;

        if self.trained.load(Ordering::Acquire) {
            // Free the quantized code slot.
            let mut code_slots = self.code_slots.write();
            if let Some(slot) = code_slots.remove(&id) {
                self.codes.write().free(slot);
            }
            // Lazy-remove from mmap store (reclaimed on rebuild).
            let mut mmap = self.mmap_store.write();
            if let Some(store) = mmap.as_mut() {
                store.remove_vector(id);
            }
        }

        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        if !self.trained.load(Ordering::Acquire) {
            return self.hnsw.search(query, k, ef_search);
        }

        let ef = ef_search.unwrap_or(50);
        let params = SearchQuantizationParams::default();
        let candidates = self.quantized_search(query, k, ef, params.oversampling);

        if params.rescore {
            Ok(self.rescore(query, candidates, k))
        } else {
            Ok(candidates.into_iter().take(k).collect())
        }
    }

    fn search_with_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
        _ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        if !self.trained.load(Ordering::Acquire) {
            return self.hnsw.search_with_candidates(query, k, candidates, _ef_search);
        }

        // Score each candidate using quantized distance.
        let codes = self.codes.read();
        let code_slots = self.code_slots.read();
        let scales = self.cached_scales.read();
        let min_vals = self.cached_min_vals.read();

        let mut scored: Vec<ScoredResult> = candidates
            .iter()
            .filter_map(|&id| {
                code_slots.get(&id).map(|&slot| {
                    let code = codes.get(slot);
                    let score = self.asymmetric_distance(query, code, &scales, &min_vals);
                    ScoredResult::new(id, score)
                })
            })
            .collect();

        scored.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        scored.truncate(k);

        // Rescore the top-k with exact f32 distances.
        Ok(self.rescore(query, scored, k))
    }

    fn search_with_quantization(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        params: &SearchQuantizationParams,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        self.search_with_quantization_impl(query, k, ef_search, params)
    }

    fn search_with_candidates_quantized(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
        ef_search: Option<usize>,
        params: &SearchQuantizationParams,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        if !self.trained.load(Ordering::Acquire) || params.ignore {
            return self.hnsw.search_with_candidates(query, k, candidates, ef_search);
        }

        // Score each candidate using quantized distance.
        let codes = self.codes.read();
        let code_slots = self.code_slots.read();
        let scales = self.cached_scales.read();
        let min_vals = self.cached_min_vals.read();

        let mut scored: Vec<ScoredResult> = candidates
            .iter()
            .filter_map(|&id| {
                code_slots.get(&id).map(|&slot| {
                    let code = codes.get(slot);
                    let score = self.asymmetric_distance(query, code, &scales, &min_vals);
                    ScoredResult::new(id, score)
                })
            })
            .collect();

        scored.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        scored.truncate(k);

        if params.rescore {
            Ok(self.rescore(query, scored, k))
        } else {
            Ok(scored)
        }
    }

    fn len(&self) -> usize {
        self.hnsw.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn contains(&self, id: VectorId) -> bool {
        self.hnsw.contains(id)
    }

    fn get_vector(&self, id: VectorId) -> Result<Vec<f32>, IndexError> {
        // Prefer exact f32 from mmap store when available.
        if self.trained.load(Ordering::Acquire) {
            let mmap = self.mmap_store.read();
            if let Some(store) = mmap.as_ref() {
                if let Some(vec) = store.get_vector(id) {
                    return Ok(vec.to_vec());
                }
            }
        }
        self.hnsw.get_vector(id)
    }

    fn iter_vectors(&self) -> Result<Vec<(VectorId, Vec<f32>)>, IndexError> {
        if self.trained.load(Ordering::Acquire) {
            let mmap = self.mmap_store.read();
            if let Some(store) = mmap.as_ref() {
                let code_slots = self.code_slots.read();
                let vecs: Vec<(VectorId, Vec<f32>)> = code_slots
                    .keys()
                    .filter_map(|&id| store.get_vector(id).map(|v| (id, v.to_vec())))
                    .collect();
                return Ok(vecs);
            }
        }
        Ok(self.hnsw.iter_vectors())
    }
}

// ── PersistableIndex implementation ─────────────────────────────────────

impl PersistableIndex for QuantizedHnswIndex {
    fn add_with_lsn(
        &self,
        id: VectorId,
        vector: &[f32],
        lsn: u64,
    ) -> Result<(), IndexError> {
        self.hnsw.add_with_lsn(id, vector, lsn)?;

        if self.trained.load(Ordering::Acquire) {
            let quantizer = self.quantizer.read();
            if let Some(q) = quantizer.as_ref() {
                let code = q.quantize(vector).map_err(|e| {
                    IndexError::Internal(format!("quantize failed: {}", e))
                })?;
                let slot = self.codes.write().push(&code);
                self.code_slots.write().insert(id, slot);
            }
            let mut mmap = self.mmap_store.write();
            if let Some(store) = mmap.as_mut() {
                let _ = store.append_vector(id, vector);
            }
        }

        Ok(())
    }

    fn remove_with_lsn(&self, id: VectorId, lsn: u64) -> Result<(), IndexError> {
        self.hnsw.remove_with_lsn(id, lsn)?;

        if self.trained.load(Ordering::Acquire) {
            let mut code_slots = self.code_slots.write();
            if let Some(slot) = code_slots.remove(&id) {
                self.codes.write().free(slot);
            }
            let mut mmap = self.mmap_store.write();
            if let Some(store) = mmap.as_mut() {
                store.remove_vector(id);
            }
        }

        Ok(())
    }

    fn snapshot_topology(&self, snapshot_lsn: u64) -> HnswTopologySnapshot {
        self.hnsw.snapshot_topology(snapshot_lsn)
    }

    fn compact(&self) {
        self.hnsw.compact();
    }

    fn is_compacted(&self) -> bool {
        self.hnsw.is_compacted()
    }

    fn build_parallel(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        self.hnsw.build_parallel(vectors)
    }

    fn set_delta_writer(&self, writer: HnswDeltaWriter) {
        self.hnsw.set_delta_writer(writer);
    }

    fn take_delta_writer(&self) -> Option<HnswDeltaWriter> {
        self.hnsw.take_delta_writer()
    }

    fn iter_vectors_owned(&self) -> Vec<(VectorId, Vec<f32>)> {
        if self.trained.load(Ordering::Acquire) {
            let mmap = self.mmap_store.read();
            if let Some(store) = mmap.as_ref() {
                let code_slots = self.code_slots.read();
                return code_slots
                    .keys()
                    .filter_map(|&id| store.get_vector(id).map(|v| (id, v.to_vec())))
                    .collect();
            }
        }
        self.hnsw.iter_vectors()
    }

    fn post_optimize(&self) {
        let dir = self.data_dir.read();
        if let Some(ref data_dir) = *dir {
            self.train_quantizer(data_dir);
        }
    }

    fn as_vector_index(&self) -> &dyn VectorIndex {
        self
    }
}
