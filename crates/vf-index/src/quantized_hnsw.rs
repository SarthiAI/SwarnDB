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
use crate::parallel_build::{parallel_encode_dense, plan_batches};
use crate::quantized_arena::QuantizedArena;
use crate::traits::{
    IndexError, ParallelBuildConfig, ParallelBuildUnit, PersistableIndex, RestoreOutcome,
    VectorIndex,
};

// Cap on the number of vectors used to TRAIN the scalar quantizer.
//
// Quantile bounds are statistically stable, so a deterministic strided
// sample of this many vectors yields the same min/max bounds (and hence
// materially identical SQ8 codes) as training on the full dataset, while
// avoiding the transient per-dimension f32 buffers the quantizer allocates
// over every vector. Encoding still runs over ALL vectors; only the bound
// estimation is sampled. At or below the cap, every vector is used.
const QUANTIZER_TRAIN_SAMPLE_CAP: usize = 262_144;

/// Build a deterministic, evenly strided sample of borrowed training slices.
///
/// Borrows directly from `vectors`, so no second owned f32 copy is made.
/// At or below `QUANTIZER_TRAIN_SAMPLE_CAP` every vector is returned; above
/// it, a fixed stride picks an even spread so the result is reproducible.
fn sampled_train_refs(vectors: &[(VectorId, Vec<f32>)]) -> Vec<&[f32]> {
    let n = vectors.len();
    if n <= QUANTIZER_TRAIN_SAMPLE_CAP {
        return vectors.iter().map(|(_, v)| v.as_slice()).collect();
    }
    let stride = n / QUANTIZER_TRAIN_SAMPLE_CAP;
    vectors
        .iter()
        .step_by(stride)
        .take(QUANTIZER_TRAIN_SAMPLE_CAP)
        .map(|(_, v)| v.as_slice())
        .collect()
}

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

        // 3. Train ScalarQuantizer on a borrowed, deterministically sampled
        //    view of the data. The refs borrow `vectors`, so training adds no
        //    owned f32 copy, and sampling caps the quantizer's transient
        //    per-dimension buffers so peak RSS right after the build no longer
        //    doubles. Quantile bounds are stable under sampling, so the codes
        //    are materially unchanged.
        let mut quantizer = ScalarQuantizer::new(self.dimension);
        let vec_refs: Vec<&[f32]> = sampled_train_refs(&vectors);
        quantizer
            .train_with_quantile(&vec_refs, self.config.quantile)
            .expect("failed to train scalar quantizer");

        // 4. Cache scales and min_vals for SIMD distance kernels.
        let scales = quantizer.scales();
        let min_vals = quantizer.min_vals().to_vec();

        // 5. Stable-sort by VectorId so the parallel encode path produces
        //    deterministic on-disk bytes regardless of worker count, then
        //    encode in parallel directly into a dense byte buffer. This
        //    replaces the old "encode in parallel into Vec<Vec<u8>> then
        //    sequentially push into QuantizedArena" bottleneck.
        let mut sorted_vectors: Vec<(VectorId, &[f32])> = vectors
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();
        sorted_vectors.sort_by_key(|(id, _)| *id);

        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let config = ParallelBuildConfig {
            workers,
            memory_cap_bytes: None,
            deterministic: true,
        };

        let (data, _slot_indices) =
            match parallel_encode_dense(&sorted_vectors, &quantizer, &config) {
                Ok(pair) => pair,
                Err(e) => {
                    log::warn!("parallel_encode_dense failed inside train_quantizer; keeping previous in-memory state unchanged: {e}");
                    return;
                }
            };

        let codes = QuantizedArena::from_dense(data, self.dimension, sorted_vectors.len());

        let mut code_slots: HashMap<VectorId, usize> =
            HashMap::with_capacity(sorted_vectors.len());
        let mut slot_ids: Vec<VectorId> = Vec::with_capacity(sorted_vectors.len());
        for (slot, (id, _)) in sorted_vectors.iter().enumerate() {
            code_slots.insert(*id, slot);
            slot_ids.push(*id);
        }

        // 6. Persist quantizer state and codes to disk so the next restart
        //    can take the fast path. Failure to persist is logged but
        //    non-fatal, the in-memory state is still valid.
        let quantizer_path = data_dir.join("quantizer.json");
        if let Err(e) = quantizer.save_to_path(&quantizer_path) {
            log::warn!("failed to persist quantizer.json: {e}");
        }
        let codes_path = data_dir.join("codes.bin");
        if let Err(e) = codes.save_to_path(&codes_path, &slot_ids) {
            log::warn!("failed to persist codes.bin: {e}");
        }

        // 7. Store everything under write locks.
        *self.quantizer.write() = Some(quantizer);
        *self.codes.write() = codes;
        *self.code_slots.write() = code_slots;
        *self.mmap_store.write() = Some(mmap_store);
        *self.cached_scales.write() = scales;
        *self.cached_min_vals.write() = min_vals;

        // Retain f32 arena: HNSW insert path reads neighbour vectors from it. The mmap rescore store on disk is the durable f32 copy.
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

    // ── SQ8 disk-state save helper ──────────────────────────────────────

    /// Persist the three SQ8 artefacts (quantizer.json, codes.bin, vectors.mmap)
    /// to the target directory. Format compatible with quantization_v3.
    ///
    /// Returns an error only if the quantizer is untrained or a hard IO
    /// failure occurs on the quantizer or codes write. The mmap file is
    /// already synced to disk by MmapVectorStore (build, append, and rebuild
    /// each call file.sync_all before remapping), so this helper does not
    /// re-flush the mmap.
    fn save_quantization_state_to_dir(&self, dir: &Path) -> Result<(), IndexError> {
        if !self.trained.load(Ordering::Acquire) {
            return Err(IndexError::Internal(
                "save_quantization_state_to_dir called on untrained QuantizedHnswIndex".into(),
            ));
        }

        if let Err(e) = std::fs::create_dir_all(dir) {
            return Err(IndexError::Internal(format!(
                "save_quantization_state_to_dir: failed to create {}: {}",
                dir.display(),
                e
            )));
        }

        // 1. Write quantizer.json via the existing atomic save path.
        {
            let quantizer_guard = self.quantizer.read();
            let quantizer = quantizer_guard.as_ref().ok_or_else(|| {
                IndexError::Internal(
                    "save_quantization_state_to_dir: quantizer is missing".into(),
                )
            })?;
            let quantizer_path = dir.join("quantizer.json");
            quantizer.save_to_path(&quantizer_path).map_err(|e| {
                IndexError::Internal(format!(
                    "save_quantization_state_to_dir: failed to write quantizer.json: {}",
                    e
                ))
            })?;
        }

        // 2. Write codes.bin via the existing atomic save path. Build the
        //    slot table from code_slots, indexed by slot number.
        {
            let codes = self.codes.read();
            let code_slots = self.code_slots.read();
            let mut slot_ids: Vec<VectorId> = vec![0; codes.len()];
            for (id, &slot) in code_slots.iter() {
                if slot >= slot_ids.len() {
                    return Err(IndexError::Internal(format!(
                        "save_quantization_state_to_dir: slot {} out of range for codes.len={}",
                        slot,
                        codes.len()
                    )));
                }
                slot_ids[slot] = *id;
            }
            let codes_path = dir.join("codes.bin");
            codes.save_to_path(&codes_path, &slot_ids).map_err(|e| {
                IndexError::Internal(format!(
                    "save_quantization_state_to_dir: failed to write codes.bin: {}",
                    e
                ))
            })?;
        }

        // 3. vectors.mmap is already file-backed and synced by MmapVectorStore
        //    on every build, append, or rebuild. No further action required.

        Ok(())
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

    fn metric_type(&self) -> DistanceMetricType {
        self.metric
    }

    fn is_quantized(&self) -> bool {
        self.trained.load(Ordering::Acquire)
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

    fn cold_build_parallel(
        &mut self,
        vectors: &[(VectorId, &[f32])],
        config: ParallelBuildConfig,
    ) -> Result<(), IndexError> {
        if vectors.is_empty() {
            return Ok(());
        }

        // 1. Pre-flight dimension check so a bad input does not leave the
        //    index half-built.
        for (_id, v) in vectors.iter() {
            if v.len() != self.dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len(),
                });
            }
        }

        // 2. Pre-flight uniqueness check.
        {
            let mut seen: HashSet<VectorId> = HashSet::with_capacity(vectors.len());
            for (id, _) in vectors.iter() {
                if !seen.insert(*id) {
                    return Err(IndexError::AlreadyExists(*id));
                }
            }
        }

        // 3. Always sort by VectorId for deterministic on-disk bytes. The
        //    field `config.deterministic` is honored implicitly: we always
        //    sort because the cost is negligible relative to encode.
        let mut sorted_vectors: Vec<(VectorId, &[f32])> = vectors.to_vec();
        sorted_vectors.sort_by_key(|(id, _)| *id);

        // 4. Build the inner HNSW graph in PARALLEL into a LOCAL HnswIndex
        //    so that any failure does not leave `self.hnsw` partially
        //    populated. Mirrors the plain HNSW cold path
        //    (hnsw.rs HnswIndex::cold_build_parallel -> build_parallel):
        //    build_parallel fully populates a fresh inner index (arena,
        //    topology, entry point, max level) using all cores, so the
        //    graph build no longer single-threads while encode workers idle.
        let local_hnsw =
            HnswIndex::new(self.dimension, self.metric, self.hnsw.params().clone());
        local_hnsw.build_parallel(&sorted_vectors)?;

        // 5. Resolve the data directory before any disk work.
        let dir_guard = self.data_dir.read();
        let data_dir = match dir_guard.as_ref() {
            Some(dir) => dir.clone(),
            None => {
                return Err(IndexError::Internal(
                    "QuantizedHnswIndex::cold_build_parallel: data_dir is unset; \
                     call set_data_dir before cold_build_parallel"
                        .into(),
                ));
            }
        };
        drop(dir_guard);

        // 6. Build the mmap rescore store into a LOCAL handle.
        let mmap_vectors: Vec<(VectorId, Vec<f32>)> = sorted_vectors
            .iter()
            .map(|(id, v)| (*id, v.to_vec()))
            .collect();
        let mmap_path = data_dir.join("vectors.mmap");
        let local_mmap_store = MmapVectorStore::build(&mmap_path, &mmap_vectors, self.dimension)
            .map_err(|e| {
                IndexError::Internal(format!(
                    "cold_build_parallel: failed to build mmap vector store: {e}"
                ))
            })?;

        // 7. Train the scalar quantizer on a deterministically sampled,
        //    borrowed view of the sorted slice. Capping the training set keeps
        //    the quantizer's transient per-dimension buffers small (no doubling
        //    of peak RSS); quantile bounds are stable under sampling, so the
        //    codes are materially unchanged. Stays local until publication.
        let mut local_quantizer = ScalarQuantizer::new(self.dimension);
        let vec_refs: Vec<&[f32]> = if sorted_vectors.len() <= QUANTIZER_TRAIN_SAMPLE_CAP {
            sorted_vectors.iter().map(|(_, v)| *v).collect()
        } else {
            let stride = sorted_vectors.len() / QUANTIZER_TRAIN_SAMPLE_CAP;
            sorted_vectors
                .iter()
                .step_by(stride)
                .take(QUANTIZER_TRAIN_SAMPLE_CAP)
                .map(|(_, v)| *v)
                .collect()
        };
        local_quantizer
            .train_with_quantile(&vec_refs, self.config.quantile)
            .map_err(|e| {
                IndexError::Internal(format!(
                    "cold_build_parallel: failed to train scalar quantizer: {e}"
                ))
            })?;

        // 8. Parallel encode into a dense byte buffer.
        let (data, _slot_indices) =
            parallel_encode_dense(&sorted_vectors, &local_quantizer, &config)?;

        let local_codes =
            QuantizedArena::from_dense(data, self.dimension, sorted_vectors.len());

        let mut local_code_slots: HashMap<VectorId, usize> =
            HashMap::with_capacity(sorted_vectors.len());
        let mut slot_ids: Vec<VectorId> = Vec::with_capacity(sorted_vectors.len());
        for (slot, (id, _)) in sorted_vectors.iter().enumerate() {
            local_code_slots.insert(*id, slot);
            slot_ids.push(*id);
        }

        // 9. Persist quantizer state and codes for the next restart's
        //    fast path. Failure to persist is logged but non-fatal, the
        //    in-memory state is still valid.
        let quantizer_path = data_dir.join("quantizer.json");
        if let Err(e) = local_quantizer.save_to_path(&quantizer_path) {
            log::warn!("cold_build_parallel: failed to persist quantizer.json: {e}");
        }
        let codes_path = data_dir.join("codes.bin");
        if let Err(e) = local_codes.save_to_path(&codes_path, &slot_ids) {
            log::warn!("cold_build_parallel: failed to persist codes.bin: {e}");
        }

        // 10. Compute cached SIMD parameters from the (already trained)
        //     local quantizer. Preserves the original ordering of
        //     scales / min_vals capture.
        let scales = local_quantizer.scales();
        let min_vals = local_quantizer.min_vals().to_vec();

        // 11. ATOMIC PUBLICATION. Every step above has succeeded, so the
        //     build is committed in one block. No `self.*` mutation
        //     happens before this point, which keeps the trait-contract
        //     promise that the index is left empty on any error.
        self.hnsw = local_hnsw;
        *self.quantizer.write() = Some(local_quantizer);
        *self.codes.write() = local_codes;
        *self.code_slots.write() = local_code_slots;
        *self.mmap_store.write() = Some(local_mmap_store);
        *self.cached_scales.write() = scales;
        *self.cached_min_vals.write() = min_vals;

        // Retain f32 arena: HNSW insert path reads neighbour vectors from it. The mmap rescore store on disk is the durable f32 copy.
        self.trained.store(true, Ordering::Release);

        log::info!(
            "QuantizedHnswIndex cold_build_parallel complete: collection_dim={} vector_count={} workers={}",
            self.dimension,
            vectors.len(),
            config.workers
        );

        Ok(())
    }

    fn parallel_build_units(
        &self,
        vectors: &[(VectorId, &[f32])],
        config: &ParallelBuildConfig,
    ) -> Vec<ParallelBuildUnit> {
        if vectors.is_empty() {
            return Vec::new();
        }
        let code_size = self.dimension;
        let batches = plan_batches(vectors.len(), code_size, config);
        batches
            .into_iter()
            .enumerate()
            .map(|(shard_index, (start, end))| ParallelBuildUnit {
                shard_index,
                vector_count: end - start,
                byte_range_hint: Some((start * code_size, end * code_size)),
            })
            .collect()
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

    // ── Fast-restart trait surface ──────────────────────────────────────

    fn serialize_state_to_dir(&self, dir: &Path) -> Result<(), IndexError> {
        // Composition: delegate the HNSW base graph snapshot to the inner
        // index, then write the SQ8 layer on top.
        self.hnsw.serialize_state_to_dir(dir)?;

        // Auto-train on first persistence if optimize was never called.
        // train_quantizer uses interior mutability and is callable through &self.
        if !self.trained.load(Ordering::Acquire) {
            let configured_dir_opt = self.data_dir.read().clone();
            if let Some(configured_dir) = configured_dir_opt {
                let target = if configured_dir == dir {
                    configured_dir
                } else {
                    dir.to_path_buf()
                };
                // Catch panics so a graceful-shutdown caller never crashes
                // the process on edge-case training failures.
                let train_result = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| self.train_quantizer(&target)),
                );
                if train_result.is_err() {
                    return Err(IndexError::Internal(
                        "auto-train inside serialize_state_to_dir panicked".into(),
                    ));
                }
                // train_quantizer returns silently on empty inner-HNSW; trained stays false,
                // save_quantization_state_to_dir below will then surface the untrained error.
            }
        }

        self.save_quantization_state_to_dir(dir)?;
        Ok(())
    }

    fn recovery_files(&self) -> &'static [&'static str] {
        &[
            "hnsw.base",
            "hnsw.delta",
            "shutdown_clean",
            "quantizer.json",
            "codes.bin",
            "vectors.mmap",
        ]
    }

    fn try_restore_from_dir(&mut self, dir: &Path) -> Result<RestoreOutcome, IndexError> {
        // Format: compatible with quantization_v3 on-disk SQ8 layout.

        // 1. Restore the inner HNSW first. Propagate StateMissing or
        //    StateCorrupt outcomes without touching the SQ8 layer.
        let hnsw_outcome = self.hnsw.try_restore_from_dir(dir)?;
        let strategy = match hnsw_outcome {
            RestoreOutcome::StateMissing => return Ok(RestoreOutcome::StateMissing),
            RestoreOutcome::StateCorrupt { reason } => {
                return Ok(RestoreOutcome::StateCorrupt { reason });
            }
            RestoreOutcome::Restored { strategy } => strategy,
        };

        // 2. Existence check on the 3 SQ8 artefacts. Treat any missing
        //    file as StateMissing so AppState falls back to FullRebuild.
        let quantizer_path = dir.join("quantizer.json");
        let codes_path = dir.join("codes.bin");
        let mmap_path = dir.join("vectors.mmap");
        if !quantizer_path.exists() || !codes_path.exists() || !mmap_path.exists() {
            return Ok(RestoreOutcome::StateMissing);
        }

        let hnsw_count = self.hnsw.vector_count();

        // 3. Load the quantizer state. Soft-fail on validation errors.
        let quantizer = match ScalarQuantizer::load_from_path(&quantizer_path) {
            Ok(q) => q,
            Err(e) => {
                return Ok(RestoreOutcome::StateCorrupt {
                    reason: format!("quantizer.json load failed: {}", e),
                });
            }
        };
        if quantizer.dimension() != self.dimension {
            return Ok(RestoreOutcome::StateCorrupt {
                reason: format!(
                    "quantizer dimension {} does not match configured {}",
                    quantizer.dimension(),
                    self.dimension
                ),
            });
        }

        // 4. Load the persisted codes plus slot table.
        let (codes, slot_ids) = match QuantizedArena::load_from_path(&codes_path, self.dimension) {
            Ok(pair) => pair,
            Err(e) => {
                return Ok(RestoreOutcome::StateCorrupt {
                    reason: format!("codes.bin load failed: {}", e),
                });
            }
        };
        if codes.len() != hnsw_count {
            return Ok(RestoreOutcome::StateCorrupt {
                reason: format!(
                    "codes.bin count {} does not match hnsw vector_count {}",
                    codes.len(),
                    hnsw_count
                ),
            });
        }
        if slot_ids.len() != codes.len() {
            return Ok(RestoreOutcome::StateCorrupt {
                reason: "codes.bin slot_ids length does not match codes length".into(),
            });
        }
        for &id in &slot_ids {
            if !self.hnsw.contains(id) {
                return Ok(RestoreOutcome::StateCorrupt {
                    reason: format!("codes.bin references id {} not present in HNSW", id),
                });
            }
        }

        // 5. Build the code_slots map and reject duplicate ids in the
        //    persisted slot table.
        let mut code_slots: HashMap<VectorId, usize> = HashMap::with_capacity(slot_ids.len());
        for (slot, id) in slot_ids.iter().enumerate() {
            code_slots.insert(*id, slot);
        }
        if code_slots.len() != hnsw_count {
            return Ok(RestoreOutcome::StateCorrupt {
                reason: format!(
                    "codes.bin slot table has duplicate ids ({} unique vs {} slots)",
                    code_slots.len(),
                    hnsw_count
                ),
            });
        }

        // 6. Open the on-disk f32 vector store and validate its count
        //    against the HNSW. No re-write of the mmap file.
        let mmap_store = match MmapVectorStore::from_file(&mmap_path) {
            Ok(s) => s,
            Err(e) => {
                return Ok(RestoreOutcome::StateCorrupt {
                    reason: format!("vectors.mmap open failed: {}", e),
                });
            }
        };
        if mmap_store.len() != hnsw_count {
            return Ok(RestoreOutcome::StateCorrupt {
                reason: format!(
                    "vectors.mmap count {} does not match hnsw vector_count {}",
                    mmap_store.len(),
                    hnsw_count
                ),
            });
        }

        // 7. Cache SIMD scratch fields before the publication step.
        let scales = quantizer.scales();
        let min_vals = quantizer.min_vals().to_vec();

        // 8. Reset any pre-existing SQ8 state on self so a partially-
        //    populated caller does not leak codes from a prior collection
        //    into the freshly-restored view.
        {
            *self.quantizer.write() = None;
            *self.codes.write() = QuantizedArena::new(self.dimension);
            self.code_slots.write().clear();
            *self.mmap_store.write() = None;
            self.cached_scales.write().clear();
            self.cached_min_vals.write().clear();
            self.trained.store(false, Ordering::Release);
        }

        // 9. Publish the loaded SQ8 state.
        *self.quantizer.write() = Some(quantizer);
        *self.codes.write() = codes;
        *self.code_slots.write() = code_slots;
        *self.mmap_store.write() = Some(mmap_store);
        *self.cached_scales.write() = scales;
        *self.cached_min_vals.write() = min_vals;
        *self.data_dir.write() = Some(dir.to_path_buf());
        self.trained.store(true, Ordering::Release);

        // Retain f32 arena: HNSW insert path reads neighbour vectors from it. The mmap rescore store on disk is the durable f32 copy.

        Ok(RestoreOutcome::Restored { strategy })
    }

    fn validate_state_on_disk(dir: &Path, dimension: usize) -> Result<bool, IndexError>
    where
        Self: Sized,
    {
        // 1. Plain HNSW base validation. Soft-fail propagates as Ok(false).
        if !HnswIndex::validate_state_on_disk(dir, dimension)? {
            return Ok(false);
        }

        // 2. quantizer.json: parseable JSON envelope with matching dim.
        let quantizer_path = dir.join("quantizer.json");
        match ScalarQuantizer::load_from_path(&quantizer_path) {
            Ok(q) => {
                if q.dimension() != dimension {
                    return Ok(false);
                }
            }
            Err(_) => return Ok(false),
        }

        // 3. codes.bin: read the 16-byte header and check magic, version,
        //    and dimension fields. Magic b"SQ8C", version u32 LE = 1.
        let codes_path = dir.join("codes.bin");
        match std::fs::read(&codes_path) {
            Ok(bytes) => {
                if bytes.len() < 16 {
                    return Ok(false);
                }
                if &bytes[0..4] != b"SQ8C" {
                    return Ok(false);
                }
                let version = u32::from_le_bytes(
                    bytes[4..8].try_into().unwrap_or([0, 0, 0, 0]),
                );
                if version != 1 {
                    return Ok(false);
                }
                let dim_on_disk = u32::from_le_bytes(
                    bytes[12..16].try_into().unwrap_or([0, 0, 0, 0]),
                ) as usize;
                if dim_on_disk != dimension {
                    return Ok(false);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => {
                return Err(IndexError::Internal(format!(
                    "validate_state_on_disk: failed to read {}: {}",
                    codes_path.display(),
                    e
                )));
            }
        }

        // 4. vectors.mmap: presence check only. Deep validation deferred
        //    to load time inside try_restore_from_dir.
        let mmap_path = dir.join("vectors.mmap");
        if !mmap_path.exists() {
            return Ok(false);
        }

        Ok(true)
    }

    fn clear_state_from_dir(dir: &Path) -> Result<(), IndexError>
    where
        Self: Sized,
    {
        // 1. Clear plain HNSW state first.
        HnswIndex::clear_state_from_dir(dir)?;

        // 2. Remove the 3 SQ8 artefacts. Tolerate ENOENT.
        for name in ["quantizer.json", "codes.bin", "vectors.mmap"].iter() {
            let path = dir.join(name);
            if path.exists() {
                std::fs::remove_file(&path).map_err(|e| {
                    IndexError::Internal(format!(
                        "clear_state_from_dir: failed to remove {}: {}",
                        path.display(),
                        e
                    ))
                })?;
            }
        }

        Ok(())
    }

    fn as_vector_index(&self) -> &dyn VectorIndex {
        self
    }
}
