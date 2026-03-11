// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! IVF+HNSW+PQ hybrid index for billion-scale vector search.
//!
//! Combines three techniques:
//! - **IVF** (Inverted File): partitions vectors into Voronoi cells for coarse filtering
//! - **HNSW**: used as the coarse quantizer for fast partition assignment (instead of flat scan)
//! - **PQ** (Product Quantization): compresses residual vectors for memory efficiency
//!
//! At search time:
//! 1. Find the `nprobe` nearest centroids via HNSW coarse search
//! 2. For each probed partition, build a PQ distance table for the query residual
//! 3. Scan PQ codes and compute approximate distances via table lookup
//! 4. Return top-k results
//!
//! This keeps only compressed data in memory, enabling billion-scale datasets
//! on a single node.

use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rayon::prelude::*;

use vf_core::simd::squared_l2_f32;
use vf_core::types::{ScoredResult, VectorId};
use vf_quantization::kmeans;
use vf_quantization::pq_distance::PqDistanceTable;
use vf_quantization::product::ProductQuantizer;

use crate::hnsw::{HnswIndex, HnswParams};
use crate::traits::{IndexError, VectorIndex};

/// A single entry in an inverted list: vector ID + PQ-compressed residual codes.
struct CompressedEntry {
    id: VectorId,
    codes: Vec<u8>,
}

/// IVF+HNSW+PQ composite index for billion-scale approximate nearest neighbor search.
///
/// The index is organized as:
/// - A small HNSW graph over partition centroids (coarse quantizer)
/// - Per-partition inverted lists storing PQ-encoded residual vectors
/// - A product quantizer trained on residual vectors
///
/// Memory footprint per vector is approximately `M` bytes (where M = number of
/// PQ sub-quantizers), plus a small overhead for the vector ID and inverted list
/// bookkeeping. For 128-dim vectors with M=16 sub-quantizers, this is ~16 bytes
/// per vector vs ~512 bytes for full f32 storage (32x compression).
///
/// **Partition limit:** The effective number of partitions is clamped to
/// `min(num_partitions, training_data_size)` during `train()`.
///
/// **Append-only:** This index supports `add` / `add_batch` but does not
/// support removing or updating individual vectors after insertion.
pub struct IvfHnswPqIndex {
    dimension: usize,
    /// Coarse quantizer: small HNSW index over partition centroids.
    coarse_quantizer: HnswIndex,
    /// Product quantizer for residual compression.
    pq: RwLock<ProductQuantizer>,
    /// Inverted lists: partition_id -> list of (vector_id, pq_codes).
    inverted_lists: Vec<RwLock<Vec<CompressedEntry>>>,
    /// Partition centroids (for residual computation).
    centroids: Vec<Vec<f32>>,
    /// Number of partitions (Voronoi cells).
    num_partitions: usize,
    /// Whether the index has been trained.
    trained: bool,
    /// Total number of vectors stored (atomic for lock-free reads).
    total_vectors: AtomicUsize,
    /// Number of PQ sub-quantizers.
    num_subquantizers: usize,
    /// Set of all inserted vector IDs for O(1) contains() and duplicate detection.
    id_index: RwLock<HashSet<VectorId>>,
}

// Safety: All mutable state is behind RwLock or AtomicUsize.
const _: () = {
    fn _assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        _assert_send_sync::<IvfHnswPqIndex>();
    }
};

impl IvfHnswPqIndex {
    /// Create a new (untrained) IVF+HNSW+PQ index.
    ///
    /// # Arguments
    /// * `dimension` - Dimensionality of the input vectors.
    /// * `num_partitions` - Number of IVF partitions (Voronoi cells). Should be
    ///   roughly `sqrt(n)` for `n` vectors for optimal balance.
    /// * `pq_subquantizers` - Number of PQ sub-quantizers (M). Must evenly divide
    ///   `dimension`. Each vector will be compressed to M bytes.
    ///
    /// # Errors
    /// Returns `IndexError::Internal` if `dimension` is not divisible by
    /// `pq_subquantizers`, or if any parameter is zero.
    pub fn new(
        dimension: usize,
        num_partitions: usize,
        pq_subquantizers: usize,
    ) -> Result<Self, IndexError> {
        if dimension == 0 || num_partitions == 0 || pq_subquantizers == 0 {
            return Err(IndexError::Internal(
                "dimension, num_partitions, and pq_subquantizers must all be > 0".into(),
            ));
        }
        if dimension % pq_subquantizers != 0 {
            return Err(IndexError::Internal(format!(
                "dimension {} is not divisible by pq_subquantizers {}",
                dimension, pq_subquantizers
            )));
        }

        let pq = ProductQuantizer::new(dimension, pq_subquantizers).map_err(|e| {
            IndexError::Internal(format!("failed to create ProductQuantizer: {}", e))
        })?;

        // Coarse quantizer HNSW: small graph, tuned for centroid search.
        // Use smaller params since the centroid set is small.
        let hnsw_params = HnswParams::new(
            16,      // m
            64,      // ef_construction — modest for a small graph
            32,      // ef_search — enough for nprobe lookups
            100_000, // max_ef
            24,      // max_level_cap
        )?;
        let coarse_quantizer = HnswIndex::new(
            dimension,
            vf_core::types::DistanceMetricType::Euclidean,
            hnsw_params,
        );

        let inverted_lists = (0..num_partitions)
            .map(|_| RwLock::new(Vec::new()))
            .collect();

        Ok(Self {
            dimension,
            coarse_quantizer,
            pq: RwLock::new(pq),
            inverted_lists,
            centroids: Vec::new(),
            num_partitions,
            trained: false,
            total_vectors: AtomicUsize::new(0),
            num_subquantizers: pq_subquantizers,
            id_index: RwLock::new(HashSet::new()),
        })
    }

    /// Train the index on a representative sample of vectors.
    ///
    /// Training performs three steps:
    /// 1. **K-means clustering** to find `num_partitions` centroids
    /// 2. **HNSW construction** over the centroids for fast coarse assignment
    /// 3. **PQ training** on the residual vectors (vector - nearest centroid)
    ///
    /// **Note:** If `num_partitions` exceeds the number of training vectors,
    /// an error is returned (you need at least as many training vectors as
    /// partitions for meaningful centroids).
    ///
    /// # Arguments
    /// * `vectors` - Training vectors. Should be a representative sample of the
    ///   dataset (typically 10x-100x the number of partitions).
    /// * `kmeans_iters` - Maximum k-means iterations for centroid computation.
    /// * `pq_iters` - Maximum k-means iterations for PQ codebook training.
    ///
    /// # Errors
    /// Returns an error if training data is empty, dimensions mismatch, or
    /// internal training fails.
    pub fn train(
        &mut self,
        vectors: &[&[f32]],
        kmeans_iters: usize,
        pq_iters: usize,
    ) -> Result<(), IndexError> {
        if vectors.is_empty() {
            return Err(IndexError::Internal("training data is empty".into()));
        }
        for v in vectors {
            if v.len() != self.dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len(),
                });
            }
        }

        // Validate that we have enough training vectors for the requested partitions.
        if vectors.len() < self.num_partitions {
            return Err(IndexError::Internal(format!(
                "not enough training vectors ({}) for {} partitions; \
                 provide at least as many training vectors as partitions",
                vectors.len(),
                self.num_partitions
            )));
        }
        let effective_k = self.num_partitions;

        // Step 1: Run k-means to find partition centroids.
        let km_result = kmeans::kmeans(vectors, effective_k, kmeans_iters, 42)
            .map_err(|e| IndexError::Internal(format!("kmeans failed: {}", e)))?;
        self.centroids = km_result.centroids;
        self.num_partitions = self.centroids.len();

        // Rebuild inverted lists for the (potentially adjusted) partition count.
        self.inverted_lists = (0..self.num_partitions)
            .map(|_| RwLock::new(Vec::new()))
            .collect();

        // Step 2: Build HNSW coarse quantizer from centroids.
        let hnsw_params = HnswParams::new(16, 64, 32, 100_000, 24)
            .map_err(|e| IndexError::Internal(format!("failed to create HnswParams: {}", e)))?;
        self.coarse_quantizer = HnswIndex::new(
            self.dimension,
            vf_core::types::DistanceMetricType::Euclidean,
            hnsw_params,
        );

        for (i, centroid) in self.centroids.iter().enumerate() {
            self.coarse_quantizer
                .add(i as VectorId, centroid)
                .map_err(|e| {
                    IndexError::Internal(format!("failed to add centroid {} to HNSW: {}", i, e))
                })?;
        }

        // Step 3: Compute residuals for all training vectors and train PQ.
        let residuals: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| {
                let partition = self.assign_partition(v);
                compute_residual(v, &self.centroids[partition])
            })
            .collect();

        let residual_refs: Vec<&[f32]> = residuals.iter().map(|r| r.as_slice()).collect();

        let mut pq = self.pq.write();
        *pq = ProductQuantizer::new(self.dimension, self.num_subquantizers).map_err(|e| {
            IndexError::Internal(format!("failed to create ProductQuantizer: {}", e))
        })?;
        pq.train(&residual_refs, pq_iters).map_err(|e| {
            IndexError::Internal(format!("PQ training failed: {}", e))
        })?;
        drop(pq);

        self.trained = true;
        self.total_vectors.store(0, Ordering::Relaxed);
        self.id_index.write().clear();

        Ok(())
    }

    /// Add a single vector to the index.
    ///
    /// The vector is assigned to its nearest partition via the HNSW coarse
    /// quantizer, its residual is PQ-encoded, and the compressed entry is
    /// stored in the corresponding inverted list.
    ///
    /// This method is safe to call concurrently from multiple threads — each
    /// inverted list has its own lock.
    ///
    /// # Errors
    /// Returns an error if the index is not trained, dimensions mismatch,
    /// or the vector ID already exists.
    pub fn add(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        self.check_trained()?;
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        // Check for duplicate ID before insertion.
        {
            let mut ids = self.id_index.write();
            if !ids.insert(id) {
                return Err(IndexError::AlreadyExists(id));
            }
        }

        let partition = self.assign_partition(vector);
        let residual = compute_residual(vector, &self.centroids[partition]);

        let pq = self.pq.read();
        let codes = pq.encode(&residual).map_err(|e| {
            IndexError::Internal(format!("PQ encode failed: {}", e))
        })?;
        drop(pq);

        let entry = CompressedEntry { id, codes };
        self.inverted_lists[partition].write().push(entry);
        self.total_vectors.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Add a batch of vectors in parallel (append-only; no remove capability).
    ///
    /// Each vector is independently assigned and PQ-encoded, leveraging rayon
    /// for parallel processing. Per-partition locking ensures safe concurrent
    /// insertion.
    ///
    /// # Errors
    /// Returns an error if the index is not trained, any vector has a
    /// dimension mismatch, or PQ encoding fails for any vector.
    pub fn add_batch(&self, vectors: &[(VectorId, &[f32])]) -> Result<(), IndexError> {
        self.check_trained()?;
        for &(_, v) in vectors {
            if v.len() != self.dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.len(),
                });
            }
        }

        // Check for duplicates against existing IDs and within the batch.
        {
            let mut ids = self.id_index.write();
            let mut inserted = Vec::new();
            for &(id, _) in vectors {
                if !ids.insert(id) {
                    // Rollback previously inserted IDs
                    for rollback_id in &inserted {
                        ids.remove(rollback_id);
                    }
                    return Err(IndexError::AlreadyExists(id));
                }
                inserted.push(id);
            }
        }

        // Pre-encode all vectors in parallel, collecting errors properly.
        let pq = self.pq.read();
        let encoded: Result<Vec<(VectorId, usize, Vec<u8>)>, IndexError> = vectors
            .par_iter()
            .map(|&(id, vector)| {
                let partition = self.assign_partition(vector);
                let residual = compute_residual(vector, &self.centroids[partition]);
                let codes = pq.encode(&residual).map_err(|e| {
                    IndexError::Internal(format!("PQ encode failed for id {}: {}", id, e))
                })?;
                Ok((id, partition, codes))
            })
            .collect();
        drop(pq);
        let encoded = match encoded {
            Ok(e) => e,
            Err(err) => {
                // Rollback id_index on encoding failure
                let mut ids = self.id_index.write();
                for &(id, _) in vectors {
                    ids.remove(&id);
                }
                return Err(err);
            }
        };

        // Insert into inverted lists (sequentially per-partition for minimal contention).
        for (id, partition, codes) in encoded {
            let entry = CompressedEntry { id, codes };
            self.inverted_lists[partition].write().push(entry);
        }

        self.total_vectors
            .fetch_add(vectors.len(), Ordering::Relaxed);

        Ok(())
    }

    /// Search for the `k` approximate nearest neighbors.
    ///
    /// # Algorithm
    /// 1. Use HNSW coarse quantizer to find the `nprobe` nearest partition centroids
    /// 2. For each probed partition:
    ///    - Compute the query residual w.r.t. that partition's centroid
    ///    - Build a PQ distance table for fast approximate distance computation
    ///    - Scan all PQ codes in the partition using table lookups
    /// 3. Merge results across partitions and return the top-k
    ///
    /// # Arguments
    /// * `query` - Query vector of length `dimension`.
    /// * `k` - Number of nearest neighbors to return.
    /// * `nprobe` - Number of partitions to probe (higher = better recall, slower).
    /// * `ef_search` - Optional override for the HNSW coarse quantizer's ef_search parameter.
    ///
    /// # Errors
    /// Returns an error if the index is not trained or dimensions mismatch.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        self.check_trained()?;
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let nprobe = nprobe.min(self.num_partitions).max(1);

        // Step 1: Find nprobe nearest centroids via HNSW coarse search.
        let centroid_results = VectorIndex::search(&self.coarse_quantizer, query, nprobe, ef_search)?;

        // Step 2+3: For each probed partition, scan PQ codes with distance table.
        let pq = self.pq.read();

        // Max-heap so we can evict the worst (furthest) candidate efficiently.
        let mut top_k: BinaryHeap<(OrderedFloat<f32>, VectorId)> = BinaryHeap::new();

        for centroid_result in &centroid_results {
            let partition_id = centroid_result.id as usize;
            if partition_id >= self.num_partitions {
                continue;
            }

            // Compute query residual for this partition's centroid.
            let query_residual = compute_residual(query, &self.centroids[partition_id]);

            // Build PQ distance table for this residual.
            let dist_table = PqDistanceTable::build_euclidean(&query_residual, &pq)
                .map_err(|e| IndexError::Internal(format!("PQ distance table build failed: {}", e)))?;

            // Scan all entries in this partition.
            let list = self.inverted_lists[partition_id].read();
            for entry in list.iter() {
                let approx_dist = dist_table.distance(&entry.codes);
                let of_dist = OrderedFloat(approx_dist);

                if top_k.len() < k {
                    top_k.push((of_dist, entry.id));
                } else if let Some(&(worst_dist, _)) = top_k.peek() {
                    if of_dist < worst_dist {
                        top_k.pop();
                        top_k.push((of_dist, entry.id));
                    }
                }
            }
        }

        drop(pq);

        // Convert heap to sorted results (ascending distance).
        let mut results: Vec<ScoredResult> = top_k
            .into_iter()
            .map(|(dist, id)| ScoredResult::new(id, dist.into_inner()))
            .collect();
        results.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));

        Ok(results)
    }

    /// Total number of vectors stored in the index.
    pub fn len(&self) -> usize {
        self.total_vectors.load(Ordering::Relaxed)
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimate the compressed memory footprint in bytes.
    ///
    /// Includes:
    /// - PQ codes for all stored vectors (`num_vectors * num_subquantizers`)
    /// - Vector IDs (`num_vectors * 8`)
    /// - Centroid storage (`num_partitions * dimension * 4`)
    /// - PQ codebooks (`num_subquantizers * 256 * subvector_dim * 4`)
    /// - HNSW coarse quantizer graph (estimated)
    pub fn memory_usage_bytes(&self) -> usize {
        let num_vectors = self.len();
        let subvector_dim = self.dimension / self.num_subquantizers;

        // PQ codes: M bytes per vector.
        let pq_codes_bytes = num_vectors * self.num_subquantizers;
        // Vector IDs: 8 bytes each (u64).
        let id_bytes = num_vectors * std::mem::size_of::<VectorId>();
        // Vec overhead per entry (pointer + len + capacity on the codes Vec).
        let entry_overhead = num_vectors * (std::mem::size_of::<CompressedEntry>());

        // Centroids: num_partitions * dimension * 4 bytes.
        let centroid_bytes = self.num_partitions * self.dimension * std::mem::size_of::<f32>();

        // PQ codebooks: M * 256 * subvector_dim * 4 bytes.
        let codebook_bytes =
            self.num_subquantizers * 256 * subvector_dim * std::mem::size_of::<f32>();

        // HNSW coarse graph: rough estimate based on num_partitions.
        // Each node stores a vector (dimension * 4) + adjacency lists.
        // Adjacency: ~M*2 neighbors per node at layer 0, 8 bytes each.
        let hnsw_vector_bytes = self.num_partitions * self.dimension * std::mem::size_of::<f32>();
        let hnsw_adj_bytes = self.num_partitions * 32 * std::mem::size_of::<VectorId>(); // ~32 neighbors avg
        let hnsw_bytes = hnsw_vector_bytes + hnsw_adj_bytes;

        // Inverted list overhead: Vec metadata per partition.
        let list_overhead =
            self.num_partitions * (std::mem::size_of::<RwLock<Vec<CompressedEntry>>>());

        pq_codes_bytes
            + id_bytes
            + entry_overhead
            + centroid_bytes
            + codebook_bytes
            + hnsw_bytes
            + list_overhead
    }

    /// Number of partitions in the index.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Returns the dimensionality of vectors in this index.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of PQ sub-quantizers (compression level).
    pub fn num_subquantizers(&self) -> usize {
        self.num_subquantizers
    }

    /// Returns whether the index has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Returns the sizes of each inverted list (useful for diagnostics).
    pub fn partition_sizes(&self) -> Vec<usize> {
        self.inverted_lists
            .iter()
            .map(|list| list.read().len())
            .collect()
    }

    // ---- Internal helpers ----

    /// Find the nearest partition for a vector using HNSW coarse search.
    fn assign_partition(&self, vector: &[f32]) -> usize {
        // Try HNSW search first (fast).
        if let Ok(results) = VectorIndex::search(&self.coarse_quantizer, vector, 1, None) {
            if let Some(result) = results.first() {
                return result.id as usize;
            }
        }

        // Fallback: brute-force scan of centroids (should only happen if HNSW is empty).
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, centroid) in self.centroids.iter().enumerate() {
            let d = squared_l2_f32(vector, centroid);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }

    fn check_trained(&self) -> Result<(), IndexError> {
        if !self.trained {
            Err(IndexError::Internal(
                "index is not trained: call train() first".into(),
            ))
        } else {
            Ok(())
        }
    }
}

impl VectorIndex for IvfHnswPqIndex {
    fn add(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        self.add(id, vector)
    }

    fn remove(&self, _id: VectorId) -> Result<(), IndexError> {
        Err(IndexError::Internal(
            "IvfHnswPqIndex is append-only and does not support removal".into(),
        ))
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        // Default nprobe: sqrt(num_partitions), clamped to [1, num_partitions].
        let nprobe = ((self.num_partitions as f64).sqrt().ceil() as usize)
            .max(1)
            .min(self.num_partitions);
        self.search(query, k, nprobe, ef_search)
    }

    fn search_with_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        // Search all partitions, then filter by candidate set.
        let results = VectorIndex::search(self, query, k.max(candidates.len()), ef_search)?;
        let candidate_set: std::collections::HashSet<VectorId> =
            candidates.iter().copied().collect();
        let filtered: Vec<ScoredResult> = results
            .into_iter()
            .filter(|r| candidate_set.contains(&r.id))
            .take(k)
            .collect();
        Ok(filtered)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn dimension(&self) -> usize {
        self.dimension()
    }

    fn contains(&self, id: VectorId) -> bool {
        // O(1) lookup via the ID index set.
        self.id_index.read().contains(&id)
    }
}

/// Compute the residual vector (vector - centroid).
#[inline]
fn compute_residual(vector: &[f32], centroid: &[f32]) -> Vec<f32> {
    vector
        .iter()
        .zip(centroid.iter())
        .map(|(&v, &c)| v - c)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let index = IvfHnswPqIndex::new(128, 4, 8).unwrap();
        assert_eq!(index.dimension(), 128);
        assert_eq!(index.num_partitions(), 4);
        assert_eq!(index.num_subquantizers(), 8);
        assert!(!index.is_trained());
        assert!(index.is_empty());
    }

    #[test]
    fn test_new_invalid_params() {
        assert!(IvfHnswPqIndex::new(0, 4, 8).is_err());
        assert!(IvfHnswPqIndex::new(128, 0, 8).is_err());
        assert!(IvfHnswPqIndex::new(128, 4, 0).is_err());
        assert!(IvfHnswPqIndex::new(127, 4, 8).is_err()); // not divisible
        assert!(IvfHnswPqIndex::new(128, 512, 8).is_ok()); // >256 partitions now supported
    }

    #[test]
    fn test_not_trained_errors() {
        let index = IvfHnswPqIndex::new(8, 2, 2).unwrap();
        let v = vec![1.0; 8];
        assert!(index.add(0, &v).is_err());
        assert!(index.search(&v, 1, 1, None).is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = IvfHnswPqIndex::new(8, 2, 2).unwrap();
        let training: Vec<Vec<f32>> = (0..300)
            .map(|i| (0..8).map(|d| (i * 8 + d) as f32 * 0.01).collect())
            .collect();
        let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        index.train(&refs, 20, 20).unwrap();

        let bad = vec![1.0; 4]; // wrong dimension
        assert!(index.add(0, &bad).is_err());
        assert!(index.search(&bad, 1, 1, None).is_err());
    }
}
