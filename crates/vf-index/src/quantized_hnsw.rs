// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Quantized HNSW index — wraps [`HnswIndex`] with quantized vector storage
//! for memory-efficient approximate nearest neighbor search with optional
//! re-ranking using full-precision vectors.

use std::collections::HashMap;
use std::mem;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;

use vf_core::distance::DistanceMetric;
use vf_core::types::{DistanceMetricType, ScoredResult, VectorId};
use vf_quantization::product::ProductQuantizer;
use vf_quantization::pq_distance::PqDistanceTable;
use vf_quantization::scalar::ScalarQuantizer;
use vf_quantization::sq_distance::{sq_cosine_distance, sq_dot_product_distance, sq_euclidean_distance};

use crate::hnsw::{HnswIndex, HnswParams};
use crate::traits::{IndexError, VectorIndex};

/// The type of quantization codec used for compact vector storage.
pub enum QuantizationType {
    /// Scalar (uniform) quantization: 1 byte per dimension.
    Scalar(ScalarQuantizer),
    /// Product quantization: M bytes per vector (M = num_subquantizers).
    Product(ProductQuantizer),
    /// No quantization — full f32 vectors only (wrapper pass-through).
    None,
}

/// Memory usage statistics for the quantized HNSW index.
#[derive(Debug, Clone)]
pub struct QuantizedMemoryStats {
    /// Bytes used by quantized codes (compact storage).
    pub quantized_codes_bytes: usize,
    /// Bytes used by full-precision vectors (for re-ranking).
    pub full_vectors_bytes: usize,
    /// Bytes used by the HNSW graph structure (nodes + adjacency lists).
    pub hnsw_graph_bytes: usize,
    /// Total number of vectors stored.
    pub num_vectors: usize,
    /// Average bytes per vector for quantized codes.
    pub avg_quantized_bytes_per_vector: f64,
    /// Average bytes per vector for full-precision storage.
    pub avg_full_bytes_per_vector: f64,
    /// Compression ratio (full / quantized), higher is better.
    pub compression_ratio: f64,
}

/// A quantized HNSW index that stores quantized codes alongside full vectors.
///
/// The HNSW graph is built and traversed using full-precision vectors (for
/// accurate graph construction), while quantized codes provide compact storage
/// for distance estimation. At search time, the index can:
///
/// 1. **Re-ranked search** (`search`): retrieve `rerank_factor * k` candidates
///    via HNSW graph traversal, then re-rank them using full-precision distances
///    to return the top `k` results.
///
/// 2. **Approximate search** (`search_approximate`): use quantized distances
///    for the final ranking without re-ranking — faster but less accurate.
pub struct QuantizedHnswIndex {
    /// The base HNSW graph (stores full vectors for graph construction).
    hnsw: HnswIndex,
    /// Quantization codec.
    quantization: QuantizationType,
    /// Quantized codes for each vector (compact storage).
    quantized_codes: RwLock<HashMap<VectorId, Vec<u8>>>,
    /// Duplicates HNSW's internal vectors for re-ranking. HnswIndex wraps
    /// nodes behind RwLock and does not expose a "get vector by id" method,
    /// so we keep a separate copy here. Future optimization: expose vector
    /// retrieval from HnswIndex to avoid this duplication.
    full_vectors: RwLock<HashMap<VectorId, Vec<f32>>>,
    /// Re-ranking factor: search for rerank_k candidates, re-rank to get k.
    rerank_factor: usize,
    /// Distance metric type (needed for choosing SQ/PQ distance functions).
    metric: DistanceMetricType,
    /// Vector dimension.
    dimension: usize,
    /// Cached distance function — enum dispatch avoids vtable overhead.
    distance_fn: DistanceMetric,
}

impl QuantizedHnswIndex {
    /// Create a new quantized HNSW index.
    ///
    /// # Arguments
    /// * `dimension` — vector dimensionality.
    /// * `params` — HNSW graph parameters.
    /// * `metric` — distance metric type.
    /// * `quantization` — quantization codec (must be already trained).
    /// * `rerank_factor` — search for `rerank_factor * k` candidates before
    ///   re-ranking with full vectors. A value of 1 means no over-retrieval.
    ///   Typical values: 2–5. Use 3 as a good default.
    pub fn new(
        dimension: usize,
        params: HnswParams,
        metric: DistanceMetricType,
        quantization: QuantizationType,
        rerank_factor: usize,
    ) -> Self {
        let rerank_factor = rerank_factor.max(1);
        let distance_fn = DistanceMetric::from_metric_type(metric);
        Self {
            hnsw: HnswIndex::new(dimension, metric, params),
            quantization,
            quantized_codes: RwLock::new(HashMap::new()),
            full_vectors: RwLock::new(HashMap::new()),
            rerank_factor,
            metric,
            dimension,
            distance_fn,
        }
    }

    /// Add a vector to the quantized HNSW index.
    ///
    /// This inserts the vector into the HNSW graph (for accurate construction),
    /// stores the quantized code (for compact storage), and keeps the full
    /// vector (for re-ranking).
    pub fn add(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        // Insert into HNSW graph first (validates dimension, checks duplicates).
        self.hnsw.add(id, vector)?;

        // Compute and store the quantized code.
        let code = self.quantize_vector(vector)?;
        self.quantized_codes.write().insert(id, code);

        // Store the full vector for re-ranking.
        self.full_vectors.write().insert(id, vector.to_vec());

        Ok(())
    }

    /// Remove a vector from the quantized HNSW index.
    pub fn remove(&self, id: VectorId) -> Result<(), IndexError> {
        self.hnsw.remove(id)?;
        self.quantized_codes.write().remove(&id);
        self.full_vectors.write().remove(&id);
        Ok(())
    }

    /// Search with over-retrieval and re-ranking.
    ///
    /// Retrieves `rerank_factor * k` candidates from the HNSW graph (which
    /// already uses full-precision vectors for traversal), then re-ranks them
    /// using exact full-precision distances. The primary benefit of
    /// over-retrieval is that HNSW's greedy traversal is approximate — it may
    /// miss true nearest neighbors. By fetching more candidates and
    /// re-computing exact distances, we confirm the graph's approximate
    /// ordering and recover neighbors that the greedy search ranked slightly
    /// too low.
    pub fn search(&self, query: &[f32], k: usize, ef_search: Option<usize>) -> Result<Vec<ScoredResult>, IndexError> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        // Over-retrieve candidates from HNSW graph.
        let expanded_k = k.saturating_mul(self.rerank_factor).max(k);
        let candidates = self.hnsw.search(query, expanded_k, ef_search)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Re-rank candidates using full-precision vectors.
        let full_vecs = self.full_vectors.read();

        let mut reranked: Vec<ScoredResult> = candidates
            .into_iter()
            .map(|candidate| {
                if let Some(full_vec) = full_vecs.get(&candidate.id) {
                    let exact_dist = self.distance_fn.compute(query, full_vec);
                    ScoredResult::new(candidate.id, exact_dist)
                } else {
                    // Fallback: keep the HNSW score if full vector is missing.
                    candidate
                }
            })
            .collect();

        reranked.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        reranked.truncate(k);

        Ok(reranked)
    }

    /// Fast approximate search using HNSW traversal + quantized re-scoring
    /// (no full-precision re-ranking).
    ///
    /// Retrieves candidates from the HNSW graph and re-scores them using
    /// quantized distance functions. Faster than `search()` but less accurate
    /// because quantized distances are lossy approximations.
    ///
    /// **Note on Manhattan metric:** Neither SQ nor PQ natively support
    /// Manhattan distance. SQ falls back to the raw HNSW score, and PQ falls
    /// back to a Euclidean distance table. Results under Manhattan will be
    /// approximate at best.
    pub fn search_approximate(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<ScoredResult>, IndexError> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        // Retrieve candidates from HNSW graph.
        let candidates = self.hnsw.search(query, k, ef_search)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Re-score using quantized distances.
        let codes_map = self.quantized_codes.read();

        let mut scored: Vec<ScoredResult> = match &self.quantization {
            QuantizationType::Scalar(sq) => {
                let query_code = sq.quantize(query).map_err(|e| {
                    IndexError::Internal(format!("scalar quantize query failed: {}", e))
                })?;
                candidates
                    .into_iter()
                    .map(|c| {
                        if let Some(code) = codes_map.get(&c.id) {
                            let dist = match self.metric {
                                DistanceMetricType::Euclidean => {
                                    sq_euclidean_distance(&query_code, code, sq)
                                }
                                DistanceMetricType::DotProduct => {
                                    sq_dot_product_distance(&query_code, code, sq)
                                }
                                DistanceMetricType::Cosine => {
                                    sq_cosine_distance(&query_code, code, sq)
                                }
                                DistanceMetricType::Manhattan => {
                                    // WARNING: SQ does not support Manhattan distance.
                                    // Falling back to the HNSW score (full-precision Manhattan).
                                    // This means no quantized re-scoring occurs for Manhattan+SQ.
                                    c.score
                                }
                            };
                            ScoredResult::new(c.id, dist)
                        } else {
                            c
                        }
                    })
                    .collect()
            }
            QuantizationType::Product(pq) => {
                let table = match self.metric {
                    DistanceMetricType::Euclidean | DistanceMetricType::Cosine => {
                        PqDistanceTable::build_euclidean(query, pq)
                    }
                    DistanceMetricType::DotProduct => {
                        PqDistanceTable::build_dot_product(query, pq)
                    }
                    DistanceMetricType::Manhattan => {
                        // WARNING: PQ does not support Manhattan distance natively.
                        // Falling back to Euclidean distance table — results will
                        // approximate Manhattan ordering but are not exact.
                        PqDistanceTable::build_euclidean(query, pq)
                    }
                };
                candidates
                    .into_iter()
                    .map(|c| {
                        if let Some(code) = codes_map.get(&c.id) {
                            let dist = table.distance(code);
                            ScoredResult::new(c.id, dist)
                        } else {
                            c
                        }
                    })
                    .collect()
            }
            QuantizationType::None => {
                // No quantization — just return HNSW results as-is.
                candidates
            }
        };

        scored.sort_by(|a, b| OrderedFloat(a.score).cmp(&OrderedFloat(b.score)));
        scored.truncate(k);

        Ok(scored)
    }

    /// Returns the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.hnsw.len() == 0
    }

    /// Report memory usage statistics: quantized codes vs full vectors.
    ///
    /// **Note:** These are estimates. They account for the raw byte capacity of
    /// `Vec<u8>` / `Vec<f32>` values but exclude HashMap overhead (bucket
    /// arrays, load factor slack) and HNSW adjacency lists (neighbor vectors
    /// at each layer). Actual RSS will be higher.
    pub fn memory_usage(&self) -> QuantizedMemoryStats {
        let codes = self.quantized_codes.read();
        let vecs = self.full_vectors.read();
        let num_vectors = codes.len();

        // Quantized codes memory: sum of all code Vec<u8> allocations.
        let quantized_codes_bytes: usize = codes
            .values()
            .map(|c| c.capacity() * mem::size_of::<u8>())
            .sum();

        // Full vectors memory: sum of all Vec<f32> allocations.
        let full_vectors_bytes: usize = vecs
            .values()
            .map(|v| v.capacity() * mem::size_of::<f32>())
            .sum();

        // HNSW graph overhead estimate: nodes + adjacency lists.
        // Each node stores a Vec<f32> (vector) + Vec<Vec<VectorId>> (neighbors).
        // We approximate the graph overhead as the vector storage portion.
        let hnsw_graph_bytes = num_vectors * self.dimension * mem::size_of::<f32>();

        let avg_quantized = if num_vectors > 0 {
            quantized_codes_bytes as f64 / num_vectors as f64
        } else {
            0.0
        };

        let avg_full = if num_vectors > 0 {
            full_vectors_bytes as f64 / num_vectors as f64
        } else {
            0.0
        };

        let compression_ratio = if quantized_codes_bytes > 0 {
            full_vectors_bytes as f64 / quantized_codes_bytes as f64
        } else {
            0.0
        };

        QuantizedMemoryStats {
            quantized_codes_bytes,
            full_vectors_bytes,
            hnsw_graph_bytes,
            num_vectors,
            avg_quantized_bytes_per_vector: avg_quantized,
            avg_full_bytes_per_vector: avg_full,
            compression_ratio,
        }
    }

    /// Returns a reference to the underlying HNSW index.
    pub fn hnsw(&self) -> &HnswIndex {
        &self.hnsw
    }

    /// Returns the re-ranking factor.
    pub fn rerank_factor(&self) -> usize {
        self.rerank_factor
    }

    /// Returns the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns true if the given vector ID exists in the index.
    pub fn contains(&self, id: VectorId) -> bool {
        self.hnsw.contains(id)
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Quantize a vector using the configured codec.
    fn quantize_vector(&self, vector: &[f32]) -> Result<Vec<u8>, IndexError> {
        match &self.quantization {
            QuantizationType::Scalar(sq) => sq.quantize(vector).map_err(|e| {
                IndexError::Internal(format!("scalar quantization failed: {}", e))
            }),
            QuantizationType::Product(pq) => pq.encode(vector).map_err(|e| {
                IndexError::Internal(format!("product quantization failed: {}", e))
            }),
            QuantizationType::None => {
                // No quantization: store raw f32 bytes as u8 for uniformity.
                Ok(vector
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect())
            }
        }
    }
}
