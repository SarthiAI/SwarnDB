// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Asymmetric distance computation for Product Quantization.
//!
//! Instead of decoding every database vector, we precompute a distance lookup
//! table from the query to every centroid. Then the approximate distance to any
//! PQ-encoded vector is just `M` table lookups + additions.
//!
//! On platforms with AVX2 support, the distance computation uses
//! `_mm256_i32gather_ps` to fetch 8 table entries at once, yielding a
//! significant speedup for typical M values (8, 16, 32, 64).

use std::sync::OnceLock;

use crate::product::ProductQuantizer;

// ---------------------------------------------------------------------------
// Runtime SIMD dispatch for PQ distance
// ---------------------------------------------------------------------------

/// Function pointer type for SIMD-dispatched PQ distance kernel.
/// Arguments: (flat_tables pointer, codes, num_subquantizers) -> distance
type PqDistanceKernelFn = fn(&[f32], &[u8], usize) -> f32;

/// Runtime dispatcher that selects the best PQ distance kernel at startup.
struct PqDistanceDispatcher {
    kernel: PqDistanceKernelFn,
}

unsafe impl Send for PqDistanceDispatcher {}
unsafe impl Sync for PqDistanceDispatcher {}

impl PqDistanceDispatcher {
    fn detect() -> Self {
        PqDistanceDispatcher {
            kernel: Self::select_kernel(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn select_kernel() -> PqDistanceKernelFn {
        if is_x86_feature_detected!("avx2") {
            |flat_tables, codes, m| unsafe { avx2_pq_distance(flat_tables, codes, m) }
        } else {
            scalar_pq_distance
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn select_kernel() -> PqDistanceKernelFn {
        // NEON has no direct gather instruction; use scalar path.
        scalar_pq_distance
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn select_kernel() -> PqDistanceKernelFn {
        scalar_pq_distance
    }
}

static PQ_DISPATCHER: OnceLock<PqDistanceDispatcher> = OnceLock::new();

/// Get the global PQ distance dispatcher (initialized on first call).
#[inline]
fn get_pq_dispatcher() -> &'static PqDistanceDispatcher {
    PQ_DISPATCHER.get_or_init(PqDistanceDispatcher::detect)
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar PQ distance: sum of M table lookups.
fn scalar_pq_distance(flat_tables: &[f32], codes: &[u8], m: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..m {
        // Each sub-quantizer table starts at offset i * 256.
        let idx = i * 256 + codes[i] as usize;
        total += flat_tables[idx];
    }
    total
}

// ---------------------------------------------------------------------------
// AVX2 gather kernel
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_pq_distance(flat_tables: &[f32], codes: &[u8], m: usize) -> f32 {
    use std::arch::x86_64::*;

    let chunks = m / 8;
    let base_ptr = flat_tables.as_ptr();

    let mut sum = _mm256_setzero_ps();

    for chunk in 0..chunks {
        let offset = chunk * 8;

        // Build 8 gather indices: index[j] = (offset + j) * 256 + codes[offset + j]
        let indices = _mm256_set_epi32(
            ((offset + 7) * 256 + codes[offset + 7] as usize) as i32,
            ((offset + 6) * 256 + codes[offset + 6] as usize) as i32,
            ((offset + 5) * 256 + codes[offset + 5] as usize) as i32,
            ((offset + 4) * 256 + codes[offset + 4] as usize) as i32,
            ((offset + 3) * 256 + codes[offset + 3] as usize) as i32,
            ((offset + 2) * 256 + codes[offset + 2] as usize) as i32,
            ((offset + 1) * 256 + codes[offset + 1] as usize) as i32,
            ((offset + 0) * 256 + codes[offset + 0] as usize) as i32,
        );

        // Gather 8 f32 values from the flat table using the computed indices.
        // Scale = 4 because each element is 4 bytes (f32).
        let gathered = _mm256_i32gather_ps::<4>(base_ptr, indices);

        sum = _mm256_add_ps(sum, gathered);
    }

    // Horizontal sum of the 256-bit accumulator.
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    // Scalar tail for remaining subquantizers (m % 8).
    let tail_start = chunks * 8;
    for i in tail_start..m {
        let idx = i * 256 + codes[i] as usize;
        total += *base_ptr.add(idx);
    }

    total
}

// ---------------------------------------------------------------------------
// PqDistanceTable
// ---------------------------------------------------------------------------

/// Precomputed per-query distance table for fast asymmetric distance computation.
///
/// For each sub-quantizer `m` and centroid index `k`, `tables[m][k]` holds the
/// partial distance between `query_m` and `centroid_{m,k}`. The total distance
/// to a PQ-encoded vector is `sum_m tables[m][codes[m]]`.
///
/// Internally maintains a flat contiguous buffer (`flat_tables`) of size M*256
/// for SIMD gather-friendly access on AVX2 platforms.
pub struct PqDistanceTable {
    /// M tables, each with K=256 entries (kept for compatibility).
    tables: Vec<Vec<f32>>,
    /// Flat contiguous buffer: `flat_tables[m * 256 + k] == tables[m][k]`.
    /// Layout is gather-friendly: all 256 entries for sub-quantizer 0 first,
    /// then all 256 for sub-quantizer 1, etc.
    flat_tables: Vec<f32>,
}

impl PqDistanceTable {
    /// Build a distance table using **squared Euclidean distance**.
    ///
    /// For each sub-quantizer `m` and centroid `k`:
    /// `tables[m][k] = ||query_m - centroid_{m,k}||^2`
    ///
    /// The total distance for a PQ code is the sum of the per-subquantizer
    /// squared distances, which equals the squared L2 distance between the
    /// query and the reconstructed vector.
    ///
    /// # Panics
    /// Panics if the quantizer is not trained or if the query length does not
    /// match the quantizer's dimension.
    pub fn build_euclidean(query: &[f32], pq: &ProductQuantizer) -> Self {
        assert!(
            pq.is_trained(),
            "PqDistanceTable: quantizer must be trained"
        );
        assert_eq!(
            query.len(),
            pq.dimension(),
            "PqDistanceTable: query dimension mismatch"
        );

        let m = pq.num_subquantizers();
        let sub_dim = pq.subvector_dim();
        let k = pq.num_centroids();
        let codebooks = pq.codebooks();

        let mut tables = Vec::with_capacity(m);
        let mut flat_tables = Vec::with_capacity(m * 256);

        for mi in 0..m {
            let q_start = mi * sub_dim;
            let q_sub = &query[q_start..q_start + sub_dim];
            let codebook = &codebooks[mi];

            let mut table = Vec::with_capacity(k);
            for ki in 0..k {
                let centroid = &codebook[ki];
                let dist: f32 = q_sub
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| {
                        let d = a - b;
                        d * d
                    })
                    .sum();
                table.push(dist);
            }
            // Pad to exactly 256 entries if k < 256 (shouldn't happen, but safe).
            flat_tables.extend_from_slice(&table);
            for _ in table.len()..256 {
                flat_tables.push(0.0);
            }
            tables.push(table);
        }

        Self {
            tables,
            flat_tables,
        }
    }

    /// Build a distance table using **negative dot product** (for maximum inner
    /// product search).
    ///
    /// For each sub-quantizer `m` and centroid `k`:
    /// `tables[m][k] = -dot(query_m, centroid_{m,k})`
    ///
    /// We negate so that _lower_ values correspond to _higher_ inner products,
    /// making it compatible with nearest-neighbor search that minimises distance.
    ///
    /// # Panics
    /// Panics if the quantizer is not trained or if the query length does not
    /// match the quantizer's dimension.
    pub fn build_dot_product(query: &[f32], pq: &ProductQuantizer) -> Self {
        assert!(
            pq.is_trained(),
            "PqDistanceTable: quantizer must be trained"
        );
        assert_eq!(
            query.len(),
            pq.dimension(),
            "PqDistanceTable: query dimension mismatch"
        );

        let m = pq.num_subquantizers();
        let sub_dim = pq.subvector_dim();
        let k = pq.num_centroids();
        let codebooks = pq.codebooks();

        let mut tables = Vec::with_capacity(m);
        let mut flat_tables = Vec::with_capacity(m * 256);

        for mi in 0..m {
            let q_start = mi * sub_dim;
            let q_sub = &query[q_start..q_start + sub_dim];
            let codebook = &codebooks[mi];

            let mut table = Vec::with_capacity(k);
            for ki in 0..k {
                let centroid = &codebook[ki];
                let dot: f32 = q_sub
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                table.push(-dot);
            }
            // Pad to exactly 256 entries.
            flat_tables.extend_from_slice(&table);
            for _ in table.len()..256 {
                flat_tables.push(0.0);
            }
            tables.push(table);
        }

        Self {
            tables,
            flat_tables,
        }
    }

    /// Compute the approximate distance to a single PQ-encoded vector.
    ///
    /// Uses SIMD gather (AVX2 `_mm256_i32gather_ps`) to fetch 8 table entries
    /// at once when available, with automatic scalar fallback on other platforms.
    /// This is the sum of `M` table lookups — extremely fast (no floating-point
    /// arithmetic beyond addition).
    #[inline]
    pub fn distance(&self, codes: &[u8]) -> f32 {
        debug_assert_eq!(
            codes.len(),
            self.tables.len(),
            "codes length must equal number of sub-quantizers"
        );

        let m = self.tables.len();
        (get_pq_dispatcher().kernel)(&self.flat_tables, codes, m)
    }

    /// Compute distances to a batch of PQ-encoded vectors.
    ///
    /// Returns one distance per code vector, in the same order.
    /// Each individual distance computation benefits from SIMD gather acceleration.
    pub fn distance_batch(&self, codes_batch: &[&[u8]]) -> Vec<f32> {
        codes_batch
            .iter()
            .map(|codes| self.distance(codes))
            .collect()
    }

    /// Number of sub-quantizers (M) this table was built for.
    pub fn num_subquantizers(&self) -> usize {
        self.tables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::product::ProductQuantizer;

    /// Helper: train a small PQ and return it.
    fn trained_pq() -> ProductQuantizer {
        let dim = 8;
        let m = 2;
        let mut pq = ProductQuantizer::new(dim, m).unwrap();

        let training: Vec<Vec<f32>> = (0..300)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.01).collect())
            .collect();
        let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 20).unwrap();
        pq
    }

    #[test]
    fn test_euclidean_table_distance_nonnegative() {
        let pq = trained_pq();
        let query: Vec<f32> = (0..8).map(|d| d as f32 * 0.1).collect();
        let table = PqDistanceTable::build_euclidean(&query, &pq);

        let codes = pq.encode(&query).unwrap();
        let dist = table.distance(&codes);
        assert!(dist >= 0.0, "squared Euclidean distance must be >= 0");
    }

    #[test]
    fn test_dot_product_table() {
        let pq = trained_pq();
        let query: Vec<f32> = (0..8).map(|d| d as f32 * 0.1).collect();
        let table = PqDistanceTable::build_dot_product(&query, &pq);

        let codes = pq.encode(&query).unwrap();
        let _dist = table.distance(&codes);
        // Just ensure it doesn't panic and returns a finite value.
        assert!(_dist.is_finite());
    }

    #[test]
    fn test_distance_batch() {
        let pq = trained_pq();
        let query: Vec<f32> = (0..8).map(|d| d as f32 * 0.1).collect();
        let table = PqDistanceTable::build_euclidean(&query, &pq);

        let v1: Vec<f32> = vec![0.0; 8];
        let v2: Vec<f32> = vec![1.0; 8];
        let c1 = pq.encode(&v1).unwrap();
        let c2 = pq.encode(&v2).unwrap();

        let batch: Vec<&[u8]> = vec![c1.as_slice(), c2.as_slice()];
        let dists = table.distance_batch(&batch);
        assert_eq!(dists.len(), 2);
    }

    #[test]
    fn test_flat_tables_layout() {
        let pq = trained_pq();
        let query: Vec<f32> = (0..8).map(|d| d as f32 * 0.1).collect();
        let table = PqDistanceTable::build_euclidean(&query, &pq);

        // Verify flat_tables matches tables for each subquantizer.
        for (mi, t) in table.tables.iter().enumerate() {
            for (ki, &val) in t.iter().enumerate() {
                let flat_idx = mi * 256 + ki;
                assert_eq!(
                    table.flat_tables[flat_idx], val,
                    "flat_tables mismatch at m={}, k={}",
                    mi, ki
                );
            }
        }
    }

    #[test]
    fn test_simd_matches_scalar() {
        let pq = trained_pq();
        let query: Vec<f32> = (0..8).map(|d| d as f32 * 0.1).collect();
        let table = PqDistanceTable::build_euclidean(&query, &pq);

        let codes = pq.encode(&query).unwrap();

        // Compute via the dispatched path (may be SIMD).
        let simd_dist = table.distance(&codes);

        // Compute via explicit scalar path.
        let scalar_dist =
            scalar_pq_distance(&table.flat_tables, &codes, table.tables.len());

        assert!(
            (simd_dist - scalar_dist).abs() < 1e-6,
            "SIMD ({}) and scalar ({}) results must match",
            simd_dist,
            scalar_dist
        );
    }
}
