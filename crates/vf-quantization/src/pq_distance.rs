// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Asymmetric distance computation for Product Quantization.
//!
//! Instead of decoding every database vector, we precompute a distance lookup
//! table from the query to every centroid. Then the approximate distance to any
//! PQ-encoded vector is just `M` table lookups + additions.

use crate::product::ProductQuantizer;

/// Precomputed per-query distance table for fast asymmetric distance computation.
///
/// For each sub-quantizer `m` and centroid index `k`, `tables[m][k]` holds the
/// partial distance between `query_m` and `centroid_{m,k}`. The total distance
/// to a PQ-encoded vector is `sum_m tables[m][codes[m]]`.
pub struct PqDistanceTable {
    /// M tables, each with K=256 entries.
    tables: Vec<Vec<f32>>,
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
            tables.push(table);
        }

        Self { tables }
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
            tables.push(table);
        }

        Self { tables }
    }

    /// Compute the approximate distance to a single PQ-encoded vector.
    ///
    /// This is the sum of `M` table lookups — extremely fast (no floating-point
    /// arithmetic beyond addition).
    #[inline]
    pub fn distance(&self, codes: &[u8]) -> f32 {
        debug_assert_eq!(
            codes.len(),
            self.tables.len(),
            "codes length must equal number of sub-quantizers"
        );

        let mut total = 0.0f32;
        for (table, &code) in self.tables.iter().zip(codes.iter()) {
            // SAFETY: code is u8 so always < 256, and tables have 256 entries.
            total += table[code as usize];
        }
        total
    }

    /// Compute distances to a batch of PQ-encoded vectors.
    ///
    /// Returns one distance per code vector, in the same order.
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
}
