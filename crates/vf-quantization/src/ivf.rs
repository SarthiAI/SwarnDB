// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Inverted File Index (IVF) for coarse-grained partitioning of vector space.
//!
//! Vectors are assigned to the nearest Voronoi cell (partition) defined by
//! k-means centroids. At search time only the `nprobe` closest partitions
//! are scanned, trading recall for speed.

use serde::{Deserialize, Serialize};
use vf_core::simd::squared_l2_f32;

use crate::error::QuantizationError;
use crate::kmeans;

/// Entry stored in an inverted list (one per partition).
///
/// This is an IVF-flat entry: only the full-precision vector is stored.
/// Residuals (vector minus centroid) are **not** stored here because IVF-flat
/// computes exact distances against the original vectors. Residual storage
/// would be used in an IVF-PQ variant where product-quantized residuals
/// replace the full vectors — that is a separate implementation.
#[derive(Clone, Serialize, Deserialize)]
pub struct InvertedListEntry {
    /// Unique vector identifier.
    pub id: u64,
    /// Full-precision vector.
    pub vector: Vec<f32>,
}

/// Inverted File Index partitioning vectors into Voronoi cells.
#[derive(Serialize, Deserialize)]
pub struct IvfIndex {
    dimension: usize,
    num_partitions: usize,
    centroids: Vec<Vec<f32>>,
    inverted_lists: Vec<Vec<InvertedListEntry>>,
    trained: bool,
}

impl IvfIndex {
    /// Create a new untrained IVF index.
    ///
    /// # Arguments
    /// * `dimension` — dimensionality of vectors
    /// * `num_partitions` — number of Voronoi cells (nlist). Must be >= 1.
    pub fn new(dimension: usize, num_partitions: usize) -> Result<Self, QuantizationError> {
        if num_partitions == 0 {
            return Err(QuantizationError::InvalidParameter(format!(
                "num_partitions must be >= 1, got {}",
                num_partitions
            )));
        }
        Ok(Self {
            dimension,
            num_partitions,
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            trained: false,
        })
    }

    /// Train the coarse quantizer by running k-means on the provided vectors.
    ///
    /// After training, vectors can be added and searched.
    pub fn train(
        &mut self,
        vectors: &[&[f32]],
        max_iters: usize,
    ) -> Result<(), QuantizationError> {
        if vectors.is_empty() {
            return Err(QuantizationError::EmptyTrainingData);
        }
        for v in vectors {
            if v.len() != self.dimension {
                return Err(QuantizationError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
        }
        let result = kmeans::kmeans(vectors, self.num_partitions, max_iters, 42)?;

        self.centroids = result.centroids;
        self.inverted_lists = vec![Vec::new(); self.num_partitions];
        self.trained = true;

        Ok(())
    }

    /// Add a single vector to the index. The vector is assigned to its nearest
    /// centroid partition.
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), QuantizationError> {
        self.check_trained()?;
        self.check_dimension(vector.len())?;

        let partition = self.assign_partition(vector);

        self.inverted_lists[partition].push(InvertedListEntry {
            id,
            vector: vector.to_vec(),
        });

        Ok(())
    }

    /// Add a batch of vectors to the index.
    pub fn add_batch(
        &mut self,
        vectors: &[(u64, &[f32])],
    ) -> Result<(), QuantizationError> {
        self.check_trained()?;
        for &(_, v) in vectors {
            self.check_dimension(v.len())?;
        }

        for &(id, vector) in vectors {
            let partition = self.assign_partition(vector);

            self.inverted_lists[partition].push(InvertedListEntry {
                id,
                vector: vector.to_vec(),
            });
        }

        Ok(())
    }

    /// Search the index for the `k` nearest vectors to `query`, probing the
    /// `nprobe` closest partitions.
    ///
    /// Returns a list of `(id, distance)` pairs sorted by ascending L2 distance.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<(u64, f32)>, QuantizationError> {
        self.check_trained()?;
        self.check_dimension(query.len())?;

        // Clamp nprobe: at least 1, at most num_partitions.
        let nprobe = nprobe.max(1).min(self.num_partitions);

        let mut centroid_dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, squared_l2_f32(query, c)))
            .collect();

        // Partial sort: we only need the top nprobe closest.
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Scan the nprobe closest partitions and collect candidates.
        let mut candidates: Vec<(u64, f32)> = Vec::new();

        for &(partition_idx, _) in centroid_dists.iter().take(nprobe) {
            for entry in &self.inverted_lists[partition_idx] {
                let dist = squared_l2_f32(query, &entry.vector);
                candidates.push((entry.id, dist));
            }
        }

        // Sort candidates by distance and return top-k.
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        Ok(candidates)
    }

    /// Total number of vectors stored across all inverted lists.
    pub fn len(&self) -> usize {
        self.inverted_lists.iter().map(|list| list.len()).sum()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of partitions (Voronoi cells) in the index.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Returns the size of each inverted list (useful for diagnostics).
    pub fn partition_sizes(&self) -> Vec<usize> {
        self.inverted_lists.iter().map(|list| list.len()).collect()
    }

    /// Find the index of the nearest centroid for the given vector.
    pub fn assign_partition(&self, vector: &[f32]) -> usize {
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

    fn check_trained(&self) -> Result<(), QuantizationError> {
        if !self.trained {
            Err(QuantizationError::NotTrained)
        } else {
            Ok(())
        }
    }

    fn check_dimension(&self, got: usize) -> Result<(), QuantizationError> {
        if got != self.dimension {
            Err(QuantizationError::DimensionMismatch {
                expected: self.dimension,
                got,
            })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_new() {
        let index = IvfIndex::new(128, 4).unwrap();
        assert_eq!(index.num_partitions(), 4);
        assert!(index.is_empty());
    }

    #[test]
    fn test_ivf_not_trained_errors() {
        let index = IvfIndex::new(4, 2).unwrap();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!(index.search(&v, 1, 1).is_err());
    }

    #[test]
    fn test_ivf_train_add_search() {
        let mut index = IvfIndex::new(2, 2).unwrap();

        // Two clusters: around (0,0) and (10,10).
        let cluster_a: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![0.0 + (i as f32) * 0.01, 0.0])
            .collect();
        let cluster_b: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![10.0 + (i as f32) * 0.01, 10.0])
            .collect();

        let training: Vec<&[f32]> = cluster_a
            .iter()
            .chain(cluster_b.iter())
            .map(|v| v.as_slice())
            .collect();

        index.train(&training, 50).unwrap();

        // Add vectors.
        for (i, v) in cluster_a.iter().enumerate() {
            index.add(i as u64, v).unwrap();
        }
        for (i, v) in cluster_b.iter().enumerate() {
            index.add((100 + i) as u64, v).unwrap();
        }

        assert_eq!(index.len(), 40);

        // Search near cluster_a with nprobe=1, should return cluster_a vectors.
        let results = index.search(&[0.05, 0.0], 5, 1).unwrap();
        assert_eq!(results.len(), 5);
        for (id, _dist) in &results {
            assert!(*id < 100, "expected cluster_a ids, got {}", id);
        }
    }

    #[test]
    fn test_ivf_add_batch() {
        let mut index = IvfIndex::new(2, 2).unwrap();

        let vecs: Vec<Vec<f32>> = (0..30)
            .map(|i| vec![(i as f32) * 0.5, (i as f32) * 0.5])
            .collect();
        let training: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        index.train(&training, 20).unwrap();

        let batch: Vec<(u64, &[f32])> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, v.as_slice()))
            .collect();
        index.add_batch(&batch).unwrap();

        assert_eq!(index.len(), 30);
        assert_eq!(
            index.partition_sizes().iter().sum::<usize>(),
            30
        );
    }

    #[test]
    fn test_ivf_dimension_mismatch() {
        let mut index = IvfIndex::new(3, 2).unwrap();
        let training = vec![vec![1.0, 2.0, 3.0]; 10];
        let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();
        index.train(&refs, 10).unwrap();

        let bad = vec![1.0, 2.0]; // dim 2, expected 3
        assert!(matches!(
            index.add(0, &bad),
            Err(QuantizationError::DimensionMismatch { .. })
        ));
    }
}
