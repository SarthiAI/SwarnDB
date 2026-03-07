// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Product Quantizer — splits vectors into M subvectors, quantizes each
//! independently using a 256-entry codebook (codes fit in `u8`).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::QuantizationError;
use crate::kmeans;

/// Number of centroids per sub-quantizer (fixed at 256 so each code is a u8).
const NUM_CENTROIDS: usize = 256;

/// Product Quantizer that compresses a D-dimensional vector into M bytes.
///
/// After training, each vector is encoded as `M` bytes where byte `m` indexes
/// into codebook `m`. Decoding concatenates the looked-up centroids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Original vector dimensionality.
    dimension: usize,
    /// Number of subvector partitions (M).
    num_subquantizers: usize,
    /// Dimensionality of each subvector (dimension / M).
    subvector_dim: usize,
    /// Number of centroids per codebook (K = 256).
    num_centroids: usize,
    /// M codebooks, each containing K centroids of `subvector_dim` dimensions.
    /// Layout: `codebooks[m][k][d]`.
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Whether `train()` has been called successfully.
    trained: bool,
}

impl ProductQuantizer {
    /// Create a new (untrained) product quantizer.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `dimension` is not evenly divisible by
    /// `num_subquantizers`, or if either value is zero.
    pub fn new(
        dimension: usize,
        num_subquantizers: usize,
    ) -> Result<Self, QuantizationError> {
        if dimension == 0 || num_subquantizers == 0 {
            return Err(QuantizationError::InvalidParameter(
                "dimension and num_subquantizers must be > 0".into(),
            ));
        }
        if dimension % num_subquantizers != 0 {
            return Err(QuantizationError::InvalidParameter(format!(
                "dimension {} is not divisible by num_subquantizers {}",
                dimension, num_subquantizers
            )));
        }

        let subvector_dim = dimension / num_subquantizers;

        Ok(Self {
            dimension,
            num_subquantizers,
            subvector_dim,
            num_centroids: NUM_CENTROIDS,
            codebooks: Vec::new(),
            trained: false,
        })
    }

    // ---- Accessors ----

    /// Full vector dimensionality.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of sub-quantizers (M).
    pub fn num_subquantizers(&self) -> usize {
        self.num_subquantizers
    }

    /// Dimensionality of each subvector partition.
    pub fn subvector_dim(&self) -> usize {
        self.subvector_dim
    }

    /// Number of centroids per codebook (always 256).
    pub fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    /// Whether the quantizer has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Reference to the internal codebooks (M x K x subvector_dim).
    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    // ---- Core API ----

    /// Train the product quantizer by running k-means on each subvector slice.
    ///
    /// # Arguments
    /// * `vectors` — training vectors, each of length `self.dimension`.
    /// * `max_iters` — maximum k-means iterations per sub-quantizer.
    ///
    /// # Errors
    /// * `EmptyTrainingData` if `vectors` is empty.
    /// * `DimensionMismatch` if any vector has incorrect length.
    pub fn train(
        &mut self,
        vectors: &[&[f32]],
        max_iters: usize,
    ) -> Result<(), QuantizationError> {
        if vectors.is_empty() {
            return Err(QuantizationError::EmptyTrainingData);
        }

        // Validate dimensions.
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dimension {
                return Err(QuantizationError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
            let _ = i; // suppress unused warning
        }

        let mut codebooks = Vec::with_capacity(self.num_subquantizers);

        for m in 0..self.num_subquantizers {
            let start = m * self.subvector_dim;
            let end = start + self.subvector_dim;

            // Extract subvectors for this partition.
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            let sub_refs: Vec<&[f32]> = subvectors.iter().map(|s| s.as_slice()).collect();

            // Use subquantizer index as part of the seed for variety across partitions.
            let seed = 42u64.wrapping_add(m as u64);
            let result = kmeans::kmeans(&sub_refs, self.num_centroids, max_iters, seed);

            codebooks.push(result.centroids);
        }

        self.codebooks = codebooks;
        self.trained = true;

        log::debug!(
            "PQ trained: M={}, K={}, subvec_dim={}, training_size={}",
            self.num_subquantizers,
            self.num_centroids,
            self.subvector_dim,
            vectors.len()
        );

        Ok(())
    }

    /// Encode a single vector into M bytes (one per sub-quantizer).
    ///
    /// Each byte is the index of the nearest centroid in that sub-quantizer's
    /// codebook.
    ///
    /// # Errors
    /// * `NotTrained` if `train()` has not been called.
    /// * `DimensionMismatch` if the vector length is wrong.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>, QuantizationError> {
        self.check_trained()?;
        self.check_dimension(vector.len())?;

        let mut codes = Vec::with_capacity(self.num_subquantizers);

        for m in 0..self.num_subquantizers {
            let start = m * self.subvector_dim;
            let end = start + self.subvector_dim;
            let sub = &vector[start..end];

            let code = nearest_centroid(sub, &self.codebooks[m]);
            codes.push(code);
        }

        Ok(codes)
    }

    /// Decode PQ codes back into an approximate vector by concatenating centroids.
    ///
    /// # Errors
    /// * `NotTrained` if `train()` has not been called.
    /// * `DimensionMismatch` if `codes.len() != num_subquantizers`.
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>, QuantizationError> {
        self.check_trained()?;

        if codes.len() != self.num_subquantizers {
            return Err(QuantizationError::DimensionMismatch {
                expected: self.num_subquantizers,
                got: codes.len(),
            });
        }

        let mut vector = Vec::with_capacity(self.dimension);

        for (m, &code) in codes.iter().enumerate() {
            let centroid = &self.codebooks[m][code as usize];
            vector.extend_from_slice(centroid);
        }

        Ok(vector)
    }

    /// Encode a batch of vectors in parallel.
    ///
    /// # Errors
    /// * `NotTrained` if `train()` has not been called.
    /// * `DimensionMismatch` if any vector has incorrect length.
    pub fn encode_batch(
        &self,
        vectors: &[&[f32]],
    ) -> Result<Vec<Vec<u8>>, QuantizationError> {
        self.check_trained()?;

        // Pre-validate all dimensions.
        for v in vectors {
            self.check_dimension(v.len())?;
        }

        let results: Vec<Vec<u8>> = vectors
            .par_iter()
            .map(|v| {
                let mut codes = Vec::with_capacity(self.num_subquantizers);
                for m in 0..self.num_subquantizers {
                    let start = m * self.subvector_dim;
                    let end = start + self.subvector_dim;
                    let sub = &v[start..end];
                    codes.push(nearest_centroid(sub, &self.codebooks[m]));
                }
                codes
            })
            .collect();

        Ok(results)
    }

    // ---- Internal helpers ----

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

/// Find the index of the nearest centroid to `sub` using squared Euclidean distance.
#[inline]
fn nearest_centroid(sub: &[f32], codebook: &[Vec<f32>]) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = f32::MAX;

    for (i, centroid) in codebook.iter().enumerate() {
        let d: f32 = sub
            .iter()
            .zip(centroid.iter())
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .sum();

        if d < best_dist {
            best_dist = d;
            best_idx = i as u8;
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let pq = ProductQuantizer::new(128, 8).unwrap();
        assert_eq!(pq.dimension(), 128);
        assert_eq!(pq.num_subquantizers(), 8);
        assert_eq!(pq.subvector_dim(), 16);
        assert!(!pq.is_trained());
    }

    #[test]
    fn test_new_invalid_divisibility() {
        let result = ProductQuantizer::new(127, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_zero_params() {
        assert!(ProductQuantizer::new(0, 8).is_err());
        assert!(ProductQuantizer::new(128, 0).is_err());
    }

    #[test]
    fn test_encode_not_trained() {
        let pq = ProductQuantizer::new(8, 2).unwrap();
        let v = vec![0.0; 8];
        assert!(pq.encode(&v).is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let dim = 8;
        let m = 2;
        let mut pq = ProductQuantizer::new(dim, m).unwrap();

        // Generate simple training data.
        let training: Vec<Vec<f32>> = (0..300)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.01).collect())
            .collect();
        let refs: Vec<&[f32]> = training.iter().map(|v| v.as_slice()).collect();

        pq.train(&refs, 20).unwrap();
        assert!(pq.is_trained());

        let test_vec: Vec<f32> = (0..dim).map(|d| d as f32 * 0.5).collect();
        let codes = pq.encode(&test_vec).unwrap();
        assert_eq!(codes.len(), m);

        let decoded = pq.decode(&codes).unwrap();
        assert_eq!(decoded.len(), dim);
    }
}
