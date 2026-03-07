// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Binary Quantization (BQ) — encodes each vector dimension as a single bit.
//!
//! BQ is the most aggressive quantization: 32x compression vs f32.
//! It's extremely fast for first-pass filtering via Hamming distance,
//! which approximates angular (cosine) distance.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::QuantizationError;

/// Binary quantizer that encodes each dimension as a single bit.
///
/// By default, uses sign-based hashing (threshold = 0.0).
/// After training, thresholds are set to per-dimension means
/// for better accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantizer {
    /// Original vector dimensionality.
    dimension: usize,
    /// Per-dimension thresholds; bit=1 when value >= threshold.
    thresholds: Vec<f32>,
    /// Whether `train()` has been called.
    trained: bool,
    /// Number of bytes needed per binary code: ceil(dimension / 8).
    code_size: usize,
}

impl BinaryQuantizer {
    /// Create a new binary quantizer with sign-based thresholds (all 0.0).
    pub fn new(dimension: usize) -> Self {
        let code_size = (dimension + 7) / 8;
        Self {
            dimension,
            thresholds: vec![0.0; dimension],
            trained: false,
            code_size,
        }
    }

    /// Train the quantizer by computing per-dimension mean thresholds.
    ///
    /// Using the mean as a threshold is more accurate than sign-based
    /// hashing because it adapts to the actual data distribution.
    pub fn train(&mut self, vectors: &[&[f32]]) -> Result<(), QuantizationError> {
        if vectors.is_empty() {
            return Err(QuantizationError::EmptyTrainingData);
        }

        // Validate dimensions
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dimension {
                return Err(QuantizationError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
            // Warn-level check: skip NaN vectors silently but count them
            if i == 0 && v.iter().any(|x| x.is_nan()) {
                log::warn!("training data contains NaN values; results may be degraded");
            }
        }

        // Compute per-dimension means as thresholds
        let n = vectors.len() as f32;
        let mut means = vec![0.0f32; self.dimension];

        for v in vectors {
            for (j, &val) in v.iter().enumerate() {
                means[j] += val;
            }
        }

        for mean in &mut means {
            *mean /= n;
        }

        self.thresholds = means;
        self.trained = true;

        log::info!(
            "binary quantizer trained on {} vectors, dim={}",
            vectors.len(),
            self.dimension
        );

        Ok(())
    }

    /// Quantize a single vector into a packed bit vector (Vec<u64>).
    ///
    /// For each dimension, if `value >= threshold` the corresponding bit is 1,
    /// otherwise 0. Bits are packed into u64 words in little-endian bit order
    /// (dimension 0 = bit 0 of word 0).
    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u64>, QuantizationError> {
        if vector.len() != self.dimension {
            return Err(QuantizationError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let num_words = self.code_size_u64s();
        let mut code = vec![0u64; num_words];

        for (i, (&val, &threshold)) in vector.iter().zip(self.thresholds.iter()).enumerate() {
            if val >= threshold {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                code[word_idx] |= 1u64 << bit_idx;
            }
        }

        Ok(code)
    }

    /// Quantize a batch of vectors in parallel using rayon.
    pub fn quantize_batch(
        &self,
        vectors: &[&[f32]],
    ) -> Result<Vec<Vec<u64>>, QuantizationError> {
        // Validate all dimensions first (fail fast)
        for v in vectors {
            if v.len() != self.dimension {
                return Err(QuantizationError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
        }

        let results: Vec<Vec<u64>> = vectors
            .par_iter()
            .map(|v| {
                // Safe to unwrap: we already validated dimensions above
                self.quantize(v).expect("dimension already validated")
            })
            .collect();

        Ok(results)
    }

    /// Returns the original vector dimensionality.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of bytes needed per binary code: ceil(dimension / 8).
    #[inline]
    pub fn code_size_bytes(&self) -> usize {
        self.code_size
    }

    /// Returns the number of u64 words needed per binary code: ceil(dimension / 64).
    #[inline]
    pub fn code_size_u64s(&self) -> usize {
        (self.dimension + 63) / 64
    }

    /// Returns whether the quantizer has been trained.
    #[inline]
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Returns memory usage per vector in bytes based on actual Vec<u64> storage.
    #[inline]
    pub fn memory_per_vector(&self) -> usize {
        self.code_size_u64s() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_defaults() {
        let bq = BinaryQuantizer::new(128);
        assert_eq!(bq.dimension(), 128);
        assert_eq!(bq.code_size_bytes(), 16);
        assert_eq!(bq.code_size_u64s(), 2);
        assert!(!bq.is_trained());
        assert_eq!(bq.memory_per_vector(), 16);
    }

    #[test]
    fn test_sign_based_quantize() {
        let bq = BinaryQuantizer::new(4);
        // thresholds are 0.0: positive => 1, negative => 0
        let code = bq.quantize(&[1.0, -1.0, 0.5, -0.5]).unwrap();
        assert_eq!(code.len(), 1);
        // bits: dim0=1, dim1=0, dim2=1, dim3=0 => 0b0101 = 5
        assert_eq!(code[0], 0b0101);
    }

    #[test]
    fn test_train_and_quantize() {
        let mut bq = BinaryQuantizer::new(4);
        let v1: &[f32] = &[2.0, 4.0, 6.0, 8.0];
        let v2: &[f32] = &[4.0, 6.0, 8.0, 10.0];
        bq.train(&[v1, v2]).unwrap();
        assert!(bq.is_trained());
        // means: [3.0, 5.0, 7.0, 9.0]
        // v1=[2,4,6,8]: all below mean => bits=0000
        let code = bq.quantize(v1).unwrap();
        assert_eq!(code[0], 0b0000);
        // v2=[4,6,8,10]: all above mean => bits=1111
        let code = bq.quantize(v2).unwrap();
        assert_eq!(code[0], 0b1111);
    }

    #[test]
    fn test_dimension_mismatch() {
        let bq = BinaryQuantizer::new(4);
        let err = bq.quantize(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            QuantizationError::DimensionMismatch {
                expected: 4,
                got: 2
            }
        ));
    }

    #[test]
    fn test_empty_training_data() {
        let mut bq = BinaryQuantizer::new(4);
        let err = bq.train(&[]).unwrap_err();
        assert!(matches!(err, QuantizationError::EmptyTrainingData));
    }

    #[test]
    fn test_code_size_edge_cases() {
        // Exact multiple of 8
        assert_eq!(BinaryQuantizer::new(64).code_size_bytes(), 8);
        assert_eq!(BinaryQuantizer::new(64).code_size_u64s(), 1);
        // Not a multiple
        assert_eq!(BinaryQuantizer::new(65).code_size_bytes(), 9);
        assert_eq!(BinaryQuantizer::new(65).code_size_u64s(), 2);
        // Small
        assert_eq!(BinaryQuantizer::new(1).code_size_bytes(), 1);
        assert_eq!(BinaryQuantizer::new(1).code_size_u64s(), 1);
    }

    #[test]
    fn test_quantize_batch() {
        let bq = BinaryQuantizer::new(4);
        let v1: &[f32] = &[1.0, -1.0, 1.0, -1.0];
        let v2: &[f32] = &[-1.0, 1.0, -1.0, 1.0];
        let codes = bq.quantize_batch(&[v1, v2]).unwrap();
        assert_eq!(codes.len(), 2);
        assert_eq!(codes[0][0], 0b0101);
        assert_eq!(codes[1][0], 0b1010);
    }
}
