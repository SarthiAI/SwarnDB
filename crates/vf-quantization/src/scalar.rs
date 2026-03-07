// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::QuantizationError;

/// Scalar quantizer that maps each f32 dimension to a u8 value [0, 255].
///
/// Per-dimension min/max values are learned during training (calibration).
/// Each float is linearly mapped to the [0, 255] range within its dimension's
/// observed range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    dimension: usize,
    min_vals: Vec<f32>,
    max_vals: Vec<f32>,
    ranges: Vec<f32>,
    trained: bool,
}

impl ScalarQuantizer {
    /// Create a new scalar quantizer for vectors of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            min_vals: Vec::new(),
            max_vals: Vec::new(),
            ranges: Vec::new(),
            trained: false,
        }
    }

    /// Train (calibrate) the quantizer by computing per-dimension min/max
    /// from the provided training vectors.
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
            if i == 0 {
                continue;
            }
        }

        if vectors.iter().any(|v| v.iter().any(|x| x.is_nan())) {
            return Err(QuantizationError::InvalidParameter("training data contains NaN".into()));
        }

        let mut min_vals = vec![f32::MAX; self.dimension];
        let mut max_vals = vec![f32::MIN; self.dimension];

        for v in vectors {
            for (d, &val) in v.iter().enumerate() {
                if val < min_vals[d] {
                    min_vals[d] = val;
                }
                if val > max_vals[d] {
                    max_vals[d] = val;
                }
            }
        }

        // Compute ranges, handling the edge case where min == max
        let ranges: Vec<f32> = min_vals
            .iter()
            .zip(max_vals.iter())
            .map(|(&mn, &mx)| {
                let r = mx - mn;
                if r == 0.0 {
                    1.0
                } else {
                    r
                }
            })
            .collect();

        self.min_vals = min_vals;
        self.max_vals = max_vals;
        self.ranges = ranges;
        self.trained = true;

        Ok(())
    }

    /// Quantize a single f32 vector to u8 codes.
    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u8>, QuantizationError> {
        if !self.trained {
            return Err(QuantizationError::NotTrained);
        }
        if vector.len() != self.dimension {
            return Err(QuantizationError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let codes: Vec<u8> = vector
            .iter()
            .enumerate()
            .map(|(d, &val)| {
                ((val - self.min_vals[d]) / self.ranges[d] * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8
            })
            .collect();

        Ok(codes)
    }

    /// Dequantize u8 codes back to approximate f32 values.
    pub fn dequantize(&self, codes: &[u8]) -> Result<Vec<f32>, QuantizationError> {
        if !self.trained {
            return Err(QuantizationError::NotTrained);
        }
        if codes.len() != self.dimension {
            return Err(QuantizationError::DimensionMismatch {
                expected: self.dimension,
                got: codes.len(),
            });
        }

        let vector: Vec<f32> = codes
            .iter()
            .enumerate()
            .map(|(d, &code)| code as f32 / 255.0 * self.ranges[d] + self.min_vals[d])
            .collect();

        Ok(vector)
    }

    /// Quantize a batch of vectors in parallel using rayon.
    pub fn quantize_batch(
        &self,
        vectors: &[&[f32]],
    ) -> Result<Vec<Vec<u8>>, QuantizationError> {
        if !self.trained {
            return Err(QuantizationError::NotTrained);
        }

        // Validate all dimensions upfront before parallel work
        for v in vectors {
            if v.len() != self.dimension {
                return Err(QuantizationError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
        }

        let results: Vec<Vec<u8>> = vectors
            .par_iter()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .map(|(d, &val)| {
                        ((val - self.min_vals[d]) / self.ranges[d] * 255.0)
                            .round()
                            .clamp(0.0, 255.0) as u8
                    })
                    .collect()
            })
            .collect();

        Ok(results)
    }

    /// Returns the vector dimension this quantizer is configured for.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns whether the quantizer has been trained.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Returns the memory usage per vector in bytes (1 byte per dimension).
    pub fn memory_per_vector(&self) -> usize {
        self.dimension
    }

    /// Returns a reference to the per-dimension minimum values.
    pub fn min_vals(&self) -> &[f32] {
        &self.min_vals
    }

    /// Returns a reference to the per-dimension maximum values.
    pub fn max_vals(&self) -> &[f32] {
        &self.max_vals
    }

    /// Returns a reference to the per-dimension ranges.
    pub fn ranges(&self) -> &[f32] {
        &self.ranges
    }
}
