// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::QuantizationError;

const PERSIST_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct PersistedScalarQuantizer {
    version: u32,
    dimension: usize,
    bits: u8,
    min_vals: Vec<f32>,
    max_vals: Vec<f32>,
}

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
        for v in vectors.iter() {
            if v.len() != self.dimension {
                return Err(QuantizationError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
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

    /// Train the quantizer using percentile-based range calibration.
    /// Instead of using absolute min/max (sensitive to outliers), clips to the
    /// given quantile boundaries. For example, quantile=0.99 uses 0.5th and 99.5th percentiles.
    pub fn train_with_quantile(
        &mut self,
        vectors: &[&[f32]],
        quantile: f32,
    ) -> Result<(), QuantizationError> {
        if vectors.is_empty() {
            return Err(QuantizationError::EmptyTrainingData);
        }

        // Validate dimensions
        for v in vectors.iter() {
            if v.len() != self.dimension {
                return Err(QuantizationError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
        }

        let n = vectors.len();
        let low_idx = ((1.0 - quantile) / 2.0 * n as f32) as usize;
        let high_idx = (n - 1).min(((1.0 + quantile) / 2.0 * n as f32) as usize);

        // Parallelize across dimensions: each thread sorts its own dim_values buffer.
        let bounds: Vec<(f32, f32)> = (0..self.dimension)
            .into_par_iter()
            .map(|d| {
                let mut dim_values: Vec<f32> = vectors.iter().map(|v| v[d]).collect();
                dim_values
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                (dim_values[low_idx], dim_values[high_idx])
            })
            .collect();

        let mut min_vals = vec![0.0f32; self.dimension];
        let mut max_vals = vec![0.0f32; self.dimension];
        let mut ranges = vec![0.0f32; self.dimension];
        for (d, (lo, hi)) in bounds.into_iter().enumerate() {
            min_vals[d] = lo;
            max_vals[d] = hi;
            let range = hi - lo;
            ranges[d] = if range == 0.0 { 1.0 } else { range };
        }

        self.min_vals = min_vals;
        self.max_vals = max_vals;
        self.ranges = ranges;
        self.trained = true;

        Ok(())
    }

    /// Create a ScalarQuantizer from pre-trained parameters.
    /// Used for restoring a trained quantizer from persisted state.
    pub fn from_trained(dimension: usize, min_vals: Vec<f32>, max_vals: Vec<f32>) -> Self {
        let ranges: Vec<f32> = min_vals
            .iter()
            .zip(max_vals.iter())
            .map(|(min, max)| {
                let r = max - min;
                if r == 0.0 {
                    1.0
                } else {
                    r
                }
            })
            .collect();

        Self {
            dimension,
            min_vals,
            max_vals,
            ranges,
            trained: true,
        }
    }

    /// Returns precomputed scales (ranges / 255.0) for SIMD distance functions.
    pub fn scales(&self) -> Vec<f32> {
        self.ranges.iter().map(|r| r / 255.0).collect()
    }

    /// Atomically persist the trained quantizer state to a JSON file.
    pub fn save_to_path(&self, path: &Path) -> Result<(), QuantizationError> {
        if !self.trained {
            return Err(QuantizationError::NotTrained);
        }

        let payload = PersistedScalarQuantizer {
            version: PERSIST_VERSION,
            dimension: self.dimension,
            bits: 8,
            min_vals: self.min_vals.clone(),
            max_vals: self.max_vals.clone(),
        };

        let bytes = serde_json::to_vec(&payload)
            .map_err(|e| QuantizationError::Serialization(e.to_string()))?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let tmp_path = path.with_extension("tmp");
        let _ = fs::remove_file(&tmp_path);

        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            file.write_all(&bytes)?;
            file.sync_data()?;
        }

        fs::rename(&tmp_path, path)?;

        if let Some(parent) = path.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }

        Ok(())
    }

    /// Load a previously persisted quantizer state from a JSON file.
    pub fn load_from_path(path: &Path) -> Result<Self, QuantizationError> {
        let bytes = fs::read(path)?;
        let payload: PersistedScalarQuantizer = serde_json::from_slice(&bytes)
            .map_err(|e| QuantizationError::Serialization(e.to_string()))?;

        if payload.version != PERSIST_VERSION {
            return Err(QuantizationError::Corrupt(format!(
                "unsupported quantizer file version {}",
                payload.version
            )));
        }
        if payload.bits != 8 {
            return Err(QuantizationError::Corrupt(format!(
                "unsupported quantizer bits {}",
                payload.bits
            )));
        }
        if payload.dimension == 0 {
            return Err(QuantizationError::Corrupt("dimension is zero".into()));
        }
        if payload.min_vals.len() != payload.dimension
            || payload.max_vals.len() != payload.dimension
        {
            return Err(QuantizationError::Corrupt(format!(
                "min/max length mismatch: dim={} min={} max={}",
                payload.dimension,
                payload.min_vals.len(),
                payload.max_vals.len()
            )));
        }

        Ok(Self::from_trained(
            payload.dimension,
            payload.min_vals,
            payload.max_vals,
        ))
    }
}
