// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use half::f16;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported vector data storage formats.
/// All distance computations convert to f32 internally.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VectorData {
    /// Full precision 32-bit floating point
    F32(Vec<f32>),
    /// Half precision 16-bit floating point (stored as half::f16)
    F16(Vec<f16>),
    /// Unsigned 8-bit integer (quantized)
    U8(Vec<u8>),
}

impl VectorData {
    /// Returns the number of dimensions
    pub fn dimension(&self) -> usize {
        match self {
            VectorData::F32(v) => v.len(),
            VectorData::F16(v) => v.len(),
            VectorData::U8(v) => v.len(),
        }
    }

    /// Convert to a new Vec<f32> for computation.
    /// Always allocates. Use `as_f32_slice()` for zero-copy access to F32 data.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            VectorData::F32(v) => v.clone(),
            VectorData::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            VectorData::U8(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }

    /// Borrow as f32 slice if already F32, otherwise None.
    /// Use this for zero-copy access when the data is already f32.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            VectorData::F32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Returns the storage type as a string identifier
    pub fn data_type(&self) -> DataType {
        match self {
            VectorData::F32(_) => DataType::F32,
            VectorData::F16(_) => DataType::F16,
            VectorData::U8(_) => DataType::U8,
        }
    }

    /// Returns the size in bytes of the raw vector data
    pub fn size_bytes(&self) -> usize {
        match self {
            VectorData::F32(v) => v.len() * std::mem::size_of::<f32>(),
            VectorData::F16(v) => v.len() * std::mem::size_of::<f16>(),
            VectorData::U8(v) => v.len() * std::mem::size_of::<u8>(),
        }
    }

    /// Validate that the vector has the expected dimension
    pub fn validate_dimension(&self, expected: usize) -> Result<(), VectorError> {
        let actual = self.dimension();
        if actual != expected {
            Err(VectorError::DimensionMismatch { expected, actual })
        } else {
            Ok(())
        }
    }

    /// Check if the vector contains any NaN or Infinity values (for f32/f16)
    pub fn validate_finite(&self) -> Result<(), VectorError> {
        match self {
            VectorData::F32(v) => {
                for (i, &val) in v.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(VectorError::NonFiniteValue { index: i });
                    }
                }
            }
            VectorData::F16(v) => {
                for (i, val) in v.iter().enumerate() {
                    if !val.to_f32().is_finite() {
                        return Err(VectorError::NonFiniteValue { index: i });
                    }
                }
            }
            VectorData::U8(_) => {} // u8 values are always finite
        }
        Ok(())
    }

    /// Normalize the vector to unit length (L2 norm = 1.0).
    /// Returns a new VectorData::F32 with the normalized values.
    pub fn normalize(&self) -> Result<VectorData, VectorError> {
        let f32_vec = self.to_f32_vec();
        let norm: f32 = f32_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Err(VectorError::ZeroNorm);
        }
        Ok(VectorData::F32(f32_vec.iter().map(|x| x / norm).collect()))
    }
}

impl fmt::Display for VectorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorData::F32(v) => write!(f, "F32[dim={}]", v.len()),
            VectorData::F16(v) => write!(f, "F16[dim={}]", v.len()),
            VectorData::U8(v) => write!(f, "U8[dim={}]", v.len()),
        }
    }
}

/// Storage data type identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F16,
    U8,
}

impl DataType {
    /// Bytes per element
    pub fn element_size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::U8 => 1,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::F32 => write!(f, "f32"),
            DataType::F16 => write!(f, "f16"),
            DataType::U8 => write!(f, "u8"),
        }
    }
}

/// Errors specific to vector operations
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("non-finite value at index {index}")]
    NonFiniteValue { index: usize },

    #[error("zero-norm vector cannot be normalized")]
    ZeroNorm,

    #[error("empty vector not allowed")]
    EmptyVector,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_dimension() {
        let v = VectorData::F32(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dimension(), 3);
    }

    #[test]
    fn test_f16_to_f32() {
        let vals = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let v = VectorData::F16(vals);
        let f32_vec = v.to_f32_vec();
        assert!((f32_vec[0] - 1.0).abs() < 1e-3);
        assert!((f32_vec[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_u8_to_f32() {
        let v = VectorData::U8(vec![0, 128, 255]);
        let f32_vec = v.to_f32_vec();
        assert_eq!(f32_vec, vec![0.0, 128.0, 255.0]);
    }

    #[test]
    fn test_validate_dimension() {
        let v = VectorData::F32(vec![1.0, 2.0, 3.0]);
        assert!(v.validate_dimension(3).is_ok());
        assert!(v.validate_dimension(4).is_err());
    }

    #[test]
    fn test_validate_finite() {
        let good = VectorData::F32(vec![1.0, 2.0, 3.0]);
        assert!(good.validate_finite().is_ok());

        let bad = VectorData::F32(vec![1.0, f32::NAN, 3.0]);
        assert!(bad.validate_finite().is_err());

        let inf = VectorData::F32(vec![1.0, f32::INFINITY, 3.0]);
        assert!(inf.validate_finite().is_err());
    }

    #[test]
    fn test_normalize() {
        let v = VectorData::F32(vec![3.0, 4.0]);
        let normalized = v.normalize().unwrap();
        let vals = normalized.to_f32_vec();
        assert!((vals[0] - 0.6).abs() < 1e-6);
        assert!((vals[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_zero_norm() {
        let v = VectorData::F32(vec![0.0, 0.0, 0.0]);
        assert!(v.normalize().is_err());
    }

    #[test]
    fn test_size_bytes() {
        let v = VectorData::F32(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.size_bytes(), 12); // 3 * 4 bytes
    }

    #[test]
    fn test_as_f32_slice() {
        let v = VectorData::F32(vec![1.0, 2.0]);
        assert!(v.as_f32_slice().is_some());

        let v2 = VectorData::U8(vec![1, 2]);
        assert!(v2.as_f32_slice().is_none());
    }

    #[test]
    fn test_display() {
        let v = VectorData::F32(vec![1.0; 128]);
        assert_eq!(format!("{}", v), "F32[dim=128]");
    }
}
