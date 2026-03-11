// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Disk-based ANN storage: quantized vectors in RAM for fast approximate search,
//! full-precision vectors on mmap'd disk for re-ranking.

use std::collections::HashMap;
use std::path::Path;

use vf_core::types::VectorId;
use vf_quantization::scalar::ScalarQuantizer;

use crate::error::{StorageError, StorageResult};
use crate::mmap;

/// Magic bytes identifying a DiskANN data file.
const DISK_ANN_MAGIC: [u8; 4] = *b"VFDA";

/// Header size: magic(4) + dimension(4) + vector_count(8) + checksum(4) = 20 bytes.
const DISK_ANN_HEADER_SIZE: usize = 20;

/// Maximum vector count to prevent unbounded memory allocation (1B).
const MAX_VECTOR_COUNT: usize = 1_000_000_000;

/// Disk-based ANN store that keeps quantized vectors in RAM for fast approximate
/// search and full-precision vectors on mmap'd disk for re-ranking.
///
/// **Immutable after construction:** Once created via `build()` or `open()`, the
/// store does not support updating or removing individual vectors. To modify
/// the dataset, rebuild the store from scratch.
pub struct DiskAnnStore {
    dimension: usize,
    /// Quantized vectors in RAM for fast approximate search.
    quantized_data: HashMap<VectorId, Vec<u8>>,
    /// Full-precision vectors on mmap'd disk for re-ranking.
    mmap_data: Option<memmap2::Mmap>,
    /// Offset index: vector_id -> byte offset in mmap file.
    /// Offsets point to the start of each entry (the 8-byte ID field);
    /// the full-precision f32 vector data follows 8 bytes later.
    offset_index: HashMap<VectorId, u64>,
    /// Scalar quantizer for RAM-resident codes (retained for dequantization).
    #[allow(dead_code)]
    quantizer: Option<ScalarQuantizer>,
    /// Number of vectors stored.
    vector_count: usize,
    /// Path to the disk file (kept for disk_usage_bytes).
    disk_path: Option<std::path::PathBuf>,
}

impl DiskAnnStore {
    /// Create a new empty DiskAnnStore for vectors of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            quantized_data: HashMap::new(),
            mmap_data: None,
            offset_index: HashMap::new(),
            quantizer: None,
            vector_count: 0,
            disk_path: None,
        }
    }

    /// Build a DiskAnnStore from a set of vectors.
    ///
    /// Quantizes all vectors to RAM using the provided ScalarQuantizer,
    /// writes full-precision vectors to a disk file at `output_path`,
    /// and builds an offset index for mmap lookups.
    pub fn build(
        vectors: &[(VectorId, &[f32])],
        quantizer: ScalarQuantizer,
        output_path: &Path,
    ) -> StorageResult<Self> {
        if !quantizer.is_trained() {
            return Err(StorageError::Serialization(
                "ScalarQuantizer must be trained before building DiskAnnStore".into(),
            ));
        }

        let dimension = quantizer.dimension();

        // Validate all vector dimensions
        for (id, vec) in vectors {
            if vec.len() != dimension {
                return Err(StorageError::SegmentDimensionMismatch {
                    expected: dimension as u32,
                    actual: vec.len() as u32,
                });
            }
            let _ = id; // used below
        }

        // Calculate file size: header + (id(8) + vector_data(dim*4)) per vector
        let bytes_per_vector = 8 + dimension * 4; // u64 id + f32 * dim
        let file_size = DISK_ANN_HEADER_SIZE + vectors.len() * bytes_per_vector;

        // Create the mmap file
        let mut mmap_mut = mmap::create(output_path, file_size)?;

        // Write header (checksum placeholder at offset 16, computed after data)
        mmap_mut[0..4].copy_from_slice(&DISK_ANN_MAGIC);
        mmap_mut[4..8].copy_from_slice(&(dimension as u32).to_le_bytes());
        mmap_mut[8..16].copy_from_slice(&(vectors.len() as u64).to_le_bytes());
        // Checksum at [16..20] written after all data

        // Write vectors and build indices
        let mut quantized_data = HashMap::with_capacity(vectors.len());
        let mut offset_index = HashMap::with_capacity(vectors.len());
        let mut offset = DISK_ANN_HEADER_SIZE;

        for (id, vec) in vectors {
            // Record offset for this vector
            offset_index.insert(*id, offset as u64);

            // Write vector id
            mmap_mut[offset..offset + 8].copy_from_slice(&id.to_le_bytes());
            offset += 8;

            // Write full-precision f32 data
            for &val in *vec {
                mmap_mut[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                offset += 4;
            }

            // Quantize and store in RAM
            let codes = quantizer.quantize(vec).map_err(|e| {
                StorageError::Serialization(format!("quantization failed for id {}: {}", id, e))
            })?;
            quantized_data.insert(*id, codes);
        }

        // Task 707: Compute CRC32 over header fields AND vector data (skip checksum slot).
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&mmap_mut[0..16]);
        hasher.update(&mmap_mut[DISK_ANN_HEADER_SIZE..]);
        let checksum = hasher.finalize();
        mmap_mut[16..20].copy_from_slice(&checksum.to_le_bytes());

        // Flush to disk
        mmap::sync(&mmap_mut)?;

        // Re-open as read-only mmap
        drop(mmap_mut);
        let mmap_read = mmap::open_read(output_path)?;

        Ok(Self {
            dimension,
            quantized_data,
            mmap_data: Some(mmap_read),
            offset_index,
            quantizer: Some(quantizer),
            vector_count: vectors.len(),
            disk_path: Some(output_path.to_path_buf()),
        })
    }

    /// Open an existing DiskAnnStore from a previously written disk file.
    pub fn open(path: &Path, quantizer: ScalarQuantizer) -> StorageResult<Self> {
        if !quantizer.is_trained() {
            return Err(StorageError::Serialization(
                "ScalarQuantizer must be trained before opening DiskAnnStore".into(),
            ));
        }

        let mmap_read = mmap::open_read(path)?;

        // Validate header
        if mmap_read.len() < DISK_ANN_HEADER_SIZE {
            return Err(StorageError::TruncatedData {
                expected: DISK_ANN_HEADER_SIZE,
                actual: mmap_read.len(),
            });
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&mmap_read[0..4]);
        if magic != DISK_ANN_MAGIC {
            return Err(StorageError::BadMagic {
                expected: DISK_ANN_MAGIC,
                found: magic,
            });
        }

        let dimension =
            u32::from_le_bytes(mmap_read[4..8].try_into().unwrap()) as usize;
        let vector_count =
            u64::from_le_bytes(mmap_read[8..16].try_into().unwrap()) as usize;

        // Task 707: Verify CRC32 checksum over header fields AND vector data.
        let stored_checksum = u32::from_le_bytes(mmap_read[16..20].try_into().unwrap());
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&mmap_read[0..16]);
        hasher.update(&mmap_read[DISK_ANN_HEADER_SIZE..]);
        let computed_checksum = hasher.finalize();
        if stored_checksum != computed_checksum {
            return Err(StorageError::ChecksumMismatch {
                expected: stored_checksum,
                computed: computed_checksum,
            });
        }

        // Task 279: Limit vector count before allocation.
        if vector_count > MAX_VECTOR_COUNT {
            return Err(StorageError::SegmentInvalidHeader(format!(
                "DiskANN vector count {} exceeds maximum allowed ({})",
                vector_count, MAX_VECTOR_COUNT
            )));
        }

        if dimension != quantizer.dimension() {
            return Err(StorageError::SegmentDimensionMismatch {
                expected: quantizer.dimension() as u32,
                actual: dimension as u32,
            });
        }

        // Read all vectors: rebuild quantized_data and offset_index
        let bytes_per_vector = 8 + dimension * 4;
        let expected_size = DISK_ANN_HEADER_SIZE + vector_count * bytes_per_vector;
        if mmap_read.len() < expected_size {
            return Err(StorageError::TruncatedData {
                expected: expected_size,
                actual: mmap_read.len(),
            });
        }

        let mut quantized_data = HashMap::with_capacity(vector_count);
        let mut offset_index = HashMap::with_capacity(vector_count);
        let mut offset = DISK_ANN_HEADER_SIZE;

        for _ in 0..vector_count {
            let id = u64::from_le_bytes(
                mmap_read[offset..offset + 8].try_into().unwrap(),
            );
            offset_index.insert(id, offset as u64);
            offset += 8;

            // Read f32 vector for quantization
            let mut vec = Vec::with_capacity(dimension);
            for d in 0..dimension {
                let start = offset + d * 4;
                let val = f32::from_le_bytes(
                    mmap_read[start..start + 4].try_into().unwrap(),
                );
                vec.push(val);
            }
            offset += dimension * 4;

            // Quantize into RAM
            let codes = quantizer.quantize(&vec).map_err(|e| {
                StorageError::Serialization(format!(
                    "quantization failed for id {}: {}",
                    id, e
                ))
            })?;
            quantized_data.insert(id, codes);
        }

        Ok(Self {
            dimension,
            quantized_data,
            mmap_data: Some(mmap_read),
            offset_index,
            quantizer: Some(quantizer),
            vector_count,
            disk_path: Some(path.to_path_buf()),
        })
    }

    /// Fast RAM lookup of quantized (u8) codes for a vector.
    pub fn get_quantized(&self, id: VectorId) -> Option<&[u8]> {
        self.quantized_data.get(&id).map(|v| v.as_slice())
    }

    /// Mmap disk lookup for re-ranking: returns the full-precision f32 vector.
    pub fn get_full_vector(&self, id: VectorId) -> Option<Vec<f32>> {
        let &byte_offset = self.offset_index.get(&id)?;
        let mmap = self.mmap_data.as_ref()?;

        // Skip past the 8-byte id field
        let data_start = byte_offset as usize + 8;
        let data_end = data_start + self.dimension * 4;

        if mmap.len() < data_end {
            return None;
        }

        let mut vector = Vec::with_capacity(self.dimension);
        for d in 0..self.dimension {
            let start = data_start + d * 4;
            let val = f32::from_le_bytes(
                mmap[start..start + 4].try_into().unwrap(),
            );
            vector.push(val);
        }

        Some(vector)
    }

    /// Batch mmap disk lookup for re-ranking multiple vectors.
    pub fn get_full_vectors_batch(
        &self,
        ids: &[VectorId],
    ) -> Vec<(VectorId, Vec<f32>)> {
        ids.iter()
            .filter_map(|&id| {
                self.get_full_vector(id).map(|vec| (id, vec))
            })
            .collect()
    }

    /// Returns the number of vectors stored.
    pub fn len(&self) -> usize {
        self.vector_count
    }

    /// Returns true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }

    /// Returns the approximate RAM usage in bytes (quantized data only).
    pub fn ram_usage_bytes(&self) -> usize {
        self.quantized_data
            .values()
            .map(|v| v.len() + std::mem::size_of::<VectorId>())
            .sum::<usize>()
            + self.offset_index.len()
                * (std::mem::size_of::<VectorId>() + std::mem::size_of::<u64>())
    }

    /// Returns the disk file size in bytes.
    pub fn disk_usage_bytes(&self) -> u64 {
        self.disk_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Returns the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}
