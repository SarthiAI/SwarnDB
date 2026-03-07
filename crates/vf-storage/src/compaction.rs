// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Segment compaction: merges multiple segments into one, removing deleted
//! vectors and deduplicating by vector id to reclaim disk space.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::error::{StorageError, StorageResult};
use crate::format::DataTypeFlag;
use crate::segment::{Segment, SegmentWriter};
use vf_core::types::{Metadata, VectorId};
use vf_core::vector::VectorData;

// ── Options ──────────────────────────────────────────────────────────────────

/// Configuration knobs for segment compaction.
#[derive(Clone, Debug)]
pub struct CompactionOptions {
    /// Minimum number of segments required before compaction triggers.
    pub min_segments_to_compact: usize,
    /// Whether to skip vectors whose ids appear in the deleted set.
    pub remove_deleted: bool,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            min_segments_to_compact: 4,
            remove_deleted: true,
        }
    }
}

// ── Result ───────────────────────────────────────────────────────────────────

/// Statistics returned after a successful compaction.
#[derive(Clone, Debug)]
pub struct CompactionResult {
    /// Number of input segments that were merged.
    pub segments_merged: usize,
    /// Number of vectors written to the new segment.
    pub vectors_written: u64,
    /// Number of vectors removed (deleted or deduplicated).
    pub vectors_removed: u64,
    /// Path of the newly created compacted segment file.
    pub new_segment_path: PathBuf,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Returns `true` if the current segment count meets the compaction threshold.
pub fn should_compact(segment_count: usize, options: &CompactionOptions) -> bool {
    segment_count >= options.min_segments_to_compact
}

/// Merge multiple segments into a single new segment, removing deleted vectors
/// and deduplicating by vector id (keeping the version from the highest segment
/// id, i.e. the newest).
///
/// # Arguments
///
/// * `segments`       - Slice of segments to compact (order does not matter).
/// * `deleted_ids`    - Set of vector ids that should be excluded from the output.
/// * `output_dir`     - Directory where the new segment file will be written.
/// * `new_segment_id` - Id used for the output segment filename.
/// * `options`        - Compaction configuration.
///
/// # Errors
///
/// Returns `StorageError` if any segment cannot be read or if the new segment
/// cannot be written.
pub fn compact_segments(
    segments: &[Segment],
    deleted_ids: &HashSet<u64>,
    output_dir: &Path,
    new_segment_id: u64,
    options: &CompactionOptions,
) -> StorageResult<CompactionResult> {
    if segments.is_empty() {
        return Err(StorageError::SegmentInvalidHeader(
            "cannot compact zero segments".into(),
        ));
    }

    // Validate that all segments share the same dimension.
    let dimension = segments[0].dimension();
    for seg in &segments[1..] {
        if seg.dimension() != dimension {
            return Err(StorageError::SegmentDimensionMismatch {
                expected: dimension,
                actual: seg.dimension(),
            });
        }
    }

    // Collect vectors, deduplicating by id.
    // For each vector id we keep the entry from the segment with the highest id
    // (newest data wins).
    //
    // Map value: (segment_id, vector_data_f32, metadata)
    let mut winner_map: HashMap<VectorId, (u64, Vec<f32>, Option<Metadata>)> = HashMap::new();
    let mut total_input_vectors: u64 = 0;

    for seg in segments {
        let seg_id = seg.id();
        let count = seg.vector_count() as usize;
        for i in 0..count {
            total_input_vectors += 1;

            let (vid, data) = seg.get_vector_data(i)?;

            // Skip deleted vectors when configured to do so.
            if options.remove_deleted && deleted_ids.contains(&vid) {
                continue;
            }

            // Keep vector only if this segment is newer than any previously seen.
            let dominated = winner_map
                .get(&vid)
                .map_or(false, |(existing_seg_id, _, _)| *existing_seg_id > seg_id);

            if !dominated {
                let meta = seg.get_metadata(vid)?;
                winner_map.insert(vid, (seg_id, data, meta));
            }
        }
    }

    // Build a sorted list of records for deterministic output.
    let mut records: Vec<(VectorId, Vec<f32>, Option<Metadata>)> = winner_map
        .into_iter()
        .map(|(vid, (_seg_id, data, meta))| (vid, data, meta))
        .collect();
    records.sort_by_key(|(vid, _, _)| *vid);

    let vectors_written = records.len() as u64;
    let vectors_removed = total_input_vectors - vectors_written;

    // Convert f32 vecs into VectorData for the writer.
    let vector_data: Vec<VectorData> = records
        .iter()
        .map(|(_, floats, _)| VectorData::F32(floats.clone()))
        .collect();

    let refs: Vec<(VectorId, &VectorData, Option<&Metadata>)> = records
        .iter()
        .zip(vector_data.iter())
        .map(|((vid, _, meta), vd)| (*vid, vd, meta.as_ref()))
        .collect();

    // Write the compacted segment.
    let filename = format!("segment_{new_segment_id:08}.vfs");
    let new_segment_path = output_dir.join(&filename);

    SegmentWriter::write_segment(
        &new_segment_path,
        new_segment_id,
        &refs,
        dimension,
        DataTypeFlag::F32,
    )?;

    Ok(CompactionResult {
        segments_merged: segments.len(),
        vectors_written,
        vectors_removed,
        new_segment_path,
    })
}
