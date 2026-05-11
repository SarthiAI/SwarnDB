// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! HNSW topology-only persistence (Base snapshot format).
//!
//! Serializes only the graph structure (adjacency lists, entry point, params)
//! without any vector data, since vectors are already persisted in .vfs segment
//! files. This makes snapshots dramatically smaller and faster.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::Mmap;

use crc32fast::Hasher as Crc32Hasher;
use vf_core::types::VectorId;

use crate::traits::IndexError;

/// Magic bytes identifying the HNSW topology binary format.
const TOPOLOGY_MAGIC: &[u8; 4] = b"HNSW";
/// Current format version.
const FORMAT_VERSION: u32 = 1;
/// Envelope size in bytes.
const ENVELOPE_SIZE: usize = 80;
/// Compression flag: none.
const COMPRESSION_NONE: u8 = 0;
/// Compression flag: LZ4.
const COMPRESSION_LZ4: u8 = 1;
/// Sentinel value for absent entry point.
const NO_ENTRY_POINT: u64 = u64::MAX;

// ── Snapshot types ──────────────────────────────────────────────────────

/// Snapshot of a single node's topology (no vector data).
pub struct TopologyNode {
    pub id: VectorId,
    pub level: usize,
    pub vector_slot: usize,
    pub neighbors: Vec<Vec<VectorId>>, // neighbors[layer] = [neighbor_ids]
}

/// Complete HNSW topology snapshot, reconstructable into an HnswIndex.
pub struct HnswTopologySnapshot {
    pub snapshot_lsn: u64,
    pub timestamp: u64,
    pub dimension: u32,
    pub metric: u8, // 0=Cosine, 1=Euclidean, 2=DotProduct, 3=Manhattan
    pub m: u32,
    pub m0: u32,
    pub ef_construction: u32,
    pub ef_search: u32,
    pub entry_point: Option<VectorId>,
    pub max_level: u32,
    pub nodes: Vec<TopologyNode>,
}

impl HnswTopologySnapshot {
    /// Returns the current time as unix milliseconds.
    pub fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

// ── Serialization helpers ───────────────────────────────────────────────

/// Build the binary payload for all topology nodes (uncompressed).
fn build_payload(snapshot: &HnswTopologySnapshot) -> Vec<u8> {
    // Estimate capacity: each node needs at least 20 bytes + neighbor data.
    let estimated = snapshot.nodes.len() * 64;
    let mut buf = Vec::with_capacity(estimated);

    for node in &snapshot.nodes {
        buf.extend_from_slice(&node.id.to_le_bytes()); // 8 bytes
        buf.extend_from_slice(&(node.level as u32).to_le_bytes()); // 4 bytes
        buf.extend_from_slice(&(node.vector_slot as u64).to_le_bytes()); // 8 bytes

        for layer in 0..=node.level {
            let neighbors = if layer < node.neighbors.len() {
                &node.neighbors[layer]
            } else {
                // Should not happen in a well-formed snapshot, but handle gracefully.
                &[][..]
            };
            buf.extend_from_slice(&(neighbors.len() as u32).to_le_bytes()); // 4 bytes
            for &nid in neighbors {
                buf.extend_from_slice(&nid.to_le_bytes()); // 8 bytes each
            }
        }
    }

    buf
}

/// Build the 80-byte envelope.
fn build_envelope(
    snapshot: &HnswTopologySnapshot,
    compression: u8,
    uncompressed_size: u64,
) -> [u8; ENVELOPE_SIZE] {
    let mut env = [0u8; ENVELOPE_SIZE];
    let mut pos = 0;

    // [0-3] Magic
    env[pos..pos + 4].copy_from_slice(TOPOLOGY_MAGIC);
    pos += 4;

    // [4-7] format_version
    env[pos..pos + 4].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    pos += 4;

    // [8-15] snapshot_lsn
    env[pos..pos + 8].copy_from_slice(&snapshot.snapshot_lsn.to_le_bytes());
    pos += 8;

    // [16-23] timestamp
    env[pos..pos + 8].copy_from_slice(&snapshot.timestamp.to_le_bytes());
    pos += 8;

    // [24-31] node_count
    env[pos..pos + 8].copy_from_slice(&(snapshot.nodes.len() as u64).to_le_bytes());
    pos += 8;

    // [32-35] dimension
    env[pos..pos + 4].copy_from_slice(&snapshot.dimension.to_le_bytes());
    pos += 4;

    // [36] metric
    env[pos] = snapshot.metric;
    pos += 1;

    // [37-40] m
    env[pos..pos + 4].copy_from_slice(&snapshot.m.to_le_bytes());
    pos += 4;

    // [41-44] m0
    env[pos..pos + 4].copy_from_slice(&snapshot.m0.to_le_bytes());
    pos += 4;

    // [45-48] ef_construction
    env[pos..pos + 4].copy_from_slice(&snapshot.ef_construction.to_le_bytes());
    pos += 4;

    // [49-52] ef_search
    env[pos..pos + 4].copy_from_slice(&snapshot.ef_search.to_le_bytes());
    pos += 4;

    // [53-60] entry_point
    let ep = snapshot.entry_point.unwrap_or(NO_ENTRY_POINT);
    env[pos..pos + 8].copy_from_slice(&ep.to_le_bytes());
    pos += 8;

    // [61-64] max_level
    env[pos..pos + 4].copy_from_slice(&snapshot.max_level.to_le_bytes());
    pos += 4;

    // [65] compression
    env[pos] = compression;
    pos += 1;

    // [66-73] uncompressed_size
    env[pos..pos + 8].copy_from_slice(&uncompressed_size.to_le_bytes());
    pos += 8;

    // [74-79] reserved (already zeroed)
    debug_assert_eq!(pos, 74);
    // pos += 6; // reserved zeros, already in the array

    env
}

/// Parse the 80-byte envelope from a byte slice.
fn parse_envelope(env: &[u8; ENVELOPE_SIZE]) -> Result<EnvelopeFields, IndexError> {
    let mut pos = 0;

    // [0-3] Magic
    if &env[pos..pos + 4] != TOPOLOGY_MAGIC {
        return Err(IndexError::Internal(
            "topology snapshot: invalid magic bytes".into(),
        ));
    }
    pos += 4;

    // [4-7] format_version
    let version = u32::from_le_bytes(env[pos..pos + 4].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(IndexError::Internal(format!(
            "topology snapshot: unsupported version {}",
            version
        )));
    }
    pos += 4;

    // [8-15] snapshot_lsn
    let snapshot_lsn = u64::from_le_bytes(env[pos..pos + 8].try_into().unwrap());
    pos += 8;

    // [16-23] timestamp
    let timestamp = u64::from_le_bytes(env[pos..pos + 8].try_into().unwrap());
    pos += 8;

    // [24-31] node_count
    let node_count = u64::from_le_bytes(env[pos..pos + 8].try_into().unwrap());
    pos += 8;

    // [32-35] dimension
    let dimension = u32::from_le_bytes(env[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // [36] metric
    let metric = env[pos];
    pos += 1;

    // [37-40] m
    let m = u32::from_le_bytes(env[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // [41-44] m0
    let m0 = u32::from_le_bytes(env[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // [45-48] ef_construction
    let ef_construction = u32::from_le_bytes(env[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // [49-52] ef_search
    let ef_search = u32::from_le_bytes(env[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // [53-60] entry_point
    let ep_raw = u64::from_le_bytes(env[pos..pos + 8].try_into().unwrap());
    let entry_point = if ep_raw == NO_ENTRY_POINT {
        None
    } else {
        Some(ep_raw)
    };
    pos += 8;

    // [61-64] max_level
    let max_level = u32::from_le_bytes(env[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // [65] compression
    let compression = env[pos];
    pos += 1;

    // [66-73] uncompressed_size
    let uncompressed_size = u64::from_le_bytes(env[pos..pos + 8].try_into().unwrap());
    // pos += 8;
    // [74-79] reserved, skip

    Ok(EnvelopeFields {
        snapshot_lsn,
        timestamp,
        node_count,
        dimension,
        metric,
        m,
        m0,
        ef_construction,
        ef_search,
        entry_point,
        max_level,
        compression,
        uncompressed_size,
    })
}

/// Parsed envelope fields.
struct EnvelopeFields {
    snapshot_lsn: u64,
    timestamp: u64,
    node_count: u64,
    dimension: u32,
    metric: u8,
    m: u32,
    m0: u32,
    ef_construction: u32,
    ef_search: u32,
    entry_point: Option<VectorId>,
    max_level: u32,
    compression: u8,
    uncompressed_size: u64,
}

/// Parse topology nodes from decompressed payload bytes.
fn parse_nodes(data: &[u8], node_count: u64) -> Result<Vec<TopologyNode>, IndexError> {
    // Sanity: each node is at minimum 24 bytes (id:8 + level:4 + slot:8 + neighbor_count:4).
    let min_bytes_per_node = 24u64;
    if node_count.saturating_mul(min_bytes_per_node) > data.len() as u64 {
        return Err(IndexError::Internal(
            "topology snapshot: node_count exceeds payload size".into(),
        ));
    }
    let mut nodes = Vec::with_capacity(node_count as usize);
    let mut pos = 0;

    let read_u32 = |pos: &mut usize| -> Result<u32, IndexError> {
        if *pos + 4 > data.len() {
            return Err(IndexError::Internal(
                "topology snapshot: unexpected end of payload (u32)".into(),
            ));
        }
        let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(val)
    };

    let read_u64 = |pos: &mut usize| -> Result<u64, IndexError> {
        if *pos + 8 > data.len() {
            return Err(IndexError::Internal(
                "topology snapshot: unexpected end of payload (u64)".into(),
            ));
        }
        let val = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        Ok(val)
    };

    for _ in 0..node_count {
        let id = read_u64(&mut pos)?;
        let level = read_u32(&mut pos)? as usize;
        let vector_slot = read_u64(&mut pos)? as usize;

        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            let count = read_u32(&mut pos)? as usize;
            // Guard against corrupt neighbor counts causing OOM.
            if count > 10_000 {
                return Err(IndexError::Internal(format!(
                    "topology snapshot: neighbor count {count} exceeds max 10000"
                )));
            }
            let mut layer_neighbors = Vec::with_capacity(count);
            for _ in 0..count {
                layer_neighbors.push(read_u64(&mut pos)?);
            }
            neighbors.push(layer_neighbors);
        }

        nodes.push(TopologyNode {
            id,
            level,
            vector_slot,
            neighbors,
        });
    }

    Ok(nodes)
}

// ── Public API ──────────────────────────────────────────────────────────

/// Serialize HNSW topology to a writer. Uses LZ4 compression.
pub fn serialize_topology(
    snapshot: &HnswTopologySnapshot,
    writer: &mut impl Write,
) -> Result<(), IndexError> {
    let wrap = |e: std::io::Error| IndexError::Internal(format!("topology write error: {}", e));

    // 1. Build the uncompressed payload.
    let payload = build_payload(snapshot);
    let uncompressed_size = payload.len() as u64;

    // 2. LZ4-compress the payload.
    let compressed = lz4_flex::compress_prepend_size(&payload);

    // 3. Build the 80-byte envelope.
    let envelope = build_envelope(snapshot, COMPRESSION_LZ4, uncompressed_size);

    // 4. Write envelope + compressed payload.
    writer.write_all(&envelope).map_err(wrap)?;
    writer.write_all(&compressed).map_err(wrap)?;

    // 5. Compute CRC32 over (envelope + compressed payload) and write footer.
    let mut hasher = Crc32Hasher::new();
    hasher.update(&envelope);
    hasher.update(&compressed);
    let crc = hasher.finalize();
    writer.write_all(&crc.to_le_bytes()).map_err(wrap)?;

    Ok(())
}

/// Deserialize HNSW topology from a reader. Validates magic, version, CRC.
pub fn deserialize_topology(reader: &mut impl Read) -> Result<HnswTopologySnapshot, IndexError> {
    let wrap = |e: std::io::Error| IndexError::Internal(format!("topology read error: {}", e));

    // 1. Read 80-byte envelope.
    let mut envelope = [0u8; ENVELOPE_SIZE];
    reader.read_exact(&mut envelope).map_err(wrap)?;

    // 2. Parse and validate envelope.
    let fields = parse_envelope(&envelope)?;

    // 3. Read the rest of the data (compressed payload + 4-byte CRC footer).
    let mut remaining = Vec::new();
    reader.read_to_end(&mut remaining).map_err(wrap)?;

    if remaining.len() < 4 {
        return Err(IndexError::Internal(
            "topology snapshot: file too short for CRC footer".into(),
        ));
    }

    // Split into compressed payload and CRC footer.
    let crc_offset = remaining.len() - 4;
    let compressed = &remaining[..crc_offset];
    let stored_crc = u32::from_le_bytes(remaining[crc_offset..].try_into().unwrap());

    // 4. Verify CRC over (envelope + compressed payload).
    let mut hasher = Crc32Hasher::new();
    hasher.update(&envelope);
    hasher.update(compressed);
    let computed_crc = hasher.finalize();

    if stored_crc != computed_crc {
        return Err(IndexError::Internal(format!(
            "topology snapshot: CRC mismatch (stored={:#010x}, computed={:#010x})",
            stored_crc, computed_crc
        )));
    }

    // 5. Decompress the payload.
    let decompressed = match fields.compression {
        COMPRESSION_NONE => compressed.to_vec(),
        COMPRESSION_LZ4 => lz4_flex::decompress_size_prepended(compressed).map_err(|e| {
            IndexError::Internal(format!("topology snapshot: LZ4 decompression failed: {}", e))
        })?,
        other => {
            return Err(IndexError::Internal(format!(
                "topology snapshot: unknown compression type {}",
                other
            )));
        }
    };

    // Validate uncompressed size matches.
    if decompressed.len() as u64 != fields.uncompressed_size {
        return Err(IndexError::Internal(format!(
            "topology snapshot: uncompressed size mismatch (expected={}, actual={})",
            fields.uncompressed_size,
            decompressed.len()
        )));
    }

    // 6. Parse topology nodes from decompressed bytes.
    let nodes = parse_nodes(&decompressed, fields.node_count)?;

    Ok(HnswTopologySnapshot {
        snapshot_lsn: fields.snapshot_lsn,
        timestamp: fields.timestamp,
        dimension: fields.dimension,
        metric: fields.metric,
        m: fields.m,
        m0: fields.m0,
        ef_construction: fields.ef_construction,
        ef_search: fields.ef_search,
        entry_point: fields.entry_point,
        max_level: fields.max_level,
        nodes,
    })
}

/// Deserialize HNSW topology using mmap for zero-copy file access.
/// Falls back to regular deserialize_topology if mmap fails.
pub fn deserialize_topology_mmap(path: &Path) -> Result<HnswTopologySnapshot, IndexError> {
    // Attempt to open and mmap the file; fall back on any I/O or mmap error.
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            log::warn!("mmap open failed, falling back to sequential read: {}", e);
            let mut f = std::fs::File::open(path)
                .map_err(|e| IndexError::Internal(format!("topology read error: {}", e)))?;
            return deserialize_topology(&mut f);
        }
    };

    // SAFETY: The file is opened read-only and we do not modify it while mapped.
    let mmap = match unsafe { Mmap::map(&file) } {
        Ok(m) => m,
        Err(e) => {
            log::warn!("mmap map failed, falling back to sequential read: {}", e);
            let mut f = std::fs::File::open(path)
                .map_err(|e| IndexError::Internal(format!("topology read error: {}", e)))?;
            return deserialize_topology(&mut f);
        }
    };

    let data = &mmap[..];
    let len = data.len();

    // Minimum: 80-byte envelope + 4-byte CRC footer.
    if len < ENVELOPE_SIZE + 4 {
        return Err(IndexError::Internal(
            "topology snapshot: file too small for envelope + CRC".into(),
        ));
    }

    // Parse the 80-byte envelope.
    let envelope: &[u8; ENVELOPE_SIZE] = data[..ENVELOPE_SIZE].try_into().unwrap();
    let fields = parse_envelope(envelope)?;

    // Read and verify CRC32 over everything except the trailing 4 bytes.
    let crc_offset = len - 4;
    let stored_crc = u32::from_le_bytes(data[crc_offset..].try_into().unwrap());

    let mut hasher = Crc32Hasher::new();
    hasher.update(&data[..crc_offset]);
    let computed_crc = hasher.finalize();

    if stored_crc != computed_crc {
        return Err(IndexError::Internal(format!(
            "topology snapshot: CRC mismatch (stored={:#010x}, computed={:#010x})",
            stored_crc, computed_crc
        )));
    }

    // Extract the compressed payload between envelope and CRC footer.
    let compressed = &data[ENVELOPE_SIZE..crc_offset];

    // Decompress the payload.
    let decompressed = match fields.compression {
        COMPRESSION_NONE => compressed.to_vec(),
        COMPRESSION_LZ4 => lz4_flex::decompress_size_prepended(compressed).map_err(|e| {
            IndexError::Internal(format!("topology snapshot: LZ4 decompression failed: {}", e))
        })?,
        other => {
            return Err(IndexError::Internal(format!(
                "topology snapshot: unknown compression type {}",
                other
            )));
        }
    };

    // Validate uncompressed size.
    if decompressed.len() as u64 != fields.uncompressed_size {
        return Err(IndexError::Internal(format!(
            "topology snapshot: uncompressed size mismatch (expected={}, actual={})",
            fields.uncompressed_size,
            decompressed.len()
        )));
    }

    // Parse topology nodes from decompressed bytes.
    let nodes = parse_nodes(&decompressed, fields.node_count)?;

    Ok(HnswTopologySnapshot {
        snapshot_lsn: fields.snapshot_lsn,
        timestamp: fields.timestamp,
        dimension: fields.dimension,
        metric: fields.metric,
        m: fields.m,
        m0: fields.m0,
        ef_construction: fields.ef_construction,
        ef_search: fields.ef_search,
        entry_point: fields.entry_point,
        max_level: fields.max_level,
        nodes,
    })
}

/// Quick validation: read header + verify CRC without full deserialization.
/// Returns (snapshot_lsn, node_count).
pub fn validate_base(path: &Path) -> Result<(u64, u64), IndexError> {
    let wrap = |e: std::io::Error| IndexError::Internal(format!("topology validate error: {}", e));

    let data = std::fs::read(path).map_err(wrap)?;

    if data.len() < ENVELOPE_SIZE + 4 {
        return Err(IndexError::Internal(
            "topology snapshot: file too small".into(),
        ));
    }

    // Parse envelope for magic/version validation and to extract lsn + node_count.
    let envelope: &[u8; ENVELOPE_SIZE] = data[..ENVELOPE_SIZE].try_into().unwrap();
    let fields = parse_envelope(envelope)?;

    // Verify CRC over everything except the last 4 bytes.
    let crc_offset = data.len() - 4;
    let stored_crc = u32::from_le_bytes(data[crc_offset..].try_into().unwrap());

    let mut hasher = Crc32Hasher::new();
    hasher.update(&data[..crc_offset]);
    let computed_crc = hasher.finalize();

    if stored_crc != computed_crc {
        return Err(IndexError::Internal(format!(
            "topology snapshot: CRC mismatch (stored={:#010x}, computed={:#010x})",
            stored_crc, computed_crc
        )));
    }

    Ok((fields.snapshot_lsn, fields.node_count))
}

/// Validate the on-disk topology envelope at `path` against `expected_dimension`.
///
/// Returns:
///   Ok(true) when the file exists, the envelope's magic and version are
///     recognised, the CRC32 footer verifies, and the dimension stored in
///     the envelope matches `expected_dimension`.
///   Ok(false) when the file is missing, truncated, has a bad magic, a bad
///     version, a bad CRC, or a dimension mismatch. These are all
///     soft-recoverable conditions; the caller falls back to a full rebuild.
///   Err(IndexError::Internal) for hard IO failures that should not be
///     silently swallowed (e.g., permission denied while opening a file).
///
/// This helper is intentionally cheap: it reads only the envelope and the
/// CRC footer, plus the bytes in between to recompute the CRC. It performs
/// no decompression and no node-by-node parsing.
pub(crate) fn validate_envelope_at_path(
    path: &Path,
    expected_dimension: usize,
) -> Result<bool, IndexError> {
    // Missing file is a soft failure.
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => {
            return Err(IndexError::Internal(format!(
                "topology validate: failed to read {}: {}",
                path.display(),
                e
            )));
        }
    };

    // Truncated below envelope plus CRC footer is a soft failure (corrupt).
    if data.len() < ENVELOPE_SIZE + 4 {
        return Ok(false);
    }

    // Soft-parse the envelope. Bad magic, bad version, etc. are soft failures.
    let envelope: &[u8; ENVELOPE_SIZE] = match data[..ENVELOPE_SIZE].try_into() {
        Ok(arr) => arr,
        Err(_) => return Ok(false),
    };
    let fields = match parse_envelope(envelope) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };

    // Dimension mismatch is a soft failure (cross-collection contamination).
    if fields.dimension as usize != expected_dimension {
        return Ok(false);
    }

    // CRC32 over (envelope plus compressed payload).
    let crc_offset = data.len() - 4;
    let stored_crc = match data[crc_offset..].try_into() {
        Ok(bytes) => u32::from_le_bytes(bytes),
        Err(_) => return Ok(false),
    };
    let mut hasher = Crc32Hasher::new();
    hasher.update(&data[..crc_offset]);
    let computed_crc = hasher.finalize();
    if stored_crc != computed_crc {
        return Ok(false);
    }

    Ok(true)
}
