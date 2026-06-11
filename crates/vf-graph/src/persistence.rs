// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crc32fast::Hasher as Crc32Hasher;
use vf_core::types::DistanceMetricType;

use crate::error::GraphError;
use crate::types::{Edge, GraphConfig, VirtualGraph};

// ── V1 legacy constants ──────────────────────────────────────────────
const MAGIC: &[u8; 4] = b"VGRF";
const LEGACY_VERSION: u32 = 1;
const LEGACY_RESERVED_BYTES: usize = 16;

// ── V2 base snapshot constants ───────────────────────────────────────
const BASE_FORMAT_VERSION: u32 = 2;
const ENVELOPE_SIZE: usize = 72;
const COMPRESSION_NONE: u8 = 0;
const COMPRESSION_LZ4: u8 = 1;

// ── Shared helpers ───────────────────────────────────────────────────

fn io_err(e: std::io::Error) -> GraphError {
    GraphError::Internal(e.to_string())
}

fn metric_to_u8(m: DistanceMetricType) -> u8 {
    match m {
        DistanceMetricType::Cosine => 0,
        DistanceMetricType::Euclidean => 1,
        DistanceMetricType::DotProduct => 2,
        DistanceMetricType::Manhattan => 3,
    }
}

fn u8_to_metric(v: u8) -> Result<DistanceMetricType, GraphError> {
    match v {
        0 => Ok(DistanceMetricType::Cosine),
        1 => Ok(DistanceMetricType::Euclidean),
        2 => Ok(DistanceMetricType::DotProduct),
        3 => Ok(DistanceMetricType::Manhattan),
        _ => Err(GraphError::InvalidFormat(format!(
            "unknown distance metric: {v}"
        ))),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// V1 LEGACY PERSISTENCE (deprecated)
// ═══════════════════════════════════════════════════════════════════════

#[deprecated(since = "0.2.0", note = "use serialize_base / deserialize_base instead")]
pub struct GraphPersistence;

#[allow(deprecated)]
impl GraphPersistence {
    #[deprecated(
        since = "0.2.0",
        note = "use serialize_base() instead"
    )]
    pub fn serialize(graph: &VirtualGraph, writer: &mut impl IoWrite) -> Result<(), GraphError> {
        let config = graph.config();
        let nodes = graph.nodes();

        // Header
        writer.write_all(MAGIC).map_err(io_err)?;
        writer
            .write_all(&LEGACY_VERSION.to_le_bytes())
            .map_err(io_err)?;
        writer
            .write_all(&(nodes.len() as u64).to_le_bytes())
            .map_err(io_err)?;

        // Config
        writer
            .write_all(&config.default_threshold.to_le_bytes())
            .map_err(io_err)?;
        writer
            .write_all(&(config.max_edges_per_node as u32).to_le_bytes())
            .map_err(io_err)?;
        writer
            .write_all(&(config.max_traversal_depth as u32).to_le_bytes())
            .map_err(io_err)?;
        writer
            .write_all(&[metric_to_u8(config.distance_metric)])
            .map_err(io_err)?;

        // Reserved
        writer
            .write_all(&[0u8; LEGACY_RESERVED_BYTES])
            .map_err(io_err)?;

        // Nodes
        for (&id, node) in nodes {
            writer.write_all(&id.to_le_bytes()).map_err(io_err)?;

            match node.threshold_override {
                Some(t) => {
                    writer.write_all(&[1u8]).map_err(io_err)?;
                    writer.write_all(&t.to_le_bytes()).map_err(io_err)?;
                }
                None => {
                    writer.write_all(&[0u8]).map_err(io_err)?;
                }
            }

            writer
                .write_all(&(node.edges.len() as u32).to_le_bytes())
                .map_err(io_err)?;

            for edge in &node.edges {
                writer
                    .write_all(&edge.target.to_le_bytes())
                    .map_err(io_err)?;
                writer
                    .write_all(&edge.similarity.to_le_bytes())
                    .map_err(io_err)?;
                writer
                    .write_all(&edge.created_at.to_le_bytes())
                    .map_err(io_err)?;
                writer
                    .write_all(&[edge.refined as u8])
                    .map_err(io_err)?;
            }
        }

        Ok(())
    }

    #[deprecated(
        since = "0.2.0",
        note = "use deserialize_base() instead"
    )]
    pub fn deserialize(reader: &mut impl IoRead) -> Result<VirtualGraph, GraphError> {
        // Header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(io_err)?;
        if &magic != MAGIC {
            return Err(GraphError::Internal("invalid magic bytes".into()));
        }

        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4).map_err(io_err)?;
        let version = u32::from_le_bytes(buf4);
        if version != LEGACY_VERSION {
            return Err(GraphError::Internal(format!(
                "unsupported version: {version}"
            )));
        }

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8).map_err(io_err)?;
        let node_count = u64::from_le_bytes(buf8) as usize;

        // Config
        reader.read_exact(&mut buf4).map_err(io_err)?;
        let default_threshold = f32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4).map_err(io_err)?;
        let max_edges_per_node = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4).map_err(io_err)?;
        let max_traversal_depth = u32::from_le_bytes(buf4) as usize;

        let mut metric_byte = [0u8; 1];
        reader.read_exact(&mut metric_byte).map_err(io_err)?;
        let distance_metric = u8_to_metric(metric_byte[0])?;

        // Reserved
        let mut reserved = [0u8; LEGACY_RESERVED_BYTES];
        reader.read_exact(&mut reserved).map_err(io_err)?;

        let config = GraphConfig {
            default_threshold,
            max_edges_per_node,
            max_traversal_depth,
            distance_metric,
        };

        let mut graph = VirtualGraph::new(config);

        // Nodes
        for _ in 0..node_count {
            reader.read_exact(&mut buf8).map_err(io_err)?;
            let id = u64::from_le_bytes(buf8);

            reader.read_exact(&mut metric_byte).map_err(io_err)?;
            let threshold_override = if metric_byte[0] == 1 {
                reader.read_exact(&mut buf4).map_err(io_err)?;
                Some(f32::from_le_bytes(buf4))
            } else {
                None
            };

            reader.read_exact(&mut buf4).map_err(io_err)?;
            let edge_count = u32::from_le_bytes(buf4) as usize;

            let mut edges = Vec::with_capacity(edge_count);
            for _ in 0..edge_count {
                reader.read_exact(&mut buf8).map_err(io_err)?;
                let target = u64::from_le_bytes(buf8);

                reader.read_exact(&mut buf4).map_err(io_err)?;
                let similarity = f32::from_le_bytes(buf4);

                reader.read_exact(&mut buf8).map_err(io_err)?;
                let created_at = u64::from_le_bytes(buf8);

                let mut refined_byte = [0u8; 1];
                reader.read_exact(&mut refined_byte).map_err(io_err)?;
                let refined = refined_byte[0] != 0;

                edges.push(Edge {
                    target,
                    similarity,
                    created_at,
                    refined,
                });
            }

            graph.add_node(id);
            if let Some(node) = graph.get_node_mut(id) {
                node.edges = edges;
                node.threshold_override = threshold_override;
            }
        }

        Ok(graph)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// V2 BASE SNAPSHOT PERSISTENCE
// ═══════════════════════════════════════════════════════════════════════

/// Counts total edges across all nodes (including both directions).
fn total_directed_edge_count(graph: &VirtualGraph) -> u64 {
    graph.nodes().values().map(|n| n.edges.len() as u64).sum()
}

/// Build the raw (uncompressed) node+edge payload.
fn build_payload(graph: &VirtualGraph) -> Vec<u8> {
    let nodes = graph.nodes();
    // Rough estimate: 13 bytes base per node + 21 bytes per edge
    let capacity = nodes.len() * 13 + total_directed_edge_count(graph) as usize * 21;
    let mut buf = Vec::with_capacity(capacity);

    for (&node_id, node) in nodes {
        buf.extend_from_slice(&node_id.to_le_bytes()); // 8 bytes

        match node.threshold_override {
            Some(t) => {
                buf.push(1u8);
                buf.extend_from_slice(&t.to_le_bytes()); // 4 bytes
            }
            None => {
                buf.push(0u8);
            }
        }

        buf.extend_from_slice(&(node.edges.len() as u32).to_le_bytes()); // 4 bytes

        for edge in &node.edges {
            buf.extend_from_slice(&edge.target.to_le_bytes()); // 8 bytes
            buf.extend_from_slice(&edge.similarity.to_le_bytes()); // 4 bytes
            buf.extend_from_slice(&edge.created_at.to_le_bytes()); // 8 bytes
            buf.push(edge.refined as u8); // 1 byte
        }
    }

    buf
}

/// Build the 72-byte envelope.
fn build_envelope(
    graph: &VirtualGraph,
    snapshot_lsn: u64,
    uncompressed_size: u64,
    compression: u8,
) -> [u8; ENVELOPE_SIZE] {
    let config = graph.config();
    let mut env = [0u8; ENVELOPE_SIZE];

    // [0-3] Magic
    env[0..4].copy_from_slice(MAGIC);
    // [4-7] format_version
    env[4..8].copy_from_slice(&BASE_FORMAT_VERSION.to_le_bytes());
    // [8-15] snapshot_lsn
    env[8..16].copy_from_slice(&snapshot_lsn.to_le_bytes());
    // [16-23] timestamp (unix millis)
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    env[16..24].copy_from_slice(&ts.to_le_bytes());
    // [24-31] node_count
    env[24..32].copy_from_slice(&(graph.node_count() as u64).to_le_bytes());
    // [32-39] edge_count (total directed edges)
    env[32..40].copy_from_slice(&total_directed_edge_count(graph).to_le_bytes());
    // [40-43] default_threshold
    env[40..44].copy_from_slice(&config.default_threshold.to_le_bytes());
    // [44-47] max_edges_per_node
    env[44..48].copy_from_slice(&(config.max_edges_per_node as u32).to_le_bytes());
    // [48-51] max_traversal_depth
    env[48..52].copy_from_slice(&(config.max_traversal_depth as u32).to_le_bytes());
    // [52] distance_metric
    env[52] = metric_to_u8(config.distance_metric);
    // [53] compression
    env[53] = compression;
    // [54-61] uncompressed_size
    env[54..62].copy_from_slice(&uncompressed_size.to_le_bytes());
    // [62-71] reserved (already zeros)

    env
}

/// Serialize a VirtualGraph to the base snapshot format (v2).
pub fn serialize_base(
    graph: &VirtualGraph,
    snapshot_lsn: u64,
    writer: &mut impl IoWrite,
) -> Result<(), GraphError> {
    // 1. Build raw payload
    let raw_payload = build_payload(graph);
    let uncompressed_size = raw_payload.len() as u64;

    // 2. LZ4-compress
    let compressed = lz4_flex::compress_prepend_size(&raw_payload);

    // 3. Build envelope
    let envelope = build_envelope(graph, snapshot_lsn, uncompressed_size, COMPRESSION_LZ4);

    // 4. Compute CRC32 over envelope + compressed payload
    let mut crc = Crc32Hasher::new();
    crc.update(&envelope);
    crc.update(&compressed);
    let checksum = crc.finalize();

    // 5. Write envelope, compressed payload, CRC32 footer
    writer.write_all(&envelope)?;
    writer.write_all(&compressed)?;
    writer.write_all(&checksum.to_le_bytes())?;

    Ok(())
}

/// Deserialize a base snapshot into a VirtualGraph.
/// Returns `(snapshot_lsn, graph)`.
pub fn deserialize_base(reader: &mut impl IoRead) -> Result<(u64, VirtualGraph), GraphError> {
    // 1. Read envelope
    let mut envelope = [0u8; ENVELOPE_SIZE];
    reader.read_exact(&mut envelope)?;

    // Validate magic
    if &envelope[0..4] != MAGIC {
        return Err(GraphError::InvalidFormat(
            "invalid magic bytes, expected VGRF".into(),
        ));
    }

    // Validate version. v2 is the similarity-graph base parsed here. v3 is the
    // typed-graph base (ADR-007 R4): recognise it and route the caller to
    // deserialize_typed_base rather than failing as "unknown version". Reject
    // anything that is neither v2 nor v3.
    let version = u32::from_le_bytes(envelope[4..8].try_into().unwrap());
    if version == 3 {
        return Err(GraphError::InvalidFormat(
            "v3 typed graph base; load via typed_persistence::deserialize_typed_base".into(),
        ));
    }
    if version != BASE_FORMAT_VERSION {
        return Err(GraphError::InvalidFormat(format!(
            "unsupported format version: {version}, expected 2 (v2) or 3 (v3 typed)"
        )));
    }

    let snapshot_lsn = u64::from_le_bytes(envelope[8..16].try_into().unwrap());
    let node_count = u64::from_le_bytes(envelope[24..32].try_into().unwrap()) as usize;
    let _edge_count = u64::from_le_bytes(envelope[32..40].try_into().unwrap());
    let default_threshold = f32::from_le_bytes(envelope[40..44].try_into().unwrap());
    let max_edges_per_node =
        u32::from_le_bytes(envelope[44..48].try_into().unwrap()) as usize;
    let max_traversal_depth =
        u32::from_le_bytes(envelope[48..52].try_into().unwrap()) as usize;
    let distance_metric = u8_to_metric(envelope[52])?;
    let compression = envelope[53];
    let _uncompressed_size = u64::from_le_bytes(envelope[54..62].try_into().unwrap());

    // 2. Read remaining data (compressed payload + 4-byte CRC footer)
    let mut rest = Vec::new();
    reader.read_to_end(&mut rest)?;

    if rest.len() < 4 {
        return Err(GraphError::Corrupted(
            "file too short, missing CRC32 footer".into(),
        ));
    }

    let crc_offset = rest.len() - 4;
    let compressed_payload = &rest[..crc_offset];
    let stored_crc =
        u32::from_le_bytes(rest[crc_offset..].try_into().unwrap());

    // 3. Verify CRC32
    let mut crc = Crc32Hasher::new();
    crc.update(&envelope);
    crc.update(compressed_payload);
    let computed_crc = crc.finalize();

    if stored_crc != computed_crc {
        return Err(GraphError::Corrupted(format!(
            "CRC32 mismatch: stored={stored_crc:#010x}, computed={computed_crc:#010x}"
        )));
    }

    // 4. Decompress
    let raw_payload = match compression {
        COMPRESSION_NONE => compressed_payload.to_vec(),
        COMPRESSION_LZ4 => lz4_flex::decompress_size_prepended(compressed_payload).map_err(
            |e| GraphError::Corrupted(format!("LZ4 decompression failed: {e}")),
        )?,
        other => {
            return Err(GraphError::InvalidFormat(format!(
                "unknown compression type: {other}"
            )));
        }
    };

    // Validate uncompressed size.
    if raw_payload.len() as u64 != _uncompressed_size {
        return Err(GraphError::Corrupted(format!(
            "uncompressed size mismatch: envelope says {_uncompressed_size}, got {}",
            raw_payload.len()
        )));
    }

    // Sanity: each node is at minimum 9 bytes (id:8 + has_override:1).
    if node_count > raw_payload.len() {
        return Err(GraphError::Corrupted(
            "node_count exceeds payload size".into(),
        ));
    }

    // 5. Parse nodes and edges from raw payload
    let config = GraphConfig {
        default_threshold,
        max_edges_per_node,
        max_traversal_depth,
        distance_metric,
    };

    let mut graph = VirtualGraph::new(config);
    let mut cursor = 0usize;
    let data = &raw_payload;

    for _ in 0..node_count {
        // node_id: u64
        let node_id = read_u64(data, &mut cursor)?;

        // has_threshold_override: u8
        let has_override = read_u8(data, &mut cursor)?;
        let threshold_override = if has_override == 1 {
            Some(read_f32(data, &mut cursor)?)
        } else {
            None
        };

        // edge_count: u32
        let edge_count = read_u32(data, &mut cursor)? as usize;

        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let target = read_u64(data, &mut cursor)?;
            let similarity = read_f32(data, &mut cursor)?;
            let created_at = read_u64(data, &mut cursor)?;
            let refined = read_u8(data, &mut cursor)? != 0;

            edges.push(Edge {
                target,
                similarity,
                created_at,
                refined,
            });
        }

        graph.add_node(node_id);
        if let Some(node) = graph.get_node_mut(node_id) {
            node.edges = edges;
            node.threshold_override = threshold_override;
        }
    }

    Ok((snapshot_lsn, graph))
}

/// Quick validation: read header + verify CRC without full deserialization.
/// Returns `(snapshot_lsn, node_count, edge_count)`.
pub fn validate_graph_base(path: &Path) -> Result<(u64, u64, u64), GraphError> {
    let data = std::fs::read(path)?;

    if data.len() < ENVELOPE_SIZE + 4 {
        return Err(GraphError::Corrupted(
            "file too short for base snapshot".into(),
        ));
    }

    // Validate magic
    if &data[0..4] != MAGIC {
        return Err(GraphError::InvalidFormat(
            "invalid magic bytes, expected VGRF".into(),
        ));
    }

    // Validate version. Accept v2 (similarity base) and v3 (typed base); both
    // share this envelope layout for the header fields read below (ADR-007 R4).
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != BASE_FORMAT_VERSION && version != 3 {
        return Err(GraphError::InvalidFormat(format!(
            "unsupported format version: {version}, expected 2 (v2) or 3 (v3 typed)"
        )));
    }

    let snapshot_lsn = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let node_count = u64::from_le_bytes(data[24..32].try_into().unwrap());
    let edge_count = u64::from_le_bytes(data[32..40].try_into().unwrap());

    // Verify CRC32
    let crc_offset = data.len() - 4;
    let stored_crc =
        u32::from_le_bytes(data[crc_offset..].try_into().unwrap());

    let mut crc = Crc32Hasher::new();
    crc.update(&data[..crc_offset]);
    let computed_crc = crc.finalize();

    if stored_crc != computed_crc {
        return Err(GraphError::Corrupted(format!(
            "CRC32 mismatch: stored={stored_crc:#010x}, computed={computed_crc:#010x}"
        )));
    }

    Ok((snapshot_lsn, node_count, edge_count))
}

// ── Payload cursor helpers ───────────────────────────────────────────

fn ensure_remaining(data: &[u8], cursor: usize, need: usize) -> Result<(), GraphError> {
    if cursor + need > data.len() {
        return Err(GraphError::Corrupted(format!(
            "unexpected end of payload at offset {cursor}, need {need} bytes but only {} remain",
            data.len() - cursor
        )));
    }
    Ok(())
}

fn read_u8(data: &[u8], cursor: &mut usize) -> Result<u8, GraphError> {
    ensure_remaining(data, *cursor, 1)?;
    let v = data[*cursor];
    *cursor += 1;
    Ok(v)
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32, GraphError> {
    ensure_remaining(data, *cursor, 4)?;
    let v = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64, GraphError> {
    ensure_remaining(data, *cursor, 8)?;
    let v = u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_f32(data: &[u8], cursor: &mut usize) -> Result<f32, GraphError> {
    ensure_remaining(data, *cursor, 4)?;
    let v = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}
