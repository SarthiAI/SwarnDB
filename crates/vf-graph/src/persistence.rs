// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use std::io::{Read as IoRead, Write as IoWrite};

use crc32fast::Hasher as Crc32Hasher;
use vf_core::types::DistanceMetricType;

use crate::error::GraphError;
use crate::types::{Edge, GraphConfig, VirtualGraph};

const MAGIC: &[u8; 4] = b"VGRF";
const VERSION: u32 = 2;
const RESERVED_BYTES: usize = 16;

/// Maximum number of nodes allowed during deserialization to prevent
/// unbounded memory allocation from malicious or corrupted graph files.
const MAX_GRAPH_NODES: u64 = 50_000_000;

/// Maximum number of edges per node allowed during deserialization.
const MAX_GRAPH_EDGES: u64 = 10_000_000_000;

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
        _ => Err(GraphError::Internal(format!("unknown distance metric: {v}"))),
    }
}

pub struct GraphPersistence;

impl GraphPersistence {
    pub fn serialize(graph: &VirtualGraph, writer: &mut impl IoWrite) -> Result<(), GraphError> {
        let config = graph.config();
        let nodes = graph.nodes();

        // Write all data to a buffer first so we can compute CRC32
        let mut buf: Vec<u8> = Vec::new();

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(nodes.len() as u64).to_le_bytes());

        // Config
        buf.extend_from_slice(&config.default_threshold.to_le_bytes());
        buf.extend_from_slice(&(config.max_edges_per_node as u32).to_le_bytes());
        buf.extend_from_slice(&(config.max_traversal_depth as u32).to_le_bytes());
        buf.push(metric_to_u8(config.distance_metric));

        // Reserved
        buf.extend_from_slice(&[0u8; RESERVED_BYTES]);

        // Nodes
        for (&id, node) in nodes {
            buf.extend_from_slice(&id.to_le_bytes());

            match node.threshold_override {
                Some(t) => {
                    buf.push(1u8);
                    buf.extend_from_slice(&t.to_le_bytes());
                }
                None => {
                    buf.push(0u8);
                }
            }

            buf.extend_from_slice(&(node.edges.len() as u64).to_le_bytes());

            for edge in &node.edges {
                buf.extend_from_slice(&edge.target.to_le_bytes());
                buf.extend_from_slice(&edge.similarity.to_le_bytes());
                buf.extend_from_slice(&edge.created_at.to_le_bytes());
                buf.push(edge.refined as u8);
            }
        }

        // Compute and append CRC32 checksum
        let mut hasher = Crc32Hasher::new();
        hasher.update(&buf);
        let checksum = hasher.finalize();
        buf.extend_from_slice(&checksum.to_le_bytes());

        writer.write_all(&buf).map_err(io_err)?;

        Ok(())
    }

    pub fn deserialize(reader: &mut impl IoRead) -> Result<VirtualGraph, GraphError> {
        // Read all bytes and verify CRC32 checksum
        let mut all_bytes = Vec::new();
        reader.read_to_end(&mut all_bytes).map_err(io_err)?;

        if all_bytes.len() < 4 {
            return Err(GraphError::Internal("data too short for CRC32 checksum".into()));
        }

        let data_len = all_bytes.len() - 4;
        let data = &all_bytes[..data_len];
        let stored_checksum = u32::from_le_bytes([
            all_bytes[data_len],
            all_bytes[data_len + 1],
            all_bytes[data_len + 2],
            all_bytes[data_len + 3],
        ]);

        let mut hasher = Crc32Hasher::new();
        hasher.update(data);
        let computed_checksum = hasher.finalize();

        if stored_checksum != computed_checksum {
            return Err(GraphError::Internal(format!(
                "CRC32 checksum mismatch: stored={stored_checksum:#010x}, computed={computed_checksum:#010x}"
            )));
        }

        // Now parse the verified data using a cursor
        let mut reader = std::io::Cursor::new(data);

        // Header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(io_err)?;
        if &magic != MAGIC {
            return Err(GraphError::Internal("invalid magic bytes".into()));
        }

        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4).map_err(io_err)?;
        let version = u32::from_le_bytes(buf4);
        if version != VERSION {
            return Err(GraphError::Internal(format!(
                "unsupported version: {version}"
            )));
        }

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8).map_err(io_err)?;
        let node_count_u64 = u64::from_le_bytes(buf8);
        if node_count_u64 > MAX_GRAPH_NODES {
            return Err(GraphError::Internal(format!(
                "node count {} exceeds maximum allowed {}",
                node_count_u64, MAX_GRAPH_NODES
            )));
        }
        let node_count = node_count_u64 as usize;

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
        let mut reserved = [0u8; RESERVED_BYTES];
        reader.read_exact(&mut reserved).map_err(io_err)?;

        // Validate deserialized config values
        if max_edges_per_node == 0 {
            return Err(GraphError::Internal("max_edges_per_node must be > 0".into()));
        }
        if max_traversal_depth == 0 {
            return Err(GraphError::Internal("max_traversal_depth must be > 0".into()));
        }
        if default_threshold.is_nan() || default_threshold.is_infinite() {
            return Err(GraphError::Internal("default_threshold must be a finite number".into()));
        }
        if !(0.0..=1.0).contains(&default_threshold) {
            return Err(GraphError::Internal(format!(
                "default_threshold must be in [0.0, 1.0], got {default_threshold}"
            )));
        }

        let config = GraphConfig {
            default_threshold,
            max_edges_per_node,
            max_traversal_depth,
            distance_metric,
            ..GraphConfig::default()
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

            reader.read_exact(&mut buf8).map_err(io_err)?;
            let edge_count_u64 = u64::from_le_bytes(buf8);
            if edge_count_u64 > MAX_GRAPH_EDGES {
                return Err(GraphError::Internal(format!(
                    "edge count {} exceeds maximum allowed {}",
                    edge_count_u64, MAX_GRAPH_EDGES
                )));
            }
            let edge_count = edge_count_u64 as usize;

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
