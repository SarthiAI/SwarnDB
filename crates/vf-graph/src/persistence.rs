// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::io::{Read as IoRead, Write as IoWrite};

use vf_core::types::DistanceMetricType;

use crate::error::GraphError;
use crate::types::{Edge, GraphConfig, VirtualGraph};

const MAGIC: &[u8; 4] = b"VGRF";
const VERSION: u32 = 1;
const RESERVED_BYTES: usize = 16;

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

        // Header
        writer.write_all(MAGIC).map_err(io_err)?;
        writer.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
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
        writer.write_all(&[0u8; RESERVED_BYTES]).map_err(io_err)?;

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
                writer.write_all(&edge.target.to_le_bytes()).map_err(io_err)?;
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
        if version != VERSION {
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
        let mut reserved = [0u8; RESERVED_BYTES];
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
