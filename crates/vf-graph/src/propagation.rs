// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use vf_core::types::VectorId;

use crate::error::GraphError;
use crate::types::VirtualGraph;

pub struct ThresholdPropagator;

impl ThresholdPropagator {
    pub fn update_collection_threshold(graph: &mut VirtualGraph, new_threshold: f32) -> usize {
        graph.config_mut().default_threshold = new_threshold;

        let mut to_remove: Vec<(VectorId, VectorId)> = Vec::new();

        for (&node_id, node) in graph.nodes() {
            if node.threshold_override.is_some() {
                continue;
            }
            for edge in &node.edges {
                if edge.similarity < new_threshold && node_id < edge.target {
                    to_remove.push((node_id, edge.target));
                }
            }
        }

        let count = to_remove.len();
        for (from, to) in to_remove {
            graph.remove_edge(from, to);
        }

        count
    }

    pub fn update_vector_threshold(
        graph: &mut VirtualGraph,
        vector_id: VectorId,
        new_threshold: f32,
    ) -> Result<usize, GraphError> {
        if graph.get_node(vector_id).is_none() {
            return Err(GraphError::NodeNotFound(vector_id));
        }

        graph.set_vector_threshold(vector_id, new_threshold);

        let to_remove: Vec<VectorId> = graph
            .get_node(vector_id)
            .map(|node| {
                node.edges
                    .iter()
                    .filter(|e| e.similarity < new_threshold)
                    .map(|e| e.target)
                    .collect()
            })
            .unwrap_or_default();

        let count = to_remove.len();
        for target in to_remove {
            graph.remove_edge(vector_id, target);
        }

        Ok(count)
    }

    pub fn clear_vector_threshold(
        graph: &mut VirtualGraph,
        vector_id: VectorId,
    ) -> Result<usize, GraphError> {
        if graph.get_node(vector_id).is_none() {
            return Err(GraphError::NodeNotFound(vector_id));
        }

        graph.clear_vector_threshold(vector_id);

        let default_threshold = graph.config().default_threshold;

        let to_remove: Vec<VectorId> = graph
            .get_node(vector_id)
            .map(|node| {
                node.edges
                    .iter()
                    .filter(|e| e.similarity < default_threshold)
                    .map(|e| e.target)
                    .collect()
            })
            .unwrap_or_default();

        let count = to_remove.len();
        for target in to_remove {
            graph.remove_edge(vector_id, target);
        }

        Ok(count)
    }

    pub fn prune_all(graph: &mut VirtualGraph) -> usize {
        let mut to_remove: Vec<(VectorId, VectorId)> = Vec::new();

        for (&node_id, node) in graph.nodes() {
            let threshold = graph.resolve_threshold(node_id, None);
            for edge in &node.edges {
                if edge.similarity < threshold && node_id < edge.target {
                    to_remove.push((node_id, edge.target));
                }
            }
        }

        let count = to_remove.len();
        for (from, to) in to_remove {
            graph.remove_edge(from, to);
        }

        count
    }
}
