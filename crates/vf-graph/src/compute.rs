// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use std::collections::HashMap;

use vf_core::types::{DistanceMetricType, VectorId};
use vf_index::traits::VectorIndex;

use crate::error::GraphError;
use crate::types::VirtualGraph;

pub(crate) fn distance_to_similarity(distance: f32, metric: DistanceMetricType) -> f32 {
    match metric {
        DistanceMetricType::Cosine => 1.0 - distance,
        DistanceMetricType::Euclidean => 1.0 / (1.0 + distance),
        DistanceMetricType::DotProduct => -distance,
        DistanceMetricType::Manhattan => 1.0 / (1.0 + distance),
    }
}

pub struct RelationshipComputer;

impl RelationshipComputer {
    pub fn compute_for_vector(
        graph: &mut VirtualGraph,
        index: &dyn VectorIndex,
        vector_id: VectorId,
        vector_data: &[f32],
        k: usize,
    ) -> Result<usize, GraphError> {
        graph.add_node(vector_id);
        let metric = graph.config().distance_metric;
        let threshold = graph.resolve_threshold(vector_id, None);
        let max_edges = graph.config().max_edges_per_node;

        let results = index.search(vector_data, k, None)?;

        let mut added = 0usize;
        for result in results {
            if result.id == vector_id {
                continue;
            }

            let similarity = distance_to_similarity(result.score, metric);
            if similarity < threshold {
                continue;
            }

            let existing = graph
                .get_node(vector_id)
                .and_then(|node| node.edges.iter().find(|e| e.target == result.id))
                .map(|e| e.similarity);

            let at_capacity = graph
                .get_node(vector_id)
                .map(|n| n.edges.len() >= max_edges)
                .unwrap_or(false);

            if at_capacity {
                let min_sim = graph
                    .get_node(vector_id)
                    .and_then(|n| n.edges.last())
                    .map(|e| e.similarity);
                if let Some(min) = min_sim {
                    if similarity <= min && existing.is_none() {
                        continue;
                    }
                }
            }

            let is_new = existing.is_none();
            graph.add_edge(vector_id, result.id, similarity);
            if is_new {
                added += 1;
            }
        }

        Ok(added)
    }

    pub fn compute_all(
        graph: &mut VirtualGraph,
        index: &dyn VectorIndex,
        vectors: &HashMap<VectorId, Vec<f32>>,
        k: usize,
    ) -> Result<usize, GraphError> {
        let ids: Vec<VectorId> = graph.node_ids();
        let mut total = 0usize;

        for id in ids {
            if let Some(data) = vectors.get(&id) {
                total += Self::compute_for_vector(graph, index, id, data, k)?;
            }
        }

        Ok(total)
    }

    pub fn compute_batch(
        graph: &mut VirtualGraph,
        index: &dyn VectorIndex,
        vector_ids: &[VectorId],
        vectors: &HashMap<VectorId, Vec<f32>>,
        k: usize,
    ) -> Result<usize, GraphError> {
        let mut total = 0usize;

        for &id in vector_ids {
            if let Some(data) = vectors.get(&id) {
                total += Self::compute_for_vector(graph, index, id, data, k)?;
            }
        }

        if total == 0 && !vector_ids.is_empty() {
            log::warn!("compute_batch produced 0 edges for {} vectors — check threshold", vector_ids.len());
        }

        Ok(total)
    }
}
