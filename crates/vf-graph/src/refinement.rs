// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;

use vf_core::types::VectorId;
use vf_index::traits::VectorIndex;

use crate::compute::distance_to_similarity;
use crate::error::GraphError;
use crate::types::VirtualGraph;

#[derive(Clone, Debug)]
pub struct RefinementConfig {
    pub batch_size: usize,
    pub extra_neighbors: usize,
    pub min_improvement: f32,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            extra_neighbors: 20,
            min_improvement: 0.01,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RefinementStats {
    pub nodes_processed: usize,
    pub edges_added: usize,
    pub edges_updated: usize,
}

pub struct GraphRefiner {
    config: RefinementConfig,
    cursor: usize,
}

impl GraphRefiner {
    pub fn new(config: RefinementConfig) -> Self {
        Self { config, cursor: 0 }
    }

    pub fn with_defaults() -> Self {
        Self::new(RefinementConfig::default())
    }

    pub fn refine_step(
        &mut self,
        graph: &mut VirtualGraph,
        index: &dyn VectorIndex,
        vectors: &HashMap<VectorId, Vec<f32>>,
    ) -> Result<RefinementStats, GraphError> {
        let node_ids = graph.node_ids();
        let total = node_ids.len();
        if total == 0 {
            return Ok(RefinementStats::default());
        }

        let metric = graph.config().distance_metric;
        let max_edges = graph.config().max_edges_per_node;
        let search_k = max_edges + self.config.extra_neighbors;

        let effective_batch = self.config.batch_size.min(total);

        let batch: Vec<VectorId> = (0..effective_batch)
            .map(|i| node_ids[(self.cursor + i) % total])
            .collect();

        self.cursor = (self.cursor + effective_batch) % total;

        let mut stats = RefinementStats::default();

        for &node_id in &batch {
            let vector_data = match vectors.get(&node_id) {
                Some(v) => v,
                None => continue,
            };

            let results = index.search(vector_data, search_k)?;
            let threshold = graph.resolve_threshold(node_id, None);

            for result in results {
                if result.id == node_id {
                    continue;
                }

                let similarity = distance_to_similarity(result.score, metric);
                if similarity < threshold {
                    continue;
                }

                let existing_sim = graph
                    .get_node(node_id)
                    .and_then(|n| n.edges.iter().find(|e| e.target == result.id))
                    .map(|e| e.similarity);

                match existing_sim {
                    Some(old_sim) => {
                        if similarity - old_sim >= self.config.min_improvement {
                            graph.add_edge(node_id, result.id, similarity);
                            if let Some(node) = graph.get_node_mut(node_id) {
                                if let Some(edge) =
                                    node.edges.iter_mut().find(|e| e.target == result.id)
                                {
                                    edge.refined = true;
                                }
                            }
                            if let Some(target_node) = graph.get_node_mut(result.id) {
                                if let Some(edge) =
                                    target_node.edges.iter_mut().find(|e| e.target == node_id)
                                {
                                    edge.refined = true;
                                }
                            }
                            stats.edges_updated += 1;
                        }
                    }
                    None => {
                        graph.add_edge(node_id, result.id, similarity);
                        if let Some(node) = graph.get_node_mut(node_id) {
                            if let Some(edge) =
                                node.edges.iter_mut().find(|e| e.target == result.id)
                            {
                                edge.refined = true;
                            }
                        }
                        if let Some(target_node) = graph.get_node_mut(result.id) {
                            if let Some(edge) =
                                target_node.edges.iter_mut().find(|e| e.target == node_id)
                            {
                                edge.refined = true;
                            }
                        }
                        stats.edges_added += 1;
                    }
                }
            }

            stats.nodes_processed += 1;
        }

        Ok(stats)
    }

    pub fn is_pass_complete(&self, total_nodes: usize) -> bool {
        if total_nodes == 0 {
            return true;
        }
        self.cursor % total_nodes == 0 && self.cursor > 0
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}
