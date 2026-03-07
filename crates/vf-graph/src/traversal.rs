// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::{HashSet, VecDeque};
use vf_core::types::VectorId;

use crate::error::GraphError;
use crate::types::VirtualGraph;

#[derive(Clone, Debug)]
pub enum TraversalOrder {
    BreadthFirst,
    DepthFirst,
}

#[derive(Clone, Debug)]
pub struct TraversalResult {
    pub id: VectorId,
    pub depth: usize,
    pub path_similarity: f32,
    pub path: Vec<VectorId>,
}

pub struct GraphTraversal;

impl GraphTraversal {
    pub fn traverse(
        graph: &VirtualGraph,
        start: VectorId,
        order: &TraversalOrder,
        max_depth: usize,
        threshold: Option<f32>,
        max_results: Option<usize>,
    ) -> Result<Vec<TraversalResult>, GraphError> {
        if !graph.contains(start) {
            return Err(GraphError::NodeNotFound(start));
        }

        let effective_max_depth = max_depth.min(graph.config().max_traversal_depth);

        let mut results = match order {
            TraversalOrder::BreadthFirst => {
                Self::bfs(graph, start, effective_max_depth, threshold)?
            }
            TraversalOrder::DepthFirst => {
                Self::dfs(graph, start, effective_max_depth, threshold)?
            }
        };

        results.sort_by(|a, b| {
            b.path_similarity
                .partial_cmp(&a.path_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(max) = max_results {
            results.truncate(max);
        }

        Ok(results)
    }

    fn bfs(
        graph: &VirtualGraph,
        start: VectorId,
        max_depth: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<TraversalResult>, GraphError> {
        let mut visited = HashSet::new();
        visited.insert(start);

        let mut results = Vec::new();

        // (node_id, depth, path_similarity, path)
        let mut queue: VecDeque<(VectorId, usize, f32, Vec<VectorId>)> = VecDeque::new();
        queue.push_back((start, 0, 1.0, vec![start]));

        while let Some((current, depth, path_sim, path)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let effective_threshold = graph.resolve_threshold(current, threshold);

            if let Some(node) = graph.get_node(current) {
                let edges = node.edges_above_threshold(effective_threshold);
                for edge in edges {
                    if visited.contains(&edge.target) {
                        continue;
                    }
                    visited.insert(edge.target);

                    let new_sim = path_sim * edge.similarity;
                    let mut new_path = path.clone();
                    new_path.push(edge.target);
                    let new_depth = depth + 1;

                    results.push(TraversalResult {
                        id: edge.target,
                        depth: new_depth,
                        path_similarity: new_sim,
                        path: new_path.clone(),
                    });

                    queue.push_back((edge.target, new_depth, new_sim, new_path));
                }
            }
        }

        Ok(results)
    }

    fn dfs(
        graph: &VirtualGraph,
        start: VectorId,
        max_depth: usize,
        threshold: Option<f32>,
    ) -> Result<Vec<TraversalResult>, GraphError> {
        let mut visited = HashSet::new();
        visited.insert(start);

        let mut results = Vec::new();

        // (node_id, depth, path_similarity, path)
        let mut stack: Vec<(VectorId, usize, f32, Vec<VectorId>)> = Vec::new();
        stack.push((start, 0, 1.0, vec![start]));

        while let Some((current, depth, path_sim, path)) = stack.pop() {
            if depth >= max_depth {
                continue;
            }

            let effective_threshold = graph.resolve_threshold(current, threshold);

            if let Some(node) = graph.get_node(current) {
                let edges = node.edges_above_threshold(effective_threshold);
                for edge in edges {
                    if visited.contains(&edge.target) {
                        continue;
                    }
                    visited.insert(edge.target);

                    let new_sim = path_sim * edge.similarity;
                    let mut new_path = path.clone();
                    new_path.push(edge.target);
                    let new_depth = depth + 1;

                    results.push(TraversalResult {
                        id: edge.target,
                        depth: new_depth,
                        path_similarity: new_sim,
                        path: new_path.clone(),
                    });

                    stack.push((edge.target, new_depth, new_sim, new_path));
                }
            }
        }

        Ok(results)
    }
}
