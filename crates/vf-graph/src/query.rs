// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::{HashMap, HashSet, VecDeque};
use vf_core::types::VectorId;

use crate::error::GraphError;
use crate::traversal::{GraphTraversal, TraversalOrder, TraversalResult};
use crate::types::VirtualGraph;

#[derive(Clone, Debug)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Clone, Debug)]
pub struct RelationshipQuery {
    pub start: VectorId,
    pub depth: usize,
    pub threshold: Option<f32>,
    pub direction: Direction,
    pub max_results: Option<usize>,
    pub min_similarity: Option<f32>,
}

impl RelationshipQuery {
    pub fn new(start: VectorId) -> Self {
        Self {
            start,
            depth: 1,
            threshold: None,
            direction: Direction::Both,
            max_results: None,
            min_similarity: None,
        }
    }

    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    pub fn max_results(mut self, max: usize) -> Self {
        self.max_results = Some(max);
        self
    }

    pub fn min_similarity(mut self, min: f32) -> Self {
        self.min_similarity = Some(min);
        self
    }
}

pub struct RelationshipQueryEngine;

impl RelationshipQueryEngine {
    pub fn get_related(
        graph: &VirtualGraph,
        id: VectorId,
        threshold: Option<f32>,
    ) -> Result<Vec<(VectorId, f32)>, GraphError> {
        if !graph.contains(id) {
            return Err(GraphError::NodeNotFound(id));
        }

        let effective_threshold = graph.resolve_threshold(id, threshold);
        let node = graph
            .get_node(id)
            .ok_or(GraphError::NodeNotFound(id))?;

        let mut results: Vec<(VectorId, f32)> = node
            .edges_above_threshold(effective_threshold)
            .into_iter()
            .map(|e| (e.target, e.similarity))
            .collect();

        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    pub fn query(
        graph: &VirtualGraph,
        query: &RelationshipQuery,
    ) -> Result<Vec<TraversalResult>, GraphError> {
        let mut results = GraphTraversal::traverse(
            graph,
            query.start,
            &TraversalOrder::BreadthFirst,
            query.depth,
            query.threshold,
            query.max_results,
        )?;

        if let Some(min_sim) = query.min_similarity {
            results.retain(|r| r.path_similarity >= min_sim);
        }

        Ok(results)
    }

    pub fn mutual_neighbors(
        graph: &VirtualGraph,
        a: VectorId,
        b: VectorId,
        threshold: Option<f32>,
    ) -> Result<Vec<VectorId>, GraphError> {
        let neighbors_a: HashSet<VectorId> = Self::get_related(graph, a, threshold)?
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let neighbors_b: HashSet<VectorId> = Self::get_related(graph, b, threshold)?
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let mut mutual: Vec<VectorId> = neighbors_a
            .intersection(&neighbors_b)
            .copied()
            .collect();

        mutual.sort();
        Ok(mutual)
    }

    pub fn find_path(
        graph: &VirtualGraph,
        from: VectorId,
        to: VectorId,
        max_depth: usize,
        threshold: Option<f32>,
    ) -> Result<Option<Vec<VectorId>>, GraphError> {
        if !graph.contains(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !graph.contains(to) {
            return Err(GraphError::NodeNotFound(to));
        }
        if from == to {
            return Ok(Some(vec![from]));
        }

        let mut visited = HashSet::new();
        visited.insert(from);

        let mut parent: HashMap<VectorId, VectorId> = HashMap::new();
        let mut queue: VecDeque<(VectorId, usize)> = VecDeque::new();
        queue.push_back((from, 0));

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let effective_threshold = graph.resolve_threshold(current, threshold);

            if let Some(node) = graph.get_node(current) {
                for edge in node.edges_above_threshold(effective_threshold) {
                    if visited.contains(&edge.target) {
                        continue;
                    }
                    visited.insert(edge.target);
                    parent.insert(edge.target, current);

                    if edge.target == to {
                        let mut path = Vec::new();
                        let mut cur = to;
                        while cur != from {
                            path.push(cur);
                            cur = parent[&cur];
                        }
                        path.push(from);
                        path.reverse();
                        return Ok(Some(path));
                    }

                    queue.push_back((edge.target, depth + 1));
                }
            }
        }

        Ok(None)
    }
}
