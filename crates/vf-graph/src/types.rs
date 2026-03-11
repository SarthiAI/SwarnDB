// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use vf_core::types::{DistanceMetricType, SimilarityThreshold, VectorId};

static LOGICAL_CLOCK: AtomicU64 = AtomicU64::new(1);

/// Returns a monotonically increasing timestamp for edge creation ordering.
/// Uses AcqRel ordering for proper cross-thread visibility and wrapping_add
/// to handle the (astronomically unlikely) u64 wrap-around gracefully.
fn next_timestamp() -> u64 {
    // fetch_add already wraps on overflow for AtomicU64. Using AcqRel
    // ensures the increment is visible to other threads immediately.
    LOGICAL_CLOCK.fetch_add(1, Ordering::AcqRel)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub target: VectorId,
    pub similarity: f32,
    pub created_at: u64,
    pub refined: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphConfig {
    pub default_threshold: f32,
    pub max_edges_per_node: usize,
    pub max_traversal_depth: usize,
    pub distance_metric: DistanceMetricType,
    /// Number of nearest neighbors to search when computing graph edges.
    pub graph_neighbors_k: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            default_threshold: 0.7,
            max_edges_per_node: 100,
            max_traversal_depth: 10,
            distance_metric: DistanceMetricType::Cosine,
            graph_neighbors_k: 10,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub edges: Vec<Edge>,
    pub threshold_override: Option<f32>,
}

impl GraphNode {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            threshold_override: None,
        }
    }

    pub fn edges_above_threshold(&self, threshold: f32) -> Vec<&Edge> {
        self.edges
            .iter()
            .filter(|e| e.similarity >= threshold)
            .collect()
    }

    pub fn upsert_edge(&mut self, edge: Edge, max_edges: usize) {
        if let Some(pos) = self.edges.iter().position(|e| e.target == edge.target) {
            self.edges[pos].similarity = edge.similarity;
            self.edges[pos].created_at = edge.created_at;
            self.edges[pos].refined = edge.refined;
        } else {
            self.edges.push(edge);
        }

        self.edges
            .sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        if self.edges.len() > max_edges {
            self.edges.truncate(max_edges);
        }
    }

    pub fn remove_edge(&mut self, target: VectorId) -> bool {
        let before = self.edges.len();
        self.edges.retain(|e| e.target != target);
        self.edges.len() < before
    }
}

impl Default for GraphNode {
    fn default() -> Self {
        Self::new()
    }
}

/// The core virtual graph data structure storing similarity-based relationships
/// between vectors.
///
/// # Thread Safety
///
/// `VirtualGraph` is **not** internally synchronized. It uses a plain `HashMap`
/// for node storage, so concurrent reads and writes will cause data races.
///
/// Callers **must** provide external synchronization (e.g., wrap in
/// `parking_lot::RwLock` or `std::sync::Mutex`) when accessing from multiple
/// threads. The server layer (`vf-server`) wraps this in an `RwLock` for
/// safe concurrent access.
pub struct VirtualGraph {
    nodes: HashMap<VectorId, GraphNode>,
    config: GraphConfig,
}

impl VirtualGraph {
    pub fn new(config: GraphConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            config,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(GraphConfig::default())
    }

    /// Create a VirtualGraph with a custom similarity threshold and distance metric.
    /// Other config fields use defaults.
    pub fn with_threshold(threshold: f32, distance_metric: DistanceMetricType) -> Self {
        Self::new(GraphConfig {
            default_threshold: threshold,
            distance_metric,
            ..GraphConfig::default()
        })
    }

    pub fn add_node(&mut self, id: VectorId) {
        self.nodes.entry(id).or_insert_with(GraphNode::new);
    }

    pub fn remove_node(&mut self, id: VectorId) {
        if let Some(node) = self.nodes.remove(&id) {
            let targets: Vec<VectorId> = node.edges.iter().map(|e| e.target).collect();
            for target in targets {
                if let Some(target_node) = self.nodes.get_mut(&target) {
                    target_node.remove_edge(id);
                }
            }
        }
    }

    pub fn add_edge(&mut self, from: VectorId, to: VectorId, similarity: f32) {
        let ts = next_timestamp();
        let max_edges = self.config.max_edges_per_node;

        let forward = Edge {
            target: to,
            similarity,
            created_at: ts,
            refined: false,
        };
        self.nodes
            .entry(from)
            .or_insert_with(GraphNode::new)
            .upsert_edge(forward, max_edges);

        let backward = Edge {
            target: from,
            similarity,
            created_at: ts,
            refined: false,
        };
        self.nodes
            .entry(to)
            .or_insert_with(GraphNode::new)
            .upsert_edge(backward, max_edges);
    }

    pub fn remove_edge(&mut self, from: VectorId, to: VectorId) {
        if let Some(node) = self.nodes.get_mut(&from) {
            node.remove_edge(to);
        }
        if let Some(node) = self.nodes.get_mut(&to) {
            node.remove_edge(from);
        }
    }

    pub fn get_node(&self, id: VectorId) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: VectorId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }

    pub fn set_vector_threshold(&mut self, id: VectorId, threshold: f32) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.threshold_override = Some(threshold);
        }
    }

    pub fn clear_vector_threshold(&mut self, id: VectorId) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.threshold_override = None;
        }
    }

    pub fn resolve_threshold(&self, vector_id: VectorId, query_threshold: Option<f32>) -> f32 {
        let collection = SimilarityThreshold::collection(self.config.default_threshold);
        let query_th = query_threshold.map(SimilarityThreshold::query);
        let vector_th = self
            .nodes
            .get(&vector_id)
            .and_then(|n| n.threshold_override)
            .map(SimilarityThreshold::vector);

        SimilarityThreshold::resolve(
            Some(&collection),
            query_th.as_ref(),
            vector_th.as_ref(),
        )
        .unwrap_or(self.config.default_threshold)
    }

    pub fn get_neighbors(&self, id: VectorId, threshold: f32) -> Vec<&Edge> {
        self.nodes
            .get(&id)
            .map(|n| n.edges_above_threshold(threshold))
            .unwrap_or_default()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        let total: usize = self.nodes.values().map(|n| n.edges.len()).sum();
        total / 2
    }

    pub fn contains(&self, id: VectorId) -> bool {
        self.nodes.contains_key(&id)
    }

    pub fn config(&self) -> &GraphConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut GraphConfig {
        &mut self.config
    }

    pub fn node_ids(&self) -> Vec<VectorId> {
        self.nodes.keys().copied().collect()
    }

    pub fn nodes(&self) -> &HashMap<VectorId, GraphNode> {
        &self.nodes
    }
}
