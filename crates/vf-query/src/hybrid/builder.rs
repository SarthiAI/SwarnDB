// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! Type-state query builder. The frontier kind is tracked in the type system so
//! illegal compositions fail at compile time rather than at run time.
//!
//! ```compile_fail
//! use vf_query::hybrid::QueryBuilder;
//! // `return_paths` exists only on the path-frontier builder, so calling it on
//! // a node frontier does not compile.
//! let _plan = QueryBuilder::new()
//!     .from_node(vf_query::hybrid::NodeId(1))
//!     .return_paths();
//! ```

use std::marker::PhantomData;

use vf_graph::{EdgeDirection, NodeId};

use super::plan::{
    OnMissingVector, QueryPlan, ReturnKind, Step, TemporalFilter, VectorMathOp, WeightParams,
};
use super::predicate::Predicate;

/// Frontier marker: builder holds no frontier yet.
pub struct OnEmpty;
/// Frontier marker: current frontier is a set of nodes.
pub struct OnNodes;
/// Frontier marker: current frontier is a set of edges.
pub struct OnEdges;
/// Frontier marker: current frontier is a set of paths.
pub struct OnPaths;

/// Composable hybrid query builder. The `F` marker records the frontier kind;
/// each transition method moves to the marker matching its output.
pub struct QueryBuilder<F> {
    steps: Vec<Step>,
    _marker: PhantomData<F>,
}

impl<F> QueryBuilder<F> {
    /// Re-tag the builder with a new frontier marker, carrying steps forward.
    fn retag<G>(self) -> QueryBuilder<G> {
        QueryBuilder {
            steps: self.steps,
            _marker: PhantomData,
        }
    }
}

impl Default for QueryBuilder<OnEmpty> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Empty frontier: seeds only ───────────────────────────────────────────

impl QueryBuilder<OnEmpty> {
    /// Start a fresh builder with no frontier.
    pub fn new() -> Self {
        QueryBuilder {
            steps: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Seed nodes from the top-k vector similarity search.
    pub fn vector_similar(mut self, vector: Vec<f32>, k: usize) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorSimilar {
            vector,
            k,
            ef_search: None,
        });
        self.retag()
    }

    /// Seed nodes from a vector search with an explicit `ef_search`.
    pub fn vector_similar_with_ef(
        mut self,
        vector: Vec<f32>,
        k: usize,
        ef_search: usize,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorSimilar {
            vector,
            k,
            ef_search: Some(ef_search),
        });
        self.retag()
    }

    /// Seed the node frontier from a single node id.
    pub fn from_node(self, node: NodeId) -> QueryBuilder<OnNodes> {
        self.from_nodes(vec![node])
    }

    /// Seed the node frontier from explicit node ids.
    pub fn from_nodes(mut self, nodes: Vec<NodeId>) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::FromNodes { nodes });
        self.retag()
    }

    /// Seed the node frontier by scanning the graph store for nodes that match an
    /// optional kind / entity label / property condition. No vector and no
    /// explicit ids: this is the filter-seed source. `is_entity` is Some(true)
    /// for entity nodes, Some(false) for content, None for either. The resulting
    /// frontier feeds the normal traversal chain.
    pub fn scan_by_filter(
        mut self,
        is_entity: Option<bool>,
        label: Option<String>,
        predicate: Option<Predicate>,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::ScanByFilter {
            is_entity,
            label,
            predicate,
        });
        self.retag()
    }
}

// ── Node frontier ────────────────────────────────────────────────────────

impl QueryBuilder<OnNodes> {
    /// One-hop neighbour expansion. The default (non-temporal) path: byte-identical
    /// to the pre-P17 behaviour.
    pub fn traverse(
        mut self,
        edge_type: Option<String>,
        direction: EdgeDirection,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::Traverse {
            edge_type,
            direction,
            temporal: None,
        });
        self
    }

    /// Opt-in time-and-context one-hop expansion (P17). Same membership semantics
    /// as `traverse`, but only edges valid at the filter's instant (and context)
    /// contribute neighbours.
    pub fn traverse_temporal(
        mut self,
        edge_type: Option<String>,
        direction: EdgeDirection,
        temporal: TemporalFilter,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::Traverse {
            edge_type,
            direction,
            temporal: Some(temporal),
        });
        self
    }

    /// Bounded multi-hop expansion with an optional per-hop node predicate. The
    /// unweighted path: byte-identical to the pre-P13 behavior.
    pub fn k_hop(
        mut self,
        edge_type: Option<String>,
        max: u32,
        predicate: Option<Predicate>,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::KHop {
            edge_type,
            max,
            predicate,
            weight: None,
            order_by_weight: false,
            temporal: None,
        });
        self
    }

    /// Opt-in weight-aware k-hop (P13). Same node membership as `k_hop`, but when
    /// `order_by_weight` is true the result is ordered by descending accumulated
    /// edge weight. `weight` selects the signals (confidence, recency, explicit).
    pub fn k_hop_weighted(
        mut self,
        edge_type: Option<String>,
        max: u32,
        predicate: Option<Predicate>,
        weight: WeightParams,
        order_by_weight: bool,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::KHop {
            edge_type,
            max,
            predicate,
            weight: Some(weight),
            order_by_weight,
            temporal: None,
        });
        self
    }

    /// Opt-in time-and-context k-hop (P17). Same membership/order semantics as
    /// `k_hop`, but every hop only expands over edges valid at the filter's
    /// instant (and context).
    pub fn k_hop_temporal(
        mut self,
        edge_type: Option<String>,
        max: u32,
        predicate: Option<Predicate>,
        temporal: TemporalFilter,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::KHop {
            edge_type,
            max,
            predicate,
            weight: None,
            order_by_weight: false,
            temporal: Some(temporal),
        });
        self
    }

    /// Opt-in weight-aware AND time-and-context k-hop (P17 + P13). Same membership
    /// as `k_hop`, ordered by descending accumulated weight when `order_by_weight`,
    /// and restricted per hop to edges valid at the filter's instant (and context).
    pub fn k_hop_weighted_temporal(
        mut self,
        edge_type: Option<String>,
        max: u32,
        predicate: Option<Predicate>,
        weight: WeightParams,
        order_by_weight: bool,
        temporal: TemporalFilter,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::KHop {
            edge_type,
            max,
            predicate,
            weight: Some(weight),
            order_by_weight,
            temporal: Some(temporal),
        });
        self
    }

    /// Common neighbours with a sub-query node result.
    pub fn mutual_neighbors(mut self, other: QueryPlan) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::MutualNeighbors { other });
        self
    }

    /// Intersect the current node set with a sub-query node result.
    pub fn intersect(mut self, other: QueryPlan) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::Intersect { other });
        self
    }

    /// Union the current node set with a sub-query node result.
    pub fn union(mut self, other: QueryPlan) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::Union { other });
        self
    }

    /// Keep nodes satisfying the predicate.
    pub fn filter(mut self, predicate: Predicate) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::Filter { predicate });
        self
    }

    /// Rank the current node frontier by similarity to `vector`, keeping the
    /// top `k` most similar. Frontier-consuming (graph -> vector): the graph
    /// has already fixed the candidate set, so scoring is exact over exactly
    /// these nodes (recall 1.0, no ANN, no ef_search). Returns the frontier in
    /// ascending-distance order (most similar first), smaller node id breaking
    /// ties, so a following `.limit(n)` yields the top-n by similarity.
    /// `on_missing` governs frontier nodes that have no vector (entity nodes):
    /// `Skip` (default) drops and counts them, `Error` fails the query.
    pub fn vector_rank(mut self, vector: Vec<f32>, k: usize, on_missing: OnMissingVector) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorRank { vector, k, on_missing, predicate: None });
        self
    }

    /// Filter-then-rank variant of `vector_rank` (ADR-034): narrow the CURRENT
    /// frontier to nodes satisfying `predicate` BEFORE ranking (pre-filter, never
    /// post-filter), then run the same exact similarity ranking inside that narrowed
    /// set. An empty frontier ranks to empty (empty in, empty out). To rank over the
    /// complete set of all nodes matching a condition, seed with `scan_by_filter`
    /// first: `scan_by_filter(predicate).vector_rank(vector, k)`.
    pub fn vector_rank_filtered(
        mut self,
        vector: Vec<f32>,
        k: usize,
        on_missing: OnMissingVector,
        predicate: Predicate,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorRank {
            vector,
            k,
            on_missing,
            predicate: Some(predicate),
        });
        self
    }

    /// Rank the current node frontier by ascending distance to the analogy vector
    /// `a - b + c` (P17), keeping the top-k. Configured-metric ranking; exact over
    /// the fixed frontier. `on_missing` governs frontier nodes with no vector.
    pub fn analogy_rank(
        mut self,
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        k: usize,
        on_missing: OnMissingVector,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorMath {
            op: VectorMathOp::Analogy { a, b, c },
            k,
            on_missing,
        });
        self
    }

    /// Re-rank the current node frontier with Maximal Marginal Relevance against
    /// `query` (P17), trading relevance for diversity by `lambda` (1.0 = pure
    /// relevance, 0.0 = pure diversity). Output is in MMR-selection order, length
    /// min(k, eligible). MMR uses its own internal cosine, not the index metric.
    pub fn diversity_rank(
        mut self,
        query: Vec<f32>,
        lambda: f32,
        k: usize,
        on_missing: OnMissingVector,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorMath {
            op: VectorMathOp::Diversity { query, lambda },
            k,
            on_missing,
        });
        self
    }

    /// Keep frontier nodes within the cone of half-angle `aperture_radians` around
    /// `direction` (P17), ordered by ascending angle, capped at k. Uses the cone
    /// op's internal cosine geometry.
    pub fn cone_filter(
        mut self,
        direction: Vec<f32>,
        aperture_radians: f32,
        k: usize,
        on_missing: OnMissingVector,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorMath {
            op: VectorMathOp::Cone {
                direction,
                aperture_radians,
            },
            k,
            on_missing,
        });
        self
    }

    /// Rank the current node frontier by isolation (P17): score each node by its
    /// minimum distance to any of `centroids` (configured metric), returning the
    /// top-k MOST isolated (descending min-distance).
    pub fn isolation_rank(
        mut self,
        centroids: Vec<Vec<f32>>,
        k: usize,
        on_missing: OnMissingVector,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorMath {
            op: VectorMathOp::Isolation { centroids },
            k,
            on_missing,
        });
        self
    }

    /// Rank the current node frontier by ascending distance to the centroid of the
    /// frontier's OWN vectors (P17), keeping the top-k most representative.
    /// Configured-metric ranking.
    pub fn centroid_rank(mut self, k: usize, on_missing: OnMissingVector) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorMath {
            op: VectorMathOp::Centroid {},
            k,
            on_missing,
        });
        self
    }

    /// Rank the current node frontier by ascending distance to the interpolated
    /// point `slerp(a, b, t)` (lerp fallback when nearly parallel) (P17), keeping
    /// the top-k. The query errors cleanly when `t` is outside [0,1] or `a`/`b`
    /// dims mismatch. Configured-metric ranking.
    pub fn interpolate_rank(
        mut self,
        a: Vec<f32>,
        b: Vec<f32>,
        t: f32,
        k: usize,
        on_missing: OnMissingVector,
    ) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::VectorMath {
            op: VectorMathOp::Interpolate { a, b, t },
            k,
            on_missing,
        });
        self
    }

    /// Truncate the node frontier to the first `n`.
    pub fn limit(mut self, n: usize) -> QueryBuilder<OnNodes> {
        self.steps.push(Step::Limit { n });
        self
    }

    /// Collect incident edges and move to an edge frontier.
    pub fn edges(
        mut self,
        edge_type: Option<String>,
        direction: EdgeDirection,
    ) -> QueryBuilder<OnEdges> {
        self.steps.push(Step::CollectEdges {
            edge_type,
            direction,
        });
        self.retag()
    }

    /// Unweighted shortest path to a target, moving to a path frontier. Byte-
    /// identical to the pre-P13 BFS.
    pub fn shortest_path(mut self, edge_types: Vec<String>, target: NodeId) -> QueryBuilder<OnPaths> {
        self.steps.push(Step::ShortestPath {
            edge_types,
            target,
            weighted: false,
            weight: None,
            temporal: None,
        });
        self.retag()
    }

    /// Opt-in weighted shortest path (P13): Dijkstra minimizing total cost, where
    /// each edge costs 1/effective_weight, so stronger edges shorten the path.
    pub fn weighted_shortest_path(
        mut self,
        edge_types: Vec<String>,
        target: NodeId,
        weight: WeightParams,
    ) -> QueryBuilder<OnPaths> {
        self.steps.push(Step::ShortestPath {
            edge_types,
            target,
            weighted: true,
            weight: Some(weight),
            temporal: None,
        });
        self.retag()
    }

    /// Opt-in time-and-context unweighted shortest path (P17). Same BFS as
    /// `shortest_path`, but only edges valid at the filter's instant (and context)
    /// are traversed.
    pub fn shortest_path_temporal(
        mut self,
        edge_types: Vec<String>,
        target: NodeId,
        temporal: TemporalFilter,
    ) -> QueryBuilder<OnPaths> {
        self.steps.push(Step::ShortestPath {
            edge_types,
            target,
            weighted: false,
            weight: None,
            temporal: Some(temporal),
        });
        self.retag()
    }

    /// Opt-in weighted AND time-and-context shortest path (P17 + P13). Dijkstra over
    /// 1/effective_weight, restricted to edges valid at the filter's instant (and
    /// context).
    pub fn weighted_shortest_path_temporal(
        mut self,
        edge_types: Vec<String>,
        target: NodeId,
        weight: WeightParams,
        temporal: TemporalFilter,
    ) -> QueryBuilder<OnPaths> {
        self.steps.push(Step::ShortestPath {
            edge_types,
            target,
            weighted: true,
            weight: Some(weight),
            temporal: Some(temporal),
        });
        self.retag()
    }

    /// Finish, returning the node-result plan.
    pub fn return_nodes(self) -> QueryPlan {
        QueryPlan {
            steps: self.steps,
            return_kind: ReturnKind::Nodes,
        }
    }
}

// ── Edge frontier ──────────────────────────────────────────────────────────

impl QueryBuilder<OnEdges> {
    /// Keep edges satisfying the predicate.
    pub fn filter(mut self, predicate: Predicate) -> QueryBuilder<OnEdges> {
        self.steps.push(Step::Filter { predicate });
        self
    }

    /// Truncate the edge frontier to the first `n`.
    pub fn limit(mut self, n: usize) -> Self {
        self.steps.push(Step::Limit { n });
        self
    }

    /// Finish, returning the edge-result plan.
    pub fn return_edges(self) -> QueryPlan {
        QueryPlan {
            steps: self.steps,
            return_kind: ReturnKind::Edges,
        }
    }
}

// ── Path frontier ──────────────────────────────────────────────────────────

impl QueryBuilder<OnPaths> {
    /// Truncate the path frontier to the first `n`.
    pub fn limit(mut self, n: usize) -> Self {
        self.steps.push(Step::Limit { n });
        self
    }

    /// Finish, returning the path-result plan.
    pub fn return_paths(self) -> QueryPlan {
        QueryPlan {
            steps: self.steps,
            return_kind: ReturnKind::Paths,
        }
    }
}
