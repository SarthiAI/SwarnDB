// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Hybrid execution engine. Runs a [`QueryPlan`] against a vector index and a
//! typed graph store, threading a runtime frontier through the steps and
//! materializing the final result per the plan's return kind.
//!
//! The executor only reads: it never mutates the index or the store. Node and
//! edge collections preserve first-seen order and dedup via a seen-guard.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::time::Instant;

use vf_core::distance::DistanceMetric;
use vf_graph::weight::WeightParams;
use vf_graph::{
    EdgeDirection, EdgeId, GraphStore, Node, NodeId, NodePageFilter, TemporalFilter,
    TypedEdge as Edge,
};
use vf_index::traits::VectorIndex;

use super::error::HybridQueryError;
use super::plan::{
    OnMissingVector, QueryPlan, ReturnKind, RrfRankSpec, Step, VectorMathOp, RRF_MENTIONS_EDGE_TYPE,
};
use crate::vector_math::{
    AnalogyComputer, CentroidComputer, ConeSearch, DiversitySampler, GhostDetector, Interpolator,
};

// ── VectorRank metric names (P09.6) ─────────────────────────────────────────
/// Count of VectorRank steps executed.
const VECTOR_RANK_TOTAL: &str = "swarndb_vector_rank_total";
/// Distribution of the incoming node-frontier size per VectorRank step.
const VECTOR_RANK_FRONTIER_SIZE: &str = "swarndb_vector_rank_frontier_size";
/// Count of frontier nodes dropped because they carried no vector (Skip policy).
const VECTOR_RANK_SKIPPED_MISSING_VECTOR_TOTAL: &str =
    "swarndb_vector_rank_skipped_missing_vector_total";
/// Count of exact distance computations performed on the plain path.
const VECTOR_RANK_DISTANCE_COMPUTATIONS_TOTAL: &str =
    "swarndb_vector_rank_distance_computations_total";
/// VectorRank end-to-end latency, in seconds.
const VECTOR_RANK_LATENCY_SECONDS: &str = "swarndb_vector_rank_latency_seconds";
/// Count of VectorRank steps rejected because the frontier exceeded the cap.
const VECTOR_RANK_CAP_EXCEEDED_TOTAL: &str = "swarndb_vector_rank_cap_exceeded_total";

// ── VectorMath metric names (P17) ───────────────────────────────────────────
/// Count of VectorMath steps executed, labelled by op.
const VECTOR_MATH_TOTAL: &str = "swarndb_vector_math_total";
/// Distribution of the incoming node-frontier size per VectorMath step, by op.
const VECTOR_MATH_FRONTIER_SIZE: &str = "swarndb_vector_math_frontier_size";
/// Count of frontier nodes dropped because they carried no vector (Skip policy), by op.
const VECTOR_MATH_SKIPPED_MISSING_VECTOR_TOTAL: &str =
    "swarndb_vector_math_skipped_missing_vector_total";
/// VectorMath end-to-end latency, in seconds, by op.
const VECTOR_MATH_LATENCY_SECONDS: &str = "swarndb_vector_math_latency_seconds";
/// Count of VectorMath steps rejected because the frontier exceeded the cap, by op.
const VECTOR_MATH_CAP_EXCEEDED_TOTAL: &str = "swarndb_vector_math_cap_exceeded_total";

// ── Quality-aware traversal/ranking metric names (P13) ───────────────────────
/// Count of weight-ordered k-hop steps executed (order_by_weight path only).
const WEIGHTED_KHOP_TOTAL: &str = "swarndb_weighted_khop_total";
/// Weight-ordered k-hop end-to-end latency, in seconds.
const WEIGHTED_KHOP_LATENCY_SECONDS: &str = "swarndb_weighted_khop_latency_seconds";
/// Count of weighted (Dijkstra) shortest-path steps executed.
const WEIGHTED_SHORTEST_PATH_TOTAL: &str = "swarndb_weighted_shortest_path_total";
/// Count of RRF rankings that applied the opt-in edge-quality factor.
const RANKING_EDGE_WEIGHTED_TOTAL: &str = "swarndb_ranking_edge_weighted_total";

/// Largest node frontier VectorRank will rank before erroring (no silent
/// truncation). Read once from `SWARNDB_MAX_FRONTIER_FOR_RANK`, default 100_000.
const DEFAULT_MAX_FRONTIER_FOR_RANK: usize = 100_000;

/// Cached frontier cap. vf-query has no config plumbing of its own (the server
/// config lives in vf-server), so this guardrail reads its env directly, once.
static MAX_FRONTIER_FOR_RANK: OnceLock<usize> = OnceLock::new();

/// The configured VectorRank frontier cap, parsed once on first use.
fn max_frontier_for_rank() -> usize {
    *MAX_FRONTIER_FOR_RANK.get_or_init(|| {
        std::env::var("SWARNDB_MAX_FRONTIER_FOR_RANK")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_MAX_FRONTIER_FOR_RANK)
    })
}

/// A materialized node row. `node` is `Some` when the store resolved the id.
pub struct NodeRecord {
    pub id: NodeId,
    pub node: Option<Node>,
}

/// An ordered chain of node ids from a seed to a target, inclusive.
#[derive(Clone, Debug)]
pub struct Path {
    pub nodes: Vec<NodeId>,
}

/// The final result of executing a plan.
pub enum QueryResult {
    Nodes(Vec<NodeRecord>),
    Edges(Vec<Edge>),
    Paths(Vec<Path>),
}

/// Borrowed handles to the index and graph store for the lifetime of a query.
pub struct HybridExecutor<'a> {
    index: &'a dyn VectorIndex,
    store: &'a dyn GraphStore,
}

/// Runtime frontier flowing between steps. Order-preserving and deduped.
enum Frontier {
    Nodes(Vec<NodeId>),
    Edges(Vec<EdgeId>),
    Paths(Vec<Path>),
}

/// Order-preserving dedup accumulator for ids.
struct OrderedSet<T> {
    items: Vec<T>,
    seen: HashSet<T>,
}

impl<T: Copy + Eq + std::hash::Hash> OrderedSet<T> {
    fn new() -> Self {
        Self {
            items: Vec::new(),
            seen: HashSet::new(),
        }
    }

    /// Push `value` if not yet seen; returns true when it was newly added.
    fn push(&mut self, value: T) -> bool {
        if self.seen.insert(value) {
            self.items.push(value);
            true
        } else {
            false
        }
    }

    fn into_vec(self) -> Vec<T> {
        self.items
    }
}

impl<'a> HybridExecutor<'a> {
    /// Build an executor over a vector index and a typed graph store.
    pub fn new(index: &'a dyn VectorIndex, store: &'a dyn GraphStore) -> Self {
        Self { index, store }
    }

    /// Run a plan to completion and materialize the result.
    ///
    /// This is the DEFAULT execution path. The default graph-augmented ranking
    /// is the graph-first scope-then-rank order (ADR-024): the plan scopes the
    /// candidate set by structure, then ranks it with a `VectorRank` step. RRF
    /// is the opt-in path via `execute_rrf`, reached only on an explicit spec.
    pub fn execute(&self, plan: &QueryPlan) -> Result<QueryResult, HybridQueryError> {
        if plan.steps.is_empty() {
            return Err(HybridQueryError::EmptyPlan);
        }
        // Single wall-clock read per query, threaded into any weight-aware step so
        // recency decay is computed against one consistent instant.
        let now_ms = now_unix_millis();
        let frontier = self.run_steps(&plan.steps, now_ms)?;
        self.materialize(frontier, &plan.return_kind)
    }

    /// Run a plan, then re-rank its node pool with graph-aware Reciprocal Rank
    /// Fusion per `spec`, returning the fused top-k as node records.
    ///
    /// This is the OPT-IN ranking path. It is only reached when the request
    /// carries an explicit RRF spec; the default `execute` path never calls it,
    /// so a query with no spec computes no proximity and adds zero latency. Per
    /// ADR-024 the default graph-augmented ranking is the `VectorRank` step run
    /// by `execute`, not this fusion; RRF stays available here on explicit opt-in.
    ///
    /// Faithful to the validated harness (`examples/real_business`):
    ///   - vector ranking  = the vector seed in similarity order (the first
    ///                       VectorSimilar step's results);
    ///   - candidate pool  = the full plan's node frontier (vector seed UNION
    ///                       graph expansion);
    ///   - graph ranking   = the pool ordered by graph-proximity, the count of
    ///                       distinct bridge routes from the vector seed to each
    ///                       candidate (graph structure only, no vector score);
    ///   - fused order     = standard RRF, 1 / (rrf_k + rank) summed over each
    ///                       ranking the candidate appears in, equal weight,
    ///                       cut to k.
    pub fn execute_rrf(
        &self,
        plan: &QueryPlan,
        spec: &RrfRankSpec,
    ) -> Result<QueryResult, HybridQueryError> {
        if plan.steps.is_empty() {
            return Err(HybridQueryError::EmptyPlan);
        }
        // Single wall-clock read per query, threaded into the steps and proximity.
        let now_ms = now_unix_millis();
        // The vector ranking and the proximity seeds both come from the plan's
        // leading vector seed. RRF ranking requires that seed.
        let seed_vector = match plan.steps.first() {
            Some(Step::VectorSimilar { vector, k, ef_search }) => {
                let results = self.index.search(vector, *k, *ef_search)?;
                dedup_ids(results.into_iter().map(|r| NodeId(r.id)))
            }
            _ => {
                return Err(HybridQueryError::InvalidPlan(
                    "rrf ranking requires the plan to begin with a vector_similar seed".into(),
                ))
            }
        };

        // Candidate pool = the full plan's node frontier (vector seed UNION the
        // graph expansion the caller composed).
        let pool = match self.run_steps(&plan.steps, now_ms)? {
            Frontier::Nodes(ids) => ids,
            _ => {
                return Err(HybridQueryError::InvalidPlan(
                    "rrf ranking requires a node-result plan".into(),
                ))
            }
        };

        // Graph-proximity score per candidate. With hub damping off this is the
        // count of distinct bridge routes; with damping on each route is weighted
        // by the inverse hub function of its bridging entities. Graph only. When
        // spec.edge_weight is set, the structural route also gains an edge-quality
        // factor (P13); otherwise this is byte-identical to the pre-P13 path.
        let proximity = self.graph_proximity(&seed_vector, spec, now_ms);

        let fused = rrf_fuse_topk(&seed_vector, &proximity, &pool, spec);
        let rows = fused
            .into_iter()
            .map(|id| NodeRecord {
                id,
                node: self.store.get_node(id),
            })
            .collect();
        Ok(QueryResult::Nodes(rows))
    }

    /// Per-candidate graph-proximity score: the weighted sum of distinct bridge
    /// routes from the vector seed that land on a candidate content node. Pure
    /// graph structure, no vector similarity.
    ///
    /// Two shapes mirror the harness exactly:
    ///   entity_bridge (relation_edge_types non-empty): one route per
    ///     (seed node, relation type) sub-plan, each being
    ///     seed -> mentions(out) -> entity -> k_hop(relation) -> entity ->
    ///     mentions(in) -> candidate;
    ///   structural (relation_edge_types empty): one route per seed node,
    ///     seed -> k_hop(any edge type) -> candidate (content-to-content).
    /// Each sub-plan is deduped, so it contributes at most one route to a
    /// candidate.
    ///
    /// Hub-aware damping (ADR-019): each route contributes a WEIGHT, not a flat
    /// 1. When `hub_damping == 0.0` the weight is exactly 1.0 per route, so the
    /// score is the plain route count and the ordering is byte-identical to the
    /// prior behavior. When `hub_damping > 0.0` the weight of a route is the
    /// product over the route's bridging nodes of
    /// `1 / (1 + hub_damping * ln(1 + degree(node)))`, so routes through
    /// high-degree hubs contribute less. The bridging nodes are:
    ///   entity_bridge: the entity nodes reached by
    ///     seed -> mentions(out) -> entity -> k_hop(relation) (the entities that
    ///     link back to candidates via mentions(in)); degree = a node's mentions
    ///     in-degree (how many content nodes mention it), the hub signal;
    ///   structural: the single candidate content node itself; degree = its total
    ///     incident-edge count (content-to-content graphs have no entity layer).
    /// Per-node degree is cached for the duration of the call.
    ///
    /// Edge-quality weighting (P13): when `spec.edge_weight` is None the path is
    /// byte-identical to the above. When it is Some(p), the STRUCTURAL route also
    /// gains a per-candidate factor = the best product of edge effective weights
    /// along the route from the seed, multiplied into the contribution alongside
    /// the hub factor (both multiply, hub_damping is not replaced). The candidate
    /// SET stays identical (same bridge membership); only each score is scaled.
    fn graph_proximity(
        &self,
        seed: &[NodeId],
        spec: &RrfRankSpec,
        now_ms: u64,
    ) -> HashMap<NodeId, f64> {
        let mut scores: HashMap<NodeId, f64> = HashMap::new();
        if seed.is_empty() {
            return scores;
        }
        let seed_set: HashSet<NodeId> = seed.iter().copied().collect();
        let k_hop_max = spec.effective_k_hop_max();
        let structural = spec.relation_edge_types.is_empty();
        let hub_damping = spec.hub_damping;
        let damping_on = spec.hub_damping_on();
        let edge_weight = spec.edge_weight.as_ref();
        if edge_weight.is_some() {
            metrics::counter!(RANKING_EDGE_WEIGHTED_TOTAL).increment(1);
        }
        // Per-entity mentions-in-degree cache, reused across seeds/relations.
        let mut degree_cache: HashMap<NodeId, u32> = HashMap::new();

        for &seed_node in seed {
            if structural {
                if let Some(p) = edge_weight {
                    // Weighted structural bridge: same membership as bridge_structural,
                    // each candidate paired with its best edge-quality product.
                    let reached = self.bridge_structural_weighted(seed_node, k_hop_max, p, now_ms);
                    let mut counted: HashSet<NodeId> = HashSet::new();
                    for (cand, edge_quality) in reached {
                        if seed_set.contains(&cand) || !counted.insert(cand) {
                            continue;
                        }
                        // Hub factor (unchanged) composed multiplicatively with the
                        // edge-quality factor; both default to 1.0 when off.
                        let hub = if damping_on {
                            self.hub_weight_total_degree(cand, hub_damping, &mut degree_cache)
                        } else {
                            1.0
                        };
                        *scores.entry(cand).or_insert(0.0) += hub * edge_quality;
                    }
                } else {
                    // One structural route per seed: seed -> k_hop(any) -> candidate.
                    let reached = self.bridge_structural(seed_node, k_hop_max, now_ms);
                    let mut counted: HashSet<NodeId> = HashSet::new();
                    for cand in reached {
                        if seed_set.contains(&cand) || !counted.insert(cand) {
                            continue;
                        }
                        // Structural mode has no entity layer; the bridging node is
                        // the candidate itself, weighted by its total degree.
                        let weight = if damping_on {
                            self.hub_weight_total_degree(cand, hub_damping, &mut degree_cache)
                        } else {
                            1.0
                        };
                        *scores.entry(cand).or_insert(0.0) += weight;
                    }
                }
            } else {
                // One entity-bridge route per (seed, relation) sub-plan. The
                // entity-bridge shape is not edge-quality-weighted (P13 weights the
                // structural bridge); this stays byte-identical to the prior path.
                for rel in &spec.relation_edge_types {
                    // Bridging entities = entities reached by
                    // seed -> mentions(out) -> entity -> k_hop(relation).
                    // The route weight is the product over them of the hub factor
                    // (computed once per sub-plan, shared by all its candidates).
                    let route_weight = if damping_on {
                        let entities = self.bridge_entities(seed_node, rel, k_hop_max, now_ms);
                        self.route_weight_entities(&entities, hub_damping, &mut degree_cache)
                    } else {
                        1.0
                    };
                    let reached = self.bridge_entity(seed_node, rel, k_hop_max, now_ms);
                    let mut counted: HashSet<NodeId> = HashSet::new();
                    for cand in reached {
                        if seed_set.contains(&cand) || !counted.insert(cand) {
                            continue;
                        }
                        *scores.entry(cand).or_insert(0.0) += route_weight;
                    }
                }
            }
        }
        scores
    }

    /// Mentions in-degree of an entity node: how many content nodes link to it
    /// via the native passage-to-entity link (the hub signal). Cached per call.
    fn mentions_in_degree(&self, entity: NodeId, cache: &mut HashMap<NodeId, u32>) -> u32 {
        if let Some(&d) = cache.get(&entity) {
            return d;
        }
        let d = self
            .store
            .neighbors(entity, Some(RRF_MENTIONS_EDGE_TYPE), EdgeDirection::Incoming)
            .len() as u32;
        cache.insert(entity, d);
        d
    }

    /// Total incident-edge count of a node (any direction, any type). Used as the
    /// hub signal in structural mode, where there is no entity layer. Cached.
    fn total_degree(&self, node: NodeId, cache: &mut HashMap<NodeId, u32>) -> u32 {
        if let Some(&d) = cache.get(&node) {
            return d;
        }
        let d = self.store.edges_for_node(node, EdgeDirection::Both).len() as u32;
        cache.insert(node, d);
        d
    }

    /// The single-node damping factor 1 / (1 + hub_damping * ln(1 + degree)).
    fn hub_factor(degree: u32, hub_damping: f64) -> f64 {
        1.0 / (1.0 + hub_damping * (1.0 + degree as f64).ln())
    }

    /// Structural route weight: the hub factor of the candidate's total degree.
    fn hub_weight_total_degree(
        &self,
        node: NodeId,
        hub_damping: f64,
        cache: &mut HashMap<NodeId, u32>,
    ) -> f64 {
        let degree = self.total_degree(node, cache);
        Self::hub_factor(degree, hub_damping)
    }

    /// Entity-bridge route weight: the product over the route's bridging entities
    /// of the hub factor of each entity's mentions in-degree. An empty entity set
    /// yields 1.0 (no bridging node to damp), matching the unweighted route.
    fn route_weight_entities(
        &self,
        entities: &[NodeId],
        hub_damping: f64,
        cache: &mut HashMap<NodeId, u32>,
    ) -> f64 {
        let mut weight = 1.0;
        for &entity in entities {
            let degree = self.mentions_in_degree(entity, cache);
            weight *= Self::hub_factor(degree, hub_damping);
        }
        weight
    }

    /// The bridging entities of one entity-bridge route from a single seed via
    /// one relation type: the entity nodes reached by
    /// seed -> mentions(out) -> entity -> k_hop(relation). These are the entities
    /// that link back to candidates via mentions(in); their degrees drive the
    /// route weight. Deduped, so each entity counts once in the product.
    fn bridge_entities(
        &self,
        seed: NodeId,
        relation: &str,
        k_hop_max: u32,
        now_ms: u64,
    ) -> Vec<NodeId> {
        let steps = vec![
            Step::FromNodes { nodes: vec![seed] },
            Step::Traverse {
                edge_type: Some(RRF_MENTIONS_EDGE_TYPE.to_string()),
                direction: EdgeDirection::Outgoing,
                temporal: None,
            },
            Step::KHop {
                edge_type: Some(relation.to_string()),
                max: k_hop_max,
                predicate: None,
                weight: None,
                order_by_weight: false,
                temporal: None,
            },
        ];
        self.node_ids_or_empty(&steps, now_ms)
    }

    /// One structural bridge route from a single seed: seed -> k_hop(any edge
    /// type) -> other content nodes.
    fn bridge_structural(&self, seed: NodeId, k_hop_max: u32, now_ms: u64) -> Vec<NodeId> {
        let steps = vec![
            Step::FromNodes { nodes: vec![seed] },
            Step::KHop {
                edge_type: None,
                max: k_hop_max,
                predicate: None,
                weight: None,
                order_by_weight: false,
                temporal: None,
            },
        ];
        self.node_ids_or_empty(&steps, now_ms)
    }

    /// One entity-bridge route from a single seed via one relation type:
    /// seed -> mentions(out) -> entity -> k_hop(relation) -> entity ->
    /// mentions(in) -> other content nodes.
    fn bridge_entity(
        &self,
        seed: NodeId,
        relation: &str,
        k_hop_max: u32,
        now_ms: u64,
    ) -> Vec<NodeId> {
        let steps = vec![
            Step::FromNodes { nodes: vec![seed] },
            Step::Traverse {
                edge_type: Some(RRF_MENTIONS_EDGE_TYPE.to_string()),
                direction: EdgeDirection::Outgoing,
                temporal: None,
            },
            Step::KHop {
                edge_type: Some(relation.to_string()),
                max: k_hop_max,
                predicate: None,
                weight: None,
                order_by_weight: false,
                temporal: None,
            },
            Step::Traverse {
                edge_type: Some(RRF_MENTIONS_EDGE_TYPE.to_string()),
                direction: EdgeDirection::Incoming,
                temporal: None,
            },
        ];
        self.node_ids_or_empty(&steps, now_ms)
    }

    /// Structural bridge from a seed, returning each reached candidate paired with
    /// the best (MAX) product of edge effective weights along its route (P13). The
    /// candidate SET is identical to `bridge_structural`: this is the same layered
    /// BFS (edge_type None, outgoing, no predicate, `!visited` push rule), only it
    /// also threads a per-node weight product. With is_noop params every product is
    /// 1.0, so the factor is neutral and the score matches the unweighted path.
    fn bridge_structural_weighted(
        &self,
        seed: NodeId,
        k_hop_max: u32,
        params: &WeightParams,
        now_ms: u64,
    ) -> Vec<(NodeId, f64)> {
        let mut visited: HashSet<NodeId> = HashSet::new();
        visited.insert(seed);
        let mut best: HashMap<NodeId, f64> = HashMap::new();
        // BFS order membership, mirroring step_k_hop's frontier progression.
        let mut order: Vec<NodeId> = Vec::new();
        let mut current: Vec<(NodeId, f64)> = vec![(seed, 1.0)];
        for _ in 0..k_hop_max {
            if current.is_empty() {
                break;
            }
            let mut next: Vec<(NodeId, f64)> = Vec::new();
            for (node, node_w) in &current {
                for (cand, edge_w) in
                    self.store.weighted_neighbors(*node, None, EdgeDirection::Outgoing, params, now_ms)
                {
                    let cand_w = node_w * edge_w;
                    if visited.contains(&cand) {
                        // Improve the best product without re-expanding (membership
                        // stays exactly bridge_structural's set).
                        best.entry(cand)
                            .and_modify(|cur| {
                                if cand_w > *cur {
                                    *cur = cand_w;
                                }
                            })
                            .or_insert(cand_w);
                        continue;
                    }
                    visited.insert(cand);
                    best.insert(cand, cand_w);
                    order.push(cand);
                    next.push((cand, cand_w));
                }
            }
            current = next;
        }
        order
            .into_iter()
            .map(|id| {
                let w = best.get(&id).copied().unwrap_or(1.0);
                (id, w)
            })
            .collect()
    }

    /// Run a sub-plan's steps and return its node ids; an empty list on any
    /// non-node frontier (a bridge sub-plan always ends on nodes here).
    fn node_ids_or_empty(&self, steps: &[Step], now_ms: u64) -> Vec<NodeId> {
        match self.run_steps(steps, now_ms) {
            Ok(Frontier::Nodes(ids)) => ids,
            _ => Vec::new(),
        }
    }

    /// Materialize a frontier into the requested result kind, erroring on a
    /// kind mismatch between the final frontier and the plan's return kind.
    fn materialize(
        &self,
        frontier: Frontier,
        return_kind: &ReturnKind,
    ) -> Result<QueryResult, HybridQueryError> {
        match (frontier, return_kind) {
            (Frontier::Nodes(ids), ReturnKind::Nodes) => {
                let rows = ids
                    .into_iter()
                    .map(|id| NodeRecord {
                        id,
                        node: self.store.get_node(id),
                    })
                    .collect();
                Ok(QueryResult::Nodes(rows))
            }
            (Frontier::Edges(ids), ReturnKind::Edges) => {
                // Drop ids that no longer resolve to a live edge.
                let edges = ids
                    .into_iter()
                    .filter_map(|id| self.store.get_edge(id))
                    .collect();
                Ok(QueryResult::Edges(edges))
            }
            (Frontier::Paths(paths), ReturnKind::Paths) => Ok(QueryResult::Paths(paths)),
            _ => Err(HybridQueryError::InvalidPlan(
                "final frontier kind does not match the plan return kind".into(),
            )),
        }
    }

    /// Drive a step list to its final frontier. Also used for nested sub-plans.
    /// `now_ms` is the per-query instant threaded into any weight-aware step.
    fn run_steps(&self, steps: &[Step], now_ms: u64) -> Result<Frontier, HybridQueryError> {
        let mut frontier = Frontier::Nodes(Vec::new());
        for step in steps {
            frontier = self.apply_step(step, frontier, now_ms)?;
        }
        Ok(frontier)
    }

    /// Resolve a sub-plan to its node ids, erroring if it is not node-shaped.
    fn run_subplan_nodes(
        &self,
        plan: &QueryPlan,
        now_ms: u64,
    ) -> Result<Vec<NodeId>, HybridQueryError> {
        match self.run_steps(&plan.steps, now_ms)? {
            Frontier::Nodes(ids) => Ok(ids),
            _ => Err(HybridQueryError::InvalidPlan(
                "sub-query must produce a node frontier".into(),
            )),
        }
    }

    /// Apply one step to the current frontier. `now_ms` is the per-query instant
    /// used by any weight-aware step (unused by the unweighted ones).
    fn apply_step(
        &self,
        step: &Step,
        frontier: Frontier,
        now_ms: u64,
    ) -> Result<Frontier, HybridQueryError> {
        match step {
            Step::VectorSimilar {
                vector,
                k,
                ef_search,
            } => self.step_vector_similar(vector, *k, *ef_search, frontier),
            Step::VectorRank {
                vector,
                k,
                on_missing,
            } => self.step_vector_rank(vector, *k, *on_missing, frontier),
            Step::VectorMath { op, k, on_missing } => {
                self.step_vector_math(op, *k, *on_missing, frontier)
            }
            Step::FromNodes { nodes } => Ok(Frontier::Nodes(dedup_ids(nodes.iter().copied()))),
            Step::ScanByFilter {
                is_entity,
                label,
                predicate,
            } => self.step_scan_by_filter(*is_entity, label.clone(), predicate.clone(), frontier),
            Step::Traverse {
                edge_type,
                direction,
                temporal,
            } => self.step_traverse(
                edge_type.as_deref(),
                *direction,
                temporal.as_ref(),
                frontier,
                now_ms,
            ),
            Step::KHop {
                edge_type,
                max,
                predicate,
                weight,
                order_by_weight,
                temporal,
            } => self.step_k_hop(
                edge_type.as_deref(),
                *max,
                predicate.as_ref(),
                weight.as_ref(),
                *order_by_weight,
                temporal.as_ref(),
                frontier,
                now_ms,
            ),
            Step::ShortestPath {
                edge_types,
                target,
                weighted,
                weight,
                temporal,
            } => self.step_shortest_path(
                edge_types,
                *target,
                *weighted,
                weight.as_ref(),
                temporal.as_ref(),
                frontier,
                now_ms,
            ),
            Step::MutualNeighbors { other } => self.step_mutual_neighbors(other, frontier, now_ms),
            Step::Intersect { other } => self.step_intersect(other, frontier, now_ms),
            Step::Union { other } => self.step_union(other, frontier, now_ms),
            Step::Filter { predicate } => self.step_filter(predicate, frontier),
            Step::CollectEdges {
                edge_type,
                direction,
            } => self.step_collect_edges(edge_type.as_deref(), *direction, frontier),
            Step::Limit { n } => Ok(limit_frontier(frontier, *n)),
        }
    }

    // ── Step implementations ────────────────────────────────────────────

    fn step_vector_similar(
        &self,
        vector: &[f32],
        k: usize,
        ef_search: Option<usize>,
        frontier: Frontier,
    ) -> Result<Frontier, HybridQueryError> {
        // Must seed an empty frontier: a vector search cannot follow prior work.
        match frontier {
            Frontier::Nodes(ref ids) if ids.is_empty() => {}
            _ => {
                return Err(HybridQueryError::InvalidPlan(
                    "vector_similar must seed an empty frontier".into(),
                ))
            }
        }
        let results = self.index.search(vector, k, ef_search)?;
        // The bridge contract: a content node id equals its vector id.
        Ok(Frontier::Nodes(dedup_ids(
            results.into_iter().map(|r| NodeId(r.id)),
        )))
    }

    /// Rank the current node frontier by exact similarity to `vector`, keeping
    /// the top-k. This is the graph-first direction (graph scope -> vector
    /// ranking): EXACT scoring over the fixed frontier (recall 1.0, no ANN, no
    /// `ef_search`), output in ascending-distance order with the smaller node id
    /// breaking ties. `on_missing` governs frontier nodes with no vector.
    fn step_vector_rank(
        &self,
        vector: &[f32],
        k: usize,
        on_missing: OnMissingVector,
        frontier: Frontier,
    ) -> Result<Frontier, HybridQueryError> {
        // Consumes a node frontier only. The type-state builder guarantees this;
        // the guard is defensive for hand-built plans.
        let ids = match frontier {
            Frontier::Nodes(ids) => ids,
            _ => {
                return Err(HybridQueryError::InvalidPlan(
                    "vector_rank requires a node frontier".into(),
                ))
            }
        };

        metrics::counter!(VECTOR_RANK_TOTAL).increment(1);
        metrics::histogram!(VECTOR_RANK_FRONTIER_SIZE).record(ids.len() as f64);
        let started = Instant::now();

        // Cap guard: never silently truncate. An oversized frontier is a planning
        // problem the caller must narrow (or raise the cap deliberately).
        let cap = max_frontier_for_rank();
        if ids.len() > cap {
            metrics::counter!(VECTOR_RANK_CAP_EXCEEDED_TOTAL).increment(1);
            return Err(HybridQueryError::InvalidPlan(format!(
                "vector_rank frontier of {} exceeds the configured cap {}; narrow the graph scope before ranking, or raise SWARNDB_MAX_FRONTIER_FOR_RANK",
                ids.len(),
                cap
            )));
        }

        // Early outs: nothing to rank, or no slots requested.
        if ids.is_empty() || k == 0 {
            metrics::histogram!(VECTOR_RANK_LATENCY_SECONDS).record(started.elapsed().as_secs_f64());
            return Ok(Frontier::Nodes(Vec::new()));
        }

        // Build the metric once, from the index's own metric type, so VectorRank
        // scores a frontier identically to a normal search on this index.
        let dm = DistanceMetric::from_metric_type(self.index.metric_type());

        let quantized = self.index.is_quantized();
        let mut skipped: u64 = 0;
        let mut distance_computations: u64 = 0;

        // Bounded top-k accumulator: hold at most k kept candidates, evicting the
        // worst when full. Used by both the plain loop and the inline merge.
        let mut heap: BinaryHeap<RankKey> = BinaryHeap::with_capacity(k.min(ids.len()) + 1);

        // The quantized path routes content nodes (those in the index) through
        // search_with_candidates for SQ8 consistency. Inline-embedding nodes (not
        // in the index) are still scored exactly and merged into the same heap.
        let mut quantized_candidates: Vec<u64> = Vec::new();

        for node_id in &ids {
            let id = node_id.0;
            // P09.6 precedence (ids are globally disjoint across content vectors and
            // entity nodes via the unified id authority, so the bridge cannot mis-resolve):
            //   1. inline embedding on the typed node -> score with dm.compute;
            //   2. else the index bridge (NodeId == VectorId) -> get_vector / quantized;
            //   3. else MISSING (on_missing).
            if let Some(node_vec) = self.store.get_node(*node_id).and_then(|n| n.embedding) {
                distance_computations += 1;
                let dist = dm.compute(vector, &node_vec);
                heap_push_bounded(&mut heap, RankKey { dist, id }, k);
                continue;
            }
            if self.index.contains(id) {
                if quantized {
                    quantized_candidates.push(id);
                } else {
                    let node_vec = self.index.get_vector(id)?;
                    distance_computations += 1;
                    let dist = dm.compute(vector, &node_vec);
                    heap_push_bounded(&mut heap, RankKey { dist, id }, k);
                }
                continue;
            }
            match on_missing {
                OnMissingVector::Error => {
                    return Err(HybridQueryError::InvalidPlan(format!(
                        "vector_rank: node {} has no vector and on_missing=error", id
                    )))
                }
                OnMissingVector::Skip => skipped += 1,
            }
        }

        // Quantized content nodes: score the whole candidate set exactly over the
        // quantized-then-rescored path (no ANN descent), then merge its top-k into
        // the same bounded heap so the final order is the union top-k.
        if quantized && !quantized_candidates.is_empty() {
            let scored =
                self.index
                    .search_with_candidates(vector, k, &quantized_candidates, None)?;
            for r in scored {
                heap_push_bounded(&mut heap, RankKey { dist: r.score, id: r.id }, k);
            }
        }

        metrics::counter!(VECTOR_RANK_SKIPPED_MISSING_VECTOR_TOTAL).increment(skipped);
        metrics::counter!(VECTOR_RANK_DISTANCE_COMPUTATIONS_TOTAL)
            .increment(distance_computations);

        // Final order: ascending distance, smaller id breaking ties. The heap holds
        // the kept candidates; drain and sort directly (each id was scored once, so
        // the output is unique). Do NOT route through the OrderedSet/dedup path:
        // that would discard the ranking.
        let mut kept: Vec<RankKey> = heap.into_vec();
        kept.sort_by(|a, b| {
            a.dist
                .total_cmp(&b.dist)
                .then_with(|| a.id.cmp(&b.id))
        });
        let ranked_ids: Vec<NodeId> = kept.into_iter().map(|rk| NodeId(rk.id)).collect();

        metrics::histogram!(VECTOR_RANK_LATENCY_SECONDS).record(started.elapsed().as_secs_f64());
        Ok(Frontier::Nodes(ranked_ids))
    }

    /// Materialize the frontier's raw f32 vectors for the vector-math ops (P17),
    /// honoring the same per-node fetch precedence as VectorRank:
    ///   1. inline embedding on the typed node;
    ///   2. else the index bridge (`NodeId == VectorId`) via `index.get_vector`,
    ///      which reconstructs the SQ8 vector when the index is quantized;
    ///   3. else MISSING -> `on_missing` (`Skip` counts a skip; `Error` fails).
    /// The math ops need RAW vectors (there is no per-vector query to rescore
    /// against for the quantized batch path), so unlike VectorRank this always
    /// reconstructs via `get_vector`; the cost is acceptable because the frontier
    /// is graph-scoped and bounded by `max_frontier_for_rank()`. Returns the
    /// resolved `(NodeId, Vec<f32>)` pairs in frontier order plus the skip count.
    fn fetch_frontier_vectors(
        &self,
        ids: &[NodeId],
        on_missing: OnMissingVector,
    ) -> Result<(Vec<(NodeId, Vec<f32>)>, u64), HybridQueryError> {
        let mut pairs: Vec<(NodeId, Vec<f32>)> = Vec::with_capacity(ids.len());
        let mut skipped: u64 = 0;
        for node_id in ids {
            let id = node_id.0;
            if let Some(node_vec) = self.store.get_node(*node_id).and_then(|n| n.embedding) {
                pairs.push((*node_id, node_vec));
                continue;
            }
            if self.index.contains(id) {
                // get_vector reconstructs the stored vector (SQ8-aware).
                let node_vec = self.index.get_vector(id)?;
                pairs.push((*node_id, node_vec));
                continue;
            }
            match on_missing {
                OnMissingVector::Error => {
                    return Err(HybridQueryError::InvalidPlan(format!(
                        "vector_math: node {} has no vector and on_missing=error",
                        id
                    )))
                }
                OnMissingVector::Skip => skipped += 1,
            }
        }
        Ok((pairs, skipped))
    }

    /// Run a vector-math operation over the current node frontier, keeping top-k
    /// (P17). Frontier-consuming (graph scope -> vector arithmetic): the graph has
    /// already fixed the candidate set, so each op runs over exactly these nodes.
    /// Reuses the VectorRank fetch precedence and frontier cap. The ranking metric
    /// is the index's CONFIGURED metric for Analogy/Centroid/Interpolate/Isolation;
    /// Diversity (MMR) and Cone use their ops' OWN internal cosine, by design.
    fn step_vector_math(
        &self,
        op: &VectorMathOp,
        k: usize,
        on_missing: OnMissingVector,
        frontier: Frontier,
    ) -> Result<Frontier, HybridQueryError> {
        let ids = expect_nodes(frontier, "vector_math requires a node frontier")?;

        let op_label = vector_math_op_label(op);
        metrics::counter!(VECTOR_MATH_TOTAL, "op" => op_label).increment(1);
        metrics::histogram!(VECTOR_MATH_FRONTIER_SIZE, "op" => op_label).record(ids.len() as f64);
        let started = Instant::now();

        // Cap guard: never silently truncate an oversized frontier (same contract
        // as VectorRank). The caller must narrow the graph scope or raise the cap.
        let cap = max_frontier_for_rank();
        if ids.len() > cap {
            metrics::counter!(VECTOR_MATH_CAP_EXCEEDED_TOTAL, "op" => op_label).increment(1);
            return Err(HybridQueryError::InvalidPlan(format!(
                "vector_math frontier of {} exceeds the configured cap {}; narrow the graph scope before the op, or raise SWARNDB_MAX_FRONTIER_FOR_RANK",
                ids.len(),
                cap
            )));
        }

        // Materialize raw vectors with the shared precedence + skip accounting.
        let (pairs, skipped) = self.fetch_frontier_vectors(&ids, on_missing)?;
        metrics::counter!(VECTOR_MATH_SKIPPED_MISSING_VECTOR_TOTAL, "op" => op_label)
            .increment(skipped);

        // The configured metric, used by the metric-ranked ops (analogy, centroid,
        // interpolate). Isolation builds its own GhostDetector with this metric.
        let dm = DistanceMetric::from_metric_type(self.index.metric_type());

        // Borrowed-slice view for the ops that take &[(VectorId, &[f32])].
        let candidates: Vec<(u64, &[f32])> =
            pairs.iter().map(|(id, v)| (id.0, v.as_slice())).collect();

        let ranked_ids: Vec<NodeId> = match op {
            VectorMathOp::Analogy { a, b, c } => {
                // derived = a - b + c; None on dimension mismatch among a/b/c.
                let derived = AnalogyComputer::analogy(a, b, c).ok_or_else(|| {
                    HybridQueryError::InvalidPlan(
                        "vector_math analogy: a, b, c must share one dimension".into(),
                    )
                })?;
                rank_ascending_by_metric(&dm, &derived, &pairs, k)
            }
            VectorMathOp::Diversity { query, lambda } => {
                // MMR selection order; min(k, eligible) length; internal cosine.
                DiversitySampler::mmr(query, &candidates, k, *lambda)
                    .into_iter()
                    .map(|r| NodeId(r.id))
                    .collect()
            }
            VectorMathOp::Cone {
                direction,
                aperture_radians,
            } => {
                // Within-cone only, already sorted by cosine desc (== angle asc);
                // cap at k. Internal cosine geometry.
                ConeSearch::search(direction, *aperture_radians, &candidates)
                    .into_iter()
                    .take(k)
                    .map(|r| NodeId(r.id))
                    .collect()
            }
            VectorMathOp::Isolation { centroids } => {
                // min distance to any centroid (configured metric); top-k MOST
                // isolated (descending min-distance). threshold unused by
                // isolation_scores. Empty centroids -> all INFINITY (still ranked).
                let detector = GhostDetector::new(0.0, self.index.metric_type());
                let mut scored = detector.isolation_scores(&candidates, centroids);
                // Descending isolation_score, smaller id breaking ties for
                // deterministic order (NaN sorts last via total_cmp).
                scored.sort_by(|x, y| {
                    y.isolation_score
                        .total_cmp(&x.isolation_score)
                        .then_with(|| x.id.cmp(&y.id))
                });
                scored
                    .into_iter()
                    .take(k)
                    .map(|r| NodeId(r.id))
                    .collect()
            }
            VectorMathOp::Centroid {} => {
                // Centroid of the frontier's OWN vectors, then rank ascending
                // distance to it (configured metric). None on empty/dim-mismatch:
                // an empty frontier yields an empty ranked result (no error).
                let vecs: Vec<&[f32]> = pairs.iter().map(|(_, v)| v.as_slice()).collect();
                match CentroidComputer::compute(&vecs) {
                    Some(centroid) => rank_ascending_by_metric(&dm, &centroid, &pairs, k),
                    None => Vec::new(),
                }
            }
            VectorMathOp::Interpolate { a, b, t } => {
                // point = slerp(a,b,t) (lerp fallback when nearly parallel). slerp
                // returns None when t is outside [0,1] or a/b dims mismatch; surface
                // a clean query error rather than a silent empty result.
                let point = Interpolator::slerp(a, b, *t).ok_or_else(|| {
                    HybridQueryError::InvalidPlan(
                        "vector_math interpolate: t must be in [0,1] and a, b must share one dimension".into(),
                    )
                })?;
                rank_ascending_by_metric(&dm, &point, &pairs, k)
            }
        };

        metrics::histogram!(VECTOR_MATH_LATENCY_SECONDS, "op" => op_label)
            .record(started.elapsed().as_secs_f64());
        Ok(Frontier::Nodes(ranked_ids))
    }

    /// Seed the frontier by scanning the graph store for nodes matching an
    /// optional kind / entity label / property condition. A SOURCE step: it must
    /// seed an EMPTY frontier (no prior work), mirroring `vector_similar`. The
    /// scan reuses the store's paged node scan with a `NodePageFilter`, so the
    /// property condition (including structural incident-edge-count terms) is
    /// evaluated against the store, and paging keeps it memory-bounded. The
    /// resulting frontier is in ascending node-id order, deduped.
    fn step_scan_by_filter(
        &self,
        is_entity: Option<bool>,
        label: Option<String>,
        predicate: Option<super::predicate::Predicate>,
        frontier: Frontier,
    ) -> Result<Frontier, HybridQueryError> {
        // Must seed an empty frontier: a scan cannot follow prior work.
        match frontier {
            Frontier::Nodes(ref ids) if ids.is_empty() => {}
            _ => {
                return Err(HybridQueryError::InvalidPlan(
                    "scan_by_filter must seed an empty frontier".into(),
                ))
            }
        }
        let filter = NodePageFilter {
            is_entity,
            label,
            predicate,
        };
        // Page through the whole matching set, walking the cursor to exhaustion.
        // SCAN_PAGE bounds each page; the frontier itself is the full match set.
        const SCAN_PAGE: usize = 10_000;
        let mut out = OrderedSet::new();
        let mut after = NodeId(0);
        loop {
            let page = self.store.nodes_page(after, SCAN_PAGE, &filter);
            for node in &page.items {
                out.push(node.id);
            }
            if !page.has_more || page.next_cursor == 0 {
                break;
            }
            after = NodeId(page.next_cursor);
        }
        Ok(Frontier::Nodes(out.into_vec()))
    }

    fn step_traverse(
        &self,
        edge_type: Option<&str>,
        direction: EdgeDirection,
        temporal: Option<&TemporalFilter>,
        frontier: Frontier,
        now_ms: u64,
    ) -> Result<Frontier, HybridQueryError> {
        let nodes = expect_nodes(frontier, "traverse requires a node frontier")?;
        let mut out = OrderedSet::new();
        for node in nodes {
            // Default-off (temporal None): the existing cached fast path,
            // byte-identical. Some(f): per-edge temporal expansion (uncached).
            match temporal {
                None => {
                    for neighbor in self.store.neighbors(node, edge_type, direction) {
                        out.push(neighbor);
                    }
                }
                Some(f) => {
                    for neighbor in
                        self.store.neighbors_temporal(node, edge_type, direction, f, now_ms)
                    {
                        out.push(neighbor);
                    }
                }
            }
        }
        Ok(Frontier::Nodes(out.into_vec()))
    }

    fn step_k_hop(
        &self,
        edge_type: Option<&str>,
        max: u32,
        predicate: Option<&super::predicate::Predicate>,
        weight: Option<&WeightParams>,
        order_by_weight: bool,
        temporal: Option<&TemporalFilter>,
        frontier: Frontier,
        now_ms: u64,
    ) -> Result<Frontier, HybridQueryError> {
        // Opt-in weight-aware path (P13). Same node membership as the unweighted
        // BFS below, only reordered by descending accumulated edge weight.
        if order_by_weight {
            return self
                .step_k_hop_weighted(edge_type, max, predicate, weight, temporal, frontier, now_ms);
        }
        let start = expect_nodes(frontier, "k_hop requires a node frontier")?;
        // Start nodes are excluded from the output but seed the visited set.
        let mut visited: HashSet<NodeId> = start.iter().copied().collect();
        let mut result = OrderedSet::new();
        let mut current = start;
        // k_hop has no direction param; outgoing matches the report-to use case.
        for _ in 0..max {
            if current.is_empty() {
                break;
            }
            let mut next: Vec<NodeId> = Vec::new();
            for node in &current {
                // Default-off (temporal None): cached neighbors() fast path,
                // byte-identical. Some(f): per-edge temporal expansion (uncached).
                let cands = match temporal {
                    None => self.store.neighbors(*node, edge_type, EdgeDirection::Outgoing),
                    Some(f) => self.store.neighbors_temporal(
                        *node,
                        edge_type,
                        EdgeDirection::Outgoing,
                        f,
                        now_ms,
                    ),
                };
                for cand in cands {
                    if visited.contains(&cand) {
                        continue;
                    }
                    if let Some(pred) = predicate {
                        // Unmaterialized candidates fail property predicates but pass Always.
                        // Store-aware eval so structural terms resolve; pure property
                        // predicates behave exactly as before.
                        let accepted = match self.store.get_node(cand) {
                            Some(node) => pred.eval_node_with_store(&node, self.store),
                            None => matches!(pred, super::predicate::Predicate::Always),
                        };
                        if !accepted {
                            continue;
                        }
                    }
                    visited.insert(cand);
                    result.push(cand);
                    next.push(cand);
                }
            }
            current = next;
        }
        Ok(Frontier::Nodes(result.into_vec()))
    }

    /// Weight-aware k-hop (P13): the SAME layered BFS as `step_k_hop`, producing
    /// the SAME node membership (the `!visited` push rule is unchanged and the
    /// predicate filter is identical), but tracking a per-node accumulated weight
    /// (parent_weight * edge_weight, kept at its MAX) and returning the result
    /// ordered by accumulated weight descending, NodeId ascending on ties.
    fn step_k_hop_weighted(
        &self,
        edge_type: Option<&str>,
        max: u32,
        predicate: Option<&super::predicate::Predicate>,
        weight: Option<&WeightParams>,
        temporal: Option<&TemporalFilter>,
        frontier: Frontier,
        now_ms: u64,
    ) -> Result<Frontier, HybridQueryError> {
        let start = expect_nodes(frontier, "k_hop requires a node frontier")?;
        metrics::counter!(WEIGHTED_KHOP_TOTAL).increment(1);
        let started = Instant::now();
        let p = weight.cloned().unwrap_or_default();
        // Start nodes seed the visited set (excluded from output), weight 1.0.
        let mut visited: HashSet<NodeId> = start.iter().copied().collect();
        // First-seen membership in BFS order (identical to the unweighted path).
        let mut result = OrderedSet::new();
        // Per-node best accumulated weight; improving it never changes membership.
        let mut acc_weight: HashMap<NodeId, f64> = HashMap::new();
        // Seed weights are 1.0; carried so children get parent_weight * edge_weight.
        let mut current: Vec<(NodeId, f64)> = start.into_iter().map(|n| (n, 1.0)).collect();
        for _ in 0..max {
            if current.is_empty() {
                break;
            }
            let mut next: Vec<(NodeId, f64)> = Vec::new();
            for (node, node_w) in &current {
                // Default-off (temporal None): weighted_neighbors_temporal with
                // filter None delegates straight to weighted_neighbors, the cached
                // path, byte-identical. Some(f): uncached per-edge temporal expansion.
                for (cand, edge_w) in self.store.weighted_neighbors_temporal(
                    *node,
                    edge_type,
                    EdgeDirection::Outgoing,
                    &p,
                    temporal,
                    now_ms,
                ) {
                    let cand_w = node_w * edge_w;
                    if visited.contains(&cand) {
                        // Already a member: only improve the recorded weight (MAX).
                        // Do NOT re-expand, so membership matches the unweighted BFS.
                        acc_weight
                            .entry(cand)
                            .and_modify(|cur| {
                                if cand_w > *cur {
                                    *cur = cand_w;
                                }
                            })
                            .or_insert(cand_w);
                        continue;
                    }
                    if let Some(pred) = predicate {
                        // Same predicate gate as the unweighted path (store-aware so
                        // structural terms resolve; pure property terms unchanged).
                        let accepted = match self.store.get_node(cand) {
                            Some(node) => pred.eval_node_with_store(&node, self.store),
                            None => matches!(pred, super::predicate::Predicate::Always),
                        };
                        if !accepted {
                            continue;
                        }
                    }
                    visited.insert(cand);
                    result.push(cand);
                    acc_weight.insert(cand, cand_w);
                    next.push((cand, cand_w));
                }
            }
            current = next;
        }
        // Order by accumulated weight desc, NodeId asc on ties (total_cmp: no NaN hazard).
        let mut ordered = result.into_vec();
        ordered.sort_by(|a, b| {
            let wa = acc_weight.get(a).copied().unwrap_or(0.0);
            let wb = acc_weight.get(b).copied().unwrap_or(0.0);
            wb.total_cmp(&wa).then_with(|| a.0.cmp(&b.0))
        });
        metrics::histogram!(WEIGHTED_KHOP_LATENCY_SECONDS).record(started.elapsed().as_secs_f64());
        Ok(Frontier::Nodes(ordered))
    }

    fn step_shortest_path(
        &self,
        edge_types: &[String],
        target: NodeId,
        weighted: bool,
        weight: Option<&WeightParams>,
        temporal: Option<&TemporalFilter>,
        frontier: Frontier,
        now_ms: u64,
    ) -> Result<Frontier, HybridQueryError> {
        let seeds = expect_nodes(frontier, "shortest_path requires a node frontier")?;
        // Opt-in weighted path (P13): Dijkstra over 1/effective_weight costs. The
        // default (weighted == false) is the unchanged unweighted BFS. The opt-in
        // temporal filter (P17) threads into both branches; None leaves each path
        // byte-identical.
        let path = if weighted {
            metrics::counter!(WEIGHTED_SHORTEST_PATH_TOTAL).increment(1);
            let p = weight.cloned().unwrap_or_default();
            self.weighted_shortest_path(&seeds, edge_types, target, &p, temporal, now_ms)
        } else {
            self.bfs_shortest_path(&seeds, edge_types, target, temporal, now_ms)
        };
        let paths = match path {
            Some(p) => vec![Path { nodes: p }],
            None => Vec::new(),
        };
        Ok(Frontier::Paths(paths))
    }

    /// Unweighted multi-source BFS to `target`, restricted to `edge_types`
    /// (empty allowlist means any type). Returns the node chain if reachable.
    fn bfs_shortest_path(
        &self,
        seeds: &[NodeId],
        edge_types: &[String],
        target: NodeId,
        temporal: Option<&TemporalFilter>,
        now_ms: u64,
    ) -> Option<Vec<NodeId>> {
        // The edge-type allowlist gate, plus the opt-in temporal gate (P17). With
        // temporal None the closure reduces to the original allowlist check, so the
        // BFS is byte-identical; the BFS already holds the full Edge here, so no new
        // store method is needed.
        let allow = |edge: &Edge| -> bool {
            let type_ok =
                edge_types.is_empty() || edge_types.iter().any(|t| t == edge.edge_type.as_str());
            if !type_ok {
                return false;
            }
            match temporal {
                None => true,
                Some(f) => vf_graph::edge_passes_temporal(edge, f, now_ms),
            }
        };
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for seed in seeds {
            if visited.insert(*seed) {
                queue.push_back(*seed);
                if *seed == target {
                    return Some(vec![*seed]);
                }
            }
        }
        while let Some(node) = queue.pop_front() {
            for edge in self.store.edges_for_node(node, EdgeDirection::Both) {
                if !allow(&edge) {
                    continue;
                }
                let other = if edge.source == node {
                    edge.target
                } else {
                    edge.source
                };
                if !visited.insert(other) {
                    continue;
                }
                parent.insert(other, node);
                if other == target {
                    return Some(reconstruct_path(&parent, other));
                }
                queue.push_back(other);
            }
        }
        None
    }

    /// Weighted multi-source shortest path (P13): Dijkstra to `target` minimizing
    /// total cost, where each edge costs 1/effective_weight (stronger edges are
    /// cheaper). Same undirected reachability as `bfs_shortest_path` (EdgeDirection
    /// ::Both). `edge_types` empty selects all types; otherwise per-type weighted
    /// neighbors are merged keeping the MAX weight per neighbor (set-filter
    /// equivalent). Determinism: cost ties break on NodeId ascending. Returns the
    /// node chain if reachable.
    fn weighted_shortest_path(
        &self,
        seeds: &[NodeId],
        edge_types: &[String],
        target: NodeId,
        params: &WeightParams,
        temporal: Option<&TemporalFilter>,
        now_ms: u64,
    ) -> Option<Vec<NodeId>> {
        // Best settled cost per node; a node is final once popped from the heap.
        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        let mut settled: HashSet<NodeId> = HashSet::new();
        // Min-heap: smallest cost first (Reverse over the total_cmp newtype), the
        // NodeId tie-break inside CostKey makes pops deterministic.
        let mut heap: BinaryHeap<Reverse<CostKey>> = BinaryHeap::new();
        for seed in seeds {
            // Multi-source: every seed starts at cost 0; keep the best on dupes.
            let entry = dist.entry(*seed).or_insert(f64::INFINITY);
            if 0.0 < *entry {
                *entry = 0.0;
                heap.push(Reverse(CostKey { cost: 0.0, node: *seed }));
            }
        }
        while let Some(Reverse(CostKey { cost, node })) = heap.pop() {
            if !settled.insert(node) {
                // A stale heap entry for an already-settled node; skip it.
                continue;
            }
            if node == target {
                return Some(reconstruct_path(&parent, target));
            }
            for (neighbor, w) in
                self.merged_weighted_neighbors(node, edge_types, params, temporal, now_ms)
            {
                if settled.contains(&neighbor) {
                    continue;
                }
                // Cost of this edge: 1/weight, guarded finite by effective_weight.
                let edge_cost = 1.0 / w;
                let nd = cost + edge_cost;
                let cur = dist.entry(neighbor).or_insert(f64::INFINITY);
                if nd < *cur {
                    *cur = nd;
                    parent.insert(neighbor, node);
                    heap.push(Reverse(CostKey { cost: nd, node: neighbor }));
                }
            }
        }
        None
    }

    /// Weighted neighbors of `node` filtered to `edge_types` (empty = all types),
    /// merged across the allowed types keeping the MAX weight per neighbor. Used
    /// by the weighted shortest path; mirrors the BFS edge-type allowlist.
    fn merged_weighted_neighbors(
        &self,
        node: NodeId,
        edge_types: &[String],
        params: &WeightParams,
        temporal: Option<&TemporalFilter>,
        now_ms: u64,
    ) -> Vec<(NodeId, f64)> {
        // temporal None routes through weighted_neighbors_temporal(.., None, ..),
        // which delegates straight to the cached weighted_neighbors, byte-identical.
        if edge_types.is_empty() {
            return self.store.weighted_neighbors_temporal(
                node,
                None,
                EdgeDirection::Both,
                params,
                temporal,
                now_ms,
            );
        }
        let mut acc: HashMap<NodeId, f64> = HashMap::new();
        for t in edge_types {
            for (n, w) in self.store.weighted_neighbors_temporal(
                node,
                Some(t.as_str()),
                EdgeDirection::Both,
                params,
                temporal,
                now_ms,
            ) {
                acc.entry(n)
                    .and_modify(|cur| {
                        if w > *cur {
                            *cur = w;
                        }
                    })
                    .or_insert(w);
            }
        }
        let mut out: Vec<(NodeId, f64)> = acc.into_iter().collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    fn step_mutual_neighbors(
        &self,
        other: &QueryPlan,
        frontier: Frontier,
        now_ms: u64,
    ) -> Result<Frontier, HybridQueryError> {
        let a = expect_nodes(frontier, "mutual_neighbors requires a node frontier")?;
        let b = self.run_subplan_nodes(other, now_ms)?;
        // Neighbours of set A, in A-neighbour first-seen order.
        let mut a_neighbors = OrderedSet::new();
        for node in &a {
            for n in self.store.neighbors(*node, None, EdgeDirection::Both) {
                a_neighbors.push(n);
            }
        }
        // Neighbours of set B as a membership set.
        let mut b_neighbors: HashSet<NodeId> = HashSet::new();
        for node in &b {
            for n in self.store.neighbors(*node, None, EdgeDirection::Both) {
                b_neighbors.insert(n);
            }
        }
        let result: Vec<NodeId> = a_neighbors
            .into_vec()
            .into_iter()
            .filter(|n| b_neighbors.contains(n))
            .collect();
        Ok(Frontier::Nodes(result))
    }

    fn step_intersect(
        &self,
        other: &QueryPlan,
        frontier: Frontier,
        now_ms: u64,
    ) -> Result<Frontier, HybridQueryError> {
        let a = expect_nodes(frontier, "intersect requires a node frontier")?;
        let b: HashSet<NodeId> = self.run_subplan_nodes(other, now_ms)?.into_iter().collect();
        let result: Vec<NodeId> = a.into_iter().filter(|n| b.contains(n)).collect();
        Ok(Frontier::Nodes(result))
    }

    fn step_union(
        &self,
        other: &QueryPlan,
        frontier: Frontier,
        now_ms: u64,
    ) -> Result<Frontier, HybridQueryError> {
        let a = expect_nodes(frontier, "union requires a node frontier")?;
        let b = self.run_subplan_nodes(other, now_ms)?;
        let mut out = OrderedSet::new();
        for n in a.into_iter().chain(b.into_iter()) {
            out.push(n);
        }
        Ok(Frontier::Nodes(out.into_vec()))
    }

    fn step_filter(
        &self,
        predicate: &super::predicate::Predicate,
        frontier: Frontier,
    ) -> Result<Frontier, HybridQueryError> {
        match frontier {
            Frontier::Nodes(ids) => {
                // Store-aware node eval so structural terms (incident-edge count)
                // resolve. Pure property predicates behave exactly as the
                // store-free path, so non-structural plans are unchanged.
                let kept = ids
                    .into_iter()
                    .filter(|id| match self.store.get_node(*id) {
                        Some(node) => predicate.eval_node_with_store(&node, self.store),
                        // Unmaterialized: only Always passes; property predicates fail.
                        None => matches!(predicate, super::predicate::Predicate::Always),
                    })
                    .collect();
                Ok(Frontier::Nodes(kept))
            }
            Frontier::Edges(ids) => {
                // Node-only structural terms (incident-edge count) resolve to None
                // on an edge frontier and would silently match nothing. Reject
                // them loudly instead of returning a misleading empty result.
                if predicate.needs_store() {
                    return Err(HybridQueryError::InvalidPlan(
                        "incident-edge-count is node-only".into(),
                    ));
                }
                let kept = ids
                    .into_iter()
                    .filter(|id| match self.store.get_edge(*id) {
                        Some(edge) => predicate.eval_edge(&edge),
                        None => matches!(predicate, super::predicate::Predicate::Always),
                    })
                    .collect();
                Ok(Frontier::Edges(kept))
            }
            Frontier::Paths(_) => Err(HybridQueryError::InvalidPlan(
                "filter is not defined on a path frontier".into(),
            )),
        }
    }

    fn step_collect_edges(
        &self,
        edge_type: Option<&str>,
        direction: EdgeDirection,
        frontier: Frontier,
    ) -> Result<Frontier, HybridQueryError> {
        let nodes = expect_nodes(frontier, "edges requires a node frontier")?;
        let mut out = OrderedSet::new();
        for node in nodes {
            for edge in self.store.edges_for_node(node, direction) {
                if let Some(t) = edge_type {
                    if edge.edge_type.as_str() != t {
                        continue;
                    }
                }
                out.push(edge.id);
            }
        }
        Ok(Frontier::Edges(out.into_vec()))
    }
}

/// Current unix-epoch millis, read once per query for weight-aware steps. A clock
/// error before the epoch yields 0 (recency decay then sees age 0, neutral).
fn now_unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ── VectorMath helpers (P17) ────────────────────────────────────────────────

/// The metric label for a vector-math op, shared across the VECTOR_MATH_* family.
fn vector_math_op_label(op: &VectorMathOp) -> &'static str {
    match op {
        VectorMathOp::Analogy { .. } => "analogy",
        VectorMathOp::Diversity { .. } => "diversity",
        VectorMathOp::Cone { .. } => "cone",
        VectorMathOp::Isolation { .. } => "isolation",
        VectorMathOp::Centroid { .. } => "centroid",
        VectorMathOp::Interpolate { .. } => "interpolate",
    }
}

/// Rank `pairs` by ascending distance to `target` under `dm`, keeping the top-k.
/// Mirrors VectorRank's bounded top-k contract exactly: a max-heap of capacity k
/// (the `RankKey` ordering makes the worst kept candidate evictable), final sort
/// ascending distance with the smaller node id breaking ties. Each frontier node
/// is scored once, so the output ids are unique.
fn rank_ascending_by_metric(
    dm: &DistanceMetric,
    target: &[f32],
    pairs: &[(NodeId, Vec<f32>)],
    k: usize,
) -> Vec<NodeId> {
    if pairs.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<RankKey> = BinaryHeap::with_capacity(k.min(pairs.len()) + 1);
    for (node_id, vec) in pairs {
        let dist = dm.compute(target, vec);
        heap_push_bounded(&mut heap, RankKey { dist, id: node_id.0 }, k);
    }
    let mut kept: Vec<RankKey> = heap.into_vec();
    kept.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
    kept.into_iter().map(|rk| NodeId(rk.id)).collect()
}

// ── VectorRank bounded top-k (P09.6) ────────────────────────────────────────

/// One scored frontier candidate for the bounded top-k heap.
///
/// The `BinaryHeap` is a MAX-heap by `Ord`, so its top is the candidate this
/// ordering ranks GREATEST, which we make the WORST kept candidate (cheap to
/// evict). "Worst" = the largest distance; on equal distance, the LARGER id.
/// Evicting the larger id on a tie keeps the smaller id, matching the final
/// smaller-id-breaks-ties contract. NaN distances sort greatest via
/// `total_cmp`, so a degenerate score is evicted first rather than poisoning
/// comparisons.
struct RankKey {
    dist: f32,
    id: u64,
}

impl PartialEq for RankKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for RankKey {}

impl PartialOrd for RankKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Greater = more evictable: larger distance first, then larger id.
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

// ── Weighted shortest-path heap key (P13) ───────────────────────────────────

/// A Dijkstra frontier entry ordered by total cost ascending, NodeId ascending
/// on equal cost. f64 is not Ord, so ordering goes through `f64::total_cmp`
/// (no NaN hazard); the queue wraps this in `Reverse` for a min-heap.
struct CostKey {
    cost: f64,
    node: NodeId,
}

impl PartialEq for CostKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for CostKey {}

impl PartialOrd for CostKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CostKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then_with(|| self.node.0.cmp(&other.node.0))
    }
}

/// Push `item` into a bounded max-heap of capacity `k`. Below capacity it is
/// inserted unconditionally; at capacity it replaces the current worst (the
/// heap top) only when it is strictly better (smaller per `RankKey::cmp`), so
/// the heap always holds the best k seen so far.
fn heap_push_bounded(heap: &mut BinaryHeap<RankKey>, item: RankKey, k: usize) {
    if heap.len() < k {
        heap.push(item);
    } else if let Some(worst) = heap.peek() {
        if item.cmp(worst) == std::cmp::Ordering::Less {
            heap.pop();
            heap.push(item);
        }
    }
}

// ── Frontier helpers ───────────────────────────────────────────────────────

/// Require a node frontier, erroring with `msg` otherwise.
fn expect_nodes(frontier: Frontier, msg: &str) -> Result<Vec<NodeId>, HybridQueryError> {
    match frontier {
        Frontier::Nodes(ids) => Ok(ids),
        _ => Err(HybridQueryError::InvalidPlan(msg.into())),
    }
}

/// Collect ids preserving first-seen order and dropping duplicates.
fn dedup_ids<I: IntoIterator<Item = NodeId>>(iter: I) -> Vec<NodeId> {
    let mut out = OrderedSet::new();
    for id in iter {
        out.push(id);
    }
    out.into_vec()
}

/// Truncate any frontier to its first `n` members.
fn limit_frontier(frontier: Frontier, n: usize) -> Frontier {
    match frontier {
        Frontier::Nodes(mut v) => {
            v.truncate(n);
            Frontier::Nodes(v)
        }
        Frontier::Edges(mut v) => {
            v.truncate(n);
            Frontier::Edges(v)
        }
        Frontier::Paths(mut v) => {
            v.truncate(n);
            Frontier::Paths(v)
        }
    }
}

// ── Reciprocal Rank Fusion (P07, opt-in ranking) ───────────────────────────

/// Map each id to its 1-based rank in an ordered list (first = rank 1),
/// keeping the first (best) position for any repeat.
fn rank_map(ranking: &[NodeId]) -> HashMap<NodeId, usize> {
    let mut ranks: HashMap<NodeId, usize> = HashMap::with_capacity(ranking.len());
    let mut pos = 0usize;
    for &id in ranking {
        if ranks.contains_key(&id) {
            continue;
        }
        pos += 1;
        ranks.insert(id, pos);
    }
    ranks
}

/// Fuse the vector ranking and a graph-proximity ranking of the pool with
/// standard Reciprocal Rank Fusion, returning the top-k by fused score.
///
/// fused(d) = sum over each ranking d appears in of 1 / (rrf_k + rank), equal
/// weight on both rankings. A candidate absent from a ranking adds no term for
/// it. Ties break on the pool's original order for a stable, reproducible cut.
fn rrf_fuse_topk(
    vector_ranking: &[NodeId],
    proximity: &HashMap<NodeId, f64>,
    pool: &[NodeId],
    spec: &RrfRankSpec,
) -> Vec<NodeId> {
    let rrf_k = spec.effective_rrf_k() as f64;
    let vec_ranks = rank_map(vector_ranking);

    // Graph ranking: pool candidates with a proximity score, ordered by score
    // descending, ties by the pool's original order. With damping off the score
    // is an exact route count (1.0 per route), so the order matches the prior
    // integer-count ordering byte-for-byte.
    let pool_order: HashMap<NodeId, usize> =
        pool.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut graph_order: Vec<NodeId> = pool
        .iter()
        .copied()
        .filter(|id| proximity.contains_key(id))
        .collect();
    graph_order.sort_by(|a, b| {
        let sa = proximity.get(a).copied().unwrap_or(0.0);
        let sb = proximity.get(b).copied().unwrap_or(0.0);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| pool_order[a].cmp(&pool_order[b]))
    });
    let graph_ranks = rank_map(&graph_order);

    // Score every pool member; keep the original pool index for a stable cut.
    let mut scored: Vec<(f64, usize, NodeId)> = Vec::with_capacity(pool.len());
    for (idx, &id) in pool.iter().enumerate() {
        let mut score = 0.0;
        if let Some(&r) = vec_ranks.get(&id) {
            score += 1.0 / (rrf_k + r as f64);
        }
        if let Some(&r) = graph_ranks.get(&id) {
            score += 1.0 / (rrf_k + r as f64);
        }
        scored.push((score, idx, id));
    }
    // Sort by fused score desc, then original pool index asc (stable tie-break).
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let mut out: Vec<NodeId> = scored.into_iter().map(|(_, _, id)| id).collect();
    if let Some(k) = spec.k {
        out.truncate(k);
    }
    out
}

/// Walk parent pointers back from `end` to a seed and return the seed-to-end chain.
fn reconstruct_path(parent: &HashMap<NodeId, NodeId>, end: NodeId) -> Vec<NodeId> {
    let mut chain = vec![end];
    let mut cur = end;
    while let Some(prev) = parent.get(&cur) {
        chain.push(*prev);
        cur = *prev;
    }
    chain.reverse();
    chain
}
