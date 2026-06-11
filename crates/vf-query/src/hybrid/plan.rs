// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Serializable hybrid query plan: an ordered list of steps plus the kind of
//! result to materialize. Plans travel on the wire and are produced by the
//! type-state builder.

use serde::{Deserialize, Serialize};

use vf_graph::{EdgeDirection, NodeId};

use super::predicate::Predicate;

// Re-export the query-time edge-weight parameters so plans carry them without a
// direct vf-graph dependency. Already derives Serialize/Deserialize, so the REST
// and gRPC surfaces get serde for free.
pub use vf_graph::weight::WeightParams;

// Re-export the opt-in traversal time-and-context filter so plans carry it
// without a direct vf-graph dependency, mirroring `WeightParams`. Owned by
// vf-graph; already derives Serialize/Deserialize, so REST and gRPC get serde
// for free. Absent on a step = byte-identical to today (ADR-007 R5).
pub use vf_graph::TemporalFilter;

/// What the executor materializes from the final frontier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReturnKind {
    Nodes,
    Edges,
    Paths,
}

/// A full hybrid query: an ordered pipeline of steps and a return kind.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryPlan {
    pub steps: Vec<Step>,
    pub return_kind: ReturnKind,
}

/// Opt-in graph-aware RRF ranking spec (P07). When present, the executor fuses
/// the vector seed ranking with a graph-proximity ranking of the candidate pool
/// using standard Reciprocal Rank Fusion. Absent means the ranking is off and
/// the plan runs unchanged. Per ADR-024 this is the OPT-IN path, not the default:
/// the default graph-augmented ranking is the `VectorRank` step (graph-first
/// scope-then-rank). RRF behaves byte-identically when explicitly requested.
///
/// Canonical defaults: `rrf_k` = 60 (Cormack et al. 2009), equal weight on the
/// two rankings, proximity from graph structure only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RrfRankSpec {
    /// Final top-k cut on the fused order. None means return the whole pool.
    pub k: Option<usize>,
    /// RRF constant in 1 / (rrf_k + rank). Canonical 60.
    pub rrf_k: u32,
    /// Bridge-hop bound for the entity-to-entity / content-to-content hop.
    pub k_hop_max: u32,
    /// Typed relation edge types for the entity-bridge middle hop. Empty selects
    /// the structural shape (content-to-content) instead.
    pub relation_edge_types: Vec<String>,
    /// Hub-aware damping strength (ADR-019). 0.0 = OFF: every bridge route counts
    /// a flat 1, byte-identical to the prior count-based proximity. > 0.0 weights
    /// each route by the product over its bridging nodes of
    /// 1 / (1 + hub_damping * ln(1 + degree)), so routes through high-degree hubs
    /// contribute less and hub-dense graphs stop flooding the pool.
    pub hub_damping: f64,
    /// Opt-in edge-quality weighting of the structural bridge (P13). None means
    /// today's flat behavior: every bridge edge counts the same and the score is
    /// byte-identical to the pre-P13 path. Some(params) multiplies into each
    /// candidate's contribution the best product of edge effective weights along
    /// its route from the seed, composing with hub_damping (both factors multiply).
    #[serde(default)]
    pub edge_weight: Option<WeightParams>,
}

/// The engine's native passage-to-entity link edge type (ADR-012). The bridge
/// hop walks out and back along this link when entity-bridge proximity is used.
pub const RRF_MENTIONS_EDGE_TYPE: &str = "mentions";

/// Canonical RRF constant; used when the spec carries a non-positive value.
pub const RRF_K_DEFAULT: u32 = 60;

/// Default bridge-hop bound when the spec carries a non-positive value.
pub const RRF_K_HOP_DEFAULT: u32 = 2;

impl RrfRankSpec {
    /// Resolve the RRF constant, applying the canonical default for 0.
    pub fn effective_rrf_k(&self) -> u32 {
        if self.rrf_k == 0 {
            RRF_K_DEFAULT
        } else {
            self.rrf_k
        }
    }

    /// Resolve the bridge-hop bound, applying the default for 0.
    pub fn effective_k_hop_max(&self) -> u32 {
        if self.k_hop_max == 0 {
            RRF_K_HOP_DEFAULT
        } else {
            self.k_hop_max
        }
    }

    /// True when hub-aware damping is enabled (a positive strength). At 0.0 the
    /// proximity score is the plain count of distinct bridge routes, unchanged.
    pub fn hub_damping_on(&self) -> bool {
        self.hub_damping > 0.0
    }
}

/// One stage in a hybrid pipeline. Steps consume the current frontier and
/// produce the next. Sub-query steps embed a nested [`QueryPlan`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Step {
    /// Seed the node frontier from a vector similarity search.
    VectorSimilar {
        vector: Vec<f32>,
        k: usize,
        ef_search: Option<usize>,
    },
    /// Seed the node frontier from explicit node ids.
    FromNodes { nodes: Vec<NodeId> },
    /// Seed the node frontier by scanning the graph store and keeping nodes that
    /// match an optional kind / entity label / property condition. A SOURCE step
    /// (no vector, no explicit ids): it produces the initial frontier that normal
    /// traversal chains then expand. All fields optional; an all-None scan yields
    /// every node. `is_entity` mirrors the page-filter kind flag (Some(true) =
    /// entity, Some(false) = content). Additive: legacy plans never carry it.
    ScanByFilter {
        #[serde(default)]
        is_entity: Option<bool>,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        predicate: Option<Predicate>,
    },
    /// One-hop neighbour expansion over the typed adjacency. `temporal` is opt-in
    /// (P17): None runs the existing cached fast path byte-identically; Some(f)
    /// expands only over edges valid at the query instant (and in `f`'s context).
    Traverse {
        edge_type: Option<String>,
        direction: EdgeDirection,
        #[serde(default)]
        temporal: Option<TemporalFilter>,
    },
    /// Bounded multi-hop expansion with an optional per-hop node predicate.
    /// `weight` and `order_by_weight` are opt-in (P13): both default off so legacy
    /// plans and existing constructors are byte-identical. With `order_by_weight`
    /// the node membership is unchanged, only the output order shifts to descending
    /// accumulated edge weight.
    KHop {
        edge_type: Option<String>,
        max: u32,
        predicate: Option<Predicate>,
        #[serde(default)]
        weight: Option<WeightParams>,
        #[serde(default)]
        order_by_weight: bool,
        /// Opt-in time-and-context filter (P17). None = existing fast path,
        /// byte-identical. Some(f) restricts every hop to edges valid at the
        /// query instant in `f`'s context.
        #[serde(default)]
        temporal: Option<TemporalFilter>,
    },
    /// Shortest path from any current node to a target. `weighted` and `weight`
    /// are opt-in (P13): both default off, so this stays the unweighted BFS unless
    /// a caller opts in. Weighted runs Dijkstra minimizing total 1/effective_weight.
    ShortestPath {
        edge_types: Vec<String>,
        target: NodeId,
        #[serde(default)]
        weighted: bool,
        #[serde(default)]
        weight: Option<WeightParams>,
        /// Opt-in time-and-context filter (P17). None = existing path search,
        /// byte-identical. Some(f) traverses only edges valid at the query
        /// instant in `f`'s context (applies to both BFS and weighted Dijkstra).
        #[serde(default)]
        temporal: Option<TemporalFilter>,
    },
    /// Common neighbours between the current node set and a sub-query result.
    MutualNeighbors { other: QueryPlan },
    /// Set intersection with a sub-query node result.
    Intersect { other: QueryPlan },
    /// Set union with a sub-query node result.
    Union { other: QueryPlan },
    /// Keep frontier members satisfying the predicate.
    Filter { predicate: Predicate },
    /// Collect incident edges of the current nodes into an edge frontier.
    CollectEdges {
        edge_type: Option<String>,
        direction: EdgeDirection,
    },
    /// Truncate the current frontier to the first `n` members.
    Limit { n: usize },
    /// Rank the current node frontier by similarity to `vector`, keep top-k.
    /// Frontier-consuming (graph -> vector direction). EXACT scoring over the
    /// fixed frontier (recall 1.0, no ANN, no ef_search). Returns nodes in
    /// ranked order: ascending distance (most similar first), smaller node id
    /// breaks ties. `on_missing` governs frontier nodes with no vector. This is
    /// the DEFAULT graph-augmented ranking step (graph-first scope-then-rank,
    /// ADR-024): the graph scopes the candidate set, then this ranks it exactly.
    VectorRank { vector: Vec<f32>, k: usize, on_missing: OnMissingVector },
    /// Run a vector-math operation over the current node frontier, keeping top-k
    /// (P17). Frontier-consuming (graph -> vector): the graph has already fixed
    /// the candidate set, so each op runs over exactly these nodes. Reuses the
    /// VectorRank fetch precedence (inline embedding -> `index.get_vector` ->
    /// `on_missing`; SQ8-aware via `get_vector`; bounded by the frontier cap).
    ///
    /// RANKING METRIC: Analogy, Centroid, Interpolate, and Isolation rank by the
    /// index's CONFIGURED metric. Diversity (MMR) and Cone use their ops' OWN
    /// internal cosine, inherent to MMR scoring and cone geometry; this is by
    /// design, not a bug.
    VectorMath { op: VectorMathOp, k: usize, on_missing: OnMissingVector },
}

/// A vector-arithmetic operation run over the current graph frontier (P17). Each
/// variant carries exactly the operands its op needs; the executor materializes
/// the frontier's raw vectors and applies the op, returning frontier nodes in the
/// per-op order documented below.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VectorMathOp {
    /// derived = a - b + c; rank frontier ASCENDING distance to derived (configured
    /// metric); top-k.
    Analogy { a: Vec<f32>, b: Vec<f32>, c: Vec<f32> },
    /// MMR over the frontier; relevance = similarity to `query`, penalty = max
    /// similarity to already-selected; output in MMR-selection order, length
    /// min(k, eligible). Uses MMR's OWN internal cosine, not the index metric.
    Diversity { query: Vec<f32>, lambda: f32 },
    /// Keep frontier nodes whose angle to `direction` <= `aperture_radians`; rank
    /// ASCENDING angle (== cosine descending); cap at k. Uses the op's internal cosine.
    Cone { direction: Vec<f32>, aperture_radians: f32 },
    /// Ghost/isolation: score each node = min distance to any centroid (configured
    /// metric); return the top-k MOST isolated (DESCENDING min-distance).
    Isolation { centroids: Vec<Vec<f32>> },
    /// Compute the centroid (mean) of the frontier's OWN vectors, then rank
    /// ASCENDING distance to that centroid (most representative, configured metric);
    /// top-k.
    Centroid {},
    /// point = slerp(a,b,t) (lerp fallback when nearly parallel); rank ASCENDING
    /// distance to point (configured metric); top-k. The executor returns a clean
    /// query error when `t` is outside [0,1] or `a`/`b` dims mismatch.
    Interpolate { a: Vec<f32>, b: Vec<f32>, t: f32 },
}

/// Policy for frontier nodes that have no vector when ranking with VectorRank.
/// Skip (default): drop the node from the ranked result, counted in a metric, never silently lost.
/// Error: fail the query cleanly the first time a frontier node has no vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OnMissingVector {
    #[default]
    Skip,
    Error,
}
