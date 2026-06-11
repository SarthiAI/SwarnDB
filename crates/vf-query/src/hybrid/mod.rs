// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Composable hybrid vector-and-graph query API.
//!
//! A query is built with the type-state [`QueryBuilder`], compiled to a
//! serializable [`QueryPlan`], and run by [`HybridExecutor`] against a vector
//! index and a typed graph store. The executor is read-only.
//!
//! The default graph-augmented ranking is graph-first scope-then-rank: scope the
//! candidate set by structure, then rank it with a `VectorRank` step run by
//! [`HybridExecutor::execute`] (ADR-024). RRF fusion via
//! [`HybridExecutor::execute_rrf`] is the opt-in path, reached only on an
//! explicit [`RrfRankSpec`].

pub mod builder;
pub mod error;
pub mod exec;
pub mod plan;
pub mod predicate;

pub use builder::{OnEdges, OnEmpty, OnNodes, OnPaths, QueryBuilder};
pub use error::HybridQueryError;
pub use exec::{HybridExecutor, NodeRecord, Path, QueryResult};
pub use plan::{
    OnMissingVector, QueryPlan, ReturnKind, RrfRankSpec, Step, TemporalFilter, VectorMathOp,
    WeightParams, RRF_K_DEFAULT, RRF_K_HOP_DEFAULT, RRF_MENTIONS_EDGE_TYPE,
};
pub use predicate::{CompareOp, Literal, Predicate, PropertyRef};

// Re-export the graph identifiers and direction so callers can build plans
// without depending on vf-graph directly.
pub use vf_graph::{EdgeDirection, EdgeId, NodeId};
