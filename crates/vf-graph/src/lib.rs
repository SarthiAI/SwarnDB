// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

pub mod types;
pub mod error;
pub mod traversal;
pub mod query;
pub mod compute;
pub mod persistence;
pub mod graph_delta;
pub mod refinement;
pub mod propagation;
pub mod model;
pub mod lru;
pub mod predicate;
pub mod typed_delta;
pub mod store;
pub mod typed_persistence;
pub mod weight;
pub mod temporal;

pub use types::{VirtualGraph, GraphConfig, GraphNode, Edge};
pub use error::GraphError;
pub use traversal::{GraphTraversal, TraversalOrder, TraversalResult};
pub use query::{RelationshipQueryEngine, RelationshipQuery, Direction};
pub use compute::RelationshipComputer;
#[allow(deprecated)]
pub use persistence::GraphPersistence;
pub use persistence::{serialize_base, deserialize_base, validate_graph_base};
pub use graph_delta::{
    GraphDeltaOp, GraphDeltaEntry, GraphDeltaWriter, GraphDeltaReader,
    replay_delta, replay_delta_after_lsn,
};
pub use refinement::{GraphRefiner, RefinementConfig, RefinementStats};
pub use propagation::ThresholdPropagator;
pub use model::{
    ChunkId, DateTime, DocId, Edge as TypedEdge, EdgeAudit, EdgeDirection, EdgeId, InternedString,
    Interner, MAX_EDGE_HISTORY, MAX_NODE_HISTORY, Node, NodeAudit, NodeId, NodeKind, NodeSource,
    PropertyValue, Provenance, now_millis,
};
pub use predicate::{CompareOp, Literal, Predicate, PropertyRef};
pub use typed_delta::{TypedDeltaEntry, TypedDeltaReader, TypedDeltaWriter, TypedGraphOp};
pub use store::{
    EdgePageFilter, GraphStore, GraphStoreConfig, GraphStoreStats, NodePageFilter, Page,
    TypedGraphStore,
};
pub use typed_persistence::{
    TYPED_BASE_FORMAT_VERSION, deserialize_typed_base, deserialize_typed_base_with_config,
    serialize_typed_base, validate_typed_base,
};
pub use weight::{effective_weight, traversal_cost, MIN_EFFECTIVE_WEIGHT, WeightParams};
pub use temporal::{edge_passes_temporal, TemporalFilter};
