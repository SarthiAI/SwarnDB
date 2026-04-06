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
