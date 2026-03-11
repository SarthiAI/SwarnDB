// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

pub mod types;
pub mod error;
pub mod traversal;
pub mod query;
pub mod compute;
pub mod persistence;
pub mod refinement;
pub mod propagation;

pub use types::{VirtualGraph, GraphConfig, GraphNode, Edge};
pub use error::GraphError;
pub use traversal::{GraphTraversal, TraversalOrder, TraversalResult};
pub use query::{RelationshipQueryEngine, RelationshipQuery, Direction};
pub use compute::RelationshipComputer;
pub use persistence::GraphPersistence;
pub use refinement::{GraphRefiner, RefinementConfig, RefinementStats};
pub use propagation::ThresholdPropagator;
