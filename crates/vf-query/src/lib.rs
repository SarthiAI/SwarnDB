// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

pub mod batch;
pub mod eval;
pub mod filter;
pub mod index_btree;
pub mod index_hash;
pub mod index_bitmap;
pub mod index_manager;
pub mod strategy;
pub mod vector_math;

pub use batch::{BatchExecutor, BatchQuery, BatchResult};
pub use eval::FilterEvaluator;
pub use filter::{FilterExpression, QueryError};
pub use index_btree::BTreeIndex;
pub use index_hash::HashIndex;
pub use index_bitmap::BitmapIndex;
pub use index_manager::{IndexManager, IndexConfig, MetadataIndex};
pub use strategy::{QueryExecutor, FilterStrategy};
