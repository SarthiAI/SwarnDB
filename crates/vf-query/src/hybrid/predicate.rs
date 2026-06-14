// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! Predicate tree for filtering nodes and edges during hybrid execution.
//!
//! The predicate AST is the ONE shared property-condition model, defined in
//! `vf-graph` so both the graph store (page scans) and this query layer evaluate
//! it without a second comparison model. This module re-exports it so existing
//! call sites (`vf_query::hybrid::Predicate`, `super::predicate::Predicate`, ...)
//! are unchanged.

pub use vf_graph::predicate::{CompareOp, Literal, Predicate, PropertyRef};
