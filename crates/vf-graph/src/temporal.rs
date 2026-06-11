// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Temporal edge filtering (P17). Opt-in, default-off: when no `TemporalFilter`
//! is supplied to a traversal step the executor uses the existing cached fast
//! paths unchanged (ADR-007 R5). A filter narrows expansion to edges valid "as
//! of" an instant and/or carrying a required regime/context label. Nothing here
//! is persisted; the per-edge validity window and context live on `model::Edge`.

use serde::{Deserialize, Serialize};

use crate::model::Edge;

/// Opt-in time-and-context filter for graph traversal (P17). Default-constructed
/// (`as_of = None`, `include_unbounded = true`, `context = None`) passes every edge,
/// but the executor only constructs one at all when the caller opts in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalFilter {
    /// The "as of" instant in unix-epoch millis. None = use now_ms supplied by the executor.
    #[serde(default)]
    pub as_of: Option<u64>,
    /// Whether unbounded edges (both bounds None) pass the time check. Default true.
    #[serde(default = "default_include_unbounded")]
    pub include_unbounded: bool,
    /// Required regime/context. Some(c) = edge must have temporal_context == Some(c).
    /// None = context ignored.
    #[serde(default)]
    pub context: Option<String>,
}

fn default_include_unbounded() -> bool {
    true
}

impl Default for TemporalFilter {
    fn default() -> Self {
        Self { as_of: None, include_unbounded: true, context: None }
    }
}

/// True iff `edge` passes both the time check and the context check of `f`.
/// `now_ms` is the per-query instant; used when `f.as_of` is None.
pub fn edge_passes_temporal(edge: &Edge, f: &TemporalFilter, now_ms: u64) -> bool {
    // CONTEXT check first (cheap): if a context is required, the edge must match it.
    if let Some(ref want) = f.context {
        match edge.temporal_context.as_deref() {
            Some(c) if c == want => {}
            _ => return false, // edges without context FAIL when a context is required
        }
    }
    // TIME check.
    let t = f.as_of.unwrap_or(now_ms);
    let unbounded = edge.valid_from.is_none() && edge.valid_until.is_none();
    if unbounded {
        return f.include_unbounded;
    }
    let after_start = edge.valid_from.map_or(true, |from| t >= from);
    let before_end = edge.valid_until.map_or(true, |until| t < until);
    after_start && before_end
}
