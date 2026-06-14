// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! Query-time effective edge weight, derived from confidence, an optional
//! explicit weight property, and recency. Strictly additive: nothing here is
//! persisted, no Edge field changes, no on-disk or proto change. With default
//! (no-op) params every edge weighs exactly 1.0, so weighted traversal and
//! ranking stay byte-identical to the unweighted path unless a caller opts in.

use serde::{Deserialize, Serialize};

use crate::model::{Edge, Value};

/// Smallest effective weight we allow, so cost = 1/weight stays finite.
pub const MIN_EFFECTIVE_WEIGHT: f64 = 1e-6;

fn default_weight_key() -> String {
    "weight".to_string()
}

/// Query-time parameters controlling how an edge's effective weight is derived.
/// Default is a strict no-op: every edge weighs exactly 1.0, so weighted traversal
/// and ranking are byte-identical to the unweighted path unless a caller opts in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeightParams {
    #[serde(default)]
    pub use_confidence: bool,
    #[serde(default)]
    pub min_confidence: f32,
    #[serde(default)]
    pub recency_half_life_ms: u64,
    #[serde(default)]
    pub use_explicit_weight: bool,
    #[serde(default = "default_weight_key")]
    pub explicit_weight_key: String,
}

impl Default for WeightParams {
    fn default() -> Self {
        Self {
            use_confidence: false,
            min_confidence: 0.0,
            recency_half_life_ms: 0,
            use_explicit_weight: false,
            explicit_weight_key: default_weight_key(),
        }
    }
}

impl WeightParams {
    /// True when no signal is enabled, so effective weight is always 1.0.
    pub fn is_noop(&self) -> bool {
        !self.use_confidence && self.recency_half_life_ms == 0 && !self.use_explicit_weight
    }
}

/// Extract a finite number from a property value. Handles JSON numeric and string
/// variants; strings are parsed and accepted only when finite. Non-numeric,
/// non-parsable, or non-finite values yield None.
fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => {
            let x = n.as_f64()?;
            if x.is_finite() {
                Some(x)
            } else {
                None
            }
        }
        Value::String(s) => match s.trim().parse::<f64>() {
            Ok(x) if x.is_finite() => Some(x),
            _ => None,
        },
        _ => None,
    }
}

/// Effective weight of `edge` under `p` at `now_ms` (unix-epoch millis).
/// Always >= MIN_EFFECTIVE_WEIGHT and finite. With is_noop() params this returns exactly 1.0.
/// Composition is multiplicative: base(explicit) * confidence_factor * recency_factor.
pub fn effective_weight(edge: &Edge, p: &WeightParams, now_ms: u64) -> f64 {
    if p.is_noop() {
        return 1.0;
    }
    let mut w = 1.0_f64;
    if p.use_explicit_weight {
        if let Some(v) = edge.properties.get(&p.explicit_weight_key) {
            if let Some(x) = value_as_f64(v) {
                if x.is_finite() && x > 0.0 {
                    w *= x;
                }
            }
        }
    }
    if p.use_confidence {
        let floor = (p.min_confidence as f64).clamp(0.0, 1.0);
        let c = (edge.confidence as f64).clamp(floor, 1.0);
        w *= c;
    }
    if p.recency_half_life_ms > 0 {
        // created_at is unix-epoch millis (u64); saturating guard for future-dated edges.
        let age = now_ms.saturating_sub(edge.created_at) as f64;
        let hl = p.recency_half_life_ms as f64;
        w *= (-(std::f64::consts::LN_2) * age / hl).exp(); // 2^(-age/half_life)
    }
    if !w.is_finite() || w < MIN_EFFECTIVE_WEIGHT {
        MIN_EFFECTIVE_WEIGHT
    } else {
        w
    }
}

/// Traversal cost = 1/effective_weight, finite and positive. Lower cost = stronger edge.
pub fn traversal_cost(edge: &Edge, p: &WeightParams, now_ms: u64) -> f64 {
    let w = effective_weight(edge, p, now_ms);
    (1.0 / w).clamp(MIN_EFFECTIVE_WEIGHT, 1e12)
}
