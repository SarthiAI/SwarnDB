// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! Shared property-condition model for filtering nodes and edges.
//!
//! This is the ONE predicate AST the whole engine reuses: hybrid query steps,
//! the filtered read API (page filters), and the filter-seed source all build on
//! it. It lives in `vf-graph` so both the graph store (page scans) and the query
//! layer can evaluate it without a second comparison model. The query crate
//! re-exports it as `vf_query::hybrid::{Predicate, PropertyRef, CompareOp,
//! Literal}`, so existing call sites are unchanged.
//!
//! A predicate resolves a [`PropertyRef`] against a node or edge, then applies a
//! comparison or set test. It serializes on the wire so a plan can travel between
//! client and server.

use serde::{Deserialize, Serialize};

use crate::model::{Edge, EdgeDirection, Node, NodeKind};
use crate::store::GraphStore;

/// A literal value used in comparisons. Reuses the JSON value model.
pub type Literal = serde_json::Value;

/// Which field a predicate reads from a node or edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PropertyRef {
    /// A key in the property bag.
    Property(String),
    /// Node: the entity label (content nodes have none). Edge: the edge type.
    Label,
    /// Node: "content" or "entity". Edge: not applicable (resolves to None).
    Kind,
    /// Node only: the count of incident edges, optionally constrained by edge
    /// type and direction. Resolves to an integer; needs the graph store, so it
    /// is meaningful only on the store-aware evaluation path (it resolves to None
    /// on the store-free path and on edges). Canonical default direction is
    /// OUTGOING on every surface (proto/gRPC/SDK and REST); see
    /// `default_incident_direction`.
    IncidentEdgeCount {
        #[serde(default)]
        edge_type: Option<String>,
        // Field-level default so an omitted `direction` on the REST/serde surface
        // matches the proto/gRPC/SDK default (OUTGOING), not the global Both.
        #[serde(default = "default_incident_direction")]
        direction: EdgeDirection,
    },
}

/// Canonical default direction for `IncidentEdgeCount` when omitted on the
/// serde/REST surface: OUTGOING, matching proto enum 0 and the SDK helper. This
/// deliberately differs from the global `EdgeDirection` default (Both), which
/// other call sites rely on.
fn default_incident_direction() -> EdgeDirection {
    EdgeDirection::Outgoing
}

/// Comparison operators over a resolved field and a literal.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A composable predicate evaluated against a single node or edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Predicate {
    /// Compare a field against a literal with an operator.
    Compare {
        field: PropertyRef,
        op: CompareOp,
        value: Literal,
    },
    /// True if the field exists and its value is in the set.
    In {
        field: PropertyRef,
        values: Vec<Literal>,
    },
    /// True if the field exists and its value is not in the set.
    NotIn {
        field: PropertyRef,
        values: Vec<Literal>,
    },
    /// True if the field resolves to a value at all.
    Exists { field: PropertyRef },
    /// Logical conjunction (empty vector is vacuously true).
    And(Vec<Predicate>),
    /// Logical disjunction (empty vector is vacuously false).
    Or(Vec<Predicate>),
    /// Logical negation.
    Not(Box<Predicate>),
    /// Always true. Constructed via `Predicate::any()`.
    Always,
}

// ── Ergonomic constructors ───────────────────────────────────────────────

impl Predicate {
    /// `field == value` over a property-bag key.
    pub fn eq(key: impl Into<String>, value: impl Into<Literal>) -> Self {
        Self::compare(key, CompareOp::Eq, value)
    }

    /// `field != value` over a property-bag key.
    pub fn ne(key: impl Into<String>, value: impl Into<Literal>) -> Self {
        Self::compare(key, CompareOp::Ne, value)
    }

    /// `field > value` over a property-bag key.
    pub fn gt(key: impl Into<String>, value: impl Into<Literal>) -> Self {
        Self::compare(key, CompareOp::Gt, value)
    }

    /// `field >= value` over a property-bag key.
    pub fn ge(key: impl Into<String>, value: impl Into<Literal>) -> Self {
        Self::compare(key, CompareOp::Ge, value)
    }

    /// `field < value` over a property-bag key.
    pub fn lt(key: impl Into<String>, value: impl Into<Literal>) -> Self {
        Self::compare(key, CompareOp::Lt, value)
    }

    /// `field <= value` over a property-bag key.
    pub fn le(key: impl Into<String>, value: impl Into<Literal>) -> Self {
        Self::compare(key, CompareOp::Le, value)
    }

    /// Shared builder for the property-key comparisons.
    fn compare(key: impl Into<String>, op: CompareOp, value: impl Into<Literal>) -> Self {
        Predicate::Compare {
            field: PropertyRef::Property(key.into()),
            op,
            value: value.into(),
        }
    }

    /// `field IN values` over a property-bag key.
    pub fn is_in(key: impl Into<String>, values: Vec<Literal>) -> Self {
        Predicate::In {
            field: PropertyRef::Property(key.into()),
            values,
        }
    }

    /// `field NOT IN values` over a property-bag key.
    pub fn not_in(key: impl Into<String>, values: Vec<Literal>) -> Self {
        Predicate::NotIn {
            field: PropertyRef::Property(key.into()),
            values,
        }
    }

    /// True when the property-bag key is present.
    pub fn exists(key: impl Into<String>) -> Self {
        Predicate::Exists {
            field: PropertyRef::Property(key.into()),
        }
    }

    /// Conjunction of sub-predicates.
    pub fn and(preds: Vec<Predicate>) -> Self {
        Predicate::And(preds)
    }

    /// Disjunction of sub-predicates.
    pub fn or(preds: Vec<Predicate>) -> Self {
        Predicate::Or(preds)
    }

    /// Negation of a sub-predicate.
    pub fn not(pred: Predicate) -> Self {
        Predicate::Not(Box::new(pred))
    }

    /// The always-true predicate.
    pub fn any() -> Self {
        Predicate::Always
    }

    /// Match a node's entity label (or, on an edge, the edge type).
    pub fn label_eq(value: impl Into<Literal>) -> Self {
        Predicate::Compare {
            field: PropertyRef::Label,
            op: CompareOp::Eq,
            value: value.into(),
        }
    }

    /// `incident-edge-count <op> value` over a node, optionally per edge type and
    /// direction. Structural; evaluated only on the store-aware path.
    pub fn incident_edges(
        edge_type: Option<String>,
        direction: EdgeDirection,
        op: CompareOp,
        value: impl Into<Literal>,
    ) -> Self {
        Predicate::Compare {
            field: PropertyRef::IncidentEdgeCount { edge_type, direction },
            op,
            value: value.into(),
        }
    }

    /// True when this predicate (anywhere in its tree) reads a structural field
    /// that needs the graph store. The store-free path treats such fields as
    /// unresolved, so callers with a store should take the store-aware path.
    pub fn needs_store(&self) -> bool {
        match self {
            Predicate::Compare { field, .. } => field.is_structural(),
            Predicate::In { field, .. } | Predicate::NotIn { field, .. } => field.is_structural(),
            Predicate::Exists { field } => field.is_structural(),
            Predicate::And(preds) | Predicate::Or(preds) => preds.iter().any(|p| p.needs_store()),
            Predicate::Not(inner) => inner.needs_store(),
            Predicate::Always => false,
        }
    }
}

// ── Field resolution ─────────────────────────────────────────────────────

impl PropertyRef {
    /// Whether this reference needs the graph store to resolve.
    fn is_structural(&self) -> bool {
        matches!(self, PropertyRef::IncidentEdgeCount { .. })
    }

    /// Resolve this reference against a node without a store. Structural refs
    /// resolve to None here (they need the store-aware path).
    fn resolve_node(&self, node: &Node) -> Option<Literal> {
        match self {
            PropertyRef::Property(k) => node.properties.get(k).cloned(),
            PropertyRef::Label => match &node.kind {
                NodeKind::Entity { label } => Some(Literal::from(label.clone())),
                NodeKind::Content => None,
            },
            PropertyRef::Kind => match &node.kind {
                NodeKind::Content => Some(Literal::from("content")),
                NodeKind::Entity { .. } => Some(Literal::from("entity")),
            },
            // Needs the store; unresolved on the store-free path.
            PropertyRef::IncidentEdgeCount { .. } => None,
        }
    }

    /// Resolve this reference against a node WITH a store, so structural refs
    /// (incident-edge count) resolve. Non-structural refs match `resolve_node`.
    fn resolve_node_with_store(&self, node: &Node, store: &dyn GraphStore) -> Option<Literal> {
        match self {
            PropertyRef::IncidentEdgeCount { edge_type, direction } => {
                let count = incident_edge_count(store, node.id, edge_type.as_deref(), *direction);
                Some(Literal::from(count))
            }
            other => other.resolve_node(node),
        }
    }

    /// Resolve this reference against an edge, owning the resulting literal.
    fn resolve_edge(&self, edge: &Edge) -> Option<Literal> {
        match self {
            PropertyRef::Property(k) => edge.properties.get(k).cloned(),
            PropertyRef::Label => Some(Literal::from(edge.edge_type.as_str())),
            // Not applicable to edges.
            PropertyRef::Kind | PropertyRef::IncidentEdgeCount { .. } => None,
        }
    }
}

/// Count incident edges of a node, optionally constrained by edge type and
/// direction. Delegates to the store's count-only API so a scan does not deep-
/// clone every incident Edge just to take a length.
fn incident_edge_count(
    store: &dyn GraphStore,
    node: crate::model::NodeId,
    edge_type: Option<&str>,
    direction: EdgeDirection,
) -> u64 {
    store.incident_edge_count(node, edge_type, direction)
}

// ── Evaluation ───────────────────────────────────────────────────────────

impl Predicate {
    /// Evaluate against a node (store-free). Structural terms see an unresolved
    /// field, so callers with a store should use `eval_node_with_store`.
    pub fn eval_node(&self, node: &Node) -> bool {
        self.eval(|r| r.resolve_node(node))
    }

    /// Evaluate against a node WITH the graph store, so structural terms (e.g.
    /// incident-edge count) resolve. Pure property predicates behave exactly as
    /// `eval_node`; no behavior change when no structural term is present.
    pub fn eval_node_with_store(&self, node: &Node, store: &dyn GraphStore) -> bool {
        self.eval(|r| r.resolve_node_with_store(node, store))
    }

    /// Evaluate against an edge.
    pub fn eval_edge(&self, edge: &Edge) -> bool {
        self.eval(|r| r.resolve_edge(edge))
    }

    /// Core evaluator parameterised by a field resolver.
    fn eval<R>(&self, resolve: R) -> bool
    where
        R: Fn(&PropertyRef) -> Option<Literal> + Copy,
    {
        match self {
            // Missing field: comparison is false.
            Predicate::Compare { field, op, value } => match resolve(field) {
                Some(actual) => compare(&actual, *op, value),
                None => false,
            },
            // In: field must exist and the value must be in the set.
            Predicate::In { field, values } => match resolve(field) {
                Some(actual) => values.iter().any(|v| value_eq(&actual, v)),
                None => false,
            },
            // NotIn: field must exist and the value must not be in the set.
            Predicate::NotIn { field, values } => match resolve(field) {
                Some(actual) => !values.iter().any(|v| value_eq(&actual, v)),
                None => false,
            },
            // Exists: missing field is false.
            Predicate::Exists { field } => resolve(field).is_some(),
            Predicate::And(preds) => preds.iter().all(|p| p.eval(resolve)),
            Predicate::Or(preds) => preds.iter().any(|p| p.eval(resolve)),
            Predicate::Not(inner) => !inner.eval(resolve),
            Predicate::Always => true,
        }
    }
}

// ── Value comparison ─────────────────────────────────────────────────────

/// Equality across JSON literals: numbers numerically, strings and bools by
/// value. Any other shape falls back to structural equality.
fn value_eq(a: &Literal, b: &Literal) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

/// Apply a comparison operator across two JSON literals. Eq/Ne work for every
/// shape; ordering operators are meaningful only for numbers and strings and
/// return false on a type mismatch.
fn compare(actual: &Literal, op: CompareOp, expected: &Literal) -> bool {
    match op {
        CompareOp::Eq => value_eq(actual, expected),
        CompareOp::Ne => !value_eq(actual, expected),
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => {
            match ordering(actual, expected) {
                Some(ord) => match op {
                    CompareOp::Lt => ord.is_lt(),
                    CompareOp::Le => ord.is_le(),
                    CompareOp::Gt => ord.is_gt(),
                    CompareOp::Ge => ord.is_ge(),
                    _ => unreachable!("non-ordering op in ordering arm"),
                },
                None => false,
            }
        }
    }
}

/// Order two literals: numbers via f64, strings lexicographically. Anything
/// else (including type mismatches and NaN) yields None.
fn ordering(a: &Literal, b: &Literal) -> Option<std::cmp::Ordering> {
    if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
        return x.partial_cmp(&y);
    }
    if let (Some(x), Some(y)) = (a.as_str(), b.as_str()) {
        return Some(x.cmp(y));
    }
    None
}
