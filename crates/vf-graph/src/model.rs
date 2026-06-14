// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! First-class typed graph data model: content and entity nodes, typed edges
//! with a property bag and full provenance baked in from day one.
//!
//! This is the Hybrid-mode model. The similarity-only `types::Edge` and
//! `VirtualGraph` stay exactly as they are for AutoSimilarity collections
//! (they keep reading and writing the v2 format). The two coexist; nothing here
//! changes vector-only or auto-similarity behavior.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Wall-clock timestamp in unix-epoch milliseconds.
///
/// We use `u64` millis rather than a `chrono` type to avoid a new dependency and
/// to match the existing graph-delta timestamp convention.
pub type DateTime = u64;

/// Identifier of a source document a node or edge was derived from.
pub type DocId = String;

/// Identifier of a source chunk within a document.
pub type ChunkId = u64;

/// A JSON-like property value. Reuses `serde_json::Value` for the property bag.
pub type Value = serde_json::Value;

/// Re-export name used by the crate root to avoid clashing with other `Value` types.
pub type PropertyValue = Value;

/// Current wall-clock time in unix-epoch milliseconds (0 if the clock is before the epoch).
pub fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Store-assigned identifier of a graph node. Distinct from `VectorId`: a content
/// node may link to a vector via its embedding, but node identity is its own space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u64);

/// Store-assigned identifier of a typed edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeId(pub u64);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── Interned edge-type labels ────────────────────────────────────────

/// An interned string used for edge-type labels. Cheap to clone (`Arc`), and
/// deduplicated by an [`Interner`] so adjacency lists stay compact in memory.
///
/// Serializes transparently as its string value; on load it is re-interned by
/// the store, so the on-disk form carries no interner-specific ids.
#[derive(Clone, Debug)]
pub struct InternedString(Arc<str>);

impl InternedString {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::ops::Deref for InternedString {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl PartialEq for InternedString {
    fn eq(&self, other: &Self) -> bool {
        // Pointer equality is the fast path for same-interner labels; the value
        // comparison covers labels interned by different interners.
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Eq for InternedString {}

impl std::hash::Hash for InternedString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl std::fmt::Display for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for InternedString {
    fn from(s: &str) -> Self {
        InternedString(Arc::from(s))
    }
}

impl From<String> for InternedString {
    fn from(s: String) -> Self {
        InternedString(Arc::from(s.as_str()))
    }
}

impl Serialize for InternedString {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for InternedString {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(InternedString(Arc::from(s.as_str())))
    }
}

/// Deduplicating pool of edge-type labels. One per graph store.
#[derive(Default)]
pub struct Interner {
    pool: HashSet<Arc<str>>,
}

impl Interner {
    pub fn new() -> Self {
        Self {
            pool: HashSet::new(),
        }
    }

    /// Intern a label, returning a shared handle. Repeated calls for the same
    /// text return clones of the same `Arc`.
    pub fn intern(&mut self, s: &str) -> InternedString {
        if let Some(existing) = self.pool.get(s) {
            return InternedString(existing.clone());
        }
        let arc: Arc<str> = Arc::from(s);
        self.pool.insert(arc.clone());
        InternedString(arc)
    }

    pub fn len(&self) -> usize {
        self.pool.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }
}

// ── Nodes ────────────────────────────────────────────────────────────

/// The two node kinds. Content nodes come from the chunk-and-embed pipeline;
/// entity nodes are typed things (people, orgs, parties) with a label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Content,
    Entity { label: String },
}

/// How a node came to exist. Drives re-extraction policy in later phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeSource {
    /// Created directly by a user via the API.
    Manual,
    /// Produced by the content ingest (chunk-and-embed) pipeline.
    Ingested,
    /// Produced by LLM extraction.
    Extracted,
}

/// A single audit entry on a node: what happened, who did it, and when.
/// Mirrors `EdgeAudit` so nodes carry the same provenance/audit shape as edges.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeAudit {
    /// e.g. "created", "updated".
    pub action: String,
    /// Optional actor identity; absent on older records.
    #[serde(default)]
    pub actor: Option<String>,
    /// Unix-epoch millis.
    pub at: DateTime,
}

/// Cap on retained audit entries per node; oldest are dropped past this.
pub const MAX_NODE_HISTORY: usize = 8;

/// A graph node. Content nodes may carry an embedding; entity nodes usually do not.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    #[serde(default)]
    pub properties: HashMap<String, Value>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub source: NodeSource,
    pub created_at: DateTime,
    #[serde(default)]
    pub created_by: Option<String>,
    /// Last mutation time; absent on create and on older records.
    #[serde(default)]
    pub updated_at: Option<DateTime>,
    /// Bounded per-node audit trail; absent on older records.
    #[serde(default)]
    pub history: Vec<NodeAudit>,
}

impl Node {
    /// A content node (optionally carrying its embedding).
    pub fn content(id: NodeId, embedding: Option<Vec<f32>>, source: NodeSource) -> Self {
        Self {
            id,
            kind: NodeKind::Content,
            properties: HashMap::new(),
            embedding,
            source,
            created_at: now_millis(),
            created_by: None,
            updated_at: None,
            history: Vec::new(),
        }
    }

    /// An entity node with the given type label (for example `"Party"`).
    pub fn entity(id: NodeId, label: impl Into<String>, source: NodeSource) -> Self {
        Self {
            id,
            kind: NodeKind::Entity {
                label: label.into(),
            },
            properties: HashMap::new(),
            embedding: None,
            source,
            created_at: now_millis(),
            created_by: None,
            updated_at: None,
            history: Vec::new(),
        }
    }

    pub fn is_entity(&self) -> bool {
        matches!(self.kind, NodeKind::Entity { .. })
    }

    pub fn is_content(&self) -> bool {
        matches!(self.kind, NodeKind::Content)
    }

    /// Append an audit entry, keeping only the most recent MAX_NODE_HISTORY entries.
    /// Also stamps `updated_at`.
    pub fn record_audit(&mut self, action: impl Into<String>, actor: Option<String>, at: DateTime) {
        self.history.push(NodeAudit { action: action.into(), actor, at });
        let len = self.history.len();
        if len > MAX_NODE_HISTORY {
            self.history.drain(0..len - MAX_NODE_HISTORY);
        }
        self.updated_at = Some(at);
    }
}

// ── Provenance ───────────────────────────────────────────────────────

/// Full provenance for an edge. Baked into the model from day one so the
/// on-disk format never has to migrate a second time (ADR-007 R4(e)). Every
/// field is individually optional: manual edges leave most of them empty.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub source_doc: Option<DocId>,
    pub source_chunk_id: Option<ChunkId>,
    pub source_text: Option<String>,
    pub model: Option<String>,
    pub prompt_version: Option<String>,
    pub extracted_at: Option<DateTime>,
    pub cache_hit_at: Option<DateTime>,
}

impl Provenance {
    /// True when no provenance field is set (the typical manual-edge case).
    pub fn is_empty(&self) -> bool {
        self.source_doc.is_none()
            && self.source_chunk_id.is_none()
            && self.source_text.is_none()
            && self.model.is_none()
            && self.prompt_version.is_none()
            && self.extracted_at.is_none()
            && self.cache_hit_at.is_none()
    }
}

// ── Edges ────────────────────────────────────────────────────────────

/// A single audit entry on an edge: what happened, who did it, and when.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EdgeAudit {
    /// e.g. "created", "updated", "verified", "imported", "reextracted".
    pub action: String,
    /// Optional actor identity; absent on older records.
    #[serde(default)]
    pub actor: Option<String>,
    /// Unix-epoch millis.
    pub at: DateTime,
}

/// Cap on retained audit entries per edge; oldest are dropped past this.
pub const MAX_EDGE_HISTORY: usize = 8;

/// A first-class typed edge: a labelled, directed relationship between two nodes,
/// with a property bag, full provenance, confidence, and manual/verified flags.
///
/// This is the Hybrid-mode edge. It does NOT replace the similarity-only
/// `types::Edge`, which is retained for the v2 format and AutoSimilarity
/// collections (ADR-007 R4(b)). On disk this lives in the v3 format only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub edge_type: InternedString,
    #[serde(default)]
    pub properties: HashMap<String, Value>,
    #[serde(default)]
    pub provenance: Provenance,
    pub confidence: f32,
    pub verified: bool,
    pub is_manual: bool,
    pub created_at: DateTime,
    /// Validity window start, unix-epoch millis. None = unbounded start. (P17)
    #[serde(default)]
    pub valid_from: Option<u64>,
    /// Validity window end, unix-epoch millis, EXCLUSIVE. None = unbounded end. (P17)
    #[serde(default)]
    pub valid_until: Option<u64>,
    /// Optional regime / version / scenario label. None = no context. (P17)
    #[serde(default)]
    pub temporal_context: Option<String>,
    /// Bounded per-edge audit trail; absent on older records.
    #[serde(default)]
    pub history: Vec<EdgeAudit>,
}

impl Edge {
    /// A manually-created edge: `is_manual = true`, `verified = true`,
    /// `confidence = 1.0`, provenance empty except for the timestamp.
    pub fn manual(
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        edge_type: InternedString,
    ) -> Self {
        Self {
            id,
            source,
            target,
            edge_type,
            properties: HashMap::new(),
            provenance: Provenance::default(),
            confidence: 1.0,
            verified: true,
            is_manual: true,
            created_at: now_millis(),
            valid_from: None,
            valid_until: None,
            temporal_context: None,
            history: Vec::new(),
        }
    }

    /// Append an audit entry, keeping only the most recent MAX_EDGE_HISTORY entries.
    pub fn record_audit(&mut self, action: impl Into<String>, actor: Option<String>, at: DateTime) {
        self.history.push(EdgeAudit { action: action.into(), actor, at });
        let len = self.history.len();
        if len > MAX_EDGE_HISTORY {
            self.history.drain(0..len - MAX_EDGE_HISTORY);
        }
    }
}

/// Direction of a neighbour query relative to a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeDirection {
    /// Edges where the node is the source.
    Outgoing,
    /// Edges where the node is the target.
    Incoming,
    /// Either direction. Default: counts any incident edge.
    #[default]
    Both,
}

impl EdgeDirection {
    /// Whether an edge with the given endpoints touches `node` in this direction.
    pub fn matches(&self, node: NodeId, src: NodeId, tgt: NodeId) -> bool {
        match self {
            EdgeDirection::Outgoing => src == node,
            EdgeDirection::Incoming => tgt == node,
            EdgeDirection::Both => src == node || tgt == node,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interner_dedups() {
        let mut i = Interner::new();
        let a = i.intern("CITES");
        let b = i.intern("CITES");
        let c = i.intern("PARTY_TO");
        assert_eq!(a, b);
        assert!(Arc::ptr_eq(&a.0, &b.0));
        assert_ne!(a, c);
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn interned_string_roundtrips_through_serde() {
        let mut i = Interner::new();
        let s = i.intern("CITES");
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"CITES\"");
        let back: InternedString = serde_json::from_str(&json).unwrap();
        assert_eq!(back.as_str(), "CITES");
    }

    #[test]
    fn manual_edge_defaults() {
        let mut i = Interner::new();
        let e = Edge::manual(EdgeId(1), NodeId(10), NodeId(20), i.intern("CITES"));
        assert!(e.is_manual);
        assert!(e.verified);
        assert_eq!(e.confidence, 1.0);
        assert!(e.provenance.is_empty());
    }

    #[test]
    fn node_kinds() {
        let c = Node::content(NodeId(1), Some(vec![0.1, 0.2]), NodeSource::Ingested);
        let e = Node::entity(NodeId(2), "Party", NodeSource::Manual);
        assert!(c.is_content());
        assert!(e.is_entity());
    }
}
