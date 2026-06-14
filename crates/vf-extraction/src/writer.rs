// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! The graph-write seam. The extraction layer never touches the server's
//! per-collection locks or the typed graph store directly; instead it writes
//! through this small object-safe trait, which the server implements over its
//! own collection handle. This keeps extraction free of server-lock knowledge
//! and keeps the crate layering acyclic.

use vf_graph::{EdgeId, Node, NodeId, TypedEdge as Edge};

use crate::error::ExtractionError;

/// The write seam between extraction and the typed graph store.
///
/// Implementations are expected to take whatever lock they need per call,
/// align the log sequence number with the typed delta, and durably persist on
/// a cadence consistent with the rest of the store. Every method that mutates
/// state returns `ExtractionError::Graph` on failure.
///
/// Object-safe and `Send + Sync` so the manager can hold an
/// `Arc<dyn GraphWriter>` per collection and share it across worker tasks.
pub trait GraphWriter: Send + Sync {
    /// Allocate the next log sequence number for a typed write.
    fn next_lsn(&self) -> u64;

    /// Insert or overwrite a node at the given log sequence number.
    fn put_node(&self, node: Node, lsn: u64) -> Result<(), ExtractionError>;

    /// Insert or overwrite an edge at the given log sequence number.
    fn put_edge(&self, edge: Edge, lsn: u64) -> Result<(), ExtractionError>;

    /// Delete an edge by id. Returns whether an edge was actually removed.
    fn delete_edge(&self, id: EdgeId, lsn: u64) -> Result<bool, ExtractionError>;

    /// Allocate a fresh, unused node id.
    fn alloc_node_id(&self) -> NodeId;

    /// Allocate a fresh, unused edge id.
    fn alloc_edge_id(&self) -> EdgeId;

    /// Find an existing entity node with the given label and name (the dedup
    /// lookup the pipeline uses before allocating a new node).
    ///
    /// Matching is on the deterministic normalized name (ADR-015): the
    /// implementation compares `normalize_entity_name(query)` against each
    /// candidate node's stored `name_norm` property, falling back to normalizing
    /// the legacy `name` property on the fly for nodes written before
    /// `name_norm` existed. This collapses whitespace, quote, trailing-
    /// punctuation, NFKC, and case variants onto one node.
    fn find_entity(&self, label: &str, name: &str) -> Option<NodeId>;

    /// Conservative deterministic fuzzy resolution (ADR-020), used only when the
    /// collection opts in to `EntityResolution::Fuzzy`. Matches `name` against
    /// existing entity nodes of the SAME label using `fuzzy_name_match` (exact
    /// normalized, initials / abbreviation, acronym / expansion, bounded typo, and
    /// strict head-shared title-prefix supersets). On a confident match it records
    /// the new surface form on the canonical node's `aliases` property for
    /// provenance and returns that node's id; otherwise returns `None` so the
    /// caller allocates a fresh node. Label-scoped and never merges across labels.
    ///
    /// The default trait body falls back to the exact `find_entity`, so an
    /// implementation that has not yet adopted fuzzy resolution stays correct
    /// (just conservative).
    fn find_entity_fuzzy(&self, label: &str, name: &str) -> Option<NodeId> {
        self.find_entity(label, name)
    }

    /// Fetch a node by id, if present.
    fn get_node(&self, id: NodeId) -> Option<Node>;

    /// True when a node with this id exists.
    fn node_exists(&self, id: NodeId) -> bool;

    /// Every edge whose provenance points at this source document and chunk.
    /// Used by re-extraction to find prior auto-edges to replace.
    fn edges_from_chunk(&self, source_doc: &str, source_chunk_id: u64) -> Vec<Edge>;

    /// Delete a node by id. Returns true if it existed.
    fn delete_node(&self, id: NodeId, lsn: u64) -> Result<bool, ExtractionError>;

    /// True if a manual OR verified edge already connects source->target with this edge_type.
    fn manual_or_verified_edge_exists(&self, source: NodeId, target: NodeId, edge_type: &str) -> bool;

    /// True if the node has any incident edge (either direction).
    fn node_has_incident_edges(&self, id: NodeId) -> bool;
}
