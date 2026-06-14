// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! The server-side `GraphWriter` implementation that the extraction manager
//! writes typed nodes and edges through. It mirrors the manual P01 typed-graph
//! handlers exactly: it sources the LSN from the same WAL high-water mark, calls
//! the same `TypedGraphStore` methods, syncs the typed delta, and marks the
//! collection dirty. This keeps the extraction crate free of any server-lock
//! knowledge.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use parking_lot::RwLock;
use vf_graph::{EdgeDirection, EdgeId, GraphStore, Node, NodeId, TypedEdge};
use vf_storage::collection::CollectionManager;

use vf_extraction::{ExtractionError, GraphWriter};

use crate::state::{metered_read, metered_write, CollectionState};

/// Writes extraction output into one collection's typed graph store. Holds the
/// per-collection handle plus the shared collection manager and the collection
/// name so it can source the LSN the way the manual handlers do.
pub struct CollectionGraphWriter {
    handle: Arc<RwLock<CollectionState>>,
    collection_manager: Arc<RwLock<CollectionManager>>,
    collection: String,
}

impl CollectionGraphWriter {
    /// Build a writer over a collection handle. `collection_manager` and
    /// `collection` are used only to read the current WAL LSN per write.
    pub fn new(
        handle: Arc<RwLock<CollectionState>>,
        collection_manager: Arc<RwLock<CollectionManager>>,
        collection: String,
    ) -> Self {
        Self {
            handle,
            collection_manager,
            collection,
        }
    }

    /// Append a merged surface form to a canonical entity node's `aliases` list
    /// (ADR-020 provenance). Idempotent: a surface form already present (including
    /// the canonical display name) is not added again, so re-extraction does not
    /// grow the list without bound. A missing node or a read/write race is a
    /// no-op; alias provenance is best-effort and never blocks resolution.
    fn record_alias(&self, id: NodeId, surface_form: &str) {
        // Fetch the LSN before taking the collection handle lock, matching the
        // established write pattern (the trait's next_lsn takes only the manager
        // read lock), so no handle-write-then-manager-read ordering is introduced.
        let lsn = {
            let cm = self.collection_manager.read();
            cm.get_collection(&self.collection)
                .map(|c| c.current_lsn())
                .unwrap_or(0)
        };
        let mut coll = metered_write(&self.handle);
        let coll_ref = &mut *coll;
        let store = match coll_ref.graph_store.as_mut() {
            Some(s) => s,
            None => return,
        };
        let mut node = match store.get_node(id) {
            Some(n) => n,
            None => return,
        };
        // Skip if it equals the display name or is already recorded.
        let is_display_name = node
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .map(|n| n == surface_form)
            .unwrap_or(false);
        if is_display_name {
            return;
        }
        let mut aliases: Vec<serde_json::Value> = node
            .properties
            .get(vf_extraction::ALIASES_KEY)
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if aliases
            .iter()
            .any(|a| a.as_str() == Some(surface_form))
        {
            return;
        }
        aliases.push(serde_json::Value::String(surface_form.to_string()));
        node.properties.insert(
            vf_extraction::ALIASES_KEY.to_string(),
            serde_json::Value::Array(aliases),
        );
        if let Err(e) = store.put_node(node, lsn) {
            tracing::warn!(error = %e, "recording entity alias failed; resolution still applied");
            return;
        }
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
    }
}

impl GraphWriter for CollectionGraphWriter {
    fn next_lsn(&self) -> u64 {
        // Mirrors the manual put_edge handler: read the WAL high-water mark from
        // the collection manager. current_lsn() returns the next LSN the WAL
        // will assign and does not advance, so repeated typed writes in one
        // chunk all carry that snapshot, exactly like a single manual RPC.
        let cm = self.collection_manager.read();
        cm.get_collection(&self.collection)
            .map(|c| c.current_lsn())
            .unwrap_or(0)
    }

    fn put_node(&self, node: Node, lsn: u64) -> Result<(), ExtractionError> {
        let mut coll = metered_write(&self.handle);
        let coll_ref = &mut *coll;
        let store = coll_ref
            .graph_store
            .as_mut()
            .ok_or_else(|| ExtractionError::Graph("collection is not in hybrid mode".to_string()))?;
        store
            .put_node(node, lsn)
            .map_err(|e| ExtractionError::Graph(format!("put_node failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn put_edge(&self, edge: TypedEdge, lsn: u64) -> Result<(), ExtractionError> {
        let mut coll = metered_write(&self.handle);
        let coll_ref = &mut *coll;
        let store = coll_ref
            .graph_store
            .as_mut()
            .ok_or_else(|| ExtractionError::Graph("collection is not in hybrid mode".to_string()))?;
        store
            .put_edge(edge, lsn)
            .map_err(|e| ExtractionError::Graph(format!("put_edge failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn delete_edge(&self, id: EdgeId, lsn: u64) -> Result<bool, ExtractionError> {
        let mut coll = metered_write(&self.handle);
        let coll_ref = &mut *coll;
        let store = coll_ref
            .graph_store
            .as_mut()
            .ok_or_else(|| ExtractionError::Graph("collection is not in hybrid mode".to_string()))?;
        let deleted = store
            .delete_edge(id, lsn)
            .map_err(|e| ExtractionError::Graph(format!("delete_edge failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(deleted)
    }

    fn alloc_node_id(&self) -> NodeId {
        let coll = metered_write(&self.handle);
        // Hybrid: draw the node id from the unified per-collection id authority
        // (the vector store's next_id) so entity node ids can never collide with
        // vector/content ids. The id is allocated but not inserted into vectors,
        // so the NodeId == VectorId bridge stays content-only.
        if coll.graph_store.is_some() {
            NodeId(coll.store.alloc_id())
        } else {
            // No store means non-hybrid; the pipeline never reaches a write in
            // that case, but return a stable sentinel rather than panic.
            NodeId(0)
        }
    }

    fn alloc_edge_id(&self) -> EdgeId {
        let mut coll = metered_write(&self.handle);
        match coll.graph_store.as_mut() {
            Some(store) => store.alloc_edge_id(),
            None => EdgeId(0),
        }
    }

    fn find_entity(&self, label: &str, name: &str) -> Option<NodeId> {
        // P09.5: exact resolution via the store's O(1) entity-name index (ADR-015).
        // Same normalized-name semantics as before, just no full-graph scan.
        let query_norm = vf_extraction::normalize_entity_name(name);
        if query_norm.is_empty() {
            return None;
        }
        let coll = metered_read(&self.handle);
        let store = coll.graph_store.as_ref()?;
        store.find_entity_by_norm(label, &query_norm)
    }

    fn find_entity_fuzzy(&self, label: &str, name: &str) -> Option<NodeId> {
        // ADR-020. Conservative deterministic fuzzy resolution, label-scoped.
        // Strategy:
        //   1. Try the exact normalized match first (the ADR-015 path). An exact
        //      hit returns without recording an alias, so exact resolution stays
        //      byte-identical to Normalized mode.
        //   2. Otherwise walk the same-label candidate bucket and apply
        //      `fuzzy_name_match` on the normalized forms. The first confident
        //      match wins; record the new surface form on that node's `aliases`
        //      and return its id.
        //   3. No confident match returns None so the caller allocates a new node.
        // P09.5: candidates now come from the store's same-label bucket (not a
        // whole-graph scan), in deterministic NodeId order, so the ADR-020
        // conservative semantics are preserved.
        if let Some(exact) = self.find_entity(label, name) {
            return Some(exact);
        }

        let query_norm = vf_extraction::normalize_entity_name(name);
        if query_norm.is_empty() {
            return None;
        }

        // Find the first same-label candidate whose normalized name fuzzily
        // matches. Read lock is dropped before any write so the alias update does
        // not deadlock against itself.
        let matched_id = {
            let coll = metered_read(&self.handle);
            let store = coll.graph_store.as_ref()?;
            let mut hit: Option<NodeId> = None;
            for (stored, id) in store.entity_candidates_for_label(label) {
                if vf_extraction::fuzzy_name_match(&stored, &query_norm) {
                    hit = Some(id);
                    break;
                }
            }
            hit
        };

        // Record the merged surface form on the canonical node for provenance.
        if let Some(id) = matched_id {
            self.record_alias(id, name);
        }
        matched_id
    }

    fn get_node(&self, id: NodeId) -> Option<Node> {
        let coll = metered_read(&self.handle);
        coll.graph_store.as_ref().and_then(|s| s.get_node(id))
    }

    fn node_exists(&self, id: NodeId) -> bool {
        let coll = metered_read(&self.handle);
        coll.graph_store
            .as_ref()
            .map(|s| s.get_node(id).is_some())
            .unwrap_or(false)
    }

    fn delete_node(&self, id: NodeId, lsn: u64) -> Result<bool, ExtractionError> {
        let mut coll = metered_write(&self.handle);
        let coll_ref = &mut *coll;
        let store = coll_ref
            .graph_store
            .as_mut()
            .ok_or_else(|| ExtractionError::Graph("collection is not in hybrid mode".to_string()))?;
        let deleted = store
            .delete_node(id, lsn)
            .map_err(|e| ExtractionError::Graph(format!("delete_node failed: {e}")))?;
        let _ = store.sync_delta();
        coll_ref.dirty.store(true, Ordering::Release);
        coll_ref.mutation_count.fetch_add(1, Ordering::Relaxed);
        Ok(deleted)
    }

    fn manual_or_verified_edge_exists(&self, source: NodeId, target: NodeId, edge_type: &str) -> bool {
        let coll = metered_read(&self.handle);
        let store = match coll.graph_store.as_ref() {
            Some(s) => s,
            None => return false,
        };
        store
            .edges_for_node(source, EdgeDirection::Outgoing)
            .iter()
            .any(|e| {
                e.target == target
                    && e.edge_type.as_str() == edge_type
                    && (e.is_manual || e.verified)
            })
    }

    fn node_has_incident_edges(&self, id: NodeId) -> bool {
        let coll = metered_read(&self.handle);
        coll.graph_store
            .as_ref()
            .map(|s| !s.edges_for_node(id, EdgeDirection::Both).is_empty())
            .unwrap_or(false)
    }

    fn edges_from_chunk(&self, source_doc: &str, source_chunk_id: u64) -> Vec<TypedEdge> {
        // O(n) scan of the edge set, matching provenance on (doc, chunk). Used by
        // re-extraction to find prior auto-edges; a provenance index is a noted
        // future optimization.
        let coll = metered_read(&self.handle);
        let store = match coll.graph_store.as_ref() {
            Some(s) => s,
            None => return Vec::new(),
        };
        store
            .edges_snapshot()
            .into_iter()
            .filter(|e| {
                e.provenance.source_doc.as_deref() == Some(source_doc)
                    && e.provenance.source_chunk_id == Some(source_chunk_id)
            })
            .collect()
    }
}
