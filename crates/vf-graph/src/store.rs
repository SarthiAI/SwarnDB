// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Typed graph store: the Hybrid-mode first-class graph.
//!
//! Authoritative state lives in memory (the same model as `VirtualGraph` and
//! the HNSW index). Durability is the v3 base snapshot (`typed_persistence`)
//! plus the typed delta log (`typed_delta`), which is LSN-aligned with
//! `hnsw.delta` so recovery never drifts (ADR-007 R4(d)). This is an LSM-shaped
//! scheme: a sorted base run, an append-only delta tail, and compaction on
//! snapshot.
//!
//! A hot adjacency cache (LRU, size-configurable) memoises resolved neighbour
//! sets per `(node, edge_type, direction)`. It uses interior mutability so reads
//! stay on `&self` and do not need a write lock. Any structural write clears the
//! cache wholesale: correctness first; finer-grained invalidation is a noted
//! future optimisation, not a correctness gap.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::GraphError;
use crate::lru::LruCache;
use crate::model::{
    DateTime, Edge, EdgeDirection, EdgeId, InternedString, Interner, Node, NodeId, NodeKind, Value,
};
use crate::typed_delta::{TypedDeltaEntry, TypedDeltaReader, TypedDeltaWriter, TypedGraphOp};
use crate::weight::{effective_weight, WeightParams};

type CacheKey = (NodeId, Option<String>, EdgeDirection);

// ADR-015 entity property keys. Mirror vf_extraction::NAME_NORM_KEY; kept local
// to avoid a reverse dependency from vf-graph back up to vf-extraction.
const NAME_NORM_PROP: &str = "name_norm";
const NAME_PROP: &str = "name";

// Resolution key for an entity node, or None for non-entity / empty-norm nodes.
// Mirrors find_entity's name_norm derivation (ADR-015).
fn entity_index_key(node: &Node) -> Option<(String, String)> {
    let label = match &node.kind {
        NodeKind::Entity { label } => label.clone(),
        _ => return None,
    };
    let norm = node
        .properties
        .get(NAME_NORM_PROP)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            node.properties
                .get(NAME_PROP)
                .and_then(|v| v.as_str())
                .map(vf_core::text_norm::normalize_entity_name)
        })?;
    if norm.is_empty() {
        return None;
    }
    Some((label, norm))
}

#[derive(Clone, Debug)]
pub struct GraphStoreConfig {
    /// Max entries in the hot adjacency result cache.
    pub adjacency_cache_capacity: usize,
    /// P09.5: when false, entity resolution uses the legacy O(n) scan (measurement switch only).
    pub name_index_enabled: bool,
}

impl Default for GraphStoreConfig {
    fn default() -> Self {
        Self {
            adjacency_cache_capacity: 4096,
            name_index_enabled: true,
        }
    }
}

/// Filter for paginated node enumeration (ADR-014). All fields optional;
/// `None` means no filter on that dimension.
#[derive(Clone, Debug, Default)]
pub struct NodePageFilter {
    /// When set, only return nodes whose kind matches (content vs entity).
    pub is_entity: Option<bool>,
    /// When set, only return entity nodes whose label equals this.
    pub label: Option<String>,
    /// When set, only return nodes satisfying this property condition. Applied
    /// after the kind/label filters. Structural terms (incident-edge count) are
    /// evaluated against the store; pure property terms are unchanged.
    pub predicate: Option<crate::predicate::Predicate>,
}

/// Filter for paginated edge enumeration (ADR-014).
#[derive(Clone, Debug, Default)]
pub struct EdgePageFilter {
    /// When set, only return edges of this type.
    pub edge_type: Option<String>,
    /// When set, only return edges satisfying this property condition. Applied
    /// after the type filter, over the edge's own properties.
    pub predicate: Option<crate::predicate::Predicate>,
    /// When set, only return edges whose endpoint node has this kind (content vs
    /// entity). The endpoint is the target for outgoing-style joins; here it is
    /// evaluated on BOTH endpoints (an edge passes if either endpoint matches),
    /// which keeps the join cheap and symmetric for undirected reads.
    pub endpoint_is_entity: Option<bool>,
    /// When set, only return edges where an endpoint node has this entity label.
    /// Evaluated by joining the edge to its endpoint nodes.
    pub endpoint_label: Option<String>,
}

/// A page of enumeration results plus the cursor state (ADR-014).
#[derive(Clone, Debug)]
pub struct Page<T> {
    pub items: Vec<T>,
    /// Last id seen in this page; 0 when exhausted. Pass as `after` next call.
    pub next_cursor: u64,
    pub has_more: bool,
}

#[derive(Clone, Debug, Default)]
pub struct GraphStoreStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub interned_labels: usize,
    pub cache_len: usize,
    pub cache_capacity: usize,
    pub last_lsn: u64,
}

/// Manual edge and node CRUD plus neighbour lookups. All mutations carry an LSN
/// so the delta tail stays aligned with `hnsw.delta`.
pub trait GraphStore: Send + Sync {
    fn put_node(&mut self, node: Node, lsn: u64) -> Result<(), GraphError>;
    fn delete_node(&mut self, id: NodeId, lsn: u64) -> Result<bool, GraphError>;

    /// Mutate an existing node's properties, preserving its immutable provenance
    /// (id, kind, source, embedding, created_at, created_by). Appends an audit
    /// entry and persists like put_node so the change survives replay. Returns
    /// the updated node, or None if the node does not exist. `new_properties`
    /// None leaves the property bag unchanged (audit-only touch).
    fn update_node(
        &mut self,
        id: NodeId,
        new_properties: Option<HashMap<String, Value>>,
        actor: Option<String>,
        at: DateTime,
        lsn: u64,
    ) -> Result<Option<Node>, GraphError> {
        let mut node = match self.get_node(id) {
            Some(n) => n,
            None => return Ok(None),
        };
        if let Some(props) = new_properties {
            node.properties = props;
        }
        node.record_audit("updated", actor, at);
        let updated = node.clone();
        self.put_node(node, lsn)?;
        Ok(Some(updated))
    }
    fn put_edge(&mut self, edge: Edge, lsn: u64) -> Result<(), GraphError>;
    fn delete_edge(&mut self, id: EdgeId, lsn: u64) -> Result<bool, GraphError>;
    fn bulk_put_edges(&mut self, edges: Vec<(Edge, u64)>) -> Result<(), GraphError>;

    fn get_node(&self, id: NodeId) -> Option<Node>;
    fn get_edge(&self, id: EdgeId) -> Option<Edge>;
    fn neighbors(&self, node: NodeId, edge_type: Option<&str>, dir: EdgeDirection) -> Vec<NodeId>;
    fn edges_for_node(&self, node: NodeId, dir: EdgeDirection) -> Vec<Edge>;

    /// Count incident edges without materialising them. Must return exactly what
    /// `edges_for_node(node, dir)` filtered by `edge_type` then `.len()` would
    /// (including Both-direction self-loop dedup). The default delegates to
    /// `edges_for_node`; TypedGraphStore overrides it to read adjacency lengths
    /// directly and avoid deep-cloning every Edge at scan scale.
    fn incident_edge_count(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
    ) -> u64 {
        self.edges_for_node(node, dir)
            .into_iter()
            .filter(|e| match edge_type {
                Some(t) => e.edge_type.as_str() == t,
                None => true,
            })
            .count() as u64
    }
    fn weighted_neighbors(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
        params: &WeightParams,
        now_ms: u64,
    ) -> Vec<(NodeId, f64)>;

    /// Temporal-filtered neighbor expansion (P17). Same direction/edge_type
    /// semantics as `neighbors`, but a neighbor is included only via edges passing
    /// `filter` at `now_ms`. UNCACHED by design: the LRU cache key has no temporal
    /// dimension, so temporal expansion bypasses it; the non-temporal `neighbors`
    /// fast path keeps the cache and stays byte-identical (ADR-007 R5). The default
    /// impl walks `edges_for_node`; `TypedGraphStore` overrides it for the fast path.
    fn neighbors_temporal(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
        filter: &crate::temporal::TemporalFilter,
        now_ms: u64,
    ) -> Vec<NodeId> {
        use crate::temporal::edge_passes_temporal;
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for e in self.edges_for_node(node, dir) {
            if let Some(t) = edge_type {
                if e.edge_type.as_str() != t {
                    continue;
                }
            }
            if !edge_passes_temporal(&e, filter, now_ms) {
                continue;
            }
            let other = if e.source == node { e.target } else { e.source };
            if seen.insert(other) {
                out.push(other);
            }
        }
        out
    }

    /// Weighted + optionally temporal neighbor expansion (P17). When `filter` is
    /// None this is exactly `weighted_neighbors` (the cached/non-temporal path).
    /// When Some, only edges passing the filter contribute; MAX effective weight is
    /// kept per neighbor and the result is sorted by NodeId ascending, matching
    /// `weighted_neighbors`' determinism. UNCACHED by design. The default impl
    /// walks `edges_for_node`; `TypedGraphStore` overrides it for the fast path.
    fn weighted_neighbors_temporal(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
        params: &WeightParams,
        filter: Option<&crate::temporal::TemporalFilter>,
        now_ms: u64,
    ) -> Vec<(NodeId, f64)> {
        let f = match filter {
            None => return self.weighted_neighbors(node, edge_type, dir, params, now_ms),
            Some(f) => f,
        };
        use crate::temporal::edge_passes_temporal;
        let mut acc: HashMap<NodeId, f64> = HashMap::new();
        for e in self.edges_for_node(node, dir) {
            if let Some(t) = edge_type {
                if e.edge_type.as_str() != t {
                    continue;
                }
            }
            if !edge_passes_temporal(&e, f, now_ms) {
                continue;
            }
            let other = if e.source == node { e.target } else { e.source };
            let w = effective_weight(&e, params, now_ms);
            acc.entry(other)
                .and_modify(|cur| {
                    if w > *cur {
                        *cur = w;
                    }
                })
                .or_insert(w);
        }
        let mut out: Vec<(NodeId, f64)> = acc.into_iter().collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;

    /// Paginated node enumeration over sorted ids (ADR-014). Returns at most
    /// `limit` nodes with id strictly greater than `after`, in ascending id
    /// order, applying the optional filter. Does not clone the whole node set.
    fn nodes_page(&self, after: NodeId, limit: usize, filter: &NodePageFilter) -> Page<Node>;

    /// Paginated edge enumeration over sorted ids (ADR-014). Returns at most
    /// `limit` edges with id strictly greater than `after`, in ascending id
    /// order, applying the optional filter. Does not clone the whole edge set.
    fn edges_page(&self, after: EdgeId, limit: usize, filter: &EdgePageFilter) -> Page<Edge>;

    fn intern(&mut self, label: &str) -> InternedString;
    /// Crate-local / test allocator only. In server-hosted collections, node ids
    /// are allocated from the unified id authority (InMemoryVectorStore::alloc_id);
    /// do not call this for server entity creation.
    fn alloc_node_id(&mut self) -> NodeId;
    fn alloc_edge_id(&mut self) -> EdgeId;
    fn stats(&self) -> GraphStoreStats;
}

pub struct TypedGraphStore {
    nodes: HashMap<NodeId, Node>,
    edges: HashMap<EdgeId, Edge>,
    out_adj: HashMap<NodeId, Vec<EdgeId>>,
    in_adj: HashMap<NodeId, Vec<EdgeId>>,
    interner: Interner,
    next_node_id: u64,
    next_edge_id: u64,
    cache: Mutex<LruCache<CacheKey, Vec<NodeId>>>,
    config: GraphStoreConfig,
    delta_writer: Option<Arc<Mutex<TypedDeltaWriter>>>,
    last_lsn: u64,
    // P09.5 derived secondary index, rebuilt on load like the adjacency lists; smallest NodeId in the set is the canonical resolution.
    name_index: HashMap<String, HashMap<String, BTreeSet<NodeId>>>,
}

impl TypedGraphStore {
    pub fn new(config: GraphStoreConfig) -> Self {
        let cap = config.adjacency_cache_capacity;
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            out_adj: HashMap::new(),
            in_adj: HashMap::new(),
            interner: Interner::new(),
            next_node_id: 1,
            next_edge_id: 1,
            cache: Mutex::new(LruCache::new(cap)),
            config,
            delta_writer: None,
            last_lsn: 0,
            name_index: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(GraphStoreConfig::default())
    }

    /// Build a store from base-snapshot parts. Recomputes adjacency and the id
    /// counters, and re-interns edge labels into this store's interner.
    pub fn from_parts(nodes: Vec<Node>, edges: Vec<Edge>, config: GraphStoreConfig) -> Self {
        let mut store = Self::new(config);
        let mut max_node = 0u64;
        for n in nodes {
            max_node = max_node.max(n.id.0);
            store.nodes.insert(n.id, n);
        }
        let mut max_edge = 0u64;
        for mut e in edges {
            max_edge = max_edge.max(e.id.0);
            e.edge_type = store.interner.intern(e.edge_type.as_str());
            store.index_edge(&e);
            store.edges.insert(e.id, e);
        }
        store.next_node_id = max_node + 1;
        store.next_edge_id = max_edge + 1;
        // P09.5: rebuild the derived entity-name index in one O(N) pass on load.
        let mut name_index: HashMap<String, HashMap<String, BTreeSet<NodeId>>> = HashMap::new();
        for node in store.nodes.values() {
            if let Some((label, norm)) = entity_index_key(node) {
                name_index
                    .entry(label)
                    .or_default()
                    .entry(norm)
                    .or_default()
                    .insert(node.id);
            }
        }
        store.name_index = name_index;
        store
    }

    // ── adjacency index maintenance ──

    fn index_edge(&mut self, e: &Edge) {
        self.out_adj.entry(e.source).or_default().push(e.id);
        self.in_adj.entry(e.target).or_default().push(e.id);
    }

    fn deindex_edge(&mut self, e: &Edge) {
        if let Some(v) = self.out_adj.get_mut(&e.source) {
            v.retain(|id| *id != e.id);
        }
        if let Some(v) = self.in_adj.get_mut(&e.target) {
            v.retain(|id| *id != e.id);
        }
    }

    // ── P09.5 derived entity-name index maintenance ──

    fn name_index_insert(&mut self, node: &Node) {
        if let Some((label, norm)) = entity_index_key(node) {
            self.name_index
                .entry(label)
                .or_default()
                .entry(norm)
                .or_default()
                .insert(node.id);
        }
    }

    fn name_index_remove(&mut self, node: &Node) {
        if let Some((label, norm)) = entity_index_key(node) {
            if let Some(bucket) = self.name_index.get_mut(&label) {
                if let Some(set) = bucket.get_mut(&norm) {
                    set.remove(&node.id);
                    if set.is_empty() {
                        bucket.remove(&norm);
                    }
                }
                if bucket.is_empty() {
                    self.name_index.remove(&label);
                }
            }
        }
    }

    // ── pure in-memory apply (no delta emission); used by replay/load ──

    pub fn apply_put_node(&mut self, node: Node) {
        // Re-key the index: drop the prior node's key, then add the new one.
        if let Some(old) = self.nodes.get(&node.id) {
            let old = old.clone();
            self.name_index_remove(&old);
        }
        self.name_index_insert(&node);
        self.nodes.insert(node.id, node);
    }

    pub fn apply_delete_node(&mut self, id: NodeId) -> bool {
        // Drop the index key before removing the node from the primary map.
        if let Some(node) = self.nodes.get(&id) {
            let node = node.clone();
            self.name_index_remove(&node);
        }
        let existed = self.nodes.remove(&id).is_some();
        let mut incident: Vec<EdgeId> = Vec::new();
        if let Some(v) = self.out_adj.remove(&id) {
            incident.extend(v);
        }
        if let Some(v) = self.in_adj.remove(&id) {
            incident.extend(v);
        }
        for eid in incident {
            if let Some(e) = self.edges.remove(&eid) {
                if e.source != id {
                    if let Some(v) = self.out_adj.get_mut(&e.source) {
                        v.retain(|x| *x != eid);
                    }
                }
                if e.target != id {
                    if let Some(v) = self.in_adj.get_mut(&e.target) {
                        v.retain(|x| *x != eid);
                    }
                }
            }
        }
        existed
    }

    pub fn apply_put_edge(&mut self, edge: Edge) {
        if let Some(old) = self.edges.get(&edge.id).cloned() {
            self.deindex_edge(&old);
        }
        self.index_edge(&edge);
        self.edges.insert(edge.id, edge);
    }

    pub fn apply_delete_edge(&mut self, id: EdgeId) -> bool {
        if let Some(e) = self.edges.remove(&id) {
            self.deindex_edge(&e);
            true
        } else {
            false
        }
    }

    fn emit(&self, lsn: u64, op: TypedGraphOp) {
        if let Some(ref w) = self.delta_writer {
            let entry = TypedDeltaEntry { lsn, op };
            if let Err(e) = w.lock().append(&entry) {
                log::warn!("failed to append typed graph delta at lsn {lsn}: {e}");
            }
        }
    }

    fn clear_cache(&self) {
        self.cache.lock().clear();
    }

    // ── delta writer management (mirrors VirtualGraph) ──

    pub fn set_delta_writer(&mut self, writer: TypedDeltaWriter) {
        self.last_lsn = self.last_lsn.max(writer.last_lsn());
        self.delta_writer = Some(Arc::new(Mutex::new(writer)));
    }

    pub fn take_delta_writer(&mut self) -> Option<TypedDeltaWriter> {
        self.delta_writer
            .take()
            .and_then(|arc| Arc::try_unwrap(arc).ok().map(|m| m.into_inner()))
    }

    pub fn sync_delta(&self) -> Result<(), GraphError> {
        if let Some(ref w) = self.delta_writer {
            w.lock().sync()?;
        }
        Ok(())
    }

    pub fn last_lsn(&self) -> u64 {
        self.last_lsn
    }

    pub fn config(&self) -> &GraphStoreConfig {
        &self.config
    }

    // ── P09.5 entity resolution lookups ──

    /// Exact entity resolution by normalized name. O(1) via the index when
    /// enabled; deterministic (smallest matching NodeId). Falls back to an O(n)
    /// scan when the index is disabled.
    pub fn find_entity_by_norm(&self, label: &str, name_norm: &str) -> Option<NodeId> {
        if self.config.name_index_enabled {
            self.name_index.get(label)?.get(name_norm)?.iter().next().copied()
        } else {
            self.scan_entity_by_norm(label, name_norm)
        }
    }

    fn scan_entity_by_norm(&self, label: &str, name_norm: &str) -> Option<NodeId> {
        let mut best: Option<NodeId> = None;
        for node in self.nodes.values() {
            if let Some((l, n)) = entity_index_key(node) {
                if l == label && n == name_norm {
                    best = Some(match best {
                        Some(b) => b.min(node.id),
                        None => node.id,
                    });
                }
            }
        }
        best
    }

    /// Same-label candidate (name_norm, node_id) pairs for fuzzy matching, sorted
    /// by NodeId ascending (deterministic). Index bucket when enabled, else a full scan.
    pub fn entity_candidates_for_label(&self, label: &str) -> Vec<(String, NodeId)> {
        let mut out: Vec<(String, NodeId)> = Vec::new();
        if self.config.name_index_enabled {
            if let Some(bucket) = self.name_index.get(label) {
                for (norm, ids) in bucket {
                    for id in ids {
                        out.push((norm.clone(), *id));
                    }
                }
            }
        } else {
            for node in self.nodes.values() {
                if let Some((l, n)) = entity_index_key(node) {
                    if l == label {
                        out.push((n, node.id));
                    }
                }
            }
        }
        out.sort_by_key(|(_, id)| id.0);
        out
    }

    pub fn name_index_enabled(&self) -> bool {
        self.config.name_index_enabled
    }

    // ── snapshot export ──

    pub fn nodes_snapshot(&self) -> Vec<Node> {
        self.nodes.values().cloned().collect()
    }

    /// Largest node id in the LIVE node set (0 when empty). Reads the actual
    /// nodes map, which after base + delta replay is the true max; do NOT use
    /// `next_node_id` for this. Recovery seeds the unified id authority above
    /// this value so reused server entity ids never collide with vector ids.
    pub fn max_node_id(&self) -> u64 {
        self.nodes.keys().map(|n| n.0).max().unwrap_or(0)
    }

    pub fn edges_snapshot(&self) -> Vec<Edge> {
        self.edges.values().cloned().collect()
    }

    // ── delta replay ──

    pub fn replay_delta(&mut self, path: &Path) -> Result<u64, GraphError> {
        self.replay_delta_after_lsn(path, 0)
    }

    pub fn replay_delta_after_lsn(&mut self, path: &Path, after: u64) -> Result<u64, GraphError> {
        let mut reader = TypedDeltaReader::open(path)?;
        let mut count = 0u64;
        while let Some(entry) = reader.next_entry()? {
            if entry.lsn > after {
                self.apply_op(entry.op);
                self.last_lsn = self.last_lsn.max(entry.lsn);
                count += 1;
            }
        }
        self.clear_cache();
        Ok(count)
    }

    fn apply_op(&mut self, op: TypedGraphOp) {
        match op {
            TypedGraphOp::PutNode(n) => self.apply_put_node(n),
            TypedGraphOp::DeleteNode(id) => {
                self.apply_delete_node(id);
            }
            TypedGraphOp::PutEdge(mut e) => {
                e.edge_type = self.interner.intern(e.edge_type.as_str());
                self.apply_put_edge(e);
            }
            TypedGraphOp::DeleteEdge(id) => {
                self.apply_delete_edge(id);
            }
        }
    }

    // ── neighbour computation (uncached) ──

    fn collect_dir(
        &self,
        ids: Option<&Vec<EdgeId>>,
        node: NodeId,
        edge_type: Option<&str>,
        out: &mut Vec<NodeId>,
        seen: &mut HashSet<NodeId>,
    ) {
        if let Some(list) = ids {
            for eid in list {
                if let Some(e) = self.edges.get(eid) {
                    if let Some(t) = edge_type {
                        if e.edge_type.as_str() != t {
                            continue;
                        }
                    }
                    let other = if e.source == node { e.target } else { e.source };
                    if seen.insert(other) {
                        out.push(other);
                    }
                }
            }
        }
    }

    fn compute_neighbors(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
    ) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        match dir {
            EdgeDirection::Outgoing => {
                self.collect_dir(self.out_adj.get(&node), node, edge_type, &mut out, &mut seen)
            }
            EdgeDirection::Incoming => {
                self.collect_dir(self.in_adj.get(&node), node, edge_type, &mut out, &mut seen)
            }
            EdgeDirection::Both => {
                self.collect_dir(self.out_adj.get(&node), node, edge_type, &mut out, &mut seen);
                self.collect_dir(self.in_adj.get(&node), node, edge_type, &mut out, &mut seen);
            }
        }
        out
    }

    // ── temporal neighbour computation (uncached, P17) ──

    // Mirrors collect_dir exactly but also applies the temporal filter per edge.
    fn collect_dir_temporal(
        &self,
        ids: Option<&Vec<EdgeId>>,
        node: NodeId,
        edge_type: Option<&str>,
        filter: &crate::temporal::TemporalFilter,
        now_ms: u64,
        out: &mut Vec<NodeId>,
        seen: &mut HashSet<NodeId>,
    ) {
        use crate::temporal::edge_passes_temporal;
        if let Some(list) = ids {
            for eid in list {
                if let Some(e) = self.edges.get(eid) {
                    if let Some(t) = edge_type {
                        if e.edge_type.as_str() != t {
                            continue;
                        }
                    }
                    if !edge_passes_temporal(e, filter, now_ms) {
                        continue;
                    }
                    let other = if e.source == node { e.target } else { e.source };
                    if seen.insert(other) {
                        out.push(other);
                    }
                }
            }
        }
    }

    fn compute_neighbors_temporal(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
        filter: &crate::temporal::TemporalFilter,
        now_ms: u64,
    ) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        match dir {
            EdgeDirection::Outgoing => self.collect_dir_temporal(
                self.out_adj.get(&node),
                node,
                edge_type,
                filter,
                now_ms,
                &mut out,
                &mut seen,
            ),
            EdgeDirection::Incoming => self.collect_dir_temporal(
                self.in_adj.get(&node),
                node,
                edge_type,
                filter,
                now_ms,
                &mut out,
                &mut seen,
            ),
            EdgeDirection::Both => {
                self.collect_dir_temporal(
                    self.out_adj.get(&node),
                    node,
                    edge_type,
                    filter,
                    now_ms,
                    &mut out,
                    &mut seen,
                );
                self.collect_dir_temporal(
                    self.in_adj.get(&node),
                    node,
                    edge_type,
                    filter,
                    now_ms,
                    &mut out,
                    &mut seen,
                );
            }
        }
        out
    }

    // ── weighted neighbour computation (uncached) ──

    // Mirrors collect_dir exactly but keeps MAX weight per neighbor for parallel edges.
    fn collect_dir_weighted(
        &self,
        ids: Option<&Vec<EdgeId>>,
        node: NodeId,
        edge_type: Option<&str>,
        params: &WeightParams,
        now_ms: u64,
        acc: &mut HashMap<NodeId, f64>,
    ) {
        if let Some(list) = ids {
            for eid in list {
                if let Some(e) = self.edges.get(eid) {
                    if let Some(t) = edge_type {
                        if e.edge_type.as_str() != t {
                            continue;
                        }
                    }
                    let other = if e.source == node { e.target } else { e.source };
                    let w = effective_weight(e, params, now_ms);
                    acc.entry(other)
                        .and_modify(|cur| {
                            if w > *cur {
                                *cur = w;
                            }
                        })
                        .or_insert(w);
                }
            }
        }
    }

    // Mirrors collect_dir_weighted exactly but also applies the temporal filter per edge (P17).
    fn collect_dir_weighted_temporal(
        &self,
        ids: Option<&Vec<EdgeId>>,
        node: NodeId,
        edge_type: Option<&str>,
        params: &WeightParams,
        filter: &crate::temporal::TemporalFilter,
        now_ms: u64,
        acc: &mut HashMap<NodeId, f64>,
    ) {
        use crate::temporal::edge_passes_temporal;
        if let Some(list) = ids {
            for eid in list {
                if let Some(e) = self.edges.get(eid) {
                    if let Some(t) = edge_type {
                        if e.edge_type.as_str() != t {
                            continue;
                        }
                    }
                    if !edge_passes_temporal(e, filter, now_ms) {
                        continue;
                    }
                    let other = if e.source == node { e.target } else { e.source };
                    let w = effective_weight(e, params, now_ms);
                    acc.entry(other)
                        .and_modify(|cur| {
                            if w > *cur {
                                *cur = w;
                            }
                        })
                        .or_insert(w);
                }
            }
        }
    }

    // ── edge endpoint-node join (filtered edge reads) ──

    /// Whether an edge satisfies the endpoint-node constraints in `filter`: an
    /// endpoint (source or target) must match the requested kind and label. An
    /// edge passes if EITHER endpoint matches, so a relationship edge touching a
    /// node of the wanted label is included regardless of edge direction. A
    /// constraint over a missing endpoint node fails (no node, no match).
    fn edge_endpoint_matches(&self, edge: &Edge, filter: &EdgePageFilter) -> bool {
        let endpoint_ok = |id: NodeId| -> bool {
            let node = match self.nodes.get(&id) {
                Some(n) => n,
                None => return false,
            };
            if let Some(want_entity) = filter.endpoint_is_entity {
                if node.is_entity() != want_entity {
                    return false;
                }
            }
            if let Some(ref want_label) = filter.endpoint_label {
                match &node.kind {
                    NodeKind::Entity { label } if label == want_label => {}
                    _ => return false,
                }
            }
            true
        };
        endpoint_ok(edge.source) || endpoint_ok(edge.target)
    }
}

impl GraphStore for TypedGraphStore {
    fn put_node(&mut self, node: Node, lsn: u64) -> Result<(), GraphError> {
        let op = TypedGraphOp::PutNode(node.clone());
        self.apply_put_node(node);
        self.emit(lsn, op);
        self.last_lsn = self.last_lsn.max(lsn);
        Ok(())
    }

    fn delete_node(&mut self, id: NodeId, lsn: u64) -> Result<bool, GraphError> {
        let existed = self.apply_delete_node(id);
        self.emit(lsn, TypedGraphOp::DeleteNode(id));
        self.last_lsn = self.last_lsn.max(lsn);
        self.clear_cache();
        Ok(existed)
    }

    fn put_edge(&mut self, mut edge: Edge, lsn: u64) -> Result<(), GraphError> {
        edge.edge_type = self.interner.intern(edge.edge_type.as_str());
        let op = TypedGraphOp::PutEdge(edge.clone());
        self.apply_put_edge(edge);
        self.emit(lsn, op);
        self.last_lsn = self.last_lsn.max(lsn);
        self.clear_cache();
        metrics::counter!("swarndb_graph_edge_writes_total").increment(1);
        Ok(())
    }

    fn delete_edge(&mut self, id: EdgeId, lsn: u64) -> Result<bool, GraphError> {
        let existed = self.apply_delete_edge(id);
        self.emit(lsn, TypedGraphOp::DeleteEdge(id));
        self.last_lsn = self.last_lsn.max(lsn);
        self.clear_cache();
        if existed {
            metrics::counter!("swarndb_graph_edge_deletes_total").increment(1);
        }
        Ok(existed)
    }

    // Each edge goes through put_edge, which records swarndb_graph_edge_writes_total
    // once per edge; no extra increment here to avoid double counting.
    fn bulk_put_edges(&mut self, edges: Vec<(Edge, u64)>) -> Result<(), GraphError> {
        for (edge, lsn) in edges {
            self.put_edge(edge, lsn)?;
        }
        Ok(())
    }

    fn get_node(&self, id: NodeId) -> Option<Node> {
        self.nodes.get(&id).cloned()
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.edges.get(&id).cloned()
    }

    fn neighbors(&self, node: NodeId, edge_type: Option<&str>, dir: EdgeDirection) -> Vec<NodeId> {
        let key: CacheKey = (node, edge_type.map(|s| s.to_string()), dir);
        {
            let mut cache = self.cache.lock();
            if let Some(v) = cache.get(&key) {
                let hit = v.clone();
                drop(cache);
                metrics::counter!("swarndb_graph_adjacency_cache_hits_total").increment(1);
                return hit;
            }
        }
        metrics::counter!("swarndb_graph_adjacency_cache_misses_total").increment(1);
        let result = self.compute_neighbors(node, edge_type, dir);
        self.cache.lock().put(key, result.clone());
        result
    }

    fn edges_for_node(&self, node: NodeId, dir: EdgeDirection) -> Vec<Edge> {
        let mut ids: Vec<EdgeId> = Vec::new();
        let mut seen: HashSet<EdgeId> = HashSet::new();
        let mut take = |list: Option<&Vec<EdgeId>>, seen: &mut HashSet<EdgeId>| {
            if let Some(v) = list {
                for id in v {
                    if seen.insert(*id) {
                        ids.push(*id);
                    }
                }
            }
        };
        match dir {
            EdgeDirection::Outgoing => take(self.out_adj.get(&node), &mut seen),
            EdgeDirection::Incoming => take(self.in_adj.get(&node), &mut seen),
            EdgeDirection::Both => {
                take(self.out_adj.get(&node), &mut seen);
                take(self.in_adj.get(&node), &mut seen);
            }
        }
        ids.into_iter()
            .filter_map(|eid| self.edges.get(&eid).cloned())
            .collect()
    }

    /// Count-only incident-edge lookup that avoids cloning Edge structs. Reads
    /// adjacency lengths directly; matches `edges_for_node().len()` exactly,
    /// including Both-direction self-loop dedup (a self-loop sits in both out_adj
    /// and in_adj for the node but is one edge).
    fn incident_edge_count(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
    ) -> u64 {
        let out = self.out_adj.get(&node);
        let in_ = self.in_adj.get(&node);
        match edge_type {
            // No type filter: count by adjacency-list lengths, subtracting the
            // self-loops once under Both (they appear in both lists for `node`).
            None => match dir {
                EdgeDirection::Outgoing => out.map_or(0, |v| v.len()) as u64,
                EdgeDirection::Incoming => in_.map_or(0, |v| v.len()) as u64,
                EdgeDirection::Both => {
                    let out_len = out.map_or(0, |v| v.len());
                    let in_len = in_.map_or(0, |v| v.len());
                    let self_loops = out.map_or(0, |v| {
                        v.iter()
                            .filter(|eid| {
                                self.edges
                                    .get(eid)
                                    .map(|e| e.source == node && e.target == node)
                                    .unwrap_or(false)
                            })
                            .count()
                    });
                    (out_len + in_len - self_loops) as u64
                }
            },
            // Typed filter: walk incident edge ids checking only the type, deduping
            // self-loops under Both so the count matches edges_for_node exactly.
            Some(t) => {
                let mut count: u64 = 0;
                let mut seen: HashSet<EdgeId> = HashSet::new();
                let mut tally = |list: Option<&Vec<EdgeId>>, seen: &mut HashSet<EdgeId>| {
                    if let Some(v) = list {
                        for eid in v {
                            if let Some(e) = self.edges.get(eid) {
                                if e.edge_type.as_str() == t && seen.insert(*eid) {
                                    count += 1;
                                }
                            }
                        }
                    }
                };
                match dir {
                    EdgeDirection::Outgoing => tally(out, &mut seen),
                    EdgeDirection::Incoming => tally(in_, &mut seen),
                    EdgeDirection::Both => {
                        tally(out, &mut seen);
                        tally(in_, &mut seen);
                    }
                }
                count
            }
        }
    }

    /// Weighted neighbor expansion. Returns (neighbor, effective_weight) pairs, deduped by
    /// neighbor keeping the MAX effective weight across parallel edges. Mirrors compute_neighbors'
    /// direction and edge_type filtering exactly. With WeightParams::is_noop() every weight is 1.0,
    /// so callers get the same neighbor set as neighbors(), each paired with 1.0.
    /// Result is sorted by neighbor NodeId ascending for determinism.
    fn weighted_neighbors(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
        params: &WeightParams,
        now_ms: u64,
    ) -> Vec<(NodeId, f64)> {
        let mut acc: HashMap<NodeId, f64> = HashMap::new();
        match dir {
            EdgeDirection::Outgoing => self.collect_dir_weighted(
                self.out_adj.get(&node),
                node,
                edge_type,
                params,
                now_ms,
                &mut acc,
            ),
            EdgeDirection::Incoming => self.collect_dir_weighted(
                self.in_adj.get(&node),
                node,
                edge_type,
                params,
                now_ms,
                &mut acc,
            ),
            EdgeDirection::Both => {
                self.collect_dir_weighted(
                    self.out_adj.get(&node),
                    node,
                    edge_type,
                    params,
                    now_ms,
                    &mut acc,
                );
                self.collect_dir_weighted(
                    self.in_adj.get(&node),
                    node,
                    edge_type,
                    params,
                    now_ms,
                    &mut acc,
                );
            }
        }
        let mut out: Vec<(NodeId, f64)> = acc.into_iter().collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    /// Temporal neighbor expansion over adjacency directly (P17), avoiding the
    /// clone-heavy `edges_for_node` default. UNCACHED: the LRU is untouched, so the
    /// non-temporal `neighbors` fast path stays byte-identical.
    fn neighbors_temporal(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
        filter: &crate::temporal::TemporalFilter,
        now_ms: u64,
    ) -> Vec<NodeId> {
        self.compute_neighbors_temporal(node, edge_type, dir, filter, now_ms)
    }

    /// Weighted + optionally temporal neighbor expansion over adjacency directly (P17).
    /// When `filter` is None this delegates to the cached/non-temporal `weighted_neighbors`
    /// so the default-off path is unchanged. When Some, only filter-passing edges contribute,
    /// MAX effective weight is kept per neighbor, and the result is sorted by NodeId ascending.
    fn weighted_neighbors_temporal(
        &self,
        node: NodeId,
        edge_type: Option<&str>,
        dir: EdgeDirection,
        params: &WeightParams,
        filter: Option<&crate::temporal::TemporalFilter>,
        now_ms: u64,
    ) -> Vec<(NodeId, f64)> {
        let f = match filter {
            None => return self.weighted_neighbors(node, edge_type, dir, params, now_ms),
            Some(f) => f,
        };
        let mut acc: HashMap<NodeId, f64> = HashMap::new();
        match dir {
            EdgeDirection::Outgoing => self.collect_dir_weighted_temporal(
                self.out_adj.get(&node),
                node,
                edge_type,
                params,
                f,
                now_ms,
                &mut acc,
            ),
            EdgeDirection::Incoming => self.collect_dir_weighted_temporal(
                self.in_adj.get(&node),
                node,
                edge_type,
                params,
                f,
                now_ms,
                &mut acc,
            ),
            EdgeDirection::Both => {
                self.collect_dir_weighted_temporal(
                    self.out_adj.get(&node),
                    node,
                    edge_type,
                    params,
                    f,
                    now_ms,
                    &mut acc,
                );
                self.collect_dir_weighted_temporal(
                    self.in_adj.get(&node),
                    node,
                    edge_type,
                    params,
                    f,
                    now_ms,
                    &mut acc,
                );
            }
        }
        let mut out: Vec<(NodeId, f64)> = acc.into_iter().collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn nodes_page(&self, after: NodeId, limit: usize, filter: &NodePageFilter) -> Page<Node> {
        // Collect ids strictly greater than the cursor, sort, then walk in order
        // applying the filter until the page is full. Bounded by the matched
        // page, never by a clone of the whole node set.
        let mut ids: Vec<NodeId> = self
            .nodes
            .keys()
            .copied()
            .filter(|id| id.0 > after.0)
            .collect();
        ids.sort_unstable();

        let mut items: Vec<Node> = Vec::new();
        let mut last_id: u64 = 0;
        let mut has_more = false;
        for id in ids {
            let node = match self.nodes.get(&id) {
                Some(n) => n,
                None => continue,
            };
            if let Some(want_entity) = filter.is_entity {
                if node.is_entity() != want_entity {
                    continue;
                }
            }
            if let Some(ref want_label) = filter.label {
                match &node.kind {
                    crate::model::NodeKind::Entity { label } if label == want_label => {}
                    _ => continue,
                }
            }
            // Property condition (incl. structural incident-edge-count terms),
            // evaluated against the store so structural terms resolve.
            if let Some(ref pred) = filter.predicate {
                if !pred.eval_node_with_store(node, self) {
                    continue;
                }
            }
            if items.len() == limit {
                // A further match exists past the page boundary.
                has_more = true;
                break;
            }
            last_id = id.0;
            items.push(node.clone());
        }

        let next_cursor = if has_more { last_id } else { 0 };
        Page { items, next_cursor, has_more }
    }

    fn edges_page(&self, after: EdgeId, limit: usize, filter: &EdgePageFilter) -> Page<Edge> {
        let mut ids: Vec<EdgeId> = self
            .edges
            .keys()
            .copied()
            .filter(|id| id.0 > after.0)
            .collect();
        ids.sort_unstable();

        let mut items: Vec<Edge> = Vec::new();
        let mut last_id: u64 = 0;
        let mut has_more = false;
        for id in ids {
            let edge = match self.edges.get(&id) {
                Some(e) => e,
                None => continue,
            };
            if let Some(ref want_type) = filter.edge_type {
                if edge.edge_type.as_str() != want_type.as_str() {
                    continue;
                }
            }
            // Property condition over the edge's own properties.
            if let Some(ref pred) = filter.predicate {
                if !pred.eval_edge(edge) {
                    continue;
                }
            }
            // Endpoint-node constraints: join the edge to its endpoint nodes and
            // require that an endpoint matches the requested kind/label.
            if filter.endpoint_is_entity.is_some() || filter.endpoint_label.is_some() {
                if !self.edge_endpoint_matches(edge, filter) {
                    continue;
                }
            }
            if items.len() == limit {
                has_more = true;
                break;
            }
            last_id = id.0;
            items.push(edge.clone());
        }

        let next_cursor = if has_more { last_id } else { 0 };
        Page { items, next_cursor, has_more }
    }

    fn intern(&mut self, label: &str) -> InternedString {
        self.interner.intern(label)
    }

    /// Crate-local / test allocator only. In server-hosted collections, node ids
    /// are allocated from the unified id authority (InMemoryVectorStore::alloc_id);
    /// do not call this for server entity creation.
    fn alloc_node_id(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        NodeId(id)
    }

    fn alloc_edge_id(&mut self) -> EdgeId {
        let id = self.next_edge_id;
        self.next_edge_id += 1;
        EdgeId(id)
    }

    fn stats(&self) -> GraphStoreStats {
        let cache = self.cache.lock();
        GraphStoreStats {
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            interned_labels: self.interner.len(),
            cache_len: cache.len(),
            cache_capacity: cache.capacity(),
            last_lsn: self.last_lsn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Node, NodeSource};

    fn content(store: &mut TypedGraphStore) -> NodeId {
        let id = store.alloc_node_id();
        store
            .put_node(Node::content(id, None, NodeSource::Ingested), 1)
            .unwrap();
        id
    }

    #[test]
    fn manual_edge_crud_and_neighbors() {
        let mut s = TypedGraphStore::with_defaults();
        let a = content(&mut s);
        let b = content(&mut s);
        let label = s.intern("CITES");
        let eid = s.alloc_edge_id();
        s.put_edge(Edge::manual(eid, a, b, label), 2).unwrap();

        assert_eq!(s.neighbors(a, Some("CITES"), EdgeDirection::Outgoing), vec![b]);
        assert_eq!(s.neighbors(b, Some("CITES"), EdgeDirection::Incoming), vec![a]);
        assert!(s.neighbors(a, Some("PARTY_TO"), EdgeDirection::Outgoing).is_empty());
        assert_eq!(s.edges_for_node(a, EdgeDirection::Both).len(), 1);

        assert!(s.delete_edge(eid, 3).unwrap());
        assert!(s.neighbors(a, Some("CITES"), EdgeDirection::Outgoing).is_empty());
    }

    #[test]
    fn delete_node_removes_incident_edges() {
        let mut s = TypedGraphStore::with_defaults();
        let a = content(&mut s);
        let b = content(&mut s);
        let label = s.intern("CITES");
        let eid = s.alloc_edge_id();
        s.put_edge(Edge::manual(eid, a, b, label), 2).unwrap();
        assert!(s.delete_node(a, 3).unwrap());
        assert_eq!(s.edge_count(), 0);
        assert!(s.edges_for_node(b, EdgeDirection::Both).is_empty());
    }
}
