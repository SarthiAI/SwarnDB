// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, RwLock};

/// Tracks the optimization state of a collection.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum CollectionStatus {
    Ready,
    PendingOptimization,
    Optimizing,
}

impl CollectionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            CollectionStatus::Ready => "ready",
            CollectionStatus::PendingOptimization => "pending_optimization",
            CollectionStatus::Optimizing => "optimizing",
        }
    }
}
use vf_core::store::InMemoryVectorStore;
use vf_core::types::{CollectionConfig, Metadata, QuantizationConfig, VectorId};
use vf_graph::VirtualGraph;
use vf_index::arena::VectorArena;
use vf_index::hnsw::HnswIndex;
use vf_index::hnsw_delta::HnswDeltaWriter;
use vf_index::hnsw_persistence::deserialize_topology_mmap;
use vf_index::quantized_hnsw::QuantizedHnswIndex;
use vf_index::traits::{PersistableIndex, VectorIndex};
use vf_graph::graph_delta::GraphDeltaWriter;
use vf_graph::persistence::deserialize_base as deserialize_graph_base;
use vf_query::IndexManager;
use vf_storage::collection::CollectionManager;
use vf_storage::recovery::{plan_recovery, RecoveryStrategy};
use vf_storage::StorageError;

/// Cached metadata store that avoids rebuilding the HashMap on every query.
/// Only rebuilds when the store's generation counter has changed.
/// Uses Arc internally so concurrent readers get a cheap reference-count bump
/// instead of cloning the entire HashMap.
pub struct MetadataCache {
    cache: Mutex<(u64, Arc<HashMap<VectorId, Metadata>>)>,
}

impl MetadataCache {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new((u64::MAX, Arc::new(HashMap::new()))),
        }
    }

    /// Returns a shared reference to the cached metadata store, rebuilding
    /// only if the store's generation has changed since the last build.
    /// Uses `iter_metadata` to avoid cloning vector data during rebuild.
    pub fn get_or_rebuild(&self, store: &InMemoryVectorStore) -> Arc<HashMap<VectorId, Metadata>> {
        let current_gen = store.generation();
        let mut guard = self.cache.lock();
        if guard.0 != current_gen {
            let metadata_store: HashMap<VectorId, Metadata> =
                store.iter_metadata().into_iter().collect();
            *guard = (current_gen, Arc::new(metadata_store));
        }
        Arc::clone(&guard.1)
    }
}

/// Result returned by the optimize endpoint.
pub struct OptimizeResult {
    pub status: String,
    pub message: String,
    pub duration_ms: u64,
    pub vectors_processed: u64,
}

/// Per-collection state holding all components needed for vector operations.
pub struct CollectionState {
    pub config: CollectionConfig,
    pub store: InMemoryVectorStore,
    pub index: Box<dyn PersistableIndex>,
    pub index_manager: IndexManager,
    pub graph: VirtualGraph,
    pub metadata_cache: MetadataCache,
    pub status: Arc<std::sync::RwLock<CollectionStatus>>,
    /// True if HNSW index build was deferred during bulk insert.
    pub deferred_index: Arc<AtomicBool>,
    /// True if virtual graph computation was deferred during bulk insert.
    pub deferred_graph: Arc<AtomicBool>,
    /// True if metadata indexing was skipped during bulk insert.
    pub deferred_metadata: Arc<AtomicBool>,
    /// True if any mutation has occurred since last snapshot.
    pub dirty: Arc<AtomicBool>,
    /// Number of mutations since last snapshot.
    pub mutation_count: Arc<AtomicU64>,
}

/// Global application state shared across all gRPC services.
#[derive(Clone)]
pub struct AppState {
    pub collections: Arc<RwLock<HashMap<String, CollectionState>>>,
    pub collection_manager: Arc<RwLock<CollectionManager>>,
    pub config: crate::config::ServerConfig,
    pub max_ef_search: usize,
    pub max_batch_lock_size: u32,
    pub max_wal_flush_interval: u32,
    pub max_ef_construction: u32,
}

impl AppState {
    /// Create a new AppState with a CollectionManager rooted at `storage_path`.
    ///
    /// The CollectionManager will create the directory if it does not exist and
    /// scan for any previously persisted collections. For each persisted
    /// collection, the in-memory vector store and HNSW index are rebuilt from
    /// the stored vectors so that search is immediately available after restart.
    pub fn new(
        storage_path: &Path,
        max_ef_search: usize,
        max_batch_lock_size: u32,
        max_wal_flush_interval: u32,
        max_ef_construction: u32,
        config: crate::config::ServerConfig,
    ) -> Result<Self, StorageError> {
        let collection_manager = CollectionManager::new(storage_path)?;

        let mut collections = HashMap::new();

        // Rebuild in-memory state for every persisted collection.
        for name in collection_manager.list_collections() {
            let name = name.to_string();

            let collection = match collection_manager.get_collection(&name) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        collection = %name,
                        "failed to load collection, skipping: {e}"
                    );
                    continue;
                }
            };

            let config = collection.config().clone();
            let dimension = config.dimension;
            let distance_metric = config.distance_metric;
            let collection_dir = collection.collection_dir().to_path_buf();

            // Plan recovery strategy based on available files.
            let plan = plan_recovery(&name, &collection_dir);
            tracing::info!(
                collection = %name,
                strategy = ?plan.strategy,
                "recovery plan: {:?}", plan.strategy
            );

            // Load all vectors from segments + memtable (needed for all strategies).
            let vectors = match collection.load_all_vectors() {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        collection = %name,
                        "failed to load vectors, skipping: {e}"
                    );
                    continue;
                }
            };
            let vector_count = vectors.len();

            // Populate the in-memory metadata store (needed for all strategies).
            let store = InMemoryVectorStore::new(dimension);
            for (id, _data, metadata) in &vectors {
                if let Err(e) = store.insert_metadata(*id, metadata.clone()) {
                    tracing::warn!(
                        collection = %name,
                        vector_id = id,
                        "failed to insert metadata into store: {e}"
                    );
                }
            }

            // Build VectorArena from loaded vectors using the topology snapshot's slot mapping.
            // Vectors must be pushed in slot order so that each vector ends up at its
            // original slot — otherwise the topology's vector_slot references break.
            let build_arena_from_topology = |vecs: &[(VectorId, Vec<f32>, Option<Metadata>)],
                                              snapshot: &vf_index::hnsw_persistence::HnswTopologySnapshot|
                                              -> VectorArena {
                let mut arena = VectorArena::new(dimension);
                // Build id→data map for O(1) lookup.
                let vec_map: HashMap<VectorId, &[f32]> = vecs.iter()
                    .map(|(id, data, _)| (*id, data.as_slice()))
                    .collect();
                // Sort nodes by vector_slot ascending so arena.push() assigns correct slots.
                let mut nodes_by_slot: Vec<_> = snapshot.nodes.iter()
                    .map(|n| (n.vector_slot, n.id))
                    .collect();
                nodes_by_slot.sort_by_key(|(slot, _)| *slot);
                // Push vectors in slot order.
                for (_slot, id) in &nodes_by_slot {
                    if let Some(data) = vec_map.get(id) {
                        arena.push(data);
                    } else {
                        // Vector not found in segments — push zeros as placeholder.
                        let zeros = vec![0.0f32; dimension];
                        arena.push(&zeros);
                    }
                }
                arena
            };

            // Attempt recovery based on strategy, falling back to full rebuild on error.
            let (index, graph) = match plan.strategy {
                RecoveryStrategy::CleanShutdown => {
                    let hnsw_path = collection_dir.join("hnsw.base");
                    let graph_path = collection_dir.join("graph.base");

                    // Try loading HNSW topology snapshot.
                    let hnsw_result = (|| -> Result<Box<dyn PersistableIndex>, String> {
                        let snapshot = deserialize_topology_mmap(&hnsw_path)
                            .map_err(|e| format!("hnsw base load failed: {e}"))?;
                        let arena = build_arena_from_topology(&vectors, &snapshot);
                        let hnsw_idx = HnswIndex::restore_from_topology(snapshot, arena)
                            .map_err(|e| format!("hnsw restore failed: {e}"))?;

                        // Wrap in QuantizedHnswIndex if quantization is configured.
                        match &config.quantization_config {
                            Some(QuantizationConfig::Scalar(sq_config)) => {
                                let q_index = QuantizedHnswIndex::from_existing_hnsw(
                                    hnsw_idx,
                                    distance_metric,
                                    sq_config.clone(),
                                );
                                q_index.set_data_dir(collection_dir.clone());
                                q_index.train_quantizer(&collection_dir);
                                Ok(Box::new(q_index))
                            }
                            None => Ok(Box::new(hnsw_idx)),
                        }
                    })();

                    // Try loading graph base snapshot.
                    let graph_result = (|| -> Result<VirtualGraph, String> {
                        let mut file = std::fs::File::open(&graph_path)
                            .map_err(|e| format!("graph base open failed: {e}"))?;
                        let (_lsn, graph) = deserialize_graph_base(&mut file)
                            .map_err(|e| format!("graph base load failed: {e}"))?;
                        Ok(graph)
                    })();

                    match (hnsw_result, graph_result) {
                        (Ok(idx), Ok(g)) => {
                            vf_storage::collection::remove_shutdown_marker(&collection_dir);
                            tracing::info!(
                                collection = %name,
                                vectors = vector_count,
                                "recovered from clean shutdown"
                            );
                            (idx, g)
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            tracing::warn!(
                                collection = %name,
                                "snapshot load failed ({e}), falling back to full rebuild"
                            );
                            Self::full_rebuild(&name, dimension, distance_metric, &vectors, &config, &collection_dir)
                        }
                    }
                }

                RecoveryStrategy::IncrementalReplay { .. } => {
                    let hnsw_path = collection_dir.join("hnsw.base");
                    let graph_path = collection_dir.join("graph.base");
                    let hnsw_delta_path = collection_dir.join("hnsw.delta");
                    let graph_delta_path = collection_dir.join("graph.delta");

                    let replay_result = (|| -> Result<(Box<dyn PersistableIndex>, VirtualGraph), String> {
                        // Load HNSW base and replay delta.
                        let mut snapshot = deserialize_topology_mmap(&hnsw_path)
                            .map_err(|e| format!("hnsw base load failed: {e}"))?;
                        let base_lsn = snapshot.snapshot_lsn;

                        if hnsw_delta_path.exists() {
                            let replayed = vf_index::hnsw_delta::replay_delta_after_lsn(
                                &mut snapshot, &hnsw_delta_path, base_lsn,
                            ).map_err(|e| format!("hnsw delta replay failed: {e}"))?;
                            tracing::info!(
                                collection = %name,
                                "replayed hnsw delta, last LSN: {replayed}"
                            );
                        }

                        let arena = build_arena_from_topology(&vectors, &snapshot);
                        let hnsw_idx = HnswIndex::restore_from_topology(snapshot, arena)
                            .map_err(|e| format!("hnsw restore failed: {e}"))?;

                        // Wrap in QuantizedHnswIndex if quantization is configured.
                        let idx: Box<dyn PersistableIndex> = match &config.quantization_config {
                            Some(QuantizationConfig::Scalar(sq_config)) => {
                                let q_index = QuantizedHnswIndex::from_existing_hnsw(
                                    hnsw_idx,
                                    distance_metric,
                                    sq_config.clone(),
                                );
                                q_index.set_data_dir(collection_dir.clone());
                                q_index.train_quantizer(&collection_dir);
                                Box::new(q_index)
                            }
                            None => Box::new(hnsw_idx),
                        };

                        // Load graph base and replay delta.
                        let mut file = std::fs::File::open(&graph_path)
                            .map_err(|e| format!("graph base open failed: {e}"))?;
                        let (graph_base_lsn, mut graph) = deserialize_graph_base(&mut file)
                            .map_err(|e| format!("graph base load failed: {e}"))?;

                        if graph_delta_path.exists() {
                            let replayed = vf_graph::graph_delta::replay_delta_after_lsn(
                                &mut graph, &graph_delta_path, graph_base_lsn,
                            ).map_err(|e| format!("graph delta replay failed: {e}"))?;
                            tracing::info!(
                                collection = %name,
                                "replayed graph delta, entries: {replayed}"
                            );
                        }

                        Ok((idx, graph))
                    })();

                    match replay_result {
                        Ok((idx, g)) => {
                            tracing::info!(
                                collection = %name,
                                vectors = vector_count,
                                "recovered via incremental replay"
                            );
                            (idx, g)
                        }
                        Err(e) => {
                            tracing::warn!(
                                collection = %name,
                                "incremental replay failed ({e}), falling back to full rebuild"
                            );
                            Self::full_rebuild(&name, dimension, distance_metric, &vectors, &config, &collection_dir)
                        }
                    }
                }

                RecoveryStrategy::FullRebuild => {
                    tracing::info!(collection = %name, "full rebuild from vectors");
                    Self::full_rebuild(&name, dimension, distance_metric, &vectors, &config, &collection_dir)
                }
            };

            // Initialize delta writers for incremental persistence.
            let hnsw_delta_path = collection_dir.join("hnsw.delta");
            match HnswDeltaWriter::create(&hnsw_delta_path) {
                Ok(writer) => index.set_delta_writer(writer),
                Err(e) => tracing::warn!(collection = %name, "failed to create hnsw delta writer: {e}"),
            }

            let graph_delta_path = collection_dir.join("graph.delta");
            let mut graph = graph;
            match GraphDeltaWriter::create(&graph_delta_path) {
                Ok(writer) => graph.set_delta_writer(writer),
                Err(e) => tracing::warn!(collection = %name, "failed to create graph delta writer: {e}"),
            }

            let index_manager = IndexManager::with_defaults();

            let collection_state = CollectionState {
                config,
                store,
                index,
                index_manager,
                graph,
                metadata_cache: MetadataCache::new(),
                status: Arc::new(std::sync::RwLock::new(CollectionStatus::Ready)),
                deferred_index: Arc::new(AtomicBool::new(false)),
                deferred_graph: Arc::new(AtomicBool::new(false)),
                deferred_metadata: Arc::new(AtomicBool::new(false)),
                dirty: Arc::new(AtomicBool::new(false)),
                mutation_count: Arc::new(AtomicU64::new(0)),
            };

            collections.insert(name.clone(), collection_state);

            tracing::info!(
                collection = %name,
                vectors = vector_count,
                "recovered collection with {} vectors",
                vector_count
            );
        }

        let recovered_count = collections.len();
        if recovered_count > 0 {
            tracing::info!(
                "recovered {} collection(s) from storage",
                recovered_count
            );
        }

        Ok(Self {
            collections: Arc::new(RwLock::new(collections)),
            collection_manager: Arc::new(RwLock::new(collection_manager)),
            config,
            max_ef_search,
            max_batch_lock_size,
            max_wal_flush_interval,
            max_ef_construction,
        })
    }

    /// Full rebuild path: load vectors into HNSW index one by one and recompute graph.
    fn full_rebuild(
        name: &str,
        dimension: usize,
        distance_metric: vf_core::types::DistanceMetricType,
        vectors: &[(VectorId, Vec<f32>, Option<Metadata>)],
        config: &CollectionConfig,
        data_dir: &Path,
    ) -> (Box<dyn PersistableIndex>, VirtualGraph) {
        let hnsw_index = HnswIndex::with_defaults(dimension, distance_metric);
        for (id, data, _metadata) in vectors {
            if let Err(e) = hnsw_index.add(*id, data) {
                tracing::warn!(
                    collection = %name,
                    vector_id = id,
                    "failed to add vector to HNSW index: {e}"
                );
            }
        }

        let mut graph = match config.default_similarity_threshold {
            Some(t) if t > 0.0 => VirtualGraph::with_threshold(t, config.distance_metric),
            _ => VirtualGraph::with_threshold(0.7, config.distance_metric),
        };

        let vector_ids: Vec<u64> = vectors.iter().map(|(id, _, _)| *id).collect();
        let vector_map: HashMap<u64, Vec<f32>> = vectors.iter()
            .map(|(id, data, _)| (*id, data.clone()))
            .collect();

        // Wrap in QuantizedHnswIndex if quantization is configured.
        let index: Box<dyn PersistableIndex> = match &config.quantization_config {
            Some(QuantizationConfig::Scalar(sq_config)) => {
                let q_index = QuantizedHnswIndex::from_existing_hnsw(
                    hnsw_index,
                    distance_metric,
                    sq_config.clone(),
                );
                q_index.set_data_dir(data_dir.to_path_buf());
                q_index.train_quantizer(data_dir);
                Box::new(q_index)
            }
            None => Box::new(hnsw_index),
        };

        if let Err(e) = vf_graph::RelationshipComputer::compute_batch(
            &mut graph, &*index, &vector_ids, &vector_map, 10,
        ) {
            tracing::warn!(collection = %name, "graph compute_batch failed: {}", e);
        }

        (index, graph)
    }

    /// Prune old WAL files for a collection.
    ///
    /// Deletes all `wal_*.log.old` files in the collection directory.
    /// Returns `(files_deleted, bytes_freed)`.
    pub fn prune_wal_for_collection(&self, name: &str) -> Result<(usize, u64), String> {
        let collection_dir = {
            let cm = self.collection_manager.read();
            let coll = cm.get_collection(name)
                .map_err(|e| format!("Collection '{}' not found: {}", name, e))?;
            coll.collection_dir().to_path_buf()
        };

        let mut deleted = 0usize;
        let mut bytes_freed = 0u64;
        if let Ok(entries) = std::fs::read_dir(&collection_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                    if fname.starts_with("wal_") && fname.ends_with(".log.old") {
                        if let Ok(meta) = std::fs::metadata(&path) {
                            bytes_freed += meta.len();
                        }
                        if std::fs::remove_file(&path).is_ok() {
                            deleted += 1;
                        }
                    }
                }
            }
        }
        Ok((deleted, bytes_freed))
    }

    /// Compact segments for a collection into a single segment.
    ///
    /// Returns compaction statistics or an error if there are too few segments.
    pub fn compact_collection(&self, name: &str, min_segments: usize, remove_deleted: bool) -> Result<vf_storage::CompactionResult, String> {
        let mut cm = self.collection_manager.write();
        let coll = cm.get_collection_mut(name)
            .map_err(|e| format!("Collection '{}' not found: {}", name, e))?;

        let segment_count = coll.segment_count();
        let effective_min = if min_segments == 0 { 4 } else { min_segments };

        if segment_count < effective_min {
            return Err(format!("Only {} segments, need at least {} to compact", segment_count, effective_min));
        }

        let options = vf_storage::CompactionOptions {
            min_segments_to_compact: effective_min,
            remove_deleted,
        };

        coll.compact(options).map_err(|e| format!("Compaction failed: {}", e))
    }

    /// Rebuild deferred operations for a collection after bulk insert.
    ///
    /// Checks which operations were deferred (index, metadata, graph) and
    /// rebuilds each one. Returns statistics about the optimization.
    pub fn optimize_collection(&self, collection_name: &str, rebuild_graph: bool) -> Result<OptimizeResult, String> {
        let start = Instant::now();

        // Check collection exists and get deferred flags.
        let (need_index, need_metadata, need_graph) = {
            let collections = self.collections.read();
            let coll = collections.get(collection_name)
                .ok_or_else(|| format!("collection '{}' not found", collection_name))?;

            let need_index = coll.deferred_index.load(Ordering::Acquire);
            let need_metadata = coll.deferred_metadata.load(Ordering::Acquire);
            let need_graph = coll.deferred_graph.load(Ordering::Acquire);
            let has_quantization = coll.config.quantization_config.is_some();

            // Nothing to do if no deferred ops, or only graph is deferred but rebuild_graph is false.
            // Exception: quantized collections always need optimization to train the quantizer.
            let effective_graph = need_graph && rebuild_graph;
            if !need_index && !need_metadata && !effective_graph && !has_quantization {
                return Ok(OptimizeResult {
                    status: "already_optimized".to_string(),
                    message: "nothing to optimize".to_string(),
                    duration_ms: 0,
                    vectors_processed: 0,
                });
            }

            // Check not already optimizing.
            {
                let status = coll.status.read().unwrap();
                if *status == CollectionStatus::Optimizing {
                    return Err(format!(
                        "collection '{}' is already being optimized",
                        collection_name
                    ));
                }
            }

            // Set status to optimizing.
            {
                let mut status = coll.status.write().unwrap();
                *status = CollectionStatus::Optimizing;
            }

            (need_index, need_metadata, need_graph)
        };

        // Perform the rebuild under a write lock.
        let result = (|| -> Result<u64, String> {
            let mut collections = self.collections.write();
            let coll = collections.get_mut(collection_name)
                .ok_or_else(|| format!("collection '{}' not found", collection_name))?;

            let vectors_processed = coll.store.len() as u64;

            // 1. Rebuild HNSW index if deferred.
            if need_index {
                let vector_data = coll.index.iter_vectors_owned();
                let refs: Vec<(VectorId, &[f32])> = vector_data
                    .iter()
                    .map(|(id, v)| (*id, v.as_slice()))
                    .collect();

                // Create a fresh HNSW index and rebuild via build_parallel.
                let new_hnsw = HnswIndex::with_defaults(
                    coll.config.dimension,
                    coll.config.distance_metric,
                );
                new_hnsw.build_parallel(&refs).map_err(|e| {
                    format!("HNSW index rebuild failed: {}", e)
                })?;
                new_hnsw.compact();

                // Wrap in QuantizedHnswIndex if quantization is configured.
                let new_index: Box<dyn PersistableIndex> = match &coll.config.quantization_config {
                    Some(QuantizationConfig::Scalar(sq_config)) => {
                        let q_index = QuantizedHnswIndex::from_existing_hnsw(
                            new_hnsw,
                            coll.config.distance_metric,
                            sq_config.clone(),
                        );
                        // Get the real collection directory for mmap files.
                        let data_dir = {
                            let cm = self.collection_manager.read();
                            cm.get_collection(collection_name)
                                .map(|c| c.collection_dir().to_path_buf())
                                .unwrap_or_else(|_| {
                                    std::env::temp_dir()
                                        .join(format!("swarndb_optimize_{}", collection_name))
                                })
                        };
                        let _ = std::fs::create_dir_all(&data_dir);
                        q_index.set_data_dir(data_dir.clone());
                        q_index.train_quantizer(&data_dir);
                        Box::new(q_index)
                    }
                    None => Box::new(new_hnsw),
                };
                coll.index = new_index;
                coll.deferred_index.store(false, Ordering::Release);
                tracing::info!(
                    collection = %collection_name,
                    vectors = vectors_processed,
                    "rebuilt HNSW index"
                );
            }

            // 2. Rebuild metadata indexes if skipped.
            if need_metadata {
                let mut new_manager = IndexManager::with_defaults();
                let metadata_pairs = coll.store.iter_metadata();
                for (id, meta) in &metadata_pairs {
                    new_manager.index_record(*id, meta);
                }
                coll.index_manager = new_manager;
                coll.deferred_metadata.store(false, Ordering::Release);
                tracing::info!(
                    collection = %collection_name,
                    "rebuilt metadata indexes"
                );
            }

            // 2.5. Train quantizer if quantized collection hasn't been trained yet.
            if !need_index {
                if let Some(QuantizationConfig::Scalar(_)) = &coll.config.quantization_config {
                    coll.index.post_optimize();
                    tracing::info!(
                        collection = %collection_name,
                        "trained scalar quantizer"
                    );
                }
            }

            // 3. Recompute virtual graph edges if deferred and rebuild_graph is requested.
            if need_graph && rebuild_graph {
                let vector_data = coll.index.iter_vectors_owned();
                let vector_ids: Vec<u64> = vector_data.iter().map(|(id, _)| *id).collect();
                let vector_map: HashMap<u64, Vec<f32>> = vector_data.into_iter().collect();

                // Reset graph and rebuild.
                let threshold = coll.config.default_similarity_threshold.unwrap_or(0.7);
                let mut new_graph = VirtualGraph::with_threshold(
                    if threshold > 0.0 { threshold } else { 0.7 },
                    coll.config.distance_metric,
                );
                if let Err(e) = vf_graph::RelationshipComputer::compute_batch(
                    &mut new_graph, coll.index.as_vector_index(), &vector_ids, &vector_map, 10,
                ) {
                    tracing::warn!(
                        collection = %collection_name,
                        "graph rebuild partially failed: {}",
                        e
                    );
                }
                coll.graph = new_graph;
                coll.deferred_graph.store(false, Ordering::Release);
                coll.dirty.store(true, Ordering::Release);
                coll.mutation_count.store(50_001, Ordering::Release);
                tracing::info!(
                    collection = %collection_name,
                    "rebuilt virtual graph"
                );
            }

            Ok(vectors_processed)
        })();

        // Always reset status back from optimizing (even on error).
        {
            let collections = self.collections.read();
            if let Some(coll) = collections.get(collection_name) {
                let mut status = coll.status.write().unwrap();
                *status = CollectionStatus::Ready;
            }
        }

        // Auto-prune WAL after optimize.
        if result.is_ok() && self.config.wal_prune_after_optimize {
            match self.prune_wal_for_collection(collection_name) {
                Ok((count, bytes)) => {
                    if count > 0 {
                        tracing::info!("Pruned {} WAL files ({} bytes) after optimize", count, bytes);
                    }
                }
                Err(e) => tracing::warn!("WAL prune after optimize failed: {}", e),
            }
        }

        // Auto-compact after optimize.
        if result.is_ok() && self.config.auto_compact_after_optimize {
            match self.compact_collection(collection_name, self.config.compaction_min_segments, true) {
                Ok(result) => {
                    tracing::info!("Auto-compacted {} segments into 1 ({} vectors)", result.segments_merged, result.vectors_written);
                }
                Err(e) => {
                    // Not an error if too few segments.
                    tracing::debug!("Auto-compact skipped: {}", e);
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(vectors_processed) => {
                let mut parts = Vec::new();
                if need_index { parts.push("HNSW index"); }
                if need_metadata { parts.push("metadata indexes"); }
                if need_graph && rebuild_graph { parts.push("virtual graph"); }

                Ok(OptimizeResult {
                    status: "completed".to_string(),
                    message: format!("rebuilt: {}", parts.join(", ")),
                    duration_ms,
                    vectors_processed,
                })
            }
            Err(e) => Err(e),
        }
    }
}
