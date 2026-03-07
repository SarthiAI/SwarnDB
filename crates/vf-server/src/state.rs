// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use vf_core::store::{InMemoryVectorStore, VectorRecord};
use vf_core::types::CollectionConfig;
use vf_core::vector::VectorData;
use vf_graph::VirtualGraph;
use vf_index::hnsw::HnswIndex;
use vf_index::traits::VectorIndex;
use vf_query::IndexManager;
use vf_storage::collection::CollectionManager;
use vf_storage::StorageError;

/// Per-collection state holding all components needed for vector operations.
pub struct CollectionState {
    pub config: CollectionConfig,
    pub store: InMemoryVectorStore,
    pub index: HnswIndex,
    pub index_manager: IndexManager,
    pub graph: VirtualGraph,
}

/// Global application state shared across all gRPC services.
#[derive(Clone)]
pub struct AppState {
    pub collections: Arc<RwLock<HashMap<String, CollectionState>>>,
    pub collection_manager: Arc<RwLock<CollectionManager>>,
}

impl AppState {
    /// Create a new AppState with a CollectionManager rooted at `storage_path`.
    ///
    /// The CollectionManager will create the directory if it does not exist and
    /// scan for any previously persisted collections. For each persisted
    /// collection, the in-memory vector store and HNSW index are rebuilt from
    /// the stored vectors so that search is immediately available after restart.
    pub fn new(storage_path: &Path) -> Result<Self, StorageError> {
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

            // Load all vectors from segments + memtable.
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

            // Populate the in-memory store and HNSW index.
            let store = InMemoryVectorStore::new(dimension);
            let index = HnswIndex::with_defaults(dimension, distance_metric);

            for (id, data, metadata) in &vectors {
                let record = VectorRecord::new(
                    *id,
                    VectorData::F32(data.clone()),
                    metadata.clone(),
                );
                if let Err(e) = store.insert(record) {
                    tracing::warn!(
                        collection = %name,
                        vector_id = id,
                        "failed to insert vector into store: {e}"
                    );
                }
                if let Err(e) = index.add(*id, data) {
                    tracing::warn!(
                        collection = %name,
                        vector_id = id,
                        "failed to add vector to HNSW index: {e}"
                    );
                }
            }

            let index_manager = IndexManager::with_defaults();
            let mut graph = match config.default_similarity_threshold {
                Some(t) if t > 0.0 => VirtualGraph::with_threshold(t, config.distance_metric),
                _ => VirtualGraph::with_threshold(0.7, config.distance_metric),
            };

            // Populate virtual graph from recovered vectors
            {
                let vector_ids: Vec<u64> = vectors.iter().map(|(id, _, _)| *id).collect();
                let vector_map: std::collections::HashMap<u64, Vec<f32>> = vectors.iter()
                    .map(|(id, data, _)| (*id, data.clone()))
                    .collect();
                if let Err(e) = vf_graph::RelationshipComputer::compute_batch(
                    &mut graph, &index, &vector_ids, &vector_map, 10,
                ) {
                    tracing::warn!(collection = %name, "graph compute_batch failed: {}", e);
                }
            }

            let collection_state = CollectionState {
                config,
                store,
                index,
                index_manager,
                graph,
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
        })
    }
}
