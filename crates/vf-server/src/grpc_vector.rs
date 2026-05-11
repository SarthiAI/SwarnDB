// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! gRPC VectorService handler implementation.

use tonic::{Request, Response, Status};

use crate::convert::{core_to_proto_metadata, proto_to_core_metadata};
use crate::proto::swarndb::v1::vector_service_server::VectorService;
use std::sync::atomic::Ordering;

use crate::proto::swarndb::v1::{
    BulkInsertResponse, BulkInsertStreamMessage, CompactRequest, CompactResponse,
    DeleteVectorRequest, DeleteVectorResponse, GetVectorRequest, GetVectorResponse,
    InsertRequest, InsertResponse, OptimizeRequest, OptimizeResponse, PruneWalRequest,
    PruneWalResponse, UpdateRequest, UpdateResponse, Vector,
    bulk_insert_stream_message::Payload,
};
use crate::state::{AppState, CollectionStatus};
use crate::validation::{
    validate_batch_lock_size, validate_bulk_insert_options, validate_ef_construction,
    validate_index_mode, validate_wal_flush_every,
};
use vf_core::vector::VectorData;

pub struct VectorServiceImpl {
    state: AppState,
    max_batch_lock_size: u32,
    max_wal_flush_interval: u32,
    max_ef_construction: u32,
}

impl VectorServiceImpl {
    pub fn new(
        state: AppState,
        max_batch_lock_size: u32,
        max_wal_flush_interval: u32,
        max_ef_construction: u32,
    ) -> Self {
        Self {
            state,
            max_batch_lock_size,
            max_wal_flush_interval,
            max_ef_construction,
        }
    }
}

#[tonic::async_trait]
impl VectorService for VectorServiceImpl {
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();

        let proto_vec = req
            .vector
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("vector field is required"))?;
        if proto_vec.values.is_empty() {
            return Err(Status::invalid_argument("vector values must not be empty"));
        }

        let values = proto_vec.values.clone();
        let core_metadata = req.metadata.as_ref().map(proto_to_core_metadata);

        let mut collections = self.state.collections.write();
        let coll = collections.get_mut(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;

        let assigned_id = if req.id == 0 {
            coll.store
                .insert_metadata_auto_id(core_metadata.clone())
                .map_err(|e| Status::internal(format!("store insert failed: {}", e)))?
        } else {
            coll.store
                .insert_metadata(req.id, core_metadata.clone())
                .map_err(|e| Status::internal(format!("store insert failed: {}", e)))?;
            req.id
        };

        // Persist to storage layer first to obtain the WAL LSN.
        let lsn = {
            let mut cm = self.state.collection_manager.write();
            if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                if let Err(e) = storage_coll.insert(assigned_id, VectorData::F32(values.clone()), core_metadata.clone()) {
                    tracing::warn!(collection = %req.collection, id = assigned_id, "storage insert failed: {}", e);
                }
                storage_coll.current_lsn().saturating_sub(1)
            } else {
                0
            }
        };

        if let Err(e) = coll.index.add_with_lsn(assigned_id, &values, lsn) {
            let _ = coll.store.delete(assigned_id);
            return Err(Status::internal(format!("index insert failed: {}", e)));
        }

        if let Some(ref meta) = core_metadata {
            coll.index_manager.index_record(assigned_id, meta);
        }

        if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
            &mut coll.graph, coll.index.as_vector_index(), assigned_id, &values, 10,
        ) {
            tracing::warn!(collection = %req.collection, id = assigned_id, "graph compute failed: {}", e);
        }

        coll.dirty.store(true, Ordering::Release);
        coll.mutation_count.fetch_add(1, Ordering::Relaxed);

        Ok(Response::new(InsertResponse {
            id: assigned_id,
            success: true,
        }))
    }

    async fn get(
        &self,
        request: Request<GetVectorRequest>,
    ) -> Result<Response<GetVectorResponse>, Status> {
        let req = request.into_inner();

        let collections = self.state.collections.read();
        let coll = collections.get(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;

        let meta_record = coll
            .store
            .get(req.id)
            .map_err(|e| Status::not_found(format!("vector not found: {}", e)))?;
        let vector_data = coll
            .index
            .get_vector(req.id)
            .map_err(|e| Status::internal(format!("vector retrieval failed: {}", e)))?;

        let proto_vector = Vector {
            values: vector_data,
        };

        let proto_metadata = meta_record.metadata.as_ref().map(core_to_proto_metadata);

        Ok(Response::new(GetVectorResponse {
            id: meta_record.id,
            vector: Some(proto_vector),
            metadata: proto_metadata,
        }))
    }

    async fn update(
        &self,
        request: Request<UpdateRequest>,
    ) -> Result<Response<UpdateResponse>, Status> {
        let req = request.into_inner();

        // Vector is now optional -- extract values if present
        let values = match req.vector.as_ref() {
            Some(proto_vec) => {
                if proto_vec.values.is_empty() {
                    return Err(Status::invalid_argument("vector values must not be empty"));
                }
                Some(proto_vec.values.clone())
            }
            None => None,
        };

        if values.is_none() && req.metadata.is_none() {
            return Err(Status::invalid_argument("at least one of 'vector' or 'metadata' must be provided"));
        }

        let core_metadata = req.metadata.as_ref().map(proto_to_core_metadata);

        let mut collections = self.state.collections.write();
        let coll = collections.get_mut(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;

        // Persist to storage layer first to obtain the WAL LSN.
        let lsn = {
            let mut cm = self.state.collection_manager.write();
            if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                let storage_data = values.clone().map(VectorData::F32);
                if let Err(e) = storage_coll.update(req.id, storage_data, core_metadata.clone()) {
                    tracing::warn!(collection = %req.collection, id = req.id, "storage update failed: {}", e);
                }
                storage_coll.current_lsn().saturating_sub(1)
            } else {
                0
            }
        };

        // Update vector index with LSN if new vector data was provided.
        if let Some(ref vals) = values {
            let _ = coll.index.remove_with_lsn(req.id, lsn);
            coll.index
                .add_with_lsn(req.id, vals, lsn)
                .map_err(|e| Status::internal(format!("index update failed: {}", e)))?;
        }

        coll.store
            .update_metadata(req.id, core_metadata.clone())
            .map_err(|e| Status::not_found(format!("update failed: {}", e)))?;

        if let Some(ref meta) = core_metadata {
            coll.index_manager.remove_record(req.id);
            coll.index_manager.index_record(req.id, meta);
        }

        coll.dirty.store(true, Ordering::Release);
        coll.mutation_count.fetch_add(1, Ordering::Relaxed);

        Ok(Response::new(UpdateResponse { success: true }))
    }

    async fn delete(
        &self,
        request: Request<DeleteVectorRequest>,
    ) -> Result<Response<DeleteVectorResponse>, Status> {
        let req = request.into_inner();

        let mut collections = self.state.collections.write();
        let coll = collections.get_mut(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;

        coll.store
            .delete(req.id)
            .map_err(|e| Status::not_found(format!("delete failed: {}", e)))?;

        // Persist to storage layer first to obtain the WAL LSN.
        let lsn = {
            let mut cm = self.state.collection_manager.write();
            if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                if let Err(e) = storage_coll.delete(req.id) {
                    tracing::warn!(collection = %req.collection, id = req.id, "storage delete failed: {}", e);
                }
                storage_coll.current_lsn().saturating_sub(1)
            } else {
                0
            }
        };

        let _ = coll.index.remove_with_lsn(req.id, lsn);
        coll.index_manager.remove_record(req.id);
        coll.graph.remove_node_with_lsn(req.id, lsn);

        coll.dirty.store(true, Ordering::Release);
        coll.mutation_count.fetch_add(1, Ordering::Relaxed);

        Ok(Response::new(DeleteVectorResponse { success: true }))
    }

    async fn bulk_insert(
        &self,
        request: Request<tonic::Streaming<InsertRequest>>,
    ) -> Result<Response<BulkInsertResponse>, Status> {
        let mut stream = request.into_inner();
        let mut inserted_count: u64 = 0;
        let mut item_index: u64 = 0;
        let mut errors: Vec<String> = Vec::new();
        let mut batch_vectors: std::collections::HashMap<String, (Vec<u64>, std::collections::HashMap<u64, Vec<f32>>)> = std::collections::HashMap::new();

        while let Some(result) = stream.message().await? {
            let req = result;
            let current_item = item_index;
            item_index += 1;

            let proto_vec = match req.vector.as_ref() {
                Some(v) if !v.values.is_empty() => v,
                _ => {
                    errors.push(format!("item {}: missing or empty vector", current_item));
                    continue;
                }
            };

            let values = proto_vec.values.clone();
            let core_metadata = req.metadata.as_ref().map(proto_to_core_metadata);

            let mut collections = self.state.collections.write();
            let coll = match collections.get_mut(&req.collection) {
                Some(c) => c,
                None => {
                    errors.push(format!("collection '{}' not found", req.collection));
                    continue;
                }
            };

            let assigned_id = if req.id == 0 {
                match coll.store.insert_metadata_auto_id(core_metadata.clone()) {
                    Ok(id) => id,
                    Err(e) => {
                        errors.push(format!("store insert failed: {}", e));
                        continue;
                    }
                }
            } else {
                match coll.store.insert_metadata(req.id, core_metadata.clone()) {
                    Ok(()) => req.id,
                    Err(e) => {
                        errors.push(format!("store insert failed for id {}: {}", req.id, e));
                        continue;
                    }
                }
            };

            // Save values for batch graph recomputation before they're moved
            let values_for_graph = values.clone();

            // Persist to storage layer first to obtain the WAL LSN.
            let lsn = {
                let mut cm = self.state.collection_manager.write();
                if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                    if let Err(e) = storage_coll.insert(
                        assigned_id,
                        VectorData::F32(values.clone()),
                        core_metadata.clone(),
                    ) {
                        tracing::warn!(collection = %req.collection, id = assigned_id, "storage bulk insert failed: {}", e);
                    }
                    storage_coll.current_lsn().saturating_sub(1)
                } else {
                    0
                }
            };

            // add_with_lsn records the entry in the hnsw.delta writer so a
            // crash before snapshot can be recovered incrementally on restart.
            if let Err(e) = coll.index.add_with_lsn(assigned_id, &values, lsn) {
                // Rollback: remove from store since index add failed
                let _ = coll.store.delete(assigned_id);
                errors.push(format!("item {}: index insert failed for id {}: {}", current_item, assigned_id, e));
                continue;
            }

            if let Some(ref meta) = core_metadata {
                coll.index_manager.index_record(assigned_id, meta);
            }

            // Compute virtual graph edges for the newly inserted vector
            if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
                &mut coll.graph, coll.index.as_vector_index(), assigned_id, &values, 10,
            ) {
                tracing::warn!(collection = %req.collection, id = assigned_id, "graph compute failed: {}", e);
            }

            let entry = batch_vectors.entry(req.collection.clone()).or_insert_with(|| (Vec::new(), std::collections::HashMap::new()));
            entry.0.push(assigned_id);
            entry.1.insert(assigned_id, values_for_graph);
            inserted_count += 1;
        }

        // Recompute graph edges for all inserted vectors now that the full index is available
        for (coll_name, (ids, vectors_map)) in &batch_vectors {
            let mut collections = self.state.collections.write();
            if let Some(coll) = collections.get_mut(coll_name) {
                if let Err(e) = vf_graph::RelationshipComputer::compute_batch(
                    &mut coll.graph, coll.index.as_vector_index(), ids, vectors_map, 10,
                ) {
                    tracing::warn!(collection = %coll_name, "graph compute_batch after bulk_insert failed: {}", e);
                }
            }
        }

        Ok(Response::new(BulkInsertResponse {
            inserted_count,
            errors,
        }))
    }

    async fn bulk_insert_with_options(
        &self,
        request: Request<tonic::Streaming<BulkInsertStreamMessage>>,
    ) -> Result<Response<BulkInsertResponse>, Status> {
        let mut stream = request.into_inner();

        // First message must be BulkInsertOptions
        let first_msg = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("stream is empty, expected options message first"))?;

        let options = match first_msg.payload {
            Some(Payload::Options(opts)) => opts,
            Some(Payload::Vector(_)) => {
                return Err(Status::invalid_argument(
                    "first message must be BulkInsertOptions, got vector",
                ));
            }
            None => {
                return Err(Status::invalid_argument(
                    "first message must contain BulkInsertOptions payload",
                ));
            }
        };

        // Validate and parse options
        let index_mode = if options.index_mode.is_empty() {
            "immediate"
        } else {
            &options.index_mode
        };
        validate_index_mode(index_mode)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let raw_batch_lock_size = if options.batch_lock_size == 0 { 1 } else { options.batch_lock_size };
        validate_batch_lock_size(raw_batch_lock_size, self.max_batch_lock_size)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        validate_wal_flush_every(options.wal_flush_every, self.max_wal_flush_interval)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        if options.ef_construction > 0 {
            validate_ef_construction(options.ef_construction, self.max_ef_construction)
                .map_err(|e| Status::invalid_argument(e.to_string()))?;
            tracing::info!(ef_construction = options.ef_construction, "ef_construction override for bulk insert");
        }

        validate_bulk_insert_options(options.parallel_build, index_mode)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        if options.defer_graph {
            tracing::warn!("defer_graph enabled: search may return stale results until optimize() is called");
        }

        let batch_lock_size = raw_batch_lock_size as usize;
        let defer_graph = options.defer_graph;
        let wal_flush_every = options.wal_flush_every;
        let _ef_construction = options.ef_construction; // TODO: thread to HNSW index
        let index_mode_deferred = index_mode == "deferred";
        let skip_metadata_index = options.skip_metadata_index;
        let _parallel_build = options.parallel_build; // TODO: use in optimize() with index_mode=deferred

        let mut inserted_count: u64 = 0;
        let mut item_index: u64 = 0;
        let mut errors: Vec<String> = Vec::new();

        // Buffer for batched lock acquisition
        struct BatchItem {
            req: InsertRequest,
            item_idx: u64,
        }
        let mut batch_buffer: Vec<BatchItem> = Vec::with_capacity(batch_lock_size);

        // Track collection names and inserted vectors for deferred graph recomputation
        let mut batch_vectors: std::collections::HashMap<
            String,
            (Vec<u64>, std::collections::HashMap<u64, Vec<f32>>),
        > = std::collections::HashMap::new();

        // Track which collections had any deferred flags set
        let mut collections_with_deferrals: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Helper closure to process a batch of items. Per item we now write
        // the WAL first to capture the LSN, then call index.add_with_lsn so
        // the hnsw.delta writer records the insert for incremental replay.
        // wal_flush_every retains its semantics: 0 disables the WAL entirely
        // (and the resulting index entry is recorded with lsn=0, matching the
        // single-insert fallback when storage is unavailable). Non-zero values
        // no longer batch (each insert writes inline) but the option is still
        // accepted for API compatibility.
        let process_batch = |batch: &mut Vec<BatchItem>,
                             state: &AppState,
                             errors: &mut Vec<String>,
                             inserted_count: &mut u64,
                             wal_flush_every: u32,
                             defer_graph: bool,
                             index_mode_deferred: bool,
                             skip_metadata_index: bool,
                             batch_vectors: &mut std::collections::HashMap<
            String,
            (Vec<u64>, std::collections::HashMap<u64, Vec<f32>>),
        >,
        | {
            if batch.is_empty() {
                return;
            }

            // Group by collection name to minimize lock re-acquisitions
            let mut by_collection: std::collections::HashMap<String, Vec<&mut BatchItem>> =
                std::collections::HashMap::new();
            for item in batch.iter_mut() {
                by_collection
                    .entry(item.req.collection.clone())
                    .or_default()
                    .push(item);
            }

            // Process each collection group under a single write lock
            for (coll_name, items) in &by_collection {
                let mut collections = state.collections.write();
                let coll = match collections.get_mut(coll_name) {
                    Some(c) => c,
                    None => {
                        for item in items {
                            errors.push(format!(
                                "item {}: collection '{}' not found",
                                item.item_idx, coll_name
                            ));
                        }
                        continue;
                    }
                };

                for item in items {
                    let proto_vec = match item.req.vector.as_ref() {
                        Some(v) if !v.values.is_empty() => v,
                        _ => {
                            errors.push(format!(
                                "item {}: missing or empty vector",
                                item.item_idx
                            ));
                            continue;
                        }
                    };

                    let values = proto_vec.values.clone();
                    let core_metadata = item.req.metadata.as_ref().map(proto_to_core_metadata);

                    // Insert metadata into store (vector data stored in index)
                    let assigned_id = if item.req.id == 0 {
                        match coll.store.insert_metadata_auto_id(core_metadata.clone()) {
                            Ok(id) => id,
                            Err(e) => {
                                errors.push(format!(
                                    "item {}: store insert failed: {}",
                                    item.item_idx, e
                                ));
                                continue;
                            }
                        }
                    } else {
                        match coll.store.insert_metadata(item.req.id, core_metadata.clone()) {
                            Ok(()) => item.req.id,
                            Err(e) => {
                                errors.push(format!(
                                    "item {}: store insert failed for id {}: {}",
                                    item.item_idx, item.req.id, e
                                ));
                                continue;
                            }
                        }
                    };

                    // Persist to storage layer first to obtain the WAL LSN.
                    // When wal_flush_every == 0 the caller has opted out of
                    // WAL persistence entirely and we fall back to lsn=0.
                    let lsn = if wal_flush_every > 0 {
                        let mut cm = state.collection_manager.write();
                        if let Ok(storage_coll) = cm.get_collection_mut(coll_name) {
                            if let Err(e) = storage_coll.insert(
                                assigned_id,
                                VectorData::F32(values.clone()),
                                core_metadata.clone(),
                            ) {
                                tracing::warn!(
                                    collection = %coll_name,
                                    id = assigned_id,
                                    "storage insert failed: {}",
                                    e
                                );
                            }
                            storage_coll.current_lsn().saturating_sub(1)
                        } else {
                            0
                        }
                    } else {
                        0
                    };

                    // HNSW index: skip if deferred. add_with_lsn records the
                    // entry in the hnsw.delta writer for incremental replay.
                    if !index_mode_deferred {
                        if let Err(e) = coll.index.add_with_lsn(assigned_id, &values, lsn) {
                            let _ = coll.store.delete(assigned_id);
                            errors.push(format!(
                                "item {}: index insert failed for id {}: {}",
                                item.item_idx, assigned_id, e
                            ));
                            continue;
                        }
                    }

                    // Metadata index: skip if flagged
                    if !skip_metadata_index {
                        if let Some(ref meta) = core_metadata {
                            coll.index_manager.index_record(assigned_id, meta);
                        }
                    }

                    // Graph computation: skip if deferred
                    if !defer_graph {
                        if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
                            &mut coll.graph,
                            coll.index.as_vector_index(),
                            assigned_id,
                            &values,
                            10,
                        ) {
                            tracing::warn!(
                                collection = %coll_name,
                                id = assigned_id,
                                "graph compute failed: {}",
                                e
                            );
                        }
                    }

                    // Track for post-insert graph recomputation
                    let entry = batch_vectors
                        .entry(coll_name.clone())
                        .or_insert_with(|| (Vec::new(), std::collections::HashMap::new()));
                    entry.0.push(assigned_id);
                    entry.1.insert(assigned_id, values.clone());

                    *inserted_count += 1;
                }
            }

            batch.clear();
        };

        // Stream vectors and collect into batches
        while let Some(result) = stream.message().await? {
            match result.payload {
                Some(Payload::Vector(req)) => {
                    let current_item = item_index;
                    item_index += 1;

                    if !req.collection.is_empty() {
                        collections_with_deferrals.insert(req.collection.clone());
                    }

                    batch_buffer.push(BatchItem {
                        req,
                        item_idx: current_item,
                    });

                    if batch_buffer.len() >= batch_lock_size {
                        process_batch(
                            &mut batch_buffer,
                            &self.state,
                            &mut errors,
                            &mut inserted_count,
                            wal_flush_every,
                            defer_graph,
                            index_mode_deferred,
                            skip_metadata_index,
                            &mut batch_vectors,
                        );
                    }
                }
                Some(Payload::Options(_)) => {
                    errors.push(format!(
                        "item {}: unexpected options message after first message",
                        item_index
                    ));
                    item_index += 1;
                }
                None => {
                    item_index += 1;
                }
            }
        }

        // Reject empty stream (options sent but no vectors followed)
        if item_index == 0 {
            return Err(Status::invalid_argument(
                "stream contained no vectors after options message",
            ));
        }

        // Process any remaining items in the buffer
        process_batch(
            &mut batch_buffer,
            &self.state,
            &mut errors,
            &mut inserted_count,
            wal_flush_every,
            defer_graph,
            index_mode_deferred,
            skip_metadata_index,
            &mut batch_vectors,
        );

        // Recompute graph edges for all inserted vectors (only if graph not deferred)
        if !defer_graph {
            for (coll_name, (ids, vectors_map)) in &batch_vectors {
                let mut collections = self.state.collections.write();
                if let Some(coll) = collections.get_mut(coll_name) {
                    if let Err(e) = vf_graph::RelationshipComputer::compute_batch(
                        &mut coll.graph, coll.index.as_vector_index(), ids, vectors_map, 10,
                    ) {
                        tracing::warn!(
                            collection = %coll_name,
                            "graph compute_batch after bulk_insert_with_options failed: {}",
                            e
                        );
                    }
                }
            }
        }

        // Set deferred flags and collection status
        let any_deferred = defer_graph || index_mode_deferred || skip_metadata_index;
        if any_deferred {
            let collections = self.state.collections.read();
            for coll_name in &collections_with_deferrals {
                if let Some(coll) = collections.get(coll_name) {
                    if defer_graph {
                        coll.deferred_graph.store(true, Ordering::Release);
                    }
                    if index_mode_deferred {
                        coll.deferred_index.store(true, Ordering::Release);
                    }
                    if skip_metadata_index {
                        coll.deferred_metadata.store(true, Ordering::Release);
                    }
                    // Set status to PendingOptimization
                    if let Ok(mut status) = coll.status.write() {
                        *status = CollectionStatus::PendingOptimization;
                    }
                }
            }
        }

        Ok(Response::new(BulkInsertResponse {
            inserted_count,
            errors,
        }))
    }

    async fn optimize(
        &self,
        request: Request<OptimizeRequest>,
    ) -> Result<Response<OptimizeResponse>, Status> {
        let req = request.into_inner();

        if req.collection.is_empty() {
            return Err(Status::invalid_argument("collection name is required"));
        }

        match self.state.optimize_collection(&req.collection, req.rebuild_graph) {
            Ok(result) => Ok(Response::new(OptimizeResponse {
                status: result.status,
                message: result.message,
                duration_ms: result.duration_ms,
                vectors_processed: result.vectors_processed,
            })),
            Err(e) if e.contains("not found") => Err(Status::not_found(e)),
            Err(e) if e.contains("already being optimized") => {
                Err(Status::failed_precondition(e))
            }
            Err(e) => Err(Status::internal(e)),
        }
    }

    async fn prune_wal(
        &self,
        request: Request<PruneWalRequest>,
    ) -> Result<Response<PruneWalResponse>, Status> {
        let req = request.into_inner();
        let start = std::time::Instant::now();

        match self.state.prune_wal_for_collection(&req.collection) {
            Ok((files_deleted, bytes_freed)) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                Ok(Response::new(PruneWalResponse {
                    status: "completed".to_string(),
                    files_deleted: files_deleted as u64,
                    bytes_freed,
                    duration_ms,
                }))
            }
            Err(e) => Err(Status::internal(e)),
        }
    }

    async fn compact(
        &self,
        request: Request<CompactRequest>,
    ) -> Result<Response<CompactResponse>, Status> {
        let req = request.into_inner();
        let start = std::time::Instant::now();

        let min_segments = if req.min_segments == 0 { 4 } else { req.min_segments as usize };
        let remove_deleted = req.remove_deleted;

        match self.state.compact_collection(&req.collection, min_segments, remove_deleted) {
            Ok(result) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                Ok(Response::new(CompactResponse {
                    status: "completed".to_string(),
                    segments_merged: result.segments_merged as u64,
                    vectors_written: result.vectors_written,
                    vectors_removed: result.vectors_removed,
                    duration_ms,
                }))
            }
            Err(e) => Err(Status::internal(e)),
        }
    }
}
