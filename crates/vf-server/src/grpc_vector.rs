// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! gRPC VectorService handler implementation.

use tonic::{Request, Response, Status};

use crate::bulk_checkpoint_token::{
    encode_resume_token, hash_collection_name, resolve_bulk_checkpoint_path,
};
use crate::convert::{core_to_proto_metadata, proto_to_core_metadata};
use crate::proto::swarndb::v1::vector_service_server::VectorService;
use std::sync::atomic::Ordering;

use crate::proto::swarndb::v1::{
    BulkInsertFromPathRequest, BulkInsertResponse, BulkInsertStreamMessage, CompactRequest,
    CompactResponse, DeleteVectorRequest, DeleteVectorResponse, GetVectorRequest,
    GetVectorResponse, InsertRequest, InsertResponse, OptimizeRequest, OptimizeResponse,
    PruneWalRequest, PruneWalResponse, UpdateRequest, UpdateResponse, Vector,
    bulk_insert_stream_message::Payload,
};
use crate::state::{
    metered_read, metered_write, AppState, CollectionAvailability, CollectionState,
    CollectionStatus,
};
use crate::validation::{
    validate_batch_lock_size, validate_bulk_insert_options, validate_ef_construction,
    validate_index_mode, validate_wal_flush_every,
};
use std::sync::Arc;
use vf_core::vector::VectorData;

/// Convert a `CollectionAvailability` returned by the readiness guard into a
/// `tonic::Status`. Recovering collections become `Unavailable`; missing
/// collections become `NotFound`.
fn status_from_availability(avail: CollectionAvailability) -> Status {
    match avail {
        CollectionAvailability::Recovering { .. } => Status::unavailable(avail.user_message()),
        CollectionAvailability::NotFound { .. } => Status::not_found(avail.user_message()),
    }
}

/// Map a path-validation / mmap error from the bulk_insert_from_path module
/// onto an appropriate tonic Status code per ADR-001 Decision 1 / 5.
fn map_bifp_error(e: crate::bulk_insert_from_path::BulkFromPathError) -> Status {
    use crate::bulk_insert_from_path::BulkFromPathError as E;
    let msg = e.to_string();
    match e {
        E::PathDenied { .. } => Status::permission_denied(msg),
        E::RelativePath { .. }
        | E::TraversalAttempt { .. }
        | E::NullByte { .. }
        | E::BadMagic { .. }
        | E::DimensionMismatch { .. }
        | E::CountMismatch { .. } => Status::invalid_argument(msg),
        E::MmapFailed { .. } | E::Io { .. } => Status::internal(msg),
    }
}

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

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let proto_vec = req
            .vector
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("vector field is required"))?;
        if proto_vec.values.is_empty() {
            return Err(Status::invalid_argument("vector values must not be empty"));
        }

        let values = proto_vec.values.clone();
        let core_metadata = req.metadata.as_ref().map(proto_to_core_metadata);

        let coll_handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&coll_handle);

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

        {
            // Reborrow the lock guard to a plain &mut CollectionState so the
            // compiler can split the disjoint field borrows of graph and index.
            let coll: &mut CollectionState = &mut *coll;
            if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
                &mut coll.graph, coll.index.as_vector_index(), assigned_id, &values, 10,
            ) {
                tracing::warn!(collection = %req.collection, id = assigned_id, "graph compute failed: {}", e);
            }
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

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let coll_handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let coll = metered_read(&coll_handle);

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

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

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

        let coll_handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&coll_handle);

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

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let coll_handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;
        let mut coll = metered_write(&coll_handle);

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
        // Pending errors carry the target collection and optional row id so the
        // final reconciliation pass can drop entries for rows that did land.
        let mut pending_errors: Vec<(Option<String>, Option<u64>, String)> = Vec::new();
        // Track ids successfully committed by THIS call only; reconciliation
        // uses this set instead of a global store lookup to avoid false
        // positives under concurrent overlapping bulk_inserts.
        let mut committed_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        // Server-assigned ids in input order, parallel to committed_ids; pushed
        // on each Ok store-insert arm and popped if the corresponding chunk
        // rolls back so the final response only carries durably committed rows.
        let mut assigned_ids: Vec<u64> = Vec::new();
        let mut last_completed_batch_idx: u64 = 0;
        let mut last_committed_lsn: u64 = 0;
        let mut chunk_idx: u64 = 0;
        // Track which collections received any successful inserts so we can
        // snapshot each one at the end of the call. Without this, the next
        // restart would FullRebuild from segments (minutes), not
        // IncrementalReplay from hnsw.base (seconds).
        let mut successful_collections: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Group incoming items by collection so each chunk is a single bulk_add_with_lsn.
        struct PendingItem {
            req: InsertRequest,
            item_idx: u64,
        }
        let mut pending_by_collection: std::collections::HashMap<String, Vec<PendingItem>> =
            std::collections::HashMap::new();

        while let Some(result) = stream.message().await? {
            let req = result;
            let current_item = item_index;
            item_index += 1;

            if req.collection.is_empty() {
                pending_errors.push((None, None, format!("item {}: collection name is empty", current_item)));
                continue;
            }
            pending_by_collection
                .entry(req.collection.clone())
                .or_default()
                .push(PendingItem { req, item_idx: current_item });
        }

        for (coll_name, items) in pending_by_collection {
            // Skip whole batches whose target collection is still recovering;
            // every item in that batch surfaces as an "unavailable" error so
            // the client can retry rather than seeing a generic 500.
            if let Err(avail) = self.state.require_collection_ready(&coll_name) {
                let msg = avail.user_message();
                for it in &items {
                    let tag = if it.req.id != 0 { Some(it.req.id) } else { None };
                    pending_errors.push((Some(coll_name.clone()), tag, format!("item {}: {}", it.item_idx, msg)));
                }
                continue;
            }

            // One chunk per collection group in the non-options variant. The per-collection
            // write lock is held across the chunk loop; concurrent searches on OTHER
            // collections are unblocked because the map RwLock is no longer held in write mode.
            let coll_handle = match self.state.collection_handle(&coll_name) {
                Some(h) => h,
                None => {
                    for it in &items {
                        let tag = if it.req.id != 0 { Some(it.req.id) } else { None };
                        pending_errors.push((Some(coll_name.clone()), tag, format!("item {}: collection '{}' not found", it.item_idx, coll_name)));
                    }
                    continue;
                }
            };
            let mut coll = metered_write(&coll_handle);

            // Each vector is wrapped once in Arc; chunk_items and the graph
            // compute map share Arc clones (refcount bump only, no byte copy).
            let mut chunk_items: Vec<(u64, Arc<Vec<f32>>, u64)> = Vec::with_capacity(items.len());
            let mut chunk_metas: Vec<Option<vf_core::types::Metadata>> = Vec::with_capacity(items.len());
            let mut chunk_max_lsn: u64 = last_committed_lsn;

            // One acquisition per (chunk, collection), held across all items in this chunk.
            let mut cm = self.state.collection_manager.write();
            let mut storage_coll_opt = cm.get_collection_mut(&coll_name).ok();

            for it in items {
                let proto_vec = match it.req.vector.as_ref() {
                    Some(v) if !v.values.is_empty() => v,
                    _ => {
                        let tag = if it.req.id != 0 { Some(it.req.id) } else { None };
                        pending_errors.push((Some(coll_name.clone()), tag, format!("item {}: missing or empty vector", it.item_idx)));
                        continue;
                    }
                };

                let values = proto_vec.values.clone();
                let core_metadata = it.req.metadata.as_ref().map(proto_to_core_metadata);

                let assigned_id = if it.req.id == 0 {
                    match coll.store.insert_metadata_auto_id(core_metadata.clone()) {
                        Ok(id) => {
                            committed_ids.insert(id);
                            assigned_ids.push(id);
                            id
                        }
                        Err(e) => {
                            pending_errors.push((Some(coll_name.clone()), None, format!("store insert failed: {}", e)));
                            continue;
                        }
                    }
                } else {
                    match coll.store.insert_metadata(it.req.id, core_metadata.clone()) {
                        Ok(()) => {
                            committed_ids.insert(it.req.id);
                            assigned_ids.push(it.req.id);
                            it.req.id
                        }
                        Err(e) => {
                            pending_errors.push((Some(coll_name.clone()), Some(it.req.id), format!("store insert failed for id {}: {}", it.req.id, e)));
                            continue;
                        }
                    }
                };

                let lsn = match storage_coll_opt.as_mut() {
                    Some(sc) => {
                        if let Err(e) = sc.insert(
                            assigned_id,
                            VectorData::F32(values.clone()),
                            core_metadata.clone(),
                        ) {
                            tracing::warn!(collection = %coll_name, id = assigned_id, "storage bulk insert failed: {}", e);
                        }
                        sc.current_lsn().saturating_sub(1)
                    }
                    None => 0,
                };
                if lsn > chunk_max_lsn {
                    chunk_max_lsn = lsn;
                }

                // Single Arc per row; the index path and the graph compute share it.
                let vec_arc = Arc::new(values);
                chunk_items.push((assigned_id, vec_arc, lsn));
                chunk_metas.push(core_metadata);
            }
            // Drop before index bulk_add to avoid deadlock against index ops.
            drop(storage_coll_opt);
            drop(cm);

            if !chunk_items.is_empty() {
                if let Err(e) = coll.index.bulk_add_with_lsn(&chunk_items) {
                    let rolled_back: std::collections::HashSet<u64> =
                        chunk_items.iter().map(|(id, _, _)| *id).collect();
                    for (id, _, _) in &chunk_items {
                        let _ = coll.store.delete(*id);
                        // Drop rolled-back ids so reconciliation does not
                        // reclassify them as successful inserts.
                        committed_ids.remove(id);
                    }
                    // Symmetric pop from assigned_ids so the response excludes
                    // rows that did not make it past the index step.
                    assigned_ids.retain(|id| !rolled_back.contains(id));
                    let err_str = format!(
                        "chunk {}: bulk index insert failed: {}",
                        chunk_idx, e
                    );
                    for (id, _, _) in &chunk_items {
                        pending_errors.push((Some(coll_name.clone()), Some(*id), err_str.clone()));
                    }
                    // Free per-chunk vector buffers before the next iteration.
                    drop(chunk_items);
                    drop(chunk_metas);
                    chunk_idx += 1;
                    continue;
                }
            }

            // Single pass: index metadata, count, and seed the graph compute map.
            let mut ids: Vec<u64> = Vec::with_capacity(chunk_items.len());
            let mut vectors_map: std::collections::HashMap<u64, Arc<Vec<f32>>> =
                std::collections::HashMap::with_capacity(chunk_items.len());
            for ((assigned_id, vec_arc, _), meta_opt) in chunk_items.iter().zip(chunk_metas.into_iter()) {
                if let Some(ref m) = meta_opt {
                    coll.index_manager.index_record(*assigned_id, m);
                }
                inserted_count += 1;
                ids.push(*assigned_id);
                vectors_map.insert(*assigned_id, Arc::clone(vec_arc));
            }
            drop(chunk_items);

            // Recompute graph edges for just this chunk while holding the same
            // write lock. Vectors are released at end of iteration.
            if !ids.is_empty() {
                // Reborrow the lock guard to a plain &mut CollectionState so
                // the compiler can split the disjoint field borrows of graph
                // and index.
                let coll: &mut CollectionState = &mut *coll;
                if let Err(e) = vf_graph::RelationshipComputer::compute_batch_parallel(
                    &mut coll.graph,
                    coll.index.as_vector_index(),
                    &ids,
                    &vectors_map,
                    10,
                ) {
                    tracing::warn!(collection = %coll_name, "graph compute_batch_parallel after bulk_insert failed: {}", e);
                }
            }
            drop(vectors_map);
            drop(ids);

            // Release capacity ratchet from this collection's metadata index
            // before dropping the write lock for the next batch.
            coll.index_manager.compact();

            last_completed_batch_idx = chunk_idx;
            last_committed_lsn = chunk_max_lsn;
            chunk_idx += 1;
            successful_collections.insert(coll_name.clone());
        }

        // Reconcile pending errors against the live store per collection. A row
        // that landed in the collection must not appear in the user-facing
        // errors list; reclaimed entries are counted as inserted so
        // errors_count + inserted_count stays equal to rows_seen.
        let mut errors: Vec<String> = Vec::with_capacity(pending_errors.len());
        {
            // Reconcile against this call's own committed set so a concurrent
            // overlapping bulk_insert by another client cannot cause us to
            // silently claim credit for rows we did not commit.
            for (_coll_tag, id_tag, msg) in pending_errors {
                let committed = match id_tag {
                    Some(id) => committed_ids.contains(&id),
                    None => false,
                };
                if committed {
                    inserted_count += 1;
                } else {
                    errors.push(msg);
                }
            }
        }

        // Snapshot the HNSW base + graph base for every collection that
        // received successful inserts. Each restart-recovery then takes the
        // fast IncrementalReplay path instead of paying the FullRebuild
        // cost. Snapshot errors are logged but not surfaced: the WAL holds
        // the durable record, and the scheduler will retry.
        for coll_name in &successful_collections {
            if let Err(e) = crate::snapshot::force_snapshot_collection(
                &self.state,
                coll_name,
            ) {
                tracing::warn!(
                    collection = %coll_name,
                    "post-bulk_insert snapshot failed: {}",
                    e
                );
            }
        }

        Ok(Response::new(BulkInsertResponse {
            inserted_count,
            errors,
            last_completed_batch_idx,
            last_committed_lsn,
            resume_token: String::new(),
            assigned_ids,
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
        // wal_flush_every is retained for API compatibility but no longer gates
        // storage writes. Every item is persisted to the storage layer inline
        // so the resulting LSN can flow into the HNSW delta writer, matching
        // the REST handler. The non-zero value is preserved for future use as
        // a batched flush cadence override; zero means "default cadence".
        let _wal_flush_every = options.wal_flush_every;
        let _ef_construction = options.ef_construction; // TODO: thread to HNSW index
        let index_mode_deferred = index_mode == "deferred";
        let skip_metadata_index = options.skip_metadata_index;
        let _parallel_build = options.parallel_build; // TODO: use in optimize() with index_mode=deferred
        let checkpoint_every = options.checkpoint_every;
        let resume_token = options.resume_token.clone();

        let mut inserted_count: u64 = 0;
        let mut item_index: u64 = 0;
        // Pending errors carry the target collection and optional row id so the
        // final reconciliation pass can drop entries for rows that did land.
        let mut pending_errors: Vec<(Option<String>, Option<u64>, String)> = Vec::new();
        // Track ids successfully committed by THIS call only; reconciliation
        // uses this set instead of a global store lookup to avoid false
        // positives under concurrent overlapping bulk_inserts.
        let mut committed_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        // Server-assigned ids in input order, parallel to committed_ids; pushed
        // on each Ok store-insert arm and popped if the corresponding chunk
        // rolls back so the final response only carries durably committed rows.
        let mut assigned_ids: Vec<u64> = Vec::new();
        // Track which collections received any successful inserts so we can
        // snapshot each one at the end of the call. Without this, the next
        // restart would FullRebuild from segments (minutes), not
        // IncrementalReplay from hnsw.base (seconds).
        let mut successful_collections: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Drain the entire stream first so we can group items into deterministic
        // chunks and support resume by chunk_idx.
        struct BatchItem {
            req: InsertRequest,
            item_idx: u64,
        }
        let mut all_items: Vec<BatchItem> = Vec::new();
        let mut collections_with_deferrals: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        while let Some(result) = stream.message().await? {
            match result.payload {
                Some(Payload::Vector(req)) => {
                    let current_item = item_index;
                    item_index += 1;
                    if !req.collection.is_empty() {
                        collections_with_deferrals.insert(req.collection.clone());
                    }
                    all_items.push(BatchItem { req, item_idx: current_item });
                }
                Some(Payload::Options(_)) => {
                    pending_errors.push((None, None, format!(
                        "item {}: unexpected options message after first message",
                        item_index
                    )));
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

        // Determine the primary collection for checkpoint paths. Checkpoint
        // and resume logic only apply when the stream targets a single
        // collection; cross-collection streams skip checkpoint IO.
        let primary_collection: Option<String> = {
            let mut iter = collections_with_deferrals.iter();
            let first = iter.next().cloned();
            if iter.next().is_some() {
                None
            } else {
                first
            }
        };

        let checkpoint_path = match primary_collection.as_ref() {
            Some(name) => resolve_bulk_checkpoint_path(&self.state, name),
            None => None,
        };
        let collection_id_hash = primary_collection
            .as_deref()
            .map(hash_collection_name)
            .unwrap_or(0);

        // Resume validation: only honored when single-collection stream and a
        // checkpoint file exists.
        let mut start_chunk_idx: usize = 0;
        let mut last_completed_batch_idx: u64 = 0;
        let mut last_committed_lsn: u64 = 0;
        let mut any_chunk_completed: bool = false;
        if !resume_token.is_empty() {
            let coll_name = primary_collection.as_ref().ok_or_else(|| {
                Status::invalid_argument(
                    "resume_token provided but stream targets multiple collections",
                )
            })?;
            let cp_path = checkpoint_path.as_ref().ok_or_else(|| {
                Status::invalid_argument(
                    "resume_token provided but server cannot resolve a data directory for this collection",
                )
            })?;
            let cp = vf_storage::bulk_checkpoint::BulkCheckpoint::read(cp_path).map_err(|e| {
                Status::invalid_argument(format!(
                    "resume_token provided but no checkpoint on disk: {}",
                    e
                ))
            })?;
            let expected = encode_resume_token(coll_name, cp.last_committed_lsn);
            if expected != resume_token {
                return Err(Status::invalid_argument(
                    "resume_token does not match on-disk checkpoint (client view diverged from server state)",
                ));
            }
            start_chunk_idx = (cp.last_completed_batch_idx as usize).saturating_add(1);
            last_completed_batch_idx = cp.last_completed_batch_idx;
            last_committed_lsn = cp.last_committed_lsn;
            any_chunk_completed = true;
        }

        // Per chunk: validate per item, write WAL, then bulk_add_with_lsn for
        // surviving items. Group by collection inside each chunk so a single
        // collection write lock covers the bulk add. Graph recompute happens
        // per chunk so vector buffers do not accumulate across the whole stream.

        // Drain all_items into owned chunk vectors so the source buffer can be
        // released immediately. Each chunk is a Vec<BatchItem> instead of a
        // borrowed slice.
        let total_items = all_items.len();
        let chunk_size = batch_lock_size.max(1);
        let total_chunks = (total_items + chunk_size - 1) / chunk_size;
        let mut chunks: Vec<Vec<BatchItem>> = Vec::with_capacity(total_chunks);
        {
            let mut iter = all_items.into_iter();
            for _ in 0..total_chunks {
                let mut bucket: Vec<BatchItem> = Vec::with_capacity(chunk_size);
                for _ in 0..chunk_size {
                    match iter.next() {
                        Some(it) => bucket.push(it),
                        None => break,
                    }
                }
                chunks.push(bucket);
            }
        }
        let mut last_resume_token = String::new();

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            if chunk_idx < start_chunk_idx {
                continue;
            }

            // Group items in this chunk by collection, owning them.
            let mut by_collection: std::collections::HashMap<String, Vec<BatchItem>> =
                std::collections::HashMap::new();
            for item in chunk {
                by_collection
                    .entry(item.req.collection.clone())
                    .or_default()
                    .push(item);
            }

            let mut chunk_max_lsn: u64 = last_committed_lsn;

            for (coll_name, items) in by_collection {
                // Skip whole batches whose target collection is still recovering.
                if let Err(avail) = self.state.require_collection_ready(&coll_name) {
                    let msg = avail.user_message();
                    for it in &items {
                        let tag = if it.req.id != 0 { Some(it.req.id) } else { None };
                        pending_errors.push((Some(coll_name.clone()), tag, format!(
                            "item {}: {}",
                            it.item_idx, msg
                        )));
                    }
                    continue;
                }

                // Per-collection write lock for the duration of this chunk; reads of
                // other collections are not impacted.
                let coll_handle = match self.state.collection_handle(&coll_name) {
                    Some(h) => h,
                    None => {
                        for it in &items {
                            let tag = if it.req.id != 0 { Some(it.req.id) } else { None };
                            pending_errors.push((Some(coll_name.clone()), tag, format!(
                                "item {}: collection '{}' not found",
                                it.item_idx, coll_name
                            )));
                        }
                        continue;
                    }
                };
                let mut coll = metered_write(&coll_handle);

                // Each vector is wrapped once in Arc; chunk_items and the graph
                // compute map share Arc clones (refcount bump only, no byte copy).
                let mut chunk_items: Vec<(u64, Arc<Vec<f32>>, u64)> = Vec::with_capacity(items.len());
                let mut chunk_metas: Vec<Option<vf_core::types::Metadata>> = Vec::with_capacity(items.len());

                // Snapshot the global counter so we can tell if this (chunk, collection) landed any rows.
                let pre_chunk_inserted = inserted_count;

                // One acquisition per (chunk, collection), held across all items in this chunk.
                let mut cm = self.state.collection_manager.write();
                let mut storage_coll_opt = cm.get_collection_mut(&coll_name).ok();

                for it in items {
                    let proto_vec = match it.req.vector.as_ref() {
                        Some(v) if !v.values.is_empty() => v,
                        _ => {
                            let tag = if it.req.id != 0 { Some(it.req.id) } else { None };
                            pending_errors.push((Some(coll_name.clone()), tag, format!(
                                "item {}: missing or empty vector",
                                it.item_idx
                            )));
                            continue;
                        }
                    };

                    let values = proto_vec.values.clone();
                    let core_metadata = it.req.metadata.as_ref().map(proto_to_core_metadata);

                    let assigned_id = if it.req.id == 0 {
                        match coll.store.insert_metadata_auto_id(core_metadata.clone()) {
                            Ok(id) => {
                                committed_ids.insert(id);
                                assigned_ids.push(id);
                                id
                            }
                            Err(e) => {
                                pending_errors.push((Some(coll_name.clone()), None, format!(
                                    "item {}: store insert failed: {}",
                                    it.item_idx, e
                                )));
                                continue;
                            }
                        }
                    } else {
                        match coll.store.insert_metadata(it.req.id, core_metadata.clone()) {
                            Ok(()) => {
                                committed_ids.insert(it.req.id);
                                assigned_ids.push(it.req.id);
                                it.req.id
                            }
                            Err(e) => {
                                pending_errors.push((Some(coll_name.clone()), Some(it.req.id), format!(
                                    "item {}: store insert failed for id {}: {}",
                                    it.item_idx, it.req.id, e
                                )));
                                continue;
                            }
                        }
                    };

                    // Persist to storage layer first to obtain the WAL LSN.
                    // wal_flush_every=0 now means default cadence, not skip WAL.
                    let lsn = match storage_coll_opt.as_mut() {
                        Some(sc) => {
                            if let Err(e) = sc.insert(
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
                            sc.current_lsn().saturating_sub(1)
                        }
                        None => 0,
                    };
                    if lsn > chunk_max_lsn {
                        chunk_max_lsn = lsn;
                    }

                    // Single Arc per row; the index path and the graph compute share it.
                    let vec_arc = Arc::new(values);
                    chunk_items.push((assigned_id, vec_arc, lsn));
                    chunk_metas.push(core_metadata);
                }
                // Drop before index bulk_add to avoid deadlock against index ops.
                drop(storage_coll_opt);
                drop(cm);

                if !index_mode_deferred && !chunk_items.is_empty() {
                    if let Err(e) = coll.index.bulk_add_with_lsn(&chunk_items) {
                        let rolled_back: std::collections::HashSet<u64> =
                            chunk_items.iter().map(|(id, _, _)| *id).collect();
                        for (id, _, _) in &chunk_items {
                            let _ = coll.store.delete(*id);
                            // Drop rolled-back ids so reconciliation does not
                            // reclassify them as successful inserts.
                            committed_ids.remove(id);
                        }
                        // Symmetric pop from assigned_ids so the response excludes
                        // rows that did not make it past the index step.
                        assigned_ids.retain(|id| !rolled_back.contains(id));
                        let err_str = format!(
                            "chunk {}: bulk index insert failed: {}",
                            chunk_idx, e
                        );
                        for (id, _, _) in &chunk_items {
                            pending_errors.push((Some(coll_name.clone()), Some(*id), err_str.clone()));
                        }
                        // Free per-chunk vector buffers before moving on.
                        drop(chunk_items);
                        drop(chunk_metas);
                        continue;
                    }
                }

                // Single pass: index metadata, count, and seed the graph compute map.
                let mut ids: Vec<u64> = Vec::with_capacity(chunk_items.len());
                let mut vectors_map: std::collections::HashMap<u64, Arc<Vec<f32>>> =
                    std::collections::HashMap::with_capacity(chunk_items.len());
                for ((assigned_id, vec_arc, _), meta_opt) in
                    chunk_items.iter().zip(chunk_metas.into_iter())
                {
                    if !skip_metadata_index {
                        if let Some(ref m) = meta_opt {
                            coll.index_manager.index_record(*assigned_id, m);
                        }
                    }
                    inserted_count += 1;
                    ids.push(*assigned_id);
                    vectors_map.insert(*assigned_id, Arc::clone(vec_arc));
                }
                drop(chunk_items);

                if !defer_graph && !ids.is_empty() {
                    // Reborrow the lock guard to a plain &mut CollectionState
                    // so the compiler can split the disjoint field borrows of
                    // graph and index.
                    let coll: &mut CollectionState = &mut *coll;
                    if let Err(e) = vf_graph::RelationshipComputer::compute_batch_parallel(
                        &mut coll.graph,
                        coll.index.as_vector_index(),
                        &ids,
                        &vectors_map,
                        10,
                    ) {
                        tracing::warn!(
                            collection = %coll_name,
                            "graph compute_batch_parallel after bulk_insert_with_options failed: {}",
                            e
                        );
                    }
                }
                drop(vectors_map);
                drop(ids);

                if inserted_count > pre_chunk_inserted {
                    successful_collections.insert(coll_name.clone());
                }
            }

            last_completed_batch_idx = chunk_idx as u64;
            last_committed_lsn = chunk_max_lsn;
            any_chunk_completed = true;

            // Write a checkpoint every N chunks. Only meaningful for single-
            // collection streams where the checkpoint path resolves.
            if checkpoint_every > 0 && (chunk_idx as u64 + 1) % (checkpoint_every as u64) == 0 {
                if let (Some(coll_name), Some(cp_path)) =
                    (primary_collection.as_ref(), checkpoint_path.as_ref())
                {
                    let cp = vf_storage::bulk_checkpoint::BulkCheckpoint::new(
                        collection_id_hash,
                        last_completed_batch_idx,
                        last_committed_lsn,
                    );
                    if let Err(e) = cp.write_atomic(cp_path) {
                        tracing::warn!(
                            collection = %coll_name,
                            chunk_idx,
                            "bulk_insert_with_options checkpoint write failed: {}",
                            e
                        );
                    } else {
                        last_resume_token = encode_resume_token(coll_name, last_committed_lsn);
                    }
                }
            }
        }

        // Release capacity ratchet from each touched collection's metadata
        // index before reporting back to the caller. Each compaction runs under
        // its own per-collection write lock so unrelated collections stay free.
        if inserted_count > 0 && !skip_metadata_index {
            for coll_name in &collections_with_deferrals {
                if let Some(handle) = self.state.collection_handle(coll_name) {
                    let mut coll = metered_write(&handle);
                    coll.index_manager.compact();
                }
            }
        }

        // Set deferred flags and collection status. Per-collection read lock is
        // enough; the atomics and inner status RwLock handle their own sync.
        let any_deferred = defer_graph || index_mode_deferred || skip_metadata_index;
        if any_deferred {
            for coll_name in &collections_with_deferrals {
                if let Some(handle) = self.state.collection_handle(coll_name) {
                    let coll = metered_read(&handle);
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

        // On full completion, scrub the checkpoint file so the next call sees a
        // clean slate.
        let fully_complete = any_chunk_completed
            && total_chunks > 0
            && last_completed_batch_idx as usize + 1 == total_chunks;
        if fully_complete {
            if let Some(ref cp_path) = checkpoint_path {
                let _ = vf_storage::bulk_checkpoint::BulkCheckpoint::delete(cp_path);
            }
            last_resume_token = String::new();
        }

        // Reconcile pending errors against the live store per collection. A row
        // that landed in the collection must not appear in the user-facing
        // errors list; reclaimed entries are counted as inserted so
        // errors_count + inserted_count stays equal to rows_seen.
        let mut errors: Vec<String> = Vec::with_capacity(pending_errors.len());
        {
            // Reconcile against this call's own committed set so a concurrent
            // overlapping bulk_insert by another client cannot cause us to
            // silently claim credit for rows we did not commit.
            for (_coll_tag, id_tag, msg) in pending_errors {
                let committed = match id_tag {
                    Some(id) => committed_ids.contains(&id),
                    None => false,
                };
                if committed {
                    inserted_count += 1;
                } else {
                    errors.push(msg);
                }
            }
        }

        // Snapshot the HNSW base + graph base for every collection that
        // received successful inserts. Each restart-recovery then takes the
        // fast IncrementalReplay path instead of paying the FullRebuild
        // cost. Snapshot errors are logged but not surfaced: the WAL holds
        // the durable record, and the scheduler will retry.
        for coll_name in &successful_collections {
            if let Err(e) = crate::snapshot::force_snapshot_collection(
                &self.state,
                coll_name,
            ) {
                tracing::warn!(
                    collection = %coll_name,
                    "post-bulk_insert_with_options snapshot failed: {}",
                    e
                );
            }
        }

        Ok(Response::new(BulkInsertResponse {
            inserted_count,
            errors,
            last_completed_batch_idx,
            last_committed_lsn,
            resume_token: last_resume_token,
            assigned_ids,
        }))
    }

    async fn bulk_insert_from_path(
        &self,
        request: Request<BulkInsertFromPathRequest>,
    ) -> Result<Response<BulkInsertResponse>, Status> {
        use crate::bulk_insert_from_path::{self as bifp, DatasetView, IdsSource};
        use std::os::fd::AsRawFd;

        let req = request.into_inner();

        if req.collection.is_empty() {
            return Err(Status::invalid_argument("collection name is required"));
        }
        if req.path.is_empty() {
            return Err(Status::invalid_argument("path must not be empty"));
        }

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

        let index_mode = if req.index_mode.is_empty() {
            "immediate"
        } else {
            &req.index_mode
        };
        validate_index_mode(index_mode)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        let index_mode_deferred = index_mode == "deferred";
        let skip_metadata_index = req.skip_metadata_index;

        if req.ef_construction > 0 {
            validate_ef_construction(req.ef_construction, self.max_ef_construction)
                .map_err(|e| Status::invalid_argument(e.to_string()))?;
        }

        let allowed_roots = &self.state.config.bulk_insert_allowed_roots;
        if allowed_roots.is_empty() {
            return Err(Status::internal(
                "bulk_insert_allowed_roots is empty; server misconfigured",
            ));
        }

        // Open every allow-list root once. Root fds stay alive in `root_files`
        // for the duration of the path validation; their raw fds are passed
        // into `open_validated` as the parent set for the openat walk.
        let mut root_files: Vec<std::fs::File> = Vec::with_capacity(allowed_roots.len());
        for root in allowed_roots {
            match std::fs::File::open(root) {
                Ok(f) => root_files.push(f),
                Err(e) => {
                    return Err(Status::internal(format!(
                        "failed to open allow-list root {}: {}",
                        root.display(),
                        e
                    )));
                }
            }
        }
        let root_fds: Vec<std::os::fd::RawFd> =
            root_files.iter().map(|f| f.as_raw_fd()).collect();

        let (data_file, mmap, view): (std::fs::File, memmap2::Mmap, DatasetView) =
            match bifp::open_validated(&req.path, &root_fds, req.dim as usize, req.expected_count) {
                Ok(t) => t,
                Err(e) => return Err(map_bifp_error(e)),
            };
        let _data_file = data_file;

        let coll_handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
            Status::not_found(format!("collection '{}' not found", req.collection))
        })?;

        // Dimension cross-check against the collection config before any work.
        {
            let coll = metered_read(&coll_handle);
            if coll.config.dimension != view.dim {
                return Err(Status::invalid_argument(format!(
                    "vector dimension mismatch: collection expects {}, file has {}",
                    coll.config.dimension, view.dim
                )));
            }
        }

        let row_bytes: &[u8] = &mmap[view.header_offset..];
        if row_bytes.len() < view.count.saturating_mul(view.dim).saturating_mul(4) {
            return Err(Status::invalid_argument(
                "file is shorter than count * dim * 4 bytes",
            ));
        }
        let flat: &[f32] = bytemuck::cast_slice(row_bytes);
        let rows: Vec<&[f32]> = flat
            .chunks_exact(view.dim)
            .take(view.count)
            .collect();

        // Optional ids source. Either mmap a sibling int64 buffer or generate
        // a sequential range starting at `id_start` (defaulting to 1 when 0).
        let ids_holder: IdsSource;
        let ids: Vec<u64> = if req.ids_path.is_empty() {
            let start = if req.id_start == 0 { 1 } else { req.id_start };
            ids_holder = IdsSource::Sequential;
            (0..view.count as u64).map(|i| start + i).collect()
        } else {
            match bifp::open_ids_validated(&req.ids_path, &root_fds, view.count) {
                Ok((file, mmap_ids, ids_vec)) => {
                    ids_holder = IdsSource::Mmap { _file: file, _mmap: mmap_ids };
                    ids_vec
                }
                Err(e) => return Err(map_bifp_error(e)),
            }
        };
        let _ids_holder = ids_holder;

        if ids.len() != rows.len() {
            return Err(Status::invalid_argument(format!(
                "ids count {} does not match vectors count {}",
                ids.len(),
                rows.len()
            )));
        }

        // Per-call D-1 reconciliation set (Decision D5): only ids landed by
        // THIS call get reclaimed during the final error reconciliation.
        let mut committed_ids: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(rows.len());
        let mut assigned_ids: Vec<u64> = Vec::with_capacity(rows.len());
        let mut pending_errors: Vec<(Option<u64>, String)> = Vec::new();
        let mut inserted_count: u64 = 0;
        let mut last_committed_lsn: u64 = 0;

        // Per-collection write lock for the duration of the bulk add (P08
        // nested-lock structural fix: never hold the map lock and the
        // collection lock together). The handle is already an Arc so the map
        // lock has been released by `collection_handle`.
        let mut coll = metered_write(&coll_handle);

        // Allocate metadata-store slots inline so each row gets a stable id
        // before we hand the batch to the index. `insert_metadata` is called
        // with `None` because path-based bulk insert does not carry metadata.
        let mut row_ids: Vec<u64> = Vec::with_capacity(rows.len());
        let mut row_vecs: Vec<&[f32]> = Vec::with_capacity(rows.len());
        for (i, vec_slice) in rows.iter().enumerate() {
            let id_in = ids[i];
            match coll.store.insert_metadata(id_in, None) {
                Ok(()) => {
                    committed_ids.insert(id_in);
                    assigned_ids.push(id_in);
                    row_ids.push(id_in);
                    row_vecs.push(*vec_slice);
                }
                Err(e) => {
                    pending_errors.push((
                        Some(id_in),
                        format!("row {}: store insert failed for id {}: {}", i, id_in, e),
                    ));
                }
            }
        }

        // Persist each surviving row through the storage layer so the WAL
        // produces an LSN per row, mirroring the bulk_insert_with_options
        // pattern at grpc_vector.rs:852-873. The borrowed mmap slice is cloned
        // into a Vec<f32> only at the storage boundary (legacy interface);
        // the index path itself uses the &[f32] without copying.
        let mut lsns: Vec<u64> = Vec::with_capacity(row_ids.len());
        {
            let mut cm = self.state.collection_manager.write();
            if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                for (idx, id_in) in row_ids.iter().enumerate() {
                    if let Err(e) = storage_coll.insert(
                        *id_in,
                        VectorData::F32(row_vecs[idx].to_vec()),
                        None,
                    ) {
                        tracing::warn!(
                            collection = %req.collection,
                            id = *id_in,
                            "storage bulk_insert_from_path insert failed: {}",
                            e
                        );
                    }
                    let lsn = storage_coll.current_lsn().saturating_sub(1);
                    if lsn > last_committed_lsn {
                        last_committed_lsn = lsn;
                    }
                    lsns.push(lsn);
                }
            } else {
                for _ in &row_ids {
                    lsns.push(0);
                }
            }
        }

        // Assemble the (id, &[f32], lsn) triples expected by the new HNSW
        // entry point. The slices borrow from the mmap held above.
        let items: Vec<(vf_core::types::VectorId, &[f32], u64)> = row_ids
            .iter()
            .zip(row_vecs.iter())
            .zip(lsns.iter())
            .map(|((id, v), lsn)| (*id, *v, *lsn))
            .collect();

        let total_hint = if req.total_count_hint == 0 {
            items.len()
        } else {
            req.total_count_hint as usize
        };

        if !index_mode_deferred && !items.is_empty() {
            if req.chunk_size == 0 {
                // Non-chunked path: preserve existing behavior, single bulk_add
                // under the already-held write guard.
                if let Err(e) = coll.index.bulk_add_from_slice_iter(&items, total_hint) {
                    let rolled_back: std::collections::HashSet<u64> =
                        items.iter().map(|(id, _, _)| *id).collect();
                    for (id, _, _) in &items {
                        let _ = coll.store.delete(*id);
                        committed_ids.remove(id);
                    }
                    assigned_ids.retain(|id| !rolled_back.contains(id));
                    let err_str = format!("bulk_insert_from_path: index insert failed: {}", e);
                    for (id, _, _) in &items {
                        pending_errors.push((Some(*id), err_str.clone()));
                    }
                } else {
                    inserted_count = items.len() as u64;
                }
            } else {
                // Chunked path: drop the outer write guard, then for each chunk
                // re-acquire a fresh write guard, run bulk_add, drop the guard,
                // snapshot, prune WAL, and release memory back to the OS. This
                // keeps resident memory bounded across very large bulk inserts.
                drop(coll);

                let chunk_sz = req.chunk_size as usize;
                for chunk in items.chunks(chunk_sz) {
                    let handle = self.state.collection_handle(&req.collection).ok_or_else(|| {
                        Status::not_found(format!(
                            "collection '{}' not found",
                            req.collection
                        ))
                    })?;
                    let coll_inner = metered_write(&handle);
                    if let Err(e) = coll_inner.index.bulk_add_from_slice_iter(chunk, total_hint) {
                        // Rollback the failed chunk's ids under the same guard
                        // we already hold, then bail out of the loop.
                        let rolled_back: std::collections::HashSet<u64> =
                            chunk.iter().map(|(id, _, _)| *id).collect();
                        for (id, _, _) in chunk {
                            let _ = coll_inner.store.delete(*id);
                            committed_ids.remove(id);
                        }
                        drop(coll_inner);
                        drop(handle);
                        assigned_ids.retain(|id| !rolled_back.contains(id));
                        let err_str = format!(
                            "bulk_insert_from_path: index insert failed: {}",
                            e
                        );
                        for (id, _, _) in chunk {
                            pending_errors.push((Some(*id), err_str.clone()));
                        }
                        break;
                    }
                    drop(coll_inner);
                    drop(handle);

                    if let Err(e) = crate::snapshot::force_snapshot_collection(
                        &self.state,
                        &req.collection,
                    ) {
                        tracing::warn!(
                            collection = %req.collection,
                            "chunked bulk_insert_from_path: snapshot failed: {}",
                            e
                        );
                    }
                    if let Err(e) = self.state.prune_wal_for_collection(&req.collection) {
                        tracing::warn!(
                            collection = %req.collection,
                            "chunked bulk_insert_from_path: prune_wal failed: {}",
                            e
                        );
                    }
                    vf_index::release_to_os();

                    inserted_count += chunk.len() as u64;
                }

                // Re-acquire the write guard on the SAME collection handle so
                // the trailing compact / deferred-flag bookkeeping below can
                // proceed under a held write lock, mirroring the non-chunked
                // path.
                coll = metered_write(&coll_handle);
            }
        } else if index_mode_deferred {
            inserted_count = items.len() as u64;
            if !items.is_empty() {
                coll.deferred_index.store(true, Ordering::Release);
            }
        }

        if !skip_metadata_index {
            coll.index_manager.compact();
        } else if !items.is_empty() {
            coll.deferred_metadata.store(true, Ordering::Release);
        }

        drop(coll);

        // Per-call D-1 reconciliation (Decision D5): a pending error is
        // reclaimed as inserted only if THIS call's committed_ids set still
        // holds the id. Mirrors grpc_vector.rs:511-531.
        let mut errors: Vec<String> = Vec::with_capacity(pending_errors.len());
        for (id_tag, msg) in pending_errors {
            let committed = match id_tag {
                Some(id) => committed_ids.contains(&id),
                None => false,
            };
            if committed {
                inserted_count += 1;
            } else {
                errors.push(msg);
            }
        }

        // Always snapshot the HNSW base + graph base at the end of a
        // successful bulk_insert_from_path call. The chunked path snapshots
        // per chunk; the single-pass (chunk_size=0) path used to skip this,
        // leaving a fat hnsw.delta with no hnsw.base on disk. Any restart
        // (clean or SIGKILL) then demoted to FullRebuild, doubling the
        // wall-clock and memory cost of recovery. Doing the snapshot here
        // makes IncrementalReplay the expected recovery path after any
        // bulk_insert_from_path completes. Failures are logged but not
        // surfaced: the data is already in the WAL, so durability is intact;
        // worst case the next snapshot (scheduler or next bulk) covers it.
        if inserted_count > 0 {
            if let Err(e) = crate::snapshot::force_snapshot_collection(
                &self.state,
                &req.collection,
            ) {
                tracing::warn!(
                    collection = %req.collection,
                    "post-bulk_insert_from_path snapshot failed: {}",
                    e
                );
            }
        }

        Ok(Response::new(BulkInsertResponse {
            inserted_count,
            errors,
            last_completed_batch_idx: 0,
            last_committed_lsn,
            resume_token: String::new(),
            assigned_ids,
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

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

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

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

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

        self.state
            .require_collection_ready(&req.collection)
            .map_err(status_from_availability)?;

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
