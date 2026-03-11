// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! gRPC VectorService handler implementation.

use tonic::{Request, Response, Status};

use crate::convert::{core_to_proto_metadata, proto_to_core_metadata};
use crate::proto::swarndb::v1::vector_service_server::VectorService;
use crate::validation::validate_vector_data;
use std::sync::atomic::Ordering;

use crate::proto::swarndb::v1::{
    BulkInsertResponse, BulkInsertStreamMessage, DeleteVectorRequest, DeleteVectorResponse,
    GetVectorRequest, GetVectorResponse, InsertRequest, InsertResponse, OptimizeRequest,
    OptimizeResponse, UpdateRequest, UpdateResponse, Vector,
    bulk_insert_stream_message::Payload,
};
use crate::state::{AppState, CollectionStatus};
use crate::validation::{
    validate_batch_lock_size, validate_bulk_insert_options, validate_ef_construction,
    validate_index_mode, validate_wal_flush_every,
};
use vf_core::store::VectorRecord;
use vf_core::vector::VectorData;
use vf_index::traits::VectorIndex;

/// Sanitize a string for safe inclusion in log messages by removing control chars.
fn sanitize_for_log(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control())
        .take(256)
        .collect()
}

// MAX_BULK_INSERT_MESSAGES and MAX_BULK_INSERT_PAYLOAD_BYTES are now
// configurable via AppState (config.max_bulk_insert_messages / max_bulk_insert_payload_bytes).

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
            Status::not_found(format!("collection '{}' not found", sanitize_for_log(&req.collection)))
        })?;

        // Validate vector dimension against collection config
        if let Err(e) = validate_vector_data(&values, coll.config.dimension) {
            return Err(Status::invalid_argument(e.to_string()));
        }

        let vector_data = VectorData::F32(values.clone());

        let assigned_id = if req.id == 0 {
            let id = coll
                .store
                .insert_auto_id(vector_data, core_metadata.clone())
                .map_err(|e| {
                    tracing::error!("store insert failed: {}", e);
                    Status::internal("internal error")
                })?;
            id
        } else {
            let record = VectorRecord::new(req.id, vector_data, core_metadata.clone());
            coll.store
                .insert(record)
                .map_err(|e| {
                    tracing::error!("store insert failed: {}", e);
                    Status::internal("internal error")
                })?;
            req.id
        };

        if let Err(e) = coll.index.add(assigned_id, &values) {
            // Rollback: remove from store since index add failed
            let _ = coll.store.delete(assigned_id);
            tracing::error!("index insert failed: {}", e);
            return Err(Status::internal("internal error"));
        }

        if let Some(ref meta) = core_metadata {
            coll.index_manager.index_record(assigned_id, meta);
        }

        // Compute virtual graph edges for the newly inserted vector
        let graph_k = coll.graph.config().graph_neighbors_k;
        if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
            &mut coll.graph, &coll.index, assigned_id, &values, graph_k,
        ) {
            tracing::warn!(collection = %req.collection, id = assigned_id, "graph compute failed: {}", e);
        }

        // Persist to storage layer (best-effort)
        {
            let mut cm = self.state.collection_manager.write();
            if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                if let Err(e) = storage_coll.insert(assigned_id, VectorData::F32(values), core_metadata) {
                    tracing::warn!(collection = %req.collection, id = assigned_id, "storage insert failed: {}", e);
                }
            }
        }

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

        let record = coll
            .store
            .get(req.id)
            .map_err(|e| Status::not_found(format!("vector not found: {}", e)))?;

        let proto_vector = Vector {
            values: record.data.to_f32_vec(),
        };

        let proto_metadata = record.metadata.as_ref().map(core_to_proto_metadata);

        Ok(Response::new(GetVectorResponse {
            id: record.id,
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

        let vector_data = values.as_ref().map(|v| VectorData::F32(v.clone()));
        coll.store
            .update(req.id, vector_data.clone(), core_metadata.clone())
            .map_err(|e| Status::not_found(format!("update failed: {}", e)))?;

        // Only update vector index if new vector data was provided
        if let Some(ref vals) = values {
            let _ = coll.index.remove(req.id);
            coll.index
                .add(req.id, vals)
                .map_err(|e| Status::internal(format!("index update failed: {}", e)))?;
        }

        // Update metadata index if metadata was provided
        if let Some(ref meta) = core_metadata {
            coll.index_manager.remove_record(req.id);
            coll.index_manager.index_record(req.id, meta);
        }

        // Persist to storage layer (best-effort)
        {
            let mut cm = self.state.collection_manager.write();
            if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                let storage_data = values.map(VectorData::F32);
                if let Err(e) = storage_coll.update(req.id, storage_data, core_metadata) {
                    tracing::warn!(collection = %req.collection, id = req.id, "storage update failed: {}", e);
                }
            }
        }

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

        let _ = coll.index.remove(req.id);
        coll.index_manager.remove_record(req.id);
        coll.graph.remove_node(req.id);

        // Persist to storage layer (best-effort)
        {
            let mut cm = self.state.collection_manager.write();
            if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                if let Err(e) = storage_coll.delete(req.id) {
                    tracing::warn!(collection = %req.collection, id = req.id, "storage delete failed: {}", e);
                }
            }
        }

        Ok(Response::new(DeleteVectorResponse { success: true }))
    }

    async fn bulk_insert(
        &self,
        request: Request<tonic::Streaming<InsertRequest>>,
    ) -> Result<Response<BulkInsertResponse>, Status> {
        let mut stream = request.into_inner();
        let mut inserted_count: u64 = 0;
        let mut item_index: u64 = 0;
        let mut total_payload_bytes: u64 = 0;
        let mut errors: Vec<String> = Vec::new();
        let mut batch_vectors: std::collections::HashMap<String, (Vec<u64>, std::collections::HashMap<u64, Vec<f32>>)> = std::collections::HashMap::new();

        // Batch messages before acquiring the write lock
        const BATCH_SIZE: usize = 100;
        let mut message_batch: Vec<(u64, InsertRequest)> = Vec::with_capacity(BATCH_SIZE);

        loop {
            // Collect a batch of messages from the stream
            message_batch.clear();
            let mut stream_done = false;
            while message_batch.len() < BATCH_SIZE {
                match stream.message().await? {
                    Some(req) => {
                        let current_item = item_index;
                        item_index += 1;
                        if item_index > self.state.max_bulk_insert_messages {
                            return Err(Status::resource_exhausted(format!(
                                "bulk_insert stream exceeded maximum message count ({})",
                                self.state.max_bulk_insert_messages
                            )));
                        }
                        if let Some(ref v) = req.vector {
                            total_payload_bytes += (v.values.len() * std::mem::size_of::<f32>()) as u64;
                        }
                        if total_payload_bytes > self.state.max_bulk_insert_payload_bytes {
                            return Err(Status::resource_exhausted(format!(
                                "bulk_insert stream exceeded maximum payload size ({} bytes)",
                                self.state.max_bulk_insert_payload_bytes
                            )));
                        }
                        message_batch.push((current_item, req));
                    }
                    None => {
                        stream_done = true;
                        break;
                    }
                }
            }

            if message_batch.is_empty() {
                break;
            }

            // Process the entire batch under a single write lock
            {
                let mut collections = self.state.collections.write();

                for (current_item, req) in &message_batch {
                    let proto_vec = match req.vector.as_ref() {
                        Some(v) if !v.values.is_empty() => v,
                        _ => {
                            errors.push(format!("item {}: missing or empty vector", current_item));
                            continue;
                        }
                    };

                    let values = proto_vec.values.clone();
                    let core_metadata = req.metadata.as_ref().map(proto_to_core_metadata);

                    let coll = match collections.get_mut(&req.collection) {
                        Some(c) => c,
                        None => {
                            errors.push(format!("item {}: collection not found", current_item));
                            continue;
                        }
                    };

                    // Validate dimension
                    if values.len() != coll.config.dimension {
                        errors.push(format!("item {}: dimension mismatch", current_item));
                        continue;
                    }

                    let vector_data = VectorData::F32(values.clone());

                    let assigned_id = if req.id == 0 {
                        match coll.store.insert_auto_id(vector_data, core_metadata.clone()) {
                            Ok(id) => id,
                            Err(_e) => {
                                errors.push(format!("item {}: store insert failed", current_item));
                                continue;
                            }
                        }
                    } else {
                        let record = VectorRecord::new(req.id, vector_data, core_metadata.clone());
                        match coll.store.insert(record) {
                            Ok(()) => req.id,
                            Err(_e) => {
                                errors.push(format!("item {}: store insert failed", current_item));
                                continue;
                            }
                        }
                    };

                    if let Err(_e) = coll.index.add(assigned_id, &values) {
                        let _ = coll.store.delete(assigned_id);
                        errors.push(format!("item {}: index insert failed", current_item));
                        continue;
                    }

                    if let Some(ref meta) = core_metadata {
                        coll.index_manager.index_record(assigned_id, meta);
                    }

                    // Compute virtual graph edges for the newly inserted vector
                    let graph_k = coll.graph.config().graph_neighbors_k;
                    if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
                        &mut coll.graph, &coll.index, assigned_id, &values, graph_k,
                    ) {
                        tracing::warn!(id = assigned_id, "graph compute failed: {}", e);
                    }

                    let entry = batch_vectors.entry(req.collection.clone()).or_insert_with(|| (Vec::new(), std::collections::HashMap::new()));
                    entry.0.push(assigned_id);
                    entry.1.insert(assigned_id, values.clone());

                    // Persist to storage layer (best-effort)
                    {
                        let mut cm = self.state.collection_manager.write();
                        if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                            if let Err(e) = storage_coll.insert(assigned_id, VectorData::F32(values), core_metadata) {
                                tracing::warn!(id = assigned_id, "storage bulk insert failed: {}", e);
                            }
                        }
                    }

                    inserted_count += 1;
                }
            } // write lock released

            if stream_done {
                break;
            }
        }

        // Recompute graph edges for all inserted vectors now that the full index is available
        for (coll_name, (ids, vectors_map)) in &batch_vectors {
            let mut collections = self.state.collections.write();
            if let Some(coll) = collections.get_mut(coll_name) {
                let graph_k = coll.graph.config().graph_neighbors_k;
                if let Err(e) = vf_graph::RelationshipComputer::compute_batch(
                    &mut coll.graph, &coll.index, ids, vectors_map, graph_k,
                ) {
                    tracing::warn!("graph compute_batch after bulk_insert failed: {}", e);
                }
            }
        }

        tracing::info!(inserted = inserted_count, errors = errors.len(), "bulk_insert completed");
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
        let any_deferred = defer_graph || index_mode_deferred || skip_metadata_index;

        let mut inserted_count: u64 = 0;
        let mut item_index: u64 = 0;
        let mut total_payload_bytes: u64 = 0;
        let mut errors: Vec<String> = Vec::new();
        let mut wal_counter: u32 = 0;
        let mut pending_wal: Vec<(String, u64, Vec<f32>, Option<vf_core::types::Metadata>)> = Vec::new();

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

        // Helper closure to process a batch of items
        let process_batch = |batch: &mut Vec<BatchItem>,
                             state: &AppState,
                             errors: &mut Vec<String>,
                             inserted_count: &mut u64,
                             wal_counter: &mut u32,
                             wal_flush_every: u32,
                             defer_graph: bool,
                             index_mode_deferred: bool,
                             skip_metadata_index: bool,
                             batch_vectors: &mut std::collections::HashMap<
            String,
            (Vec<u64>, std::collections::HashMap<u64, Vec<f32>>),
        >,
                             pending_wal: &mut Vec<(String, u64, Vec<f32>, Option<vf_core::types::Metadata>)>,
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
                    let vector_data = VectorData::F32(values.clone());

                    // Insert into store
                    let assigned_id = if item.req.id == 0 {
                        match coll.store.insert_auto_id(vector_data, core_metadata.clone()) {
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
                        let record =
                            VectorRecord::new(item.req.id, vector_data, core_metadata.clone());
                        match coll.store.insert(record) {
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

                    // HNSW index: skip if deferred
                    if !index_mode_deferred {
                        if let Err(e) = coll.index.add(assigned_id, &values) {
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
                        let graph_k = coll.graph.config().graph_neighbors_k;
                        if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
                            &mut coll.graph,
                            &coll.index,
                            assigned_id,
                            &values,
                            graph_k,
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

                    // WAL persistence: accumulate into pending buffer, flush at threshold
                    if wal_flush_every > 0 {
                        pending_wal.push((coll_name.clone(), assigned_id, values, core_metadata));
                        *wal_counter += 1;

                        if wal_flush_every <= 1 || *wal_counter >= wal_flush_every {
                            let mut cm = state.collection_manager.write();
                            for (cn, id, vals, meta) in pending_wal.drain(..) {
                                if let Ok(storage_coll) = cm.get_collection_mut(&cn) {
                                    if let Err(e) = storage_coll.insert(
                                        id,
                                        VectorData::F32(vals),
                                        meta,
                                    ) {
                                        tracing::warn!(
                                            collection = %cn,
                                            id = id,
                                            "storage insert failed: {}",
                                            e
                                        );
                                    }
                                }
                            }
                            *wal_counter = 0;
                        }
                    }
                    // wal_flush_every == 0 means skip WAL entirely
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

                    // Enforce stream limits
                    if item_index > self.state.max_bulk_insert_messages {
                        return Err(Status::resource_exhausted(format!(
                            "bulk_insert_with_options stream exceeded maximum message count ({})",
                            self.state.max_bulk_insert_messages
                        )));
                    }
                    if let Some(ref v) = req.vector {
                        total_payload_bytes += (v.values.len() * std::mem::size_of::<f32>()) as u64;
                    }
                    if total_payload_bytes > self.state.max_bulk_insert_payload_bytes {
                        return Err(Status::resource_exhausted(format!(
                            "bulk_insert_with_options stream exceeded maximum payload size ({} bytes)",
                            self.state.max_bulk_insert_payload_bytes
                        )));
                    }

                    if any_deferred && !req.collection.is_empty() {
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
                            &mut wal_counter,
                            wal_flush_every,
                            defer_graph,
                            index_mode_deferred,
                            skip_metadata_index,
                            &mut batch_vectors,
                            &mut pending_wal,
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
            &mut wal_counter,
            wal_flush_every,
            defer_graph,
            index_mode_deferred,
            skip_metadata_index,
            &mut batch_vectors,
            &mut pending_wal,
        );

        // Flush any remaining WAL entries
        if !pending_wal.is_empty() {
            let mut cm = self.state.collection_manager.write();
            for (cn, id, vals, meta) in pending_wal.drain(..) {
                if let Ok(storage_coll) = cm.get_collection_mut(&cn) {
                    if let Err(e) = storage_coll.insert(id, VectorData::F32(vals), meta) {
                        tracing::warn!(
                            collection = %cn,
                            id = id,
                            "storage insert failed: {}",
                            e
                        );
                    }
                }
            }
        }

        // Recompute graph edges for all inserted vectors (only if graph not deferred)
        if !defer_graph {
            for (coll_name, (ids, vectors_map)) in &batch_vectors {
                let mut collections = self.state.collections.write();
                if let Some(coll) = collections.get_mut(coll_name) {
                    let graph_k = coll.graph.config().graph_neighbors_k;
                    if let Err(e) = vf_graph::RelationshipComputer::compute_batch(
                        &mut coll.graph, &coll.index, ids, vectors_map, graph_k,
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

        match self.state.optimize_collection(&req.collection) {
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
}
