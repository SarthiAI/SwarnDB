// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! gRPC VectorService handler implementation.

use tonic::{Request, Response, Status};

use crate::convert::{core_to_proto_metadata, proto_to_core_metadata};
use crate::proto::swarndb::v1::vector_service_server::VectorService;
use crate::proto::swarndb::v1::{
    BulkInsertResponse, DeleteVectorRequest, DeleteVectorResponse, GetVectorRequest,
    GetVectorResponse, InsertRequest, InsertResponse, UpdateRequest, UpdateResponse, Vector,
};
use crate::state::AppState;
use vf_core::store::VectorRecord;
use vf_core::vector::VectorData;
use vf_index::traits::VectorIndex;

pub struct VectorServiceImpl {
    state: AppState,
}

impl VectorServiceImpl {
    pub fn new(state: AppState) -> Self {
        Self { state }
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

        let vector_data = VectorData::F32(values.clone());

        let assigned_id = if req.id == 0 {
            let id = coll
                .store
                .insert_auto_id(vector_data, core_metadata.clone())
                .map_err(|e| Status::internal(format!("store insert failed: {}", e)))?;
            id
        } else {
            let record = VectorRecord::new(req.id, vector_data, core_metadata.clone());
            coll.store
                .insert(record)
                .map_err(|e| Status::internal(format!("store insert failed: {}", e)))?;
            req.id
        };

        if let Err(e) = coll.index.add(assigned_id, &values) {
            // Rollback: remove from store since index add failed
            let _ = coll.store.delete(assigned_id);
            return Err(Status::internal(format!("index insert failed: {}", e)));
        }

        if let Some(ref meta) = core_metadata {
            coll.index_manager.index_record(assigned_id, meta);
        }

        // Compute virtual graph edges for the newly inserted vector
        if let Err(e) = vf_graph::RelationshipComputer::compute_for_vector(
            &mut coll.graph, &coll.index, assigned_id, &values, 10,
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

            let vector_data = VectorData::F32(values.clone());

            let assigned_id = if req.id == 0 {
                match coll.store.insert_auto_id(vector_data, core_metadata.clone()) {
                    Ok(id) => id,
                    Err(e) => {
                        errors.push(format!("store insert failed: {}", e));
                        continue;
                    }
                }
            } else {
                let record = VectorRecord::new(req.id, vector_data, core_metadata.clone());
                match coll.store.insert(record) {
                    Ok(()) => req.id,
                    Err(e) => {
                        errors.push(format!("store insert failed for id {}: {}", req.id, e));
                        continue;
                    }
                }
            };

            if let Err(e) = coll.index.add(assigned_id, &values) {
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
                &mut coll.graph, &coll.index, assigned_id, &values, 10,
            ) {
                tracing::warn!(collection = %req.collection, id = assigned_id, "graph compute failed: {}", e);
            }

            // Save values for batch graph recomputation before they're moved
            let values_for_graph = values.clone();

            // Persist to storage layer (best-effort)
            {
                let mut cm = self.state.collection_manager.write();
                if let Ok(storage_coll) = cm.get_collection_mut(&req.collection) {
                    if let Err(e) = storage_coll.insert(assigned_id, VectorData::F32(values), core_metadata) {
                        tracing::warn!(collection = %req.collection, id = assigned_id, "storage bulk insert failed: {}", e);
                    }
                }
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
                    &mut coll.graph, &coll.index, ids, vectors_map, 10,
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
}
