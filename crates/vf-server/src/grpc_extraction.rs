// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! gRPC surface for the LLM extraction service. Every handler gates on
//! collection readiness and Hybrid mode, then delegates to the shared
//! `ExtractionManager`, mapping `ExtractionError` onto tonic `Status`.

use tonic::{Request, Response, Status};

use vf_core::types::Mode;
use vf_extraction::{
    ChunkContent, EdgeTypeDef, EntityLabelDef, ExtractionError, LlmConfig, Ontology,
};

use crate::metrics;
use crate::proto::swarndb::v1::extraction_service_server::ExtractionService;
use crate::proto::swarndb::v1::{
    AppendExtractionChunksRequest, AppendExtractionChunksResponse, ApproveProposalRequest,
    ApproveProposalResponse, CancelCorpusReextractionRequest, CancelCorpusReextractionResponse,
    CancelExtractionRequest, CancelExtractionResponse, Chunk,
    ChunkDiff as ProtoChunkDiff, ChunkDiffAction as ProtoChunkDiffAction, ChunkErrorMsg,
    CorpusDocProgressMsg, CorpusReextractionStatusMsg, CostPreviewRequest, CostPreviewResponse,
    DiffDocumentRequest, DiffDocumentResponse,
    EntityResolutionMode, GetCorpusReextractionStatusRequest, GetExtractionStatusRequest,
    GetLlmConfigRequest, GetLlmConfigResponse,
    GetOntologyRequest,
    GetOntologyResponse, JobStatusMsg, ListProposalsRequest, ListProposalsResponse,
    OntologyEdgeType, OntologyEntityLabel, OntologyMsg, ProposalMsg, ReextractDocumentRequest,
    ReextractDocumentResponse, RejectProposalRequest, RejectProposalResponse,
    RotateLlmConfigRequest, RotateLlmConfigResponse, SetLlmConfigRequest, SetLlmConfigResponse,
    SetOntologyRequest, SetOntologyResponse, StartCorpusReextractionRequest,
    StartCorpusReextractionResponse, StartExtractionRequest, StartExtractionResponse,
};
use crate::state::{metered_read, AppState, CollectionAvailability};

fn status_from_availability(avail: CollectionAvailability) -> Status {
    match avail {
        CollectionAvailability::Recovering { .. } => Status::unavailable(avail.user_message()),
        CollectionAvailability::NotFound { .. } => Status::not_found(avail.user_message()),
    }
}

/// Map an `ExtractionError` onto a tonic `Status` with the right code class.
fn status_from_extraction(e: ExtractionError) -> Status {
    match e {
        ExtractionError::Config(m) => Status::invalid_argument(m),
        ExtractionError::Ontology(m) => Status::invalid_argument(m),
        ExtractionError::Parse(m) => Status::invalid_argument(m),
        ExtractionError::JobNotFound(m) => Status::not_found(format!("job not found: {m}")),
        ExtractionError::Crypto(m) => Status::failed_precondition(m),
        ExtractionError::Cancelled => Status::aborted("extraction cancelled"),
        ExtractionError::Llm(m) => Status::internal(format!("llm error: {m}")),
        ExtractionError::Io(m) => Status::internal(format!("io error: {m}")),
        ExtractionError::Graph(m) => Status::internal(format!("graph write error: {m}")),
    }
}

fn chunk_from_proto(c: Chunk) -> ChunkContent {
    ChunkContent {
        doc_id: c.doc_id,
        chunk_id: c.chunk_id,
        text: c.text,
        embedding: if c.embedding.is_empty() {
            None
        } else {
            Some(c.embedding)
        },
    }
}

fn ontology_to_proto(o: &Ontology) -> OntologyMsg {
    OntologyMsg {
        entity_labels: o
            .entity_labels
            .iter()
            .map(|l| OntologyEntityLabel {
                label: l.label.clone(),
                description: l.description.clone(),
            })
            .collect(),
        edge_types: o
            .edge_types
            .iter()
            .map(|t| OntologyEdgeType {
                edge_type: t.edge_type.clone(),
                description: t.description.clone(),
                source_labels: t.source_labels.clone(),
                target_labels: t.target_labels.clone(),
            })
            .collect(),
        // None -> empty string; the wire treats empty as "use the default prompt".
        system_prompt: o.system_prompt.clone().unwrap_or_default(),
        extra_guidance: o.extra_guidance.clone().unwrap_or_default(),
        link_passages: o.link_passages,
        // ADR-020. Map the engine resolution mode to the proto enum value.
        entity_resolution: entity_resolution_to_proto(o.entity_resolution) as i32,
    }
}

/// Map the engine resolution enum to the proto `EntityResolutionMode` enum.
fn entity_resolution_to_proto(mode: vf_extraction::EntityResolution) -> EntityResolutionMode {
    match mode {
        vf_extraction::EntityResolution::Normalized => {
            EntityResolutionMode::EntityResolutionNormalized
        }
        vf_extraction::EntityResolution::Fuzzy => EntityResolutionMode::EntityResolutionFuzzy,
    }
}

/// Map the proto enum value to the engine resolution enum. Unknown / default (0)
/// decodes to Normalized, the backward-compatible default.
fn entity_resolution_from_proto(raw: i32) -> vf_extraction::EntityResolution {
    match EntityResolutionMode::try_from(raw) {
        Ok(EntityResolutionMode::EntityResolutionFuzzy) => vf_extraction::EntityResolution::Fuzzy,
        _ => vf_extraction::EntityResolution::Normalized,
    }
}

fn ontology_from_proto(msg: OntologyMsg) -> Ontology {
    Ontology {
        entity_labels: msg
            .entity_labels
            .into_iter()
            .map(|l| EntityLabelDef::new(l.label, l.description))
            .collect(),
        edge_types: msg
            .edge_types
            .into_iter()
            .map(|t| EdgeTypeDef::new(t.edge_type, t.description, t.source_labels, t.target_labels))
            .collect(),
        // Empty proto string -> None so the engine falls back to the default prompt.
        system_prompt: if msg.system_prompt.trim().is_empty() {
            None
        } else {
            Some(msg.system_prompt.clone())
        },
        extra_guidance: if msg.extra_guidance.trim().is_empty() {
            None
        } else {
            Some(msg.extra_guidance.clone())
        },
        link_passages: msg.link_passages,
        // ADR-020. Default (0) decodes to Normalized for backward compatibility.
        entity_resolution: entity_resolution_from_proto(msg.entity_resolution),
    }
}

/// The extraction gRPC service implementation.
pub struct ExtractionServiceImpl {
    state: AppState,
}

impl ExtractionServiceImpl {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    /// Readiness + Hybrid-mode gate shared by every handler. Returns the error
    /// status to reject with, or `Ok(())` to proceed.
    fn gate(&self, collection: &str) -> Result<(), Status> {
        self.state
            .require_collection_ready(collection)
            .map_err(status_from_availability)?;
        let handle = self
            .state
            .collection_handle(collection)
            .ok_or_else(|| Status::not_found(format!("collection '{collection}' not found")))?;
        let mode = {
            let coll = metered_read(&handle);
            coll.config.effective_mode()
        };
        if mode != Mode::Hybrid {
            return Err(Status::failed_precondition(format!(
                "collection '{collection}' is not in hybrid mode; extraction is not available"
            )));
        }
        Ok(())
    }
}

#[tonic::async_trait]
impl ExtractionService for ExtractionServiceImpl {
    async fn set_llm_config(
        &self,
        request: Request<SetLlmConfigRequest>,
    ) -> Result<Response<SetLlmConfigResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let config = LlmConfig::new(
            req.base_url,
            req.api_key,
            req.model_name,
            req.temperature,
            req.max_tokens,
            req.timeout_seconds,
        );
        self.state
            .extraction
            .set_llm_config(&req.collection, config)
            .map_err(status_from_extraction)?;
        Ok(Response::new(SetLlmConfigResponse { success: true }))
    }

    async fn get_llm_config(
        &self,
        request: Request<GetLlmConfigRequest>,
    ) -> Result<Response<GetLlmConfigResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let redacted = self
            .state
            .extraction
            .get_llm_config(&req.collection)
            .map_err(status_from_extraction)?;
        Ok(Response::new(GetLlmConfigResponse {
            base_url: redacted.base_url,
            model_name: redacted.model_name,
            temperature: redacted.temperature,
            max_tokens: redacted.max_tokens,
            timeout_seconds: redacted.timeout_seconds,
            api_key_set: redacted.api_key_set,
        }))
    }

    async fn rotate_llm_config(
        &self,
        request: Request<RotateLlmConfigRequest>,
    ) -> Result<Response<RotateLlmConfigResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        self.state
            .extraction
            .rotate_llm_config(&req.collection, &req.new_api_key)
            .map_err(status_from_extraction)?;
        Ok(Response::new(RotateLlmConfigResponse { success: true }))
    }

    async fn set_ontology(
        &self,
        request: Request<SetOntologyRequest>,
    ) -> Result<Response<SetOntologyResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let base_template = if req.base_template.trim().is_empty() {
            None
        } else {
            Some(req.base_template)
        };
        let extension = req.extension.map(ontology_from_proto).unwrap_or_default();
        self.state
            .extraction
            .set_ontology(&req.collection, base_template, extension, req.replace)
            .map_err(status_from_extraction)?;
        Ok(Response::new(SetOntologyResponse { success: true }))
    }

    async fn get_ontology(
        &self,
        request: Request<GetOntologyRequest>,
    ) -> Result<Response<GetOntologyResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let ontology = self
            .state
            .extraction
            .get_ontology(&req.collection)
            .map_err(status_from_extraction)?;
        Ok(Response::new(GetOntologyResponse {
            ontology: Some(ontology_to_proto(&ontology)),
        }))
    }

    async fn cost_preview(
        &self,
        request: Request<CostPreviewRequest>,
    ) -> Result<Response<CostPreviewResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(chunk_from_proto).collect();
        let estimate = self
            .state
            .extraction
            .cost_preview(&req.collection, &chunks)
            .await
            .map_err(status_from_extraction)?;
        Ok(Response::new(CostPreviewResponse {
            chunks: estimate.chunks as u64,
            estimated_input_tokens: estimate.estimated_input_tokens,
            estimated_output_tokens: estimate.estimated_output_tokens,
            estimated_cost_usd: estimate.estimated_cost_usd,
            model: estimate.model,
            pricing_known: estimate.pricing_known,
        }))
    }

    async fn start_extraction(
        &self,
        request: Request<StartExtractionRequest>,
    ) -> Result<Response<StartExtractionResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(chunk_from_proto).collect();
        let job_id = self
            .state
            .extraction
            .start_extraction(&req.collection, chunks)
            .map_err(status_from_extraction)?;
        metrics::record_extraction_job("started");
        Ok(Response::new(StartExtractionResponse { job_id }))
    }

    async fn append_extraction_chunks(
        &self,
        request: Request<AppendExtractionChunksRequest>,
    ) -> Result<Response<AppendExtractionChunksResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(chunk_from_proto).collect();
        let total_chunks = self
            .state
            .extraction
            .append_extraction(&req.collection, &req.job_id, chunks, req.last_batch)
            .map_err(status_from_extraction)?;
        Ok(Response::new(AppendExtractionChunksResponse {
            total_chunks,
            accepted: true,
        }))
    }

    async fn get_extraction_status(
        &self,
        request: Request<GetExtractionStatusRequest>,
    ) -> Result<Response<JobStatusMsg>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let status = self
            .state
            .extraction
            .job_status(&req.collection, &req.job_id)
            .map_err(status_from_extraction)?;
        // Surface progress counters as cumulative metrics. Recording the latest
        // observed totals keeps the counters monotone for terminal jobs and the
        // cache hit-rate gauge fresh without threading a recorder into the
        // manager.
        let total = status.cache_hits + status.cache_misses;
        if total > 0 {
            metrics::set_extraction_cache_hit_rate(
                &req.collection,
                status.cache_hits as f64 / total as f64,
            );
        }
        Ok(Response::new(job_status_to_proto(status)))
    }

    async fn cancel_extraction(
        &self,
        request: Request<CancelExtractionRequest>,
    ) -> Result<Response<CancelExtractionResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        self.state
            .extraction
            .cancel_extraction(&req.collection, &req.job_id)
            .map_err(status_from_extraction)?;
        Ok(Response::new(CancelExtractionResponse { success: true }))
    }

    async fn list_proposals(
        &self,
        request: Request<ListProposalsRequest>,
    ) -> Result<Response<ListProposalsResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let proposals = self
            .state
            .extraction
            .list_proposals(&req.collection)
            .map_err(status_from_extraction)?;
        Ok(Response::new(ListProposalsResponse {
            proposals: proposals.iter().map(proposal_to_proto).collect(),
        }))
    }

    async fn approve_proposal(
        &self,
        request: Request<ApproveProposalRequest>,
    ) -> Result<Response<ApproveProposalResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        self.state
            .extraction
            .approve_proposal(&req.collection, &req.id)
            .map_err(status_from_extraction)?;
        Ok(Response::new(ApproveProposalResponse { success: true }))
    }

    async fn reject_proposal(
        &self,
        request: Request<RejectProposalRequest>,
    ) -> Result<Response<RejectProposalResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        self.state
            .extraction
            .reject_proposal(&req.collection, &req.id)
            .map_err(status_from_extraction)?;
        Ok(Response::new(RejectProposalResponse { success: true }))
    }

    // ── Document-update diff and re-extraction (P04) ──

    async fn diff_document(
        &self,
        request: Request<DiffDocumentRequest>,
    ) -> Result<Response<DiffDocumentResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(chunk_from_proto).collect();
        let diffs = self
            .state
            .extraction
            .diff_document(&req.collection, &req.doc_id, &chunks)
            .map_err(status_from_extraction)?;
        Ok(Response::new(DiffDocumentResponse {
            diffs: diffs
                .into_iter()
                .map(|d| ProtoChunkDiff {
                    chunk_id: d.chunk_id,
                    action: chunk_diff_action_to_proto(d.action) as i32,
                })
                .collect(),
        }))
    }

    async fn reextract_document(
        &self,
        request: Request<ReextractDocumentRequest>,
    ) -> Result<Response<ReextractDocumentResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let chunks: Vec<ChunkContent> = req.chunks.into_iter().map(chunk_from_proto).collect();
        let s = self
            .state
            .extraction
            .reextract_document(&req.collection, &req.doc_id, chunks)
            .map_err(status_from_extraction)?;
        Ok(Response::new(ReextractDocumentResponse {
            job_id: s.job_id,
            unchanged: s.unchanged,
            changed: s.changed,
            added: s.added,
            deleted: s.deleted,
            edges_deleted: s.edges_deleted,
            nodes_deleted: s.nodes_deleted,
        }))
    }

    // ── Corpus-level re-extraction (P16, Area 3) ──

    async fn start_corpus_reextraction(
        &self,
        request: Request<StartCorpusReextractionRequest>,
    ) -> Result<Response<StartCorpusReextractionResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        // Flatten each proto document into (doc_id, chunks) for the manager.
        let documents: Vec<(String, Vec<ChunkContent>)> = req
            .documents
            .into_iter()
            .map(|d| {
                (
                    d.doc_id,
                    d.chunks.into_iter().map(chunk_from_proto).collect(),
                )
            })
            .collect();
        let resume_token = if req.resume_token.trim().is_empty() {
            None
        } else {
            Some(req.resume_token)
        };
        let corpus_job_id = self
            .state
            .extraction
            .start_corpus_reextraction(&req.collection, documents, req.doc_ids, resume_token)
            .map_err(status_from_extraction)?;
        metrics::record_extraction_job("corpus_started");
        Ok(Response::new(StartCorpusReextractionResponse {
            corpus_job_id,
        }))
    }

    async fn get_corpus_reextraction_status(
        &self,
        request: Request<GetCorpusReextractionStatusRequest>,
    ) -> Result<Response<CorpusReextractionStatusMsg>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        let status = self
            .state
            .extraction
            .corpus_job_status(&req.collection, &req.corpus_job_id)
            .map_err(status_from_extraction)?;
        Ok(Response::new(corpus_status_to_proto(status)))
    }

    async fn cancel_corpus_reextraction(
        &self,
        request: Request<CancelCorpusReextractionRequest>,
    ) -> Result<Response<CancelCorpusReextractionResponse>, Status> {
        let req = request.into_inner();
        self.gate(&req.collection)?;
        self.state
            .extraction
            .cancel_corpus_reextraction(&req.collection, &req.corpus_job_id)
            .map_err(status_from_extraction)?;
        Ok(Response::new(CancelCorpusReextractionResponse {
            success: true,
        }))
    }
}

fn chunk_diff_action_to_proto(a: vf_extraction::ChunkDiffAction) -> ProtoChunkDiffAction {
    match a {
        vf_extraction::ChunkDiffAction::Unchanged => ProtoChunkDiffAction::ChunkDiffUnchanged,
        vf_extraction::ChunkDiffAction::Changed => ProtoChunkDiffAction::ChunkDiffChanged,
        vf_extraction::ChunkDiffAction::New => ProtoChunkDiffAction::ChunkDiffNew,
        vf_extraction::ChunkDiffAction::Deleted => ProtoChunkDiffAction::ChunkDiffDeleted,
    }
}

// ── Domain -> proto helpers ──────────────────────────────────────────

fn job_state_str(state: vf_extraction::JobState) -> &'static str {
    match state {
        vf_extraction::JobState::Queued => "queued",
        vf_extraction::JobState::Running => "running",
        vf_extraction::JobState::Completed => "completed",
        vf_extraction::JobState::CompletedWithErrors => "completed_with_errors",
        vf_extraction::JobState::Failed => "failed",
        vf_extraction::JobState::Cancelled => "cancelled",
    }
}

pub(crate) fn job_status_to_proto(s: vf_extraction::JobStatus) -> JobStatusMsg {
    JobStatusMsg {
        job_id: s.job_id,
        collection: s.collection,
        state: job_state_str(s.state).to_string(),
        total_chunks: s.total_chunks as u64,
        processed_chunks: s.processed_chunks as u64,
        entities_written: s.entities_written as u64,
        edges_written: s.edges_written as u64,
        cache_hits: s.cache_hits as u64,
        cache_misses: s.cache_misses as u64,
        error: s.error.unwrap_or_default(),
        failed_chunks: s.failed_chunks as u64,
        chunk_errors: s
            .chunk_errors
            .iter()
            .map(|c| ChunkErrorMsg {
                doc_id: c.doc_id.clone(),
                chunk_id: c.chunk_id,
                error: c.error.clone(),
            })
            .collect(),
        // Area 4. Post-job cost actuals.
        actual_input_tokens: s.actual_input_tokens,
        actual_output_tokens: s.actual_output_tokens,
        actual_cost_usd: s.actual_cost_usd,
        usage_provider_reported: s.usage_provider_reported,
    }
}

fn corpus_doc_state_str(state: vf_extraction::CorpusDocState) -> &'static str {
    match state {
        vf_extraction::CorpusDocState::Pending => "pending",
        vf_extraction::CorpusDocState::Completed => "completed",
        vf_extraction::CorpusDocState::CompletedWithErrors => "completed_with_errors",
        vf_extraction::CorpusDocState::Failed => "failed",
        vf_extraction::CorpusDocState::Skipped => "skipped",
    }
}

pub(crate) fn corpus_status_to_proto(
    s: vf_extraction::CorpusJobStatus,
) -> CorpusReextractionStatusMsg {
    CorpusReextractionStatusMsg {
        corpus_job_id: s.corpus_job_id,
        collection: s.collection,
        state: job_state_str(s.state).to_string(),
        total_documents: s.total_documents as u64,
        processed_documents: s.processed_documents as u64,
        failed_documents: s.failed_documents as u64,
        skipped_documents: s.skipped_documents as u64,
        changed_chunks: s.changed_chunks,
        added_chunks: s.added_chunks,
        deleted_chunks: s.deleted_chunks,
        edges_deleted: s.edges_deleted,
        nodes_deleted: s.nodes_deleted,
        entities_written: s.entities_written,
        edges_written: s.edges_written,
        documents: s
            .documents
            .iter()
            .map(|d| CorpusDocProgressMsg {
                doc_id: d.doc_id.clone(),
                state: corpus_doc_state_str(d.state).to_string(),
                job_id: d.job_id.clone(),
                changed: d.changed,
                added: d.added,
                deleted: d.deleted,
            })
            .collect(),
        error: s.error.unwrap_or_default(),
    }
}

fn proposal_kind_str(kind: vf_extraction::ProposalKind) -> &'static str {
    match kind {
        vf_extraction::ProposalKind::EntityLabel => "entity_label",
        vf_extraction::ProposalKind::EdgeType => "edge_type",
    }
}

fn proposal_status_str(status: vf_extraction::ProposalStatus) -> &'static str {
    match status {
        vf_extraction::ProposalStatus::Pending => "pending",
        vf_extraction::ProposalStatus::Approved => "approved",
        vf_extraction::ProposalStatus::Rejected => "rejected",
    }
}

pub(crate) fn proposal_to_proto(p: &vf_extraction::OntologyProposal) -> ProposalMsg {
    ProposalMsg {
        id: p.id.clone(),
        kind: proposal_kind_str(p.kind).to_string(),
        name: p.name.clone(),
        description: p.description.clone(),
        examples: p.examples.clone(),
        status: proposal_status_str(p.status).to_string(),
        source_doc: p.source_doc.clone().unwrap_or_default(),
        source_chunk_id: p.source_chunk_id.unwrap_or(0),
    }
}
