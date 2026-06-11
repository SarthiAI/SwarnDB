"""SwarnDB LLM extraction operations (Hybrid mode only).

This module wraps the ExtractionService gRPC API: LLM config, ontology,
cost preview, extraction jobs, and ontology proposals. Every RPC is rejected
on non-Hybrid collections by the server.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from ._proto import extraction_pb2
from .types import (
    Chunk,
    ChunkDiff,
    ChunkError,
    CorpusDocProgress,
    CorpusReextractionStatus,
    CostEstimate,
    EdgeType,
    EntityLabel,
    ExtractionJob,
    LlmConfigInfo,
    OntologyInfo,
    OntologyProposal,
    ReextractSummary,
)

if TYPE_CHECKING:
    from .client import SwarnDBClient
    from .async_client import AsyncSwarnDBClient


# A chunk supplied by the caller: a Chunk dataclass or a plain dict.
ChunkLike = Union[Chunk, Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Request builders and response converters (shared by sync and async)
# ---------------------------------------------------------------------------


def _chunk_to_proto(chunk: ChunkLike) -> extraction_pb2.Chunk:
    """Coerce a Chunk dataclass or a dict into a proto Chunk."""
    if isinstance(chunk, Chunk):
        return extraction_pb2.Chunk(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            embedding=list(chunk.embedding),
        )
    if isinstance(chunk, Mapping):
        return extraction_pb2.Chunk(
            doc_id=chunk.get("doc_id", ""),
            chunk_id=int(chunk.get("chunk_id", 0)),
            text=chunk.get("text", ""),
            embedding=list(chunk.get("embedding") or ()),
        )
    raise TypeError(
        "chunk must be a Chunk or a dict with keys "
        "doc_id, chunk_id, text, embedding?; "
        f"got {type(chunk).__name__}"
    )


def _chunks_to_proto(chunks: Sequence[ChunkLike]) -> List[extraction_pb2.Chunk]:
    """Coerce a sequence of chunks into proto Chunks."""
    return [_chunk_to_proto(c) for c in chunks]


# Map the entity-resolution mode string to its proto enum value and back
# (ADR-020). An unknown string falls back to the normalized default.
_RESOLUTION_STR_TO_PROTO = {
    "normalized": extraction_pb2.ENTITY_RESOLUTION_NORMALIZED,
    "fuzzy": extraction_pb2.ENTITY_RESOLUTION_FUZZY,
}
_RESOLUTION_PROTO_TO_STR = {
    extraction_pb2.ENTITY_RESOLUTION_NORMALIZED: "normalized",
    extraction_pb2.ENTITY_RESOLUTION_FUZZY: "fuzzy",
}


def _resolution_to_proto(mode: str) -> int:
    """Coerce a resolution-mode string to its proto enum value."""
    return _RESOLUTION_STR_TO_PROTO.get(
        (mode or "normalized").lower(),
        extraction_pb2.ENTITY_RESOLUTION_NORMALIZED,
    )


# Map the ChunkDiffAction proto enum value to its string form.
_CHUNK_DIFF_ACTION_TO_STR = {
    extraction_pb2.CHUNK_DIFF_UNCHANGED: "unchanged",
    extraction_pb2.CHUNK_DIFF_CHANGED: "changed",
    extraction_pb2.CHUNK_DIFF_NEW: "new",
    extraction_pb2.CHUNK_DIFF_DELETED: "deleted",
}


def _chunk_diff_from_proto(d: Any) -> ChunkDiff:
    """Convert a proto ChunkDiff into a ChunkDiff dataclass."""
    return ChunkDiff(
        chunk_id=int(d.chunk_id),
        action=_CHUNK_DIFF_ACTION_TO_STR.get(d.action, "unchanged"),
    )


def _reextract_from_proto(r: Any) -> ReextractSummary:
    """Convert a ReextractDocumentResponse into a ReextractSummary."""
    return ReextractSummary(
        job_id=r.job_id,
        unchanged=int(r.unchanged),
        changed=int(r.changed),
        added=int(r.added),
        deleted=int(r.deleted),
        edges_deleted=int(r.edges_deleted),
        nodes_deleted=int(r.nodes_deleted),
    )


def _llm_config_from_proto(resp: Any) -> LlmConfigInfo:
    """Convert a GetLlmConfigResponse into an LlmConfigInfo."""
    return LlmConfigInfo(
        base_url=resp.base_url,
        model_name=resp.model_name,
        temperature=resp.temperature,
        max_tokens=int(resp.max_tokens),
        timeout_seconds=int(resp.timeout_seconds),
        api_key_set=bool(resp.api_key_set),
    )


def _ontology_from_proto(ontology: Any) -> OntologyInfo:
    """Convert an OntologyMsg into an OntologyInfo."""
    return OntologyInfo(
        entity_labels=[
            EntityLabel(label=e.label, description=e.description)
            for e in ontology.entity_labels
        ],
        edge_types=[
            EdgeType(
                edge_type=t.edge_type,
                description=t.description,
                source_labels=list(t.source_labels),
                target_labels=list(t.target_labels),
            )
            for t in ontology.edge_types
        ],
        # Empty string -> None; empty means the default prompt is used.
        system_prompt=getattr(ontology, "system_prompt", "") or None,
        extra_guidance=getattr(ontology, "extra_guidance", "") or None,
        link_passages=bool(getattr(ontology, "link_passages", False)),
        entity_resolution=_RESOLUTION_PROTO_TO_STR.get(
            getattr(ontology, "entity_resolution", 0), "normalized"
        ),
    )


def _ontology_extension_to_proto(
    entity_labels: Optional[Sequence[EntityLabel]],
    edge_types: Optional[Sequence[EdgeType]],
    system_prompt: Optional[str] = None,
    extra_guidance: Optional[str] = None,
    link_passages: bool = False,
    entity_resolution: str = "normalized",
) -> extraction_pb2.OntologyMsg:
    """Build an OntologyMsg from entity-label and edge-type sequences."""
    return extraction_pb2.OntologyMsg(
        entity_labels=[
            extraction_pb2.OntologyEntityLabel(
                label=e.label,
                description=e.description,
            )
            for e in (entity_labels or ())
        ],
        edge_types=[
            extraction_pb2.OntologyEdgeType(
                edge_type=t.edge_type,
                description=t.description,
                source_labels=list(t.source_labels),
                target_labels=list(t.target_labels),
            )
            for t in (edge_types or ())
        ],
        # Empty means the default prompt is used.
        system_prompt=system_prompt or "",
        extra_guidance=extra_guidance or "",
        link_passages=link_passages,
        entity_resolution=_resolution_to_proto(entity_resolution),
    )


# Per-batch payload budget for auto-batched extraction submission (ADR-013).
# Kept well under the 128 MB server/client cap so one batch never trips a limit
# even with protobuf and gRPC framing overhead.
_EXTRACTION_BATCH_MAX_BYTES = 32 * 1024 * 1024
# Hard cap on chunks per batch as a second guard, independent of size.
_EXTRACTION_BATCH_MAX_CHUNKS = 2000


def _batch_chunks(
    protos: List[extraction_pb2.Chunk],
) -> List[List[extraction_pb2.Chunk]]:
    """Split proto chunks into size-bounded batches (ADR-013).

    Each batch stays under both the byte budget and the chunk-count cap. A
    single chunk larger than the byte budget still ships alone (it cannot be
    split further), relying on the raised server cap to accept it.
    """
    batches: List[List[extraction_pb2.Chunk]] = []
    current: List[extraction_pb2.Chunk] = []
    current_bytes = 0
    for proto in protos:
        size = proto.ByteSize()
        would_exceed = current and (
            current_bytes + size > _EXTRACTION_BATCH_MAX_BYTES
            or len(current) >= _EXTRACTION_BATCH_MAX_CHUNKS
        )
        if would_exceed:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(proto)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def _cost_from_proto(resp: Any) -> CostEstimate:
    """Convert a CostPreviewResponse into a CostEstimate."""
    return CostEstimate(
        chunks=int(resp.chunks),
        estimated_input_tokens=int(resp.estimated_input_tokens),
        estimated_output_tokens=int(resp.estimated_output_tokens),
        estimated_cost_usd=resp.estimated_cost_usd,
        model=resp.model,
        pricing_known=bool(resp.pricing_known),
    )


def _job_from_proto(msg: Any) -> ExtractionJob:
    """Convert a JobStatusMsg into an ExtractionJob."""
    return ExtractionJob(
        job_id=msg.job_id,
        collection=msg.collection,
        state=msg.state,
        total_chunks=int(msg.total_chunks),
        processed_chunks=int(msg.processed_chunks),
        entities_written=int(msg.entities_written),
        edges_written=int(msg.edges_written),
        cache_hits=int(msg.cache_hits),
        cache_misses=int(msg.cache_misses),
        error=msg.error,
        failed_chunks=int(getattr(msg, "failed_chunks", 0)),
        chunk_errors=[
            ChunkError(doc_id=e.doc_id, chunk_id=int(e.chunk_id), error=e.error)
            for e in getattr(msg, "chunk_errors", [])
        ],
        # Area 4: post-job cost actuals. getattr keeps old stubs working.
        actual_input_tokens=int(getattr(msg, "actual_input_tokens", 0)),
        actual_output_tokens=int(getattr(msg, "actual_output_tokens", 0)),
        actual_cost_usd=float(getattr(msg, "actual_cost_usd", 0.0)),
        usage_provider_reported=bool(
            getattr(msg, "usage_provider_reported", True)
        ),
    )


def _corpus_doc_from_proto(d: Any) -> CorpusDocProgress:
    """Convert a CorpusDocProgressMsg into a CorpusDocProgress dataclass."""
    return CorpusDocProgress(
        doc_id=d.doc_id,
        state=d.state,
        job_id=d.job_id,
        changed=int(d.changed),
        added=int(d.added),
        deleted=int(d.deleted),
    )


def _corpus_status_from_proto(msg: Any) -> CorpusReextractionStatus:
    """Convert a CorpusReextractionStatusMsg into a CorpusReextractionStatus."""
    return CorpusReextractionStatus(
        corpus_job_id=msg.corpus_job_id,
        collection=msg.collection,
        state=msg.state,
        total_documents=int(msg.total_documents),
        processed_documents=int(msg.processed_documents),
        failed_documents=int(msg.failed_documents),
        skipped_documents=int(msg.skipped_documents),
        changed_chunks=int(msg.changed_chunks),
        added_chunks=int(msg.added_chunks),
        deleted_chunks=int(msg.deleted_chunks),
        edges_deleted=int(msg.edges_deleted),
        nodes_deleted=int(msg.nodes_deleted),
        entities_written=int(msg.entities_written),
        edges_written=int(msg.edges_written),
        documents=[_corpus_doc_from_proto(d) for d in msg.documents],
        error=msg.error,
    )


# A corpus document supplied by the caller: a (doc_id, chunks) pair where chunks
# is a sequence of Chunk dataclasses or dicts. A mapping with "doc_id"/"chunks"
# keys is also accepted.
CorpusDocLike = Union[
    "tuple[str, Sequence[ChunkLike]]",
    Mapping[str, Any],
]


def _corpus_document_to_proto(doc: Any) -> extraction_pb2.CorpusDocument:
    """Coerce a (doc_id, chunks) pair or a dict into a proto CorpusDocument."""
    if isinstance(doc, Mapping):
        doc_id = doc.get("doc_id", "")
        chunks = doc.get("chunks") or ()
    else:
        # Assume a (doc_id, chunks) tuple / sequence pair.
        doc_id, chunks = doc[0], doc[1]
    return extraction_pb2.CorpusDocument(
        doc_id=doc_id,
        chunks=_chunks_to_proto(list(chunks)),
    )


def _proposal_from_proto(msg: Any) -> OntologyProposal:
    """Convert a ProposalMsg into an OntologyProposal."""
    return OntologyProposal(
        id=msg.id,
        kind=msg.kind,
        name=msg.name,
        description=msg.description,
        examples=list(msg.examples),
        status=msg.status,
        source_doc=msg.source_doc,
        source_chunk_id=int(msg.source_chunk_id),
    )


# ---------------------------------------------------------------------------
# ExtractionAPI (sync)
# ---------------------------------------------------------------------------


class ExtractionAPI:
    """Pythonic wrapper around the ExtractionService gRPC API (Hybrid mode)."""

    def __init__(self, client: "SwarnDBClient") -> None:
        self._client = client

    # ── LLM config ──

    def set_llm_config(
        self,
        collection: str,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: int = 30,
    ) -> bool:
        """Set the LLM config for a collection. The api key is write-only."""
        request = extraction_pb2.SetLlmConfigRequest(
            collection=collection,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        response = self._client._call(
            self._client._extraction_stub.SetLlmConfig, request
        )
        return bool(response.success)

    def get_llm_config(self, collection: str) -> LlmConfigInfo:
        """Get the redacted LLM config (never returns the api key)."""
        request = extraction_pb2.GetLlmConfigRequest(collection=collection)
        response = self._client._call(
            self._client._extraction_stub.GetLlmConfig, request
        )
        return _llm_config_from_proto(response)

    def rotate_llm_config(self, collection: str, new_api_key: str) -> bool:
        """Rotate just the api key, keeping the rest of the config."""
        request = extraction_pb2.RotateLlmConfigRequest(
            collection=collection,
            new_api_key=new_api_key,
        )
        response = self._client._call(
            self._client._extraction_stub.RotateLlmConfig, request
        )
        return bool(response.success)

    # ── Ontology ──

    def set_ontology(
        self,
        collection: str,
        *,
        base_template: Optional[str] = None,
        entity_labels: Optional[Sequence[EntityLabel]] = None,
        edge_types: Optional[Sequence[EdgeType]] = None,
        replace: bool = False,
        system_prompt: Optional[str] = None,
        extra_guidance: Optional[str] = None,
        link_passages: bool = False,
        entity_resolution: str = "normalized",
    ) -> bool:
        """Set the ontology from a template and/or an inline extension.

        ``base_template`` is a kebab-case template name (e.g.
        "research-papers"); when ``replace`` is True the extension replaces
        the template entirely. ``system_prompt`` fully overrides the generic
        task framing and ``extra_guidance`` is appended on top; an empty value
        for either means the default prompt is used. ``link_passages`` opts in
        to passage-to-entity links so GraphRAG works without hand-authoring the
        @chunk relation.

        ``entity_resolution`` selects how a new entity name is matched against
        existing nodes of the same label when deduping (ADR-020):

        - ``"normalized"`` (default): exact match after deterministic
          normalization (trim, NFKC, lowercase, quote / trailing-punctuation
          strip, whitespace collapse). This is the historical behavior, so
          existing collections are unchanged.
        - ``"fuzzy"``: opt-in conservative deterministic resolution that also
          merges alias / abbreviation variants ("J. Smith" with "John Smith"),
          acronym / expansion pairs ("USA" with "United States America"), bounded
          typos, and strict title-prefixed supersets of the same head entity
          ("Barack Obama" with "President Barack Obama"). Merged surface forms are
          recorded on the canonical node's ``aliases`` for provenance.

        Caveat: fuzzy resolution trades recall for a small risk of OVER-MERGING
        two genuinely distinct entities, which corrupts the graph. The rules are
        deliberately conservative (when in doubt they do not merge), but enable it
        only when alias / abbreviation variants of the same entity are common in
        your data; otherwise keep the default.
        """
        request = extraction_pb2.SetOntologyRequest(
            collection=collection,
            base_template=base_template or "",
            extension=_ontology_extension_to_proto(
                entity_labels,
                edge_types,
                system_prompt,
                extra_guidance,
                link_passages,
                entity_resolution,
            ),
            replace=replace,
        )
        response = self._client._call(
            self._client._extraction_stub.SetOntology, request
        )
        return bool(response.success)

    def get_ontology(self, collection: str) -> OntologyInfo:
        """Get the collection's ontology."""
        request = extraction_pb2.GetOntologyRequest(collection=collection)
        response = self._client._call(
            self._client._extraction_stub.GetOntology, request
        )
        return _ontology_from_proto(response.ontology)

    # ── Cost preview and extraction ──

    def cost_preview(
        self,
        collection: str,
        chunks: Sequence[ChunkLike],
    ) -> CostEstimate:
        """Estimate token usage and cost for extracting the given chunks.

        Large inputs are split client-side into size-bounded batches and the
        per-batch estimates are summed, so a big corpus never trips the message
        size cap. The returned estimate is the aggregate (ADR-013).
        """
        protos = _chunks_to_proto(chunks)
        batches = _batch_chunks(protos)
        if not batches:
            batches = [[]]

        total_chunks = 0
        total_in = 0
        total_out = 0
        total_cost = 0.0
        model = ""
        pricing_known = True
        for batch in batches:
            request = extraction_pb2.CostPreviewRequest(
                collection=collection,
                chunks=batch,
            )
            response = self._client._call(
                self._client._extraction_stub.CostPreview, request
            )
            part = _cost_from_proto(response)
            total_chunks += part.chunks
            total_in += part.estimated_input_tokens
            total_out += part.estimated_output_tokens
            total_cost += part.estimated_cost_usd
            model = part.model or model
            pricing_known = pricing_known and part.pricing_known
        return CostEstimate(
            chunks=total_chunks,
            estimated_input_tokens=total_in,
            estimated_output_tokens=total_out,
            estimated_cost_usd=total_cost,
            model=model,
            pricing_known=pricing_known,
        )

    def start_extraction(
        self,
        collection: str,
        chunks: Sequence[ChunkLike],
    ) -> str:
        """Start an async extraction job. Returns the job id.

        Large inputs are split client-side into size-bounded batches (ADR-013):
        the first batch creates the job via StartExtraction, the rest are
        appended via AppendExtractionChunks, and the final batch seals the job.
        One logical job spans the whole corpus, so progress and partial-success
        tracking cover the full submission. The public signature is unchanged.
        """
        protos = _chunks_to_proto(chunks)
        batches = _batch_chunks(protos)
        if not batches:
            batches = [[]]

        first = batches[0]
        request = extraction_pb2.StartExtractionRequest(
            collection=collection,
            chunks=first,
        )
        response = self._client._call(
            self._client._extraction_stub.StartExtraction, request
        )
        job_id = response.job_id

        last_index = len(batches) - 1
        for i in range(1, len(batches)):
            append_req = extraction_pb2.AppendExtractionChunksRequest(
                collection=collection,
                job_id=job_id,
                chunks=batches[i],
                last_batch=(i == last_index),
            )
            self._client._call(
                self._client._extraction_stub.AppendExtractionChunks, append_req
            )
        return job_id

    def extraction_status(self, collection: str, job_id: str) -> ExtractionJob:
        """Get a snapshot of an extraction job's progress."""
        request = extraction_pb2.GetExtractionStatusRequest(
            collection=collection,
            job_id=job_id,
        )
        response = self._client._call(
            self._client._extraction_stub.GetExtractionStatus, request
        )
        return _job_from_proto(response)

    def cancel_extraction(self, collection: str, job_id: str) -> bool:
        """Cancel a running extraction job. True on success."""
        request = extraction_pb2.CancelExtractionRequest(
            collection=collection,
            job_id=job_id,
        )
        response = self._client._call(
            self._client._extraction_stub.CancelExtraction, request
        )
        return bool(response.success)

    # ── Proposals ──

    def list_proposals(self, collection: str) -> List[OntologyProposal]:
        """List pending ontology proposals for a collection."""
        request = extraction_pb2.ListProposalsRequest(collection=collection)
        response = self._client._call(
            self._client._extraction_stub.ListProposals, request
        )
        return [_proposal_from_proto(p) for p in response.proposals]

    def approve_proposal(self, collection: str, proposal_id: str) -> bool:
        """Approve an ontology proposal. True on success."""
        request = extraction_pb2.ApproveProposalRequest(
            collection=collection,
            id=proposal_id,
        )
        response = self._client._call(
            self._client._extraction_stub.ApproveProposal, request
        )
        return bool(response.success)

    def reject_proposal(self, collection: str, proposal_id: str) -> bool:
        """Reject an ontology proposal. True on success."""
        request = extraction_pb2.RejectProposalRequest(
            collection=collection,
            id=proposal_id,
        )
        response = self._client._call(
            self._client._extraction_stub.RejectProposal, request
        )
        return bool(response.success)

    # ── Document diff and re-extraction ──

    def diff_document(
        self,
        collection: str,
        doc_id: str,
        chunks: Sequence[ChunkLike],
    ) -> List[ChunkDiff]:
        """Diff a document's chunks against the stored extraction state."""
        request = extraction_pb2.DiffDocumentRequest(
            collection=collection,
            doc_id=doc_id,
            chunks=_chunks_to_proto(chunks),
        )
        response = self._client._call(
            self._client._extraction_stub.DiffDocument, request
        )
        return [_chunk_diff_from_proto(d) for d in response.diffs]

    def reextract_document(
        self,
        collection: str,
        doc_id: str,
        chunks: Sequence[ChunkLike],
    ) -> ReextractSummary:
        """Re-extract a document, processing only its changed chunks."""
        request = extraction_pb2.ReextractDocumentRequest(
            collection=collection,
            doc_id=doc_id,
            chunks=_chunks_to_proto(chunks),
        )
        response = self._client._call(
            self._client._extraction_stub.ReextractDocument, request
        )
        return _reextract_from_proto(response)

    # ── Corpus-level re-extraction ──

    def start_corpus_reextraction(
        self,
        collection: str,
        documents: Sequence[CorpusDocLike],
        *,
        doc_ids: Optional[Sequence[str]] = None,
        resume_token: Optional[str] = None,
    ) -> str:
        """Start a corpus re-extraction over a set of documents.

        ``documents`` is a sequence of ``(doc_id, chunks)`` pairs (or dicts with
        ``doc_id``/``chunks``); each document is driven through the same
        per-document re-extract path. ``doc_ids``, when given, restricts the run
        to those ids among ``documents``; empty means every supplied document. A
        document supplied with no chunks is treated as a full deletion.

        Pass the prior run's corpus job id as ``resume_token`` to continue it:
        documents already completed are skipped. Returns the corpus job id (which
        is also the resume token). The run proceeds in the background; poll
        ``corpus_reextraction_status`` for progress.
        """
        request = extraction_pb2.StartCorpusReextractionRequest(
            collection=collection,
            documents=[_corpus_document_to_proto(d) for d in documents],
            doc_ids=list(doc_ids or ()),
            resume_token=resume_token or "",
        )
        response = self._client._call(
            self._client._extraction_stub.StartCorpusReextraction, request
        )
        return response.corpus_job_id

    def corpus_reextraction_status(
        self, collection: str, corpus_job_id: str
    ) -> CorpusReextractionStatus:
        """Get a snapshot of a corpus re-extraction job's progress."""
        request = extraction_pb2.GetCorpusReextractionStatusRequest(
            collection=collection,
            corpus_job_id=corpus_job_id,
        )
        response = self._client._call(
            self._client._extraction_stub.GetCorpusReextractionStatus, request
        )
        return _corpus_status_from_proto(response)

    def cancel_corpus_reextraction(
        self, collection: str, corpus_job_id: str
    ) -> bool:
        """Cancel a running corpus re-extraction job. True on success."""
        request = extraction_pb2.CancelCorpusReextractionRequest(
            collection=collection,
            corpus_job_id=corpus_job_id,
        )
        response = self._client._call(
            self._client._extraction_stub.CancelCorpusReextraction, request
        )
        return bool(response.success)


# ---------------------------------------------------------------------------
# AsyncExtractionAPI
# ---------------------------------------------------------------------------


class AsyncExtractionAPI:
    """Async wrapper around the ExtractionService gRPC API (Hybrid mode)."""

    def __init__(self, client: "AsyncSwarnDBClient") -> None:
        self._client = client

    # ── LLM config ──

    async def set_llm_config(
        self,
        collection: str,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: int = 30,
    ) -> bool:
        """Set the LLM config for a collection. The api key is write-only."""
        request = extraction_pb2.SetLlmConfigRequest(
            collection=collection,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        response = await self._client._call(
            self._client._extraction_stub.SetLlmConfig, request
        )
        return bool(response.success)

    async def get_llm_config(self, collection: str) -> LlmConfigInfo:
        """Get the redacted LLM config (never returns the api key)."""
        request = extraction_pb2.GetLlmConfigRequest(collection=collection)
        response = await self._client._call(
            self._client._extraction_stub.GetLlmConfig, request
        )
        return _llm_config_from_proto(response)

    async def rotate_llm_config(
        self, collection: str, new_api_key: str
    ) -> bool:
        """Rotate just the api key, keeping the rest of the config."""
        request = extraction_pb2.RotateLlmConfigRequest(
            collection=collection,
            new_api_key=new_api_key,
        )
        response = await self._client._call(
            self._client._extraction_stub.RotateLlmConfig, request
        )
        return bool(response.success)

    # ── Ontology ──

    async def set_ontology(
        self,
        collection: str,
        *,
        base_template: Optional[str] = None,
        entity_labels: Optional[Sequence[EntityLabel]] = None,
        edge_types: Optional[Sequence[EdgeType]] = None,
        replace: bool = False,
        system_prompt: Optional[str] = None,
        extra_guidance: Optional[str] = None,
        link_passages: bool = False,
        entity_resolution: str = "normalized",
    ) -> bool:
        """Set the ontology from a template and/or an inline extension.

        ``base_template`` is a kebab-case template name (e.g.
        "research-papers"); when ``replace`` is True the extension replaces
        the template entirely. ``system_prompt`` fully overrides the generic
        task framing and ``extra_guidance`` is appended on top; an empty value
        for either means the default prompt is used. ``link_passages`` opts in
        to passage-to-entity links so GraphRAG works without hand-authoring the
        @chunk relation.

        ``entity_resolution`` selects how a new entity name is matched against
        existing nodes of the same label when deduping (ADR-020):

        - ``"normalized"`` (default): exact match after deterministic
          normalization (trim, NFKC, lowercase, quote / trailing-punctuation
          strip, whitespace collapse). This is the historical behavior, so
          existing collections are unchanged.
        - ``"fuzzy"``: opt-in conservative deterministic resolution that also
          merges alias / abbreviation variants ("J. Smith" with "John Smith"),
          acronym / expansion pairs ("USA" with "United States America"), bounded
          typos, and strict title-prefixed supersets of the same head entity
          ("Barack Obama" with "President Barack Obama"). Merged surface forms are
          recorded on the canonical node's ``aliases`` for provenance.

        Caveat: fuzzy resolution trades recall for a small risk of OVER-MERGING
        two genuinely distinct entities, which corrupts the graph. The rules are
        deliberately conservative (when in doubt they do not merge), but enable it
        only when alias / abbreviation variants of the same entity are common in
        your data; otherwise keep the default.
        """
        request = extraction_pb2.SetOntologyRequest(
            collection=collection,
            base_template=base_template or "",
            extension=_ontology_extension_to_proto(
                entity_labels,
                edge_types,
                system_prompt,
                extra_guidance,
                link_passages,
                entity_resolution,
            ),
            replace=replace,
        )
        response = await self._client._call(
            self._client._extraction_stub.SetOntology, request
        )
        return bool(response.success)

    async def get_ontology(self, collection: str) -> OntologyInfo:
        """Get the collection's ontology."""
        request = extraction_pb2.GetOntologyRequest(collection=collection)
        response = await self._client._call(
            self._client._extraction_stub.GetOntology, request
        )
        return _ontology_from_proto(response.ontology)

    # ── Cost preview and extraction ──

    async def cost_preview(
        self,
        collection: str,
        chunks: Sequence[ChunkLike],
    ) -> CostEstimate:
        """Estimate token usage and cost for extracting the given chunks.

        Large inputs are split client-side into size-bounded batches and the
        per-batch estimates are summed, so a big corpus never trips the message
        size cap. The returned estimate is the aggregate (ADR-013).
        """
        protos = _chunks_to_proto(chunks)
        batches = _batch_chunks(protos)
        if not batches:
            batches = [[]]

        total_chunks = 0
        total_in = 0
        total_out = 0
        total_cost = 0.0
        model = ""
        pricing_known = True
        for batch in batches:
            request = extraction_pb2.CostPreviewRequest(
                collection=collection,
                chunks=batch,
            )
            response = await self._client._call(
                self._client._extraction_stub.CostPreview, request
            )
            part = _cost_from_proto(response)
            total_chunks += part.chunks
            total_in += part.estimated_input_tokens
            total_out += part.estimated_output_tokens
            total_cost += part.estimated_cost_usd
            model = part.model or model
            pricing_known = pricing_known and part.pricing_known
        return CostEstimate(
            chunks=total_chunks,
            estimated_input_tokens=total_in,
            estimated_output_tokens=total_out,
            estimated_cost_usd=total_cost,
            model=model,
            pricing_known=pricing_known,
        )

    async def start_extraction(
        self,
        collection: str,
        chunks: Sequence[ChunkLike],
    ) -> str:
        """Start an async extraction job. Returns the job id.

        Large inputs are split client-side into size-bounded batches (ADR-013):
        the first batch creates the job via StartExtraction, the rest are
        appended via AppendExtractionChunks, and the final batch seals the job.
        One logical job spans the whole corpus, so progress and partial-success
        tracking cover the full submission. The public signature is unchanged.
        """
        protos = _chunks_to_proto(chunks)
        batches = _batch_chunks(protos)
        if not batches:
            batches = [[]]

        first = batches[0]
        request = extraction_pb2.StartExtractionRequest(
            collection=collection,
            chunks=first,
        )
        response = await self._client._call(
            self._client._extraction_stub.StartExtraction, request
        )
        job_id = response.job_id

        last_index = len(batches) - 1
        for i in range(1, len(batches)):
            append_req = extraction_pb2.AppendExtractionChunksRequest(
                collection=collection,
                job_id=job_id,
                chunks=batches[i],
                last_batch=(i == last_index),
            )
            await self._client._call(
                self._client._extraction_stub.AppendExtractionChunks, append_req
            )
        return job_id

    async def extraction_status(
        self, collection: str, job_id: str
    ) -> ExtractionJob:
        """Get a snapshot of an extraction job's progress."""
        request = extraction_pb2.GetExtractionStatusRequest(
            collection=collection,
            job_id=job_id,
        )
        response = await self._client._call(
            self._client._extraction_stub.GetExtractionStatus, request
        )
        return _job_from_proto(response)

    async def cancel_extraction(self, collection: str, job_id: str) -> bool:
        """Cancel a running extraction job. True on success."""
        request = extraction_pb2.CancelExtractionRequest(
            collection=collection,
            job_id=job_id,
        )
        response = await self._client._call(
            self._client._extraction_stub.CancelExtraction, request
        )
        return bool(response.success)

    # ── Proposals ──

    async def list_proposals(self, collection: str) -> List[OntologyProposal]:
        """List pending ontology proposals for a collection."""
        request = extraction_pb2.ListProposalsRequest(collection=collection)
        response = await self._client._call(
            self._client._extraction_stub.ListProposals, request
        )
        return [_proposal_from_proto(p) for p in response.proposals]

    async def approve_proposal(
        self, collection: str, proposal_id: str
    ) -> bool:
        """Approve an ontology proposal. True on success."""
        request = extraction_pb2.ApproveProposalRequest(
            collection=collection,
            id=proposal_id,
        )
        response = await self._client._call(
            self._client._extraction_stub.ApproveProposal, request
        )
        return bool(response.success)

    async def reject_proposal(
        self, collection: str, proposal_id: str
    ) -> bool:
        """Reject an ontology proposal. True on success."""
        request = extraction_pb2.RejectProposalRequest(
            collection=collection,
            id=proposal_id,
        )
        response = await self._client._call(
            self._client._extraction_stub.RejectProposal, request
        )
        return bool(response.success)

    # ── Document diff and re-extraction ──

    async def diff_document(
        self,
        collection: str,
        doc_id: str,
        chunks: Sequence[ChunkLike],
    ) -> List[ChunkDiff]:
        """Diff a document's chunks against the stored extraction state."""
        request = extraction_pb2.DiffDocumentRequest(
            collection=collection,
            doc_id=doc_id,
            chunks=_chunks_to_proto(chunks),
        )
        response = await self._client._call(
            self._client._extraction_stub.DiffDocument, request
        )
        return [_chunk_diff_from_proto(d) for d in response.diffs]

    async def reextract_document(
        self,
        collection: str,
        doc_id: str,
        chunks: Sequence[ChunkLike],
    ) -> ReextractSummary:
        """Re-extract a document, processing only its changed chunks."""
        request = extraction_pb2.ReextractDocumentRequest(
            collection=collection,
            doc_id=doc_id,
            chunks=_chunks_to_proto(chunks),
        )
        response = await self._client._call(
            self._client._extraction_stub.ReextractDocument, request
        )
        return _reextract_from_proto(response)

    # ── Corpus-level re-extraction ──

    async def start_corpus_reextraction(
        self,
        collection: str,
        documents: Sequence[CorpusDocLike],
        *,
        doc_ids: Optional[Sequence[str]] = None,
        resume_token: Optional[str] = None,
    ) -> str:
        """Start a corpus re-extraction over a set of documents.

        ``documents`` is a sequence of ``(doc_id, chunks)`` pairs (or dicts with
        ``doc_id``/``chunks``); each document is driven through the same
        per-document re-extract path. ``doc_ids``, when given, restricts the run
        to those ids among ``documents``; empty means every supplied document. A
        document supplied with no chunks is treated as a full deletion.

        Pass the prior run's corpus job id as ``resume_token`` to continue it:
        documents already completed are skipped. Returns the corpus job id (which
        is also the resume token). The run proceeds in the background; poll
        ``corpus_reextraction_status`` for progress.
        """
        request = extraction_pb2.StartCorpusReextractionRequest(
            collection=collection,
            documents=[_corpus_document_to_proto(d) for d in documents],
            doc_ids=list(doc_ids or ()),
            resume_token=resume_token or "",
        )
        response = await self._client._call(
            self._client._extraction_stub.StartCorpusReextraction, request
        )
        return response.corpus_job_id

    async def corpus_reextraction_status(
        self, collection: str, corpus_job_id: str
    ) -> CorpusReextractionStatus:
        """Get a snapshot of a corpus re-extraction job's progress."""
        request = extraction_pb2.GetCorpusReextractionStatusRequest(
            collection=collection,
            corpus_job_id=corpus_job_id,
        )
        response = await self._client._call(
            self._client._extraction_stub.GetCorpusReextractionStatus, request
        )
        return _corpus_status_from_proto(response)

    async def cancel_corpus_reextraction(
        self, collection: str, corpus_job_id: str
    ) -> bool:
        """Cancel a running corpus re-extraction job. True on success."""
        request = extraction_pb2.CancelCorpusReextractionRequest(
            collection=collection,
            corpus_job_id=corpus_job_id,
        )
        response = await self._client._call(
            self._client._extraction_stub.CancelCorpusReextraction, request
        )
        return bool(response.success)
