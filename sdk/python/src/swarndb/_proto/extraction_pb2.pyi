from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EntityResolutionMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENTITY_RESOLUTION_NORMALIZED: _ClassVar[EntityResolutionMode]
    ENTITY_RESOLUTION_FUZZY: _ClassVar[EntityResolutionMode]

class ChunkDiffAction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHUNK_DIFF_UNCHANGED: _ClassVar[ChunkDiffAction]
    CHUNK_DIFF_CHANGED: _ClassVar[ChunkDiffAction]
    CHUNK_DIFF_NEW: _ClassVar[ChunkDiffAction]
    CHUNK_DIFF_DELETED: _ClassVar[ChunkDiffAction]
ENTITY_RESOLUTION_NORMALIZED: EntityResolutionMode
ENTITY_RESOLUTION_FUZZY: EntityResolutionMode
CHUNK_DIFF_UNCHANGED: ChunkDiffAction
CHUNK_DIFF_CHANGED: ChunkDiffAction
CHUNK_DIFF_NEW: ChunkDiffAction
CHUNK_DIFF_DELETED: ChunkDiffAction

class Chunk(_message.Message):
    __slots__ = ("doc_id", "chunk_id", "text", "embedding")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    chunk_id: int
    text: str
    embedding: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, doc_id: _Optional[str] = ..., chunk_id: _Optional[int] = ..., text: _Optional[str] = ..., embedding: _Optional[_Iterable[float]] = ...) -> None: ...

class SetLlmConfigRequest(_message.Message):
    __slots__ = ("collection", "base_url", "api_key", "model_name", "temperature", "max_tokens", "timeout_seconds")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    BASE_URL_FIELD_NUMBER: _ClassVar[int]
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    base_url: str
    api_key: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    def __init__(self, collection: _Optional[str] = ..., base_url: _Optional[str] = ..., api_key: _Optional[str] = ..., model_name: _Optional[str] = ..., temperature: _Optional[float] = ..., max_tokens: _Optional[int] = ..., timeout_seconds: _Optional[int] = ...) -> None: ...

class SetLlmConfigResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class GetLlmConfigRequest(_message.Message):
    __slots__ = ("collection",)
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    collection: str
    def __init__(self, collection: _Optional[str] = ...) -> None: ...

class GetLlmConfigResponse(_message.Message):
    __slots__ = ("base_url", "model_name", "temperature", "max_tokens", "timeout_seconds", "api_key_set")
    BASE_URL_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    API_KEY_SET_FIELD_NUMBER: _ClassVar[int]
    base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    api_key_set: bool
    def __init__(self, base_url: _Optional[str] = ..., model_name: _Optional[str] = ..., temperature: _Optional[float] = ..., max_tokens: _Optional[int] = ..., timeout_seconds: _Optional[int] = ..., api_key_set: bool = ...) -> None: ...

class RotateLlmConfigRequest(_message.Message):
    __slots__ = ("collection", "new_api_key")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    NEW_API_KEY_FIELD_NUMBER: _ClassVar[int]
    collection: str
    new_api_key: str
    def __init__(self, collection: _Optional[str] = ..., new_api_key: _Optional[str] = ...) -> None: ...

class RotateLlmConfigResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class OntologyEntityLabel(_message.Message):
    __slots__ = ("label", "description")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    label: str
    description: str
    def __init__(self, label: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class OntologyEdgeType(_message.Message):
    __slots__ = ("edge_type", "description", "source_labels", "target_labels")
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LABELS_FIELD_NUMBER: _ClassVar[int]
    TARGET_LABELS_FIELD_NUMBER: _ClassVar[int]
    edge_type: str
    description: str
    source_labels: _containers.RepeatedScalarFieldContainer[str]
    target_labels: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, edge_type: _Optional[str] = ..., description: _Optional[str] = ..., source_labels: _Optional[_Iterable[str]] = ..., target_labels: _Optional[_Iterable[str]] = ...) -> None: ...

class OntologyMsg(_message.Message):
    __slots__ = ("entity_labels", "edge_types", "system_prompt", "extra_guidance", "link_passages", "entity_resolution")
    ENTITY_LABELS_FIELD_NUMBER: _ClassVar[int]
    EDGE_TYPES_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    EXTRA_GUIDANCE_FIELD_NUMBER: _ClassVar[int]
    LINK_PASSAGES_FIELD_NUMBER: _ClassVar[int]
    ENTITY_RESOLUTION_FIELD_NUMBER: _ClassVar[int]
    entity_labels: _containers.RepeatedCompositeFieldContainer[OntologyEntityLabel]
    edge_types: _containers.RepeatedCompositeFieldContainer[OntologyEdgeType]
    system_prompt: str
    extra_guidance: str
    link_passages: bool
    entity_resolution: EntityResolutionMode
    def __init__(self, entity_labels: _Optional[_Iterable[_Union[OntologyEntityLabel, _Mapping]]] = ..., edge_types: _Optional[_Iterable[_Union[OntologyEdgeType, _Mapping]]] = ..., system_prompt: _Optional[str] = ..., extra_guidance: _Optional[str] = ..., link_passages: bool = ..., entity_resolution: _Optional[_Union[EntityResolutionMode, str]] = ...) -> None: ...

class SetOntologyRequest(_message.Message):
    __slots__ = ("collection", "base_template", "extension", "replace")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    BASE_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    EXTENSION_FIELD_NUMBER: _ClassVar[int]
    REPLACE_FIELD_NUMBER: _ClassVar[int]
    collection: str
    base_template: str
    extension: OntologyMsg
    replace: bool
    def __init__(self, collection: _Optional[str] = ..., base_template: _Optional[str] = ..., extension: _Optional[_Union[OntologyMsg, _Mapping]] = ..., replace: bool = ...) -> None: ...

class SetOntologyResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class GetOntologyRequest(_message.Message):
    __slots__ = ("collection",)
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    collection: str
    def __init__(self, collection: _Optional[str] = ...) -> None: ...

class GetOntologyResponse(_message.Message):
    __slots__ = ("ontology",)
    ONTOLOGY_FIELD_NUMBER: _ClassVar[int]
    ontology: OntologyMsg
    def __init__(self, ontology: _Optional[_Union[OntologyMsg, _Mapping]] = ...) -> None: ...

class CostPreviewRequest(_message.Message):
    __slots__ = ("collection", "chunks")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    chunks: _containers.RepeatedCompositeFieldContainer[Chunk]
    def __init__(self, collection: _Optional[str] = ..., chunks: _Optional[_Iterable[_Union[Chunk, _Mapping]]] = ...) -> None: ...

class CostPreviewResponse(_message.Message):
    __slots__ = ("chunks", "estimated_input_tokens", "estimated_output_tokens", "estimated_cost_usd", "model", "pricing_known")
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_COST_USD_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PRICING_KNOWN_FIELD_NUMBER: _ClassVar[int]
    chunks: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    model: str
    pricing_known: bool
    def __init__(self, chunks: _Optional[int] = ..., estimated_input_tokens: _Optional[int] = ..., estimated_output_tokens: _Optional[int] = ..., estimated_cost_usd: _Optional[float] = ..., model: _Optional[str] = ..., pricing_known: bool = ...) -> None: ...

class StartExtractionRequest(_message.Message):
    __slots__ = ("collection", "chunks")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    chunks: _containers.RepeatedCompositeFieldContainer[Chunk]
    def __init__(self, collection: _Optional[str] = ..., chunks: _Optional[_Iterable[_Union[Chunk, _Mapping]]] = ...) -> None: ...

class StartExtractionResponse(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class AppendExtractionChunksRequest(_message.Message):
    __slots__ = ("collection", "job_id", "chunks", "last_batch")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    LAST_BATCH_FIELD_NUMBER: _ClassVar[int]
    collection: str
    job_id: str
    chunks: _containers.RepeatedCompositeFieldContainer[Chunk]
    last_batch: bool
    def __init__(self, collection: _Optional[str] = ..., job_id: _Optional[str] = ..., chunks: _Optional[_Iterable[_Union[Chunk, _Mapping]]] = ..., last_batch: bool = ...) -> None: ...

class AppendExtractionChunksResponse(_message.Message):
    __slots__ = ("total_chunks", "accepted")
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    total_chunks: int
    accepted: bool
    def __init__(self, total_chunks: _Optional[int] = ..., accepted: bool = ...) -> None: ...

class GetExtractionStatusRequest(_message.Message):
    __slots__ = ("collection", "job_id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    job_id: str
    def __init__(self, collection: _Optional[str] = ..., job_id: _Optional[str] = ...) -> None: ...

class JobStatusMsg(_message.Message):
    __slots__ = ("job_id", "collection", "state", "total_chunks", "processed_chunks", "entities_written", "edges_written", "cache_hits", "cache_misses", "error", "failed_chunks", "chunk_errors", "actual_input_tokens", "actual_output_tokens", "actual_cost_usd", "usage_provider_reported")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    EDGES_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    CACHE_HITS_FIELD_NUMBER: _ClassVar[int]
    CACHE_MISSES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    FAILED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_ERRORS_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_COST_USD_FIELD_NUMBER: _ClassVar[int]
    USAGE_PROVIDER_REPORTED_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    collection: str
    state: str
    total_chunks: int
    processed_chunks: int
    entities_written: int
    edges_written: int
    cache_hits: int
    cache_misses: int
    error: str
    failed_chunks: int
    chunk_errors: _containers.RepeatedCompositeFieldContainer[ChunkErrorMsg]
    actual_input_tokens: int
    actual_output_tokens: int
    actual_cost_usd: float
    usage_provider_reported: bool
    def __init__(self, job_id: _Optional[str] = ..., collection: _Optional[str] = ..., state: _Optional[str] = ..., total_chunks: _Optional[int] = ..., processed_chunks: _Optional[int] = ..., entities_written: _Optional[int] = ..., edges_written: _Optional[int] = ..., cache_hits: _Optional[int] = ..., cache_misses: _Optional[int] = ..., error: _Optional[str] = ..., failed_chunks: _Optional[int] = ..., chunk_errors: _Optional[_Iterable[_Union[ChunkErrorMsg, _Mapping]]] = ..., actual_input_tokens: _Optional[int] = ..., actual_output_tokens: _Optional[int] = ..., actual_cost_usd: _Optional[float] = ..., usage_provider_reported: bool = ...) -> None: ...

class ChunkErrorMsg(_message.Message):
    __slots__ = ("doc_id", "chunk_id", "error")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    chunk_id: int
    error: str
    def __init__(self, doc_id: _Optional[str] = ..., chunk_id: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class CancelExtractionRequest(_message.Message):
    __slots__ = ("collection", "job_id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    job_id: str
    def __init__(self, collection: _Optional[str] = ..., job_id: _Optional[str] = ...) -> None: ...

class CancelExtractionResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class ProposalMsg(_message.Message):
    __slots__ = ("id", "kind", "name", "description", "examples", "status", "source_doc", "source_chunk_id")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_DOC_FIELD_NUMBER: _ClassVar[int]
    SOURCE_CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: str
    name: str
    description: str
    examples: _containers.RepeatedScalarFieldContainer[str]
    status: str
    source_doc: str
    source_chunk_id: int
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., examples: _Optional[_Iterable[str]] = ..., status: _Optional[str] = ..., source_doc: _Optional[str] = ..., source_chunk_id: _Optional[int] = ...) -> None: ...

class ListProposalsRequest(_message.Message):
    __slots__ = ("collection",)
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    collection: str
    def __init__(self, collection: _Optional[str] = ...) -> None: ...

class ListProposalsResponse(_message.Message):
    __slots__ = ("proposals",)
    PROPOSALS_FIELD_NUMBER: _ClassVar[int]
    proposals: _containers.RepeatedCompositeFieldContainer[ProposalMsg]
    def __init__(self, proposals: _Optional[_Iterable[_Union[ProposalMsg, _Mapping]]] = ...) -> None: ...

class ApproveProposalRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: str
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[str] = ...) -> None: ...

class ApproveProposalResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class RejectProposalRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: str
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[str] = ...) -> None: ...

class RejectProposalResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class ChunkDiff(_message.Message):
    __slots__ = ("chunk_id", "action")
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    chunk_id: int
    action: ChunkDiffAction
    def __init__(self, chunk_id: _Optional[int] = ..., action: _Optional[_Union[ChunkDiffAction, str]] = ...) -> None: ...

class DiffDocumentRequest(_message.Message):
    __slots__ = ("collection", "doc_id", "chunks")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    doc_id: str
    chunks: _containers.RepeatedCompositeFieldContainer[Chunk]
    def __init__(self, collection: _Optional[str] = ..., doc_id: _Optional[str] = ..., chunks: _Optional[_Iterable[_Union[Chunk, _Mapping]]] = ...) -> None: ...

class DiffDocumentResponse(_message.Message):
    __slots__ = ("diffs",)
    DIFFS_FIELD_NUMBER: _ClassVar[int]
    diffs: _containers.RepeatedCompositeFieldContainer[ChunkDiff]
    def __init__(self, diffs: _Optional[_Iterable[_Union[ChunkDiff, _Mapping]]] = ...) -> None: ...

class ReextractDocumentRequest(_message.Message):
    __slots__ = ("collection", "doc_id", "chunks")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    doc_id: str
    chunks: _containers.RepeatedCompositeFieldContainer[Chunk]
    def __init__(self, collection: _Optional[str] = ..., doc_id: _Optional[str] = ..., chunks: _Optional[_Iterable[_Union[Chunk, _Mapping]]] = ...) -> None: ...

class ReextractDocumentResponse(_message.Message):
    __slots__ = ("job_id", "unchanged", "changed", "added", "deleted", "edges_deleted", "nodes_deleted")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    UNCHANGED_FIELD_NUMBER: _ClassVar[int]
    CHANGED_FIELD_NUMBER: _ClassVar[int]
    ADDED_FIELD_NUMBER: _ClassVar[int]
    DELETED_FIELD_NUMBER: _ClassVar[int]
    EDGES_DELETED_FIELD_NUMBER: _ClassVar[int]
    NODES_DELETED_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    unchanged: int
    changed: int
    added: int
    deleted: int
    edges_deleted: int
    nodes_deleted: int
    def __init__(self, job_id: _Optional[str] = ..., unchanged: _Optional[int] = ..., changed: _Optional[int] = ..., added: _Optional[int] = ..., deleted: _Optional[int] = ..., edges_deleted: _Optional[int] = ..., nodes_deleted: _Optional[int] = ...) -> None: ...

class CorpusDocument(_message.Message):
    __slots__ = ("doc_id", "chunks")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNKS_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    chunks: _containers.RepeatedCompositeFieldContainer[Chunk]
    def __init__(self, doc_id: _Optional[str] = ..., chunks: _Optional[_Iterable[_Union[Chunk, _Mapping]]] = ...) -> None: ...

class StartCorpusReextractionRequest(_message.Message):
    __slots__ = ("collection", "documents", "doc_ids", "resume_token")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    DOC_IDS_FIELD_NUMBER: _ClassVar[int]
    RESUME_TOKEN_FIELD_NUMBER: _ClassVar[int]
    collection: str
    documents: _containers.RepeatedCompositeFieldContainer[CorpusDocument]
    doc_ids: _containers.RepeatedScalarFieldContainer[str]
    resume_token: str
    def __init__(self, collection: _Optional[str] = ..., documents: _Optional[_Iterable[_Union[CorpusDocument, _Mapping]]] = ..., doc_ids: _Optional[_Iterable[str]] = ..., resume_token: _Optional[str] = ...) -> None: ...

class StartCorpusReextractionResponse(_message.Message):
    __slots__ = ("corpus_job_id",)
    CORPUS_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    corpus_job_id: str
    def __init__(self, corpus_job_id: _Optional[str] = ...) -> None: ...

class GetCorpusReextractionStatusRequest(_message.Message):
    __slots__ = ("collection", "corpus_job_id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    CORPUS_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    corpus_job_id: str
    def __init__(self, collection: _Optional[str] = ..., corpus_job_id: _Optional[str] = ...) -> None: ...

class CorpusDocProgressMsg(_message.Message):
    __slots__ = ("doc_id", "state", "job_id", "changed", "added", "deleted")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    CHANGED_FIELD_NUMBER: _ClassVar[int]
    ADDED_FIELD_NUMBER: _ClassVar[int]
    DELETED_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    state: str
    job_id: str
    changed: int
    added: int
    deleted: int
    def __init__(self, doc_id: _Optional[str] = ..., state: _Optional[str] = ..., job_id: _Optional[str] = ..., changed: _Optional[int] = ..., added: _Optional[int] = ..., deleted: _Optional[int] = ...) -> None: ...

class CorpusReextractionStatusMsg(_message.Message):
    __slots__ = ("corpus_job_id", "collection", "state", "total_documents", "processed_documents", "failed_documents", "skipped_documents", "changed_chunks", "added_chunks", "deleted_chunks", "edges_deleted", "nodes_deleted", "entities_written", "edges_written", "documents", "error")
    CORPUS_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    FAILED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    SKIPPED_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    CHANGED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    ADDED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    DELETED_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    EDGES_DELETED_FIELD_NUMBER: _ClassVar[int]
    NODES_DELETED_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    EDGES_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    corpus_job_id: str
    collection: str
    state: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    skipped_documents: int
    changed_chunks: int
    added_chunks: int
    deleted_chunks: int
    edges_deleted: int
    nodes_deleted: int
    entities_written: int
    edges_written: int
    documents: _containers.RepeatedCompositeFieldContainer[CorpusDocProgressMsg]
    error: str
    def __init__(self, corpus_job_id: _Optional[str] = ..., collection: _Optional[str] = ..., state: _Optional[str] = ..., total_documents: _Optional[int] = ..., processed_documents: _Optional[int] = ..., failed_documents: _Optional[int] = ..., skipped_documents: _Optional[int] = ..., changed_chunks: _Optional[int] = ..., added_chunks: _Optional[int] = ..., deleted_chunks: _Optional[int] = ..., edges_deleted: _Optional[int] = ..., nodes_deleted: _Optional[int] = ..., entities_written: _Optional[int] = ..., edges_written: _Optional[int] = ..., documents: _Optional[_Iterable[_Union[CorpusDocProgressMsg, _Mapping]]] = ..., error: _Optional[str] = ...) -> None: ...

class CancelCorpusReextractionRequest(_message.Message):
    __slots__ = ("collection", "corpus_job_id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    CORPUS_JOB_ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    corpus_job_id: str
    def __init__(self, collection: _Optional[str] = ..., corpus_job_id: _Optional[str] = ...) -> None: ...

class CancelCorpusReextractionResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
