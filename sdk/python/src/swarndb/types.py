"""SwarnDB data types.

Python dataclasses mirroring the protobuf response types used across
the SwarnDB gRPC API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GraphEdge:
    """An edge in the virtual graph."""

    target_id: int
    similarity: float


@dataclass(frozen=True)
class NodeAudit:
    """A single entry in a typed node's audit trail (Hybrid mode)."""

    action: str
    actor: str = ""
    at: int = 0


@dataclass(frozen=True)
class TypedNode:
    """A node in the first-class typed graph (Hybrid mode)."""

    id: int
    kind: str  # "content" or "entity"
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    source: str = "manual"
    created_at: int = 0
    created_by: str = ""
    history: List[NodeAudit] = field(default_factory=list)
    updated_at: int = 0


@dataclass(frozen=True)
class EdgeAudit:
    """A single entry in a typed edge's audit trail (Hybrid mode)."""

    action: str
    actor: str = ""
    at: int = 0


@dataclass(frozen=True)
class TypedEdge:
    """A typed, directed edge in the first-class graph (Hybrid mode)."""

    id: int
    source: int
    target: int
    edge_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    verified: bool = False
    is_manual: bool = False
    created_at: int = 0
    history: List[EdgeAudit] = field(default_factory=list)
    # Temporal validity window + context (P17). None = unbounded / no context.
    valid_from: Optional[int] = None
    valid_until: Optional[int] = None
    temporal_context: Optional[str] = None


@dataclass(frozen=True)
class NodePage:
    """One page of a paginated whole-graph node enumeration (ADR-014)."""

    nodes: List["TypedNode"] = field(default_factory=list)
    # Pass as ``after_id`` to fetch the next page; 0 when exhausted.
    next_cursor: int = 0
    has_more: bool = False


@dataclass(frozen=True)
class EdgePage:
    """One page of a paginated whole-graph edge enumeration (ADR-014)."""

    edges: List["TypedEdge"] = field(default_factory=list)
    # Pass as ``after_id`` to fetch the next page; 0 when exhausted.
    next_cursor: int = 0
    has_more: bool = False


@dataclass(frozen=True)
class EdgeRejectResult:
    """Result of rejecting a typed edge (Hybrid mode)."""

    deleted: bool
    rule_added: bool


@dataclass(frozen=True)
class BulkImportRowError:
    """A per-row failure from a bulk edge import (Hybrid mode)."""

    row: int
    message: str


@dataclass(frozen=True)
class BulkImportResult:
    """Result of a bulk edge import (Hybrid mode)."""

    total_rows: int
    imported: int
    failed: int
    errors: List[BulkImportRowError] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkDiff:
    """A per-chunk diff result from a document re-extraction (Hybrid mode)."""

    chunk_id: int
    action: str  # "unchanged" | "changed" | "new" | "deleted"


@dataclass(frozen=True)
class ReextractSummary:
    """Summary of a document re-extraction run (Hybrid mode)."""

    job_id: str = ""
    unchanged: int = 0
    changed: int = 0
    added: int = 0
    deleted: int = 0
    edges_deleted: int = 0
    nodes_deleted: int = 0


@dataclass(frozen=True)
class HybridQueryResult:
    """Result of a hybrid graph query (Hybrid mode).

    Exactly one of ``nodes``, ``edges``, or ``paths`` is populated,
    depending on the terminal return kind requested on the builder.
    """

    nodes: List["TypedNode"] = field(default_factory=list)
    edges: List["TypedEdge"] = field(default_factory=list)
    paths: List[List[int]] = field(default_factory=list)


@dataclass(frozen=True)
class ScoredResult:
    """A search result with distance score (lower = more similar)."""

    id: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_edges: List[GraphEdge] = field(default_factory=list)


@dataclass(frozen=True)
class ScalarQuantizationConfig:
    """Configuration for scalar (SQ8) quantization."""

    quantile: float = 0.99
    always_ram: bool = True


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization configuration for a collection."""

    type: str = "scalar"
    scalar: Optional[ScalarQuantizationConfig] = None


@dataclass(frozen=True)
class SearchQuantizationParams:
    """Per-query quantization parameters."""

    rescore: bool = True
    oversampling: float = 3.0
    ignore: bool = False


@dataclass(frozen=True)
class CollectionInfo:
    """Metadata about a collection."""

    name: str
    dimension: int
    distance_metric: str
    vector_count: int
    default_threshold: float
    quantization_type: Optional[str] = None
    # Live HNSW index node count. May trail vector_count when an index build is
    # deferred or pending optimization; vector_count is the stored-row count.
    indexed_count: int = 0


@dataclass(frozen=True)
class VectorRecord:
    """A stored vector with its metadata."""

    id: int
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraversalNode:
    """A node visited during graph traversal."""

    id: int
    depth: int
    path_similarity: float
    path: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class GhostVector:
    """A vector identified as isolated (ghost) in the graph."""

    id: int
    isolation_score: float


@dataclass(frozen=True)
class ConeSearchResult:
    """A result from a cone (angular) search."""

    id: int
    cosine_similarity: float
    angle_radians: float


@dataclass(frozen=True)
class DriftReport:
    """Report from a distribution drift detection analysis."""

    centroid_shift: float
    mean_distance_window1: float
    mean_distance_window2: float
    spread_change: float
    has_drifted: bool


@dataclass(frozen=True)
class ClusterAssignment:
    """Assignment of a vector to a cluster."""

    id: int
    cluster: int
    distance_to_centroid: float


@dataclass(frozen=True)
class ClusterResult:
    """Result of k-means clustering."""

    centroids: List[List[float]]
    assignments: List[ClusterAssignment]
    iterations: int
    converged: bool


@dataclass(frozen=True)
class PCAResult:
    """Result of PCA dimensionality reduction."""

    components: List[List[float]]
    explained_variance: List[float]
    mean: List[float]
    projected: List[List[float]]


@dataclass(frozen=True)
class DiversityResult:
    """A result from MMR (Maximal Marginal Relevance) diversity search."""

    id: int
    relevance_score: float
    mmr_score: float


@dataclass(frozen=True)
class BulkInsertOptions:
    """Configuration options for optimized bulk insert operations.

    These parameters control server-side behavior during bulk ingestion,
    allowing callers to trade consistency for throughput.
    """

    batch_lock_size: Optional[int] = None
    defer_graph: bool = False
    wal_flush_every: Optional[int] = None
    ef_construction: Optional[int] = None
    index_mode: Optional[str] = None
    skip_metadata_index: bool = False
    parallel_build: bool = False
    checkpoint_every: Optional[int] = None
    resume_token: Optional[str] = None

    def has_non_defaults(self) -> bool:
        """Return True if any parameter differs from its default."""
        return (
            self.batch_lock_size is not None
            or self.defer_graph
            or self.wal_flush_every is not None
            or self.ef_construction is not None
            or self.index_mode is not None
            or self.skip_metadata_index
            or self.parallel_build
            or self.checkpoint_every is not None
            or self.resume_token is not None
        )


@dataclass(frozen=True)
class OptimizeResult:
    """Result of a collection optimize operation."""

    status: str
    message: str
    duration_ms: int
    vectors_processed: int


@dataclass(frozen=True)
class PruneWALResult:
    """Result of a WAL prune operation."""

    status: str
    files_deleted: int
    bytes_freed: int
    duration_ms: int


@dataclass(frozen=True)
class CompactResult:
    """Result of a compaction operation."""

    status: str
    segments_merged: int
    vectors_written: int
    vectors_removed: int
    duration_ms: int


@dataclass(frozen=True)
class BulkInsertResult:
    """Result of a bulk insert operation."""

    inserted_count: int
    errors: List[str] = field(default_factory=list)
    last_completed_batch_idx: int = 0
    last_committed_lsn: int = 0
    resume_token: str = ""
    assigned_ids: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class BulkInsertFromPathRequest:
    """Parameters for an mmap-based bulk insert from a server-side path.

    The server reads vector bytes directly from a file on its local
    filesystem rather than streaming them over gRPC. This avoids the
    client-side allocation cost for very large ingests.
    """

    collection: str
    path: str
    dim: int = 0
    expected_count: int = 0
    total_count_hint: int = 0
    id_start: int = 1
    ids_path: str = ""
    skip_metadata_index: bool = False
    index_mode: str = "immediate"
    ef_construction: int = 0


@dataclass(frozen=True)
class SearchResult:
    """Result of a single search query."""

    results: List[ScoredResult]
    search_time_us: int
    warning: str = ""


@dataclass(frozen=True)
class BatchSearchResult:
    """Result of a batch search operation."""

    results: List[SearchResult]
    total_time_us: int


@dataclass(frozen=True)
class RecoveryStatus:
    """Server recovery status snapshot."""

    path: str
    elapsed_secs: int
    collections: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PersistenceStatus:
    """Per-collection persistence and LSN state."""

    last_snapshot_lsn: int
    current_lsn: int
    next_lsn: int


@dataclass(frozen=True)
class CollectionMetrics:
    """Per-collection lock-contention counters."""

    map_lock_acquisitions: int
    collection_read_acquisitions: int
    collection_write_acquisitions: int
    total_blocked_microseconds: int


@dataclass(frozen=True)
class ReadinessStatus:
    """Result of the readiness probe."""

    ready: bool
    status: str
    checks: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthStatus:
    """Result of the liveness probe."""

    healthy: bool
    status: str
    checks: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction (LLM-driven, Hybrid mode only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """A client-supplied unit of extraction.

    The embedding is optional and is used only for entity dedup when present.
    """

    doc_id: str
    chunk_id: int
    text: str
    embedding: Tuple[float, ...] = ()


@dataclass(frozen=True)
class LlmConfigInfo:
    """Redacted view of a collection's LLM config; never carries the api key."""

    base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    api_key_set: bool


@dataclass(frozen=True)
class EntityLabel:
    """An ontology entity label."""

    label: str
    description: str = ""


@dataclass(frozen=True)
class EdgeType:
    """An ontology edge type with optional endpoint label constraints."""

    edge_type: str
    description: str = ""
    source_labels: List[str] = field(default_factory=list)
    target_labels: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OntologyInfo:
    """A collection's extraction ontology."""

    entity_labels: List[EntityLabel] = field(default_factory=list)
    edge_types: List[EdgeType] = field(default_factory=list)
    # Custom prompt fields; None means the default prompt is used.
    system_prompt: Optional[str] = None
    extra_guidance: Optional[str] = None
    # Opt-in passage-to-entity linking for GraphRAG (ADR-012).
    link_passages: bool = False
    # Entity-resolution mode (ADR-020): "normalized" (default) or "fuzzy".
    entity_resolution: str = "normalized"


@dataclass(frozen=True)
class CostEstimate:
    """Estimated token usage and cost for an extraction run."""

    chunks: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    model: str
    pricing_known: bool


@dataclass(frozen=True)
class ChunkError:
    """One per-chunk extraction failure from a partially-successful job."""

    doc_id: str
    chunk_id: int
    error: str


@dataclass(frozen=True)
class ExtractionJob:
    """A snapshot of an extraction job's progress."""

    job_id: str
    collection: str
    state: str  # queued | running | completed | completed_with_errors | failed | cancelled
    total_chunks: int = 0
    processed_chunks: int = 0
    entities_written: int = 0
    edges_written: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    error: str = ""
    failed_chunks: int = 0
    chunk_errors: List[ChunkError] = field(default_factory=list)
    # Area 4: post-job cost actuals. Cache hits cost zero tokens; actual_cost_usd
    # is 0.0 when the model's pricing is unknown (never fabricated).
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    actual_cost_usd: float = 0.0
    # True only while every priced LLM call reported its own usage; False once
    # any call's tokens had to be estimated locally.
    usage_provider_reported: bool = True


@dataclass(frozen=True)
class CorpusDocProgress:
    """Per-document progress within a corpus re-extraction run (Hybrid mode)."""

    doc_id: str
    state: str  # pending | completed | completed_with_errors | failed | skipped
    job_id: str = ""
    changed: int = 0
    added: int = 0
    deleted: int = 0


@dataclass(frozen=True)
class CorpusReextractionStatus:
    """Master status of a corpus re-extraction job (Hybrid mode).

    The ``corpus_job_id`` doubles as the resume token: re-issuing
    ``start_corpus_reextraction`` with it continues the run and skips documents
    already completed.
    """

    corpus_job_id: str
    collection: str
    state: str  # queued | running | completed | completed_with_errors | failed | cancelled
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    skipped_documents: int = 0
    changed_chunks: int = 0
    added_chunks: int = 0
    deleted_chunks: int = 0
    edges_deleted: int = 0
    nodes_deleted: int = 0
    entities_written: int = 0
    edges_written: int = 0
    documents: List[CorpusDocProgress] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class OntologyProposal:
    """A proposed ontology entity label or edge type awaiting review."""

    id: str
    kind: str  # entity_label | edge_type
    name: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | approved | rejected
    source_doc: str = ""
    source_chunk_id: int = 0
