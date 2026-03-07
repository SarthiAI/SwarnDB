"""SwarnDB data types.

Python dataclasses mirroring the protobuf response types used across
the SwarnDB gRPC API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GraphEdge:
    """An edge in the virtual graph."""

    target_id: int
    similarity: float


@dataclass(frozen=True)
class ScoredResult:
    """A search result with similarity score."""

    id: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_edges: List[GraphEdge] = field(default_factory=list)


@dataclass(frozen=True)
class CollectionInfo:
    """Metadata about a collection."""

    name: str
    dimension: int
    distance_metric: str
    vector_count: int
    default_threshold: float


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
class BulkInsertResult:
    """Result of a bulk insert operation."""

    inserted_count: int
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResult:
    """Result of a single search query."""

    results: List[ScoredResult]
    search_time_us: int


@dataclass(frozen=True)
class BatchSearchResult:
    """Result of a batch search operation."""

    results: List[SearchResult]
    total_time_us: int
