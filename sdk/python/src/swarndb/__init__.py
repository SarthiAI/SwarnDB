"""SwarnDB Python SDK.

High-performance vector database client for Python.
"""

from swarndb._version import __version__
from swarndb.exceptions import (
    AuthenticationError,
    CollectionError,
    CollectionExistsError,
    CollectionNotFoundError,
    ConnectionError,
    DimensionMismatchError,
    GraphError,
    MathError,
    SearchError,
    SwarnDBError,
    VectorError,
    VectorNotFoundError,
)
from swarndb.types import (
    BatchSearchResult,
    BulkInsertOptions,
    BulkInsertResult,
    ClusterAssignment,
    ClusterResult,
    CollectionInfo,
    ConeSearchResult,
    DiversityResult,
    DriftReport,
    GhostVector,
    GraphEdge,
    OptimizeResult,
    PCAResult,
    ScoredResult,
    SearchResult,
    TraversalNode,
    VectorRecord,
)
from swarndb.search import Filter

try:
    from swarndb.numpy_utils import (
        NumpyMixin,
        NumpyPCAResult,
        NumpyClusterResult,
        VectorLike,
        create_numpy_client,
        to_list,
        to_numpy,
        to_numpy_batch,
    )
except ImportError:
    # numpy is an optional dependency
    NumpyMixin = None  # type: ignore[assignment,misc]
    NumpyPCAResult = None  # type: ignore[assignment,misc]
    NumpyClusterResult = None  # type: ignore[assignment,misc]
    VectorLike = None  # type: ignore[assignment,misc]
    create_numpy_client = None  # type: ignore[assignment]
    to_list = None  # type: ignore[assignment]
    to_numpy = None  # type: ignore[assignment]
    to_numpy_batch = None  # type: ignore[assignment]

# Lazy imports for client classes (implementations not yet available)
_LAZY_IMPORTS = {
    "SwarnDBClient": "swarndb.client",
    "AsyncSwarnDBClient": "swarndb.async_client",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'swarndb' has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Clients
    "SwarnDBClient",
    "AsyncSwarnDBClient",
    # Types
    "Filter",
    "ScoredResult",
    "SearchResult",
    "BatchSearchResult",
    "CollectionInfo",
    "VectorRecord",
    "GraphEdge",
    "TraversalNode",
    "GhostVector",
    "ConeSearchResult",
    "DriftReport",
    "ClusterResult",
    "ClusterAssignment",
    "PCAResult",
    "DiversityResult",
    "BulkInsertOptions",
    "BulkInsertResult",
    "OptimizeResult",
    # Exceptions
    "SwarnDBError",
    "ConnectionError",
    "AuthenticationError",
    "CollectionError",
    "CollectionNotFoundError",
    "CollectionExistsError",
    "VectorError",
    "VectorNotFoundError",
    "DimensionMismatchError",
    "SearchError",
    "GraphError",
    "MathError",
    # NumPy integration
    "NumpyMixin",
    "NumpyPCAResult",
    "NumpyClusterResult",
    "VectorLike",
    "create_numpy_client",
    "to_list",
    "to_numpy",
    "to_numpy_batch",
]
