"""SwarnDB NumPy integration utilities.

Type conversion helpers, NumPy-aware client mixin, and convenience
result types for zero-copy-friendly interop between NumPy arrays and
the SwarnDB Python SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from .types import BulkInsertResult, ClusterAssignment

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

VectorLike = Union[List[float], np.ndarray]

# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def to_list(v: VectorLike) -> List[float]:
    """Convert a numpy array or list to a Python list of floats.

    Uses ``.tolist()`` for numpy arrays which delegates to CPython list
    construction -- the fastest path from a contiguous buffer.
    """
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tolist()
    return list(v)


def to_numpy(v: List[float]) -> np.ndarray:
    """Convert a list of floats to a numpy float32 array."""
    return np.array(v, dtype=np.float32)


def to_numpy_batch(vectors: List[List[float]]) -> np.ndarray:
    """Convert a list of vectors to a 2D numpy array (n, dim)."""
    return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# NumPy-specific result types
# ---------------------------------------------------------------------------


@dataclass
class NumpyPCAResult:
    """PCA result with numpy arrays instead of nested lists."""

    components: np.ndarray  # (n_components, dim)
    explained_variance: np.ndarray  # (n_components,)
    mean: np.ndarray  # (dim,)
    projected: np.ndarray  # (n_vectors, n_components)


@dataclass
class NumpyClusterResult:
    """K-means result with numpy centroids."""

    centroids: np.ndarray  # (k, dim)
    assignments: list  # List[ClusterAssignment]
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# NumpyMixin
# ---------------------------------------------------------------------------


class NumpyMixin:
    """Mixin that adds numpy-aware methods to SwarnDBClient.

    Usage::

        class NumpyClient(NumpyMixin, SwarnDBClient):
            pass

        client = NumpyClient("localhost", 50051)
        # Or use the convenience function:
        client = create_numpy_client("localhost", 50051)
    """

    def np_insert(self, collection: str, vector: np.ndarray, **kwargs) -> int:
        """Insert a vector from a numpy array."""
        return self.vectors.insert(collection, to_list(vector), **kwargs)

    def np_bulk_insert(
        self, collection: str, vectors: np.ndarray, **kwargs
    ) -> BulkInsertResult:
        """Bulk insert from a 2D numpy array (n, dim)."""
        vector_lists = vectors.astype(np.float32).tolist()
        return self.vectors.bulk_insert(collection, vector_lists, **kwargs)

    def np_search(
        self, collection: str, query: np.ndarray, k: int = 10, **kwargs
    ):
        """Search with a numpy query vector.

        Returns the same ``SearchResult`` as the regular search API.
        """
        return self.search.query(collection, to_list(query), k, **kwargs)

    def np_get(self, collection: str, id: int) -> Tuple[np.ndarray, dict]:
        """Get a vector as a numpy array.

        Returns:
            A tuple of ``(np.ndarray, metadata_dict)``.
        """
        record = self.vectors.get(collection, id)
        return to_numpy(record.vector), record.metadata

    def np_centroid(self, collection: str, **kwargs) -> np.ndarray:
        """Compute centroid, return as numpy array."""
        result = self.math.centroid(collection, **kwargs)
        return to_numpy(result)

    def np_interpolate(
        self,
        a: np.ndarray,
        b: np.ndarray,
        t: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        """Interpolate between numpy vectors."""
        result = self.math.interpolate(to_list(a), to_list(b), t, **kwargs)
        return to_numpy(result)

    def np_pca(
        self, collection: str, n_components: int, **kwargs
    ) -> NumpyPCAResult:
        """PCA with numpy returns.

        Returns a ``NumpyPCAResult`` whose fields are numpy arrays.
        """
        result = self.math.reduce_dimensions(collection, n_components, **kwargs)
        return NumpyPCAResult(
            components=to_numpy_batch(result.components),
            explained_variance=np.array(result.explained_variance, dtype=np.float32),
            mean=to_numpy(result.mean),
            projected=to_numpy_batch(result.projected),
        )

    def np_cluster(
        self, collection: str, k: int, **kwargs
    ) -> NumpyClusterResult:
        """K-means with numpy centroids."""
        result = self.math.cluster(collection, k, **kwargs)
        return NumpyClusterResult(
            centroids=to_numpy_batch(result.centroids),
            assignments=result.assignments,
            iterations=result.iterations,
            converged=result.converged,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_numpy_client(*args, **kwargs):
    """Create a SwarnDBClient with NumPy support.

    Accepts the same arguments as ``SwarnDBClient.__init__``.

    Usage::

        client = create_numpy_client("localhost", 50051)
        vec = np.random.randn(128).astype(np.float32)
        client.np_insert("my_collection", vec)
    """
    from .client import SwarnDBClient

    class NumpySwarnDBClient(NumpyMixin, SwarnDBClient):
        """SwarnDBClient with NumPy helper methods."""

        pass

    return NumpySwarnDBClient(*args, **kwargs)
