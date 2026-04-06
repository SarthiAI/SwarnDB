"""SwarnDB collection operations.

This module provides the collection management API: create, delete, list,
get info, and check existence of collections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from ._proto import collection_pb2, vector_pb2
from .exceptions import CollectionNotFoundError
from .types import CollectionInfo, CompactResult, OptimizeResult, PruneWALResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .client import SwarnDBClient


class CollectionAPI:
    """Pythonic wrapper around the CollectionService gRPC API."""

    def __init__(self, client: SwarnDBClient) -> None:
        self._client = client

    def create(
        self,
        name: str,
        dimension: int,
        *,
        distance_metric: str = "cosine",
        default_threshold: float = 0.0,
        max_vectors: int = 0,
    ) -> CollectionInfo:
        """Create a new collection.

        Args:
            name: Unique collection name.
            dimension: Vector dimensionality.
            distance_metric: Distance function (e.g. "cosine", "euclidean").
            default_threshold: Default similarity threshold for searches.
            max_vectors: Maximum number of vectors (0 = unlimited).

        Returns:
            CollectionInfo with the created collection's metadata.

        Raises:
            CollectionExistsError: If a collection with this name already exists.
        """
        request = collection_pb2.CreateCollectionRequest(
            name=name,
            dimension=dimension,
            distance_metric=distance_metric,
            default_threshold=default_threshold,
            max_vectors=max_vectors,
        )
        self._client._call(
            self._client._collection_stub.CreateCollection, request
        )
        return CollectionInfo(
            name=name,
            dimension=dimension,
            distance_metric=distance_metric,
            vector_count=0,
            default_threshold=default_threshold,
        )

    def get(self, name: str) -> CollectionInfo:
        """Get collection metadata.

        Args:
            name: Collection name.

        Returns:
            CollectionInfo for the requested collection.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        request = collection_pb2.GetCollectionRequest(name=name)
        response = self._client._call(
            self._client._collection_stub.GetCollection, request
        )
        return CollectionInfo(
            name=response.name,
            dimension=response.dimension,
            distance_metric=response.distance_metric,
            vector_count=response.vector_count,
            default_threshold=response.default_threshold,
        )

    def delete(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name.

        Returns:
            True on success.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        request = collection_pb2.DeleteCollectionRequest(name=name)
        response = self._client._call(
            self._client._collection_stub.DeleteCollection, request
        )
        return bool(response.success)

    def list(self) -> List[CollectionInfo]:
        """List all collections.

        Returns:
            List of CollectionInfo for every collection on the server.
        """
        request = collection_pb2.ListCollectionsRequest()
        response = self._client._call(
            self._client._collection_stub.ListCollections, request
        )
        return [
            CollectionInfo(
                name=c.name,
                dimension=c.dimension,
                distance_metric=c.distance_metric,
                vector_count=c.vector_count,
                default_threshold=c.default_threshold,
            )
            for c in response.collections
        ]

    def exists(self, name: str) -> bool:
        """Check whether a collection exists.

        Args:
            name: Collection name.

        Returns:
            True if the collection exists, False otherwise.
        """
        try:
            self.get(name)
            return True
        except CollectionNotFoundError:
            return False

    def optimize(self, collection: str, rebuild_graph: bool = False) -> OptimizeResult:
        """Optimize a collection (rebuild deferred HNSW index).

        Args:
            collection: Collection name.
            rebuild_graph: If True, also rebuild the virtual graph. Default False.
        """
        request = vector_pb2.OptimizeRequest(
            collection=collection,
            rebuild_graph=rebuild_graph,
        )
        response = self._client._call(
            self._client._vector_stub.Optimize, request
        )
        return OptimizeResult(
            status=response.status,
            message=response.message,
            duration_ms=response.duration_ms,
            vectors_processed=response.vectors_processed,
        )

    def prune_wal(self, collection: str) -> PruneWALResult:
        """Prune old WAL files for a collection.

        Removes write-ahead log files that are no longer needed after
        data has been flushed to segments.
        """
        request = vector_pb2.PruneWALRequest(collection=collection)
        response = self._client._call(
            self._client._vector_stub.PruneWAL, request
        )
        return PruneWALResult(
            status=response.status,
            files_deleted=response.files_deleted,
            bytes_freed=response.bytes_freed,
            duration_ms=response.duration_ms,
        )

    def compact(self, collection: str, min_segments: int = 0, remove_deleted: bool = True) -> CompactResult:
        """Compact collection segments into fewer, larger files.

        Args:
            collection: Collection name.
            min_segments: Minimum segment count to trigger compaction. 0 = use server default (4).
            remove_deleted: Whether to remove deleted vectors during compaction. Default True.
        """
        request = vector_pb2.CompactRequest(
            collection=collection,
            min_segments=min_segments,
            remove_deleted=remove_deleted,
        )
        response = self._client._call(
            self._client._vector_stub.Compact, request
        )
        return CompactResult(
            status=response.status,
            segments_merged=response.segments_merged,
            vectors_written=response.vectors_written,
            vectors_removed=response.vectors_removed,
            duration_ms=response.duration_ms,
        )

    def get_status(self, collection: str) -> str:
        """Get collection optimization status.

        Returns the current optimization state of a collection, which
        indicates whether deferred indexes need to be rebuilt.

        Args:
            collection: Collection name.

        Returns:
            One of: ``"ready"``, ``"pending_optimization"``, or
            ``"optimizing"``.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        request = collection_pb2.GetCollectionRequest(name=collection)
        response = self._client._call(
            self._client._collection_stub.GetCollection, request
        )
        return response.status or "ready"
