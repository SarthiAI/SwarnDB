"""SwarnDB collection operations.

This module provides the collection management API: create, delete, list,
get info, and check existence of collections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from ._proto import collection_pb2, vector_pb2
from .exceptions import CollectionNotFoundError
from .types import (
    CollectionInfo,
    CollectionMetrics,
    CompactResult,
    OptimizeResult,
    PersistenceStatus,
    PruneWALResult,
    QuantizationConfig,
    ScalarQuantizationConfig,
)

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
        quantization: Optional[QuantizationConfig] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
    ) -> CollectionInfo:
        """Create a new collection.

        Args:
            name: Unique collection name.
            dimension: Vector dimensionality.
            distance_metric: Distance function (e.g. "cosine", "euclidean").
            default_threshold: Default similarity threshold for searches.
            max_vectors: Maximum number of vectors (0 = unlimited).
            quantization: Optional quantization configuration (e.g. SQ8).
            m: Optional HNSW M parameter override. When None, the server
                uses its default.
            ef_construction: Optional HNSW ef_construction override. When
                None, the server uses its default.

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
        if quantization is not None:
            if quantization.scalar is not None or quantization.type == "scalar":
                sq = quantization.scalar or ScalarQuantizationConfig()
                proto_sq = collection_pb2.ScalarQuantization(
                    quantile=sq.quantile,
                    always_ram=sq.always_ram,
                )
                request.quantization.scalar.CopyFrom(proto_sq)
        if m is not None:
            request.m = m
        if ef_construction is not None:
            request.ef_construction = ef_construction
        self._client._call(
            self._client._collection_stub.CreateCollection, request
        )
        quantization_type = quantization.type if quantization is not None else None
        return CollectionInfo(
            name=name,
            dimension=dimension,
            distance_metric=distance_metric,
            vector_count=0,
            default_threshold=default_threshold,
            quantization_type=quantization_type,
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
        qt = getattr(response, 'quantization_type', '') or None
        return CollectionInfo(
            name=response.name,
            dimension=response.dimension,
            distance_metric=response.distance_metric,
            vector_count=response.vector_count,
            default_threshold=response.default_threshold,
            quantization_type=qt,
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
                quantization_type=getattr(c, 'quantization_type', '') or None,
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

    def snapshot(self, name: str) -> int:
        """Force a synchronous snapshot for a collection.

        Args:
            name: Collection name.

        Returns:
            The LSN of the snapshot just written.
        """
        request = collection_pb2.SnapshotCollectionRequest(name=name)
        response = self._client._call(
            self._client._collection_stub.SnapshotCollection, request
        )
        return int(response.last_snapshot_lsn)

    def persistence_status(self, name: str) -> PersistenceStatus:
        """Return the snapshot and WAL LSN state for a collection."""
        request = collection_pb2.GetPersistenceStatusRequest(name=name)
        response = self._client._call(
            self._client._collection_stub.GetPersistenceStatus, request
        )
        return PersistenceStatus(
            last_snapshot_lsn=int(response.last_snapshot_lsn),
            current_lsn=int(response.current_lsn),
            next_lsn=int(response.next_lsn),
        )

    def metrics(self, name: str) -> CollectionMetrics:
        """Return per-collection lock-contention counters."""
        request = collection_pb2.GetCollectionMetricsRequest(name=name)
        response = self._client._call(
            self._client._collection_stub.GetCollectionMetrics, request
        )
        return CollectionMetrics(
            map_lock_acquisitions=int(response.map_lock_acquisitions),
            collection_read_acquisitions=int(response.collection_read_acquisitions),
            collection_write_acquisitions=int(response.collection_write_acquisitions),
            total_blocked_microseconds=int(response.total_blocked_microseconds),
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
