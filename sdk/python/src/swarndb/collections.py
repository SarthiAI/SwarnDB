"""SwarnDB collection operations.

This module provides the collection management API: create, delete, list,
get info, and check existence of collections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ._proto import collection_pb2
from .exceptions import CollectionNotFoundError
from .types import CollectionInfo

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
