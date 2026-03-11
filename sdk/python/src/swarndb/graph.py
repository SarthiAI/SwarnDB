"""SwarnDB virtual graph operations.

This module provides graph-based operations: get related vectors, traverse
the virtual similarity graph, and configure similarity thresholds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ._proto import graph_pb2
from .types import GraphEdge, TraversalNode

if TYPE_CHECKING:
    from .client import SwarnDBClient


class GraphAPI:
    """Pythonic wrapper around the GraphService gRPC API."""

    def __init__(self, client: SwarnDBClient) -> None:
        self._client = client

    def get_related(
        self,
        collection: str,
        vector_id: int,
        *,
        threshold: float = 0.0,
        max_results: int = 10,
    ) -> List[GraphEdge]:
        """Get vectors related to the given vector via the virtual graph.

        Args:
            collection: Collection name.
            vector_id: ID of the source vector.
            threshold: Minimum similarity threshold for edges.
            max_results: Maximum number of related vectors to return.

        Returns:
            List of GraphEdge(target_id, similarity).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            VectorNotFoundError: If the vector does not exist.
            GraphError: If the graph operation fails.
        """
        request = graph_pb2.GetRelatedRequest(
            collection=collection,
            vector_id=vector_id,
            threshold=threshold,
            max_results=max_results,
        )
        response = self._client._call(
            self._client._graph_stub.GetRelated, request
        )
        return [
            GraphEdge(
                target_id=edge.target_id,
                similarity=edge.similarity,
            )
            for edge in response.edges
        ]

    def traverse(
        self,
        collection: str,
        start_id: int,
        *,
        depth: int = 2,
        threshold: float = 0.0,
        max_results: int = 100,
    ) -> List[TraversalNode]:
        """Multi-hop graph traversal from a starting vector.

        Args:
            collection: Collection name.
            start_id: ID of the starting vector.
            depth: Maximum traversal depth (number of hops).
            threshold: Minimum similarity threshold for traversal edges.
            max_results: Maximum number of nodes to return.

        Returns:
            List of TraversalNode(id, depth, path_similarity, path).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            VectorNotFoundError: If the starting vector does not exist.
            GraphError: If the graph operation fails.
        """
        request = graph_pb2.TraverseRequest(
            collection=collection,
            start_id=start_id,
            depth=depth,
            threshold=threshold,
            max_results=max_results,
        )
        response = self._client._call(
            self._client._graph_stub.Traverse, request
        )
        return [
            TraversalNode(
                id=node.id,
                depth=node.depth,
                path_similarity=node.path_similarity,
                path=list(node.path),
            )
            for node in response.nodes
        ]

    def set_threshold(
        self,
        collection: str,
        threshold: float,
        *,
        vector_id: int = 0,
    ) -> bool:
        """Set similarity threshold for the virtual graph.

        If vector_id is 0, sets the collection-level threshold.
        Otherwise sets a per-vector threshold override.

        Args:
            collection: Collection name.
            threshold: Similarity threshold value.
            vector_id: Vector ID for per-vector threshold, or 0 for
                collection-level.

        Returns:
            True on success.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            GraphError: If the operation fails.
        """
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError("threshold must be >0.0 and <=1.0")

        request = graph_pb2.SetThresholdRequest(
            collection=collection,
            vector_id=vector_id,
            threshold=threshold,
        )
        response = self._client._call(
            self._client._graph_stub.SetThreshold, request
        )
        return bool(response.success)
