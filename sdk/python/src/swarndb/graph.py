"""SwarnDB virtual graph operations.

This module provides graph-based operations: get related vectors, traverse
the virtual similarity graph, and configure similarity thresholds.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from ._proto import graph_pb2
from .types import (
    BulkImportResult,
    BulkImportRowError,
    EdgeAudit,
    EdgePage,
    EdgeRejectResult,
    GraphEdge,
    NodeAudit,
    NodePage,
    TraversalNode,
    TypedEdge,
    TypedNode,
)

if TYPE_CHECKING:
    from .client import SwarnDBClient
    from .hybrid import HybridQueryBuilder
    from .types import HybridQueryResult


def _kind_to_proto(kind: str):
    """Map a node-kind string to its proto enum value."""
    mapping = {
        "content": graph_pb2.TYPED_NODE_CONTENT,
        "entity": graph_pb2.TYPED_NODE_ENTITY,
    }
    if kind not in mapping:
        raise ValueError(f"kind must be 'content' or 'entity', got {kind!r}")
    return mapping[kind]


def _node_from_proto(n: Any) -> TypedNode:
    kind = "entity" if n.kind == graph_pb2.TYPED_NODE_ENTITY else "content"
    return TypedNode(
        id=n.id,
        kind=kind,
        label=n.label,
        properties=json.loads(n.properties_json) if n.properties_json else {},
        embedding=list(n.embedding),
        source=n.source,
        created_at=n.created_at,
        created_by=n.created_by,
        history=[
            NodeAudit(action=a.action, actor=a.actor, at=a.at)
            for a in n.history
        ],
        updated_at=n.updated_at,
    )


def _bulk_import_format_to_proto(fmt: str):
    """Map a bulk-import format string to its proto enum value."""
    mapping = {
        "csv": graph_pb2.BULK_IMPORT_FORMAT_CSV,
        "jsonl": graph_pb2.BULK_IMPORT_FORMAT_JSONL,
    }
    key = fmt.lower()
    if key not in mapping:
        raise ValueError(f"format must be 'csv' or 'jsonl', got {fmt!r}")
    return mapping[key]


def _edge_from_proto(e: Any) -> TypedEdge:
    return TypedEdge(
        id=e.id,
        source=e.source,
        target=e.target,
        edge_type=e.edge_type,
        properties=json.loads(e.properties_json) if e.properties_json else {},
        provenance=json.loads(e.provenance_json) if e.provenance_json else {},
        confidence=e.confidence,
        verified=e.verified,
        is_manual=e.is_manual,
        created_at=e.created_at,
        history=[
            EdgeAudit(action=a.action, actor=a.actor, at=a.at)
            for a in e.history
        ],
        valid_from=e.valid_from if e.HasField("valid_from") else None,
        valid_until=e.valid_until if e.HasField("valid_until") else None,
        temporal_context=(
            e.temporal_context if e.HasField("temporal_context") else None
        ),
    )


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
        request = graph_pb2.SetThresholdRequest(
            collection=collection,
            vector_id=vector_id,
            threshold=threshold,
        )
        response = self._client._call(
            self._client._graph_stub.SetThreshold, request
        )
        return bool(response.success)

    # ── Typed graph (Hybrid mode) ──

    def put_node(
        self,
        collection: str,
        *,
        kind: str = "content",
        label: str = "",
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        source: str = "manual",
        created_by: str = "",
    ) -> int:
        """Create a typed node (Hybrid collections only). Returns the node id.

        For a searchable content vector use ``vectors.insert`` instead: the vector id it
        assigns becomes the content node id via the NodeId==VectorId bridge. ``put_node``
        accepts an ``embedding`` only for entity nodes (kind="entity"), where the inline
        embedding feeds vector_rank's graph-scoped inline-embedding fallback, or for
        content placeholders that carry no embedding. A content node supplied WITH an
        embedding is rejected, because it would be stored inline but never indexed into the
        vector index (not searchable).
        """
        request = graph_pb2.PutNodeRequest(
            collection=collection,
            kind=_kind_to_proto(kind),
            label=label,
            properties_json=json.dumps(properties or {}),
            embedding=list(embedding or []),
            source=source,
            created_by=created_by,
        )
        response = self._client._call(self._client._graph_stub.PutNode, request)
        return int(response.id)

    def get_node(self, collection: str, node_id: int) -> Optional[TypedNode]:
        """Fetch a typed node by id, or None if absent."""
        request = graph_pb2.GetNodeRequest(collection=collection, id=node_id)
        response = self._client._call(self._client._graph_stub.GetNode, request)
        return _node_from_proto(response.node) if response.found else None

    def delete_node(self, collection: str, node_id: int) -> bool:
        """Delete a typed node and its incident edges. True if it existed."""
        request = graph_pb2.DeleteNodeRequest(collection=collection, id=node_id)
        response = self._client._call(self._client._graph_stub.DeleteNode, request)
        return bool(response.deleted)

    def update_node(
        self,
        collection: str,
        node_id: int,
        *,
        properties: Optional[Dict[str, Any]] = None,
        actor: str = "",
    ) -> TypedNode:
        """Update a typed node's properties, recording an audit entry.

        Only the property bag is mutable; provenance (source, created_at,
        created_by) and the embedding are immutable. Omitting ``properties``
        leaves them unchanged (an audit-only touch).
        """
        request = graph_pb2.UpdateNodeRequest(
            collection=collection,
            node_id=node_id,
            actor=actor,
        )
        if properties is not None:
            request.properties_json = json.dumps(properties)
        response = self._client._call(
            self._client._graph_stub.UpdateNode, request
        )
        return _node_from_proto(response.node)

    def put_edge(
        self,
        collection: str,
        source: int,
        target: int,
        edge_type: str,
        *,
        properties: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        verified: bool = False,
        is_manual: bool = True,
        valid_from: Optional[int] = None,
        valid_until: Optional[int] = None,
        temporal_context: Optional[str] = None,
    ) -> int:
        """Create a typed edge (Hybrid collections only). Returns the edge id.

        ``valid_from`` / ``valid_until`` (unix-epoch millis, ``valid_until``
        exclusive) and ``temporal_context`` set an optional validity window and
        regime label (P17); omit them for an always-valid, context-free edge.
        """
        request = graph_pb2.PutEdgeRequest(
            collection=collection,
            source=source,
            target=target,
            edge_type=edge_type,
            properties_json=json.dumps(properties or {}),
            provenance_json=json.dumps(provenance or {}),
            confidence=confidence,
            verified=verified,
            is_manual=is_manual,
        )
        if valid_from is not None:
            request.valid_from = valid_from
        if valid_until is not None:
            request.valid_until = valid_until
        if temporal_context is not None:
            request.temporal_context = temporal_context
        response = self._client._call(self._client._graph_stub.PutEdge, request)
        return int(response.id)

    def get_edge(self, collection: str, edge_id: int) -> Optional[TypedEdge]:
        """Fetch a typed edge by id, or None if absent."""
        request = graph_pb2.GetEdgeRequest(collection=collection, id=edge_id)
        response = self._client._call(self._client._graph_stub.GetEdge, request)
        return _edge_from_proto(response.edge) if response.found else None

    def delete_edge(self, collection: str, edge_id: int) -> bool:
        """Delete a typed edge. True if it existed."""
        request = graph_pb2.DeleteEdgeRequest(collection=collection, id=edge_id)
        response = self._client._call(self._client._graph_stub.DeleteEdge, request)
        return bool(response.deleted)

    def list_edges(
        self,
        collection: str,
        node: int,
        *,
        direction: str = "outgoing",
        edge_type: str = "",
    ) -> List[TypedEdge]:
        """List typed edges incident to a node, optionally filtered by type."""
        request = graph_pb2.ListEdgesRequest(
            collection=collection,
            node=node,
            direction=direction,
            edge_type=edge_type,
        )
        response = self._client._call(self._client._graph_stub.ListEdges, request)
        return [_edge_from_proto(e) for e in response.edges]

    def update_edge(
        self,
        collection: str,
        edge_id: int,
        *,
        properties: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        verified: Optional[bool] = None,
        actor: str = "",
    ) -> TypedEdge:
        """Update a typed edge's properties, confidence, or verified flag.

        Only the supplied fields are changed; omitted fields keep their value.
        """
        request = graph_pb2.UpdateEdgeRequest(
            collection=collection,
            edge_id=edge_id,
            actor=actor,
        )
        if properties is not None:
            request.properties_json = json.dumps(properties)
        if confidence is not None:
            request.confidence = confidence
        if verified is not None:
            request.verified = verified
        response = self._client._call(
            self._client._graph_stub.UpdateEdge, request
        )
        return _edge_from_proto(response.edge)

    def verify_edge(
        self,
        collection: str,
        edge_id: int,
        *,
        actor: str = "",
    ) -> TypedEdge:
        """Mark a typed edge as verified. Returns the updated edge."""
        request = graph_pb2.VerifyEdgeRequest(
            collection=collection,
            edge_id=edge_id,
            actor=actor,
        )
        response = self._client._call(
            self._client._graph_stub.VerifyEdge, request
        )
        return _edge_from_proto(response.edge)

    def reject_edge(
        self,
        collection: str,
        edge_id: int,
        *,
        actor: str = "",
    ) -> EdgeRejectResult:
        """Reject a typed edge, deleting it and optionally adding a rule."""
        request = graph_pb2.RejectEdgeRequest(
            collection=collection,
            edge_id=edge_id,
            actor=actor,
        )
        response = self._client._call(
            self._client._graph_stub.RejectEdge, request
        )
        return EdgeRejectResult(
            deleted=bool(response.deleted),
            rule_added=bool(response.rule_added),
        )

    def bulk_import_edges(
        self,
        collection: str,
        data: str,
        *,
        format: str = "csv",
        auto_add_edge_types: bool = False,
        actor: str = "",
    ) -> BulkImportResult:
        """Bulk-import typed edges from CSV or JSONL data."""
        request = graph_pb2.BulkImportEdgesRequest(
            collection=collection,
            format=_bulk_import_format_to_proto(format),
            data=data,
            auto_add_edge_types=auto_add_edge_types,
            actor=actor,
        )
        response = self._client._call(
            self._client._graph_stub.BulkImportEdges, request
        )
        return BulkImportResult(
            total_rows=int(response.total_rows),
            imported=int(response.imported),
            failed=int(response.failed),
            errors=[
                BulkImportRowError(row=int(e.row), message=e.message)
                for e in response.errors
            ],
        )

    # ── Paginated whole-graph enumeration (ADR-014) ──

    def enumerate_nodes(
        self,
        collection: str,
        *,
        after_id: int = 0,
        limit: int = 1000,
        kind: Optional[str] = None,
        label: str = "",
        predicate: Optional[Any] = None,
    ) -> NodePage:
        """Fetch one page of nodes in ascending id order (Hybrid mode).

        ``after_id`` is the cursor (0 to start; pass back ``next_cursor``).
        ``kind`` optionally filters to "content" or "entity"; ``label`` filters
        entity nodes by label. ``predicate`` (a Predicate from the hybrid module)
        applies an arbitrary property condition, including the structural
        incident-edge-count term, with no vector input. The page size is
        server-clamped.
        """
        request = graph_pb2.EnumerateNodesRequest(
            collection=collection,
            after_id=after_id,
            limit=limit,
            label=label,
        )
        if kind is not None:
            request.filter_by_kind = True
            request.kind = _kind_to_proto(kind)
        if predicate is not None:
            request.predicate.CopyFrom(predicate)
        response = self._client._call(
            self._client._graph_stub.EnumerateNodes, request
        )
        return NodePage(
            nodes=[_node_from_proto(n) for n in response.nodes],
            next_cursor=int(response.next_cursor),
            has_more=bool(response.has_more),
        )

    def enumerate_edges(
        self,
        collection: str,
        *,
        after_id: int = 0,
        limit: int = 1000,
        edge_type: str = "",
        predicate: Optional[Any] = None,
        endpoint_label: str = "",
        endpoint_kind: Optional[str] = None,
    ) -> EdgePage:
        """Fetch one page of edges in ascending id order (Hybrid mode).

        ``after_id`` is the cursor (0 to start; pass back ``next_cursor``).
        ``edge_type`` optionally filters by type. ``predicate`` (a Predicate from
        the hybrid module) applies a property condition over edge properties.
        ``endpoint_label`` / ``endpoint_kind`` constrain by the endpoint node: an
        edge passes when an endpoint (source or target) node matches. The page
        size is server-clamped.
        """
        request = graph_pb2.EnumerateEdgesRequest(
            collection=collection,
            after_id=after_id,
            limit=limit,
            edge_type=edge_type,
            endpoint_label=endpoint_label,
        )
        if predicate is not None:
            request.predicate.CopyFrom(predicate)
        if endpoint_kind is not None:
            request.filter_by_endpoint_kind = True
            request.endpoint_kind = _kind_to_proto(endpoint_kind)
        response = self._client._call(
            self._client._graph_stub.EnumerateEdges, request
        )
        return EdgePage(
            edges=[_edge_from_proto(e) for e in response.edges],
            next_cursor=int(response.next_cursor),
            has_more=bool(response.has_more),
        )

    def iter_nodes(
        self,
        collection: str,
        *,
        page_size: int = 1000,
        kind: Optional[str] = None,
        label: str = "",
    ) -> Iterator[TypedNode]:
        """Iterate every node in the graph, walking pages to exhaustion."""
        after = 0
        while True:
            page = self.enumerate_nodes(
                collection,
                after_id=after,
                limit=page_size,
                kind=kind,
                label=label,
            )
            for node in page.nodes:
                yield node
            if not page.has_more or page.next_cursor == 0:
                break
            after = page.next_cursor

    def iter_edges(
        self,
        collection: str,
        *,
        page_size: int = 1000,
        edge_type: str = "",
    ) -> Iterator[TypedEdge]:
        """Iterate every edge in the graph, walking pages to exhaustion."""
        after = 0
        while True:
            page = self.enumerate_edges(
                collection,
                after_id=after,
                limit=page_size,
                edge_type=edge_type,
            )
            for edge in page.edges:
                yield edge
            if not page.has_more or page.next_cursor == 0:
                break
            after = page.next_cursor

    # ── Hybrid query engine (Hybrid mode) ──

    def query(self, collection: str) -> "HybridQueryBuilder":
        """Start a composable hybrid query against a collection.

        Returns a chainable builder; finish with a terminal
        ``return_nodes()`` / ``return_edges()`` / ``return_paths()``.
        """
        from .hybrid import HybridQueryBuilder

        return HybridQueryBuilder(self._client, collection)

    def graph_rag(
        self,
        collection: str,
        query_vector: List[float],
        k: int = 10,
        *,
        fusion: str = "vector_rank",
        mentions_edge_type: str = "mentions",
        relation_edge_types: Optional[List[str]] = None,
        k_hop_max: int = 2,
        rrf_k: int = 60,
        hub_damping: float = 0.0,
        ef_search: Optional[int] = None,
    ) -> "HybridQueryResult":
        """One-call graph-aware GraphRAG retrieval (the documented composed plan).

        This composes the proven GraphRAG plan so you do not have to hand-build
        it. From a vector seed it expands across the graph so the candidate pool
        includes graph-reached passages, then ranks within that pool. The
        composed plan is exactly::

            seed = vector_similar(query_vector, k)
            for each relation R in relation_edge_types:
                bridge_R = vector_similar(query_vector, k)
                            .traverse(mentions_edge_type, outgoing)
                            .k_hop(R, max=k_hop_max)
                            .traverse(mentions_edge_type, incoming)
                seed = seed.union(bridge_R)
            seed.<ranking step from fusion>.return_nodes()

        When ``relation_edge_types`` is empty/None it falls back to the
        structural form for content-to-content graphs (graphs whose edges link
        passages directly rather than through entities)::

            seed = vector_similar(query_vector, k)
            bridge = vector_similar(query_vector, k).k_hop(any, max=k_hop_max)
            seed.union(bridge).<ranking step from fusion>.return_nodes()

        The default ``fusion="vector_rank"`` is graph-first scope-then-rank: the
        graph fixes the candidate pool, then the pool is ranked EXACTLY by
        similarity to ``query_vector`` (recall 1.0, no ANN). Pass
        ``fusion="rrf"`` to opt into Reciprocal Rank Fusion instead, which fuses
        the vector and graph-proximity rankings and then applies ``rrf_k`` and
        ``hub_damping``.

        The graph expansion is what recovers gold the vector arm misses on
        multi-hop or vector-dissimilar bridges. It is OPTIONAL value that ADDS
        LATENCY; plain
        ``client.graph.query(...).vector_similar(...).return_nodes()`` stays the
        zero-cost default.

        Example::

            result = client.graph.graph_rag(
                "docs_graph", query_vector, k=10,
                relation_edge_types=["CITES"],
            )
            for node in result.nodes:
                print(node.id, node.label)

        Args:
            collection: Collection name (hybrid mode).
            query_vector: The query embedding to seed from.
            k: Final top-k cut and the seed size. Default 10.
            fusion: Ranking step over the graph-scoped pool. ``"vector_rank"``
                (default) ranks the pool exactly by similarity to
                ``query_vector``. ``"rrf"`` opts into Reciprocal Rank Fusion,
                which then uses ``rrf_k`` and ``hub_damping``.
            mentions_edge_type: Content-to-entity edge type bridged through.
                Default "mentions".
            relation_edge_types: Typed relation edge types to bridge across (for
                example ["CITES"]). Empty/None uses the structural fallback.
            k_hop_max: Bridge-hop bound for the relation walk. Default 2.
            rrf_k: RRF constant in 1 / (rrf_k + rank). Default 60 (canonical).
                Applies only when ``fusion="rrf"``.
            hub_damping: Hub-aware damping strength. Applies only when
                ``fusion="rrf"``. Default 0.0 = OFF (the validated default
                behavior). When > 0.0, bridge routes through high-degree hub
                entities are down-weighted (weight = product of
                1 / (1 + hub_damping * ln(1 + degree))), which helps hub-dense
                graphs such as conversational memory where a few entities are
                mentioned almost everywhere and otherwise flood the pool. A
                principled starting value is 1.0.
            ef_search: Optional HNSW search width for the vector seed.

        Returns:
            A HybridQueryResult whose ``nodes`` hold the ranked top-k.
        """
        from .hybrid import HybridQueryBuilder, compose_graph_rag

        seed = HybridQueryBuilder(self._client, collection)
        bridge = HybridQueryBuilder(self._client, collection)
        composed = compose_graph_rag(
            seed,
            bridge,
            query_vector,
            k,
            fusion=fusion,
            mentions_edge_type=mentions_edge_type,
            relation_edge_types=relation_edge_types,
            k_hop_max=k_hop_max,
            rrf_k=rrf_k,
            hub_damping=hub_damping,
            ef_search=ef_search,
        )
        return composed.return_nodes()
