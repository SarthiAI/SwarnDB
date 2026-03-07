"""SwarnDB asynchronous client.

This module provides the async gRPC client for interacting with a SwarnDB
server. It mirrors the synchronous SwarnDBClient API but uses async/await
for all operations. All five API namespaces (collections, vectors, search,
graph, math) are inlined as async wrapper classes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import grpc
import grpc.aio

from swarndb._proto import (
    collection_pb2,
    collection_pb2_grpc,
    common_pb2,
    graph_pb2,
    graph_pb2_grpc,
    search_pb2,
    search_pb2_grpc,
    vector_math_pb2,
    vector_math_pb2_grpc,
    vector_pb2,
    vector_pb2_grpc,
)
from swarndb._helpers import _translate_error as _translate_error_fn
from swarndb.exceptions import (
    AuthenticationError,
    CollectionExistsError,
    CollectionNotFoundError,
    ConnectionError,
    DimensionMismatchError,
    GraphError,
    MathError,
    SearchError,
    SwarnDBError,
    VectorNotFoundError,
)
from swarndb.search import Filter, _to_proto_value, _scored_result_from_proto
from swarndb.types import (
    BatchSearchResult,
    BulkInsertResult,
    ClusterAssignment,
    ClusterResult,
    CollectionInfo,
    ConeSearchResult,
    DiversityResult,
    DriftReport,
    GhostVector,
    GraphEdge,
    PCAResult,
    ScoredResult,
    SearchResult,
    TraversalNode,
    VectorRecord,
)
from swarndb._helpers import _to_proto_vector
from swarndb.vectors import _from_proto_metadata, _to_proto_metadata

logger = logging.getLogger(__name__)

# gRPC status codes eligible for retry.
_RETRYABLE_CODES = frozenset({
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
})

_VALID_STRATEGIES = frozenset({"auto", "pre_filter", "post_filter"})


# ---------------------------------------------------------------------------
# Async Auth Interceptor
# ---------------------------------------------------------------------------


class _AsyncClientCallDetails(grpc.aio.ClientCallDetails):
    """Concrete implementation of ClientCallDetails for async interceptors."""

    def __init__(self, method, timeout, metadata, credentials, wait_for_ready):
        self.method = method
        self.timeout = timeout
        self.metadata = metadata
        self.credentials = credentials
        self.wait_for_ready = wait_for_ready


class _AsyncAuthInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
    """Async interceptor that injects an API key into every unary-unary call."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def intercept_unary_unary(
        self,
        continuation,
        client_call_details,
        request,
    ):
        """Add X-API-Key metadata to the outgoing call."""
        metadata: list[tuple[str, str]] = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)
        metadata.append(("x-api-key", self._api_key))

        new_details = _AsyncClientCallDetails(
            method=client_call_details.method,
            timeout=client_call_details.timeout,
            metadata=metadata,
            credentials=client_call_details.credentials,
            wait_for_ready=client_call_details.wait_for_ready,
        )
        return await continuation(new_details, request)


# ---------------------------------------------------------------------------
# AsyncSwarnDBClient
# ---------------------------------------------------------------------------


class AsyncSwarnDBClient:
    """Asynchronous SwarnDB client with gRPC connection management.

    Provides access to all SwarnDB operations through lazy-initialized
    async API namespaces: ``collections``, ``vectors``, ``search``,
    ``graph``, and ``math``.

    Usage::

        async with AsyncSwarnDBClient("localhost", 50051) as client:
            await client.collections.create("my_collection", dimension=128)
            vid = await client.vectors.insert("my_collection", [0.1] * 128)
            results = await client.search.query("my_collection", [0.1] * 128, k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        *,
        api_key: Optional[str] = None,
        secure: bool = False,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        timeout: float = 30.0,
        options: Optional[Sequence[Tuple[str, Any]]] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._secure = secure
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout

        target = f"{host}:{port}"
        channel_options = list(options) if options else []

        # Build interceptor list
        interceptors: list[grpc.aio.ClientInterceptor] = []
        if api_key:
            interceptors.append(_AsyncAuthInterceptor(api_key))

        # Create async channel
        if secure:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.aio.secure_channel(
                target, credentials,
                options=channel_options,
                interceptors=interceptors or None,
            )
        else:
            self._channel = grpc.aio.insecure_channel(
                target,
                options=channel_options,
                interceptors=interceptors or None,
            )

        # Create all 5 service stubs (same stub classes work with aio channels)
        self._collection_stub = collection_pb2_grpc.CollectionServiceStub(self._channel)
        self._vector_stub = vector_pb2_grpc.VectorServiceStub(self._channel)
        self._search_stub = search_pb2_grpc.SearchServiceStub(self._channel)
        self._graph_stub = graph_pb2_grpc.GraphServiceStub(self._channel)
        self._vector_math_stub = vector_math_pb2_grpc.VectorMathServiceStub(self._channel)

        # Lazy-initialized API facades
        self._collections: Optional[AsyncCollectionAPI] = None
        self._vectors: Optional[AsyncVectorAPI] = None
        self._search: Optional[AsyncSearchAPI] = None
        self._graph: Optional[AsyncGraphAPI] = None
        self._math: Optional[AsyncMathAPI] = None

    # ------------------------------------------------------------------
    # API namespace properties (lazy initialization)
    # ------------------------------------------------------------------

    @property
    def collections(self) -> AsyncCollectionAPI:
        """Access async collection management operations."""
        if self._collections is None:
            self._collections = AsyncCollectionAPI(self)
        return self._collections

    @property
    def vectors(self) -> AsyncVectorAPI:
        """Access async vector CRUD operations."""
        if self._vectors is None:
            self._vectors = AsyncVectorAPI(self)
        return self._vectors

    @property
    def search(self) -> AsyncSearchAPI:
        """Access async search operations."""
        if self._search is None:
            self._search = AsyncSearchAPI(self)
        return self._search

    @property
    def graph(self) -> AsyncGraphAPI:
        """Access async graph operations."""
        if self._graph is None:
            self._graph = AsyncGraphAPI(self)
        return self._graph

    @property
    def math(self) -> AsyncMathAPI:
        """Access async vector math operations."""
        if self._math is None:
            self._math = AsyncMathAPI(self)
        return self._math

    # ------------------------------------------------------------------
    # Internal call helpers
    # ------------------------------------------------------------------

    async def _call(self, stub_method, request, *, timeout: Optional[float] = None):
        """Execute an async gRPC call with retry logic and error handling.

        Wraps gRPC errors into SwarnDB exceptions. Implements exponential
        backoff retry for transient errors (UNAVAILABLE, DEADLINE_EXCEEDED).

        Args:
            stub_method: The gRPC stub method to invoke.
            request: The protobuf request object.
            timeout: Optional per-call timeout override (seconds).

        Returns:
            The protobuf response object.

        Raises:
            SwarnDBError: On any non-retryable gRPC failure.
        """
        call_timeout = timeout if timeout is not None else self._timeout
        last_error: Optional[grpc.RpcError] = None

        for attempt in range(self._max_retries + 1):
            try:
                return await stub_method(
                    request,
                    timeout=call_timeout,
                )
            except grpc.RpcError as exc:
                last_error = exc
                code = exc.code()

                # Only retry on transient errors
                if code in _RETRYABLE_CODES and attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.debug(
                        "Retrying async gRPC call (attempt %d/%d) after %s "
                        "error, backoff %.2fs",
                        attempt + 1,
                        self._max_retries,
                        code.name,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable or exhausted retries
                raise self._translate_error(exc) from exc

        # Should not reach here, but guard anyway
        assert last_error is not None
        raise self._translate_error(last_error) from last_error

    def _metadata(self) -> Optional[Sequence[Tuple[str, str]]]:
        """Return auth metadata if api_key is set."""
        if self._api_key:
            return (("x-api-key", self._api_key),)
        return None

    @staticmethod
    def _translate_error(exc: grpc.RpcError) -> SwarnDBError:
        """Map a gRPC RpcError to the appropriate SwarnDB exception."""
        return _translate_error_fn(exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the async gRPC channel."""
        await self._channel.close()

    async def __aenter__(self) -> AsyncSwarnDBClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"AsyncSwarnDBClient(host={self._host!r}, port={self._port}, "
            f"secure={self._secure})"
        )


# ---------------------------------------------------------------------------
# AsyncCollectionAPI
# ---------------------------------------------------------------------------


class AsyncCollectionAPI:
    """Async wrapper around the CollectionService gRPC API."""

    def __init__(self, client: AsyncSwarnDBClient) -> None:
        self._client = client

    async def create(
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
        await self._client._call(
            self._client._collection_stub.CreateCollection, request
        )
        return CollectionInfo(
            name=name,
            dimension=dimension,
            distance_metric=distance_metric,
            vector_count=0,
            default_threshold=default_threshold,
        )

    async def get(self, name: str) -> CollectionInfo:
        """Get collection metadata.

        Args:
            name: Collection name.

        Returns:
            CollectionInfo for the requested collection.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        request = collection_pb2.GetCollectionRequest(name=name)
        response = await self._client._call(
            self._client._collection_stub.GetCollection, request
        )
        return CollectionInfo(
            name=response.name,
            dimension=response.dimension,
            distance_metric=response.distance_metric,
            vector_count=response.vector_count,
            default_threshold=response.default_threshold,
        )

    async def delete(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name.

        Returns:
            True on success.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        request = collection_pb2.DeleteCollectionRequest(name=name)
        response = await self._client._call(
            self._client._collection_stub.DeleteCollection, request
        )
        return bool(response.success)

    async def list(self) -> List[CollectionInfo]:
        """List all collections.

        Returns:
            List of CollectionInfo for every collection on the server.
        """
        request = collection_pb2.ListCollectionsRequest()
        response = await self._client._call(
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

    async def exists(self, name: str) -> bool:
        """Check whether a collection exists.

        Args:
            name: Collection name.

        Returns:
            True if the collection exists, False otherwise.
        """
        try:
            await self.get(name)
            return True
        except CollectionNotFoundError:
            return False


# ---------------------------------------------------------------------------
# AsyncVectorAPI
# ---------------------------------------------------------------------------


class AsyncVectorAPI:
    """Async vector CRUD operations for a SwarnDB collection."""

    def __init__(self, client: AsyncSwarnDBClient) -> None:
        self._client = client

    async def insert(
        self,
        collection: str,
        vector: List[float],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        id: int = 0,
    ) -> int:
        """Insert a vector into a collection.

        Args:
            collection: Target collection name.
            vector: The vector values as a list of floats.
            metadata: Optional metadata dict to attach to the vector.
            id: Optional vector ID. Pass 0 (default) for auto-assignment.

        Returns:
            The assigned vector ID.
        """
        request = vector_pb2.InsertRequest(
            collection=collection,
            vector=_to_proto_vector(vector),
            id=id,
        )
        if metadata:
            request.metadata.CopyFrom(_to_proto_metadata(metadata))

        response = await self._client._call(
            self._client._vector_stub.Insert, request
        )
        return response.id

    async def get(self, collection: str, id: int) -> VectorRecord:
        """Get a vector by ID.

        Args:
            collection: Collection name.
            id: Vector ID to retrieve.

        Returns:
            A VectorRecord with the vector data and metadata.

        Raises:
            VectorNotFoundError: If the vector does not exist.
        """
        request = vector_pb2.GetVectorRequest(
            collection=collection,
            id=id,
        )
        response = await self._client._call(
            self._client._vector_stub.Get, request
        )
        return VectorRecord(
            id=str(response.id),
            vector=list(response.vector.values),
            metadata=_from_proto_metadata(response.metadata),
        )

    async def update(
        self,
        collection: str,
        id: int,
        *,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a vector's data and/or metadata.

        Args:
            collection: Collection name.
            id: Vector ID to update.
            vector: New vector values (optional).
            metadata: New metadata dict (optional).

        Returns:
            True if the update was successful.
        """
        request = vector_pb2.UpdateRequest(
            collection=collection,
            id=id,
        )
        if vector is not None:
            request.vector.CopyFrom(_to_proto_vector(vector))
        if metadata is not None:
            request.metadata.CopyFrom(_to_proto_metadata(metadata))

        response = await self._client._call(
            self._client._vector_stub.Update, request
        )
        return response.success

    async def delete(self, collection: str, id: int) -> bool:
        """Delete a vector by ID.

        Args:
            collection: Collection name.
            id: Vector ID to delete.

        Returns:
            True if the deletion was successful.
        """
        request = vector_pb2.DeleteVectorRequest(
            collection=collection,
            id=id,
        )
        response = await self._client._call(
            self._client._vector_stub.Delete, request
        )
        return response.success

    async def bulk_insert(
        self,
        collection: str,
        vectors: List[List[float]],
        *,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
        batch_size: int = 1000,
    ) -> BulkInsertResult:
        """Bulk insert vectors using async streaming RPC.

        Streams InsertRequest messages to the server via an async generator.

        Args:
            collection: Target collection name.
            vectors: List of vector value lists.
            metadata_list: Optional per-vector metadata dicts (must match
                length of ``vectors`` if provided).
            ids: Optional per-vector IDs (must match length of ``vectors``
                if provided). Use 0 for auto-assignment.
            batch_size: Number of vectors per streaming batch.

        Returns:
            A BulkInsertResult with inserted_count and any errors.

        Raises:
            ValueError: If metadata_list or ids length doesn't match vectors.
        """
        total = len(vectors)

        if metadata_list is not None and len(metadata_list) != total:
            raise ValueError(
                f"metadata_list length ({len(metadata_list)}) must match "
                f"vectors length ({total})"
            )
        if ids is not None and len(ids) != total:
            raise ValueError(
                f"ids length ({len(ids)}) must match vectors length ({total})"
            )

        async def _request_iterator() -> AsyncIterator[vector_pb2.InsertRequest]:
            """Yield InsertRequest messages asynchronously."""
            for i in range(total):
                req = vector_pb2.InsertRequest(
                    collection=collection,
                    vector=_to_proto_vector(vectors[i]),
                    id=ids[i] if ids is not None else 0,
                )
                if metadata_list is not None and metadata_list[i] is not None:
                    req.metadata.CopyFrom(
                        _to_proto_metadata(metadata_list[i])
                    )
                yield req

        # BulkInsert is stream_unary: stream requests, get one response.
        # Cannot use _call (unary-unary), so call stub directly with retry.
        metadata = self._client._metadata()
        call_timeout = self._client._timeout
        last_error: Optional[grpc.RpcError] = None

        for attempt in range(self._client._max_retries + 1):
            try:
                response = await self._client._vector_stub.BulkInsert(
                    _request_iterator(),
                    timeout=call_timeout,
                    metadata=metadata,
                )
                return BulkInsertResult(
                    inserted_count=response.inserted_count,
                    errors=list(response.errors),
                )
            except grpc.RpcError as exc:
                last_error = exc
                code = exc.code()

                if code in _RETRYABLE_CODES and attempt < self._client._max_retries:
                    delay = self._client._retry_delay * (2 ** attempt)
                    logger.debug(
                        "Retrying async BulkInsert (attempt %d/%d) after %s, "
                        "backoff %.2fs",
                        attempt + 1,
                        self._client._max_retries,
                        code.name,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                raise self._client._translate_error(exc) from exc

        assert last_error is not None
        raise self._client._translate_error(last_error) from last_error


# ---------------------------------------------------------------------------
# AsyncSearchAPI
# ---------------------------------------------------------------------------


class AsyncSearchAPI:
    """Async search facade wrapping the gRPC SearchService."""

    def __init__(self, client: AsyncSwarnDBClient) -> None:
        self._client = client

    async def query(
        self,
        collection: str,
        vector: List[float],
        k: int = 10,
        *,
        filter: Optional[Filter] = None,
        strategy: str = "auto",
        include_metadata: bool = True,
        include_graph: bool = False,
        graph_threshold: float = 0.0,
        max_graph_edges: int = 10,
    ) -> SearchResult:
        """Search for nearest neighbors.

        Args:
            collection: Name of the collection to search.
            vector: Query vector as a list of floats.
            k: Number of nearest neighbors to return.
            filter: Optional metadata filter built via the Filter class.
            strategy: Filter strategy -- "auto", "pre_filter", or "post_filter".
            include_metadata: Whether to include metadata in results.
            include_graph: Whether to include graph edges in results.
            graph_threshold: Minimum similarity for graph edges.
            max_graph_edges: Maximum number of graph edges per result.

        Returns:
            SearchResult with ``.results`` list and ``.search_time_us``.

        Raises:
            SearchError: If the search operation fails.
            ValueError: If an invalid strategy is provided.
        """
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Invalid search strategy {strategy!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_STRATEGIES))}"
            )

        request = search_pb2.SearchRequest(
            collection=collection,
            query=common_pb2.Vector(values=vector),
            k=k,
            strategy=strategy,
            include_metadata=include_metadata,
            include_graph=include_graph,
            graph_threshold=graph_threshold,
            max_graph_edges=max_graph_edges,
        )
        if filter is not None:
            request.filter.CopyFrom(filter._to_proto())

        response = await self._client._call(
            self._client._search_stub.Search,
            request,
        )

        results = [_scored_result_from_proto(r) for r in response.results]
        return SearchResult(
            results=results,
            search_time_us=response.search_time_us,
        )

    async def batch(
        self,
        collection: str,
        queries: List[List[float]],
        k: int = 10,
        *,
        filter: Optional[Filter] = None,
        strategy: str = "auto",
        include_metadata: bool = True,
        include_graph: bool = False,
        graph_threshold: float = 0.0,
        max_graph_edges: int = 10,
    ) -> BatchSearchResult:
        """Batch search multiple queries against a collection.

        Args:
            collection: Name of the collection to search.
            queries: List of query vectors, each a list of floats.
            k: Number of nearest neighbors to return per query.
            filter: Optional metadata filter applied to all queries.
            strategy: Filter strategy -- "auto", "pre_filter", or "post_filter".
            include_metadata: Whether to include metadata in results.
            include_graph: Whether to include graph edges in results.
            graph_threshold: Minimum similarity for graph edges.
            max_graph_edges: Maximum number of graph edges per result.

        Returns:
            BatchSearchResult with ``.results`` (list of SearchResult)
            and ``.total_time_us``.

        Raises:
            SearchError: If the batch search operation fails.
            ValueError: If an invalid strategy is provided.
        """
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Invalid search strategy {strategy!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_STRATEGIES))}"
            )

        proto_filter = filter._to_proto() if filter is not None else None

        search_requests = []
        for query_vec in queries:
            req = search_pb2.SearchRequest(
                collection=collection,
                query=common_pb2.Vector(values=query_vec),
                k=k,
                strategy=strategy,
                include_metadata=include_metadata,
                include_graph=include_graph,
                graph_threshold=graph_threshold,
                max_graph_edges=max_graph_edges,
            )
            if proto_filter is not None:
                req.filter.CopyFrom(proto_filter)
            search_requests.append(req)

        batch_request = search_pb2.BatchSearchRequest(queries=search_requests)

        response = await self._client._call(
            self._client._search_stub.BatchSearch,
            batch_request,
        )

        search_results = []
        for sr in response.results:
            results = [_scored_result_from_proto(r) for r in sr.results]
            search_results.append(
                SearchResult(results=results, search_time_us=sr.search_time_us)
            )

        return BatchSearchResult(
            results=search_results,
            total_time_us=response.total_time_us,
        )


# ---------------------------------------------------------------------------
# AsyncGraphAPI
# ---------------------------------------------------------------------------


class AsyncGraphAPI:
    """Async wrapper around the GraphService gRPC API."""

    def __init__(self, client: AsyncSwarnDBClient) -> None:
        self._client = client

    async def get_related(
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
        response = await self._client._call(
            self._client._graph_stub.GetRelated, request
        )
        return [
            GraphEdge(
                target_id=edge.target_id,
                similarity=edge.similarity,
            )
            for edge in response.edges
        ]

    async def traverse(
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
        response = await self._client._call(
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

    async def set_threshold(
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
        response = await self._client._call(
            self._client._graph_stub.SetThreshold, request
        )
        return bool(response.success)


# ---------------------------------------------------------------------------
# AsyncMathAPI
# ---------------------------------------------------------------------------


class AsyncMathAPI:
    """Async wrapper around the VectorMathService gRPC API."""

    def __init__(self, client: AsyncSwarnDBClient) -> None:
        self._client = client

    async def detect_ghosts(
        self,
        collection: str,
        threshold: float,
        *,
        centroids: Optional[List[List[float]]] = None,
        auto_k: int = 8,
        metric: str = "euclidean",
    ) -> List[GhostVector]:
        """Detect isolated 'ghost' vectors far from any cluster centroid.

        Args:
            collection: Collection name.
            threshold: Distance threshold above which a vector is a ghost.
            centroids: Optional explicit centroid vectors. If omitted the
                server auto-computes centroids using ``auto_k``.
            auto_k: Number of centroids to auto-compute when ``centroids``
                is not provided.
            metric: Distance metric (e.g. ``"euclidean"``).

        Returns:
            List of GhostVector(id, isolation_score).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        proto_centroids = (
            [_to_proto_vector(c) for c in centroids] if centroids else []
        )
        request = vector_math_pb2.DetectGhostsRequest(
            collection=collection,
            threshold=threshold,
            centroids=proto_centroids,
            auto_k=auto_k,
            metric=metric,
        )
        response = await self._client._call(
            self._client._vector_math_stub.DetectGhosts, request
        )
        return [
            GhostVector(
                id=g.id,
                isolation_score=g.isolation_score,
            )
            for g in response.ghosts
        ]

    async def cone_search(
        self,
        collection: str,
        direction: List[float],
        aperture_radians: float,
    ) -> List[ConeSearchResult]:
        """Find vectors within an angular cone around a direction vector.

        Args:
            collection: Collection name.
            direction: Unit direction vector defining the cone axis.
            aperture_radians: Half-angle of the cone in radians.

        Returns:
            List of ConeSearchResult(id, cosine_similarity, angle_radians).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ConeSearchRequest(
            collection=collection,
            direction=_to_proto_vector(direction),
            aperture_radians=aperture_radians,
        )
        response = await self._client._call(
            self._client._vector_math_stub.ConeSearch, request
        )
        return [
            ConeSearchResult(
                id=r.id,
                cosine_similarity=r.cosine_similarity,
                angle_radians=r.angle_radians,
            )
            for r in response.results
        ]

    async def centroid(
        self,
        collection: str,
        *,
        vector_ids: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute the (optionally weighted) centroid of vectors.

        Args:
            collection: Collection name.
            vector_ids: IDs of vectors to include. If omitted, uses all
                vectors in the collection.
            weights: Optional per-vector weights for a weighted centroid.

        Returns:
            Centroid as a list of floats.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ComputeCentroidRequest(
            collection=collection,
            vector_ids=vector_ids or [],
            weights=weights or [],
        )
        response = await self._client._call(
            self._client._vector_math_stub.ComputeCentroid, request
        )
        return list(response.centroid.values)

    async def interpolate(
        self,
        a: List[float],
        b: List[float],
        t: float = 0.5,
        *,
        method: str = "lerp",
    ) -> List[float]:
        """Interpolate between two vectors at parameter t.

        Args:
            a: Start vector.
            b: End vector.
            t: Interpolation parameter in [0, 1].
            method: ``"lerp"`` for linear or ``"slerp"`` for spherical.

        Returns:
            Interpolated vector as a list of floats.

        Raises:
            MathError: If the operation fails.
        """
        request = vector_math_pb2.InterpolateRequest(
            a=_to_proto_vector(a),
            b=_to_proto_vector(b),
            t=t,
            method=method,
            sequence_count=0,
        )
        response = await self._client._call(
            self._client._vector_math_stub.Interpolate, request
        )
        return list(response.results[0].values)

    async def interpolate_sequence(
        self,
        a: List[float],
        b: List[float],
        n: int,
        *,
        method: str = "lerp",
    ) -> List[List[float]]:
        """Generate a sequence of n interpolated vectors between a and b.

        Args:
            a: Start vector.
            b: End vector.
            n: Number of interpolation steps.
            method: ``"lerp"`` for linear or ``"slerp"`` for spherical.

        Returns:
            List of interpolated vectors.

        Raises:
            MathError: If the operation fails.
        """
        request = vector_math_pb2.InterpolateRequest(
            a=_to_proto_vector(a),
            b=_to_proto_vector(b),
            t=0.0,
            method=method,
            sequence_count=n,
        )
        response = await self._client._call(
            self._client._vector_math_stub.Interpolate, request
        )
        return [list(v.values) for v in response.results]

    async def detect_drift(
        self,
        collection: str,
        window1_ids: List[int],
        window2_ids: List[int],
        *,
        metric: str = "euclidean",
        threshold: float = 0.0,
    ) -> DriftReport:
        """Detect distribution drift between two temporal windows of vectors.

        Args:
            collection: Collection name.
            window1_ids: Vector IDs for the first (baseline) window.
            window2_ids: Vector IDs for the second (comparison) window.
            metric: Distance metric (e.g. ``"euclidean"``).
            threshold: Drift threshold; if centroid shift exceeds this
                the report marks ``has_drifted`` as True.

        Returns:
            DriftReport with centroid shift, spread change, and drift flag.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.DetectDriftRequest(
            collection=collection,
            window1_ids=window1_ids,
            window2_ids=window2_ids,
            metric=metric,
            threshold=threshold,
        )
        response = await self._client._call(
            self._client._vector_math_stub.DetectDrift, request
        )
        return DriftReport(
            centroid_shift=response.centroid_shift,
            mean_distance_window1=response.mean_distance_window1,
            mean_distance_window2=response.mean_distance_window2,
            spread_change=response.spread_change,
            has_drifted=response.has_drifted,
        )

    async def cluster(
        self,
        collection: str,
        k: int,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        metric: str = "euclidean",
    ) -> ClusterResult:
        """Run k-means clustering on vectors in a collection.

        Args:
            collection: Collection name.
            k: Number of clusters.
            max_iterations: Maximum iteration count.
            tolerance: Convergence tolerance.
            metric: Distance metric (e.g. ``"euclidean"``).

        Returns:
            ClusterResult with centroids, assignments, iteration count,
            and convergence flag.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ClusterRequest(
            collection=collection,
            k=k,
            max_iterations=max_iterations,
            tolerance=tolerance,
            metric=metric,
        )
        response = await self._client._call(
            self._client._vector_math_stub.Cluster, request
        )
        return ClusterResult(
            centroids=[list(c.values) for c in response.centroids],
            assignments=[
                ClusterAssignment(
                    id=a.id,
                    cluster=a.cluster,
                    distance_to_centroid=a.distance_to_centroid,
                )
                for a in response.assignments
            ],
            iterations=response.iterations,
            converged=response.converged,
        )

    async def reduce_dimensions(
        self,
        collection: str,
        n_components: int,
        *,
        vector_ids: Optional[List[int]] = None,
    ) -> PCAResult:
        """Perform PCA dimensionality reduction on collection vectors.

        Args:
            collection: Collection name.
            n_components: Number of principal components to keep.
            vector_ids: Optional subset of vector IDs. If omitted, uses
                all vectors in the collection.

        Returns:
            PCAResult with components, explained variance, mean, and
            projected vectors.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ReduceDimensionsRequest(
            collection=collection,
            n_components=n_components,
            vector_ids=vector_ids or [],
        )
        response = await self._client._call(
            self._client._vector_math_stub.ReduceDimensions, request
        )
        return PCAResult(
            components=[list(c.values) for c in response.components],
            explained_variance=list(response.explained_variance),
            mean=list(response.mean.values),
            projected=[list(p.values) for p in response.projected],
        )

    async def analogy(
        self,
        a: List[float],
        b: List[float],
        c: List[float],
        *,
        normalize: bool = True,
    ) -> List[float]:
        """Compute vector analogy: a - b + c.

        Args:
            a: First vector (the "is to" side).
            b: Second vector (the "as" side).
            c: Third vector (the query side).
            normalize: Whether to L2-normalize the result.

        Returns:
            Result vector as a list of floats.

        Raises:
            MathError: If the operation fails.
        """
        request = vector_math_pb2.ComputeAnalogyRequest(
            a=_to_proto_vector(a),
            b=_to_proto_vector(b),
            c=_to_proto_vector(c),
            normalize=normalize,
        )
        response = await self._client._call(
            self._client._vector_math_stub.ComputeAnalogy, request
        )
        return list(response.result.values)

    async def weighted_sum(
        self,
        vectors: List[List[float]],
        weights: List[float],
        *,
        normalize: bool = True,
    ) -> List[float]:
        """Compute a weighted sum of vectors using the analogy endpoint.

        Uses the ``terms`` field of ComputeAnalogyRequest to send
        arbitrary (vector, weight) pairs.

        Args:
            vectors: List of vectors to combine.
            weights: Corresponding weight for each vector.
            normalize: Whether to L2-normalize the result.

        Returns:
            Result vector as a list of floats.

        Raises:
            ValueError: If vectors and weights have different lengths.
            MathError: If the operation fails.
        """
        if len(vectors) != len(weights):
            raise ValueError(
                f"vectors and weights must have the same length, "
                f"got {len(vectors)} and {len(weights)}"
            )
        dim = len(vectors[0]) if vectors else 0
        zero = [0.0] * dim
        terms = [
            vector_math_pb2.ArithmeticTerm(
                vector=_to_proto_vector(v),
                weight=w,
            )
            for v, w in zip(vectors, weights)
        ]
        request = vector_math_pb2.ComputeAnalogyRequest(
            a=_to_proto_vector(zero),
            b=_to_proto_vector(zero),
            c=_to_proto_vector(zero),
            normalize=normalize,
            terms=terms,
        )
        response = await self._client._call(
            self._client._vector_math_stub.ComputeAnalogy, request
        )
        return list(response.result.values)

    async def diversity_sample(
        self,
        collection: str,
        query: List[float],
        k: int,
        *,
        lambda_: float = 0.5,
        candidate_ids: Optional[List[int]] = None,
    ) -> List[DiversityResult]:
        """Maximal Marginal Relevance diversity sampling.

        Selects k vectors that balance relevance to the query with
        diversity among the selected set.

        Args:
            collection: Collection name.
            query: Query vector.
            k: Number of vectors to select.
            lambda_: Trade-off parameter in [0, 1]. Higher values favour
                relevance; lower values favour diversity.
            candidate_ids: Optional subset of candidate vector IDs to
                consider. If omitted, considers all vectors.

        Returns:
            List of DiversityResult(id, relevance_score, mmr_score).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            MathError: If the operation fails.
        """
        request = vector_math_pb2.DiversitySampleRequest(
            collection=collection,
            query=_to_proto_vector(query),
            k=k,
            candidate_ids=candidate_ids or [],
            **{"lambda": lambda_},
        )
        response = await self._client._call(
            self._client._vector_math_stub.DiversitySample, request
        )
        return [
            DiversityResult(
                id=r.id,
                relevance_score=r.relevance_score,
                mmr_score=r.mmr_score,
            )
            for r in response.results
        ]
