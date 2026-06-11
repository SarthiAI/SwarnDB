"""SwarnDB asynchronous client.

This module provides the async gRPC client for interacting with a SwarnDB
server. It mirrors the synchronous SwarnDBClient API but uses async/await
for all operations. All five API namespaces (collections, vectors, search,
graph, math) are inlined as async wrapper classes.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
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
    extraction_pb2_grpc,
    graph_pb2,
    graph_pb2_grpc,
    search_pb2,
    search_pb2_grpc,
    vector_math_pb2,
    vector_math_pb2_grpc,
    vector_pb2,
    vector_pb2_grpc,
)
from swarndb.client import (
    _DEFAULT_MAX_MESSAGE_BYTES,
    _DEFAULT_REST_PORT,
    _recovery_path_name,
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
from swarndb.graph import (
    _bulk_import_format_to_proto,
    _edge_from_proto,
    _kind_to_proto,
    _node_from_proto,
)

if TYPE_CHECKING:
    from swarndb.hybrid import AsyncHybridQueryBuilder
    from swarndb.extraction import AsyncExtractionAPI
    from swarndb.types import HybridQueryResult
from swarndb.types import (
    BatchSearchResult,
    BulkImportResult,
    BulkImportRowError,
    BulkInsertOptions,
    BulkInsertResult,
    ClusterAssignment,
    ClusterResult,
    CollectionInfo,
    CollectionMetrics,
    CompactResult,
    ConeSearchResult,
    DiversityResult,
    DriftReport,
    EdgePage,
    EdgeRejectResult,
    GhostVector,
    GraphEdge,
    NodePage,
    HealthStatus,
    TypedEdge,
    TypedNode,
    OptimizeResult,
    PCAResult,
    PersistenceStatus,
    PruneWALResult,
    QuantizationConfig,
    ReadinessStatus,
    RecoveryStatus,
    ScalarQuantizationConfig,
    ScoredResult,
    SearchQuantizationParams,
    SearchResult,
    TraversalNode,
    VectorRecord,
)
from swarndb._helpers import _to_proto_vector
from swarndb.vectors import (
    _BULK_FROM_PATH_MIN_DEADLINE,
    _assigned_ids_view,
    _from_proto_metadata,
    _scaled_bulk_timeout,
    _to_proto_metadata,
)
from swarndb.collections import _MODE_TO_PROTO, _maintenance_timeout

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

    Concurrency:
        This client is concurrent by design: many awaited calls (for example
        via ``asyncio.gather``) are in flight at once without blocking a
        thread. They still share the channel's HTTP/2 connection, which caps
        how many requests can be in flight before further ones queue. For a
        large fan-out, raise ``channels=N`` to spread requests across ``N``
        independent channels, round robin, so they do not queue behind a
        single channel's stream limit.

    Args:
        channels: Number of independent gRPC channels to spread requests
            across, round robin. ``1`` (default) keeps the prior single-channel
            behavior exactly. Raise it for a large concurrent fan-out.
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
        rest_port: int = _DEFAULT_REST_PORT,
        channels: int = 1,
    ) -> None:
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        self._host = host
        self._port = port
        self._rest_port = rest_port
        self._api_key = api_key
        self._secure = secure
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._options = tuple(options) if options else None
        self._num_channels = channels

        target = f"{host}:{port}"
        channel_options: list[Tuple[str, Any]] = [
            ("grpc.max_receive_message_length", _DEFAULT_MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", _DEFAULT_MAX_MESSAGE_BYTES),
        ]
        if options:
            channel_options.extend(options)

        # Build interceptor list
        interceptors: list[grpc.aio.ClientInterceptor] = []
        if api_key:
            interceptors.append(_AsyncAuthInterceptor(api_key))

        # Build the channel pool. One channel keeps the prior behavior; more
        # channels let a large concurrent fan-out spread out instead of
        # queueing behind a single channel's HTTP/2 stream limit.
        #
        # With >1 channel, each gets its own subchannel pool so identical
        # target+options do not coalesce onto one shared HTTP/2 connection in
        # gRPC's global pool. Single-channel callers keep byte-identical options.
        per_channel_options = channel_options
        if channels > 1:
            per_channel_options = channel_options + [
                ("grpc.use_local_subchannel_pool", 1),
            ]
        self._channels: List[grpc.aio.Channel] = []
        for _ in range(channels):
            if secure:
                credentials = grpc.ssl_channel_credentials()
                ch = grpc.aio.secure_channel(
                    target, credentials,
                    options=per_channel_options,
                    interceptors=interceptors or None,
                )
            else:
                ch = grpc.aio.insecure_channel(
                    target,
                    options=per_channel_options,
                    interceptors=interceptors or None,
                )
            self._channels.append(ch)

        # First channel kept for single-channel callers and lifecycle code.
        self._channel = self._channels[0]

        # One stub set per channel, plus a round-robin cursor. The event loop
        # runs coroutines on one thread, so the cursor needs no lock.
        self._collection_stubs = [
            collection_pb2_grpc.CollectionServiceStub(ch) for ch in self._channels
        ]
        self._vector_stubs = [
            vector_pb2_grpc.VectorServiceStub(ch) for ch in self._channels
        ]
        self._search_stubs = [
            search_pb2_grpc.SearchServiceStub(ch) for ch in self._channels
        ]
        self._graph_stubs = [
            graph_pb2_grpc.GraphServiceStub(ch) for ch in self._channels
        ]
        self._extraction_stubs = [
            extraction_pb2_grpc.ExtractionServiceStub(ch) for ch in self._channels
        ]
        self._vector_math_stubs = [
            vector_math_pb2_grpc.VectorMathServiceStub(ch) for ch in self._channels
        ]
        self._rr_counter = itertools.count()

        # Lazy-initialized API facades
        self._collections: Optional[AsyncCollectionAPI] = None
        self._vectors: Optional[AsyncVectorAPI] = None
        self._search: Optional[AsyncSearchAPI] = None
        self._graph: Optional[AsyncGraphAPI] = None
        self._extraction: Optional["AsyncExtractionAPI"] = None
        self._math: Optional[AsyncMathAPI] = None

    # ------------------------------------------------------------------
    # Round-robin stub access (channel pool)
    # ------------------------------------------------------------------
    #
    # The whole async SDK reads these as ``client._search_stub`` etc. With one
    # channel they return that single stub (prior behavior). With more channels
    # each read advances a cursor so successive calls ride different channels.

    def _next_index(self) -> int:
        if self._num_channels == 1:
            return 0
        return next(self._rr_counter) % self._num_channels

    @property
    def _collection_stub(self):
        return self._collection_stubs[self._next_index()]

    @property
    def _vector_stub(self):
        return self._vector_stubs[self._next_index()]

    @property
    def _search_stub(self):
        return self._search_stubs[self._next_index()]

    @property
    def _graph_stub(self):
        return self._graph_stubs[self._next_index()]

    @property
    def _extraction_stub(self):
        return self._extraction_stubs[self._next_index()]

    @property
    def _vector_math_stub(self):
        return self._vector_math_stubs[self._next_index()]

    def clone(self) -> "AsyncSwarnDBClient":
        """Return a new async client with its own channel(s) and same settings.

        The returned client is fully independent: closing one does not close
        the other.
        """
        return AsyncSwarnDBClient(
            self._host,
            self._port,
            api_key=self._api_key,
            secure=self._secure,
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
            timeout=self._timeout,
            options=self._options,
            rest_port=self._rest_port,
            channels=self._num_channels,
        )

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
    def extraction(self) -> "AsyncExtractionAPI":
        """Access async LLM extraction operations (Hybrid mode)."""
        if self._extraction is None:
            from swarndb.extraction import AsyncExtractionAPI
            self._extraction = AsyncExtractionAPI(self)
        return self._extraction

    @property
    def math(self) -> AsyncMathAPI:
        """Access async vector math operations."""
        if self._math is None:
            self._math = AsyncMathAPI(self)
        return self._math

    # ------------------------------------------------------------------
    # Internal call helpers
    # ------------------------------------------------------------------

    async def _call(
        self,
        stub_method,
        request,
        *,
        timeout: Optional[float] = None,
        retry_deadline: bool = True,
    ):
        """Execute an async gRPC call with retry logic and error handling.

        Wraps gRPC errors into SwarnDB exceptions. Implements exponential
        backoff retry for transient errors (UNAVAILABLE, DEADLINE_EXCEEDED).

        Args:
            stub_method: The gRPC stub method to invoke.
            request: The protobuf request object.
            timeout: Optional per-call timeout override (seconds).
            retry_deadline: When False, a DEADLINE_EXCEEDED is treated as
                non-retryable (UNAVAILABLE retries still apply). Use for
                non-idempotent calls where a retry would re-run server-side
                work. Defaults to True to preserve existing behavior.

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

                retryable = code in _RETRYABLE_CODES
                if not retry_deadline and code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    retryable = False

                # Only retry on transient errors
                if retryable and attempt < self._max_retries:
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

    async def _rest_probe(self, path: str) -> Tuple[int, Dict[str, Any]]:
        """Async wrapper around the sync stdlib HTTP probe call.

        Delegates the blocking urllib call to a worker thread so the event
        loop stays unblocked. Error semantics mirror the sync client.
        """
        import json
        import urllib.error
        import urllib.request

        scheme = "https" if self._secure else "http"
        url = f"{scheme}://{self._host}:{self._rest_port}{path}"
        api_key = self._api_key
        call_timeout = self._timeout

        def _fetch() -> Tuple[int, bytes]:
            req = urllib.request.Request(url, method="GET")
            if api_key:
                req.add_header("X-API-Key", api_key)
            try:
                with urllib.request.urlopen(req, timeout=call_timeout) as resp:
                    return resp.getcode(), resp.read()
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read()
                except Exception:
                    body = b""
                return exc.code, body

        try:
            status, body_bytes = await asyncio.to_thread(_fetch)
        except urllib.error.URLError as exc:
            raise ConnectionError(
                message=f"failed to reach REST endpoint {path}",
                details=str(exc.reason),
            ) from exc
        except OSError as exc:
            raise ConnectionError(
                message=f"failed to reach REST endpoint {path}",
                details=str(exc),
            ) from exc

        if status == 401:
            raise AuthenticationError(
                message="authentication failed",
                details=body_bytes.decode("utf-8", errors="replace") or None,
            )
        if status not in (200, 503):
            raise SwarnDBError(
                message=f"HTTP {status} from {path}",
                details=body_bytes.decode("utf-8", errors="replace") or None,
            )

        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SwarnDBError(
                message=f"invalid JSON from {path}",
                details=str(exc),
            ) from exc
        if not isinstance(body, dict):
            body = {}
        return status, body

    # ------------------------------------------------------------------
    # Operational endpoints
    # ------------------------------------------------------------------

    async def recovery_status(self) -> RecoveryStatus:
        """Return the boot recovery snapshot for the server."""
        request = collection_pb2.GetRecoveryStatusRequest()
        response = await self._call(
            self._collection_stub.GetRecoveryStatus, request
        )
        collections = {
            entry.name: _recovery_path_name(entry.path)
            for entry in response.collections
        }
        return RecoveryStatus(
            path=_recovery_path_name(response.path),
            elapsed_secs=int(response.elapsed_secs),
            collections=collections,
        )

    async def readyz(self) -> ReadinessStatus:
        """Return the Kubernetes-style readiness probe result.

        A 503 response is not an error; it maps to ``ready=False`` with the
        probe body. Transport, auth, or other failures raise ``SwarnDBError``.
        """
        status, body = await self._rest_probe("/readyz")
        return ReadinessStatus(
            ready=status == 200,
            status=str(body.get("status", "")),
            checks={k: str(v) for k, v in (body.get("checks") or {}).items()},
        )

    async def healthz(self) -> HealthStatus:
        """Return the Kubernetes-style liveness probe result.

        A 503 response is not an error; it maps to ``healthy=False`` with the
        probe body. Transport, auth, or other failures raise ``SwarnDBError``.
        """
        status, body = await self._rest_probe("/healthz")
        return HealthStatus(
            healthy=status == 200,
            status=str(body.get("status", "")),
            checks={k: str(v) for k, v in (body.get("checks") or {}).items()},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close every async gRPC channel in the pool."""
        for ch in self._channels:
            await ch.close()

    async def __aenter__(self) -> AsyncSwarnDBClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"AsyncSwarnDBClient(host={self._host!r}, port={self._port}, "
            f"secure={self._secure}, channels={self._num_channels})"
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
        quantization: Optional[QuantizationConfig] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        mode: Optional[str] = None,
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
            mode: Optional graph mode ("vector_only", "auto_similarity",
                "hybrid"). When None, the server defaults to vector-only.

        Returns:
            CollectionInfo with the created collection's metadata.

        Raises:
            CollectionExistsError: If a collection with this name already exists.
            ValueError: If mode is not a recognized value.
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
        if mode is not None:
            try:
                request.mode = _MODE_TO_PROTO[mode]
            except KeyError:
                raise ValueError(
                    f"unknown mode: {mode!r}; expected one of "
                    "'vector_only', 'auto_similarity', 'hybrid'"
                ) from None
        await self._client._call(
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
        qt = getattr(response, 'quantization_type', '') or None
        return CollectionInfo(
            name=response.name,
            dimension=response.dimension,
            distance_metric=response.distance_metric,
            vector_count=response.vector_count,
            default_threshold=response.default_threshold,
            quantization_type=qt,
            indexed_count=getattr(response, 'indexed_count', 0),
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
                quantization_type=getattr(c, 'quantization_type', '') or None,
                indexed_count=getattr(c, 'indexed_count', 0),
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

    async def optimize(
        self,
        collection: str,
        rebuild_graph: bool = False,
        *,
        timeout: Optional[float] = None,
    ) -> OptimizeResult:
        """Rebuild deferred indexes, graph, and metadata indexes.

        Call this after bulk inserting with ``defer_graph=True`` or
        ``index_mode="deferred"`` to finalize the collection's indexes.

        Args:
            collection: Collection name to optimize.
            rebuild_graph: If True, also rebuild the virtual graph. Default False.
            timeout: optional per-call deadline in seconds. When None
                (default), a generous deadline is derived so a multi-minute
                index rebuild on a large collection is never cut short. An
                explicit value is honored verbatim. This RPC is server
                serialized and non-idempotent, so a deadline never re-issues it.

        Returns:
            An OptimizeResult with status, message, duration, and
            number of vectors processed.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        request = vector_pb2.OptimizeRequest(
            collection=collection,
            rebuild_graph=rebuild_graph,
        )
        call_timeout = _maintenance_timeout(timeout, self._client._timeout)
        response = await self._client._call(
            self._client._vector_stub.Optimize,
            request,
            timeout=call_timeout,
            retry_deadline=False,
        )
        return OptimizeResult(
            status=response.status,
            message=response.message,
            duration_ms=response.duration_ms,
            vectors_processed=response.vectors_processed,
        )

    async def prune_wal(
        self,
        collection: str,
        *,
        timeout: Optional[float] = None,
    ) -> PruneWALResult:
        """Prune old WAL files for a collection.

        Removes write-ahead log files that are no longer needed after
        data has been flushed to segments.

        Args:
            collection: Collection name.
            timeout: optional per-call deadline in seconds. When None
                (default), a generous deadline is derived so a long prune on a
                large collection is never cut short. An explicit value is
                honored verbatim. This RPC is server serialized and
                non-idempotent, so a deadline never re-issues it.
        """
        request = vector_pb2.PruneWALRequest(collection=collection)
        call_timeout = _maintenance_timeout(timeout, self._client._timeout)
        response = await self._client._call(
            self._client._vector_stub.PruneWAL,
            request,
            timeout=call_timeout,
            retry_deadline=False,
        )
        return PruneWALResult(
            status=response.status,
            files_deleted=response.files_deleted,
            bytes_freed=response.bytes_freed,
            duration_ms=response.duration_ms,
        )

    async def compact(
        self,
        collection: str,
        min_segments: int = 0,
        remove_deleted: bool = True,
        *,
        timeout: Optional[float] = None,
    ) -> CompactResult:
        """Compact collection segments into fewer, larger files.

        Args:
            collection: Collection name.
            min_segments: Minimum segment count to trigger compaction. 0 = use server default (4).
            remove_deleted: Whether to remove deleted vectors during compaction. Default True.
            timeout: optional per-call deadline in seconds. When None
                (default), a generous deadline is derived so a multi-minute
                compaction on a large collection is never cut short. An
                explicit value is honored verbatim. This RPC is server
                serialized and non-idempotent, so a deadline never re-issues it.
        """
        request = vector_pb2.CompactRequest(
            collection=collection,
            min_segments=min_segments,
            remove_deleted=remove_deleted,
        )
        call_timeout = _maintenance_timeout(timeout, self._client._timeout)
        response = await self._client._call(
            self._client._vector_stub.Compact,
            request,
            timeout=call_timeout,
            retry_deadline=False,
        )
        return CompactResult(
            status=response.status,
            segments_merged=response.segments_merged,
            vectors_written=response.vectors_written,
            vectors_removed=response.vectors_removed,
            duration_ms=response.duration_ms,
        )

    async def snapshot(
        self,
        name: str,
        *,
        timeout: Optional[float] = None,
    ) -> int:
        """Force a synchronous snapshot for a collection.

        Args:
            name: Collection name.
            timeout: optional per-call deadline in seconds. When None
                (default), a generous deadline is derived so a long snapshot on
                a large collection is never cut short. An explicit value is
                honored verbatim. This RPC is server serialized and
                non-idempotent, so a deadline never re-issues it.

        Returns:
            The LSN of the snapshot just written.
        """
        request = collection_pb2.SnapshotCollectionRequest(name=name)
        call_timeout = _maintenance_timeout(timeout, self._client._timeout)
        response = await self._client._call(
            self._client._collection_stub.SnapshotCollection,
            request,
            timeout=call_timeout,
            retry_deadline=False,
        )
        return int(response.last_snapshot_lsn)

    async def persistence_status(self, name: str) -> PersistenceStatus:
        """Return the snapshot and WAL LSN state for a collection."""
        request = collection_pb2.GetPersistenceStatusRequest(name=name)
        response = await self._client._call(
            self._client._collection_stub.GetPersistenceStatus, request
        )
        return PersistenceStatus(
            last_snapshot_lsn=int(response.last_snapshot_lsn),
            current_lsn=int(response.current_lsn),
            next_lsn=int(response.next_lsn),
        )

    async def metrics(self, name: str) -> CollectionMetrics:
        """Return per-collection lock-contention counters."""
        request = collection_pb2.GetCollectionMetricsRequest(name=name)
        response = await self._client._call(
            self._client._collection_stub.GetCollectionMetrics, request
        )
        return CollectionMetrics(
            map_lock_acquisitions=int(response.map_lock_acquisitions),
            collection_read_acquisitions=int(response.collection_read_acquisitions),
            collection_write_acquisitions=int(response.collection_write_acquisitions),
            total_blocked_microseconds=int(response.total_blocked_microseconds),
        )

    async def get_status(self, collection: str) -> str:
        """Get collection optimization status.

        Args:
            collection: Collection name.

        Returns:
            One of: ``"ready"``, ``"pending_optimization"``, or
            ``"optimizing"``.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
        """
        request = collection_pb2.GetCollectionRequest(name=collection)
        response = await self._client._call(
            self._client._collection_stub.GetCollection, request
        )
        return response.status or "ready"


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

    async def get(self, collection: str, id: int) -> Optional[VectorRecord]:
        """Get a vector by ID.

        Args:
            collection: Collection name.
            id: Vector ID to retrieve.

        Returns:
            A VectorRecord with the vector data and metadata, or
            ``None`` if no vector with the given id exists.

        Raises:
            SwarnDBError: On transport, auth, quota, or other non
                NotFound failures.
        """
        request = vector_pb2.GetVectorRequest(
            collection=collection,
            id=id,
        )
        try:
            response = await self._client._call(
                self._client._vector_stub.Get, request
            )
        except VectorNotFoundError:
            return None
        return VectorRecord(
            id=response.id,
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
        batch_lock_size: Optional[int] = None,
        defer_graph: bool = False,
        wal_flush_every: Optional[int] = None,
        ef_construction: Optional[int] = None,
        index_mode: Optional[str] = None,
        skip_metadata_index: bool = False,
        parallel_build: bool = False,
        checkpoint_every: Optional[int] = None,
        resume_token: Optional[str] = None,
        timeout: Optional[float] = None,
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
            batch_lock_size: Vectors per lock acquisition (server default=1,
                max=10000). Higher values reduce lock overhead.
            defer_graph: If True, skip graph computation during insert.
                Use ``client.collections.optimize()`` afterward.
            wal_flush_every: Preserved for backward compatibility. Only the
                value ``0`` has effect: it opts out of storage-layer WAL
                writes during this bulk insert (max throughput, no crash
                recovery for these entries). Any positive integer is
                accepted but ignored, and behaves identically to ``None``;
                per-entry WAL writes happen unconditionally in that case.
                The legacy "flush every N operations" batching behavior
                no longer exists.
            ef_construction: Override HNSW ef_construction for this batch.
            index_mode: Index build mode: ``"immediate"`` (default) or
                ``"deferred"`` (build index after all inserts).
            skip_metadata_index: If True, skip metadata indexing during
                insert for faster ingestion.
            parallel_build: If True, use parallel HNSW construction
                (only effective with ``index_mode="deferred"``).
            checkpoint_every: optional, if non-zero the server writes a
                checkpoint file every N batches; use the returned
                resume_token to resume a failed bulk insert. 0 (default)
                disables checkpoints.
            resume_token: optional, opaque token from a prior partial bulk
                insert response; the server validates it against its
                on-disk checkpoint and skips already-committed batches.
            timeout: optional per-call deadline in seconds for this bulk
                insert RPC. When None (default), the client default timeout
                is used, preserving prior behavior. Pass a larger value for
                large immediate-index loads that would otherwise exceed the
                default deadline; for very large loads prefer the
                server-side ``bulk_insert_from_path`` path instead.

        Returns:
            A BulkInsertResult with inserted_count and any errors.

        Raises:
            ValueError: If metadata_list or ids length doesn't match vectors,
                or if index_mode is not a valid value.
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
        if index_mode is not None and index_mode not in ("immediate", "deferred"):
            raise ValueError(
                f"index_mode must be 'immediate' or 'deferred', "
                f"got {index_mode!r}"
            )
        if batch_lock_size is not None and (batch_lock_size < 1 or batch_lock_size > 10000):
            raise ValueError(
                f"batch_lock_size must be between 1 and 10000, "
                f"got {batch_lock_size}"
            )
        if parallel_build and index_mode not in (None, "deferred"):
            raise ValueError(
                "parallel_build=True requires index_mode='deferred'"
            )

        # Build options to determine which RPC to use
        options = BulkInsertOptions(
            batch_lock_size=batch_lock_size,
            defer_graph=defer_graph,
            wal_flush_every=wal_flush_every,
            ef_construction=ef_construction,
            index_mode=index_mode,
            skip_metadata_index=skip_metadata_index,
            parallel_build=parallel_build,
            checkpoint_every=checkpoint_every,
            resume_token=resume_token,
        )
        use_optimized_rpc = options.has_non_defaults()

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

        async def _options_request_iterator() -> AsyncIterator:
            """Yield options message first, then vector InsertRequests.

            Uses the BulkInsertWithOptions streaming RPC where the first
            message contains BulkInsertOptions and subsequent messages
            contain the vectors.
            """
            options_msg = vector_pb2.BulkInsertStreamMessage(
                options=vector_pb2.BulkInsertOptions(
                    batch_lock_size=options.batch_lock_size or 0,
                    defer_graph=options.defer_graph,
                    wal_flush_every=options.wal_flush_every or 0,
                    ef_construction=options.ef_construction or 0,
                    index_mode=options.index_mode or "",
                    skip_metadata_index=options.skip_metadata_index,
                    parallel_build=options.parallel_build,
                    checkpoint_every=options.checkpoint_every or 0,
                    resume_token=options.resume_token or "",
                )
            )
            yield options_msg

            async for req in _request_iterator():
                yield vector_pb2.BulkInsertStreamMessage(vector=req)

        # BulkInsert is stream_unary: stream requests, get one response.
        # Cannot use _call (unary-unary), so call stub directly with retry.
        metadata = self._client._metadata()
        # Auto-scale the deadline with the workload when the caller did not pass
        # an explicit timeout; an explicit timeout is honored verbatim. Pass the
        # vector dimension so high-dim loads (e.g. 1536) get a deadline that
        # reflects their heavier per-vector cost instead of timing out early.
        insert_dim = len(vectors[0]) if total > 0 else 0
        call_timeout = _scaled_bulk_timeout(
            timeout, self._client._timeout, num_vectors=total, dim=insert_dim
        )

        # Streaming bulk_insert is NOT safely retryable once a single message
        # has left the client: the server is non-idempotent on explicit ids, so
        # re-streaming the same ids after a partial commit collides with the
        # committed prefix and yields a misleading inserted_count plus a wall of
        # "already exists" errors. A DEADLINE_EXCEEDED only ever fires after
        # streaming has begun, so it is never re-streamed. We retry ONLY genuine
        # pre-stream transient failures (connection setup), where nothing could
        # have been committed; the marker flips true at the first yielded
        # message.
        stream_started = {"v": False}

        async def _marking_iter(inner: AsyncIterator) -> AsyncIterator:
            async for item in inner:
                stream_started["v"] = True
                yield item

        last_error: Optional[grpc.RpcError] = None

        for attempt in range(self._client._max_retries + 1):
            stream_started["v"] = False
            try:
                if use_optimized_rpc:
                    logger.info(
                        "Async BulkInsert with options: batch_lock_size=%s, "
                        "defer_graph=%s, wal_flush_every=%s, "
                        "ef_construction=%s, index_mode=%s, "
                        "skip_metadata_index=%s, parallel_build=%s, "
                        "checkpoint_every=%s, resume_token=%s",
                        options.batch_lock_size,
                        options.defer_graph,
                        options.wal_flush_every,
                        options.ef_construction,
                        options.index_mode,
                        options.skip_metadata_index,
                        options.parallel_build,
                        options.checkpoint_every,
                        options.resume_token,
                    )
                    response = await self._client._vector_stub.BulkInsertWithOptions(
                        _marking_iter(_options_request_iterator()),
                        timeout=call_timeout,
                        metadata=metadata,
                    )
                else:
                    response = await self._client._vector_stub.BulkInsert(
                        _marking_iter(_request_iterator()),
                        timeout=call_timeout,
                        metadata=metadata,
                    )
                return BulkInsertResult(
                    inserted_count=response.inserted_count,
                    errors=list(response.errors),
                    last_completed_batch_idx=getattr(
                        response, "last_completed_batch_idx", 0
                    ),
                    last_committed_lsn=getattr(
                        response, "last_committed_lsn", 0
                    ),
                    resume_token=getattr(response, "resume_token", ""),
                    assigned_ids=list(getattr(response, "assigned_ids", []) or []),
                )
            except grpc.RpcError as exc:
                last_error = exc
                code = exc.code()

                # A deadline that still trips after auto-scaling means the load
                # genuinely outran the deadline. Do not re-stream: surface one
                # clear, actionable error instead of a misleading partial count.
                if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise SwarnDBError(
                        f"bulk_insert exceeded the {call_timeout:.0f}s deadline "
                        f"for {total} vectors. The streaming insert cannot be "
                        "auto-retried because re-sending the same ids would "
                        "collide with the partially committed prefix. Retry "
                        "with a larger timeout= (or pass checkpoint_every= and "
                        "resume with the returned resume_token), or use "
                        "bulk_insert_from_path for very large loads."
                    ) from exc

                # Only a pre-stream transient (e.g. UNAVAILABLE during connection
                # setup, before any message was sent) is safe to retry.
                if (
                    code == grpc.StatusCode.UNAVAILABLE
                    and not stream_started["v"]
                    and attempt < self._client._max_retries
                ):
                    delay = self._client._retry_delay * (2 ** attempt)
                    logger.debug(
                        "Retrying async BulkInsert (attempt %d/%d) after "
                        "pre-stream %s, backoff %.2fs",
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

    async def bulk_insert_from_path(
        self,
        collection: str,
        path: str,
        *,
        dim: int = 0,
        expected_count: int = 0,
        total_count_hint: int = 0,
        id_start: int = 1,
        ids_path: str = "",
        skip_metadata_index: bool = False,
        index_mode: str = "immediate",
        ef_construction: int = 0,
        chunk_size: int = 0,
        timeout: Optional[float] = None,
    ) -> BulkInsertResult:
        """Bulk insert vectors from a server-side file via mmap.

        The server memory-maps the file at ``path`` and ingests vectors
        directly without streaming them over gRPC. Use this for very
        large ingests where avoiding a client-side allocation matters.

        Args:
            collection: Target collection name.
            path: Absolute path to the vector file on the server's
                local filesystem.
            dim: Vector dimensionality. Pass 0 to defer to the
                collection's configured dimension.
            expected_count: Expected number of vectors in the file. Use
                0 to let the server infer from file size.
            total_count_hint: Optional hint for total vectors across
                multiple bulk inserts; used for capacity planning.
            id_start: Starting ID for auto-assigned IDs (default 1).
            ids_path: Optional path to a sidecar file containing explicit
                IDs (one per vector). Empty string means use auto-assigned
                IDs starting at ``id_start``.
            skip_metadata_index: If True, skip metadata indexing during
                insert for faster ingestion.
            index_mode: Index build mode: ``"immediate"`` (default) or
                ``"deferred"`` (build index after all inserts).
            ef_construction: Optional HNSW ef_construction override for
                this batch (0 = use collection default).
            chunk_size: Server-side chunked compact-insert mode. ``0``
                (default) preserves the existing single-pass behavior.
                When ``> 0``, the server processes the load in chunks of
                that many rows and snapshots, prunes the WAL, and
                releases scratch memory between chunks. This trades
                insert wall-clock for a lower peak RSS and is intended
                for memory-tight boxes willing to accept a slower
                one-time load. Resume mid-call is not supported in this
                release; callers must restart the full call on failure.
            timeout: optional per-call deadline in seconds. When None
                (default), the deadline is derived from a client-known
                signal and never collapses to the bare client default: it
                scales from ``expected_count`` when that is given, refines by
                the on-disk file size if ``path`` happens to be locally
                visible (a shared filesystem), and otherwise falls back to a
                generous from-path floor so a large server-side load is not
                cut short. An explicit value is honored verbatim. The call
                does not retry on DEADLINE_EXCEEDED, because the server-side
                mmap ingest is non-idempotent and a retry would re-ingest.

        Returns:
            A BulkInsertResult with inserted_count, assigned_ids, and
            any errors.

        Raises:
            ValueError: If index_mode is not a valid value.
            SwarnDBError: On transport, auth, quota, or other failures.
        """
        if index_mode not in ("immediate", "deferred"):
            raise ValueError(
                f"index_mode must be 'immediate' or 'deferred', "
                f"got {index_mode!r}"
            )

        # Derive the deadline from whatever signal the client actually has,
        # never the bare default. An explicit timeout always wins. With
        # expected_count we scale from the vector count (same model as the
        # streaming path). Otherwise, if the path is locally visible (client
        # and server share a filesystem), refine by file size. With no signal
        # at all (the normal server-side path) we use a generous from-path
        # floor rather than guessing, since this API is the very-large-load
        # path and must over-provision.
        if timeout is not None:
            call_timeout = timeout
        elif expected_count > 0:
            # from-path is the very-large-load path; the server-side index
            # build dominates and is far slower than a file read, so never let
            # a count/byte estimate drop the deadline below the generous floor.
            # The estimate may only raise it (for multi-million loads).
            call_timeout = max(
                _BULK_FROM_PATH_MIN_DEADLINE,
                _scaled_bulk_timeout(
                    None,
                    self._client._timeout,
                    num_vectors=expected_count,
                    dim=dim if dim > 0 else None,
                ),
            )
        else:
            file_bytes: Optional[int] = None
            try:
                file_bytes = os.path.getsize(path)
            except OSError:
                file_bytes = None
            if file_bytes is not None:
                call_timeout = max(
                    _BULK_FROM_PATH_MIN_DEADLINE,
                    _scaled_bulk_timeout(
                        None, self._client._timeout, num_bytes=file_bytes
                    ),
                )
            else:
                # No client-side signal: over-provision via the floor.
                call_timeout = _BULK_FROM_PATH_MIN_DEADLINE

        request = vector_pb2.BulkInsertFromPathRequest(
            collection=collection,
            path=path,
            dim=dim,
            expected_count=expected_count,
            total_count_hint=total_count_hint,
            id_start=id_start,
            ids_path=ids_path,
            skip_metadata_index=skip_metadata_index,
            index_mode=index_mode,
            ef_construction=ef_construction,
            chunk_size=chunk_size,
        )

        # Do not retry DEADLINE_EXCEEDED: the server-side mmap ingest is
        # non-idempotent, so a retry would re-ingest the file and corrupt the
        # client-visible count. UNAVAILABLE (connection setup) may still retry.
        response = await self._client._call(
            self._client._vector_stub.BulkInsertFromPath,
            request,
            timeout=call_timeout,
            retry_deadline=False,
        )
        # Avoid rebuilding the full 1M id list the from-path API exists to skip:
        # keep a plain list for small results, lazily view the proto field for
        # large ones. inserted_count carries the count without touching ids.
        return BulkInsertResult(
            inserted_count=response.inserted_count,
            errors=list(response.errors),
            last_completed_batch_idx=getattr(
                response, "last_completed_batch_idx", 0
            ),
            last_committed_lsn=getattr(
                response, "last_committed_lsn", 0
            ),
            resume_token=getattr(response, "resume_token", ""),
            assigned_ids=_assigned_ids_view(response),
        )


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
        ef_search: Optional[int] = None,
        quantization: Optional[SearchQuantizationParams] = None,
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
            ef_search: Optional HNSW ef_search override for this query.
            quantization: Optional per-query quantization parameters.

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
        if ef_search is not None:
            request.ef_search = ef_search
        if filter is not None:
            request.filter.CopyFrom(filter._to_proto())
        if quantization is not None:
            request.quantization.rescore = quantization.rescore
            request.quantization.oversampling = quantization.oversampling
            request.quantization.ignore = quantization.ignore

        response = await self._client._call(
            self._client._search_stub.Search,
            request,
        )

        results = [_scored_result_from_proto(r) for r in response.results]
        return SearchResult(
            results=results,
            search_time_us=response.search_time_us,
            warning=getattr(response, 'warning', '') or '',
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
        ef_search: Optional[int] = None,
        quantization: Optional[SearchQuantizationParams] = None,
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
            ef_search: Optional HNSW ef_search override applied to all queries.
            quantization: Optional per-query quantization parameters applied to all queries.

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
            if ef_search is not None:
                req.ef_search = ef_search
            if proto_filter is not None:
                req.filter.CopyFrom(proto_filter)
            if quantization is not None:
                req.quantization.rescore = quantization.rescore
                req.quantization.oversampling = quantization.oversampling
                req.quantization.ignore = quantization.ignore
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
                SearchResult(
                    results=results,
                    search_time_us=sr.search_time_us,
                    warning=getattr(sr, 'warning', '') or '',
                )
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

    # ── Typed graph (Hybrid mode) ──

    async def put_node(
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
        """Create a typed node (Hybrid collections only). Returns the node id."""
        request = graph_pb2.PutNodeRequest(
            collection=collection,
            kind=_kind_to_proto(kind),
            label=label,
            properties_json=json.dumps(properties or {}),
            embedding=list(embedding or []),
            source=source,
            created_by=created_by,
        )
        response = await self._client._call(self._client._graph_stub.PutNode, request)
        return int(response.id)

    async def get_node(self, collection: str, node_id: int) -> Optional[TypedNode]:
        """Fetch a typed node by id, or None if absent."""
        request = graph_pb2.GetNodeRequest(collection=collection, id=node_id)
        response = await self._client._call(self._client._graph_stub.GetNode, request)
        return _node_from_proto(response.node) if response.found else None

    async def delete_node(self, collection: str, node_id: int) -> bool:
        """Delete a typed node and its incident edges. True if it existed."""
        request = graph_pb2.DeleteNodeRequest(collection=collection, id=node_id)
        response = await self._client._call(
            self._client._graph_stub.DeleteNode, request
        )
        return bool(response.deleted)

    async def update_node(
        self,
        collection: str,
        node_id: int,
        *,
        properties: Optional[Dict[str, Any]] = None,
        actor: str = "",
    ) -> TypedNode:
        """Update a typed node's properties, recording an audit entry.

        Only the property bag is mutable; provenance and the embedding are
        immutable. Omitting ``properties`` leaves them unchanged.
        """
        request = graph_pb2.UpdateNodeRequest(
            collection=collection,
            node_id=node_id,
            actor=actor,
        )
        if properties is not None:
            request.properties_json = json.dumps(properties)
        response = await self._client._call(
            self._client._graph_stub.UpdateNode, request
        )
        return _node_from_proto(response.node)

    async def put_edge(
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
        response = await self._client._call(self._client._graph_stub.PutEdge, request)
        return int(response.id)

    async def get_edge(self, collection: str, edge_id: int) -> Optional[TypedEdge]:
        """Fetch a typed edge by id, or None if absent."""
        request = graph_pb2.GetEdgeRequest(collection=collection, id=edge_id)
        response = await self._client._call(self._client._graph_stub.GetEdge, request)
        return _edge_from_proto(response.edge) if response.found else None

    async def delete_edge(self, collection: str, edge_id: int) -> bool:
        """Delete a typed edge. True if it existed."""
        request = graph_pb2.DeleteEdgeRequest(collection=collection, id=edge_id)
        response = await self._client._call(
            self._client._graph_stub.DeleteEdge, request
        )
        return bool(response.deleted)

    async def list_edges(
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
        response = await self._client._call(self._client._graph_stub.ListEdges, request)
        return [_edge_from_proto(e) for e in response.edges]

    async def update_edge(
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
        response = await self._client._call(
            self._client._graph_stub.UpdateEdge, request
        )
        return _edge_from_proto(response.edge)

    async def verify_edge(
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
        response = await self._client._call(
            self._client._graph_stub.VerifyEdge, request
        )
        return _edge_from_proto(response.edge)

    async def reject_edge(
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
        response = await self._client._call(
            self._client._graph_stub.RejectEdge, request
        )
        return EdgeRejectResult(
            deleted=bool(response.deleted),
            rule_added=bool(response.rule_added),
        )

    async def bulk_import_edges(
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
        response = await self._client._call(
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

    async def enumerate_nodes(
        self,
        collection: str,
        *,
        after_id: int = 0,
        limit: int = 1000,
        kind: Optional[str] = None,
        label: str = "",
    ) -> NodePage:
        """Fetch one page of nodes in ascending id order (Hybrid mode)."""
        request = graph_pb2.EnumerateNodesRequest(
            collection=collection,
            after_id=after_id,
            limit=limit,
            label=label,
        )
        if kind is not None:
            request.filter_by_kind = True
            request.kind = _kind_to_proto(kind)
        response = await self._client._call(
            self._client._graph_stub.EnumerateNodes, request
        )
        return NodePage(
            nodes=[_node_from_proto(n) for n in response.nodes],
            next_cursor=int(response.next_cursor),
            has_more=bool(response.has_more),
        )

    async def enumerate_edges(
        self,
        collection: str,
        *,
        after_id: int = 0,
        limit: int = 1000,
        edge_type: str = "",
    ) -> EdgePage:
        """Fetch one page of edges in ascending id order (Hybrid mode)."""
        request = graph_pb2.EnumerateEdgesRequest(
            collection=collection,
            after_id=after_id,
            limit=limit,
            edge_type=edge_type,
        )
        response = await self._client._call(
            self._client._graph_stub.EnumerateEdges, request
        )
        return EdgePage(
            edges=[_edge_from_proto(e) for e in response.edges],
            next_cursor=int(response.next_cursor),
            has_more=bool(response.has_more),
        )

    async def iter_nodes(
        self,
        collection: str,
        *,
        page_size: int = 1000,
        kind: Optional[str] = None,
        label: str = "",
    ) -> AsyncIterator[TypedNode]:
        """Iterate every node in the graph, walking pages to exhaustion."""
        after = 0
        while True:
            page = await self.enumerate_nodes(
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

    async def iter_edges(
        self,
        collection: str,
        *,
        page_size: int = 1000,
        edge_type: str = "",
    ) -> AsyncIterator[TypedEdge]:
        """Iterate every edge in the graph, walking pages to exhaustion."""
        after = 0
        while True:
            page = await self.enumerate_edges(
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

    def query(self, collection: str) -> "AsyncHybridQueryBuilder":
        """Start a composable hybrid query against a collection.

        Returns a chainable builder; finish by awaiting a terminal
        ``return_nodes()`` / ``return_edges()`` / ``return_paths()``.
        """
        from swarndb.hybrid import AsyncHybridQueryBuilder

        return AsyncHybridQueryBuilder(self._client, collection)

    async def graph_rag(
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

        Async twin of ``GraphAPI.graph_rag``. It composes the proven GraphRAG
        plan for you: a vector seed expanded across the graph so the candidate
        pool includes graph-reached passages, then ranked within that pool. The
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
        structural form for content-to-content graphs::

            seed = vector_similar(query_vector, k)
            bridge = vector_similar(query_vector, k).k_hop(any, max=k_hop_max)
            seed.union(bridge).<ranking step from fusion>.return_nodes()

        The default ``fusion="vector_rank"`` is graph-first scope-then-rank: the
        graph fixes the candidate pool, then the pool is ranked EXACTLY by
        similarity to ``query_vector`` (recall 1.0, no ANN). Pass
        ``fusion="rrf"`` to opt into Reciprocal Rank Fusion instead, which fuses
        the vector and graph-proximity rankings and then applies ``rrf_k`` and
        ``hub_damping``.

        The graph expansion recovers gold the vector arm misses on multi-hop or
        vector-dissimilar bridges. It is OPTIONAL value that ADDS LATENCY; plain
        vector search stays the zero-cost default.

        Example::

            result = await client.graph.graph_rag(
                "docs_graph", query_vector, k=10,
                relation_edge_types=["CITES"],
            )
            for node in result.nodes:
                print(node.id, node.label)

        Args mirror ``GraphAPI.graph_rag`` (including ``fusion``: "vector_rank"
        default, "rrf" to opt in). Returns a HybridQueryResult whose ``nodes``
        hold the ranked top-k.
        """
        from swarndb.hybrid import AsyncHybridQueryBuilder, compose_graph_rag

        seed = AsyncHybridQueryBuilder(self._client, collection)
        bridge = AsyncHybridQueryBuilder(self._client, collection)
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
        return await composed.return_nodes()


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
