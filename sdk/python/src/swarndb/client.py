"""SwarnDB synchronous client.

This module provides the main synchronous gRPC client for interacting with
a SwarnDB server. It wraps all collection, vector, search, graph, and math
operations behind a single connection-managed interface.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import grpc

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
from swarndb._proto import collection_pb2
from swarndb._proto import collection_pb2_grpc
from swarndb._proto import vector_pb2_grpc
from swarndb._proto import search_pb2_grpc
from swarndb._proto import graph_pb2_grpc
from swarndb._proto import extraction_pb2_grpc
from swarndb._proto import vector_math_pb2_grpc
from swarndb.types import HealthStatus, ReadinessStatus, RecoveryStatus

if TYPE_CHECKING:
    from swarndb.collections import CollectionAPI
    from swarndb.vectors import VectorAPI
    from swarndb.search import SearchAPI
    from swarndb.graph import GraphAPI
    from swarndb.extraction import ExtractionAPI
    from swarndb.math_ops import MathAPI

logger = logging.getLogger(__name__)

# 128 MB ceiling: headroom above server default (64 MB) so operators can raise SWARNDB_MAX_REQUEST_BODY_BYTES.
_DEFAULT_MAX_MESSAGE_BYTES = 128 * 1024 * 1024

# Default REST port; mirrors the server's default_rest_port in config.rs.
_DEFAULT_REST_PORT = 8080

# gRPC status codes that are considered transient and eligible for retry.
_RETRYABLE_CODES = frozenset({
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
})

# Map the proto RecoveryPath enum integer to the wire string used by the REST
# /recovery_status endpoint. Keeps the SDK return value identical regardless of
# which transport carried the response.
_RECOVERY_PATH_NAMES = {
    0: "Unknown",
    1: "CleanShutdown",
    2: "IncrementalReplay",
    3: "FullRebuild",
}


def _recovery_path_name(value: int) -> str:
    return _RECOVERY_PATH_NAMES.get(int(value), "Unknown")


class _AuthInterceptor(grpc.UnaryUnaryClientInterceptor):
    """Interceptor that injects an API key into every unary-unary call."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def intercept_unary_unary(
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

        new_details = _ClientCallDetails(
            method=client_call_details.method,
            timeout=client_call_details.timeout,
            metadata=metadata,
            credentials=client_call_details.credentials,
            wait_for_ready=client_call_details.wait_for_ready,
            compression=client_call_details.compression,
        )
        return continuation(new_details, request)


class _ClientCallDetails(
    grpc.ClientCallDetails,
):
    """Concrete implementation of ClientCallDetails for interceptor use."""

    def __init__(
        self,
        method: str,
        timeout: Optional[float],
        metadata: Optional[Sequence[Tuple[str, str]]],
        credentials: Optional[grpc.CallCredentials],
        wait_for_ready: Optional[bool],
        compression: Optional[grpc.Compression],
    ) -> None:
        self.method = method
        self.timeout = timeout
        self.metadata = metadata
        self.credentials = credentials
        self.wait_for_ready = wait_for_ready
        self.compression = compression


class SwarnDBClient:
    """Synchronous SwarnDB client with gRPC connection management.

    Provides access to all SwarnDB operations through lazy-initialized
    API namespaces: ``collections``, ``vectors``, ``search``, ``graph``,
    and ``math``.

    Usage::

        with SwarnDBClient("localhost", 50051) as client:
            client.collections.create("my_collection", dimension=128)
            client.vectors.insert("my_collection", "v1", [0.1] * 128)
            results = client.search.query("my_collection", [0.1] * 128, k=5)

    Concurrency:
        Every call here is a blocking call: the calling thread waits for the
        server reply. A single gRPC channel rides on a small number of HTTP/2
        connections, each with a cap on how many requests can be in flight at
        once; once that cap is reached, further requests queue at the channel
        instead of going out concurrently. So a multi-thread fan-out (for
        example a ``ThreadPoolExecutor`` running many ``search.query`` calls)
        sharing one default client stops scaling well past a couple of
        in-flight requests.

        To genuinely run many requests at once from a thread pool, give the
        client room with ``channels=N`` (it spreads each request across ``N``
        independent channels, round robin), or give each worker its own client
        via :meth:`clone`. As a rule of thumb, set ``channels`` to about the
        number of worker threads. For an asyncio program, use
        :class:`AsyncSwarnDBClient` instead, which is concurrent by design.

    Args:
        channels: Number of independent gRPC channels to spread requests
            across, round robin. ``1`` (default) keeps the prior single-channel
            behavior exactly. Raise it for multi-thread fan-out so concurrent
            requests do not serialize behind one channel's stream limit.
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

        # Build the channel pool. One channel keeps the prior behavior; more
        # channels let concurrent blocking calls fan out instead of queueing
        # behind a single channel's HTTP/2 stream limit.
        #
        # With >1 channel, each gets its own subchannel pool so identical
        # target+options do not coalesce onto one shared HTTP/2 connection in
        # gRPC's global pool. Single-channel callers keep byte-identical options.
        per_channel_options = channel_options
        if channels > 1:
            per_channel_options = channel_options + [
                ("grpc.use_local_subchannel_pool", 1),
            ]
        self._channels: List[grpc.Channel] = []
        for _ in range(channels):
            if secure:
                credentials = grpc.ssl_channel_credentials()
                ch = grpc.secure_channel(target, credentials, options=per_channel_options)
            else:
                ch = grpc.insecure_channel(target, options=per_channel_options)
            if api_key:
                ch = grpc.intercept_channel(ch, _AuthInterceptor(api_key))
            self._channels.append(ch)

        # First channel kept for single-channel callers and lifecycle code.
        self._channel = self._channels[0]

        # One stub set per channel, plus a round-robin cursor so each stub
        # access lands on the next channel. Guarded by a lock because a thread
        # pool reads these cursors concurrently.
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
        self._rr_lock = threading.Lock()
        self._rr_counter = itertools.count()

        # Lazy-initialized API facades
        self._collections: Optional[CollectionAPI] = None
        self._vectors: Optional[VectorAPI] = None
        self._search: Optional[SearchAPI] = None
        self._graph: Optional[GraphAPI] = None
        self._extraction: Optional[ExtractionAPI] = None
        self._math: Optional[MathAPI] = None

    # ------------------------------------------------------------------
    # Round-robin stub access (channel pool)
    # ------------------------------------------------------------------
    #
    # The whole SDK reads these as ``client._search_stub`` etc. When the pool
    # holds one channel they return that single stub (zero overhead, prior
    # behavior). With more channels each read advances a shared cursor so
    # successive calls ride different channels and run concurrently.

    def _next_index(self) -> int:
        if self._num_channels == 1:
            return 0
        with self._rr_lock:
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

    def clone(self) -> "SwarnDBClient":
        """Return a new client with its own channel(s) and the same settings.

        Use this to give each worker thread its own client when you prefer a
        client-per-worker layout over ``channels=N``. The returned client is
        fully independent: closing one does not close the other.
        """
        return SwarnDBClient(
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
    def collections(self) -> CollectionAPI:
        """Access collection management operations."""
        if self._collections is None:
            from swarndb.collections import CollectionAPI
            self._collections = CollectionAPI(self)
        return self._collections

    @property
    def vectors(self) -> VectorAPI:
        """Access vector CRUD operations."""
        if self._vectors is None:
            from swarndb.vectors import VectorAPI
            self._vectors = VectorAPI(self)
        return self._vectors

    @property
    def search(self) -> SearchAPI:
        """Access search operations."""
        if self._search is None:
            from swarndb.search import SearchAPI
            self._search = SearchAPI(self)
        return self._search

    @property
    def graph(self) -> GraphAPI:
        """Access graph operations."""
        if self._graph is None:
            from swarndb.graph import GraphAPI
            self._graph = GraphAPI(self)
        return self._graph

    @property
    def extraction(self) -> ExtractionAPI:
        """Access LLM extraction operations (Hybrid mode)."""
        if self._extraction is None:
            from swarndb.extraction import ExtractionAPI
            self._extraction = ExtractionAPI(self)
        return self._extraction

    @property
    def math(self) -> MathAPI:
        """Access vector math operations."""
        if self._math is None:
            from swarndb.math_ops import MathAPI
            self._math = MathAPI(self)
        return self._math

    # ------------------------------------------------------------------
    # Internal call helpers
    # ------------------------------------------------------------------

    def _call(
        self,
        stub_method,
        request,
        *,
        timeout: Optional[float] = None,
        retry_deadline: bool = True,
    ):
        """Execute a gRPC call with retry logic and error handling.

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
                return stub_method(
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
                        "Retrying gRPC call (attempt %d/%d) after %s error, "
                        "backoff %.2fs",
                        attempt + 1,
                        self._max_retries,
                        code.name,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                # Non-retryable or exhausted retries, convert to SDK error
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

    def _rest_url(self, path: str) -> str:
        scheme = "https" if self._secure else "http"
        return f"{scheme}://{self._host}:{self._rest_port}{path}"

    def _rest_probe(self, path: str) -> Tuple[int, Dict[str, Any]]:
        """GET an HTTP probe endpoint that may legitimately return 503.

        Returns the status code and parsed JSON body. Raises SwarnDBError on
        transport / auth / quota failures and on unexpected HTTP statuses.
        """
        url = self._rest_url(path)
        req = urllib.request.Request(url, method="GET")
        if self._api_key:
            req.add_header("X-API-Key", self._api_key)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status = resp.getcode()
                body_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                body_bytes = exc.read()
            except Exception:
                body_bytes = b""
            if status == 401:
                raise AuthenticationError(
                    message="authentication failed",
                    details=body_bytes.decode("utf-8", errors="replace") or None,
                ) from exc
            if status != 503:
                raise SwarnDBError(
                    message=f"HTTP {status} from {path}",
                    details=body_bytes.decode("utf-8", errors="replace") or None,
                ) from exc
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

    def recovery_status(self) -> RecoveryStatus:
        """Return the boot recovery snapshot for the server."""
        request = collection_pb2.GetRecoveryStatusRequest()
        response = self._call(
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

    def readyz(self) -> ReadinessStatus:
        """Return the Kubernetes-style readiness probe result.

        A 503 response is not an error; it maps to ``ready=False`` with the
        probe body. Transport, auth, or other failures raise ``SwarnDBError``.
        """
        status, body = self._rest_probe("/readyz")
        return ReadinessStatus(
            ready=status == 200,
            status=str(body.get("status", "")),
            checks={k: str(v) for k, v in (body.get("checks") or {}).items()},
        )

    def healthz(self) -> HealthStatus:
        """Return the Kubernetes-style liveness probe result.

        A 503 response is not an error; it maps to ``healthy=False`` with the
        probe body. Transport, auth, or other failures raise ``SwarnDBError``.
        """
        status, body = self._rest_probe("/healthz")
        return HealthStatus(
            healthy=status == 200,
            status=str(body.get("status", "")),
            checks={k: str(v) for k, v in (body.get("checks") or {}).items()},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close every gRPC channel in the pool."""
        for ch in self._channels:
            ch.close()

    def __enter__(self) -> SwarnDBClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SwarnDBClient(host={self._host!r}, port={self._port}, "
            f"secure={self._secure}, channels={self._num_channels})"
        )
