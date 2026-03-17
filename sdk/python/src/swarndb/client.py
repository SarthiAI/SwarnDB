"""SwarnDB synchronous client.

This module provides the main synchronous gRPC client for interacting with
a SwarnDB server. It wraps all collection, vector, search, graph, and math
operations behind a single connection-managed interface.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

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
from swarndb._proto import collection_pb2_grpc
from swarndb._proto import vector_pb2_grpc
from swarndb._proto import search_pb2_grpc
from swarndb._proto import graph_pb2_grpc
from swarndb._proto import vector_math_pb2_grpc

if TYPE_CHECKING:
    from swarndb.collections import CollectionAPI
    from swarndb.vectors import VectorAPI
    from swarndb.search import SearchAPI
    from swarndb.graph import GraphAPI
    from swarndb.math_ops import MathAPI

logger = logging.getLogger(__name__)

# gRPC status codes that are considered transient and eligible for retry.
_RETRYABLE_CODES = frozenset({
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
})


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

        # Create channel
        if secure:
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(target, credentials, options=channel_options)
        else:
            self._channel = grpc.insecure_channel(target, options=channel_options)

        # Wrap channel with auth interceptor if api_key is provided
        if api_key:
            interceptor = _AuthInterceptor(api_key)
            self._channel = grpc.intercept_channel(self._channel, interceptor)

        # Create all 5 service stubs
        self._collection_stub = collection_pb2_grpc.CollectionServiceStub(self._channel)
        self._vector_stub = vector_pb2_grpc.VectorServiceStub(self._channel)
        self._search_stub = search_pb2_grpc.SearchServiceStub(self._channel)
        self._graph_stub = graph_pb2_grpc.GraphServiceStub(self._channel)
        self._vector_math_stub = vector_math_pb2_grpc.VectorMathServiceStub(self._channel)

        # Lazy-initialized API facades
        self._collections: Optional[CollectionAPI] = None
        self._vectors: Optional[VectorAPI] = None
        self._search: Optional[SearchAPI] = None
        self._graph: Optional[GraphAPI] = None
        self._math: Optional[MathAPI] = None

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
    def math(self) -> MathAPI:
        """Access vector math operations."""
        if self._math is None:
            from swarndb.math_ops import MathAPI
            self._math = MathAPI(self)
        return self._math

    # ------------------------------------------------------------------
    # Internal call helpers
    # ------------------------------------------------------------------

    def _call(self, stub_method, request, *, timeout: Optional[float] = None):
        """Execute a gRPC call with retry logic and error handling.

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
                return stub_method(
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
                        "Retrying gRPC call (attempt %d/%d) after %s error, "
                        "backoff %.2fs",
                        attempt + 1,
                        self._max_retries,
                        code.name,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                # Non-retryable or exhausted retries — convert to SDK error
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

    def close(self) -> None:
        """Close the gRPC channel."""
        self._channel.close()

    def __enter__(self) -> SwarnDBClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"SwarnDBClient(host={self._host!r}, port={self._port}, "
            f"secure={self._secure})"
        )
