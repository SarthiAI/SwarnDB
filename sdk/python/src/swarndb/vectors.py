"""SwarnDB vector operations.

This module provides vector CRUD operations: insert, get, update, delete,
and bulk insert with optional progress bars for bulk operations.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import grpc

from swarndb._helpers import _to_proto_vector
from swarndb._proto import common_pb2, vector_pb2
from swarndb.exceptions import SwarnDBError, VectorNotFoundError
from swarndb.types import BulkInsertOptions, BulkInsertResult, VectorRecord

if TYPE_CHECKING:
    from swarndb.client import SwarnDBClient

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proto conversion helpers
# ---------------------------------------------------------------------------


def _to_proto_metadata(d: Dict[str, Any]) -> common_pb2.Metadata:
    """Convert a Python dict to a proto Metadata message.

    Supported value types:
    - bool  (checked before int, since bool is a subclass of int)
    - int   -> int_value
    - float -> float_value
    - str   -> string_value
    - list of str -> string_list_value
    """
    fields: Dict[str, common_pb2.MetadataValue] = {}
    for key, value in d.items():
        if isinstance(value, bool):
            fields[key] = common_pb2.MetadataValue(bool_value=value)
        elif isinstance(value, int):
            fields[key] = common_pb2.MetadataValue(int_value=value)
        elif isinstance(value, float):
            fields[key] = common_pb2.MetadataValue(float_value=value)
        elif isinstance(value, str):
            fields[key] = common_pb2.MetadataValue(string_value=value)
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            string_list = common_pb2.StringList(values=value)
            fields[key] = common_pb2.MetadataValue(string_list_value=string_list)
        else:
            raise ValueError(
                f"Unsupported metadata value type for key '{key}': "
                f"{type(value).__name__}"
            )
    return common_pb2.Metadata(fields=fields)


def _from_proto_metadata(m: common_pb2.Metadata) -> Dict[str, Any]:
    """Convert a proto Metadata message to a Python dict."""
    result: Dict[str, Any] = {}
    for key, mv in m.fields.items():
        try:
            which = mv.WhichOneof("value")
        except ValueError:
            which = None

        if which == "string_value":
            result[key] = mv.string_value
        elif which == "int_value":
            result[key] = mv.int_value
        elif which == "float_value":
            result[key] = mv.float_value
        elif which == "bool_value":
            result[key] = mv.bool_value
        elif which == "string_list_value":
            result[key] = list(mv.string_list_value.values)
        else:
            # Fallback: try string_value (default for proto3)
            result[key] = mv.string_value
    return result


# ---------------------------------------------------------------------------
# VectorAPI
# ---------------------------------------------------------------------------


class VectorAPI:
    """Vector CRUD operations for a SwarnDB collection."""

    def __init__(self, client: SwarnDBClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(
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

        response = self._client._call(
            self._client._vector_stub.Insert, request
        )
        return response.id

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(self, collection: str, id: int) -> VectorRecord:
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
        response = self._client._call(
            self._client._vector_stub.Get, request
        )
        return VectorRecord(
            id=response.id,
            vector=list(response.vector.values),
            metadata=_from_proto_metadata(response.metadata),
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
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

        response = self._client._call(
            self._client._vector_stub.Update, request
        )
        return response.success

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, collection: str, id: int) -> bool:
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
        response = self._client._call(
            self._client._vector_stub.Delete, request
        )
        return response.success

    # ------------------------------------------------------------------
    # Bulk Insert
    # ------------------------------------------------------------------

    def bulk_insert(
        self,
        collection: str,
        vectors: List[List[float]],
        *,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
        batch_size: int = 1000,
        show_progress: bool = False,
        batch_lock_size: Optional[int] = None,
        defer_graph: bool = False,
        wal_flush_every: Optional[int] = None,
        ef_construction: Optional[int] = None,
        index_mode: Optional[str] = None,
        skip_metadata_index: bool = False,
        parallel_build: bool = False,
    ) -> BulkInsertResult:
        """Bulk insert vectors using streaming RPC.

        Streams InsertRequest messages to the server. Optionally displays a
        progress bar via tqdm when ``show_progress=True``.

        Args:
            collection: Target collection name.
            vectors: List of vector value lists.
            metadata_list: Optional per-vector metadata dicts (must match
                length of ``vectors`` if provided).
            ids: Optional per-vector IDs (must match length of ``vectors``
                if provided). Use 0 for auto-assignment.
            batch_size: Number of vectors per streaming batch (controls
                progress bar granularity).
            show_progress: If True, display a tqdm progress bar. Requires
                tqdm to be installed.
            batch_lock_size: Vectors per lock acquisition (server default=1,
                max=10000). Higher values reduce lock overhead.
            defer_graph: If True, skip graph computation during insert.
                Use ``client.collections.optimize()`` afterward.
            wal_flush_every: WAL flush interval in operations (default=1,
                0=disable WAL flushing for max throughput).
            ef_construction: Override HNSW ef_construction for this batch.
            index_mode: Index build mode: ``"immediate"`` (default) or
                ``"deferred"`` (build index after all inserts).
            skip_metadata_index: If True, skip metadata indexing during
                insert for faster ingestion.
            parallel_build: If True, use parallel HNSW construction
                (only effective with ``index_mode="deferred"``).

        Returns:
            A BulkInsertResult with inserted_count and any errors.

        Raises:
            ValueError: If metadata_list or ids length doesn't match vectors,
                or if index_mode is not a valid value.
            ImportError: If show_progress is True but tqdm is not installed.
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

        if show_progress and tqdm is None:
            raise ImportError(
                "tqdm is required for progress bars. "
                "Install it with: pip install tqdm"
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
        )
        use_optimized_rpc = options.has_non_defaults()

        def _request_iterator() -> Iterator[vector_pb2.InsertRequest]:
            """Yield InsertRequest messages, optionally with progress."""
            iterator = range(total)

            if show_progress and tqdm is not None:
                iterator = tqdm(
                    iterator,
                    total=total,
                    desc="Bulk inserting vectors",
                    unit="vec",
                )

            for i in iterator:
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

        def _options_request_iterator() -> Iterator:
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
                )
            )
            yield options_msg

            for req in _request_iterator():
                yield vector_pb2.BulkInsertStreamMessage(vector=req)

        # BulkInsert is stream_unary: we stream requests, get one response.
        # Cannot use _call (which is for unary-unary), so call stub directly
        # with retry logic.
        metadata = self._client._metadata()
        call_timeout = self._client._timeout
        last_error: Optional[grpc.RpcError] = None

        for attempt in range(self._client._max_retries + 1):
            try:
                if use_optimized_rpc:
                    logger.info(
                        "BulkInsert with options: batch_lock_size=%s, "
                        "defer_graph=%s, wal_flush_every=%s, "
                        "ef_construction=%s, index_mode=%s, "
                        "skip_metadata_index=%s, parallel_build=%s",
                        options.batch_lock_size,
                        options.defer_graph,
                        options.wal_flush_every,
                        options.ef_construction,
                        options.index_mode,
                        options.skip_metadata_index,
                        options.parallel_build,
                    )
                    response = self._client._vector_stub.BulkInsertWithOptions(
                        _options_request_iterator(),
                        timeout=call_timeout,
                        metadata=metadata,
                    )
                else:
                    response = self._client._vector_stub.BulkInsert(
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

                retryable = frozenset({
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                })
                if code in retryable and attempt < self._client._max_retries:
                    delay = self._client._retry_delay * (2 ** attempt)
                    logger.debug(
                        "Retrying BulkInsert (attempt %d/%d) after %s, "
                        "backoff %.2fs",
                        attempt + 1,
                        self._client._max_retries,
                        code.name,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                raise self._client._translate_error(exc) from exc

        # Should not reach here, but guard anyway
        assert last_error is not None
        raise self._client._translate_error(last_error) from last_error
