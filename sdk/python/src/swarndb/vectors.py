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
from swarndb.types import (
    BulkInsertFromPathRequest,
    BulkInsertOptions,
    BulkInsertResult,
    VectorRecord,
)

# Re-exported so other modules can import VectorNotFoundError from here if
# they need to. `vectors.get` no longer raises it for the missing-id case;
# update/delete still do.
__all__ = ["VectorAPI", "VectorNotFoundError"]

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

    def get(self, collection: str, id: int) -> Optional[VectorRecord]:
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
            response = self._client._call(
                self._client._vector_stub.Get, request
            )
        except VectorNotFoundError:
            return None
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
        checkpoint_every: Optional[int] = None,
        resume_token: Optional[str] = None,
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
            checkpoint_every=checkpoint_every,
            resume_token=resume_token,
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
                    checkpoint_every=options.checkpoint_every or 0,
                    resume_token=options.resume_token or "",
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

    # ------------------------------------------------------------------
    # Bulk Insert From Path (mmap)
    # ------------------------------------------------------------------

    def bulk_insert_from_path(
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

        response = self._client._call(
            self._client._vector_stub.BulkInsertFromPath, request
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
