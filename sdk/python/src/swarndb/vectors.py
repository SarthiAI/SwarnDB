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
from swarndb.types import BulkInsertResult, VectorRecord

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

        Returns:
            A BulkInsertResult with inserted_count and any errors.

        Raises:
            ValueError: If metadata_list or ids length doesn't match vectors.
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

        if show_progress and tqdm is None:
            raise ImportError(
                "tqdm is required for progress bars. "
                "Install it with: pip install tqdm"
            )

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

        # BulkInsert is stream_unary: we stream requests, get one response.
        # Cannot use _call (which is for unary-unary), so call stub directly
        # with retry logic.
        metadata = self._client._metadata()
        call_timeout = self._client._timeout
        last_error: Optional[grpc.RpcError] = None

        for attempt in range(self._client._max_retries + 1):
            try:
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
