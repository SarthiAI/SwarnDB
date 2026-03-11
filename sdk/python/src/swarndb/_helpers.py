"""Shared helper functions used across SwarnDB SDK modules.

This module centralises small utilities that would otherwise be duplicated
in multiple places (e.g. vectors.py, math_ops.py, client.py, async_client.py).
"""

from __future__ import annotations

import re
from typing import List

import grpc

from swarndb._proto import common_pb2
from swarndb.exceptions import (
    AuthenticationError,
    CollectionExistsError,
    CollectionNotFoundError,
    SwarnDBConnectionError,
    DimensionMismatchError,
    GraphError,
    MathError,
    SearchError,
    SwarnDBError,
    VectorNotFoundError,
)

_MAX_ERROR_DETAIL_LENGTH = 2000


def _to_proto_vector(v: List[float]) -> common_pb2.Vector:
    """Convert a Python list of floats to a proto Vector message."""
    return common_pb2.Vector(values=v)


def _truncate_details(details: str) -> str:
    """Truncate error details to prevent leaking verbose server internals."""
    if len(details) > _MAX_ERROR_DETAIL_LENGTH:
        return details[:_MAX_ERROR_DETAIL_LENGTH] + "...(truncated)"
    return details


def _translate_error(exc: grpc.RpcError) -> SwarnDBError:
    """Map a gRPC RpcError to the appropriate SwarnDB exception."""
    code = exc.code()
    details = _truncate_details(exc.details() or "")

    if code == grpc.StatusCode.NOT_FOUND:
        lower = details.lower()
        if "collection" in lower:
            name = details.split("'")[1] if "'" in details else details
            return CollectionNotFoundError(name)
        if "vector" in lower:
            vid = details.split("'")[1] if "'" in details else details
            return VectorNotFoundError(vid)
        return SwarnDBError(details)

    if code == grpc.StatusCode.ALREADY_EXISTS:
        name = details.split("'")[1] if "'" in details else details
        return CollectionExistsError(name)

    if code == grpc.StatusCode.UNAUTHENTICATED:
        return AuthenticationError(details)

    if code == grpc.StatusCode.PERMISSION_DENIED:
        return AuthenticationError(details)

    if code == grpc.StatusCode.INVALID_ARGUMENT:
        lower = details.lower()
        if "dimension" in lower:
            match = re.search(r'expected\s+(?:dimension\s+)?(\d+).*?got\s+(\d+)', lower)
            if match:
                return DimensionMismatchError(int(match.group(1)), int(match.group(2)))
            return DimensionMismatchError(0, 0)
        if "search" in lower:
            return SearchError(details)
        if "graph" in lower:
            return GraphError(details)
        if "math" in lower:
            return MathError(details)
        return SwarnDBError(details)

    if code == grpc.StatusCode.UNAVAILABLE:
        return SwarnDBConnectionError(
            f"SwarnDB server unavailable: {details}"
        )

    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return SwarnDBConnectionError(
            f"Request timed out: {details}"
        )

    # Catch-all
    return SwarnDBError(
        message=f"gRPC error ({code.name}): {details}",
        details=details,
    )
