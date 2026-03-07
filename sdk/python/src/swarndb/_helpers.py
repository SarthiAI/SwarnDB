"""Shared helper functions used across SwarnDB SDK modules.

This module centralises small utilities that would otherwise be duplicated
in multiple places (e.g. vectors.py, math_ops.py, client.py, async_client.py).
"""

from __future__ import annotations

from typing import List

import grpc

from swarndb._proto import common_pb2
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


def _to_proto_vector(v: List[float]) -> common_pb2.Vector:
    """Convert a Python list of floats to a proto Vector message."""
    return common_pb2.Vector(values=v)


def _translate_error(exc: grpc.RpcError) -> SwarnDBError:
    """Map a gRPC RpcError to the appropriate SwarnDB exception."""
    code = exc.code()
    details = exc.details() or ""

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
            return DimensionMismatchError(0, 0)
        if "search" in lower:
            return SearchError(details)
        if "graph" in lower:
            return GraphError(details)
        if "math" in lower:
            return MathError(details)
        return SwarnDBError(details)

    if code == grpc.StatusCode.UNAVAILABLE:
        return ConnectionError(
            f"SwarnDB server unavailable: {details}"
        )

    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return ConnectionError(
            f"Request timed out: {details}"
        )

    # Catch-all
    return SwarnDBError(
        message=f"gRPC error ({code.name}): {details}",
        details=details,
    )
