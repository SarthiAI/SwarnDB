"""SwarnDB exception hierarchy.

All SwarnDB-specific exceptions inherit from SwarnDBError, making it easy
to catch any SDK error with a single except clause.
"""

from __future__ import annotations

from typing import Optional


class SwarnDBError(Exception):
    """Base exception for all SwarnDB errors."""

    def __init__(self, message: str = "", details: Optional[str] = None) -> None:
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message


# --- Connection errors ---


class ConnectionError(SwarnDBError):
    """Raised when a connection to SwarnDB cannot be established or is lost."""


class AuthenticationError(SwarnDBError):
    """Raised when authentication fails (invalid or missing credentials)."""


# --- Collection errors ---


class CollectionError(SwarnDBError):
    """Base exception for collection-related errors."""


class CollectionNotFoundError(CollectionError):
    """Raised when a referenced collection does not exist."""

    def __init__(self, collection_name: str) -> None:
        super().__init__(
            message=f"Collection not found: '{collection_name}'",
            details=collection_name,
        )
        self.collection_name = collection_name


class CollectionExistsError(CollectionError):
    """Raised when attempting to create a collection that already exists."""

    def __init__(self, collection_name: str) -> None:
        super().__init__(
            message=f"Collection already exists: '{collection_name}'",
            details=collection_name,
        )
        self.collection_name = collection_name


# --- Vector errors ---


class VectorError(SwarnDBError):
    """Base exception for vector-related errors."""


class VectorNotFoundError(VectorError):
    """Raised when a referenced vector does not exist."""

    def __init__(self, vector_id: str) -> None:
        super().__init__(
            message=f"Vector not found: '{vector_id}'",
            details=vector_id,
        )
        self.vector_id = vector_id


class DimensionMismatchError(VectorError):
    """Raised when a vector's dimension does not match the collection's dimension."""

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(
            message=f"Dimension mismatch: expected {expected}, got {got}",
            details=f"expected={expected}, got={got}",
        )
        self.expected = expected
        self.got = got


# --- Search errors ---


class SearchError(SwarnDBError):
    """Raised when a search operation fails."""


# --- Graph errors ---


class GraphError(SwarnDBError):
    """Raised when a graph operation fails."""


# --- Math errors ---


class MathError(SwarnDBError):
    """Raised when a vector math operation fails."""
