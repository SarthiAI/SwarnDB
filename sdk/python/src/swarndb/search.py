"""SwarnDB search operations.

This module provides similarity search with a fluent filter builder,
single and batch search, and filter strategy selection.

Classes:
    Filter       -- Fluent builder for composing metadata filter expressions.
    FieldBuilder -- Chained field-level filter operations.
    SearchAPI    -- Search facade wrapping the gRPC SearchService.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from swarndb._proto import common_pb2, search_pb2
from swarndb.exceptions import SearchError, SwarnDBError
from swarndb.types import GraphEdge, ScoredResult, SearchResult, BatchSearchResult
from swarndb.vectors import _from_proto_metadata

if TYPE_CHECKING:
    from swarndb.client import SwarnDBClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_OPS = frozenset({
    "eq", "ne", "gt", "gte", "lt", "lte", "in", "between", "exists", "contains",
})

_VALID_STRATEGIES = frozenset({"auto", "pre_filter", "post_filter"})


def _to_proto_value(value: Any) -> common_pb2.MetadataValue:
    """Convert a Python value to a protobuf MetadataValue."""
    if isinstance(value, bool):
        return common_pb2.MetadataValue(bool_value=value)
    if isinstance(value, int):
        return common_pb2.MetadataValue(int_value=value)
    if isinstance(value, float):
        return common_pb2.MetadataValue(float_value=value)
    if isinstance(value, str):
        return common_pb2.MetadataValue(string_value=value)
    if isinstance(value, (list, tuple)):
        # Assume list of strings for StringList
        return common_pb2.MetadataValue(
            string_list_value=common_pb2.StringList(values=[str(v) for v in value]),
        )
    raise TypeError(f"Unsupported metadata value type: {type(value).__name__}")


def _scored_result_from_proto(proto: common_pb2.ScoredResult) -> ScoredResult:
    """Convert a protobuf ScoredResult to the SDK dataclass."""
    metadata: dict[str, Any] = {}
    if proto.HasField("metadata"):
        metadata = _from_proto_metadata(proto.metadata)
    graph_edges = [
        GraphEdge(target_id=e.target_id, similarity=e.similarity)
        for e in proto.graph_edges
    ]
    return ScoredResult(
        id=proto.id,
        score=proto.score,
        metadata=metadata,
        graph_edges=graph_edges,
    )


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------


class Filter:
    """Fluent filter builder for search queries.

    Supports composing complex metadata filters using Python operators:

        Filter.eq("category", "electronics")
        Filter.gt("price", 50) & Filter.lt("price", 200)
        Filter.eq("brand", "acme") | Filter.eq("brand", "globex")
        ~Filter.eq("discontinued", True)
        Filter.field("tags").contains("sale")
        Filter.field("price").between(10, 100)
        Filter.field("color").in_(["red", "blue", "green"])
    """

    __slots__ = ("_kind", "_data")

    # Internal kinds: "field", "and", "or", "not"
    def __init__(self, kind: str, data: Any) -> None:
        self._kind = kind
        self._data = data

    # ------------------------------------------------------------------
    # Static factory methods for field-level filters
    # ------------------------------------------------------------------

    @staticmethod
    def eq(field: str, value: Any) -> Filter:
        """Equality filter: field == value."""
        return Filter("field", {"field": field, "op": "eq", "value": value})

    @staticmethod
    def ne(field: str, value: Any) -> Filter:
        """Not-equal filter: field != value."""
        return Filter("field", {"field": field, "op": "ne", "value": value})

    @staticmethod
    def gt(field: str, value: Any) -> Filter:
        """Greater-than filter: field > value."""
        return Filter("field", {"field": field, "op": "gt", "value": value})

    @staticmethod
    def gte(field: str, value: Any) -> Filter:
        """Greater-than-or-equal filter: field >= value."""
        return Filter("field", {"field": field, "op": "gte", "value": value})

    @staticmethod
    def lt(field: str, value: Any) -> Filter:
        """Less-than filter: field < value."""
        return Filter("field", {"field": field, "op": "lt", "value": value})

    @staticmethod
    def lte(field: str, value: Any) -> Filter:
        """Less-than-or-equal filter: field <= value."""
        return Filter("field", {"field": field, "op": "lte", "value": value})

    @staticmethod
    def in_(field: str, values: list) -> Filter:
        """Membership filter: field in values."""
        return Filter("field", {"field": field, "op": "in", "values": values})

    @staticmethod
    def between(field: str, low: Any, high: Any) -> Filter:
        """Range filter: low <= field <= high."""
        return Filter("field", {"field": field, "op": "between", "values": [low, high]})

    @staticmethod
    def exists(field: str) -> Filter:
        """Existence filter: field is present."""
        return Filter("field", {"field": field, "op": "exists", "value": True})

    @staticmethod
    def contains(field: str, value: str) -> Filter:
        """Contains filter: field contains value."""
        return Filter("field", {"field": field, "op": "contains", "value": value})

    @staticmethod
    def field(name: str) -> FieldBuilder:
        """Start a chained field filter: Filter.field("price").gt(50)."""
        return FieldBuilder(name)

    # ------------------------------------------------------------------
    # Boolean combinators (Python operator overloads)
    # ------------------------------------------------------------------

    def __and__(self, other: Filter) -> Filter:
        """Combine two filters with AND logic."""
        if not isinstance(other, Filter):
            return NotImplemented
        # Flatten nested ANDs for cleaner proto output
        left = self._data if self._kind == "and" else [self]
        right = other._data if other._kind == "and" else [other]
        return Filter("and", list(left) + list(right))

    def __or__(self, other: Filter) -> Filter:
        """Combine two filters with OR logic."""
        if not isinstance(other, Filter):
            return NotImplemented
        left = self._data if self._kind == "or" else [self]
        right = other._data if other._kind == "or" else [other]
        return Filter("or", list(left) + list(right))

    def __invert__(self) -> Filter:
        """Negate a filter with NOT logic."""
        return Filter("not", self)

    # ------------------------------------------------------------------
    # Protobuf serialization
    # ------------------------------------------------------------------

    def _to_proto(self) -> search_pb2.FilterExpression:
        """Convert this filter tree into a protobuf FilterExpression."""
        if self._kind == "field":
            data = self._data
            field_filter = search_pb2.FieldFilter(
                field=data["field"],
                op=data["op"],
            )
            if "value" in data:
                field_filter.value.CopyFrom(_to_proto_value(data["value"]))
            if "values" in data:
                for v in data["values"]:
                    field_filter.values.append(_to_proto_value(v))
            return search_pb2.FilterExpression(field=field_filter)

        if self._kind == "and":
            sub_filters = [f._to_proto() for f in self._data]
            and_filter = search_pb2.AndFilter(filters=sub_filters)
            return search_pb2.FilterExpression(**{"and": and_filter})

        if self._kind == "or":
            sub_filters = [f._to_proto() for f in self._data]
            or_filter = search_pb2.OrFilter(filters=sub_filters)
            return search_pb2.FilterExpression(**{"or": or_filter})

        if self._kind == "not":
            inner = self._data._to_proto()
            not_filter = search_pb2.NotFilter(filter=inner)
            return search_pb2.FilterExpression(**{"not": not_filter})

        raise ValueError(f"Unknown filter kind: {self._kind}")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._kind == "field":
            d = self._data
            if "values" in d:
                return f"Filter.{d['op']}({d['field']!r}, {d['values']!r})"
            return f"Filter.{d['op']}({d['field']!r}, {d.get('value')!r})"
        if self._kind == "and":
            return " & ".join(repr(f) for f in self._data)
        if self._kind == "or":
            return " | ".join(repr(f) for f in self._data)
        if self._kind == "not":
            return f"~({self._data!r})"
        return f"Filter(kind={self._kind!r})"


# ---------------------------------------------------------------------------
# FieldBuilder for chained field operations
# ---------------------------------------------------------------------------


class FieldBuilder:
    """Chained field filter builder: ``Filter.field("price").gt(50)``."""

    __slots__ = ("_field_name",)

    def __init__(self, field_name: str) -> None:
        self._field_name = field_name

    def eq(self, value: Any) -> Filter:
        """Equality filter."""
        return Filter.eq(self._field_name, value)

    def ne(self, value: Any) -> Filter:
        """Not-equal filter."""
        return Filter.ne(self._field_name, value)

    def gt(self, value: Any) -> Filter:
        """Greater-than filter."""
        return Filter.gt(self._field_name, value)

    def gte(self, value: Any) -> Filter:
        """Greater-than-or-equal filter."""
        return Filter.gte(self._field_name, value)

    def lt(self, value: Any) -> Filter:
        """Less-than filter."""
        return Filter.lt(self._field_name, value)

    def lte(self, value: Any) -> Filter:
        """Less-than-or-equal filter."""
        return Filter.lte(self._field_name, value)

    def in_(self, values: list) -> Filter:
        """Membership filter."""
        return Filter.in_(self._field_name, values)

    def between(self, low: Any, high: Any) -> Filter:
        """Range filter."""
        return Filter.between(self._field_name, low, high)

    def exists(self) -> Filter:
        """Existence filter."""
        return Filter.exists(self._field_name)

    def contains(self, value: str) -> Filter:
        """Contains filter."""
        return Filter.contains(self._field_name, value)

    def __repr__(self) -> str:
        return f"FieldBuilder({self._field_name!r})"


# ---------------------------------------------------------------------------
# SearchAPI
# ---------------------------------------------------------------------------


class SearchAPI:
    """Search facade wrapping the gRPC SearchService.

    Provides single-query and batch search with optional metadata filtering
    and configurable filter strategy.

    Usage::

        results = client.search.query("my_collection", vector, k=10)
        results = client.search.query(
            "my_collection", vector, k=5,
            filter=Filter.eq("category", "electronics"),
        )
        batch = client.search.batch(
            "my_collection", [vec1, vec2], k=10,
        )
    """

    def __init__(self, client: SwarnDBClient) -> None:
        self._client = client

    def query(
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

        response = self._client._call(
            self._client._search_stub.Search,
            request,
        )

        results = [_scored_result_from_proto(r) for r in response.results]
        return SearchResult(
            results=results,
            search_time_us=response.search_time_us,
            warning=getattr(response, 'warning', '') or '',
        )

    def batch(
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
            search_requests.append(req)

        batch_request = search_pb2.BatchSearchRequest(queries=search_requests)

        response = self._client._call(
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
