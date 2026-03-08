from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FilterExpression(_message.Message):
    __slots__ = ("field",)
    AND_FIELD_NUMBER: _ClassVar[int]
    OR_FIELD_NUMBER: _ClassVar[int]
    NOT_FIELD_NUMBER: _ClassVar[int]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    field: FieldFilter
    def __init__(self, field: _Optional[_Union[FieldFilter, _Mapping]] = ..., **kwargs) -> None: ...

class AndFilter(_message.Message):
    __slots__ = ("filters",)
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    filters: _containers.RepeatedCompositeFieldContainer[FilterExpression]
    def __init__(self, filters: _Optional[_Iterable[_Union[FilterExpression, _Mapping]]] = ...) -> None: ...

class OrFilter(_message.Message):
    __slots__ = ("filters",)
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    filters: _containers.RepeatedCompositeFieldContainer[FilterExpression]
    def __init__(self, filters: _Optional[_Iterable[_Union[FilterExpression, _Mapping]]] = ...) -> None: ...

class NotFilter(_message.Message):
    __slots__ = ("filter",)
    FILTER_FIELD_NUMBER: _ClassVar[int]
    filter: FilterExpression
    def __init__(self, filter: _Optional[_Union[FilterExpression, _Mapping]] = ...) -> None: ...

class FieldFilter(_message.Message):
    __slots__ = ("field", "op", "value", "values")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    field: str
    op: str
    value: _common_pb2.MetadataValue
    values: _containers.RepeatedCompositeFieldContainer[_common_pb2.MetadataValue]
    def __init__(self, field: _Optional[str] = ..., op: _Optional[str] = ..., value: _Optional[_Union[_common_pb2.MetadataValue, _Mapping]] = ..., values: _Optional[_Iterable[_Union[_common_pb2.MetadataValue, _Mapping]]] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("collection", "query", "k", "filter", "strategy", "include_metadata", "include_graph", "graph_threshold", "max_graph_edges", "ef_search")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_METADATA_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_GRAPH_FIELD_NUMBER: _ClassVar[int]
    GRAPH_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_GRAPH_EDGES_FIELD_NUMBER: _ClassVar[int]
    EF_SEARCH_FIELD_NUMBER: _ClassVar[int]
    collection: str
    query: _common_pb2.Vector
    k: int
    filter: FilterExpression
    strategy: str
    include_metadata: bool
    include_graph: bool
    graph_threshold: float
    max_graph_edges: int
    ef_search: int
    def __init__(self, collection: _Optional[str] = ..., query: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., k: _Optional[int] = ..., filter: _Optional[_Union[FilterExpression, _Mapping]] = ..., strategy: _Optional[str] = ..., include_metadata: bool = ..., include_graph: bool = ..., graph_threshold: _Optional[float] = ..., max_graph_edges: _Optional[int] = ..., ef_search: _Optional[int] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("results", "search_time_us", "warning")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    SEARCH_TIME_US_FIELD_NUMBER: _ClassVar[int]
    WARNING_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[_common_pb2.ScoredResult]
    search_time_us: int
    warning: str
    def __init__(self, results: _Optional[_Iterable[_Union[_common_pb2.ScoredResult, _Mapping]]] = ..., search_time_us: _Optional[int] = ..., warning: _Optional[str] = ...) -> None: ...

class BatchSearchRequest(_message.Message):
    __slots__ = ("queries",)
    QUERIES_FIELD_NUMBER: _ClassVar[int]
    queries: _containers.RepeatedCompositeFieldContainer[SearchRequest]
    def __init__(self, queries: _Optional[_Iterable[_Union[SearchRequest, _Mapping]]] = ...) -> None: ...

class BatchSearchResponse(_message.Message):
    __slots__ = ("results", "total_time_us")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TIME_US_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[SearchResponse]
    total_time_us: int
    def __init__(self, results: _Optional[_Iterable[_Union[SearchResponse, _Mapping]]] = ..., total_time_us: _Optional[int] = ...) -> None: ...
