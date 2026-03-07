from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetRelatedRequest(_message.Message):
    __slots__ = ("collection", "vector_id", "threshold", "max_results")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    VECTOR_ID_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    vector_id: int
    threshold: float
    max_results: int
    def __init__(self, collection: _Optional[str] = ..., vector_id: _Optional[int] = ..., threshold: _Optional[float] = ..., max_results: _Optional[int] = ...) -> None: ...

class GraphEdge(_message.Message):
    __slots__ = ("target_id", "similarity")
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    target_id: int
    similarity: float
    def __init__(self, target_id: _Optional[int] = ..., similarity: _Optional[float] = ...) -> None: ...

class GetRelatedResponse(_message.Message):
    __slots__ = ("edges",)
    EDGES_FIELD_NUMBER: _ClassVar[int]
    edges: _containers.RepeatedCompositeFieldContainer[GraphEdge]
    def __init__(self, edges: _Optional[_Iterable[_Union[GraphEdge, _Mapping]]] = ...) -> None: ...

class TraverseRequest(_message.Message):
    __slots__ = ("collection", "start_id", "depth", "threshold", "max_results")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    START_ID_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    start_id: int
    depth: int
    threshold: float
    max_results: int
    def __init__(self, collection: _Optional[str] = ..., start_id: _Optional[int] = ..., depth: _Optional[int] = ..., threshold: _Optional[float] = ..., max_results: _Optional[int] = ...) -> None: ...

class TraversalNode(_message.Message):
    __slots__ = ("id", "depth", "path_similarity", "path")
    ID_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    PATH_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    id: int
    depth: int
    path_similarity: float
    path: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, id: _Optional[int] = ..., depth: _Optional[int] = ..., path_similarity: _Optional[float] = ..., path: _Optional[_Iterable[int]] = ...) -> None: ...

class TraverseResponse(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[TraversalNode]
    def __init__(self, nodes: _Optional[_Iterable[_Union[TraversalNode, _Mapping]]] = ...) -> None: ...

class SetThresholdRequest(_message.Message):
    __slots__ = ("collection", "vector_id", "threshold")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    VECTOR_ID_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    collection: str
    vector_id: int
    threshold: float
    def __init__(self, collection: _Optional[str] = ..., vector_id: _Optional[int] = ..., threshold: _Optional[float] = ...) -> None: ...

class SetThresholdResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
