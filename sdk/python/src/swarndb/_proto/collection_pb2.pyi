from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreateCollectionRequest(_message.Message):
    __slots__ = ("name", "dimension", "distance_metric", "default_threshold", "max_vectors")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_VECTORS_FIELD_NUMBER: _ClassVar[int]
    name: str
    dimension: int
    distance_metric: str
    default_threshold: float
    max_vectors: int
    def __init__(self, name: _Optional[str] = ..., dimension: _Optional[int] = ..., distance_metric: _Optional[str] = ..., default_threshold: _Optional[float] = ..., max_vectors: _Optional[int] = ...) -> None: ...

class CreateCollectionResponse(_message.Message):
    __slots__ = ("name", "success")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    name: str
    success: bool
    def __init__(self, name: _Optional[str] = ..., success: bool = ...) -> None: ...

class DeleteCollectionRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class DeleteCollectionResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class GetCollectionRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetCollectionResponse(_message.Message):
    __slots__ = ("name", "dimension", "distance_metric", "vector_count", "default_threshold", "status")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    VECTOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    name: str
    dimension: int
    distance_metric: str
    vector_count: int
    default_threshold: float
    status: str
    def __init__(self, name: _Optional[str] = ..., dimension: _Optional[int] = ..., distance_metric: _Optional[str] = ..., vector_count: _Optional[int] = ..., default_threshold: _Optional[float] = ..., status: _Optional[str] = ...) -> None: ...

class ListCollectionsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListCollectionsResponse(_message.Message):
    __slots__ = ("collections",)
    COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    collections: _containers.RepeatedCompositeFieldContainer[GetCollectionResponse]
    def __init__(self, collections: _Optional[_Iterable[_Union[GetCollectionResponse, _Mapping]]] = ...) -> None: ...
