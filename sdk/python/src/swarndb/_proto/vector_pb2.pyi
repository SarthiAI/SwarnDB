from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InsertRequest(_message.Message):
    __slots__ = ("collection", "vector", "metadata", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    vector: _common_pb2.Vector
    metadata: _common_pb2.Metadata
    id: int
    def __init__(self, collection: _Optional[str] = ..., vector: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., metadata: _Optional[_Union[_common_pb2.Metadata, _Mapping]] = ..., id: _Optional[int] = ...) -> None: ...

class InsertResponse(_message.Message):
    __slots__ = ("id", "success")
    ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    id: int
    success: bool
    def __init__(self, id: _Optional[int] = ..., success: bool = ...) -> None: ...

class GetVectorRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: int
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[int] = ...) -> None: ...

class GetVectorResponse(_message.Message):
    __slots__ = ("id", "vector", "metadata")
    ID_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: int
    vector: _common_pb2.Vector
    metadata: _common_pb2.Metadata
    def __init__(self, id: _Optional[int] = ..., vector: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., metadata: _Optional[_Union[_common_pb2.Metadata, _Mapping]] = ...) -> None: ...

class UpdateRequest(_message.Message):
    __slots__ = ("collection", "id", "vector", "metadata")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: int
    vector: _common_pb2.Vector
    metadata: _common_pb2.Metadata
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[int] = ..., vector: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., metadata: _Optional[_Union[_common_pb2.Metadata, _Mapping]] = ...) -> None: ...

class UpdateResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class DeleteVectorRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: int
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[int] = ...) -> None: ...

class DeleteVectorResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class BulkInsertResponse(_message.Message):
    __slots__ = ("inserted_count", "errors")
    INSERTED_COUNT_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    inserted_count: int
    errors: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, inserted_count: _Optional[int] = ..., errors: _Optional[_Iterable[str]] = ...) -> None: ...
