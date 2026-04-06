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

class BulkInsertOptions(_message.Message):
    __slots__ = ("batch_lock_size", "defer_graph", "wal_flush_every", "ef_construction", "index_mode", "skip_metadata_index", "parallel_build")
    BATCH_LOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    DEFER_GRAPH_FIELD_NUMBER: _ClassVar[int]
    WAL_FLUSH_EVERY_FIELD_NUMBER: _ClassVar[int]
    EF_CONSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    INDEX_MODE_FIELD_NUMBER: _ClassVar[int]
    SKIP_METADATA_INDEX_FIELD_NUMBER: _ClassVar[int]
    PARALLEL_BUILD_FIELD_NUMBER: _ClassVar[int]
    batch_lock_size: int
    defer_graph: bool
    wal_flush_every: int
    ef_construction: int
    index_mode: str
    skip_metadata_index: bool
    parallel_build: bool
    def __init__(self, batch_lock_size: _Optional[int] = ..., defer_graph: bool = ..., wal_flush_every: _Optional[int] = ..., ef_construction: _Optional[int] = ..., index_mode: _Optional[str] = ..., skip_metadata_index: bool = ..., parallel_build: bool = ...) -> None: ...

class BulkInsertStreamMessage(_message.Message):
    __slots__ = ("options", "vector")
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    options: BulkInsertOptions
    vector: InsertRequest
    def __init__(self, options: _Optional[_Union[BulkInsertOptions, _Mapping]] = ..., vector: _Optional[_Union[InsertRequest, _Mapping]] = ...) -> None: ...

class OptimizeRequest(_message.Message):
    __slots__ = ("collection", "rebuild_graph")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    REBUILD_GRAPH_FIELD_NUMBER: _ClassVar[int]
    collection: str
    rebuild_graph: bool
    def __init__(self, collection: _Optional[str] = ..., rebuild_graph: bool = ...) -> None: ...

class OptimizeResponse(_message.Message):
    __slots__ = ("status", "message", "duration_ms", "vectors_processed")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    VECTORS_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    status: str
    message: str
    duration_ms: int
    vectors_processed: int
    def __init__(self, status: _Optional[str] = ..., message: _Optional[str] = ..., duration_ms: _Optional[int] = ..., vectors_processed: _Optional[int] = ...) -> None: ...

class PruneWALRequest(_message.Message):
    __slots__ = ("collection",)
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    collection: str
    def __init__(self, collection: _Optional[str] = ...) -> None: ...

class PruneWALResponse(_message.Message):
    __slots__ = ("status", "files_deleted", "bytes_freed", "duration_ms")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    FILES_DELETED_FIELD_NUMBER: _ClassVar[int]
    BYTES_FREED_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    status: str
    files_deleted: int
    bytes_freed: int
    duration_ms: int
    def __init__(self, status: _Optional[str] = ..., files_deleted: _Optional[int] = ..., bytes_freed: _Optional[int] = ..., duration_ms: _Optional[int] = ...) -> None: ...

class CompactRequest(_message.Message):
    __slots__ = ("collection", "min_segments", "remove_deleted")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    MIN_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    REMOVE_DELETED_FIELD_NUMBER: _ClassVar[int]
    collection: str
    min_segments: int
    remove_deleted: bool
    def __init__(self, collection: _Optional[str] = ..., min_segments: _Optional[int] = ..., remove_deleted: bool = ...) -> None: ...

class CompactResponse(_message.Message):
    __slots__ = ("status", "segments_merged", "vectors_written", "vectors_removed", "duration_ms")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_MERGED_FIELD_NUMBER: _ClassVar[int]
    VECTORS_WRITTEN_FIELD_NUMBER: _ClassVar[int]
    VECTORS_REMOVED_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    status: str
    segments_merged: int
    vectors_written: int
    vectors_removed: int
    duration_ms: int
    def __init__(self, status: _Optional[str] = ..., segments_merged: _Optional[int] = ..., vectors_written: _Optional[int] = ..., vectors_removed: _Optional[int] = ..., duration_ms: _Optional[int] = ...) -> None: ...
