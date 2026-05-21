from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RecoveryPath(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RECOVERY_UNKNOWN: _ClassVar[RecoveryPath]
    RECOVERY_CLEAN_SHUTDOWN: _ClassVar[RecoveryPath]
    RECOVERY_INCREMENTAL_REPLAY: _ClassVar[RecoveryPath]
    RECOVERY_FULL_REBUILD: _ClassVar[RecoveryPath]
RECOVERY_UNKNOWN: RecoveryPath
RECOVERY_CLEAN_SHUTDOWN: RecoveryPath
RECOVERY_INCREMENTAL_REPLAY: RecoveryPath
RECOVERY_FULL_REBUILD: RecoveryPath

class ScalarQuantization(_message.Message):
    __slots__ = ("quantile", "always_ram")
    QUANTILE_FIELD_NUMBER: _ClassVar[int]
    ALWAYS_RAM_FIELD_NUMBER: _ClassVar[int]
    quantile: float
    always_ram: bool
    def __init__(self, quantile: _Optional[float] = ..., always_ram: bool = ...) -> None: ...

class QuantizationConfig(_message.Message):
    __slots__ = ("scalar",)
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    scalar: ScalarQuantization
    def __init__(self, scalar: _Optional[_Union[ScalarQuantization, _Mapping]] = ...) -> None: ...

class CreateCollectionRequest(_message.Message):
    __slots__ = ("name", "dimension", "distance_metric", "default_threshold", "max_vectors", "quantization", "m", "ef_construction")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_VECTORS_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    M_FIELD_NUMBER: _ClassVar[int]
    EF_CONSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    dimension: int
    distance_metric: str
    default_threshold: float
    max_vectors: int
    quantization: QuantizationConfig
    m: int
    ef_construction: int
    def __init__(self, name: _Optional[str] = ..., dimension: _Optional[int] = ..., distance_metric: _Optional[str] = ..., default_threshold: _Optional[float] = ..., max_vectors: _Optional[int] = ..., quantization: _Optional[_Union[QuantizationConfig, _Mapping]] = ..., m: _Optional[int] = ..., ef_construction: _Optional[int] = ...) -> None: ...

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
    __slots__ = ("name", "dimension", "distance_metric", "vector_count", "default_threshold", "status", "quantization_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    VECTOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    dimension: int
    distance_metric: str
    vector_count: int
    default_threshold: float
    status: str
    quantization_type: str
    def __init__(self, name: _Optional[str] = ..., dimension: _Optional[int] = ..., distance_metric: _Optional[str] = ..., vector_count: _Optional[int] = ..., default_threshold: _Optional[float] = ..., status: _Optional[str] = ..., quantization_type: _Optional[str] = ...) -> None: ...

class ListCollectionsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListCollectionsResponse(_message.Message):
    __slots__ = ("collections",)
    COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    collections: _containers.RepeatedCompositeFieldContainer[GetCollectionResponse]
    def __init__(self, collections: _Optional[_Iterable[_Union[GetCollectionResponse, _Mapping]]] = ...) -> None: ...

class GetRecoveryStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CollectionRecoveryEntry(_message.Message):
    __slots__ = ("name", "path")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    name: str
    path: RecoveryPath
    def __init__(self, name: _Optional[str] = ..., path: _Optional[_Union[RecoveryPath, str]] = ...) -> None: ...

class GetRecoveryStatusResponse(_message.Message):
    __slots__ = ("elapsed_secs", "collections", "path")
    ELAPSED_SECS_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    elapsed_secs: int
    collections: _containers.RepeatedCompositeFieldContainer[CollectionRecoveryEntry]
    path: RecoveryPath
    def __init__(self, elapsed_secs: _Optional[int] = ..., collections: _Optional[_Iterable[_Union[CollectionRecoveryEntry, _Mapping]]] = ..., path: _Optional[_Union[RecoveryPath, str]] = ...) -> None: ...

class SnapshotCollectionRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class SnapshotCollectionResponse(_message.Message):
    __slots__ = ("last_snapshot_lsn",)
    LAST_SNAPSHOT_LSN_FIELD_NUMBER: _ClassVar[int]
    last_snapshot_lsn: int
    def __init__(self, last_snapshot_lsn: _Optional[int] = ...) -> None: ...

class GetPersistenceStatusRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetPersistenceStatusResponse(_message.Message):
    __slots__ = ("last_snapshot_lsn", "current_lsn", "next_lsn")
    LAST_SNAPSHOT_LSN_FIELD_NUMBER: _ClassVar[int]
    CURRENT_LSN_FIELD_NUMBER: _ClassVar[int]
    NEXT_LSN_FIELD_NUMBER: _ClassVar[int]
    last_snapshot_lsn: int
    current_lsn: int
    next_lsn: int
    def __init__(self, last_snapshot_lsn: _Optional[int] = ..., current_lsn: _Optional[int] = ..., next_lsn: _Optional[int] = ...) -> None: ...

class GetCollectionMetricsRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetCollectionMetricsResponse(_message.Message):
    __slots__ = ("map_lock_acquisitions", "collection_read_acquisitions", "collection_write_acquisitions", "total_blocked_microseconds")
    MAP_LOCK_ACQUISITIONS_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_READ_ACQUISITIONS_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_WRITE_ACQUISITIONS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BLOCKED_MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
    map_lock_acquisitions: int
    collection_read_acquisitions: int
    collection_write_acquisitions: int
    total_blocked_microseconds: int
    def __init__(self, map_lock_acquisitions: _Optional[int] = ..., collection_read_acquisitions: _Optional[int] = ..., collection_write_acquisitions: _Optional[int] = ..., total_blocked_microseconds: _Optional[int] = ...) -> None: ...
