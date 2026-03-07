from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DetectGhostsRequest(_message.Message):
    __slots__ = ("collection", "threshold", "centroids", "auto_k", "metric")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    CENTROIDS_FIELD_NUMBER: _ClassVar[int]
    AUTO_K_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    collection: str
    threshold: float
    centroids: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vector]
    auto_k: int
    metric: str
    def __init__(self, collection: _Optional[str] = ..., threshold: _Optional[float] = ..., centroids: _Optional[_Iterable[_Union[_common_pb2.Vector, _Mapping]]] = ..., auto_k: _Optional[int] = ..., metric: _Optional[str] = ...) -> None: ...

class GhostVector(_message.Message):
    __slots__ = ("id", "isolation_score")
    ID_FIELD_NUMBER: _ClassVar[int]
    ISOLATION_SCORE_FIELD_NUMBER: _ClassVar[int]
    id: int
    isolation_score: float
    def __init__(self, id: _Optional[int] = ..., isolation_score: _Optional[float] = ...) -> None: ...

class DetectGhostsResponse(_message.Message):
    __slots__ = ("ghosts", "compute_time_us")
    GHOSTS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    ghosts: _containers.RepeatedCompositeFieldContainer[GhostVector]
    compute_time_us: int
    def __init__(self, ghosts: _Optional[_Iterable[_Union[GhostVector, _Mapping]]] = ..., compute_time_us: _Optional[int] = ...) -> None: ...

class ConeSearchRequest(_message.Message):
    __slots__ = ("collection", "direction", "aperture_radians")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    APERTURE_RADIANS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    direction: _common_pb2.Vector
    aperture_radians: float
    def __init__(self, collection: _Optional[str] = ..., direction: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., aperture_radians: _Optional[float] = ...) -> None: ...

class ConeSearchResult(_message.Message):
    __slots__ = ("id", "cosine_similarity", "angle_radians")
    ID_FIELD_NUMBER: _ClassVar[int]
    COSINE_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    ANGLE_RADIANS_FIELD_NUMBER: _ClassVar[int]
    id: int
    cosine_similarity: float
    angle_radians: float
    def __init__(self, id: _Optional[int] = ..., cosine_similarity: _Optional[float] = ..., angle_radians: _Optional[float] = ...) -> None: ...

class ConeSearchResponse(_message.Message):
    __slots__ = ("results", "compute_time_us")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[ConeSearchResult]
    compute_time_us: int
    def __init__(self, results: _Optional[_Iterable[_Union[ConeSearchResult, _Mapping]]] = ..., compute_time_us: _Optional[int] = ...) -> None: ...

class ComputeCentroidRequest(_message.Message):
    __slots__ = ("collection", "vector_ids", "weights")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    VECTOR_IDS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    vector_ids: _containers.RepeatedScalarFieldContainer[int]
    weights: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, collection: _Optional[str] = ..., vector_ids: _Optional[_Iterable[int]] = ..., weights: _Optional[_Iterable[float]] = ...) -> None: ...

class ComputeCentroidResponse(_message.Message):
    __slots__ = ("centroid", "compute_time_us")
    CENTROID_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    centroid: _common_pb2.Vector
    compute_time_us: int
    def __init__(self, centroid: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., compute_time_us: _Optional[int] = ...) -> None: ...

class InterpolateRequest(_message.Message):
    __slots__ = ("a", "b", "t", "method", "sequence_count")
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    T_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_COUNT_FIELD_NUMBER: _ClassVar[int]
    a: _common_pb2.Vector
    b: _common_pb2.Vector
    t: float
    method: str
    sequence_count: int
    def __init__(self, a: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., b: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., t: _Optional[float] = ..., method: _Optional[str] = ..., sequence_count: _Optional[int] = ...) -> None: ...

class InterpolateResponse(_message.Message):
    __slots__ = ("results", "compute_time_us")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vector]
    compute_time_us: int
    def __init__(self, results: _Optional[_Iterable[_Union[_common_pb2.Vector, _Mapping]]] = ..., compute_time_us: _Optional[int] = ...) -> None: ...

class DetectDriftRequest(_message.Message):
    __slots__ = ("collection", "window1_ids", "window2_ids", "metric", "threshold")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    WINDOW1_IDS_FIELD_NUMBER: _ClassVar[int]
    WINDOW2_IDS_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    collection: str
    window1_ids: _containers.RepeatedScalarFieldContainer[int]
    window2_ids: _containers.RepeatedScalarFieldContainer[int]
    metric: str
    threshold: float
    def __init__(self, collection: _Optional[str] = ..., window1_ids: _Optional[_Iterable[int]] = ..., window2_ids: _Optional[_Iterable[int]] = ..., metric: _Optional[str] = ..., threshold: _Optional[float] = ...) -> None: ...

class DetectDriftResponse(_message.Message):
    __slots__ = ("centroid_shift", "mean_distance_window1", "mean_distance_window2", "spread_change", "has_drifted", "compute_time_us")
    CENTROID_SHIFT_FIELD_NUMBER: _ClassVar[int]
    MEAN_DISTANCE_WINDOW1_FIELD_NUMBER: _ClassVar[int]
    MEAN_DISTANCE_WINDOW2_FIELD_NUMBER: _ClassVar[int]
    SPREAD_CHANGE_FIELD_NUMBER: _ClassVar[int]
    HAS_DRIFTED_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    centroid_shift: float
    mean_distance_window1: float
    mean_distance_window2: float
    spread_change: float
    has_drifted: bool
    compute_time_us: int
    def __init__(self, centroid_shift: _Optional[float] = ..., mean_distance_window1: _Optional[float] = ..., mean_distance_window2: _Optional[float] = ..., spread_change: _Optional[float] = ..., has_drifted: bool = ..., compute_time_us: _Optional[int] = ...) -> None: ...

class ClusterRequest(_message.Message):
    __slots__ = ("collection", "k", "max_iterations", "tolerance", "metric")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    MAX_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    TOLERANCE_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    collection: str
    k: int
    max_iterations: int
    tolerance: float
    metric: str
    def __init__(self, collection: _Optional[str] = ..., k: _Optional[int] = ..., max_iterations: _Optional[int] = ..., tolerance: _Optional[float] = ..., metric: _Optional[str] = ...) -> None: ...

class ClusterAssignmentProto(_message.Message):
    __slots__ = ("id", "cluster", "distance_to_centroid")
    ID_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_TO_CENTROID_FIELD_NUMBER: _ClassVar[int]
    id: int
    cluster: int
    distance_to_centroid: float
    def __init__(self, id: _Optional[int] = ..., cluster: _Optional[int] = ..., distance_to_centroid: _Optional[float] = ...) -> None: ...

class ClusterResponse(_message.Message):
    __slots__ = ("centroids", "assignments", "iterations", "converged", "compute_time_us")
    CENTROIDS_FIELD_NUMBER: _ClassVar[int]
    ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    CONVERGED_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    centroids: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vector]
    assignments: _containers.RepeatedCompositeFieldContainer[ClusterAssignmentProto]
    iterations: int
    converged: bool
    compute_time_us: int
    def __init__(self, centroids: _Optional[_Iterable[_Union[_common_pb2.Vector, _Mapping]]] = ..., assignments: _Optional[_Iterable[_Union[ClusterAssignmentProto, _Mapping]]] = ..., iterations: _Optional[int] = ..., converged: bool = ..., compute_time_us: _Optional[int] = ...) -> None: ...

class ReduceDimensionsRequest(_message.Message):
    __slots__ = ("collection", "n_components", "vector_ids")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    N_COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    VECTOR_IDS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    n_components: int
    vector_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, collection: _Optional[str] = ..., n_components: _Optional[int] = ..., vector_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class ReduceDimensionsResponse(_message.Message):
    __slots__ = ("components", "explained_variance", "mean", "projected", "compute_time_us")
    COMPONENTS_FIELD_NUMBER: _ClassVar[int]
    EXPLAINED_VARIANCE_FIELD_NUMBER: _ClassVar[int]
    MEAN_FIELD_NUMBER: _ClassVar[int]
    PROJECTED_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    components: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vector]
    explained_variance: _containers.RepeatedScalarFieldContainer[float]
    mean: _common_pb2.Vector
    projected: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vector]
    compute_time_us: int
    def __init__(self, components: _Optional[_Iterable[_Union[_common_pb2.Vector, _Mapping]]] = ..., explained_variance: _Optional[_Iterable[float]] = ..., mean: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., projected: _Optional[_Iterable[_Union[_common_pb2.Vector, _Mapping]]] = ..., compute_time_us: _Optional[int] = ...) -> None: ...

class ComputeAnalogyRequest(_message.Message):
    __slots__ = ("a", "b", "c", "normalize", "terms")
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    C_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_FIELD_NUMBER: _ClassVar[int]
    TERMS_FIELD_NUMBER: _ClassVar[int]
    a: _common_pb2.Vector
    b: _common_pb2.Vector
    c: _common_pb2.Vector
    normalize: bool
    terms: _containers.RepeatedCompositeFieldContainer[ArithmeticTerm]
    def __init__(self, a: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., b: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., c: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., normalize: bool = ..., terms: _Optional[_Iterable[_Union[ArithmeticTerm, _Mapping]]] = ...) -> None: ...

class ArithmeticTerm(_message.Message):
    __slots__ = ("vector", "weight")
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    vector: _common_pb2.Vector
    weight: float
    def __init__(self, vector: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., weight: _Optional[float] = ...) -> None: ...

class ComputeAnalogyResponse(_message.Message):
    __slots__ = ("result", "compute_time_us")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    result: _common_pb2.Vector
    compute_time_us: int
    def __init__(self, result: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., compute_time_us: _Optional[int] = ...) -> None: ...

class DiversitySampleRequest(_message.Message):
    __slots__ = ("collection", "query", "k", "candidate_ids")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    LAMBDA_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_IDS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    query: _common_pb2.Vector
    k: int
    candidate_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, collection: _Optional[str] = ..., query: _Optional[_Union[_common_pb2.Vector, _Mapping]] = ..., k: _Optional[int] = ..., candidate_ids: _Optional[_Iterable[int]] = ..., **kwargs) -> None: ...

class DiversitySampleResult(_message.Message):
    __slots__ = ("id", "relevance_score", "mmr_score")
    ID_FIELD_NUMBER: _ClassVar[int]
    RELEVANCE_SCORE_FIELD_NUMBER: _ClassVar[int]
    MMR_SCORE_FIELD_NUMBER: _ClassVar[int]
    id: int
    relevance_score: float
    mmr_score: float
    def __init__(self, id: _Optional[int] = ..., relevance_score: _Optional[float] = ..., mmr_score: _Optional[float] = ...) -> None: ...

class DiversitySampleResponse(_message.Message):
    __slots__ = ("results", "compute_time_us")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_US_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[DiversitySampleResult]
    compute_time_us: int
    def __init__(self, results: _Optional[_Iterable[_Union[DiversitySampleResult, _Mapping]]] = ..., compute_time_us: _Optional[int] = ...) -> None: ...
