from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TypedNodeKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TYPED_NODE_CONTENT: _ClassVar[TypedNodeKind]
    TYPED_NODE_ENTITY: _ClassVar[TypedNodeKind]

class HybridDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HYBRID_DIR_OUTGOING: _ClassVar[HybridDirection]
    HYBRID_DIR_INCOMING: _ClassVar[HybridDirection]
    HYBRID_DIR_BOTH: _ClassVar[HybridDirection]

class HybridReturnKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HYBRID_RETURN_NODES: _ClassVar[HybridReturnKind]
    HYBRID_RETURN_EDGES: _ClassVar[HybridReturnKind]
    HYBRID_RETURN_PATHS: _ClassVar[HybridReturnKind]

class HybridCompareOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HYBRID_CMP_EQ: _ClassVar[HybridCompareOp]
    HYBRID_CMP_NE: _ClassVar[HybridCompareOp]
    HYBRID_CMP_LT: _ClassVar[HybridCompareOp]
    HYBRID_CMP_LE: _ClassVar[HybridCompareOp]
    HYBRID_CMP_GT: _ClassVar[HybridCompareOp]
    HYBRID_CMP_GE: _ClassVar[HybridCompareOp]

class HybridOnMissingVector(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HYBRID_ON_MISSING_SKIP: _ClassVar[HybridOnMissingVector]
    HYBRID_ON_MISSING_ERROR: _ClassVar[HybridOnMissingVector]

class BulkImportFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BULK_IMPORT_FORMAT_CSV: _ClassVar[BulkImportFormat]
    BULK_IMPORT_FORMAT_JSONL: _ClassVar[BulkImportFormat]
TYPED_NODE_CONTENT: TypedNodeKind
TYPED_NODE_ENTITY: TypedNodeKind
HYBRID_DIR_OUTGOING: HybridDirection
HYBRID_DIR_INCOMING: HybridDirection
HYBRID_DIR_BOTH: HybridDirection
HYBRID_RETURN_NODES: HybridReturnKind
HYBRID_RETURN_EDGES: HybridReturnKind
HYBRID_RETURN_PATHS: HybridReturnKind
HYBRID_CMP_EQ: HybridCompareOp
HYBRID_CMP_NE: HybridCompareOp
HYBRID_CMP_LT: HybridCompareOp
HYBRID_CMP_LE: HybridCompareOp
HYBRID_CMP_GT: HybridCompareOp
HYBRID_CMP_GE: HybridCompareOp
HYBRID_ON_MISSING_SKIP: HybridOnMissingVector
HYBRID_ON_MISSING_ERROR: HybridOnMissingVector
BULK_IMPORT_FORMAT_CSV: BulkImportFormat
BULK_IMPORT_FORMAT_JSONL: BulkImportFormat

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

class NodeAudit(_message.Message):
    __slots__ = ("action", "actor", "at")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    AT_FIELD_NUMBER: _ClassVar[int]
    action: str
    actor: str
    at: int
    def __init__(self, action: _Optional[str] = ..., actor: _Optional[str] = ..., at: _Optional[int] = ...) -> None: ...

class TypedNode(_message.Message):
    __slots__ = ("id", "kind", "label", "properties_json", "embedding", "source", "created_at", "created_by", "history", "updated_at")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_JSON_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    id: int
    kind: TypedNodeKind
    label: str
    properties_json: str
    embedding: _containers.RepeatedScalarFieldContainer[float]
    source: str
    created_at: int
    created_by: str
    history: _containers.RepeatedCompositeFieldContainer[NodeAudit]
    updated_at: int
    def __init__(self, id: _Optional[int] = ..., kind: _Optional[_Union[TypedNodeKind, str]] = ..., label: _Optional[str] = ..., properties_json: _Optional[str] = ..., embedding: _Optional[_Iterable[float]] = ..., source: _Optional[str] = ..., created_at: _Optional[int] = ..., created_by: _Optional[str] = ..., history: _Optional[_Iterable[_Union[NodeAudit, _Mapping]]] = ..., updated_at: _Optional[int] = ...) -> None: ...

class EdgeAudit(_message.Message):
    __slots__ = ("action", "actor", "at")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    AT_FIELD_NUMBER: _ClassVar[int]
    action: str
    actor: str
    at: int
    def __init__(self, action: _Optional[str] = ..., actor: _Optional[str] = ..., at: _Optional[int] = ...) -> None: ...

class TypedEdge(_message.Message):
    __slots__ = ("id", "source", "target", "edge_type", "properties_json", "provenance_json", "confidence", "verified", "is_manual", "created_at", "history", "valid_from", "valid_until", "temporal_context")
    ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_JSON_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_JSON_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    VERIFIED_FIELD_NUMBER: _ClassVar[int]
    IS_MANUAL_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    VALID_FROM_FIELD_NUMBER: _ClassVar[int]
    VALID_UNTIL_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    id: int
    source: int
    target: int
    edge_type: str
    properties_json: str
    provenance_json: str
    confidence: float
    verified: bool
    is_manual: bool
    created_at: int
    history: _containers.RepeatedCompositeFieldContainer[EdgeAudit]
    valid_from: int
    valid_until: int
    temporal_context: str
    def __init__(self, id: _Optional[int] = ..., source: _Optional[int] = ..., target: _Optional[int] = ..., edge_type: _Optional[str] = ..., properties_json: _Optional[str] = ..., provenance_json: _Optional[str] = ..., confidence: _Optional[float] = ..., verified: bool = ..., is_manual: bool = ..., created_at: _Optional[int] = ..., history: _Optional[_Iterable[_Union[EdgeAudit, _Mapping]]] = ..., valid_from: _Optional[int] = ..., valid_until: _Optional[int] = ..., temporal_context: _Optional[str] = ...) -> None: ...

class PutNodeRequest(_message.Message):
    __slots__ = ("collection", "kind", "label", "properties_json", "embedding", "source", "created_by")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_JSON_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CREATED_BY_FIELD_NUMBER: _ClassVar[int]
    collection: str
    kind: TypedNodeKind
    label: str
    properties_json: str
    embedding: _containers.RepeatedScalarFieldContainer[float]
    source: str
    created_by: str
    def __init__(self, collection: _Optional[str] = ..., kind: _Optional[_Union[TypedNodeKind, str]] = ..., label: _Optional[str] = ..., properties_json: _Optional[str] = ..., embedding: _Optional[_Iterable[float]] = ..., source: _Optional[str] = ..., created_by: _Optional[str] = ...) -> None: ...

class PutNodeResponse(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class GetNodeRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: int
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[int] = ...) -> None: ...

class GetNodeResponse(_message.Message):
    __slots__ = ("found", "node")
    FOUND_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    found: bool
    node: TypedNode
    def __init__(self, found: bool = ..., node: _Optional[_Union[TypedNode, _Mapping]] = ...) -> None: ...

class DeleteNodeRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: int
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[int] = ...) -> None: ...

class DeleteNodeResponse(_message.Message):
    __slots__ = ("deleted",)
    DELETED_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    def __init__(self, deleted: bool = ...) -> None: ...

class PutEdgeRequest(_message.Message):
    __slots__ = ("collection", "source", "target", "edge_type", "properties_json", "provenance_json", "confidence", "verified", "is_manual", "valid_from", "valid_until", "temporal_context")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_JSON_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_JSON_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    VERIFIED_FIELD_NUMBER: _ClassVar[int]
    IS_MANUAL_FIELD_NUMBER: _ClassVar[int]
    VALID_FROM_FIELD_NUMBER: _ClassVar[int]
    VALID_UNTIL_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    collection: str
    source: int
    target: int
    edge_type: str
    properties_json: str
    provenance_json: str
    confidence: float
    verified: bool
    is_manual: bool
    valid_from: int
    valid_until: int
    temporal_context: str
    def __init__(self, collection: _Optional[str] = ..., source: _Optional[int] = ..., target: _Optional[int] = ..., edge_type: _Optional[str] = ..., properties_json: _Optional[str] = ..., provenance_json: _Optional[str] = ..., confidence: _Optional[float] = ..., verified: bool = ..., is_manual: bool = ..., valid_from: _Optional[int] = ..., valid_until: _Optional[int] = ..., temporal_context: _Optional[str] = ...) -> None: ...

class PutEdgeResponse(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class GetEdgeRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: int
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[int] = ...) -> None: ...

class GetEdgeResponse(_message.Message):
    __slots__ = ("found", "edge")
    FOUND_FIELD_NUMBER: _ClassVar[int]
    EDGE_FIELD_NUMBER: _ClassVar[int]
    found: bool
    edge: TypedEdge
    def __init__(self, found: bool = ..., edge: _Optional[_Union[TypedEdge, _Mapping]] = ...) -> None: ...

class DeleteEdgeRequest(_message.Message):
    __slots__ = ("collection", "id")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    collection: str
    id: int
    def __init__(self, collection: _Optional[str] = ..., id: _Optional[int] = ...) -> None: ...

class DeleteEdgeResponse(_message.Message):
    __slots__ = ("deleted",)
    DELETED_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    def __init__(self, deleted: bool = ...) -> None: ...

class ListEdgesRequest(_message.Message):
    __slots__ = ("collection", "node", "direction", "edge_type")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    collection: str
    node: int
    direction: str
    edge_type: str
    def __init__(self, collection: _Optional[str] = ..., node: _Optional[int] = ..., direction: _Optional[str] = ..., edge_type: _Optional[str] = ...) -> None: ...

class ListEdgesResponse(_message.Message):
    __slots__ = ("edges",)
    EDGES_FIELD_NUMBER: _ClassVar[int]
    edges: _containers.RepeatedCompositeFieldContainer[TypedEdge]
    def __init__(self, edges: _Optional[_Iterable[_Union[TypedEdge, _Mapping]]] = ...) -> None: ...

class HybridIncidentEdgeCount(_message.Message):
    __slots__ = ("edge_type", "direction")
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    edge_type: str
    direction: HybridDirection
    def __init__(self, edge_type: _Optional[str] = ..., direction: _Optional[_Union[HybridDirection, str]] = ...) -> None: ...

class HybridPropertyRef(_message.Message):
    __slots__ = ("property", "label", "kind", "incident_edge_count")
    PROPERTY_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    INCIDENT_EDGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    property: str
    label: bool
    kind: bool
    incident_edge_count: HybridIncidentEdgeCount
    def __init__(self, property: _Optional[str] = ..., label: bool = ..., kind: bool = ..., incident_edge_count: _Optional[_Union[HybridIncidentEdgeCount, _Mapping]] = ...) -> None: ...

class HybridCompare(_message.Message):
    __slots__ = ("field", "op", "value_json")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    VALUE_JSON_FIELD_NUMBER: _ClassVar[int]
    field: HybridPropertyRef
    op: HybridCompareOp
    value_json: str
    def __init__(self, field: _Optional[_Union[HybridPropertyRef, _Mapping]] = ..., op: _Optional[_Union[HybridCompareOp, str]] = ..., value_json: _Optional[str] = ...) -> None: ...

class HybridInList(_message.Message):
    __slots__ = ("field", "values_json")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUES_JSON_FIELD_NUMBER: _ClassVar[int]
    field: HybridPropertyRef
    values_json: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, field: _Optional[_Union[HybridPropertyRef, _Mapping]] = ..., values_json: _Optional[_Iterable[str]] = ...) -> None: ...

class HybridPredicateList(_message.Message):
    __slots__ = ("preds",)
    PREDS_FIELD_NUMBER: _ClassVar[int]
    preds: _containers.RepeatedCompositeFieldContainer[HybridPredicate]
    def __init__(self, preds: _Optional[_Iterable[_Union[HybridPredicate, _Mapping]]] = ...) -> None: ...

class HybridPredicate(_message.Message):
    __slots__ = ("compare", "in_list", "not_in_list", "exists", "always")
    COMPARE_FIELD_NUMBER: _ClassVar[int]
    IN_LIST_FIELD_NUMBER: _ClassVar[int]
    NOT_IN_LIST_FIELD_NUMBER: _ClassVar[int]
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    AND_FIELD_NUMBER: _ClassVar[int]
    OR_FIELD_NUMBER: _ClassVar[int]
    NOT_FIELD_NUMBER: _ClassVar[int]
    ALWAYS_FIELD_NUMBER: _ClassVar[int]
    compare: HybridCompare
    in_list: HybridInList
    not_in_list: HybridInList
    exists: HybridPropertyRef
    always: bool
    def __init__(self, compare: _Optional[_Union[HybridCompare, _Mapping]] = ..., in_list: _Optional[_Union[HybridInList, _Mapping]] = ..., not_in_list: _Optional[_Union[HybridInList, _Mapping]] = ..., exists: _Optional[_Union[HybridPropertyRef, _Mapping]] = ..., always: bool = ..., **kwargs) -> None: ...

class HybridVectorSimilar(_message.Message):
    __slots__ = ("vector", "k", "ef_search")
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    EF_SEARCH_FIELD_NUMBER: _ClassVar[int]
    vector: _containers.RepeatedScalarFieldContainer[float]
    k: int
    ef_search: int
    def __init__(self, vector: _Optional[_Iterable[float]] = ..., k: _Optional[int] = ..., ef_search: _Optional[int] = ...) -> None: ...

class HybridVectorRank(_message.Message):
    __slots__ = ("vector", "k", "on_missing", "predicate")
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    ON_MISSING_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    vector: _containers.RepeatedScalarFieldContainer[float]
    k: int
    on_missing: HybridOnMissingVector
    predicate: HybridPredicate
    def __init__(self, vector: _Optional[_Iterable[float]] = ..., k: _Optional[int] = ..., on_missing: _Optional[_Union[HybridOnMissingVector, str]] = ..., predicate: _Optional[_Union[HybridPredicate, _Mapping]] = ...) -> None: ...

class HybridVector(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class HybridAnalogy(_message.Message):
    __slots__ = ("a", "b", "c")
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    C_FIELD_NUMBER: _ClassVar[int]
    a: _containers.RepeatedScalarFieldContainer[float]
    b: _containers.RepeatedScalarFieldContainer[float]
    c: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, a: _Optional[_Iterable[float]] = ..., b: _Optional[_Iterable[float]] = ..., c: _Optional[_Iterable[float]] = ...) -> None: ...

class HybridDiversity(_message.Message):
    __slots__ = ("query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    LAMBDA_FIELD_NUMBER: _ClassVar[int]
    query: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, query: _Optional[_Iterable[float]] = ..., **kwargs) -> None: ...

class HybridCone(_message.Message):
    __slots__ = ("direction", "aperture_radians")
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    APERTURE_RADIANS_FIELD_NUMBER: _ClassVar[int]
    direction: _containers.RepeatedScalarFieldContainer[float]
    aperture_radians: float
    def __init__(self, direction: _Optional[_Iterable[float]] = ..., aperture_radians: _Optional[float] = ...) -> None: ...

class HybridIsolation(_message.Message):
    __slots__ = ("centroids",)
    CENTROIDS_FIELD_NUMBER: _ClassVar[int]
    centroids: _containers.RepeatedCompositeFieldContainer[HybridVector]
    def __init__(self, centroids: _Optional[_Iterable[_Union[HybridVector, _Mapping]]] = ...) -> None: ...

class HybridCentroid(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HybridInterpolate(_message.Message):
    __slots__ = ("a", "b", "t")
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    T_FIELD_NUMBER: _ClassVar[int]
    a: _containers.RepeatedScalarFieldContainer[float]
    b: _containers.RepeatedScalarFieldContainer[float]
    t: float
    def __init__(self, a: _Optional[_Iterable[float]] = ..., b: _Optional[_Iterable[float]] = ..., t: _Optional[float] = ...) -> None: ...

class HybridVectorMath(_message.Message):
    __slots__ = ("analogy", "diversity", "cone", "isolation", "centroid", "interpolate", "k", "on_missing")
    ANALOGY_FIELD_NUMBER: _ClassVar[int]
    DIVERSITY_FIELD_NUMBER: _ClassVar[int]
    CONE_FIELD_NUMBER: _ClassVar[int]
    ISOLATION_FIELD_NUMBER: _ClassVar[int]
    CENTROID_FIELD_NUMBER: _ClassVar[int]
    INTERPOLATE_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    ON_MISSING_FIELD_NUMBER: _ClassVar[int]
    analogy: HybridAnalogy
    diversity: HybridDiversity
    cone: HybridCone
    isolation: HybridIsolation
    centroid: HybridCentroid
    interpolate: HybridInterpolate
    k: int
    on_missing: HybridOnMissingVector
    def __init__(self, analogy: _Optional[_Union[HybridAnalogy, _Mapping]] = ..., diversity: _Optional[_Union[HybridDiversity, _Mapping]] = ..., cone: _Optional[_Union[HybridCone, _Mapping]] = ..., isolation: _Optional[_Union[HybridIsolation, _Mapping]] = ..., centroid: _Optional[_Union[HybridCentroid, _Mapping]] = ..., interpolate: _Optional[_Union[HybridInterpolate, _Mapping]] = ..., k: _Optional[int] = ..., on_missing: _Optional[_Union[HybridOnMissingVector, str]] = ...) -> None: ...

class HybridFromNodes(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, nodes: _Optional[_Iterable[int]] = ...) -> None: ...

class HybridScanByFilter(_message.Message):
    __slots__ = ("filter_by_kind", "kind", "label", "predicate")
    FILTER_BY_KIND_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    filter_by_kind: bool
    kind: TypedNodeKind
    label: str
    predicate: HybridPredicate
    def __init__(self, filter_by_kind: bool = ..., kind: _Optional[_Union[TypedNodeKind, str]] = ..., label: _Optional[str] = ..., predicate: _Optional[_Union[HybridPredicate, _Mapping]] = ...) -> None: ...

class HybridTemporalFilter(_message.Message):
    __slots__ = ("as_of", "include_unbounded", "context")
    AS_OF_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_UNBOUNDED_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    as_of: int
    include_unbounded: bool
    context: str
    def __init__(self, as_of: _Optional[int] = ..., include_unbounded: bool = ..., context: _Optional[str] = ...) -> None: ...

class HybridTraverse(_message.Message):
    __slots__ = ("edge_type", "direction", "temporal")
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_FIELD_NUMBER: _ClassVar[int]
    edge_type: str
    direction: HybridDirection
    temporal: HybridTemporalFilter
    def __init__(self, edge_type: _Optional[str] = ..., direction: _Optional[_Union[HybridDirection, str]] = ..., temporal: _Optional[_Union[HybridTemporalFilter, _Mapping]] = ...) -> None: ...

class HybridWeightSpec(_message.Message):
    __slots__ = ("use_confidence", "min_confidence", "recency_half_life_ms", "use_explicit_weight", "explicit_weight_key")
    USE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    MIN_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    RECENCY_HALF_LIFE_MS_FIELD_NUMBER: _ClassVar[int]
    USE_EXPLICIT_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    EXPLICIT_WEIGHT_KEY_FIELD_NUMBER: _ClassVar[int]
    use_confidence: bool
    min_confidence: float
    recency_half_life_ms: int
    use_explicit_weight: bool
    explicit_weight_key: str
    def __init__(self, use_confidence: bool = ..., min_confidence: _Optional[float] = ..., recency_half_life_ms: _Optional[int] = ..., use_explicit_weight: bool = ..., explicit_weight_key: _Optional[str] = ...) -> None: ...

class HybridKHop(_message.Message):
    __slots__ = ("edge_type", "max", "predicate", "weight", "order_by_weight", "temporal")
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_FIELD_NUMBER: _ClassVar[int]
    edge_type: str
    max: int
    predicate: HybridPredicate
    weight: HybridWeightSpec
    order_by_weight: bool
    temporal: HybridTemporalFilter
    def __init__(self, edge_type: _Optional[str] = ..., max: _Optional[int] = ..., predicate: _Optional[_Union[HybridPredicate, _Mapping]] = ..., weight: _Optional[_Union[HybridWeightSpec, _Mapping]] = ..., order_by_weight: bool = ..., temporal: _Optional[_Union[HybridTemporalFilter, _Mapping]] = ...) -> None: ...

class HybridShortestPath(_message.Message):
    __slots__ = ("edge_types", "target", "weighted", "weight", "temporal")
    EDGE_TYPES_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    WEIGHTED_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_FIELD_NUMBER: _ClassVar[int]
    edge_types: _containers.RepeatedScalarFieldContainer[str]
    target: int
    weighted: bool
    weight: HybridWeightSpec
    temporal: HybridTemporalFilter
    def __init__(self, edge_types: _Optional[_Iterable[str]] = ..., target: _Optional[int] = ..., weighted: bool = ..., weight: _Optional[_Union[HybridWeightSpec, _Mapping]] = ..., temporal: _Optional[_Union[HybridTemporalFilter, _Mapping]] = ...) -> None: ...

class HybridCollectEdges(_message.Message):
    __slots__ = ("edge_type", "direction")
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    edge_type: str
    direction: HybridDirection
    def __init__(self, edge_type: _Optional[str] = ..., direction: _Optional[_Union[HybridDirection, str]] = ...) -> None: ...

class HybridStep(_message.Message):
    __slots__ = ("vector_similar", "from_nodes", "traverse", "k_hop", "shortest_path", "mutual_neighbors", "intersect", "union", "filter", "collect_edges", "limit", "vector_rank", "scan_by_filter", "vector_math")
    VECTOR_SIMILAR_FIELD_NUMBER: _ClassVar[int]
    FROM_NODES_FIELD_NUMBER: _ClassVar[int]
    TRAVERSE_FIELD_NUMBER: _ClassVar[int]
    K_HOP_FIELD_NUMBER: _ClassVar[int]
    SHORTEST_PATH_FIELD_NUMBER: _ClassVar[int]
    MUTUAL_NEIGHBORS_FIELD_NUMBER: _ClassVar[int]
    INTERSECT_FIELD_NUMBER: _ClassVar[int]
    UNION_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    COLLECT_EDGES_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    VECTOR_RANK_FIELD_NUMBER: _ClassVar[int]
    SCAN_BY_FILTER_FIELD_NUMBER: _ClassVar[int]
    VECTOR_MATH_FIELD_NUMBER: _ClassVar[int]
    vector_similar: HybridVectorSimilar
    from_nodes: HybridFromNodes
    traverse: HybridTraverse
    k_hop: HybridKHop
    shortest_path: HybridShortestPath
    mutual_neighbors: HybridQueryPlan
    intersect: HybridQueryPlan
    union: HybridQueryPlan
    filter: HybridPredicate
    collect_edges: HybridCollectEdges
    limit: int
    vector_rank: HybridVectorRank
    scan_by_filter: HybridScanByFilter
    vector_math: HybridVectorMath
    def __init__(self, vector_similar: _Optional[_Union[HybridVectorSimilar, _Mapping]] = ..., from_nodes: _Optional[_Union[HybridFromNodes, _Mapping]] = ..., traverse: _Optional[_Union[HybridTraverse, _Mapping]] = ..., k_hop: _Optional[_Union[HybridKHop, _Mapping]] = ..., shortest_path: _Optional[_Union[HybridShortestPath, _Mapping]] = ..., mutual_neighbors: _Optional[_Union[HybridQueryPlan, _Mapping]] = ..., intersect: _Optional[_Union[HybridQueryPlan, _Mapping]] = ..., union: _Optional[_Union[HybridQueryPlan, _Mapping]] = ..., filter: _Optional[_Union[HybridPredicate, _Mapping]] = ..., collect_edges: _Optional[_Union[HybridCollectEdges, _Mapping]] = ..., limit: _Optional[int] = ..., vector_rank: _Optional[_Union[HybridVectorRank, _Mapping]] = ..., scan_by_filter: _Optional[_Union[HybridScanByFilter, _Mapping]] = ..., vector_math: _Optional[_Union[HybridVectorMath, _Mapping]] = ...) -> None: ...

class HybridQueryPlan(_message.Message):
    __slots__ = ("steps", "return_kind")
    STEPS_FIELD_NUMBER: _ClassVar[int]
    RETURN_KIND_FIELD_NUMBER: _ClassVar[int]
    steps: _containers.RepeatedCompositeFieldContainer[HybridStep]
    return_kind: HybridReturnKind
    def __init__(self, steps: _Optional[_Iterable[_Union[HybridStep, _Mapping]]] = ..., return_kind: _Optional[_Union[HybridReturnKind, str]] = ...) -> None: ...

class HybridPath(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, nodes: _Optional[_Iterable[int]] = ...) -> None: ...

class RrfRankSpec(_message.Message):
    __slots__ = ("k", "rrf_k", "k_hop_max", "relation_edge_types", "hub_damping", "edge_weight")
    K_FIELD_NUMBER: _ClassVar[int]
    RRF_K_FIELD_NUMBER: _ClassVar[int]
    K_HOP_MAX_FIELD_NUMBER: _ClassVar[int]
    RELATION_EDGE_TYPES_FIELD_NUMBER: _ClassVar[int]
    HUB_DAMPING_FIELD_NUMBER: _ClassVar[int]
    EDGE_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    k: int
    rrf_k: int
    k_hop_max: int
    relation_edge_types: _containers.RepeatedScalarFieldContainer[str]
    hub_damping: float
    edge_weight: HybridWeightSpec
    def __init__(self, k: _Optional[int] = ..., rrf_k: _Optional[int] = ..., k_hop_max: _Optional[int] = ..., relation_edge_types: _Optional[_Iterable[str]] = ..., hub_damping: _Optional[float] = ..., edge_weight: _Optional[_Union[HybridWeightSpec, _Mapping]] = ...) -> None: ...

class HybridQueryRequest(_message.Message):
    __slots__ = ("collection", "plan", "rrf_rank")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    PLAN_FIELD_NUMBER: _ClassVar[int]
    RRF_RANK_FIELD_NUMBER: _ClassVar[int]
    collection: str
    plan: HybridQueryPlan
    rrf_rank: RrfRankSpec
    def __init__(self, collection: _Optional[str] = ..., plan: _Optional[_Union[HybridQueryPlan, _Mapping]] = ..., rrf_rank: _Optional[_Union[RrfRankSpec, _Mapping]] = ...) -> None: ...

class HybridQueryResponse(_message.Message):
    __slots__ = ("nodes", "edges", "paths")
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    PATHS_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[TypedNode]
    edges: _containers.RepeatedCompositeFieldContainer[TypedEdge]
    paths: _containers.RepeatedCompositeFieldContainer[HybridPath]
    def __init__(self, nodes: _Optional[_Iterable[_Union[TypedNode, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[TypedEdge, _Mapping]]] = ..., paths: _Optional[_Iterable[_Union[HybridPath, _Mapping]]] = ...) -> None: ...

class UpdateNodeRequest(_message.Message):
    __slots__ = ("collection", "node_id", "properties_json", "actor")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_JSON_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    collection: str
    node_id: int
    properties_json: str
    actor: str
    def __init__(self, collection: _Optional[str] = ..., node_id: _Optional[int] = ..., properties_json: _Optional[str] = ..., actor: _Optional[str] = ...) -> None: ...

class UpdateNodeResponse(_message.Message):
    __slots__ = ("node",)
    NODE_FIELD_NUMBER: _ClassVar[int]
    node: TypedNode
    def __init__(self, node: _Optional[_Union[TypedNode, _Mapping]] = ...) -> None: ...

class UpdateEdgeRequest(_message.Message):
    __slots__ = ("collection", "edge_id", "properties_json", "confidence", "verified", "actor")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_JSON_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    VERIFIED_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    collection: str
    edge_id: int
    properties_json: str
    confidence: float
    verified: bool
    actor: str
    def __init__(self, collection: _Optional[str] = ..., edge_id: _Optional[int] = ..., properties_json: _Optional[str] = ..., confidence: _Optional[float] = ..., verified: bool = ..., actor: _Optional[str] = ...) -> None: ...

class UpdateEdgeResponse(_message.Message):
    __slots__ = ("edge",)
    EDGE_FIELD_NUMBER: _ClassVar[int]
    edge: TypedEdge
    def __init__(self, edge: _Optional[_Union[TypedEdge, _Mapping]] = ...) -> None: ...

class VerifyEdgeRequest(_message.Message):
    __slots__ = ("collection", "edge_id", "actor")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    collection: str
    edge_id: int
    actor: str
    def __init__(self, collection: _Optional[str] = ..., edge_id: _Optional[int] = ..., actor: _Optional[str] = ...) -> None: ...

class VerifyEdgeResponse(_message.Message):
    __slots__ = ("edge",)
    EDGE_FIELD_NUMBER: _ClassVar[int]
    edge: TypedEdge
    def __init__(self, edge: _Optional[_Union[TypedEdge, _Mapping]] = ...) -> None: ...

class RejectEdgeRequest(_message.Message):
    __slots__ = ("collection", "edge_id", "actor")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    collection: str
    edge_id: int
    actor: str
    def __init__(self, collection: _Optional[str] = ..., edge_id: _Optional[int] = ..., actor: _Optional[str] = ...) -> None: ...

class RejectEdgeResponse(_message.Message):
    __slots__ = ("deleted", "rule_added")
    DELETED_FIELD_NUMBER: _ClassVar[int]
    RULE_ADDED_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    rule_added: bool
    def __init__(self, deleted: bool = ..., rule_added: bool = ...) -> None: ...

class BulkImportEdgesRequest(_message.Message):
    __slots__ = ("collection", "format", "data", "auto_add_edge_types", "actor")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    AUTO_ADD_EDGE_TYPES_FIELD_NUMBER: _ClassVar[int]
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    collection: str
    format: BulkImportFormat
    data: str
    auto_add_edge_types: bool
    actor: str
    def __init__(self, collection: _Optional[str] = ..., format: _Optional[_Union[BulkImportFormat, str]] = ..., data: _Optional[str] = ..., auto_add_edge_types: bool = ..., actor: _Optional[str] = ...) -> None: ...

class BulkImportRowError(_message.Message):
    __slots__ = ("row", "message")
    ROW_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    row: int
    message: str
    def __init__(self, row: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class BulkImportEdgesResponse(_message.Message):
    __slots__ = ("total_rows", "imported", "failed", "errors")
    TOTAL_ROWS_FIELD_NUMBER: _ClassVar[int]
    IMPORTED_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    total_rows: int
    imported: int
    failed: int
    errors: _containers.RepeatedCompositeFieldContainer[BulkImportRowError]
    def __init__(self, total_rows: _Optional[int] = ..., imported: _Optional[int] = ..., failed: _Optional[int] = ..., errors: _Optional[_Iterable[_Union[BulkImportRowError, _Mapping]]] = ...) -> None: ...

class EnumerateNodesRequest(_message.Message):
    __slots__ = ("collection", "after_id", "limit", "kind", "filter_by_kind", "label", "predicate")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    AFTER_ID_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    FILTER_BY_KIND_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    collection: str
    after_id: int
    limit: int
    kind: TypedNodeKind
    filter_by_kind: bool
    label: str
    predicate: HybridPredicate
    def __init__(self, collection: _Optional[str] = ..., after_id: _Optional[int] = ..., limit: _Optional[int] = ..., kind: _Optional[_Union[TypedNodeKind, str]] = ..., filter_by_kind: bool = ..., label: _Optional[str] = ..., predicate: _Optional[_Union[HybridPredicate, _Mapping]] = ...) -> None: ...

class EnumerateNodesResponse(_message.Message):
    __slots__ = ("nodes", "next_cursor", "has_more")
    NODES_FIELD_NUMBER: _ClassVar[int]
    NEXT_CURSOR_FIELD_NUMBER: _ClassVar[int]
    HAS_MORE_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[TypedNode]
    next_cursor: int
    has_more: bool
    def __init__(self, nodes: _Optional[_Iterable[_Union[TypedNode, _Mapping]]] = ..., next_cursor: _Optional[int] = ..., has_more: bool = ...) -> None: ...

class EnumerateEdgesRequest(_message.Message):
    __slots__ = ("collection", "after_id", "limit", "edge_type", "predicate", "endpoint_label", "endpoint_kind", "filter_by_endpoint_kind")
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    AFTER_ID_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    EDGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_LABEL_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_KIND_FIELD_NUMBER: _ClassVar[int]
    FILTER_BY_ENDPOINT_KIND_FIELD_NUMBER: _ClassVar[int]
    collection: str
    after_id: int
    limit: int
    edge_type: str
    predicate: HybridPredicate
    endpoint_label: str
    endpoint_kind: TypedNodeKind
    filter_by_endpoint_kind: bool
    def __init__(self, collection: _Optional[str] = ..., after_id: _Optional[int] = ..., limit: _Optional[int] = ..., edge_type: _Optional[str] = ..., predicate: _Optional[_Union[HybridPredicate, _Mapping]] = ..., endpoint_label: _Optional[str] = ..., endpoint_kind: _Optional[_Union[TypedNodeKind, str]] = ..., filter_by_endpoint_kind: bool = ...) -> None: ...

class EnumerateEdgesResponse(_message.Message):
    __slots__ = ("edges", "next_cursor", "has_more")
    EDGES_FIELD_NUMBER: _ClassVar[int]
    NEXT_CURSOR_FIELD_NUMBER: _ClassVar[int]
    HAS_MORE_FIELD_NUMBER: _ClassVar[int]
    edges: _containers.RepeatedCompositeFieldContainer[TypedEdge]
    next_cursor: int
    has_more: bool
    def __init__(self, edges: _Optional[_Iterable[_Union[TypedEdge, _Mapping]]] = ..., next_cursor: _Optional[int] = ..., has_more: bool = ...) -> None: ...
