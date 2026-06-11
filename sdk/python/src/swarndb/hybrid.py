"""SwarnDB hybrid graph query builder.

A composable, chainable builder that mirrors the Rust hybrid query API.
Start from ``client.graph.query("collection")`` and chain steps such as
``vector_similar``, ``from_node``, ``traverse``, ``k_hop``, ``filter``, and
``limit``; finish with a terminal ``return_nodes()`` / ``return_edges()`` /
``return_paths()`` that executes the query and returns a HybridQueryResult.

Sub-queries (for ``intersect`` / ``union`` / ``mutual_neighbors``) are built
with ``to_plan()`` (alias ``build_plan()``), which returns a proto plan without
executing it, so it can be embedded into an outer query.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

from ._proto import graph_pb2
from .types import HybridQueryResult

if TYPE_CHECKING:
    from .client import SwarnDBClient


@dataclass
class WeightSpec:
    """Edge quality weighting for quality-aware traversal and ranking (P13).

    All defaults are OFF, so omitting a WeightSpec (or passing one built with no
    args) keeps every edge at weight 1.0 and leaves behavior unchanged. Pass one
    to fold edge confidence, recency decay, or an explicit numeric property into
    the per-edge weight used by ``k_hop``, ``shortest_path``, and RRF ranking.
    """

    use_confidence: bool = False
    min_confidence: float = 0.0
    recency_half_life_ms: int = 0
    use_explicit_weight: bool = False
    explicit_weight_key: str = "weight"

    def _to_proto(self) -> "graph_pb2.HybridWeightSpec":
        """Build the proto weight spec mirroring these fields."""
        return graph_pb2.HybridWeightSpec(
            use_confidence=self.use_confidence,
            min_confidence=self.min_confidence,
            recency_half_life_ms=self.recency_half_life_ms,
            use_explicit_weight=self.use_explicit_weight,
            explicit_weight_key=self.explicit_weight_key,
        )


# Map direction strings to proto enum values; mirrors the list_edges convention.
_DIRECTION_TO_PROTO = {
    "outgoing": graph_pb2.HYBRID_DIR_OUTGOING,
    "incoming": graph_pb2.HYBRID_DIR_INCOMING,
    "both": graph_pb2.HYBRID_DIR_BOTH,
}


def _direction_to_proto(direction: str):
    """Map a direction string to its proto enum value."""
    try:
        return _DIRECTION_TO_PROTO[direction]
    except KeyError:
        raise ValueError(
            f"direction must be 'outgoing', 'incoming', or 'both', "
            f"got {direction!r}"
        ) from None


def _temporal_filter(
    as_of: Optional[int],
    include_unbounded: bool,
    context: Optional[str],
) -> Optional["graph_pb2.HybridTemporalFilter"]:
    """Build a HybridTemporalFilter, or None when the caller passed pure defaults.

    Returning None keeps the wire empty (no time/context filtering, byte-identical
    to today). When a filter IS built, ``include_unbounded`` is always set
    explicitly because the proto bool defaults to false on the wire (P17 / D.1.c).
    """
    if as_of is None and include_unbounded is True and context is None:
        return None
    tf = graph_pb2.HybridTemporalFilter(include_unbounded=include_unbounded)
    if as_of is not None:
        tf.as_of = as_of
    if context is not None:
        tf.context = context
    return tf


class Direction:
    """Direction string constants for traversal and edge collection."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


def _field_ref(key: str) -> graph_pb2.HybridPropertyRef:
    """Build a property-key field reference."""
    return graph_pb2.HybridPropertyRef(property=key)


def _cmp(key: str, op, value: Any) -> graph_pb2.HybridPredicate:
    """Build a single compare predicate; the value travels as JSON."""
    return graph_pb2.HybridPredicate(
        compare=graph_pb2.HybridCompare(
            field=_field_ref(key),
            op=op,
            value_json=json.dumps(value),
        )
    )


class Predicate:
    """Constructors for hybrid filter predicates.

    Scalars are wire-encoded as JSON literals (matching the server). A bare
    property key maps to a property-bag reference; ``label_eq`` references the
    node label flag instead.
    """

    @staticmethod
    def eq(key: str, value: Any) -> graph_pb2.HybridPredicate:
        """field == value."""
        return _cmp(key, graph_pb2.HYBRID_CMP_EQ, value)

    @staticmethod
    def ne(key: str, value: Any) -> graph_pb2.HybridPredicate:
        """field != value."""
        return _cmp(key, graph_pb2.HYBRID_CMP_NE, value)

    @staticmethod
    def gt(key: str, value: Any) -> graph_pb2.HybridPredicate:
        """field > value."""
        return _cmp(key, graph_pb2.HYBRID_CMP_GT, value)

    @staticmethod
    def ge(key: str, value: Any) -> graph_pb2.HybridPredicate:
        """field >= value."""
        return _cmp(key, graph_pb2.HYBRID_CMP_GE, value)

    @staticmethod
    def lt(key: str, value: Any) -> graph_pb2.HybridPredicate:
        """field < value."""
        return _cmp(key, graph_pb2.HYBRID_CMP_LT, value)

    @staticmethod
    def le(key: str, value: Any) -> graph_pb2.HybridPredicate:
        """field <= value."""
        return _cmp(key, graph_pb2.HYBRID_CMP_LE, value)

    @staticmethod
    def is_in(key: str, values: Sequence[Any]) -> graph_pb2.HybridPredicate:
        """field IN values."""
        return graph_pb2.HybridPredicate(
            in_list=graph_pb2.HybridInList(
                field=_field_ref(key),
                values_json=[json.dumps(v) for v in values],
            )
        )

    @staticmethod
    def not_in(key: str, values: Sequence[Any]) -> graph_pb2.HybridPredicate:
        """field NOT IN values."""
        return graph_pb2.HybridPredicate(
            not_in_list=graph_pb2.HybridInList(
                field=_field_ref(key),
                values_json=[json.dumps(v) for v in values],
            )
        )

    @staticmethod
    def exists(key: str) -> graph_pb2.HybridPredicate:
        """Property key is present."""
        return graph_pb2.HybridPredicate(exists=_field_ref(key))

    @staticmethod
    def label_eq(value: Any) -> graph_pb2.HybridPredicate:
        """Node label == value."""
        return graph_pb2.HybridPredicate(
            compare=graph_pb2.HybridCompare(
                field=graph_pb2.HybridPropertyRef(label=True),
                op=graph_pb2.HYBRID_CMP_EQ,
                value_json=json.dumps(value),
            )
        )

    @staticmethod
    def and_(*preds: graph_pb2.HybridPredicate) -> graph_pb2.HybridPredicate:
        """All sub-predicates hold."""
        # 'and' is a Python keyword; set the oneof field via a kwargs dict.
        return graph_pb2.HybridPredicate(
            **{"and": graph_pb2.HybridPredicateList(preds=list(preds))}
        )

    @staticmethod
    def or_(*preds: graph_pb2.HybridPredicate) -> graph_pb2.HybridPredicate:
        """Any sub-predicate holds."""
        return graph_pb2.HybridPredicate(
            **{"or": graph_pb2.HybridPredicateList(preds=list(preds))}
        )

    @staticmethod
    def not_(pred: graph_pb2.HybridPredicate) -> graph_pb2.HybridPredicate:
        """Negate a predicate."""
        return graph_pb2.HybridPredicate(**{"not": pred})

    @staticmethod
    def any_() -> graph_pb2.HybridPredicate:
        """Always-true predicate (matches everything)."""
        return graph_pb2.HybridPredicate(always=True)

    @staticmethod
    def incident_edges(
        op,
        value: Any,
        *,
        edge_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> graph_pb2.HybridPredicate:
        """Structural node predicate: compare the count of a node's incident edges.

        Counts edges incident to the node, optionally constrained by ``edge_type``
        and ``direction`` ("outgoing" / "incoming" / "both"), and compares the
        count against ``value`` with ``op`` (one of the ``graph_pb2.HYBRID_CMP_*``
        constants). Node-only; needs the graph store to resolve. Example: at least
        three outgoing CITES edges is
        ``Predicate.incident_edges(graph_pb2.HYBRID_CMP_GE, 3, edge_type="CITES")``.
        """
        ref = graph_pb2.HybridPropertyRef(
            incident_edge_count=graph_pb2.HybridIncidentEdgeCount(
                edge_type=edge_type or "",
                direction=_direction_to_proto(direction),
            )
        )
        return graph_pb2.HybridPredicate(
            compare=graph_pb2.HybridCompare(
                field=ref,
                op=op,
                value_json=json.dumps(value),
            )
        )


# A sub-plan source: either a builder (sync or async) or a prebuilt proto plan.
PlanLike = Union["_BaseHybridQueryBuilder", graph_pb2.HybridQueryPlan]


def _resolve_plan(plan: PlanLike) -> graph_pb2.HybridQueryPlan:
    """Coerce a builder or proto plan into a proto HybridQueryPlan."""
    if isinstance(plan, graph_pb2.HybridQueryPlan):
        return plan
    if isinstance(plan, _BaseHybridQueryBuilder):
        return plan.to_plan()
    raise TypeError(
        "expected a HybridQueryBuilder or HybridQueryPlan, "
        f"got {type(plan).__name__}"
    )


def _result_from_response(response: Any) -> HybridQueryResult:
    """Convert a HybridQueryResponse into a HybridQueryResult."""
    # Local import avoids a circular import at module load time.
    from .graph import _edge_from_proto, _node_from_proto

    return HybridQueryResult(
        nodes=[_node_from_proto(n) for n in response.nodes],
        edges=[_edge_from_proto(e) for e in response.edges],
        paths=[list(p.nodes) for p in response.paths],
    )


class _BaseHybridQueryBuilder:
    """Shared step-building logic for the sync and async builders.

    Subclasses add the terminal execution methods (sync vs async). All step
    methods append a HybridStep and return ``self`` for chaining.
    """

    def __init__(self, client: Any, collection: str) -> None:
        self._client = client
        self._collection = collection
        self._steps: List[graph_pb2.HybridStep] = []
        # Optional graph-aware RRF ranking spec. None = ranking OFF (default).
        self._rrf_rank: Optional[graph_pb2.RrfRankSpec] = None

    # ── Source steps ──

    def vector_similar(
        self,
        vector: Sequence[float],
        k: int,
        *,
        ef_search: Optional[int] = None,
    ):
        """Seed the result set with the k nearest neighbors of a vector."""
        sim = graph_pb2.HybridVectorSimilar(vector=list(vector), k=k)
        if ef_search is not None:
            sim.ef_search = ef_search
        self._steps.append(graph_pb2.HybridStep(vector_similar=sim))
        return self

    def vector_rank(self, vector, k, *, on_missing="skip"):
        """Rank the current node frontier by similarity to `vector`, keeping the
        top `k`. Graph-first (scope by structure, then rank by meaning): the graph
        has already fixed the candidate set, so scoring is exact over exactly those
        nodes (recall 1.0, no ANN, no ef_search). Returns nodes in ascending-distance
        order (most similar first), smaller node id breaking ties. `on_missing`
        governs frontier nodes with no vector: "skip" (default) drops and counts
        them, "error" fails the query.
        """
        on_missing_map = {
            "skip": graph_pb2.HYBRID_ON_MISSING_SKIP,
            "error": graph_pb2.HYBRID_ON_MISSING_ERROR,
        }
        key = str(on_missing).lower()
        if key not in on_missing_map:
            raise ValueError(f"on_missing must be 'skip' or 'error', got {on_missing!r}")
        rank = graph_pb2.HybridVectorRank(vector=list(vector), k=k, on_missing=on_missing_map[key])
        self._steps.append(graph_pb2.HybridStep(vector_rank=rank))
        return self

    # ── Vector-math steps over the graph frontier (P17) ──

    @staticmethod
    def _on_missing_to_proto(on_missing):
        """Map an on_missing string to its proto enum value (skip / error)."""
        on_missing_map = {
            "skip": graph_pb2.HYBRID_ON_MISSING_SKIP,
            "error": graph_pb2.HYBRID_ON_MISSING_ERROR,
        }
        key = str(on_missing).lower()
        if key not in on_missing_map:
            raise ValueError(
                f"on_missing must be 'skip' or 'error', got {on_missing!r}"
            )
        return on_missing_map[key]

    def analogy_rank(self, a, b, c, k, *, on_missing="skip"):
        """Rank the frontier by distance to the analogy vector ``a - b + c``,
        keeping the top ``k`` (ascending distance). Ranks by the index metric.
        ``on_missing`` governs frontier nodes with no vector ("skip" / "error").
        """
        vm = graph_pb2.HybridVectorMath(
            analogy=graph_pb2.HybridAnalogy(a=list(a), b=list(b), c=list(c)),
            k=k,
            on_missing=self._on_missing_to_proto(on_missing),
        )
        self._steps.append(graph_pb2.HybridStep(vector_math=vm))
        return self

    def diversity_rank(self, query, lambda_, k, *, on_missing="skip"):
        """Select up to ``k`` diverse frontier nodes via MMR against ``query``
        (``lambda_`` in [0,1] trades relevance vs. diversity), returned in
        MMR-selection order. MMR uses internal cosine similarity, not the index
        metric. ``on_missing`` governs frontier nodes with no vector.
        """
        vm = graph_pb2.HybridVectorMath(
            diversity=graph_pb2.HybridDiversity(
                query=list(query), **{"lambda": lambda_}
            ),
            k=k,
            on_missing=self._on_missing_to_proto(on_missing),
        )
        self._steps.append(graph_pb2.HybridStep(vector_math=vm))
        return self

    def cone_filter(self, direction, aperture_radians, k, *, on_missing="skip"):
        """Keep frontier nodes whose angle to ``direction`` is <=
        ``aperture_radians``, ranked by ascending angle, capped at ``k``. Uses
        internal cosine geometry. ``on_missing`` governs nodes with no vector.
        """
        vm = graph_pb2.HybridVectorMath(
            cone=graph_pb2.HybridCone(
                direction=list(direction), aperture_radians=aperture_radians
            ),
            k=k,
            on_missing=self._on_missing_to_proto(on_missing),
        )
        self._steps.append(graph_pb2.HybridStep(vector_math=vm))
        return self

    def isolation_rank(self, centroids, k, *, on_missing="skip"):
        """Rank frontier nodes by how isolated they are: score = min distance to
        any of ``centroids`` (a sequence of vectors); top ``k`` MOST isolated
        first (descending). Ranks by the index metric. ``on_missing`` governs
        frontier nodes with no vector.
        """
        vm = graph_pb2.HybridVectorMath(
            isolation=graph_pb2.HybridIsolation(
                centroids=[graph_pb2.HybridVector(values=list(c)) for c in centroids]
            ),
            k=k,
            on_missing=self._on_missing_to_proto(on_missing),
        )
        self._steps.append(graph_pb2.HybridStep(vector_math=vm))
        return self

    def centroid_rank(self, k, *, on_missing="skip"):
        """Compute the centroid (mean) of the frontier's own vectors, then rank
        the frontier by ascending distance to it (most representative first),
        keeping the top ``k``. Ranks by the index metric. ``on_missing`` governs
        frontier nodes with no vector.
        """
        vm = graph_pb2.HybridVectorMath(
            centroid=graph_pb2.HybridCentroid(),
            k=k,
            on_missing=self._on_missing_to_proto(on_missing),
        )
        self._steps.append(graph_pb2.HybridStep(vector_math=vm))
        return self

    def interpolate_rank(self, a, b, t, k, *, on_missing="skip"):
        """Rank the frontier by distance to the interpolated point between ``a``
        and ``b`` at fraction ``t`` in [0,1] (slerp with lerp fallback), keeping
        the top ``k`` (ascending). Ranks by the index metric. ``on_missing``
        governs frontier nodes with no vector.
        """
        vm = graph_pb2.HybridVectorMath(
            interpolate=graph_pb2.HybridInterpolate(a=list(a), b=list(b), t=t),
            k=k,
            on_missing=self._on_missing_to_proto(on_missing),
        )
        self._steps.append(graph_pb2.HybridStep(vector_math=vm))
        return self

    def from_node(self, node_id: int):
        """Seed the result set with a single node id."""
        return self.from_nodes([node_id])

    def from_nodes(self, node_ids: Sequence[int]):
        """Seed the result set with explicit node ids."""
        self._steps.append(
            graph_pb2.HybridStep(
                from_nodes=graph_pb2.HybridFromNodes(nodes=list(node_ids))
            )
        )
        return self

    def scan_by_filter(
        self,
        *,
        kind: Optional[str] = None,
        label: str = "",
        predicate: Optional[graph_pb2.HybridPredicate] = None,
    ):
        """Seed the result set by scanning the graph for matching nodes.

        A SOURCE step with no vector and no explicit ids: it produces the initial
        frontier from nodes matching an optional ``kind`` ("content" / "entity"),
        entity ``label``, and ``predicate`` (a property condition, including the
        structural incident-edge-count term). Chain the normal traversal steps
        after it.
        """
        scan = graph_pb2.HybridScanByFilter(label=label or "")
        if kind is not None:
            scan.filter_by_kind = True
            from .graph import _kind_to_proto

            scan.kind = _kind_to_proto(kind)
        if predicate is not None:
            scan.predicate.CopyFrom(predicate)
        self._steps.append(graph_pb2.HybridStep(scan_by_filter=scan))
        return self

    # ── Traversal steps ──

    def traverse(
        self,
        edge_type: Optional[str] = None,
        direction: str = "outgoing",
        *,
        as_of: Optional[int] = None,
        include_unbounded: bool = True,
        context: Optional[str] = None,
    ):
        """Walk one hop along edges (empty edge_type = any type).

        Pass ``as_of`` (unix-epoch millis), ``include_unbounded``, and/or
        ``context`` to filter the hop to edges valid at that time and regime
        (P17). All default to no filtering, leaving the hop unchanged.
        """
        traverse = graph_pb2.HybridTraverse(
            edge_type=edge_type or "",
            direction=_direction_to_proto(direction),
        )
        tf = _temporal_filter(as_of, include_unbounded, context)
        if tf is not None:
            traverse.temporal.CopyFrom(tf)
        self._steps.append(graph_pb2.HybridStep(traverse=traverse))
        return self

    def k_hop(
        self,
        edge_type: Optional[str] = None,
        max: int = 1,
        predicate: Optional[graph_pb2.HybridPredicate] = None,
        *,
        weight: Optional[WeightSpec] = None,
        order_by_weight: bool = False,
        as_of: Optional[int] = None,
        include_unbounded: bool = True,
        context: Optional[str] = None,
    ):
        """Expand up to ``max`` hops, optionally gated by a predicate.

        Pass ``weight`` to quality-weight the hops (confidence / recency /
        explicit) and ``order_by_weight`` to order the frontier by accumulated
        weight. Both default OFF, leaving unweighted hops unchanged.

        Pass ``as_of`` (unix-epoch millis), ``include_unbounded``, and/or
        ``context`` to filter the hops to edges valid at that time and regime
        (P17). All default to no filtering.
        """
        k_hop = graph_pb2.HybridKHop(edge_type=edge_type or "", max=max)
        if predicate is not None:
            k_hop.predicate.CopyFrom(predicate)
        if weight is not None:
            k_hop.weight.CopyFrom(weight._to_proto())
        k_hop.order_by_weight = order_by_weight
        tf = _temporal_filter(as_of, include_unbounded, context)
        if tf is not None:
            k_hop.temporal.CopyFrom(tf)
        self._steps.append(graph_pb2.HybridStep(k_hop=k_hop))
        return self

    def shortest_path(
        self,
        edge_types: Sequence[str],
        target: int,
        *,
        weighted: bool = False,
        weight: Optional[WeightSpec] = None,
        as_of: Optional[int] = None,
        include_unbounded: bool = True,
        context: Optional[str] = None,
    ):
        """Find the shortest path to a target node along the given edge types.

        Set ``weighted`` for a cost-weighted path instead of hop count, and pass
        ``weight`` to choose the edge-cost source. Both default OFF, leaving the
        hop-count shortest path unchanged.

        Pass ``as_of`` (unix-epoch millis), ``include_unbounded``, and/or
        ``context`` to restrict the path to edges valid at that time and regime
        (P17). All default to no filtering.
        """
        sp = graph_pb2.HybridShortestPath(
            edge_types=list(edge_types),
            target=target,
            weighted=weighted,
        )
        if weight is not None:
            sp.weight.CopyFrom(weight._to_proto())
        tf = _temporal_filter(as_of, include_unbounded, context)
        if tf is not None:
            sp.temporal.CopyFrom(tf)
        self._steps.append(graph_pb2.HybridStep(shortest_path=sp))
        return self

    # ── Set-combination steps (embed a sub-plan) ──

    def mutual_neighbors(self, other_plan: PlanLike):
        """Keep nodes reachable by both the current and the other plan."""
        self._steps.append(
            graph_pb2.HybridStep(mutual_neighbors=_resolve_plan(other_plan))
        )
        return self

    def intersect(self, other_plan: PlanLike):
        """Intersect the current result set with another plan's result set."""
        self._steps.append(
            graph_pb2.HybridStep(intersect=_resolve_plan(other_plan))
        )
        return self

    def union(self, other_plan: PlanLike):
        """Union the current result set with another plan's result set."""
        self._steps.append(
            graph_pb2.HybridStep(union=_resolve_plan(other_plan))
        )
        return self

    # ── Refinement steps ──

    def filter(self, predicate: graph_pb2.HybridPredicate):
        """Keep only nodes matching a predicate (see the Predicate helpers)."""
        self._steps.append(graph_pb2.HybridStep(filter=predicate))
        return self

    def edges(
        self,
        edge_type: Optional[str] = None,
        direction: str = "outgoing",
    ):
        """Collect edges incident to the current nodes (empty type = any)."""
        self._steps.append(
            graph_pb2.HybridStep(
                collect_edges=graph_pb2.HybridCollectEdges(
                    edge_type=edge_type or "",
                    direction=_direction_to_proto(direction),
                )
            )
        )
        return self

    def limit(self, n: int):
        """Cap the result set to ``n`` items."""
        self._steps.append(graph_pb2.HybridStep(limit=n))
        return self

    # ── Optional graph-aware ranking (opt-in, off by default) ──

    def rank_rrf(
        self,
        k: int,
        *,
        rrf_k: int = 60,
        k_hop_max: int = 2,
        relation_edge_types: Optional[Sequence[str]] = None,
        hub_damping: float = 0.0,
        edge_weight: Optional[WeightSpec] = None,
    ):
        """Opt in to graph-aware Reciprocal Rank Fusion of the result.

        OPTIONAL and OFF BY DEFAULT. When set, the server fuses two rankings of
        the candidate pool and returns the fused top-k nodes:

          - the vector seed in its similarity order, and
          - a graph-proximity ranking by the count of distinct bridge routes
            from the vector seed to each candidate (graph structure only).

        Fusion is standard RRF: score(d) = sum of 1 / (rrf_k + rank) over each
        ranking d appears in, equal weight, canonical ``rrf_k`` of 60.

        IMPORTANT, the seed-only form gives ZERO lift. The server fuses only
        the candidate pool the plan actually builds. If the plan is just
        ``vector_similar(qv, k).rank_rrf(...)``, the pool is only the vector
        seed, the graph-reached bridges are never in the pool, there is nothing
        for the fusion to pull in, and the result is the vector order unchanged.
        To get the recall lift, the plan MUST ALSO compose the graph expansion
        so the pool includes the graph-reached nodes. For GraphRAG that is:
        traverse the ``mentions`` edge outgoing, ``k_hop`` the relation types,
        then traverse ``mentions`` incoming, and only then ``rank_rrf(...)``.
        The composed plan that recovers missed gold looks like::

            result = (
                client.graph.query("docs_graph")
                .vector_similar(qv, k=10)
                .traverse("mentions", direction="outgoing")
                .k_hop("CITES", max=2)
                .traverse("mentions", direction="incoming")
                .rank_rrf(k=10, relation_edge_types=["CITES"])
                .return_nodes()
            )

        This RECOVERS gold the vector arm missed on multi-hop or
        vector-dissimilar bridges. It ADDS LATENCY because it computes graph
        proximity, which is why it is opt-in; leave it unset for the default,
        unchanged behavior. The plan MUST begin with ``vector_similar`` (the
        seed the ranking fuses from), MUST compose the graph expansion above so
        the pool has bridges to fuse, and MUST return nodes. See
        ``docs/graph-guide.md`` for the full worked example.

        Args:
            k: Final top-k cut on the fused order. Pass <= 0 to return the
                whole pool fused (no cut).
            rrf_k: RRF constant in 1 / (rrf_k + rank). Default 60 (canonical).
            k_hop_max: Bridge-hop bound for the proximity walk. Default 2.
            relation_edge_types: Typed relation edge types for the entity-bridge
                proximity hop. When empty/None the structural shape is used
                (content-to-content), matching a collection whose graph carries
                direct content edges rather than entity relations.
            hub_damping: Hub-aware damping strength. Default 0.0 = OFF, which is
                byte-identical to the prior count-based fusion. When > 0.0 each
                bridge route is down-weighted by the degree of the hub entities
                it crosses (weight = product of 1 / (1 + hub_damping *
                ln(1 + degree))), so routes through high-degree hub entities
                contribute less. Helps hub-dense graphs (for example
                conversational memory, where a few entities are mentioned almost
                everywhere) whose bridges otherwise flood the candidate pool. A
                principled starting value is 1.0; leave at 0.0 for the validated
                default behavior.
            edge_weight: Optional quality weighting for the bridge routes
                (confidence / recency / explicit). Default None = unweighted
                route counting, unchanged behavior. Pass a WeightSpec to fold
                edge quality into the proximity ranking.
        """
        rrf = graph_pb2.RrfRankSpec(
            k=k,
            rrf_k=rrf_k,
            k_hop_max=k_hop_max,
            relation_edge_types=list(relation_edge_types or []),
            hub_damping=hub_damping,
        )
        if edge_weight is not None:
            rrf.edge_weight.CopyFrom(edge_weight._to_proto())
        self._rrf_rank = rrf
        return self

    # ── Plan assembly (no execution) ──

    def to_plan(
        self,
        return_kind=graph_pb2.HYBRID_RETURN_NODES,
    ) -> graph_pb2.HybridQueryPlan:
        """Build the proto plan without executing it (for use as a sub-query)."""
        return graph_pb2.HybridQueryPlan(
            steps=list(self._steps),
            return_kind=return_kind,
        )

    # Alias mirroring the spec's build_plan name.
    build_plan = to_plan

    def _request(self, return_kind) -> graph_pb2.HybridQueryRequest:
        """Assemble the request proto for a given return kind.

        The optional RRF ranking spec is attached only when ``rank_rrf`` was
        called; otherwise the field is left absent so the server runs the
        default, unchanged path.
        """
        request = graph_pb2.HybridQueryRequest(
            collection=self._collection,
            plan=self.to_plan(return_kind=return_kind),
        )
        if self._rrf_rank is not None:
            request.rrf_rank.CopyFrom(self._rrf_rank)
        return request


# Graph-augmented fusion modes for compose_graph_rag. vector_rank is the default
# (graph-first scope-then-rank, ADR-024); rrf is the explicit opt-in.
FUSION_VECTOR_RANK = "vector_rank"
FUSION_RRF = "rrf"


def compose_graph_rag(
    seed: _BaseHybridQueryBuilder,
    bridge: _BaseHybridQueryBuilder,
    query_vector: Sequence[float],
    k: int,
    *,
    fusion: str = FUSION_VECTOR_RANK,
    mentions_edge_type: str = "mentions",
    relation_edge_types: Optional[Sequence[str]] = None,
    k_hop_max: int = 2,
    rrf_k: int = 60,
    hub_damping: float = 0.0,
    on_missing: str = "skip",
    ef_search: Optional[int] = None,
) -> _BaseHybridQueryBuilder:
    """Compose the documented graph-augmented plan onto a fresh builder.

    Shared by the sync and async ``graph_rag`` helpers. Both ``seed`` and
    ``bridge`` are fresh, empty builders on the same collection (the bridge is
    only used to assemble sub-plans and is never executed). Returns the ``seed``
    builder with all steps attached, ready for ``return_nodes``.

    Both fusion modes build the SAME candidate pool (the vector seed unioned with
    the graph expansion); they differ only in the final ranking step:

      - vector seed: ``vector_similar(query_vector, k)``
      - for each relation type R, a bridge sub-plan:
        ``vector_similar(qv, k) -> traverse(mentions, outgoing)
          -> k_hop(R, max=k_hop_max) -> traverse(mentions, incoming)``
      - union the seed with every bridge sub-plan so the candidate pool holds
        the graph-reached nodes
      - then the ranking step chosen by ``fusion``.

    When ``relation_edge_types`` is empty/None, the structural fallback is used:
    a single bridge ``vector_similar(qv, k) -> k_hop(any, max=k_hop_max)``
    unioned with the seed. This fits content-to-content graphs that carry direct
    edges rather than entity relations.

    ``fusion`` selects the ranking step (ADR-024):

      - ``"vector_rank"`` (DEFAULT): graph-first scope-then-rank. The graph has
        already fixed the candidate pool; ``.vector_rank(query_vector, k)`` then
        ranks EXACTLY within that pool by similarity (recall 1.0, no ANN). This
        is the default graph-augmented path: it ties plain vector retrieval and
        wins on the hard multi-hop and adversarial cases.
      - ``"rrf"`` (OPT-IN): the prior Reciprocal Rank Fusion ranking via
        ``rank_rrf(k, rrf_k, k_hop_max, relation_edge_types, hub_damping)``.
        Reachable only by explicitly passing ``fusion="rrf"``; its behavior is
        byte-identical to before.
    """
    relations = list(relation_edge_types or [])

    # The vector seed: the arm both ranking modes start from.
    seed.vector_similar(query_vector, k, ef_search=ef_search)

    if relations:
        # One bridge sub-plan per relation type, unioned into the pool.
        for relation in relations:
            sub = bridge.__class__(bridge._client, bridge._collection)
            sub_plan = (
                sub.vector_similar(query_vector, k, ef_search=ef_search)
                .traverse(mentions_edge_type, direction="outgoing")
                .k_hop(relation, max=k_hop_max)
                .traverse(mentions_edge_type, direction="incoming")
                .to_plan()
            )
            seed.union(sub_plan)
    else:
        # Structural fallback: content-to-content graphs, any edge type.
        sub = bridge.__class__(bridge._client, bridge._collection)
        sub_plan = (
            sub.vector_similar(query_vector, k, ef_search=ef_search)
            .k_hop(None, max=k_hop_max)
            .to_plan()
        )
        seed.union(sub_plan)

    mode = str(fusion).lower()
    if mode == FUSION_RRF:
        # Opt-in: fuse the vector and graph-proximity rankings with RRF.
        seed.rank_rrf(
            k,
            rrf_k=rrf_k,
            k_hop_max=k_hop_max,
            relation_edge_types=relations,
            hub_damping=hub_damping,
        )
    elif mode == FUSION_VECTOR_RANK:
        # Default: rank exactly within the graph-scoped pool by similarity.
        seed.vector_rank(query_vector, k, on_missing=on_missing)
    else:
        raise ValueError(
            f"fusion must be {FUSION_VECTOR_RANK!r} or {FUSION_RRF!r}, "
            f"got {fusion!r}"
        )
    return seed


class HybridQueryBuilder(_BaseHybridQueryBuilder):
    """Synchronous, chainable hybrid graph query builder.

    Usage::

        result = (
            client.graph.query("col")
            .vector_similar([0.1] * 128, k=10)
            .traverse(":BOUGHT")
            .filter(Predicate.eq("in_stock", True))
            .limit(20)
            .return_nodes()
        )
    """

    def __init__(self, client: "SwarnDBClient", collection: str) -> None:
        super().__init__(client, collection)

    def return_nodes(self) -> HybridQueryResult:
        """Execute the plan and return matching nodes."""
        request = self._request(graph_pb2.HYBRID_RETURN_NODES)
        response = self._client._call(
            self._client._graph_stub.HybridQuery, request
        )
        return _result_from_response(response)

    def return_edges(self) -> HybridQueryResult:
        """Execute the plan and return collected edges."""
        request = self._request(graph_pb2.HYBRID_RETURN_EDGES)
        response = self._client._call(
            self._client._graph_stub.HybridQuery, request
        )
        return _result_from_response(response)

    def return_paths(self) -> HybridQueryResult:
        """Execute the plan and return matching paths (lists of node ids)."""
        request = self._request(graph_pb2.HYBRID_RETURN_PATHS)
        response = self._client._call(
            self._client._graph_stub.HybridQuery, request
        )
        return _result_from_response(response)


class AsyncHybridQueryBuilder(_BaseHybridQueryBuilder):
    """Asynchronous, chainable hybrid graph query builder.

    Feature-identical to HybridQueryBuilder; the terminal methods are
    awaitable. Usage::

        result = await (
            client.graph.query("col")
            .vector_similar([0.1] * 128, k=10)
            .traverse(":BOUGHT")
            .return_nodes()
        )
    """

    async def return_nodes(self) -> HybridQueryResult:
        """Execute the plan and return matching nodes."""
        request = self._request(graph_pb2.HYBRID_RETURN_NODES)
        response = await self._client._call(
            self._client._graph_stub.HybridQuery, request
        )
        return _result_from_response(response)

    async def return_edges(self) -> HybridQueryResult:
        """Execute the plan and return collected edges."""
        request = self._request(graph_pb2.HYBRID_RETURN_EDGES)
        response = await self._client._call(
            self._client._graph_stub.HybridQuery, request
        )
        return _result_from_response(response)

    async def return_paths(self) -> HybridQueryResult:
        """Execute the plan and return matching paths (lists of node ids)."""
        request = self._request(graph_pb2.HYBRID_RETURN_PATHS)
        response = await self._client._call(
            self._client._graph_stub.HybridQuery, request
        )
        return _result_from_response(response)
