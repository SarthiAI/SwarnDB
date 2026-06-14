# Typed Graph: Complete Guide

This is the complete, end-to-end how-to and API reference for the **typed graph** (the explicit, first-class graph available on `hybrid` collections) and how to drive it from the Python SDK. Every method, field, state, and metric named here is taken from the live SDK source.

SwarnDB has two graph surfaces. This guide is the reference for the typed graph; the other surface, the virtual graph, has its own dedicated doc:

1. The **virtual graph (SwarnDB's automatic similarity graph)**, available on `auto_similarity` collections. SwarnDB connects similar vectors automatically using a similarity threshold; you never build it by hand. Section 1 below is a short recap; the full treatment, including REST, tuning, and graph-enriched search, is in [Virtual Graph](virtual-graph.md).
2. The **typed graph**, present only on collections created in `hybrid` mode and the subject of this guide. Here you store typed nodes (content and entity) and typed, directed, labelled edges, run a composable hybrid query that mixes vector similarity with graph walks, and optionally let an LLM extract entities and relationships from your text.

Vector-only users can ignore the typed graph entirely: it is opt-in and adds nothing to the cost of a plain vector collection.

Related docs:

- [Typed Graph: Overview](graph-first-class.md): the short overview and where to start; explains the two graph surfaces and which one to use.
- [Virtual Graph](virtual-graph.md): the virtual graph (the automatic similarity graph) in depth.
- [LLM Extraction](llm-extraction.md): the extraction concept and ontology templates.
- [Python SDK](python-sdk.md): the full SDK reference; the graph and extraction sections cross-link here.
- [API Reference](api-reference.md): the gRPC service contracts.

Throughout, `client` is a connected `SwarnDBClient`. Every method shown on `client.graph` and `client.extraction` has an async twin on `AsyncSwarnDBClient` with the same name and signature; the async terminals are awaitable. The async section near the end shows the pattern.

---

## 1. The virtual graph (recap)

The virtual graph (SwarnDB's automatic similarity graph) lives on `auto_similarity` collections. It connects vectors whose similarity is at or above a threshold, built automatically as you insert, with no curation. This is a short recap so the typed-graph reference below stands on its own; the full how-to (REST, tuning, graph-enriched search, deferred mode, use cases) is in [Virtual Graph](virtual-graph.md).

The key calls, all on `client.graph`:

- `set_threshold(collection, threshold, *, vector_id=0)`: set the collection-level threshold (a higher threshold means fewer, tighter edges), or a per-vector override when `vector_id` is non-zero. After changing a collection-level threshold, call `client.collections.optimize(collection)` to rebuild the graph.
- `get_related(collection, vector_id, *, threshold=0.0, max_results=10) -> list[GraphEdge]`: read a vector's direct neighbours, each a `GraphEdge` with `target_id` and `similarity`.
- `traverse(collection, start_id, *, depth=2, threshold=0.0, max_results=100) -> list[TraversalNode]`: walk multiple hops outward, returning `TraversalNode`s with `id`, `depth`, `path_similarity`, and `path`.

Graph-enriched search (returning neighbours alongside ranked results) lives on `client.search.query(..., include_graph=True)`. For the threshold-precedence rules, tuning guidance, REST routes, and worked use cases, see [Virtual Graph](virtual-graph.md).

The rest of this guide covers the typed graph, which is the explicit, curated graph on `hybrid` collections.

---

## 2. Typed nodes (hybrid mode)

Hybrid collections add a first-class typed graph. A node has a `kind` that is either `"content"` (a unit of source material, usually with an embedding) or `"entity"` (a thing extracted or curated, such as a person or product). Typed-graph calls only work on `hybrid` collections; the server rejects them elsewhere.

### 2.1 Create: `put_node`

```python
node_id = client.graph.put_node(
    "docs_graph",
    kind="entity",                       # "content" or "entity"
    label="Person",
    properties={"name": "Ada Lovelace"},
    embedding=[0.0] * 1536,              # optional; used for entity dedup
    source="manual",
    created_by="curator",
)
```

Signature:

```python
put_node(
    collection, *,
    kind="content", label="", properties=None,
    embedding=None, source="manual", created_by="",
) -> int
```

Returns the new node id. `kind` must be exactly `"content"` or `"entity"`; any other value raises `ValueError` before the call leaves the client. `properties` is a free-form dict that travels to the server as JSON.

### 2.2 Read: `get_node`

```python
node = client.graph.get_node("docs_graph", node_id)
if node is not None:
    print(node.id, node.kind, node.label, node.properties)
```

Signature:

```python
get_node(collection, node_id) -> TypedNode | None
```

Returns `None` when the node does not exist. A `TypedNode` carries: `id`, `kind`, `label`, `properties`, `embedding`, `source`, `created_at`, and `created_by`.

### 2.3 Delete: `delete_node`

```python
existed = client.graph.delete_node("docs_graph", node_id)
```

Signature:

```python
delete_node(collection, node_id) -> bool
```

Returns `True` if the node existed and was removed. Deleting a node also removes the edges incident to it.

### 2.4 Update a node's properties: `update_node`

You can change a node's property bag after it is created, while everything that anchors the node stays put. The node's provenance (its `source`, `created_at`, and `created_by`) and its `embedding` are immutable: only `properties` can change. The embedding is held fixed on purpose, because a content node shares its id with the vector it stands for (the NodeId == VectorId bridge), and letting the embedding drift would break that link.

```python
node = client.graph.update_node(
    "docs_graph", node_id,
    properties={"name": "Ada Lovelace", "verified_by": "curator"},
    actor="curator",
)
print(node.properties, node.updated_at)
```

Signature:

```python
update_node(collection, node_id, *, properties=None, actor="") -> TypedNode
```

The whole property bag is replaced by what you pass in `properties`; omitting `properties` leaves the bag unchanged and records an audit-only touch. The `actor` is written to the node's audit history. Returns the updated `TypedNode`, whose `updated_at` reflects the change and whose `history` carries a `NodeAudit` entry (each with `action`, `actor`, and `at`) for the update.

---

## 3. Typed edges (hybrid mode)

A typed edge is directed and labelled. It carries a confidence, a verified flag, a manual flag, a free-form property bag, a provenance bag, and an append-only audit history.

### 3.1 Create: `put_edge`

```python
edge_id = client.graph.put_edge(
    "docs_graph",
    source=content_node_id,
    target=person_node_id,
    edge_type="AUTHORED_BY",
    properties={"page": 1},
    provenance={"doc_id": "paper-1", "chunk_id": 3},
    confidence=1.0,
    verified=False,
    is_manual=True,
)
```

Signature:

```python
put_edge(
    collection, source, target, edge_type, *,
    properties=None, provenance=None,
    confidence=1.0, verified=False, is_manual=True,
) -> int
```

Returns the new edge id. `source` and `target` are node ids. `is_manual=True` marks the edge as human-authored, which protects it during re-extraction (see Section 5.6).

### 3.2 Read: `get_edge`

```python
edge = client.graph.get_edge("docs_graph", edge_id)
if edge is not None:
    print(edge.edge_type, edge.confidence, edge.verified, edge.is_manual)
    for entry in edge.history:
        print(entry.action, entry.actor, entry.at)
```

Signature:

```python
get_edge(collection, edge_id) -> TypedEdge | None
```

Returns `None` when the edge does not exist. A `TypedEdge` carries: `id`, `source`, `target`, `edge_type`, `properties`, `provenance`, `confidence`, `verified`, `is_manual`, `created_at`, and `history`. The `history` is a list of `EdgeAudit` entries, each with `action`, `actor`, and `at` (a timestamp).

### 3.3 List edges on a node: `list_edges`

```python
out = client.graph.list_edges(
    "docs_graph", node=content_node_id,
    direction="outgoing",        # "outgoing", "incoming", or "both"
    edge_type="AUTHORED_BY",     # empty string = any type
)
```

Signature:

```python
list_edges(collection, node, *, direction="outgoing", edge_type="") -> list[TypedEdge]
```

`direction` selects which edges to list relative to the node; an empty `edge_type` returns all types.

### 3.4 Update: `update_edge`

```python
updated = client.graph.update_edge(
    "docs_graph", edge_id,
    properties={"page": 2},
    confidence=0.9,
    verified=True,
    actor="curator",
)
```

Signature:

```python
update_edge(
    collection, edge_id, *,
    properties=None, confidence=None, verified=None, actor="",
) -> TypedEdge
```

Only the supplied fields change; any field left as `None` keeps its current value. The `actor` is recorded in the edge's audit history. Returns the updated `TypedEdge`.

### 3.5 Delete: `delete_edge`

```python
existed = client.graph.delete_edge("docs_graph", edge_id)
```

Signature:

```python
delete_edge(collection, edge_id) -> bool
```

Returns `True` if the edge existed.

### 3.6 Verify versus reject: curation semantics

These two calls are the heart of edge curation, and they are not symmetric.

`verify_edge` is a positive judgement. It marks the edge as trustworthy and records the action in the audit trail. The edge stays in the graph, and being verified protects it from being removed by a later re-extraction (Section 5.6).

```python
edge = client.graph.verify_edge("docs_graph", edge_id, actor="curator")
print(edge.verified)   # True
```

Signature:

```python
verify_edge(collection, edge_id, *, actor="") -> TypedEdge
```

`reject_edge` is a negative judgement. It deletes the edge, and it can also remember the rejection as a rule so the same wrong relationship is not re-proposed on a future extraction.

```python
result = client.graph.reject_edge("docs_graph", edge_id, actor="curator")
print(result.deleted, result.rule_added)
```

Signature:

```python
reject_edge(collection, edge_id, *, actor="") -> EdgeRejectResult
```

An `EdgeRejectResult` carries two booleans: `deleted` (whether the edge was removed) and `rule_added` (whether a rejection rule was recorded to suppress the pattern in future).

So: verify keeps and blesses; reject removes and remembers. Both write to the audit history through the `actor` you pass.

### 3.7 The audit trail: `EdgeAudit`

Every edge carries a `history` list of `EdgeAudit` entries. Each entry records one mutation, with `action` (what happened, for example a create, update, or verify), `actor` (who did it, from the `actor` argument), and `at` (when). Read it from any `TypedEdge`:

```python
edge = client.graph.get_edge("docs_graph", edge_id)
for entry in edge.history:
    print(f"{entry.at}: {entry.action} by {entry.actor or '(unknown)'}")
```

### 3.8 Bulk import edges: `bulk_import_edges`

For loading many edges at once, supply CSV or JSONL text.

```python
csv_data = (
    "source,target,edge_type,confidence\n"
    "42,99,CITES,1.0\n"
    "42,17,CITES,0.8\n"
)
report = client.graph.bulk_import_edges(
    "docs_graph", csv_data,
    format="csv",                 # "csv" or "jsonl"
    auto_add_edge_types=False,    # True lets unknown edge types be added on the fly
    actor="import-job",
)
print(report.total_rows, report.imported, report.failed)
for err in report.errors:
    print(f"row {err.row}: {err.message}")
```

Signature:

```python
bulk_import_edges(
    collection, data, *,
    format="csv", auto_add_edge_types=False, actor="",
) -> BulkImportResult
```

`format` must be `"csv"` or `"jsonl"` (case-insensitive); any other value raises `ValueError` before the call. A `BulkImportResult` carries `total_rows`, `imported`, `failed`, and `errors`, where each entry in `errors` is a `BulkImportRowError` with `row` and `message`. Rows that fail do not stop the others: the import is per-row, and you read exactly which rows failed and why.

When `auto_add_edge_types=False` (the default), any edge whose type is not already in the ontology fails its row. Set it to `True` to let the import register new edge types as it goes.

### 3.9 Temporal edges: validity windows and context

An edge can optionally carry a validity window and a context label, so the same graph can hold facts that were true over different stretches of time, or under different regimes. This is opt-in: leave the fields out and the edge is always valid and context-free, exactly as before.

```python
edge_id = client.graph.put_edge(
    "people_graph",
    source=alice_id, target=acme_id, edge_type="WORKS_AT",
    valid_from=1577836800000,        # unix-epoch millis; when the fact starts
    valid_until=1640995200000,       # unix-epoch millis, EXCLUSIVE; when it stops
    temporal_context="employment-v2",
)
```

The three optional fields on `put_edge`:

- `valid_from`: the start of the window, in unix-epoch milliseconds. Omit it for an unbounded start (valid since forever).
- `valid_until`: the end of the window, in unix-epoch milliseconds, and it is EXCLUSIVE (the instant equal to `valid_until` is already outside the window). Omit it for an unbounded end (still valid).
- `temporal_context`: a free-form label naming the regime or version this edge belongs to (for example `"employment-v2"`). Omit it for a context-free edge.

When you read an edge back, these surface on the `TypedEdge` as `valid_from`, `valid_until`, and `temporal_context`, each `None` when the edge does not set it.

```python
edge = client.graph.get_edge("people_graph", edge_id)
print(edge.valid_from, edge.valid_until, edge.temporal_context)
```

To query against these windows, see the time-filtered traversal in Section 4.8.

---

## 4. The hybrid query builder (hybrid mode)

`client.graph.query(collection)` starts a composable, chainable query that mixes vector similarity with graph walks. You chain steps, then finish with one of three terminals that executes the plan and returns a `HybridQueryResult`. The async client returns an awaitable builder with the same steps.

```python
from swarndb import Predicate, Direction

result = (
    client.graph.query("docs_graph")
    .vector_similar(query_vector, k=20)
    .traverse("AUTHORED_BY", direction=Direction.OUTGOING)
    .filter(Predicate.label_eq("Person"))
    .limit(10)
    .return_nodes()
)
for node in result.nodes:
    print(node.id, node.label, node.properties)
```

A `HybridQueryResult` carries `nodes`, `edges`, and `paths`; exactly one is populated, matching the terminal you call. `paths` is a list of paths, each a list of node ids.

### 4.1 The steps

The steps group by role. Each returns the builder so you can keep chaining. The source, traversal, set-combination, and refinement steps are below; the opt-in quality, temporal, scan, and vector-math steps follow in Sections 4.7 through 4.12.

**Source steps** (seed the working set):

- `vector_similar(vector, k, *, ef_search=None)`: seed with the `k` nearest neighbours of a vector. `ef_search` tunes HNSW search quality for this seed: higher means better recall and more work. Left unset, it uses the collection default.
- `from_node(node_id)`: seed with a single node id.
- `from_nodes(node_ids)`: seed with an explicit list of node ids.
- `scan_by_filter(*, kind=None, label="", predicate=None)`: seed by scanning the graph for nodes that match a filter, with no vector input (see Section 4.9).

```python
client.graph.query("docs_graph").vector_similar(query_vector, k=20, ef_search=200)
client.graph.query("docs_graph").from_node(42)
client.graph.query("docs_graph").from_nodes([42, 99, 17])
```

**Traversal steps** (walk the graph):

- `traverse(edge_type=None, direction="outgoing")`: walk exactly one hop. An empty `edge_type` walks edges of any type.
- `k_hop(edge_type=None, max=1, predicate=None)`: expand up to `max` hops, optionally gated so each hop only crosses edges to nodes matching `predicate`.
- `shortest_path(edge_types, target)`: find the shortest path from the current set to `target`, crossing only the listed edge types.

```python
client.graph.query("docs_graph").from_node(42).traverse("CITES", direction=Direction.INCOMING)
client.graph.query("docs_graph").from_node(42).k_hop("CITES", max=3, predicate=Predicate.eq("year", 2024))
client.graph.query("docs_graph").from_node(paper_a).shortest_path(["CITES"], target=paper_b)
```

**Set-combination steps** (combine with another plan, see Section 4.3):

- `mutual_neighbors(other_plan)`: keep nodes reachable by both the current plan and the other plan.
- `intersect(other_plan)`: keep nodes that appear in both result sets.
- `union(other_plan)`: combine both result sets.

**Refinement steps** (shape the working set):

- `filter(predicate)`: keep only nodes matching a predicate (see Section 4.4).
- `edges(edge_type=None, direction="outgoing")`: collect the edges incident to the current nodes; an empty `edge_type` collects any type. This is what populates `result.edges` when you call `return_edges()`.
- `limit(n)`: cap the working set to `n` items.

```python
client.graph.query("docs_graph").from_node(42).edges("AUTHORED_BY", direction=Direction.BOTH)
client.graph.query("docs_graph").vector_similar(query_vector, k=100).filter(Predicate.gt("price", 50)).limit(10)
```

`Direction` exposes the three constants `Direction.OUTGOING`, `Direction.INCOMING`, and `Direction.BOTH`. Passing any other direction string raises `ValueError`.

### 4.2 The terminals

Pick exactly one terminal; it executes the plan and returns a `HybridQueryResult`.

- `return_nodes()`: populates `result.nodes` with the working set as `TypedNode`s.
- `return_edges()`: populates `result.edges` with the `TypedEdge`s collected by `edges(...)` steps.
- `return_paths()`: populates `result.paths` with the matching paths (each a list of node ids).

```python
nodes = client.graph.query("docs_graph").vector_similar(qv, k=10).return_nodes().nodes
edges = client.graph.query("docs_graph").from_node(42).edges("CITES").return_edges().edges
paths = client.graph.query("docs_graph").from_node(a).shortest_path(["CITES"], b).return_paths().paths
```

### 4.3 Sub-plans: `to_plan` / `build_plan`

The set-combination steps take another plan rather than a live result. Build that inner plan with `to_plan()` (or its alias `build_plan()`), which assembles the plan proto without executing it, so it can be embedded into the outer query.

```python
from swarndb import Predicate

# Inner plan: papers tagged as benchmarks, built but not executed.
benchmarks = (
    client.graph.query("docs_graph")
    .from_node(survey_paper)
    .traverse("CITES")
    .filter(Predicate.eq("is_benchmark", True))
    .to_plan()            # alias: .build_plan()
)

# Outer plan: papers near the query vector that ALSO appear in the inner plan.
result = (
    client.graph.query("docs_graph")
    .vector_similar(query_vector, k=50)
    .intersect(benchmarks)
    .return_nodes()
)
```

`to_plan()` accepts an optional return-kind argument, but for sub-plans the default is what you want. Both `intersect`, `union`, and `mutual_neighbors` accept either a builder (it is converted to a plan for you) or a plan you built explicitly.

### 4.4 The predicate helpers

`Predicate` builds the filter expressions used by `filter(...)` and the `k_hop` gate. By default a predicate references a key in the node's property bag; `label_eq` references the node label instead, and `incident_edges` (Section 4.11) references the node's edge count instead. Scalar values are encoded as JSON literals on the wire, which is how the server matches them.

Comparisons:

- `Predicate.eq(key, value)`: field equals value.
- `Predicate.ne(key, value)`: field not equal to value.
- `Predicate.gt(key, value)`: field greater than value.
- `Predicate.ge(key, value)`: field greater than or equal to value.
- `Predicate.lt(key, value)`: field less than value.
- `Predicate.le(key, value)`: field less than or equal to value.

Membership and presence:

- `Predicate.is_in(key, values)`: field is one of the values.
- `Predicate.not_in(key, values)`: field is none of the values.
- `Predicate.exists(key)`: the property key is present.

Label:

- `Predicate.label_eq(value)`: the node's label equals value (references the label flag, not the property bag).

Logic:

- `Predicate.and_(*preds)`: all sub-predicates hold.
- `Predicate.or_(*preds)`: any sub-predicate holds.
- `Predicate.not_(pred)`: negate a predicate.
- `Predicate.any_()`: always true (matches everything).

```python
from swarndb import Predicate

p = Predicate.and_(
    Predicate.label_eq("Product"),
    Predicate.gt("price", 10),
    Predicate.or_(
        Predicate.eq("in_stock", True),
        Predicate.is_in("warehouse", ["east", "west"]),
    ),
    Predicate.not_(Predicate.exists("discontinued")),
)

result = (
    client.graph.query("docs_graph")
    .vector_similar(query_vector, k=100)
    .filter(p)
    .limit(20)
    .return_nodes()
)
```

#### Property-bag filtering and the JSON wire encoding

When you write `Predicate.eq("price", 29.99)`, the key `"price"` is sent as a reference into the node's property bag, and the value `29.99` is serialised with `json.dumps`, so it travels as the JSON literal `29.99`. The same rule holds for strings (`"east"` becomes `"east"` with quotes), booleans (`True` becomes `true`), and the lists passed to `is_in` / `not_in` (each element is JSON-encoded independently). This is why predicate values match the way they were stored in the property bag: both sides use the same JSON encoding. `label_eq` is the single helper that targets the node's label flag rather than a property key; everything else reads from the property bag.

### 4.5 Graph-aware ranking: `vector_rank` (default) and RRF (opt-in)

A hybrid query returns its results in the order the plan produced them. To rank a graph-augmented result there are two ranking steps, and the default is `vector_rank`.

The default graph-augmented ranking is graph-first scope-then-rank: scope the candidate set with the graph (a traversal or `k_hop` over the structure), then add `.vector_rank(query_vector, k)` as the final step. The graph has already fixed the candidate set, so the ranking is exact over exactly those nodes (recall 1.0, no approximate search):

```python
result = (
    client.graph.query("docs_graph")
    .vector_similar(query_vector, k=10)
    .traverse("mentions", direction="outgoing")
    .k_hop("CITES", max=2)
    .traverse("mentions", direction="incoming")
    .vector_rank(query_vector, k=10)
    .return_nodes()
)
```

This is the default because, on real data at scale, it ties plain vector retrieval on overall accuracy and wins on the hard multi-hop and adversarial questions. The one-call `graph_rag` helper in Section 4.6 composes exactly this for you.

You can instead OPT IN to a graph-aware re-ranking that fuses the vector signal with the graph signal with Reciprocal Rank Fusion. RRF stays fully supported; it is just no longer the default. Add `.rank_rrf(...)` before the terminal:

```python
result = (
    client.graph.query("docs_graph")
    .vector_similar(query_vector, k=10)
    .traverse("mentions", direction="outgoing")
    .k_hop("CITES", max=2)
    .traverse("mentions", direction="incoming")
    .rank_rrf(k=10, rrf_k=60, k_hop_max=2, relation_edge_types=["CITES"])
    .return_nodes()
)
```

What it does: the server builds the candidate pool from your plan (the vector seed together with the graph expansion), then ranks it two ways and fuses them with standard Reciprocal Rank Fusion:

- the vector seed in its similarity order, and
- a graph-proximity ranking, scoring each candidate by the number of distinct bridge routes that reach it from the vector seed through the graph (graph structure only, no vector score).

It then returns the fused top-k. A candidate that the vector arm ranked low but the graph reached strongly rises into the top-k, which is the whole point.

When to use it: turn it on when relevant answers are reachable through the graph but are dissimilar to the query vector, for example a multi-hop bridge or an entity-linked passage that does not share the query's wording. In a real public-benchmark run this lifted recall on a question-answering set from 0.900 to 0.963 at equal k, recovering most of the gold the vector arm missed.

The cost: it ADDS LATENCY, because it computes graph proximity by walking bridge routes from the seed. That, plus the at-scale accuracy result above, is why RRF is opt-in and is no longer the default. If you do not call `.rank_rrf(...)`, the query runs as composed (the `vector_rank` step ranks the graph-scoped pool by default).

Arguments:

- `k`: the final top-k cut on the fused order. Pass `0` or less to return the whole fused pool.
- `rrf_k`: the RRF constant in `1 / (rrf_k + rank)`. Default `60` (the canonical value); leave it unless you have a specific reason.
- `k_hop_max`: how many hops the proximity walk may take. Default `2`.
- `relation_edge_types`: the typed relation edge types for the entity-bridge proximity walk (for example `["CITES"]`). Leave it empty for the structural shape, used when your graph carries direct content-to-content edges rather than entity relations.

The plan must begin with `vector_similar` (the seed the ranking fuses from) and return nodes. Ranking is supported on `return_nodes()`; it does not apply to edge or path results.

One trap to know: the seed-only form gives ZERO lift. If the plan is just `vector_similar(qv, k).rank_rrf(...)`, the candidate pool is only the vector seed, so there are no graph-reached bridges for the fusion to pull in and the order comes back unchanged. The lift only appears when the plan ALSO composes the graph expansion (the mentions/relation bridge) so the pool includes the graph-reached nodes. The one-call helper below composes that expansion for you (defaulting to `vector_rank`), so it is the recommended path.

### 4.6 One-call graph-augmented retrieval: `graph_rag` (recommended)

`client.graph.graph_rag(...)` is the recommended way to get graph-augmented retrieval. It composes the candidate pool for you (the vector seed plus the graph expansion that the seed-only form is missing), then ranks it. By default it uses `vector_rank` (graph-first scope-then-rank): the graph fixes the candidate set, and the result is ranked exactly within that set by similarity to your query. This is the default graph-augmented path.

```python
result = client.graph.graph_rag(
    "docs_graph",
    query_vector,
    k=10,
    relation_edge_types=["CITES"],
)
for node in result.nodes:
    print(node.id, node.label, node.properties)
```

Signature:

```python
graph_rag(
    collection, query_vector, k=10, *,
    fusion="vector_rank", mentions_edge_type="mentions",
    relation_edge_types=None, k_hop_max=2, rrf_k=60,
    hub_damping=0.0, ef_search=None,
) -> HybridQueryResult
```

It composes and runs exactly this plan: a vector seed (`vector_similar(query_vector, k)`); for every relation type in `relation_edge_types` a bridge sub-plan (`vector_similar(query_vector, k)` then `traverse(mentions_edge_type, outgoing)` then `k_hop(relation, max=k_hop_max)` then `traverse(mentions_edge_type, incoming)`); the seed unioned with all the bridge sub-plans so the candidate pool holds the graph-reached nodes; then `vector_rank(query_vector, k)` and `return_nodes()`, which ranks the graph-scoped pool exactly by similarity.

When `relation_edge_types` is empty or `None`, it falls back to the structural form for content-to-content graphs (graphs whose edges link passages directly rather than through entities): the seed unioned with `vector_similar(query_vector, k).k_hop(any, max=k_hop_max)`, then the ranking step.

Why `vector_rank` is the default: on real data at scale it ties plain vector retrieval on overall answer accuracy and wins on the hard multi-hop and adversarial questions, while the older RRF fusion regressed below plain vector retrieval. So the default graph-augmented path gives you a real lift on the hard questions and never a worse result than plain vector search.

RRF stays fully supported as an explicit opt-in; it is just no longer the default. The one-call path is `graph_rag(..., fusion="rrf")`, which runs the same composed plan but ends in RRF instead of `vector_rank` (and then applies `rrf_k` and `hub_damping`). Building the explicit chain from Section 4.5 and calling `.rank_rrf(...)` before the terminal is the customizable equivalent, for when you need to tune the composition yourself.

When you only want plain vector results, stay on `client.graph.query(...).vector_similar(...).return_nodes()`, which does no graph work. Use `graph_rag` when you want the graph-augmented result in one call; the explicit Section 4.5 chain stays available when you need to customise the composition or opt in to RRF.

The async client mirrors this: `await client.graph.graph_rag(...)`.

### 4.7 Quality-aware traversal and ranking: `WeightSpec`

By default every edge counts the same when you walk the graph. Quality-aware traversal lets you weight an edge by how much you trust it, so stronger relationships steer the walk and the ranking. This is opt-in and off by default: a query that does not pass a `WeightSpec` is unchanged, and so is every plain vector query.

An edge's weight can be built from three signals, which combine together:

- its confidence (the `confidence` you store on the edge),
- an explicit numeric weight you keep in a property (any property key, defaulting to `"weight"`), and
- recency, so older edges decay toward a smaller weight over a half-life you choose.

You describe which signals to use with a `WeightSpec`, imported from `swarndb`:

```python
from swarndb import WeightSpec

w = WeightSpec(
    use_confidence=True,           # fold the edge's confidence into the weight
    min_confidence=0.2,            # floor confidence at 0.2 before using it
    recency_half_life_ms=2_592_000_000,  # 30 days in ms; older edges decay
    use_explicit_weight=True,      # also read a numeric weight from a property
    explicit_weight_key="weight",  # which property holds it (default "weight")
)
```

The `WeightSpec` fields:

- `use_confidence` (default `False`): when `True`, the edge's stored confidence multiplies into the weight.
- `min_confidence` (default `0.0`): a floor applied to confidence before it is used, so very low confidence does not drag the weight to nothing.
- `recency_half_life_ms` (default `0`, meaning no decay): when positive, older edges are down-weighted with this half-life in milliseconds (an edge one half-life old carries about half the weight of a brand-new one).
- `use_explicit_weight` (default `False`): when `True`, a numeric value read from the edge's property bag multiplies into the weight.
- `explicit_weight_key` (default `"weight"`): which property key holds that explicit number.

Leaving every field at its default makes the spec a no-op (every edge weighs the same), so passing such a spec changes nothing.

A `WeightSpec` plugs into three places:

**Weighted `k_hop` ordering.** Pass `weight=` and set `order_by_weight=True` to order the expanded frontier by accumulated edge weight (strongest paths first). The set of nodes reached is the same as the unweighted hop; only the order changes.

```python
result = (
    client.graph.query("docs_graph")
    .from_node(42)
    .k_hop("CITES", max=3, weight=w, order_by_weight=True)
    .return_nodes()
)
```

**Weighted `shortest_path`.** Set `weighted=True` and pass `weight=` so the path cost is driven by edge weight instead of plain hop count. Stronger edges cost less, so the path that wins is the one made of the most trustworthy links, not simply the one with the fewest hops.

```python
result = (
    client.graph.query("docs_graph")
    .from_node(paper_a)
    .shortest_path(["CITES"], target=paper_b, weighted=True, weight=w)
    .return_paths()
)
```

**Weighted hybrid RRF ranking.** Pass `edge_weight=` to `rank_rrf(...)` so the graph-proximity arm folds edge quality into its bridge routes, letting strong relationships pull a candidate up more than weak ones.

```python
result = (
    client.graph.query("docs_graph")
    .vector_similar(query_vector, k=10)
    .traverse("mentions", direction="outgoing")
    .k_hop("CITES", max=2)
    .traverse("mentions", direction="incoming")
    .rank_rrf(k=10, relation_edge_types=["CITES"], edge_weight=w)
    .return_nodes()
)
```

In every case, omitting the spec (or passing one with all defaults) keeps the unweighted behaviour.

### 4.8 Time-filtered traversal: `as_of`, `include_unbounded`, `context`

When your edges carry validity windows (Section 3.9), you can ask the graph what it looked like at a particular instant, or under a particular context. Three keyword arguments are available on `traverse`, `k_hop`, and `shortest_path`. They are opt-in: leave them out and the traversal runs exactly as before over every edge.

```python
# 1577836800000 is some instant in unix-epoch millis.
result = (
    client.graph.query("people_graph")
    .from_node(alice_id)
    .traverse(
        "WORKS_AT",
        direction="outgoing",
        as_of=1577836800000,        # only edges valid at this instant
        include_unbounded=True,     # also keep edges that carry no window
        context="employment-v2",    # only edges in this context
    )
    .return_nodes()
)
```

The three arguments:

- `as_of`: a unix-epoch-millisecond instant. Only edges whose window covers that instant are crossed (remember `valid_until` is exclusive). Left unset, the server uses "now".
- `include_unbounded` (default `True`): whether edges that carry no validity window at all still pass the time check. With the default they do; set it to `False` to keep only edges that explicitly declare a window covering `as_of`.
- `context`: when set, only edges whose `temporal_context` matches are crossed. Left unset, context is ignored and edges of any context are eligible.

The same three arguments work identically on `k_hop` and `shortest_path`:

```python
client.graph.query("people_graph").from_node(a).k_hop(
    "WORKS_AT", max=3, as_of=1577836800000, context="employment-v2"
)
client.graph.query("people_graph").from_node(a).shortest_path(
    ["WORKS_AT"], target=b, as_of=1577836800000
)
```

If you pass none of these (the default), the traversal is byte-identical to a non-temporal query and sees every edge.

### 4.9 Start from a filtered scan: `scan_by_filter`

Sometimes you do not have a starting vector or a node id, you just want to begin from "every node that looks like this" and traverse from there. `scan_by_filter` is a source step that produces the initial set by scanning the graph for matching nodes, all in the same query.

```python
from swarndb import Predicate

result = (
    client.graph.query("docs_graph")
    .scan_by_filter(
        kind="entity",                          # "content" or "entity"
        label="Company",                        # entity label to match
        predicate=Predicate.eq("sector", "fintech"),  # a property condition
    )
    .traverse("EMPLOYS", direction="outgoing")
    .return_nodes()
)
```

Signature:

```python
scan_by_filter(*, kind=None, label="", predicate=None)
```

Each part is optional. `kind` narrows to content or entity nodes, `label` narrows entity nodes to one label, and `predicate` applies any property condition (including the structural incident-edge count from Section 4.11). With all three left out, the scan yields every node. After the scan you chain the normal traversal, filter, and ranking steps just as you would after any other source step.

For attribute or condition-constrained retrieval, filter-then-search is the recommended pattern: follow `scan_by_filter` with `vector_rank(query_vector, k)`. The scan fixes the candidate set to only the nodes that satisfy the condition, so the ranking returns the correct top-k among exactly those nodes, rather than ranking the whole collection and hoping the matches surface.

### 4.10 Filtered graph reads: nodes and edges by filter

When you simply want to list parts of the graph, with no vector input, there are paged read calls on `client.graph`. They walk the graph in id order, a page at a time, narrowing by label, kind, edge type, and property conditions.

```python
from swarndb import Predicate

# One page of entity nodes labelled "Company" in the fintech sector.
page = client.graph.enumerate_nodes(
    "docs_graph",
    kind="entity",
    label="Company",
    predicate=Predicate.eq("sector", "fintech"),
    limit=500,
)
for node in page.nodes:
    print(node.id, node.label, node.properties)

# To get the next page, pass the returned cursor back as after_id.
if page.has_more:
    next_page = client.graph.enumerate_nodes(
        "docs_graph", kind="entity", label="Company",
        after_id=page.next_cursor, limit=500,
    )
```

Signature:

```python
enumerate_nodes(
    collection, *,
    after_id=0, limit=1000, kind=None, label="", predicate=None,
) -> NodePage
```

A `NodePage` carries `nodes`, `next_cursor` (pass it back as `after_id` to fetch the next page; `0` when exhausted), and `has_more`. `after_id` starts at `0` for the first page.

Edges read the same way, with a filter on edge type, edge properties, and the endpoint nodes:

```python
page = client.graph.enumerate_edges(
    "docs_graph",
    edge_type="CITES",
    predicate=Predicate.ge("confidence", 0.9),   # over edge properties
    endpoint_label="Paper",                       # an endpoint node's label
    endpoint_kind="content",                      # an endpoint node's kind
    limit=500,
)
for edge in page.edges:
    print(edge.id, edge.source, edge.target, edge.edge_type)
```

Signature:

```python
enumerate_edges(
    collection, *,
    after_id=0, limit=1000, edge_type="", predicate=None,
    endpoint_label="", endpoint_kind=None,
) -> EdgePage
```

An `EdgePage` mirrors `NodePage` with `edges`, `next_cursor`, and `has_more`. An edge passes the `endpoint_label` / `endpoint_kind` filter when either of its endpoints (source or target) matches. The page size you ask for is clamped by the server.

For convenience, `iter_nodes` and `iter_edges` walk every page for you and yield one item at a time, so you can iterate the whole graph without managing the cursor:

```python
for node in client.graph.iter_nodes("docs_graph", kind="entity", label="Company"):
    print(node.id, node.label)

for edge in client.graph.iter_edges("docs_graph", edge_type="CITES"):
    print(edge.source, "->", edge.target)
```

Signatures:

```python
iter_nodes(collection, *, page_size=1000, kind=None, label="") -> Iterator[TypedNode]
iter_edges(collection, *, page_size=1000, edge_type="") -> Iterator[TypedEdge]
```

### 4.11 Filter by how connected a node is: `Predicate.incident_edges`

Beyond matching on properties and labels, you can filter nodes by a structural fact: how many edges touch them. `Predicate.incident_edges` compares the count of a node's incident edges against a number, optionally narrowed to one edge type and direction. Use it anywhere a predicate is accepted (`filter`, the `k_hop` gate, `scan_by_filter`, and the filtered reads above).

```python
from swarndb import Predicate
from swarndb._proto import graph_pb2

# Nodes with at least three outgoing CITES edges (well-cited content).
well_cited = Predicate.incident_edges(
    graph_pb2.HYBRID_CMP_GE, 3,
    edge_type="CITES",
    direction="outgoing",
)

result = (
    client.graph.query("docs_graph")
    .scan_by_filter(kind="content")
    .filter(well_cited)
    .return_nodes()
)
```

Signature:

```python
Predicate.incident_edges(op, value, *, edge_type=None, direction="outgoing")
```

`op` is one of the `graph_pb2.HYBRID_CMP_*` comparison constants (the same set the other comparison predicates use, for example `HYBRID_CMP_EQ`, `HYBRID_CMP_GE`, `HYBRID_CMP_LT`), and `value` is the count to compare against. `edge_type` left as `None` counts edges of any type, and `direction` is one of `"outgoing"`, `"incoming"`, or `"both"`. The count is resolved against the graph store, so this predicate needs the typed graph to evaluate.

### 4.12 Vector math over a graph-built frontier

The vector-rank step in Section 4.5 ranks a graph-scoped set by plain similarity. There is a richer family of vector operations that work the same way: first you scope a set of nodes with the graph (any chain of source and traversal steps), then you apply one vector operation exactly over that set. Because the graph has already fixed the candidate set, each operation runs exactly over those nodes, not an approximate search.

There are six builder methods, each a terminal-style step you place after the graph scoping and before `return_nodes()`:

- `analogy_rank(a, b, c, k, *, on_missing="skip")`: rank the frontier by closeness to the analogy point `a - b + c`, keeping the top `k`. This answers "a is to b as c is to what?" over exactly the scoped nodes.
- `diversity_rank(query, lambda_, k, *, on_missing="skip")`: pick up to `k` nodes that are both relevant to `query` and varied among themselves, using Maximal Marginal Relevance. `lambda_` in [0, 1] trades relevance against diversity. Results come back in selection order.
- `cone_filter(direction, aperture_radians, k, *, on_missing="skip")`: keep only the nodes that point within `aperture_radians` of the `direction` vector (a cone around a direction in vector space), ordered by how tightly they align, capped at `k`.
- `isolation_rank(centroids, k, *, on_missing="skip")`: rank nodes by how far they sit from a set of reference points (`centroids`, a list of vectors), surfacing the most isolated, off-to-the-side nodes first. Keeps the top `k`.
- `centroid_rank(k, *, on_missing="skip")`: compute the average of the frontier's own vectors, then rank the frontier by closeness to that average, surfacing the most representative nodes first. Keeps the top `k`.
- `interpolate_rank(a, b, t, k, *, on_missing="skip")`: rank the frontier by closeness to a point interpolated between vectors `a` and `b` at fraction `t` in [0, 1], keeping the top `k`. This finds nodes that sit "partway between" two anchors.

A short example, scoping with the graph and then ranking the scoped set by analogy:

```python
result = (
    client.graph.query("words_graph")
    .from_node(seed_id)
    .k_hop("RELATED", max=2)
    .analogy_rank(vec_king, vec_man, vec_woman, k=10)
    .return_nodes()
)
for node in result.nodes:
    print(node.id, node.label)
```

**Vectorless nodes and `on_missing`.** A frontier node that carries no vector cannot take part in a vector operation. The `on_missing` argument decides what happens: `"skip"` (the default) drops those nodes from the operation and counts them, while `"error"` fails the query if any scoped node has no vector. Use `"skip"` when a mix of vectored and vectorless nodes is expected, and `"error"` when every node is supposed to have a vector and you want to catch it if one does not.

**Which similarity is used.** `diversity_rank` and `cone_filter` use an internal cosine similarity, because that is inherent to how Maximal Marginal Relevance scores and how a cone is defined. The other four (`analogy_rank`, `centroid_rank`, `interpolate_rank`, and `isolation_rank`) rank by the collection's configured distance metric, so they line up with how that collection measures nearness everywhere else.

**A note on scope size.** Because each operation runs exactly over the scoped frontier, the frontier must stay within a configured size cap. If a graph scope produces a frontier larger than the cap, the query is rejected with a clear error rather than silently dropping nodes, so you scope the graph more tightly before the vector operation (or, if you must, raise the cap on the server via `SWARNDB_MAX_FRONTIER_FOR_RANK`). The same cap applies to `vector_rank`.

---

## 5. LLM extraction (hybrid mode)

`client.extraction` turns text chunks into typed entities and edges using your own LLM. Every method works only on `hybrid` collections; the server rejects extraction calls elsewhere. To store api keys at rest, the server needs `SWARNDB_MASTER_KEY` set. See [LLM Extraction](llm-extraction.md) for the concept and the ontology templates.

The LLM is OpenAI-compatible, so any provider that speaks that protocol works (OpenAI directly, OpenRouter, a self-hosted endpoint, and so on).

### 5.1 LLM config: `set_llm_config` / `get_llm_config` / `rotate_llm_config`

```python
client.extraction.set_llm_config(
    "docs_graph",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",            # write-only; never read back
    model_name="openai/gpt-4o-mini",
    temperature=0.0,
    max_tokens=2048,
    timeout_seconds=30,
)

info = client.extraction.get_llm_config("docs_graph")
print(info.base_url, info.model_name, info.api_key_set)   # api_key_set is a bool

client.extraction.rotate_llm_config("docs_graph", new_api_key="sk-or-...")
```

Signatures:

```python
set_llm_config(
    collection, *,
    base_url, api_key, model_name,
    temperature=0.0, max_tokens=2048, timeout_seconds=30,
) -> bool

get_llm_config(collection) -> LlmConfigInfo
rotate_llm_config(collection, new_api_key) -> bool
```

`get_llm_config` returns an `LlmConfigInfo` with `base_url`, `model_name`, `temperature`, `max_tokens`, `timeout_seconds`, and `api_key_set`. It never returns the key itself; `api_key_set` just tells you whether one is stored. `rotate_llm_config` swaps only the key and leaves the rest of the config untouched.

### 5.2 Ontology: `set_ontology` / `get_ontology`

The ontology defines which entity labels and edge types the extractor is allowed to produce. You can start from a named template, extend it with your own labels and edge types, and tune the prompt.

```python
from swarndb import EntityLabel, EdgeType

client.extraction.set_ontology(
    "docs_graph",
    base_template="research-papers",       # a kebab-case template name
    entity_labels=[EntityLabel(label="Dataset", description="A named dataset")],
    edge_types=[
        EdgeType(
            edge_type="USES_DATASET",
            description="Paper uses a dataset",
            source_labels=["Paper"],
            target_labels=["Dataset"],
        )
    ],
    replace=False,                          # True replaces the template entirely
    system_prompt="You are a research-paper analyst extracting a citation graph.",
    extra_guidance="Treat 'we' as the authors of the current paper.",
)

ontology = client.extraction.get_ontology("docs_graph")
for lbl in ontology.entity_labels:
    print(lbl.label, lbl.description)
for et in ontology.edge_types:
    print(et.edge_type, et.source_labels, "->", et.target_labels)
print(ontology.system_prompt)    # your override, or None for the default
print(ontology.extra_guidance)   # your hint, or None
```

Signatures:

```python
set_ontology(
    collection, *,
    base_template=None, entity_labels=None, edge_types=None,
    replace=False, system_prompt=None, extra_guidance=None,
) -> bool

get_ontology(collection) -> OntologyInfo
```

An `OntologyInfo` carries `entity_labels` (a list of `EntityLabel`, each with `label` and `description`), `edge_types` (a list of `EdgeType`, each with `edge_type`, `description`, `source_labels`, and `target_labels`), and the two prompt fields `system_prompt` and `extra_guidance`, each `None` when left at the default.

When `replace=False` (the default), your `entity_labels` and `edge_types` extend the template. When `replace=True`, the extension replaces the template entirely.

#### Customising the extraction prompt

Two optional knobs shape the prompt the model sees, per collection:

- `system_prompt` fully overrides SwarnDB's generic task framing with your own. Leave it unset (or empty) to keep the built-in framing.
- `extra_guidance` is a short domain hint appended on top of whichever framing is in effect (the default one, or your `system_prompt`). Use it to teach the model things it cannot infer from the text alone.

In every case SwarnDB still enforces the machine contract: your ontology's allowed labels and edge types, the JSON output schema, and a fixed contract footer (output only the JSON object, stay within the allowed types or propose a new one, cite the span and a confidence, do not invent) are always part of the prompt. A custom prompt can shape the task but cannot break parsing or your ontology. Changing either value recomputes the extraction cache, so the next run re-extracts under the new prompt.

#### Ontology validation on label violations

The ontology is a contract the server enforces. An edge type with `source_labels` and `target_labels` constrains which node labels its endpoints may carry; an extracted or imported edge whose endpoints violate those label constraints is rejected rather than written. During extraction, the model is told the allowed labels and edge types and is asked to either stay within them or propose a new one (which surfaces as an ontology proposal, Section 5.5) rather than silently inventing types. This keeps the graph's schema honest: what lands in the graph always conforms to the ontology you set.

### 5.3 Cost preview: `cost_preview`

Before spending tokens, estimate the cost.

```python
from swarndb import Chunk

chunks = [
    Chunk(doc_id="paper-1", chunk_id=0, text="...", embedding=vec_0),
    Chunk(doc_id="paper-1", chunk_id=1, text="...", embedding=vec_1),
]
estimate = client.extraction.cost_preview("docs_graph", chunks)
print(estimate.chunks, estimate.estimated_input_tokens, estimate.estimated_output_tokens)
print(estimate.estimated_cost_usd, estimate.model, estimate.pricing_known)
```

Signature:

```python
cost_preview(collection, chunks) -> CostEstimate
```

A `Chunk` carries `doc_id`, `chunk_id`, `text`, and an optional `embedding` (used only for entity dedup when present). You may pass a `Chunk` or a plain dict with the same keys. A `CostEstimate` carries `chunks`, `estimated_input_tokens`, `estimated_output_tokens`, `estimated_cost_usd`, `model`, and `pricing_known` (which is `False` when the model's pricing is unknown to the server, in which case the dollar figure is best-effort).

### 5.4 Run an extraction: `start_extraction` / `extraction_status` / `cancel_extraction`

Extraction is asynchronous on the server: you start a job, then poll its status.

```python
job_id = client.extraction.start_extraction("docs_graph", chunks)

job = client.extraction.extraction_status("docs_graph", job_id)
print(job.state, job.processed_chunks, "/", job.total_chunks)
print(job.entities_written, job.edges_written, job.cache_hits, job.cache_misses)

# To stop a running job early:
client.extraction.cancel_extraction("docs_graph", job_id)
```

Signatures:

```python
start_extraction(collection, chunks) -> str          # returns the job id
extraction_status(collection, job_id) -> ExtractionJob
cancel_extraction(collection, job_id) -> bool
```

An `ExtractionJob` carries `job_id`, `collection`, `state`, `total_chunks`, `processed_chunks`, `entities_written`, `edges_written`, `cache_hits`, `cache_misses`, `error`, `failed_chunks`, and `chunk_errors`. The cache counters reflect SwarnDB's per-chunk extraction cache: an unchanged chunk under an unchanged prompt is a cache hit and costs nothing.

#### Job state and partial success

`state` is one of `"queued"`, `"running"`, `"completed"`, `"completed_with_errors"`, `"failed"`, or `"cancelled"`. A single chunk failing does not fail the whole job: it finishes in `"completed_with_errors"`, keeps everything that succeeded, and records what failed on two fields. `failed_chunks` is the true total number of failed chunks; `chunk_errors` is a sample (up to 100) of `ChunkError` entries, each with `doc_id`, `chunk_id`, and `error`.

```python
job = client.extraction.extraction_status("docs_graph", job_id)

if job.state == "completed_with_errors":
    print(f"{job.failed_chunks} chunk(s) failed; sample of {len(job.chunk_errors)}:")
    for e in job.chunk_errors:
        print(f"  doc={e.doc_id} chunk={e.chunk_id}: {e.error}")
```

`ChunkError` is importable from `swarndb` alongside the other extraction types.

#### Truncation auto-retry

When a model reply is cut off at the model's output-token limit, the chunk is retried once automatically with a higher token budget, so most truncations are recovered and never reach `chunk_errors`. Each such retry increments the server metric `swarndb_extraction_truncation_retries_total`, which you can watch on the server's metrics endpoint to see how often truncation is happening and whether you should raise `max_tokens` in the LLM config.

### 5.5 Ontology proposals: `list_proposals` / `approve_proposal` / `reject_proposal`

When the model encounters something that does not fit the current ontology, it can propose a new entity label or edge type instead of inventing one silently. Proposals wait for your review.

```python
for p in client.extraction.list_proposals("docs_graph"):
    print(p.id, p.kind, p.name, p.description, p.status)
    print("  seen in", p.source_doc, "chunk", p.source_chunk_id, "examples:", p.examples)

client.extraction.approve_proposal("docs_graph", p.id)   # adds it to the ontology
client.extraction.reject_proposal("docs_graph", p.id)    # discards it
```

Signatures:

```python
list_proposals(collection) -> list[OntologyProposal]
approve_proposal(collection, proposal_id) -> bool
reject_proposal(collection, proposal_id) -> bool
```

An `OntologyProposal` carries `id`, `kind` (`"entity_label"` or `"edge_type"`), `name`, `description`, `examples`, `status` (`"pending"`, `"approved"`, or `"rejected"`), `source_doc`, and `source_chunk_id`. Approving a proposal adds it to the ontology so future extractions may use it.

### 5.6 Incremental updates: `diff_document` / `reextract_document`

When a document changes, you do not need to re-extract the whole thing. Diff it first to see what changed, then re-extract, which processes only the changed chunks.

```python
diffs = client.extraction.diff_document("docs_graph", "paper-1", new_chunks)
for d in diffs:
    print(d.chunk_id, d.action)    # "unchanged" | "changed" | "new" | "deleted"

summary = client.extraction.reextract_document("docs_graph", "paper-1", new_chunks)
print(summary.job_id)
print(summary.unchanged, summary.changed, summary.added, summary.deleted)
print(summary.edges_deleted, summary.nodes_deleted)
```

Signatures:

```python
diff_document(collection, doc_id, chunks) -> list[ChunkDiff]
reextract_document(collection, doc_id, chunks) -> ReextractSummary
```

A `ChunkDiff` carries `chunk_id` and `action`. A `ReextractSummary` carries `job_id`, `unchanged`, `changed`, `added`, `deleted`, `edges_deleted`, and `nodes_deleted`.

#### Manual versus auto-extracted edge workflows

Re-extraction is safe to run repeatedly because it respects your curation. Edges that you authored by hand (`is_manual=True`) and edges you have verified are preserved across a re-extraction: re-extraction only churns the edges and nodes it owns (the auto-extracted ones) for the chunks that changed. The `edges_deleted` and `nodes_deleted` counts on the summary reflect that churn of auto-extracted structure, not your manual or verified edges. This is why the recommended pattern is: let extraction produce the first-pass graph, then curate with `verify_edge`, `reject_edge`, `put_edge`, and `update_edge`; subsequent re-extractions keep your curation intact while refreshing the rest.

---

## 6. Async patterns

Everything above has an async twin on `AsyncSwarnDBClient`. The method names and arguments are identical; you `await` each call, and the hybrid query terminal is awaitable.

```python
import asyncio
from swarndb import AsyncSwarnDBClient, Predicate, Direction, Chunk

async def main():
    async with AsyncSwarnDBClient(host="localhost", port=50051) as client:
        # Typed graph
        node_id = await client.graph.put_node("docs_graph", kind="entity", label="Person")
        node = await client.graph.get_node("docs_graph", node_id)

        # Hybrid query: build the chain, await the terminal
        result = await (
            client.graph.query("docs_graph")
            .vector_similar(query_vector, k=20)
            .traverse("AUTHORED_BY", direction=Direction.OUTGOING)
            .filter(Predicate.label_eq("Person"))
            .return_nodes()
        )
        print(len(result.nodes), "nodes")

        # Extraction
        chunks = [Chunk(doc_id="paper-1", chunk_id=0, text="...")]
        await client.extraction.cost_preview("docs_graph", chunks)
        job_id = await client.extraction.start_extraction("docs_graph", chunks)
        job = await client.extraction.extraction_status("docs_graph", job_id)
        print(job.state)

asyncio.run(main())
```

Because the async client overlaps I/O, it is the right choice when you run many extractions or queries concurrently, or when your app already lives inside `asyncio` (FastAPI, aiohttp, and the like). Build the same chains and gather them:

```python
async def concurrent_queries(client, vectors):
    tasks = [
        client.graph.query("docs_graph").vector_similar(v, k=10).return_nodes()
        for v in vectors
    ]
    results = await asyncio.gather(*tasks)
    return [r.nodes for r in results]
```

---

## 7. End-to-end: extract, curate, query

A full hybrid workflow ties the pieces together: configure the LLM, set the ontology, preview the cost, extract, curate the result, then query across vectors and graph.

```python
from swarndb import SwarnDBClient, Chunk, EntityLabel, EdgeType, Predicate, Direction

with SwarnDBClient(host="localhost", port=50051) as client:
    col = "papers_graph"
    client.collections.create(col, dimension=1536, distance_metric="cosine")
    # (the collection must be created in hybrid mode for the typed graph;
    #  see graph-first-class.md for how mode is selected.)

    # 1. LLM config and ontology
    client.extraction.set_llm_config(
        col,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-...",
        model_name="openai/gpt-4o-mini",
    )
    client.extraction.set_ontology(
        col,
        base_template="research-papers",
        edge_types=[EdgeType(edge_type="USES_DATASET",
                             source_labels=["Paper"], target_labels=["Dataset"])],
    )

    # 2. Preview and extract
    chunks = [Chunk(doc_id="paper-1", chunk_id=i, text=t, embedding=e)
              for i, (t, e) in enumerate(my_chunks)]
    print(client.extraction.cost_preview(col, chunks).estimated_cost_usd)
    job_id = client.extraction.start_extraction(col, chunks)

    # 3. Poll to completion
    job = client.extraction.extraction_status(col, job_id)
    while job.state in ("queued", "running"):
        job = client.extraction.extraction_status(col, job_id)
    if job.state == "completed_with_errors":
        for e in job.chunk_errors:
            print("failed chunk", e.chunk_id, e.error)

    # 4. Review proposals, curate edges
    for p in client.extraction.list_proposals(col):
        client.extraction.approve_proposal(col, p.id)   # or reject

    # 5. Query: similar papers that USE_DATASET something, returning the entities
    result = (
        client.graph.query(col)
        .vector_similar(query_vector, k=50)
        .traverse("USES_DATASET", direction=Direction.OUTGOING)
        .filter(Predicate.label_eq("Dataset"))
        .limit(20)
        .return_nodes()
    )
    for node in result.nodes:
        print(node.label, node.properties)
```

This is the full surface: virtual graph for similarity structure, typed nodes and edges for curated facts, the hybrid builder to query both at once, and extraction to populate the typed graph from text while keeping a human in the loop.
