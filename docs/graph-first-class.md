# Graph as a First-Class Layer

SwarnDB started as a vector database with an automatic similarity graph layered on top (see [Virtual Graph](virtual-graph.md)). This guide covers the next step: a collection where the graph is a first-class, typed structure you build and curate, sitting next to your vectors instead of being derived from them.

You get to choose how much graph you want. A collection picks one of three modes when it is created, and that choice decides what graph features are available. Vector-only users see no change at all. Users who want a typed graph of entities and relationships get one, with optional LLM-driven extraction on top (see [LLM Extraction](llm-extraction.md)) and a bulk way to load edges (see [Bulk Ingestion](bulk-ingestion.md)).

---

## 1. Collection modes

A collection is created in one of three modes. The mode is fixed at creation time.

| Mode | What it gives you | Pick it when |
|------|-------------------|--------------|
| `vector_only` | Pure vector search and metadata filtering. No graph at all. | You only need nearest-neighbor search and filters. |
| `auto_similarity` | The automatic similarity graph: edges appear between vectors that exceed a similarity threshold. This is the legacy behavior. | You want the "related vectors" and traversal features driven purely by similarity, with no manual curation. |
| `hybrid` | A typed graph of nodes and edges that lives alongside your vectors, plus the composable hybrid query engine and optional LLM extraction. | You want real relationships (entities, typed edges, properties, provenance) and want to combine vector search with graph traversal in one query. |

### Creating each mode

The mode is passed as a keyword when you create the collection.

REST:

```bash
# Vector-only (the default when "mode" is omitted)
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 384, "distance_metric": "cosine", "mode": "vector_only"}'

# Automatic similarity graph
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 384, "distance_metric": "cosine", "mode": "auto_similarity"}'

# First-class typed graph
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 384, "distance_metric": "cosine", "mode": "hybrid"}'
```

Python SDK:

```python
from swarndb import SwarnDBClient

with SwarnDBClient("localhost", 50051) as client:
    client.collections.create("docs", dimension=384, distance_metric="cosine", mode="vector_only")
    client.collections.create("docs_sim", dimension=384, distance_metric="cosine", mode="auto_similarity")
    client.collections.create("docs_graph", dimension=384, distance_metric="cosine", mode="hybrid")
```

When `mode` is omitted, the server defaults to vector-only.

### Backward compatibility

Existing collections keep working exactly as they did. The guarantees, in plain terms:

1. A collection created before modes existed (no mode recorded) resolves to `auto_similarity`. It keeps its similarity graph, and `get_related`, `traverse`, and the `include_graph` option on search behave as before.
2. Vector-only behavior is unchanged. If you do not ask for a graph, nothing about search, insert, or storage changes.
3. The graph RPCs and REST routes that existed before are frozen. Old clients keep calling them and get the same answers.
4. The typed-graph and extraction features are additive. They appear only on `hybrid` collections and never alter how the other two modes work.
5. Upgrading the server does not change the mode or behavior of any existing collection.

These are the high-level commitments of the upgrade contract; nothing you already run needs to change.

---

## 2. The typed graph (hybrid mode)

In a hybrid collection the graph is made of **nodes** and **typed edges** that you can read, write, and curate directly.

### Nodes

A node is one of two kinds:

- **Content node**: a node that stands for a piece of content, typically carrying an embedding. Content nodes are the bridge between vectors and the graph (see below).
- **Entity node**: a node that stands for a thing extracted or declared in your domain, such as a person, an organization, or a product. Entity nodes carry a `label` (the entity type) and a property bag.

Every node has: an `id`, a `kind` (`content` or `entity`), a `label`, a `properties` bag (free-form JSON key-values), an optional `embedding`, a `source` (`manual`, `ingested`, or `extracted`), and audit fields (`created_at`, `created_by`).

### Edges

An edge is a typed, directed link from a source node to a target node. Every edge carries:

- `edge_type`: the relationship name, for example `CITES`, `WORKS_AT`, or `BOUGHT`.
- `properties`: a free-form JSON property bag.
- `provenance`: where this edge came from (for auto-extracted edges, the source document, chunk, model, and prompt version; see [Provenance and trust](#5-provenance-and-trust)).
- `confidence`: a score from 0 to 1.
- `verified`: whether a human has confirmed this edge.
- `is_manual`: whether a human created it (versus the extractor).
- `history`: a bounded per-edge audit trail (a short list of `{action, actor, at}` records) so you can see how an edge was created, updated, verified, or rejected.

### The NodeId == VectorId bridge

The key idea that makes this hybrid and not just a graph database: **a vector and its content node share the same id**. When you insert a vector with id `42`, the content node for that vector is also id `42`. There is no separate mapping to keep in sync.

In plain terms, this means you can attach a typed edge directly to a vector. If vector `42` is a paragraph that cites the paper stored as vector `99`, you write an edge `42 --CITES--> 99` and both endpoints are the vectors you already have. A vector-similarity search returns ids, and those same ids are valid graph endpoints.

---

## 3. Manual edge management

Hybrid collections let you build and curate the graph by hand. All of these are on `client.graph`.

### Nodes

```python
# Create an entity node (returns the new node id)
person_id = client.graph.put_node(
    "docs_graph",
    kind="entity",
    label="Person",
    properties={"name": "Ada Lovelace"},
    created_by="curator@example.com",
)

node = client.graph.get_node("docs_graph", person_id)   # TypedNode or None
client.graph.delete_node("docs_graph", person_id)        # deletes the node and its incident edges
```

### Edges

```python
# Create a typed edge (returns the new edge id)
edge_id = client.graph.put_edge(
    "docs_graph",
    source=42,                 # may be an existing vector id (content node)
    target=person_id,          # or a materialized entity node
    edge_type="AUTHORED_BY",
    properties={"page": 1},
    confidence=1.0,
    verified=False,
)

edge = client.graph.get_edge("docs_graph", edge_id)      # TypedEdge or None
client.graph.delete_edge("docs_graph", edge_id)

# List edges around a node, optionally filtered by type or direction
edges = client.graph.list_edges("docs_graph", node=42, direction="outgoing", edge_type="AUTHORED_BY")
```

`direction` is one of `"outgoing"`, `"incoming"`, or `"both"`.

### Curating edges

These three operations are how you build trust in the graph over time:

```python
# Update a manual edge's properties, confidence, or verified flag.
# Only the fields you pass are changed. (Manual edges only.)
client.graph.update_edge("docs_graph", edge_id, properties={"page": 2}, confidence=0.9, actor="curator@example.com")

# Verify locks an edge: re-extraction will never replace or remove it.
client.graph.verify_edge("docs_graph", edge_id, actor="curator@example.com")

# Reject deletes the edge AND remembers the pattern, so the same edge is
# never auto-recreated from the same source again.
result = client.graph.reject_edge("docs_graph", edge_id, actor="curator@example.com")
# result.deleted, result.rule_added
```

`update_edge` applies to manual edges (the ones you created), and only the supplied fields change; omitted fields keep their value. `verify_edge` marks an edge as human-confirmed and pins it against future re-extraction. `reject_edge` removes a bad edge and records the pattern so the extractor will not bring it back from the same document and chunk.

### Bulk importing edges

To load many edges at once, use `bulk_import_edges` with CSV or JSONL. Endpoints can be existing vector ids or materialized nodes. Each row is validated and you get a per-row error report back.

```python
csv_data = """source,target,edge_type,confidence
42,99,CITES,1.0
42,100,CITES,0.8
7,99,CITES,1.0
"""

result = client.graph.bulk_import_edges("docs_graph", csv_data, format="csv", actor="importer")
print(result.total_rows, result.imported, result.failed)
for err in result.errors:
    print(f"row {err.row}: {err.message}")
```

The CSV header names the columns; `source`, `target`, and `edge_type` are required, and `confidence` plus any property columns are optional. JSONL is the same data as one JSON object per line:

```json
{"source": 42, "target": 99, "edge_type": "CITES", "confidence": 1.0}
{"source": 42, "target": 100, "edge_type": "CITES", "properties": {"section": "intro"}}
```

Pass `auto_add_edge_types=True` if you want edge types not yet in the ontology to be added automatically as you import.

---

## 4. Composable hybrid query API

Hybrid collections expose a chainable query builder that mixes vector similarity and graph traversal in a single plan. Start from `client.graph.query(collection)`, chain steps, and finish with a terminal `return_nodes()`, `return_edges()`, or `return_paths()` that runs the query.

### Steps

| Step | What it does |
|------|--------------|
| `.vector_similar(vector, k, ef_search=None)` | Seed the result set with the k nearest neighbors of a query vector. |
| `.from_node(node_id)` / `.from_nodes([...])` | Seed the result set with explicit node ids. |
| `.traverse(edge_type=None, direction="outgoing")` | Walk one hop along edges of a given type (empty type means any). |
| `.k_hop(edge_type=None, max=1, predicate=None)` | Expand up to `max` hops, optionally gated by a predicate. |
| `.shortest_path(edge_types, target)` | Find the shortest path to a target node along the given edge types. |
| `.mutual_neighbors(other_plan)` | Keep nodes reachable by both the current plan and the other plan. |
| `.intersect(other_plan)` | Keep only nodes in both the current result set and the other plan's result set. |
| `.union(other_plan)` | Combine the current result set with the other plan's result set. |
| `.filter(Predicate...)` | Keep only nodes matching a predicate. |
| `.edges(edge_type=None, direction="outgoing")` | Collect edges incident to the current nodes. |
| `.vector_rank(vector, k, on_missing="skip")` | Rank the current (graph-scoped) node frontier by similarity to a query vector, keeping the top `k`. Exact scoring over exactly those nodes (recall 1.0, no approximate search). This is the default graph-augmented ranking step. |
| `.limit(n)` | Cap the result set to `n` items. |

`direction` uses the `Direction` constants: `Direction.OUTGOING`, `Direction.INCOMING`, `Direction.BOTH` (the plain strings `"outgoing"`, `"incoming"`, `"both"` work too).

### Predicates

Filters are built with the `Predicate` helpers (imported from `swarndb`):

| Helper | Meaning |
|--------|---------|
| `Predicate.eq(key, value)` | property `==` value |
| `Predicate.ne(key, value)` | property `!=` value |
| `Predicate.gt(key, value)` | property `>` value |
| `Predicate.ge(key, value)` | property `>=` value |
| `Predicate.lt(key, value)` | property `<` value |
| `Predicate.le(key, value)` | property `<=` value |
| `Predicate.is_in(key, values)` | property is in the list |
| `Predicate.not_in(key, values)` | property is not in the list |
| `Predicate.exists(key)` | property key is present |
| `Predicate.label_eq(value)` | node label `==` value |
| `Predicate.and_(*preds)` | all sub-predicates hold |
| `Predicate.or_(*preds)` | any sub-predicate holds |
| `Predicate.not_(pred)` | negate a predicate |
| `Predicate.any_()` | always true (matches everything) |

### Examples

Find documents similar to a query, then keep only those authored by a verified person:

```python
from swarndb import SwarnDBClient, Predicate, Direction

with SwarnDBClient("localhost", 50051) as client:
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

In-catalog product recommendations: start from a product, walk "also bought" edges up to two hops, keep in-stock items:

```python
result = (
    client.graph.query("catalog")
    .from_node(product_id)
    .k_hop("ALSO_BOUGHT", max=2, predicate=Predicate.eq("in_stock", True))
    .limit(25)
    .return_nodes()
)
```

Trace a citation chain between two papers and return the path:

```python
result = (
    client.graph.query("papers")
    .from_node(paper_a)
    .shortest_path(["CITES"], target=paper_b)
    .return_paths()
)
for path in result.paths:
    print(" -> ".join(str(node_id) for node_id in path))
```

The set-combination steps (`mutual_neighbors`, `intersect`, `union`) take another plan as their argument. Build that sub-plan with `.to_plan()` (alias `.build_plan()`), which assembles a plan without running it, then pass it to the outer query. Keep only documents that are both similar to a query vector and authored by a verified person:

```python
verified = (
    client.graph.query("docs_graph")
    .vector_similar(query_vector, k=50)
    .traverse("AUTHORED_BY", direction=Direction.OUTGOING)
    .filter(Predicate.eq("verified", True))
    .to_plan()
)

result = (
    client.graph.query("docs_graph")
    .vector_similar(query_vector, k=50)
    .intersect(verified)
    .limit(10)
    .return_nodes()
)
```

A `HybridQueryResult` carries `nodes`, `edges`, and `paths`; which ones are populated depends on the terminal you call.

### Graph-augmented ranking (default: `vector_rank`)

When you want a graph-augmented result ranked for relevance, the default path is graph-first scope-then-rank: scope the candidate set with the graph, then add `.vector_rank(query_vector, k)` as the final step so the graph-scoped set is ranked exactly by similarity. The one-call `client.graph.graph_rag(...)` helper composes this for you and uses `vector_rank` by default. The older Reciprocal Rank Fusion ranking stays available as an explicit opt-in via `.rank_rrf(...)`; it is no longer the default. See the [Graph Guide](graph-guide.md) sections 4.5 and 4.6 for the full ranking story.

---

## 5. Provenance and trust

Every edge the extractor creates carries **provenance**: which document it came from (`source_doc`), which chunk (`source_chunk_id`), which model (`model`), the prompt version (`prompt_version`), and timestamps. You can read this on any edge via `get_edge` and the `provenance` field. It answers the question "why is this edge here?" without guessing.

Trust is built by the curate operations:

- An auto-extracted edge starts unverified, with whatever confidence the extractor assigned.
- `verify_edge` marks an edge as human-confirmed and locks it.
- `reject_edge` removes a bad edge and remembers the pattern so it does not come back.

The rule that ties this together: **manual edges and verified edges always survive re-extraction.** When a document is re-extracted, only the unverified auto-edges from the changed chunks are replaced; anything you created or verified stays, and anything you rejected is not recreated. This is what makes it safe to keep an extraction pipeline running on top of a curated graph.

---

## See also

- [LLM Extraction](llm-extraction.md): turn text chunks into typed entities and edges using your own LLM (hybrid mode only).
- [Bulk Ingestion](bulk-ingestion.md): load vectors at scale and control when the index is built.
- [Virtual Graph](virtual-graph.md): the automatic similarity graph used by `auto_similarity` collections.
- [API Reference](api-reference.md): the GraphService RPCs and REST routes behind these methods.
- [Python SDK](python-sdk.md): the full method-by-method SDK reference.
