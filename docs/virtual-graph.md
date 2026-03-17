# Virtual Graph Layer

SwarnDB's virtual graph layer is the feature that separates it from every other vector database. While traditional vector databases return a flat list of nearest neighbors, SwarnDB automatically builds a similarity graph on top of your vectors, enabling multi-hop discovery, relationship traversal, and graph-amplified search.

This guide explains how the virtual graph works, how to configure it, and how to use it effectively.

---

## What Is the Virtual Graph?

Most vector databases treat search as a one-shot operation: you provide a query vector, and you get back the k most similar vectors. That is useful, but it misses a fundamental insight: **relationships between vectors carry information too**.

SwarnDB's virtual graph captures these relationships automatically. When vectors are inserted, the engine computes similarity scores between neighbors and stores them as bidirectional edges. The result is a traversable network where each vector knows which other vectors are most similar to it.

This enables a class of queries that flat vector search cannot answer:

- **Multi-hop discovery**: Find vectors that are similar to vectors that are similar to your query. A 2-hop traversal from a "machine learning" document might surface "neural architecture search" papers that would never appear in a flat top-k search.
- **Relationship mapping**: Given any vector, instantly retrieve its neighborhood of related vectors, with similarity scores on every edge.
- **Graph-amplified search**: Run a fast HNSW search with a small k, then expand each result through the graph to discover additional relevant vectors. This is often faster and more comprehensive than simply increasing k.

The key distinction from external graph databases: SwarnDB's graph is **computed directly from the vector index structure**. There is no separate graph storage, no ETL pipeline, and no manual relationship creation. The graph lives within the index and stays in sync automatically.

---

## How It Works

The virtual graph is built incrementally as vectors are inserted into a collection. Here is the process:

1. **Vector insertion**: When a new vector is added to the HNSW index, the index identifies its nearest neighbors as part of the normal insertion algorithm.

2. **Edge computation**: SwarnDB takes those nearest-neighbor connections and computes the similarity between the new vector and each neighbor using the collection's configured distance metric (cosine, euclidean, dot_product, or manhattan).

3. **Threshold filtering**: Only edges where the similarity meets or exceeds the collection's threshold are stored. This ensures the graph contains only meaningful relationships.

4. **Bidirectional storage**: Every edge is stored in both directions. If Vector A is connected to Vector B with similarity 0.92, then Vector B is also connected to Vector A with the same score.

5. **Edge replacement**: Each node has a maximum number of edges (configurable via `max_edges_per_node`, default 100). When a node is full, a new edge can only replace an existing one if it has a higher similarity score. This keeps the graph lean and high-quality.

Because the graph piggybacks on the HNSW index structure, it adds minimal overhead to insertion. The nearest-neighbor search that HNSW already performs provides the candidate edges; the graph layer simply filters and stores them.

---

## Similarity Thresholds

Thresholds control which edges are visible in the graph. A higher threshold means fewer, stronger edges. A lower threshold means more edges, including weaker ones.

### Three Levels of Precedence

SwarnDB supports thresholds at three levels, evaluated in this order:

| Priority | Level | Description |
|----------|-------|-------------|
| Highest | Per-vector | Override threshold for a specific vector |
| Middle | Per-query | Passed at query time (graph or search request) |
| Lowest | Per-collection | Default set when the collection is created |

The engine checks for a per-vector threshold first. If none is set, it falls back to the per-query threshold. If that is also absent or zero, it uses the collection default.

### Setting Thresholds

**At collection creation** (REST):

```bash
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "articles",
    "dimension": 384,
    "distance_metric": "cosine",
    "default_threshold": 0.75
  }'
```

**Updating the collection-level threshold** (REST):

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/graph/threshold \
  -H "Content-Type: application/json" \
  -d '{
    "vector_id": 0,
    "threshold": 0.8
  }'
```

Setting `vector_id` to `0` updates the collection-level default. Any non-zero `vector_id` sets a per-vector override.

**Setting a per-vector threshold** (REST):

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/graph/threshold \
  -H "Content-Type: application/json" \
  -d '{
    "vector_id": 42,
    "threshold": 0.9
  }'
```

**Python SDK**:

```python
from swarndb import SwarnDBClient

client = SwarnDBClient("localhost:50051")

# Set collection-level threshold (vector_id=0)
client.graph.set_threshold("articles", 0.8, vector_id=0)

# Set per-vector threshold
client.graph.set_threshold("articles", 0.9, vector_id=42)
```

### Practical Guidance

| Threshold | Behavior | Best For |
|-----------|----------|----------|
| 0.6 to 0.7 | Loose. Many edges, broad connections | Exploratory analysis, sparse data |
| 0.75 to 0.85 | Balanced. Good coverage with meaningful edges | Most production workloads |
| 0.85 to 0.95 | Strict. Only strong relationships | Precision-critical applications |

> **Note**: After changing a collection-level threshold, call `optimize()` to rebuild the graph with the new threshold. Per-vector thresholds take effect immediately for read queries.

---

## Getting Related Vectors

The simplest graph operation: given a vector ID, retrieve its direct neighbors in the graph.

### REST API

**Endpoint**: `GET /api/v1/collections/{collection}/graph/related/{id}`

**Query parameters**:
- `threshold` (float, optional): Minimum similarity for returned edges. Overrides the collection default for this query.
- `max_results` (integer, optional): Maximum number of edges to return.

```bash
curl "http://localhost:8080/api/v1/collections/articles/graph/related/42?threshold=0.8&max_results=5"
```

**Response**:

```json
{
  "edges": [
    { "target_id": 17, "similarity": 0.94 },
    { "target_id": 83, "similarity": 0.91 },
    { "target_id": 5, "similarity": 0.87 },
    { "target_id": 201, "similarity": 0.85 },
    { "target_id": 64, "similarity": 0.82 }
  ]
}
```

### Python SDK

```python
edges = client.graph.get_related(
    "articles",
    vector_id=42,
    threshold=0.8,
    max_results=5,
)

for edge in edges:
    print(f"Vector {edge.target_id}: similarity {edge.similarity:.3f}")
```

### Understanding the Response

Each edge in the response contains:

- **target_id**: The ID of the related vector.
- **similarity**: The similarity score between the source vector and this neighbor, computed using the collection's configured distance metric. For cosine, values range from 0.0 to 1.0; for other metrics, the range and interpretation vary accordingly.

Results are ordered by similarity, highest first. The source vector itself is never included in the results.

---

## Graph Traversal

While `get_related` returns direct neighbors (1-hop), graph traversal lets you explore deeper into the network. Starting from any vector, you can walk the graph to discover vectors that are 2, 3, or more hops away.

### How Depth Works

- **depth=1**: Returns direct neighbors only (equivalent to `get_related`).
- **depth=2**: Returns direct neighbors, plus the neighbors of those neighbors.
- **depth=3**: Adds one more layer of expansion.

Each additional depth level expands the discovery radius exponentially, so `max_results` becomes important for controlling response size.

### Path Tracking

Every traversal result includes the full path from the start vector to the destination. This lets you understand *how* two vectors are connected, not just *that* they are connected.

**Path similarity** is the cumulative similarity along the traversal path. For a path A to B to C, the path similarity is `similarity(A,B) * similarity(B,C)`. This naturally penalizes longer, weaker paths.

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/graph/traverse`

**Request body**:

```json
{
  "start_id": 42,
  "depth": 2,
  "threshold": 0.75,
  "max_results": 20
}
```

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/graph/traverse \
  -H "Content-Type: application/json" \
  -d '{
    "start_id": 42,
    "depth": 2,
    "threshold": 0.75,
    "max_results": 20
  }'
```

**Response**:

```json
{
  "nodes": [
    {
      "id": 17,
      "depth": 1,
      "path_similarity": 0.94,
      "path": [42, 17]
    },
    {
      "id": 83,
      "depth": 1,
      "path_similarity": 0.91,
      "path": [42, 83]
    },
    {
      "id": 112,
      "depth": 2,
      "path_similarity": 0.86,
      "path": [42, 17, 112]
    },
    {
      "id": 205,
      "depth": 2,
      "path_similarity": 0.81,
      "path": [42, 83, 205]
    }
  ]
}
```

### Python SDK

```python
nodes = client.graph.traverse(
    "articles",
    start_id=42,
    depth=2,
    threshold=0.75,
    max_results=20,
)

for node in nodes:
    hop_path = " -> ".join(str(v) for v in node.path)
    print(f"Vector {node.id} (depth {node.depth}, "
          f"path_sim {node.path_similarity:.3f}): {hop_path}")
```

### Traversal Strategy

SwarnDB uses breadth-first search (BFS) by default, which explores all neighbors at depth N before moving to depth N+1. This ensures that closer, higher-similarity vectors are discovered first.

At each hop, the threshold filter is applied: only edges meeting the threshold are followed. This prevents the traversal from wandering into weakly related territory.

### Use Cases

- **Topic exploration**: Start from a known document and traverse to discover the broader topic cluster, including subtopics you might not have thought to search for.
- **Recommendation chains**: "Users who liked X also liked Y, and people who liked Y also liked Z." A depth-2 traversal naturally produces these chains.
- **Knowledge graph navigation**: In a collection of concept embeddings, traverse from one concept to discover related concepts, creating a navigable knowledge map.

---

## Graph-Enriched Search

Graph-enriched search combines traditional vector search with graph context. Instead of returning bare search results, each result includes its graph neighbors as additional context.

This is the recommended way to use the virtual graph in production. It gives you the speed of HNSW search with the discovery power of graph traversal.

### How It Works

1. SwarnDB runs a standard HNSW search to find the top-k results.
2. For each result, it looks up the vector's graph edges.
3. The edges are included in the response alongside the search scores.

This adds minimal latency (graph edge lookups are O(1) per vector) while significantly enriching the results.

### REST API

**Endpoint**: `POST /api/v1/collections/{collection}/search`

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, 0.4],
    "k": 5,
    "include_graph": true,
    "graph_threshold": 0.8,
    "max_graph_edges": 10
  }'
```

**Response**:

```json
{
  "results": [
    {
      "id": 42,
      "score": 0.05,
      "graph_edges": [
        { "target_id": 17, "similarity": 0.94 },
        { "target_id": 83, "similarity": 0.91 }
      ]
    },
    {
      "id": 17,
      "score": 0.08,
      "graph_edges": [
        { "target_id": 42, "similarity": 0.94 },
        { "target_id": 112, "similarity": 0.88 }
      ]
    }
  ],
  "search_time_us": 245
}
```

### Python SDK

```python
from swarndb import SwarnDBClient
from swarndb.search import Filter

client = SwarnDBClient("localhost:50051")

results = client.search.search(
    "articles",
    query=[0.1, 0.2, 0.3, 0.4],
    k=5,
    include_graph=True,
    graph_threshold=0.8,
    max_graph_edges=10,
)

for result in results.results:
    print(f"Vector {result.id} (score: {result.score:.4f})")
    for edge in result.graph_edges:
        print(f"  Related: {edge.target_id} (similarity: {edge.similarity:.3f})")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_graph` | bool | false | Enable graph enrichment |
| `graph_threshold` | float | 0.0 | Minimum similarity for included edges |
| `max_graph_edges` | int | 10 | Maximum edges per result |

### When to Use

Graph-enriched search is ideal for:

- **"More like this" features**: Each search result comes pre-loaded with related items.
- **Result expansion**: If 5 search results are not enough, the graph edges provide additional candidates without running another search.
- **Relationship-aware ranking**: Use the graph edges to re-rank or cluster search results based on their interconnections.

---

## Deferred Graph Mode

For large bulk imports where insertion speed is the priority, you can defer graph computation until after all vectors are loaded.

### How It Works

1. During bulk insert, set `defer_graph: true`. Vectors are inserted into the HNSW index but graph edges are not computed.
2. The collection status changes to `"pending_optimization"` to indicate the graph is stale.
3. After loading is complete, call the `optimize` endpoint to rebuild the graph.
4. The collection status returns to `"ready"` once optimization finishes.

### REST API

**Bulk insert with deferred graph**:

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/vectors/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      { "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"title": "Doc A"} },
      { "values": [0.5, 0.6, 0.7, 0.8], "metadata": {"title": "Doc B"} }
    ],
    "defer_graph": true
  }'
```

**Trigger graph rebuild**:

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/optimize
```

### Python SDK

```python
# Bulk insert with deferred graph
client.vectors.bulk_insert(
    "articles",
    vectors=vectors_list,
    defer_graph=True,
)

# Rebuild the graph after loading
client.collections.optimize("articles")
```

### When to Use

- **Initial data loading**: When populating a new collection with thousands or millions of vectors, deferring the graph and building it once at the end is significantly faster than computing edges on every insert.
- **Batch updates**: When replacing a large portion of a collection's data, defer graph computation for the batch and rebuild afterward.
- **Threshold changes**: After changing the collection-level threshold, call `optimize()` to rebuild the graph with the new threshold.

### Performance Impact

In benchmarks with 1 million vectors (1536 dimensions), deferred graph mode reduces bulk insert time by approximately 30 to 40%, depending on the threshold and edge density. The one-time `optimize()` call typically completes in seconds for collections under 1M vectors.

---

## Use Cases

### Recommendation Systems

Traditional: Search for items similar to what a user liked.
With virtual graph: Traverse from a liked item to find second-degree recommendations. "Users who liked X also liked Y, and people who liked Y also liked Z."

```python
# User liked product 42. Find 2-hop recommendations.
recs = client.graph.traverse(
    "products", start_id=42, depth=2, threshold=0.8, max_results=20
)
# Depth-1 results are direct alternatives.
# Depth-2 results are discovery recommendations.
```

### Knowledge Exploration

Build a collection of concept embeddings (from Wikipedia articles, research papers, or internal docs). Traverse the graph to create navigable topic maps.

```python
# Start from "quantum computing" and explore related concepts
nodes = client.graph.traverse(
    "concepts", start_id=concept_id, depth=3, threshold=0.7, max_results=50
)
# Discover connections: quantum computing -> quantum entanglement ->
#   quantum cryptography -> post-quantum algorithms
```

### Content Networks

Map relationships between articles, products, or documents. The graph reveals clusters and bridges between content areas.

```python
# Find the content network around an article
edges = client.graph.get_related("articles", vector_id=article_id, max_results=20)

# Use graph-enriched search to find articles with rich context
results = client.search.search(
    "articles", vector=query_vector, k=10,
    include_graph=True, graph_threshold=0.8
)
```

### Anomaly Investigation

When an anomalous vector is detected (via ghost detection or monitoring), trace its graph connections to understand its relationship to normal clusters.

```python
# Investigate an anomalous vector's neighborhood
edges = client.graph.get_related("sensors", vector_id=anomaly_id, threshold=0.5)

# If the vector has few or no edges above the threshold,
# it is genuinely isolated. If it has edges, follow them
# to understand which cluster it is closest to.
nodes = client.graph.traverse(
    "sensors", start_id=anomaly_id, depth=2, threshold=0.5, max_results=10
)
```

---

## Quick Reference

| Operation | REST Endpoint | Method |
|-----------|---------------|--------|
| Get related vectors | `/api/v1/collections/{collection}/graph/related/{id}` | GET |
| Graph traversal | `/api/v1/collections/{collection}/graph/traverse` | POST |
| Set threshold | `/api/v1/collections/{collection}/graph/threshold` | POST |
| Graph-enriched search | `/api/v1/collections/{collection}/search` (with `include_graph: true`) | POST |
| Optimize (rebuild graph) | `/api/v1/collections/{collection}/optimize` | POST |
