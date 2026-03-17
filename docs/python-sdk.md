# Python SDK

The SwarnDB Python SDK provides synchronous and asynchronous clients for interacting with a SwarnDB server over gRPC. It covers collections, vectors, search, virtual graph, and vector math operations.

**Source:** [github.com/SarthiAI/swarndb](https://github.com/SarthiAI/swarndb)

**Requirements:** Python 3.9+, grpcio>=1.60.0, protobuf>=4.25.0, numpy>=1.24.0

---

## 1. Installation

Install the SDK from PyPI:

```bash
pip install swarndb
```

For async support (adds `grpcio` async extras):

```bash
pip install swarndb[async]
```

---

## 2. Quick Start

A complete working example, from connection to search:

```python
from swarndb import SwarnDBClient

with SwarnDBClient(host="localhost", port=50051) as client:
    # Create a collection
    client.collections.create("articles", dimension=128, distance_metric="cosine")

    # Insert vectors with metadata
    for i in range(10):
        client.vectors.insert(
            "articles",
            vector=[0.1 * (i + 1)] * 128,
            metadata={"category": "science", "year": 2024},
        )

    # Search for nearest neighbors
    results = client.search.query("articles", vector=[0.5] * 128, k=5)
    for r in results.results:
        print(f"ID: {r.id}, Score: {r.score:.4f}, Metadata: {r.metadata}")
```

---

## 3. Connecting

### Basic Connection

```python
from swarndb import SwarnDBClient

client = SwarnDBClient(host="localhost", port=50051)
```

### With Authentication

```python
client = SwarnDBClient(
    host="localhost",
    port=50051,
    api_key="your-api-key",
)
```

### Context Manager (recommended)

The context manager automatically closes the gRPC channel on exit:

```python
with SwarnDBClient(host="localhost", port=50051) as client:
    collections = client.collections.list()
    print(collections)
```

### Connection Options

| Parameter     | Type                | Default       | Description                                      |
|---------------|---------------------|---------------|--------------------------------------------------|
| `host`        | `str`               | `"localhost"` | Server hostname or IP address                    |
| `port`        | `int`               | `50051`       | gRPC port number                                 |
| `api_key`     | `str` or `None`     | `None`        | API key for authentication                       |
| `secure`      | `bool`              | `False`       | Use TLS/SSL encrypted channel                    |
| `max_retries` | `int`               | `3`           | Max retry attempts for transient gRPC errors     |
| `retry_delay` | `float`             | `0.5`         | Base delay in seconds between retries (exponential backoff) |
| `timeout`     | `float`             | `30.0`        | Default per-call timeout in seconds              |
| `options`     | `list[tuple]` or `None` | `None`     | Additional gRPC channel options                  |

---

## 4. Collections

Access via `client.collections`.

### Create a Collection

```python
info = client.collections.create(
    "products",
    dimension=1536,
    distance_metric="cosine",    # "cosine", "euclidean", "dot_product"
    default_threshold=0.7,
)
print(info.name, info.dimension)
```

**Signature:**

```python
create(name, dimension, *, distance_metric="cosine", default_threshold=0.0, max_vectors=0) -> CollectionInfo
```

### List All Collections

```python
for col in client.collections.list():
    print(f"{col.name}: {col.vector_count} vectors, {col.dimension}d")
```

### Get Collection Info

```python
info = client.collections.get("products")
print(f"Metric: {info.distance_metric}, Vectors: {info.vector_count}")
```

### Check If a Collection Exists

```python
if client.collections.exists("products"):
    print("Collection exists")
```

### Delete a Collection

```python
client.collections.delete("products")
```

### Optimize a Collection

After bulk inserting with `defer_graph=True` or `index_mode="deferred"`, call optimize to rebuild indexes and the virtual graph:

```python
result = client.collections.optimize("products")
print(f"Status: {result.status}, Vectors processed: {result.vectors_processed}")
print(f"Duration: {result.duration_ms}ms")
```

### Get Collection Status

```python
status = client.collections.get_status("products")
# Returns: "ready", "pending_optimization", or "optimizing"
```

---

## 5. Vectors

Access via `client.vectors`.

### Insert a Vector

```python
# Auto-assigned ID (pass id=0 or omit)
vec_id = client.vectors.insert(
    "products",
    vector=[0.1, 0.2, 0.3, ...],  # must match collection dimension
    metadata={"name": "Widget", "price": 29.99, "tags": ["sale", "new"]},
)
print(f"Inserted with ID: {vec_id}")

# Explicit ID
vec_id = client.vectors.insert(
    "products",
    vector=[0.4, 0.5, 0.6, ...],
    id=42,
    metadata={"name": "Gadget", "price": 49.99},
)
```

**Signature:**

```python
insert(collection, vector, *, metadata=None, id=0) -> int
```

### Get a Vector

```python
record = client.vectors.get("products", id=42)
print(f"ID: {record.id}")
print(f"Vector: {record.vector[:5]}...")  # first 5 values
print(f"Metadata: {record.metadata}")
```

### Update a Vector

You can update the vector values, the metadata, or both:

```python
# Update metadata only
client.vectors.update("products", id=42, metadata={"price": 39.99})

# Update vector values only
client.vectors.update("products", id=42, vector=[0.7, 0.8, 0.9, ...])

# Update both
client.vectors.update(
    "products", id=42,
    vector=[0.7, 0.8, 0.9, ...],
    metadata={"price": 39.99, "on_sale": True},
)
```

### Delete a Vector

```python
client.vectors.delete("products", id=42)
```

### Bulk Insert

For high-throughput ingestion with performance tuning options:

```python
import numpy as np

# Generate 10,000 random vectors
vectors = np.random.rand(10000, 1536).tolist()
metadata_list = [{"batch": "2024-Q1", "index": i} for i in range(10000)]

result = client.vectors.bulk_insert(
    "products",
    vectors=vectors,
    metadata_list=metadata_list,
    batch_size=1000,
    show_progress=True,          # requires tqdm
    defer_graph=True,            # skip graph during insert
    wal_flush_every=0,           # disable WAL for max speed
    index_mode="deferred",       # build index after all inserts
    parallel_build=True,         # parallel HNSW construction on optimize
)
print(f"Inserted: {result.inserted_count}, Errors: {len(result.errors)}")

# After bulk insert, rebuild indexes and graph
opt = client.collections.optimize("products")
print(f"Optimized in {opt.duration_ms}ms")
```

**Signature:**

```python
bulk_insert(
    collection, vectors, *,
    metadata_list=None, ids=None, batch_size=1000,
    show_progress=False, batch_lock_size=None,
    defer_graph=False, wal_flush_every=None,
    ef_construction=None, index_mode=None,
    skip_metadata_index=False, parallel_build=False,
) -> BulkInsertResult
```

**Bulk Insert Options:**

| Parameter            | Type          | Default  | Description                                         |
|----------------------|---------------|----------|-----------------------------------------------------|
| `metadata_list`      | `list[dict]`  | `None`   | Per-vector metadata (must match vectors length)      |
| `ids`                | `list[int]`   | `None`   | Per-vector IDs (0 for auto-assign)                   |
| `batch_size`         | `int`         | `1000`   | Vectors per streaming batch                          |
| `show_progress`      | `bool`        | `False`  | Display tqdm progress bar                            |
| `batch_lock_size`    | `int`         | `None`   | Vectors per lock acquisition (1 to 10000)            |
| `defer_graph`        | `bool`        | `False`  | Skip graph computation during insert                 |
| `wal_flush_every`    | `int`         | `None`   | WAL flush interval (0 = disable)                     |
| `ef_construction`    | `int`         | `None`   | Override HNSW ef_construction for this batch         |
| `index_mode`         | `str`         | `None`   | `"immediate"` or `"deferred"`                        |
| `skip_metadata_index`| `bool`        | `False`  | Skip metadata indexing during insert                 |
| `parallel_build`     | `bool`        | `False`  | Parallel HNSW build (requires `index_mode="deferred"`) |

### NumPy Integration

The SDK accepts NumPy arrays anywhere a `list[float]` is expected:

```python
import numpy as np

embedding = np.random.rand(1536).astype(np.float32)
vec_id = client.vectors.insert("products", vector=embedding.tolist())

query = np.random.rand(1536).astype(np.float32)
results = client.search.query("products", vector=query.tolist(), k=10)
```

---

## 6. Search

Access via `client.search`.

### Basic Search

```python
results = client.search.query("products", vector=[0.5] * 1536, k=10)

for r in results.results:
    print(f"ID: {r.id}, Score: {r.score:.4f}")
print(f"Search took {results.search_time_us}us")
```

**Signature:**

```python
query(
    collection, vector, k=10, *,
    filter=None, strategy="auto",
    include_metadata=True, include_graph=False,
    graph_threshold=0.0, max_graph_edges=10,
    ef_search=None,
) -> SearchResult
```

### Filtered Search

Use the `Filter` class to build metadata filters with Python operators:

```python
from swarndb import Filter

# Equality filter
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=Filter.eq("category", "electronics"),
)

# Range filter
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=Filter.field("price").between(10.0, 100.0),
)

# Combine with AND (& operator)
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=Filter.eq("category", "electronics") & Filter.field("price").lt(50.0),
)

# Combine with OR (| operator)
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=Filter.eq("brand", "Acme") | Filter.eq("brand", "Globex"),
)

# Negate with NOT (~ operator)
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=~Filter.eq("discontinued", True),
)

# Membership filter
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=Filter.in_("color", ["red", "blue", "green"]),
)

# Existence check
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=Filter.exists("discount_price"),
)

# Contains filter
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=Filter.contains("description", "wireless"),
)
```

**Available Filter Operations:**

| Method                            | Description                      |
|-----------------------------------|----------------------------------|
| `Filter.eq(field, value)`         | Equality: field == value         |
| `Filter.ne(field, value)`         | Not equal: field != value        |
| `Filter.gt(field, value)`         | Greater than: field > value      |
| `Filter.gte(field, value)`        | Greater than or equal            |
| `Filter.lt(field, value)`         | Less than: field < value         |
| `Filter.lte(field, value)`        | Less than or equal               |
| `Filter.in_(field, values)`       | Membership: field in values      |
| `Filter.between(field, lo, hi)`   | Range: lo <= field <= hi         |
| `Filter.exists(field)`            | Field is present                 |
| `Filter.contains(field, value)`   | Field contains value             |

**Chained syntax** is also supported via `Filter.field()`:

```python
Filter.field("price").gt(50)
Filter.field("tags").contains("sale")
Filter.field("year").between(2020, 2024)
```

**Boolean combinators:**

```python
f1 & f2          # AND
f1 | f2          # OR
~f1              # NOT
(f1 & f2) | f3   # nested logic
```

### Graph-Enriched Search

Include virtual graph edges alongside search results for relationship discovery:

```python
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    include_graph=True,
    graph_threshold=0.7,
    max_graph_edges=5,
)

for r in results.results:
    print(f"ID: {r.id}, Score: {r.score:.4f}")
    for edge in r.graph_edges:
        print(f"  Related to {edge.target_id} (similarity: {edge.similarity:.3f})")
```

### Search with ef_search Override

Tune HNSW search quality per query:

```python
results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    ef_search=200,  # higher = better recall, slower
)
```

### Filter Strategy

Control when metadata filtering is applied:

```python
# "auto" (default): engine picks the best strategy
# "pre_filter": filter before ANN search (exact, slower for low selectivity)
# "post_filter": filter after ANN search (fast, may return fewer results)

results = client.search.query(
    "products", vector=[0.5] * 1536, k=10,
    filter=Filter.eq("category", "electronics"),
    strategy="pre_filter",
)
```

### Batch Search

Search multiple queries in a single RPC call:

```python
queries = [
    [0.1] * 1536,
    [0.5] * 1536,
    [0.9] * 1536,
]

batch = client.search.batch(
    "products", queries=queries, k=5,
    filter=Filter.eq("category", "electronics"),
    include_metadata=True,
)

for i, sr in enumerate(batch.results):
    print(f"Query {i}: {len(sr.results)} results in {sr.search_time_us}us")
print(f"Total batch time: {batch.total_time_us}us")
```

---

## 7. Graph Operations

Access via `client.graph`. SwarnDB's virtual graph automatically connects similar vectors based on a similarity threshold.

### Set Collection Threshold

Set the similarity threshold that determines which vectors are connected in the graph:

```python
# Collection-level threshold
client.graph.set_threshold("products", threshold=0.75)

# Per-vector threshold override
client.graph.set_threshold("products", threshold=0.9, vector_id=42)
```

After setting a threshold, call `client.collections.optimize("products")` to rebuild the graph.

### Get Related Vectors

Find vectors connected to a given vector via the virtual graph:

```python
edges = client.graph.get_related(
    "products",
    vector_id=42,
    threshold=0.7,
    max_results=20,
)

for edge in edges:
    print(f"Related to {edge.target_id}, similarity: {edge.similarity:.3f}")
```

### Graph Traversal

Multi-hop traversal discovers vectors connected through chains of similarity:

```python
nodes = client.graph.traverse(
    "products",
    start_id=42,
    depth=3,          # max hops
    threshold=0.6,    # minimum edge similarity
    max_results=50,
)

for node in nodes:
    print(f"ID: {node.id}, Depth: {node.depth}, "
          f"Path similarity: {node.path_similarity:.3f}, "
          f"Path: {node.path}")
```

### Complete Graph Exploration Workflow

```python
from swarndb import SwarnDBClient

with SwarnDBClient(host="localhost", port=50051) as client:
    # 1. Create collection and insert data
    client.collections.create("articles", dimension=128, distance_metric="cosine")
    for i in range(100):
        client.vectors.insert(
            "articles",
            vector=[float(i % 10) / 10.0 + j * 0.01 for j in range(128)],
            metadata={"topic": f"topic_{i % 5}"},
        )

    # 2. Set threshold and rebuild graph
    client.graph.set_threshold("articles", threshold=0.8)
    client.collections.optimize("articles")

    # 3. Explore relationships
    edges = client.graph.get_related("articles", vector_id=1, max_results=10)
    print(f"Vector 1 has {len(edges)} related vectors")

    # 4. Traverse the graph
    nodes = client.graph.traverse("articles", start_id=1, depth=2, max_results=25)
    print(f"Traversal found {len(nodes)} reachable vectors within 2 hops")
```

---

## 8. Vector Math

Access via `client.math`. All operations run server-side for performance.

### Ghost Detection

Find isolated vectors that are far from any cluster centroid:

```python
ghosts = client.math.detect_ghosts(
    "products",
    threshold=5.0,     # distance threshold
    auto_k=8,          # auto-compute 8 centroids
    metric="euclidean",
)

for g in ghosts:
    print(f"Ghost vector {g.id}, isolation score: {g.isolation_score:.2f}")
```

### Cone Search

Find vectors within an angular cone around a direction:

```python
import math

results = client.math.cone_search(
    "products",
    direction=[1.0] + [0.0] * 1535,       # unit direction vector
    aperture_radians=math.pi / 6,          # 30-degree cone
)

for r in results:
    print(f"ID: {r.id}, cosine: {r.cosine_similarity:.3f}, "
          f"angle: {math.degrees(r.angle_radians):.1f} degrees")
```

### Centroid Computation

Compute the centroid of all or a subset of vectors:

```python
# Centroid of the entire collection
centroid = client.math.centroid("products")

# Centroid of specific vectors
centroid = client.math.centroid("products", vector_ids=[1, 2, 3, 4, 5])

# Weighted centroid
centroid = client.math.centroid(
    "products",
    vector_ids=[1, 2, 3],
    weights=[0.5, 0.3, 0.2],
)
```

### Interpolation

Interpolate between two vectors using linear (LERP) or spherical (SLERP) interpolation:

```python
vec_a = [1.0, 0.0, 0.0, ...]
vec_b = [0.0, 1.0, 0.0, ...]

# Single interpolation at t=0.5
midpoint = client.math.interpolate(vec_a, vec_b, t=0.5, method="slerp")

# Generate a sequence of 10 interpolated vectors
sequence = client.math.interpolate_sequence(vec_a, vec_b, n=10, method="slerp")
print(f"Generated {len(sequence)} intermediate vectors")
```

### Drift Detection

Detect distribution shift between two temporal windows of vectors:

```python
# Compare old vs new embeddings
report = client.math.detect_drift(
    "products",
    window1_ids=[1, 2, 3, 4, 5],        # baseline window
    window2_ids=[96, 97, 98, 99, 100],   # comparison window
    metric="euclidean",
    threshold=2.0,
)

print(f"Centroid shift: {report.centroid_shift:.4f}")
print(f"Spread change: {report.spread_change:.4f}")
print(f"Has drifted: {report.has_drifted}")
```

### K-Means Clustering

Run k-means clustering on collection vectors:

```python
result = client.math.cluster(
    "products",
    k=5,
    max_iterations=100,
    tolerance=1e-4,
    metric="euclidean",
)

print(f"Converged: {result.converged} in {result.iterations} iterations")
print(f"Found {len(result.centroids)} clusters")

for assignment in result.assignments[:5]:
    print(f"Vector {assignment.id} -> Cluster {assignment.cluster} "
          f"(distance: {assignment.distance_to_centroid:.3f})")
```

### PCA Dimensionality Reduction

Project high-dimensional vectors to lower dimensions:

```python
pca = client.math.reduce_dimensions(
    "products",
    n_components=2,
    vector_ids=[1, 2, 3, 4, 5],  # optional subset
)

print(f"Explained variance: {pca.explained_variance}")
for i, point in enumerate(pca.projected):
    print(f"Vector -> ({point[0]:.3f}, {point[1]:.3f})")
```

### Analogy Computation

Solve vector analogies of the form "a is to b as c is to ?":

```python
# king - man + woman = queen (conceptually)
result = client.math.analogy(
    a=king_vec,
    b=man_vec,
    c=woman_vec,
    normalize=True,
)
# Use result as a query vector to find the closest match
```

### Diversity Sampling (MMR)

Select vectors that balance relevance with diversity using Maximal Marginal Relevance:

```python
results = client.math.diversity_sample(
    "products",
    query=[0.5] * 1536,
    k=10,
    lambda_=0.7,                          # 0.7 = favor relevance, 0.3 = favor diversity
    candidate_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # optional
)

for r in results:
    print(f"ID: {r.id}, Relevance: {r.relevance_score:.3f}, MMR: {r.mmr_score:.3f}")
```

---

## 9. Async Client

The `AsyncSwarnDBClient` provides the same API surface using `async`/`await`. It uses `grpc.aio` under the hood.

### Basic Usage

```python
import asyncio
from swarndb import AsyncSwarnDBClient

async def main():
    async with AsyncSwarnDBClient(host="localhost", port=50051) as client:
        # Create collection
        await client.collections.create("articles", dimension=128)

        # Insert vectors
        for i in range(100):
            await client.vectors.insert(
                "articles",
                vector=[0.1 * (i % 10)] * 128,
                metadata={"index": i},
            )

        # Search
        results = await client.search.query("articles", [0.5] * 128, k=5)
        for r in results.results:
            print(f"ID: {r.id}, Score: {r.score:.4f}")

asyncio.run(main())
```

### Concurrent Operations

The async client excels at running multiple operations in parallel:

```python
async def concurrent_search(client):
    queries = [[0.1 * i] * 128 for i in range(10)]

    tasks = [
        client.search.query("articles", q, k=5)
        for q in queries
    ]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(f"Query {i}: {len(result.results)} results")
```

### When to Use Async vs Sync

Use the **async client** when:
- Your application already uses `asyncio` (web frameworks like FastAPI, aiohttp)
- You need to run many concurrent searches or inserts
- You want to overlap I/O with other async operations

Use the **sync client** when:
- You are writing scripts, notebooks, or batch jobs
- Your application does not use asyncio
- Simplicity is more important than concurrency

---

## 10. Error Handling

All SDK exceptions inherit from `SwarnDBError`, so you can catch any error with a single clause or handle specific cases.

### Exception Hierarchy

```text
SwarnDBError (base)
  ConnectionError          # cannot reach server
  AuthenticationError      # invalid or missing API key
  CollectionError          # base for collection issues
    CollectionNotFoundError
    CollectionExistsError
  VectorError              # base for vector issues
    VectorNotFoundError
    DimensionMismatchError
  SearchError              # search operation failure
  GraphError               # graph operation failure
  MathError                # math operation failure
```

### Catching Errors

```python
from swarndb import SwarnDBClient
from swarndb.exceptions import (
    SwarnDBError,
    ConnectionError,
    CollectionNotFoundError,
    VectorNotFoundError,
    DimensionMismatchError,
    AuthenticationError,
)

with SwarnDBClient(host="localhost", port=50051) as client:
    try:
        info = client.collections.get("nonexistent")
    except CollectionNotFoundError as e:
        print(f"Collection not found: {e.collection_name}")

    try:
        client.vectors.get("products", id=999999)
    except VectorNotFoundError as e:
        print(f"Vector missing: {e.vector_id}")

    try:
        client.vectors.insert("products", vector=[0.1, 0.2])  # wrong dimension
    except DimensionMismatchError as e:
        print(f"Expected {e.expected}d, got {e.got}d")

    try:
        client.search.query("products", [0.5] * 1536, k=10)
    except SwarnDBError as e:
        # Catch-all for any SDK error
        print(f"SwarnDB error: {e.message}")
        if e.details:
            print(f"Details: {e.details}")
```

---

## 11. Type Reference

All types are frozen dataclasses imported from `swarndb.types`.

### ScoredResult

A single search result with similarity score and optional graph edges.

| Field         | Type              | Description                          |
|---------------|-------------------|--------------------------------------|
| `id`          | `int`             | Vector ID                            |
| `score`       | `float`           | Similarity/distance score            |
| `metadata`    | `dict[str, Any]`  | Attached metadata (empty dict if not requested) |
| `graph_edges` | `list[GraphEdge]` | Related vectors via virtual graph    |

### SearchResult

Result of a single search query.

| Field            | Type                 | Description                    |
|------------------|----------------------|--------------------------------|
| `results`        | `list[ScoredResult]` | Matching vectors               |
| `search_time_us` | `int`                | Search duration in microseconds|
| `warning`        | `str`                | Optional warning message       |

### BatchSearchResult

Result of a batch search operation.

| Field          | Type                  | Description                      |
|----------------|-----------------------|----------------------------------|
| `results`      | `list[SearchResult]`  | One SearchResult per query       |
| `total_time_us`| `int`                 | Total batch duration in microseconds |

### CollectionInfo

Metadata about a collection.

| Field               | Type    | Description                         |
|---------------------|---------|-------------------------------------|
| `name`              | `str`   | Collection name                     |
| `dimension`         | `int`   | Vector dimensionality               |
| `distance_metric`   | `str`   | Distance function name              |
| `vector_count`      | `int`   | Number of stored vectors            |
| `default_threshold` | `float` | Default similarity threshold        |

### VectorRecord

A stored vector with its metadata.

| Field      | Type             | Description            |
|------------|------------------|------------------------|
| `id`       | `int`            | Vector ID              |
| `vector`   | `list[float]`    | Vector values          |
| `metadata` | `dict[str, Any]` | Attached metadata      |

### BulkInsertResult

Result of a bulk insert operation.

| Field            | Type         | Description                  |
|------------------|--------------|------------------------------|
| `inserted_count` | `int`        | Number of vectors inserted   |
| `errors`         | `list[str]`  | Error messages (if any)      |

### OptimizeResult

Result of a collection optimize operation.

| Field               | Type  | Description                       |
|---------------------|-------|-----------------------------------|
| `status`            | `str` | Operation status                  |
| `message`           | `str` | Human-readable message            |
| `duration_ms`       | `int` | Duration in milliseconds          |
| `vectors_processed` | `int` | Number of vectors processed       |

### GraphEdge

An edge in the virtual graph.

| Field        | Type    | Description              |
|--------------|---------|--------------------------|
| `target_id`  | `int`   | Connected vector ID      |
| `similarity` | `float` | Edge similarity score    |

### TraversalNode

A node visited during graph traversal.

| Field             | Type         | Description                        |
|-------------------|--------------|------------------------------------|
| `id`              | `int`        | Vector ID                          |
| `depth`           | `int`        | Hop distance from start            |
| `path_similarity` | `float`      | Cumulative similarity along path   |
| `path`            | `list[int]`  | Vector IDs along the traversal path|

### GhostVector

A vector identified as isolated.

| Field             | Type    | Description                      |
|-------------------|---------|----------------------------------|
| `id`              | `int`   | Vector ID                        |
| `isolation_score` | `float` | Distance to nearest centroid     |

### ConeSearchResult

A result from angular cone search.

| Field               | Type    | Description                  |
|---------------------|---------|------------------------------|
| `id`                | `int`   | Vector ID                    |
| `cosine_similarity` | `float` | Cosine similarity to direction |
| `angle_radians`     | `float` | Angle from cone axis         |

### DriftReport

Report from distribution drift detection.

| Field                    | Type    | Description                          |
|--------------------------|---------|--------------------------------------|
| `centroid_shift`         | `float` | Distance between window centroids    |
| `mean_distance_window1`  | `float` | Mean distance to centroid in window 1|
| `mean_distance_window2`  | `float` | Mean distance to centroid in window 2|
| `spread_change`          | `float` | Change in spread between windows     |
| `has_drifted`            | `bool`  | Whether drift exceeds the threshold  |

### ClusterResult

Result of k-means clustering.

| Field         | Type                      | Description                     |
|---------------|---------------------------|---------------------------------|
| `centroids`   | `list[list[float]]`       | Computed cluster centroids      |
| `assignments` | `list[ClusterAssignment]` | Per-vector cluster assignments  |
| `iterations`  | `int`                     | Number of iterations run        |
| `converged`   | `bool`                    | Whether k-means converged       |

### ClusterAssignment

Assignment of a vector to a cluster.

| Field                  | Type    | Description                    |
|------------------------|---------|--------------------------------|
| `id`                   | `int`   | Vector ID                      |
| `cluster`              | `int`   | Assigned cluster index         |
| `distance_to_centroid` | `float` | Distance to cluster centroid   |

### PCAResult

Result of PCA dimensionality reduction.

| Field                | Type                | Description                    |
|----------------------|---------------------|--------------------------------|
| `components`         | `list[list[float]]` | Principal component vectors    |
| `explained_variance` | `list[float]`       | Variance explained per component|
| `mean`               | `list[float]`       | Mean vector of input data      |
| `projected`          | `list[list[float]]` | Projected lower-dimensional vectors |

### DiversityResult

A result from MMR diversity sampling.

| Field             | Type    | Description                    |
|-------------------|---------|--------------------------------|
| `id`              | `int`   | Vector ID                      |
| `relevance_score` | `float` | Relevance to the query         |
| `mmr_score`       | `float` | Combined MMR score             |
