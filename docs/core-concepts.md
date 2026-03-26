# Core Concepts

This guide covers the foundational concepts behind SwarnDB: what it stores, how it indexes data, and how searches are performed. Whether you are building a recommendation engine, a semantic search pipeline, or a knowledge graph, understanding these concepts will help you get the most out of SwarnDB.

---

## 1. Vectors

A **vector** is a fixed-length array of floating-point numbers (f32) that represents a piece of data in a high-dimensional space. In practice, vectors come from embedding models (such as OpenAI, Cohere, or Sentence Transformers) that convert text, images, audio, or other content into numerical representations.

**Key properties of a vector in SwarnDB:**

- **Dimensions**: Every vector belongs to a collection, and every collection enforces a single fixed dimension. For example, OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors. All vectors in that collection must have exactly 1536 values.
- **Vector ID**: Each vector has a `uint64` identifier. You can let SwarnDB auto-assign IDs (monotonically increasing), or you can specify your own ID when inserting.
- **Values**: The raw floating-point data. Stored as `Vec<f32>` internally.

**How vectors are stored:**

```text
  Insert request
       |
       v
  +----------+     flush      +------------+     compaction     +----------+
  |   WAL    | ------------> |  Memtable   | ----------------> | Segments |
  | (append) |               | (in-memory) |                   | (on-disk)|
  +----------+               +------------+                    +----------+
```

1. Every write is first appended to the **Write-Ahead Log (WAL)** for durability.
2. The vector is then inserted into the in-memory **Memtable** for fast access.
3. When the memtable reaches a threshold, it is flushed to an immutable on-disk **Segment**.
4. Background **compaction** merges small segments into larger ones.

---

## 2. Collections

A **collection** is a named container of vectors that share the same configuration: dimension, distance metric, and optional similarity threshold for graph edges.

### Creating a collection

When you create a collection, you specify:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | Yes | Unique string identifier (e.g., `"products"`) |
| `dimension` | Yes | Number of dimensions for all vectors (e.g., `1536`) |
| `distance_metric` | Yes | How similarity is measured: `cosine`, `euclidean`, `dot_product`, or `manhattan` |
| `default_threshold` | No | Similarity threshold for virtual graph edges (0.0 to 1.0) |

### Collection status

A collection moves through these states:

```text
  "ready"  ------>  "pending_optimization"  ------>  "optimizing"  ------>  "ready"
            (bulk inserts done)              (optimize() called)      (index rebuilt)
```

- **ready**: Normal operating state. Searches and inserts work.
- **pending_optimization**: Data has been inserted but indexes may not reflect it yet.
- **optimizing**: The `optimize()` operation is running, rebuilding indexes and compacting segments.

### Collection lifecycle

1. **Create** the collection with a name, dimension, and distance metric.
2. **Insert** vectors (individually or in bulk).
3. **Optimize** after bulk inserts to rebuild the HNSW index and compact storage.
4. **Search** with query vectors, filters, and graph enrichment.

---

## 3. Distance Metrics

Distance metrics determine how SwarnDB measures the similarity (or dissimilarity) between two vectors. The choice of metric should match how your embedding model was trained.

### Cosine Similarity

Measures the angle between two vectors, ignoring magnitude. Returns a value between -1 and 1 (1 = identical direction, 0 = orthogonal, -1 = opposite).

**Best for**: Text embeddings, normalized embeddings, any case where direction matters more than magnitude.

```text
              A . B
  cosine = -----------
            |A| * |B|
```

### Euclidean Distance (L2)

Measures the straight-line distance between two points in space. Lower values mean more similar.

**Best for**: Spatial data, image features, cases where magnitude matters.

```text
  euclidean = sqrt( sum( (a_i - b_i)^2 ) )
```

### Dot Product

The raw inner product of two vectors. Higher values mean more similar. Unlike cosine, this is sensitive to vector magnitude.

**Best for**: Maximum inner product search (MIPS), unnormalized embeddings, recommendation scores.

```text
  dot_product = sum( a_i * b_i )
```

### Manhattan Distance (L1)

The sum of absolute differences along each dimension. Sometimes called "taxicab distance."

**Best for**: Sparse feature vectors, discrete or categorical embeddings, cases where outlier dimensions should not be amplified.

```text
  manhattan = sum( |a_i - b_i| )
```

### When to use which metric

| Scenario | Recommended Metric |
|----------|-------------------|
| OpenAI, Cohere, or most text embeddings | Cosine |
| Image feature vectors (CNN outputs) | Euclidean |
| Recommendation scores, MIPS | Dot Product |
| Sparse or categorical features | Manhattan |
| Unsure / general purpose | Cosine |

---

## 4. Indexing

SwarnDB uses approximate nearest neighbor (ANN) indexes to make search fast. Instead of comparing your query against every single vector (brute force), the index narrows the search space dramatically.

### a. HNSW (Hierarchical Navigable Small World)

HNSW is the default and primary index in SwarnDB. It builds a multi-layer graph where each layer is a progressively sparser version of the one below.

```text
  Layer 3:    o . . . . . . . . . . . o        (very sparse, long-range links)
              |                         |
  Layer 2:    o . . . o . . . . o . . . o       (medium density)
              |       |         |       |
  Layer 1:    o . o . o . o . . o . o . o       (denser, shorter links)
              |   |   |   |     |   |   |
  Layer 0:    o o o o o o o o o o o o o o       (all vectors, short links)
```

**How search works**: Start at the top layer with a single entry point. Greedily descend through layers, expanding the search at each level until you reach layer 0, where the final candidates are scored and ranked.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Number of connections per node. Higher M = better recall, more memory. |
| `ef_construction` | 200 | Search width during index building. Higher = better quality index, slower build. |
| `ef_search` | 50 | Search width during queries. Higher = better recall, slower search. |

**Tuning guidance:**

- For high recall (>0.99): set `ef_search` to 200 or higher.
- For low latency: keep `ef_search` between 50 and 100.
- `M = 16` is a strong default. Increase to 32 or 64 for very high-dimensional data.
- `ef_construction` only affects build time. Set it high (200 to 500) and leave it.

SwarnDB allows per-query `ef_search` overrides, so you can tune recall vs. speed on every request.

### b. IVF+PQ (Inverted File + Product Quantization)

For billion-scale datasets where memory is the constraint, SwarnDB offers IVF+PQ indexing.

**How IVF works**: The vector space is partitioned into regions (called Voronoi cells) using k-means clustering. Each region has a centroid. At query time, only the nearest clusters are searched.

```text
  +--------+--------+--------+
  |   C1   |   C2   |   C3   |  <-- Voronoi partitions
  | . . .  | . . .  | . .    |
  | . .    | . . .  | . . .  |
  +--------+--------+--------+
  |   C4   |   C5   |   C6   |
  | . . .  | . .    | . . .  |
  | .      | . . .  | . .    |
  +--------+--------+--------+
```

**How PQ works**: Each vector is split into sub-vectors, and each sub-vector is compressed into a small codebook index. This reduces memory by 10x to 50x at the cost of some recall.

```text
  Original vector (1536 floats = 6144 bytes)
       |
       v
  Split into 192 sub-vectors of 8 floats each
       |
       v
  Each sub-vector -> 1-byte codebook index
       |
       v
  Compressed vector (192 bytes, 32x smaller)
```

**Tradeoff**: IVF+PQ uses far less memory than HNSW, but recall is typically 5 to 15% lower. Use it when your dataset exceeds available RAM.

---

## 5. Metadata

Every vector can optionally carry **metadata**: a set of key-value pairs that describe the data the vector represents.

### Supported types

| Type | Example | Notes |
|------|---------|-------|
| `string` | `"category": "electronics"` | UTF-8 text |
| `int` (i64) | `"price_cents": 1999` | 64-bit signed integer |
| `float` (f64) | `"rating": 4.7` | 64-bit floating point |
| `bool` | `"in_stock": true` | Boolean |
| `string_list` | `"tags": ["sale", "new"]` | Array of strings |

### Metadata indexing

SwarnDB automatically selects the best index type for each metadata field:

| Field Type | Index Type | Use Case |
|------------|-----------|----------|
| Low cardinality (e.g., status) | **Bitmap** | Fast set membership checks |
| High cardinality (e.g., user ID) | **Hash** | Exact equality lookups |
| Numeric / ordered (e.g., price) | **B-tree** | Range queries (gt, lt, between) |

You do not need to configure metadata indexes manually. They are created and maintained automatically.

---

## 6. Filtering

Filters let you narrow search results based on metadata conditions. Filters are applied in combination with vector similarity, so you get results that are both semantically similar and match your business criteria.

### Filter operators

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equal to | `{"field": "category", "op": "eq", "value": "electronics"}` |
| `ne` | Not equal to | `{"field": "status", "op": "ne", "value": "archived"}` |
| `gt` | Greater than | `{"field": "price", "op": "gt", "value": 100}` |
| `gte` | Greater than or equal | `{"field": "rating", "op": "gte", "value": 4.0}` |
| `lt` | Less than | `{"field": "price", "op": "lt", "value": 500}` |
| `lte` | Less than or equal | `{"field": "year", "op": "lte", "value": 2024}` |
| `in` | Value in a set | `{"field": "brand", "op": "in", "value": ["Apple", "Sony"]}` |
| `between` | Within a range (inclusive) | `{"field": "price", "op": "between", "value": [100, 500]}` |
| `exists` | Field is present | `{"field": "description", "op": "exists"}` |
| `contains` | String/list contains | `{"field": "tags", "op": "contains", "value": "sale"}` |

### Logical operators

Combine conditions with `AND`, `OR`, and `NOT`:

```json
{
  "and": [
    {"field": "category", "op": "eq", "value": "electronics"},
    {"or": [
      {"field": "brand", "op": "eq", "value": "Apple"},
      {"field": "brand", "op": "eq", "value": "Sony"}
    ]},
    {"not": {"field": "status", "op": "eq", "value": "discontinued"}}
  ]
}
```

### Filter strategies

| Strategy | Behavior | Best When |
|----------|----------|-----------|
| `auto` (default) | SwarnDB picks the best strategy based on estimated selectivity | Almost always |
| `pre_filter` | Filter metadata first, then search only matching vectors | Filter is very selective (<10% of data matches) |
| `post_filter` | Search all vectors first, then remove non-matching results | Filter is very broad (>50% of data matches) |

**How `auto` works**: SwarnDB estimates what fraction of vectors will pass the filter. If the filter is selective, it pre-filters to avoid unnecessary distance computations. If the filter is broad, it post-filters to avoid missing good candidates.

---

## 7. Search

Search is the core operation: given a query vector, find the most similar vectors in a collection.

### How search works

```text
  Query vector
       |
       v
  +----------------+
  | ANN Index      |  Approximate nearest neighbor search (HNSW or IVF+PQ)
  | (ef_search=50) |
  +----------------+
       |
       v
  Candidate set (oversampled)
       |
       v
  +----------------+
  | Filter         |  Apply metadata filters (if any)
  +----------------+
       |
       v
  +----------------+
  | Score + Rank   |  Compute final scores, sort, take top-k
  +----------------+
       |
       v
  Top-k results with scores
       |
       v (optional)
  +----------------+
  | Graph Enrich   |  Add virtual graph edges to results
  +----------------+
       |
       v
  Final response
```

### Top-k results

Every search returns up to `k` results, each containing:
- **Vector ID**: The identifier of the matched vector.
- **Score**: The distance value (lower = more similar).
- **Metadata**: The vector's metadata (if requested).
- **Graph edges**: Related vectors from the virtual graph (if `include_graph` is enabled).

### Per-query ef_search tuning

You can override `ef_search` on each search request. This lets you trade recall for speed per query:

```json
{
  "query": [0.1, 0.2, ...],
  "k": 10,
  "ef_search": 200
}
```

Higher `ef_search` values explore more of the graph, improving recall at the cost of latency.

### Batch search

Send multiple query vectors in a single request. SwarnDB processes them concurrently:

```json
{
  "queries": [
    {"query": [0.1, 0.2, ...], "k": 10},
    {"query": [0.3, 0.4, ...], "k": 5}
  ]
}
```

### Graph-enriched search

When `include_graph` is enabled, each search result is enriched with edges from SwarnDB's virtual graph. This surfaces related vectors that pure vector similarity might miss, enabling multi-hop discovery and knowledge-graph-style traversal.

---

## 8. Persistence

SwarnDB is designed for durability. No data is lost on crash, restart, or unexpected shutdown.

### Write-Ahead Log (WAL)

Every mutation (insert, update, delete) is written to the WAL before it is applied to the in-memory store. The WAL is an append-only file on disk.

```text
  Client write
       |
       v
  +----------+     success     +------------+
  | WAL      | -------------> | In-memory   |
  | (fsync)  |                | store       |
  +----------+                +------------+
```

If the server crashes after the WAL write but before the in-memory update, the WAL is replayed on startup to recover the data.

### Memtable

The memtable is the in-memory buffer for recent writes. It provides fast reads for recently inserted data. When the memtable reaches a configured size threshold, it is flushed to an immutable on-disk segment.

### Segments

Segments are immutable, memory-mapped files on disk. Each segment contains a batch of vectors, their metadata, and index data. Segments are never modified after creation.

### Compaction

Over time, many small segments accumulate. Compaction merges them into fewer, larger segments. This:
- Reduces the number of files the system must manage.
- Removes tombstoned (deleted) vectors permanently.
- Improves read performance by reducing I/O scatter.

### Crash recovery

On startup, SwarnDB:
1. Loads all existing segments from disk.
2. Replays the WAL to recover any writes that were not yet flushed to segments.
3. Rebuilds in-memory indexes from the recovered data.

This ensures zero data loss for any acknowledged write.

---

## 9. Optimization

The `optimize()` operation is a manual trigger that performs a full maintenance pass on a collection.

### When to call optimize()

Call `optimize()` after completing a bulk insert batch. During streaming inserts, the system handles incremental maintenance automatically, but after large bulk loads, an explicit optimize ensures everything is up to date.

### What optimize does

1. **Rebuilds the HNSW index** from scratch with all current vectors, ensuring optimal graph quality.
2. **Computes virtual graph edges** based on the collection's similarity threshold.
3. **Compacts segments** by merging small on-disk segments into larger ones and removing deleted vectors.
4. **Flushes the WAL** to ensure all data is persisted to segments.

### Collection status during optimization

While `optimize()` runs, the collection status changes to `"optimizing"`. During this time:
- **Reads (search) continue to work** using the previous index state.
- **Writes are queued** and applied after optimization completes.
- The status returns to `"ready"` when optimization finishes.

```text
  optimize() called
       |
       v
  Status: "optimizing"
       |
       +-- Rebuild HNSW index
       +-- Compute graph edges
       +-- Compact segments
       +-- Flush WAL
       |
       v
  Status: "ready"
```
