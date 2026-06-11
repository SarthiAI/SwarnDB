# Bulk Ingestion and Indexing

Loading vectors and building the index that makes them searchable are two separate jobs. SwarnDB lets you decide whether they happen together or apart, which matters a lot when you are loading millions of vectors. This guide covers how to ingest at scale and when to build the index.

---

## 1. Index modes: immediate vs deferred

Bulk insert takes an `index_mode` that controls when the search index is built.

| Mode | What happens | Use it when |
|------|--------------|-------------|
| `immediate` (default) | The index is built inline as vectors are inserted. The collection is queryable as soon as the call returns. | You want the data searchable right away and the load is small to moderate. |
| `deferred` | Vectors are stored durably, but the index is not built yet. The collection is not fully queryable until you call `optimize()`. | You are loading a large batch and want the fastest possible ingest, then one index build at the end. |

With `immediate`, the index build runs off the request runtime, so the server stays responsive to other requests while the build proceeds. You are not trading away the health of the rest of the server to get an inline build.

REST:

```bash
curl -X POST http://localhost:8080/api/v1/collections/docs/vectors/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [ {"id": 1, "values": [0.1, 0.2, 0.3]}, {"id": 2, "values": [0.4, 0.5, 0.6]} ],
    "index_mode": "deferred"
  }'
```

Python SDK:

```python
# Immediate: searchable when this returns
client.vectors.bulk_insert("docs", vectors, ids=ids, index_mode="immediate")

# Deferred: store now, build the index later
client.vectors.bulk_insert("docs", vectors, ids=ids, index_mode="deferred")
client.collections.optimize("docs")   # build the index
```

### The timeout parameter for large immediate loads

A large `immediate` load can take longer than the client's default deadline, because the call only returns once the index is built. The SDK `bulk_insert` method takes a `timeout` (seconds) for exactly this case:

```python
client.vectors.bulk_insert("docs", vectors, ids=ids, index_mode="immediate", timeout=600)
```

When `timeout` is `None` (the default), the SDK does not just use the bare default deadline for a bulk insert. It derives a deadline that scales with the load so a healthy load is never cut short. The estimate accounts for both how many vectors you are inserting and their dimension, since a higher-dimension vector (for example 1536) costs more to index than a low-dimension one. So a high-dimension immediate load gets a proportionally larger deadline automatically and does not time out early. A `timeout` you pass yourself always wins and is used exactly.

That said, the auto-scaled deadline is a safety net, not a substitute for the right ingestion path. For anything beyond a small or moderate load, especially high-dimension data, prefer one of the scale paths below rather than leaning on a long immediate-mode deadline:

- Switch to `index_mode="deferred"` and call `optimize()` once at the end (see the next section). This stores the data fast, then builds the index in a single pass.
- For very large datasets, use the file-based `bulk_insert_from_path` path described in [Large-scale loads](#3-large-scale-loads). It avoids streaming every vector over the wire and keeps memory bounded.

---

## 2. optimize()

`optimize()` builds the search index from durable storage and rebuilds the graph. It is the partner to deferred loads: after a `deferred` bulk insert, calling `optimize()` makes the collection fully queryable.

```python
client.collections.optimize("docs", rebuild_graph=True)
```

Because it reads from durable storage, "deferred insert, then optimize" is exactly equivalent in queryability to an immediate load; the only difference is when the build happens. Like the immediate inline build, `optimize()` runs off the request runtime, so other requests keep flowing while it works.

See [Core Concepts](core-concepts.md) for how the collection status moves through `pending_optimization` and `optimizing` during this.

---

## 3. Large-scale loads

For very large datasets, the recommended production path is **`bulk_insert_from_path`**. Instead of streaming every vector over the wire, the server reads a file on its own filesystem (memory-mapped), in chunks, with bounded memory.

```python
result = client.vectors.bulk_insert_from_path(
    "docs",
    path="/data/staging/vectors.f32",
    dim=1536,
    index_mode="immediate",
    chunk_size=100_000,     # process in chunks for a lower memory peak
)
print(result.inserted_count)
```

Why this is the scale path:

- The in-memory SDK `bulk_insert` is convenient and great for small to moderate loads, but it sends data over gRPC and the per-core CPU cost of that path becomes the bottleneck on very large loads.
- `bulk_insert_from_path` avoids the streaming and client-side allocation entirely. The working memory for the load is bounded by the index being built, not by the size of the input file, and `chunk_size` lets you cap the peak further on memory-tight machines.

See [Benchmarks](benchmarks.md) for measured throughput, peak memory, and the full reproduction recipe for a 1M-vector, 1536-dimensional load via this path.

---

## 4. Health during ingestion

A large ingest does not take the server offline. The liveness and readiness probes stay responsive throughout:

- `GET /readyz` and `GET /healthz` report readiness and liveness.
- `GET /health` reports overall health.

Because the index build (whether inline `immediate` or via `optimize()`) runs off the request runtime, these probes keep answering while a big load is in progress. Orchestrators relying on them will not flap during ingestion. See [Configuration](configuration.md) for the related storage, snapshot, and WAL settings that govern durability during a load.

---

## See also

- [Configuration](configuration.md): storage, snapshot, WAL, and bulk-insert settings.
- [Benchmarks](benchmarks.md): throughput and memory numbers for large loads.
- [API Reference](api-reference.md): the bulk insert, bulk-insert-from-path, and optimize endpoints in full.
- [Python SDK](python-sdk.md): the `bulk_insert` and `bulk_insert_from_path` method reference.
