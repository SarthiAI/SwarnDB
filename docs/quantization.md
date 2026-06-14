# Quantization

Quantization stores your vectors in a compressed form: instead of keeping every dimension as a full 32-bit float, it encodes each one in a smaller representation. A raw vector is a list of 32-bit floating-point numbers, and at high dimension that adds up fast. Quantization is the family of techniques for encoding those numbers more compactly.

A quantized vector is an approximation of the original, so the index may occasionally rank results slightly differently than it would on the raw floats. To keep results accurate, SwarnDB keeps the full-precision vectors on the side and uses them to rescore candidates, so the approximation is used to find candidates quickly and the exact vectors settle the final ranking.

This guide covers the quantization options SwarnDB offers, when to choose each one, and exactly how to turn them on.

> **Today, scalar quantization (SQ8) is the quantization mode you select when you create a collection.** Product quantization (PQ), binary quantization (BQ), and the hybrid IVF + HNSW + PQ index ship inside the engine and are described below so you understand the design space, but SQ8 is the user-facing quantization knob on collection create. See [How to enable quantization](#how-to-enable-quantization) for the exact field and keyword.

---

## Scalar Quantization (SQ8)

SQ8 is the selectable quantization mode in SwarnDB, and the one most users should reach for first when they want quantization.

### What it is

SQ8 is 8-bit scalar quantization. Each 32-bit float in a vector is mapped onto an 8-bit value, so every dimension is encoded as a single byte. The mapping is learned from your own data: SwarnDB looks at the range your values actually occupy (controlled by the `quantile` setting, which trims extreme outliers so the limited 8-bit range is spent where most of your data lives) and fits the encoding to that range. This keeps the approximation tight for the vectors you really have, rather than wasting precision on a worst-case range.

Searches run against the compact 8-bit codes, and the original full-precision vectors are kept on the side so candidates can be rescored against the exact values. That rescoring is what rescues accuracy: the 8-bit codes find candidates quickly, and the exact vectors settle the final ranking.

### Measured behavior

On a 1,000,000-vector, 1536-dimension benchmark (cosine distance, 32 cores), SQ8 holds full-precision-grade recall across the speed range: recall@10 about 0.99 at ef_search 50 (around 1,460 queries per second at 8 threads), rising to about 0.998 at ef_search 800, with recall@100 up to about 0.996 and p99 latency in the low tens of milliseconds. SQ8 search quality tracks uncompressed HNSW on the same workload, and SQ8 restarts on the same fast path as plain HNSW, so a quantized collection becomes queryable about as quickly as an uncompressed one.

### When to choose it

Choose SQ8 when:

- You want quantization that keeps recall close to plain HNSW, not a steep accuracy cliff.
- You want a single, well-supported knob rather than a tuning exercise.
- You want the quantized collection to restart on the same fast path as a plain one.

SQ8 is the selectable quantization mode: it preserves recall at high QPS and restarts as fast as plain HNSW. It sits between full-precision HNSW and the more aggressive encodings of PQ or BQ.

### Fast restart parity with plain HNSW

A quantized collection used to pay a penalty when the server restarted: rebuilding the compressed index from scratch took noticeably longer than bringing a plain HNSW collection back up. That gap is gone. SQ8 collections now restart on the same fast path as plain HNSW. On a clean shutdown the saved quantizer state and codes are loaded back directly, so a quantized collection becomes queryable about as quickly as a plain one, regardless of size. If the server comes back from an unclean shutdown, recovery falls back to a transparent rebuild, and the collection still comes up correct. The practical result: choosing SQ8 does not cost you a slower restart.

---

## Product Quantization (PQ)

### What it is

Product quantization splits each vector into several short sub-vectors and replaces each sub-vector with a small codebook index. Instead of storing the numbers, the vector is stored as a handful of references into learned codebooks. In principle this encodes more aggressively than scalar quantization, because a whole group of dimensions collapses to a single small code, at the cost of more accuracy loss.

### What it is in principle

PQ is the technique you reach for when very aggressive encoding matters more than recall. The encoding is dramatic and recall is lower than SQ8, which is the textbook trade-off product quantization makes.

---

## Binary Quantization (BQ)

### What it is

Binary quantization is the most aggressive option. Each dimension is reduced to a single bit, so a vector becomes a compact bit-string and distances are computed with fast bitwise operations. This is the most compact encoding and gives the fastest raw distance math.

### What it is in principle

BQ is the technique for cases where very fast scanning matters more than fine-grained accuracy, and where the embeddings tolerate aggressive encoding. Because a single bit per dimension discards the most information, BQ is best paired with a refinement step on the full-precision vectors for the final ranking. It suits coarse first-pass filtering.

---

## IVF and the hybrid IVF + HNSW + PQ index

### IVF

IVF (inverted file) is a partitioning scheme rather than an encoding scheme. It clusters the vector space into regions using k-means, and at query time it searches only the few regions nearest the query instead of the whole collection. This cuts the amount of work per query when the collection is very large.

### IVF + HNSW + PQ

The hybrid index combines all three ideas: IVF narrows the search to the relevant regions, HNSW navigates within them quickly, and PQ encodes each vector compactly. This is the design aimed at the billion-scale end of the spectrum, where the goal is a search path that does not touch every vector.

---

## How to enable quantization

A collection is plain HNSW by default. To make it an SQ8 collection, you set the quantization configuration at create time. There is nothing to do after creation: the quantizer is trained as the collection is built and optimized.

### What you set

| Surface | Field / keyword | Value |
|---------|-----------------|-------|
| REST body | `quantization.type` | `"scalar"` (selects SQ8) |
| REST body | `quantization.quantile` | optional float, defaults to `0.99` |
| REST body | `quantization.always_ram` | optional bool, defaults to `true` |
| Python SDK | `quantization=` keyword on `collections.create` | a `QuantizationConfig` (see below) |

Omit the `quantization` field entirely and the collection is plain HNSW.

### REST

```bash
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products",
    "dimension": 1536,
    "distance_metric": "cosine",
    "quantization": {
      "type": "scalar",
      "quantile": 0.99,
      "always_ram": true
    }
  }'
```

Leaving out the `quantization` object creates a plain HNSW collection:

```bash
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products",
    "dimension": 1536,
    "distance_metric": "cosine"
  }'
```

### Python SDK

```python
from swarndb import SwarnDBClient, QuantizationConfig, ScalarQuantizationConfig

client = SwarnDBClient(host="localhost", port=50051)

# SQ8 collection
client.collections.create(
    "products",
    dimension=1536,
    distance_metric="cosine",
    quantization=QuantizationConfig(
        type="scalar",
        scalar=ScalarQuantizationConfig(quantile=0.99, always_ram=True),
    ),
)

# Plain HNSW collection (no quantization argument)
client.collections.create(
    "products_plain",
    dimension=1536,
    distance_metric="cosine",
)
```

`ScalarQuantizationConfig` defaults `quantile` to `0.99` and `always_ram` to `True`, so `QuantizationConfig(type="scalar", scalar=ScalarQuantizationConfig())` is enough to take the defaults.

> **Note on PQ, BQ, and IVF:** these live inside the engine and back the design described above, but collection create currently exposes scalar quantization (SQ8) as the selectable mode. A create request that asks for any other quantization type is rejected. For an SQ8-quantized collection, use the configuration shown here.

---

## How to choose

Use this as a starting point and tune against your own data and recall targets.

| Situation | Recommended mode |
|-----------|------------------|
| Highest possible recall, no quantization | Plain HNSW (no quantization) |
| Want quantization that preserves recall at speed, with fast restart | **SQ8** (the selectable quantization mode) |
| In principle: more aggressive encoding, accepting more accuracy cost | PQ |
| In principle: most aggressive encoding and fast scanning, with a refinement pass for final ranking | BQ |
| In principle: a partitioned search path that does not touch every vector at the largest scales | IVF + HNSW + PQ |

The short version: use plain HNSW when you want the highest possible recall and no quantization. Choose SQ8 when you want quantization, since it keeps recall close to plain HNSW at high QPS and restarts just as fast. PQ, BQ, and IVF + HNSW + PQ describe the broader design space (more aggressive encodings, with accuracy falling as the encoding gets more aggressive), but SQ8 is the mode you select on collection create.

---

## Related

- [Core Concepts](core-concepts.md) for how indexing, HNSW, and IVF+PQ fit into the overall engine.
- [Configuration](configuration.md) for server-level tuning.
- [Python SDK](python-sdk.md) for the full collection-create surface.
