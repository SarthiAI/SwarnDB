# SwarnDB Python SDK

Official Python SDK for [SwarnDB](https://github.com/SarthiAI/SwarnDB), a high-performance vector database that combines HNSW and IVF + product quantization indexing with a virtual graph layer and 15+ built-in vector math operations.

## Installation

```bash
pip install swarndb
```

Requires Python 3.9 or higher.

## Quick Start

```python
from swarndb import SwarnDBClient

with SwarnDBClient(host="localhost", port=50051) as client:
    # Create a collection
    client.collections.create(
        "articles",
        dimension=384,
        distance_metric="cosine",
    )

    # Insert a vector
    vec_id = client.vectors.insert(
        "articles",
        vector=[0.1, 0.2, 0.3, ...],   # length 384
        metadata={"topic": "physics", "year": 2024},
    )

    # Search
    results = client.search.query("articles", vector=[0.1, 0.2, 0.3, ...], k=10)
    for r in results.results:
        print(r.id, r.score)
```

## Bulk Insert From a File

For large loads, stage your vectors as a `.npy` (or flat `.f32`) file in a directory listed in the server's `SWARNDB_BULK_INSERT_ALLOWED_ROOTS` (which defaults to `SWARNDB_DATA_DIR`), then point the server at the file. The server reads the file via memory mapping, so the working memory for the load is bounded by the index being built rather than by the input file size.

```python
import numpy as np
from swarndb import SwarnDBClient

vectors = np.random.rand(1_000_000, 1536).astype(np.float32)
np.save("/data/ingest/embeddings.npy", vectors)

with SwarnDBClient(host="localhost", port=50051) as client:
    client.collections.create("docs", dimension=1536, distance_metric="cosine")

    result = client.vectors.bulk_insert_from_path(
        collection="docs",
        path="/data/ingest/embeddings.npy",
        dim=1536,
        expected_count=1_000_000,
        total_count_hint=1_000_000,
        index_mode="immediate",
    )

    print(result.inserted_count, len(result.assigned_ids))
```

For tight-memory hosts where the single-pass load would not fit, set `chunk_size` to a positive value (for example `100_000`). The server then processes the load in chunks and releases scratch memory between chunks, trading wall-clock for a lower peak resident memory footprint.

## Async Client

The async client mirrors the full API surface using `asyncio`, including `bulk_insert_from_path`.

```python
import asyncio
from swarndb import AsyncSwarnDBClient

async def main():
    async with AsyncSwarnDBClient(host="localhost", port=50051) as client:
        await client.collections.create("articles", dimension=384)
        await client.vectors.insert(
            "articles",
            vector=[0.1, 0.2, 0.3, ...],
            metadata={"topic": "physics"},
        )
        results = await client.search.query("articles", vector=[0.1, 0.2, 0.3, ...], k=10)
        for r in results.results:
            print(r.id, r.score)

asyncio.run(main())
```

## Features

- Sync (`SwarnDBClient`) and async (`AsyncSwarnDBClient`) clients with identical method names and return types
- Single insert, streaming bulk insert, and file-based bulk insert (`bulk_insert_from_path`)
- HNSW tuning knobs (`ef_construction`, `ef_search`, `M`) settable per collection and per query
- Vector similarity search with metadata filtering (`Filter.eq`, `Filter.in_`, `Filter.between`, boolean combinators with `&`, `|`, `~`)
- Batch search across multiple queries in one round trip
- Virtual graph traversal with per-collection and per-vector similarity thresholds
- 15+ vector math operations (centroid, cone search, SLERP, drift detection, k-means, PCA, MMR, analogies)
- Bulk insert checkpoints and resume via `resume_token` for long-running loads
- NumPy arrays accepted anywhere a `list[float]` is expected

## Documentation

For the complete reference, see the SwarnDB documentation:

- [Python SDK reference](https://github.com/SarthiAI/SwarnDB/blob/main/docs/python-sdk.md)
- [API reference (REST and gRPC)](https://github.com/SarthiAI/SwarnDB/blob/main/docs/api-reference.md)
- [Benchmarks](https://github.com/SarthiAI/SwarnDB/blob/main/docs/benchmarks.md)
- [Getting started](https://github.com/SarthiAI/SwarnDB/blob/main/docs/getting-started.md)

## License

Elastic License 2.0 (ELv2).
