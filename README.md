<p align="center">
  <img src="assets/logo.png" alt="SwarnDB Logo" width="150">
</p>

<p align="center">
  <h1 align="center">SwarnDB</h1>
  <p align="center">
    <strong>The vector database that thinks in graphs.</strong>
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-ELv2-blue?style=flat-square" alt="License"></a>
    <a href="https://github.com/SarthiAI/SwarnDB/releases"><img src="https://img.shields.io/badge/version-1.0-brightgreen?style=flat-square" alt="Version"></a>
  </p>
</p>

---

## What is SwarnDB

SwarnDB is a high-performance vector database written in Rust that combines HNSW and IVF+PQ indexing with a virtual graph layer and 15+ built-in vector math operations. Unlike traditional vector databases that stop at nearest-neighbor search, SwarnDB automatically computes relationships between vectors and exposes them as a traversable graph.

**One engine, three capabilities: vector search, graph traversal, and vector mathematics.**

---

## Why SwarnDB

- **Vector search + graph traversal in one engine.**
  The virtual graph layer computes nearest-neighbor edges and threshold-filtered relationships automatically. Query vectors, then traverse their connections; no external graph database required.

- **15+ vector math operations built in.**
  Ghost vectors, cone search, SLERP interpolation, k-means, PCA, maximal marginal relevance, centroid computation, vector drift detection, and more.

- **Billion-scale without compromise.**
  IVF + HNSW + product quantization keeps memory bounded while maintaining high recall on datasets with hundreds of millions of vectors.

- **Rust-native performance with SIMD acceleration.**
  AVX2, SSE4.1, NEON, and scalar fallback. Zero-copy mmap, arena allocators, DashMap lock-free concurrency, and fine-grained HNSW locking.

- **Dual API: gRPC + REST.**
  High-throughput gRPC for production pipelines. REST for rapid prototyping, debugging, and curl-friendly workflows.

---

## Performance

| ef_search | QPS | Recall@10 | p50 (ms) | p95 (ms) | p99 (ms) |
|-----------|------|-----------|----------|----------|----------|
| 50 | 1,563 | 98.8% | 4.76 | 7.62 | 9.35 |
| 100 | 1,271 | 99.0% | 5.74 | 10.93 | 14.06 |
| 200 | 984 | 99.2% | 7.55 | 13.33 | 18.17 |
| 400 | 666 | 99.5% | 11.41 | 19.29 | 23.56 |
| 800 | 388 | 99.8% | 19.64 | 33.54 | 41.25 |

*DBpedia 1M (1536-dim) on a 32-core, 64 GB RAM system.*

---

## Quick Start

Clone the repository and build the Docker image locally:

```bash
git clone https://github.com/SarthiAI/SwarnDB.git
cd SwarnDB
docker build -t swarndb .
docker run -d \
  --name swarndb \
  -p 8080:8080 \
  -p 50051:50051 \
  -v swarndb_data:/data \
  swarndb
```

Verify it is running:

```bash
curl http://localhost:8080/health
```

### Create Your First Collection and Search

**1. Create a collection:**

```bash
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "articles",
    "dimension": 384,
    "distance_metric": "cosine"
  }'
```

**2. Insert vectors:**

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "values": [0.1, 0.2, 0.3, 0.4],
    "metadata": {"topic": "physics", "year": 2024}
  }'
```

**3. Search:**

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, 0.4],
    "k": 10
  }'
```

---

## Docker Compose

Create a `docker-compose.yml`:

```yaml
services:
  swarndb:
    build: .
    container_name: swarndb
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - swarndb_data:/data
    environment:
      - SWARNDB_LOG_LEVEL=info
      - SWARNDB_DATA_DIR=/data
    restart: unless-stopped

volumes:
  swarndb_data:
```

Start it:

```bash
docker compose up -d
```

---

## Python SDK

```bash
pip install swarndb
```

```python
from swarndb import SwarnDBClient

with SwarnDBClient(host="localhost", port=50051) as client:
    # Create a collection
    client.collections.create("articles", dimension=384, distance_metric="cosine")

    # Insert vectors
    client.vectors.insert("articles", vector=[0.1, 0.2, ...], metadata={"topic": "physics"})
    client.vectors.insert("articles", vector=[0.3, 0.1, ...], metadata={"topic": "math"})
    client.vectors.insert("articles", vector=[0.2, 0.4, ...], metadata={"topic": "physics"})

    # Search
    results = client.search.query("articles", vector=[0.1, 0.2, ...], k=10)
    for r in results.results:
        print(r.id, r.score)  # distance score (lower = more similar)

    # Graph: set a similarity threshold, then traverse relationships
    client.graph.set_threshold("articles", threshold=0.85)
    client.collections.optimize("articles")
    edges = client.graph.get_related("articles", vector_id=1)
    for edge in edges:
        print(edge.target_id, edge.similarity)

    # Search with graph-enriched results
    results = client.search.query(
        "articles",
        vector=[0.1, 0.2, ...],
        k=10,
        include_graph=True,
        graph_threshold=0.85,
    )
```

Async support is available via `AsyncSwarnDBClient` with the same API surface.

---

## Architecture

SwarnDB is organized as seven Rust crates with clean dependency boundaries:

| Crate | Role |
|:--|:--|
| `vf-core` | Core types, distance functions, SIMD kernels |
| `vf-storage` | WAL, segment management, memory-mapped I/O, collections |
| `vf-index` | HNSW and brute-force index implementations |
| `vf-query` | Filter evaluation, query execution, batch processing |
| `vf-quantization` | Scalar, product, and binary quantization; IVF partitioning |
| `vf-graph` | Virtual relationship graph, traversal algorithms |
| `vf-server` | gRPC and REST servers, authentication, health checks |

---

## Key Capabilities

### Vector Operations

- **HNSW index** with configurable `ef_construction`, `ef_search`, and `M` parameters
- **IVF + Product Quantization** for billion-scale datasets with bounded memory
- **Batch search** with multi-query execution and shared overhead
- **Pre-filtering** with adaptive index selection (B-tree, hash, bitmap) for metadata-filtered queries
- **Per-query ef_search** to tune recall/latency tradeoff at query time

### Virtual Graph Layer

- **Automatic relationship computation** from HNSW structure with configurable similarity thresholds
- **Graph traversal** via BFS/DFS across vector relationships for multi-hop discovery
- **Threshold-based filtering** with per-collection, per-query, and per-vector precedence
- **Graph-enriched search** where results are automatically annotated with related vectors and edge weights
- **Deferred graph mode** for batch inserts that defer graph computation until `optimize()` is called

### Math Engine

15+ vector math operations available through both gRPC and REST APIs:

| Operation | Description |
|:--|:--|
| Ghost vectors | Synthetic vectors representing absent concepts in a space |
| Cone search | Angular proximity search within a cone aperture |
| SLERP interpolation | Spherical linear interpolation between vectors |
| Centroid computation | Weighted and unweighted centroids of vector sets |
| Vector drift detection | Track how vector representations change over time |
| K-means clustering | Partition vectors into k clusters |
| PCA | Dimensionality reduction via principal component analysis |
| Analogy completion | Vector arithmetic for analogy tasks (A:B :: C:?) |
| Maximal marginal relevance | Diversity-aware result re-ranking |
| Vector normalization | L2 normalization for angular similarity |

### SIMD Acceleration

All distance computations are SIMD-accelerated with runtime dispatch:

| Instruction Set | Platform | Width |
|:--|:--|:--|
| **AVX2** | x86_64 | 256-bit |
| **SSE4.1** | x86_64 | 128-bit |
| **NEON** | ARM / Apple Silicon | 128-bit |
| **Scalar** | All platforms | Portable fallback |

Specialized kernels include fused cosine distance (dot product + norms in a single pass), batched multi-vector distance computation, and SIMD gather for PQ distance table lookups.

---

## Configuration

All configuration is via environment variables. See `.env.example` for the full list.

| Variable | Default | Description |
|:--|:--|:--|
| `SWARNDB_HOST` | `0.0.0.0` | Bind address |
| `SWARNDB_GRPC_PORT` | `50051` | gRPC listener port |
| `SWARNDB_REST_PORT` | `8080` | REST listener port |
| `SWARNDB_DATA_DIR` | `./data` | Data storage directory |
| `SWARNDB_LOG_LEVEL` | `info` | Log verbosity (`trace`, `debug`, `info`, `warn`, `error`) |
| `SWARNDB_API_KEYS` | *(empty)* | Comma-separated API keys; empty disables auth |
| `SWARNDB_MAX_CONNECTIONS` | `1000` | Maximum concurrent connections |
| `SWARNDB_REQUEST_TIMEOUT_MS` | `10000` | Request timeout in milliseconds |

---

## API Reference

SwarnDB exposes dual API surfaces: **gRPC** on port `50051` and **REST** on port `8080`.

| Operation | gRPC Service | REST Endpoint |
|:--|:--|:--|
| Collection CRUD | `CollectionService` | `POST/GET/DELETE /api/v1/collections` |
| Vector CRUD | `VectorService` | `POST/GET/DELETE /api/v1/collections/{id}/vectors` |
| Search | `SearchService` | `POST /api/v1/collections/{id}/search` |
| Batch search | `SearchService` | `POST /api/v1/search/batch` |
| Graph operations | `GraphService` | `POST/GET /api/v1/collections/{id}/graph/*` |
| Math operations | `MathService` | `POST /api/v1/collections/{id}/math/*` |
| Health / Readiness | `HealthService` | `GET /health`, `GET /ready` |

For complete API documentation, see [API Reference](docs/api-reference.md).

---

## Documentation

| Guide | Description |
|:--|:--|
| [Getting Started](docs/getting-started.md) | Installation, first steps, basic usage |
| [Core Concepts](docs/core-concepts.md) | Collections, vectors, metadata, indexing |
| [API Reference](docs/api-reference.md) | Complete gRPC and REST API documentation |
| [Python SDK](docs/python-sdk.md) | SDK installation, client usage, async support |
| [Virtual Graph](docs/virtual-graph.md) | Graph layer concepts, traversal, thresholds |
| [Vector Math](docs/vector-math.md) | All 15+ math operations with examples |
| [Configuration](docs/configuration.md) | Environment variables and tuning guide |
| [Deployment](docs/deployment.md) | Docker, Kubernetes, and Helm deployment |

---

## Issues and Feedback

Found a bug or have a feature request? Open an issue on [GitHub Issues](https://github.com/SarthiAI/SwarnDB/issues).

---

## License

SwarnDB is licensed under the [Elastic License 2.0 (ELv2)](LICENSE).

---

The SwarnDB project is designed, developed and maintained by <a href="https://www.linkedin.com/in/chirotpal/" target="_blank">Chirotpal</a>
