# SwarnDB

**A billion-scale vector database with a built-in virtual graph layer and advanced math engine.**

[![Build](https://img.shields.io/github/actions/workflow/status/chirotpal/swarndb/ci.yml?branch=main&label=build)](https://github.com/swarndb/swarndb/actions)
[![License](https://img.shields.io/badge/license-ELv2-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0-brightgreen)]()

SwarnDB is a high-performance vector database written in Rust that goes beyond nearest-neighbor search. It combines HNSW and IVF+PQ indexing with a virtual graph layer that automatically computes relationships between vectors, and ships with 15+ built-in vector math operations -- from ghost vectors and cone search to SLERP interpolation and k-means clustering. One engine, three capabilities: vector search, graph traversal, and vector mathematics.

---

## Why SwarnDB

- **Vector search + graph traversal in one engine.** The virtual graph layer computes nearest-neighbor edges and threshold-filtered relationships automatically. Query vectors, then traverse their connections -- no external graph database required.

- **15+ vector math operations built in.** Ghost vectors, cone search, SLERP interpolation, k-means, PCA, maximal marginal relevance, centroid computation, vector drift detection, and more. Not just cosine similarity and dot product.

- **Billion-scale without compromise.** IVF + HNSW + product quantization keeps memory bounded while maintaining high recall on datasets with billions of vectors.

- **Rust-native performance throughout.** SIMD-accelerated distance kernels (AVX2, SSE4.1, NEON, scalar fallback), zero-copy mmap, arena allocators, DashMap lock-free concurrency, and fine-grained HNSW locking.

---

## Performance

Benchmarked on real-world datasets with production-grade configurations:

| Metric | Result | Dataset |
|---|---|---|
| Search throughput | **1,620 QPS** | 500K OpenAI vectors, 8 threads |
| Recall@10 | **0.965+** | 500K OpenAI vectors |
| Search throughput | **1,562 QPS** | DBPedia 1M vectors, 32-core |
| Recall@10 | **0.988 -- 0.998** | DBPedia 1M vectors |

---

## Quick Start

### Docker Compose

```bash
git clone https://github.com/swarndb/swarndb.git
cd swarndb
docker compose up -d
```

SwarnDB is now running with gRPC on port `50051` and REST on port `8080`.

### Build from Source

Requires Rust 1.85+.

```bash
git clone https://github.com/swarndb/swarndb.git
cd swarndb
cargo build --release
./target/release/vf-server
```

---

## Python SDK

```bash
pip install swarndb
```

```python
from swarndb import SwarnDBClient

client = SwarnDBClient(host="localhost", grpc_port=50051)

# Create a collection
client.create_collection("articles", dimension=384)

# Insert vectors
client.insert("articles", id="vec-1", vector=[0.1, 0.2, ...], metadata={"topic": "physics"})
client.insert("articles", id="vec-2", vector=[0.3, 0.1, ...], metadata={"topic": "math"})
client.insert("articles", id="vec-3", vector=[0.2, 0.4, ...], metadata={"topic": "physics"})

# Search
results = client.search("articles", query_vector=[0.1, 0.2, ...], top_k=10)
for r in results:
    print(r.id, r.score)

# Graph: set a similarity threshold, then traverse relationships
client.set_threshold("articles", threshold=0.85)
neighbors = client.get_neighbors("articles", vector_id="vec-1")
for edge in neighbors:
    print(edge.target_id, edge.weight)

# Search with graph-enriched results
results = client.search(
    "articles",
    query_vector=[0.1, 0.2, ...],
    top_k=10,
    include_graph=True,
    graph_threshold=0.85,
)
```

Async support is available via `AsyncSwarnDBClient` with the same API surface.

---

## Architecture

SwarnDB is organized as seven Rust crates with clean dependency boundaries:

| Crate | Role |
|---|---|
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

- **HNSW index** -- approximate nearest neighbor search with configurable `ef_construction`, `ef_search`, and `M` parameters
- **IVF + Product Quantization** -- inverted file indexing with PQ compression for billion-scale datasets
- **Batch search** -- multi-query execution with shared overhead
- **Pre-filtering** -- adaptive index selection (B-tree, hash, bitmap) for metadata-filtered queries
- **Configurable per-query ef_search** -- tune recall/latency tradeoff at query time

### Virtual Graph Layer

- **Automatic relationship computation** -- nearest-neighbor edges generated from HNSW structure with configurable similarity thresholds
- **Graph traversal** -- BFS/DFS traversal across vector relationships for multi-hop discovery
- **Threshold-based filtering** -- per-collection, per-query, and per-vector threshold precedence
- **Graph-enriched search** -- search results automatically annotated with related vectors and edge weights
- **Deferred graph mode** -- batch inserts defer graph computation until `optimize()` is called

### Math Engine

15+ vector math operations available through both gRPC and REST APIs:

| Operation | Description |
|---|---|
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

- **AVX2** -- 256-bit operations on x86_64
- **SSE4.1** -- 128-bit fallback on x86_64
- **NEON** -- 128-bit operations on ARM/Apple Silicon
- **Scalar** -- portable fallback for all platforms

Specialized kernels include fused cosine distance (dot product + norms in a single pass), batched multi-vector distance computation, and SIMD gather for PQ distance table lookups.

---

## Deployment

### Docker Compose

```bash
docker compose up -d
```

### Kubernetes

```bash
kubectl apply -k k8s/
```

### Helm

```bash
helm install swarndb helm/swarndb
```

Production deployment includes read-only filesystem, security contexts (`no-new-privileges`), resource limits, health probes, network policies, and configurable logging.

---

## Configuration

All configuration is via environment variables. See `.env.example` for defaults.

| Variable | Default | Description |
|---|---|---|
| `SWARNDB_HOST` | `0.0.0.0` | Bind address |
| `SWARNDB_GRPC_PORT` | `50051` | gRPC listener port |
| `SWARNDB_REST_PORT` | `8080` | REST listener port |
| `SWARNDB_DATA_DIR` | `./data` | Data storage directory |
| `SWARNDB_LOG_LEVEL` | `info` | Log verbosity (`trace`, `debug`, `info`, `warn`, `error`) |
| `SWARNDB_API_KEYS` | *(empty)* | Comma-separated API keys; empty disables auth |
| `SWARNDB_MAX_CONNECTIONS` | `1000` | Maximum concurrent connections |
| `SWARNDB_REQUEST_TIMEOUT_MS` | `30000` | Request timeout in milliseconds |

---

## API Reference

SwarnDB exposes dual API surfaces:

- **gRPC** on port `50051` -- high-throughput, streaming-capable
- **REST** on port `8080` -- browser and curl-friendly

| Operation | gRPC Service | REST Endpoint |
|---|---|---|
| Collection CRUD | `CollectionService` | `POST/GET/DELETE /v1/collections` |
| Vector CRUD | `VectorService` | `POST/GET/DELETE /v1/collections/{id}/vectors` |
| Search | `SearchService` | `POST /v1/search` |
| Batch search | `SearchService` | `POST /v1/search/batch` |
| Graph operations | `GraphService` | `POST/GET /v1/graph` |
| Math operations | `MathService` | `POST /v1/math/*` |
| Health / Readiness | `HealthService` | `GET /v1/health`, `GET /v1/ready` |

Proto definitions are in the `proto/` directory.

---

## License

SwarnDB is licensed under the [Elastic License 2.0 (ELv2)](LICENSE).

---

Built by the SwarnDB team.
