# SwarnDB Benchmarks

This page documents the workloads SwarnDB is measured against, the hardware those measurements are taken on, the exact API calls used to reproduce them, and the measured numbers.

Every number on this page is a standalone fact about the workload and hardware listed in the same section. Numbers from one workload do not transfer to another. When you tune SwarnDB for your own data, treat the numbers here as reference points, not guarantees.

## Table of contents

- [Reference host](#reference-host)
- [Workload reference](#workload-reference)
- [Search throughput and recall](#search-throughput-and-recall)
- [Worker saturation](#worker-saturation)
- [Ingestion via bulk insert from path](#ingestion-via-bulk-insert-from-path)
- [Restart and recovery](#restart-and-recovery)
- [Memory behavior](#memory-behavior)
- [Reproducing these results](#reproducing-these-results)

---

## Reference host

All measurements on this page were taken on a single reference host class.

| Spec     | Value                                                                |
|----------|----------------------------------------------------------------------|
| CPU      | 32 cores, x86_64                                                      |
| Memory   | 64 GB (59 GiB usable), no swap                                        |
| Storage  | NVMe SSD                                                              |
| OS       | Linux                                                                 |
| Runtime  | SwarnDB Docker image (`docker compose` at the default 56 GiB / 32 core caps) |
| WAL mode | `SWARNDB_WAL_FSYNC_MODE=per_batch:10000`                              |

`SWARNDB_WAL_FSYNC_MODE=per_batch:10000` is required on this disk class for slow-disk WAL throughput; without it, WAL fsync saturates the disk during bulk loads and queries time out. See [Configuration](configuration.md) for the full list of WAL fsync options.

---

## Workload reference

| Workload id | Dataset    | Vector count | Dimension | Distance metric | Notes                        |
|-------------|------------|--------------|-----------|-----------------|------------------------------|
| W1M-1536    | DBpedia 1M | 1,000,000    | 1,536     | cosine          | Real text embeddings, dense. |

The id is shorthand for the dataset, dimension, and distance metric, and is referenced from every measurement section below.

---

## Search throughput and recall

Workload: **W1M-1536**. Default HNSW parameters (`M=16`, `ef_construction=200`). 1,000 queries per `ef_search` setting averaged across 3 iterations. 8 concurrent searcher threads (see [Worker saturation](#worker-saturation) for why 8). Clean-room measurement: fresh Docker container, fresh named volume, dataset loaded via `bulk_insert_from_path`.

| ef_search | QPS   | Recall@10 | p50 (ms) | p95 (ms) | p99 (ms) |
|-----------|-------|-----------|----------|----------|----------|
| 25        | 2,398 | 0.9816    | 3.16     | 4.91     | 6.06     |
| 50        | 2,214 | 0.9894    | 3.33     | 5.26     | 6.77     |
| 100       | 1,801 | 0.9921    | 4.16     | 6.85     | 8.02     |
| 200       | 1,233 | 0.9935    | 6.18     | 10.19    | 12.26    |
| 400       |   760 | 0.9960    | 10.00    | 16.83    | 20.48    |
| 800       |   437 | 0.9974    | 17.42    | 30.43    | 35.90    |

`ef_search` is a per-query knob. Set it via the `ef_search` parameter on `client.search.query` (Python SDK) or via the `ef_search` field in the REST search request. The default is 50.

For workloads with concurrent inserts in flight, search latency stays in the same band as the table above, because the search path takes shared read access to the HNSW graph and does not serialize behind writers. Sustained same-collection contention (concurrent searcher threads against an active writer holding a per-collection write lock for a 2,000 row chunk) does lower search QPS on that collection while the chunk is in flight; other collections are unaffected.

---

## Worker saturation

Workload: **W1M-1536** at `ef_search=50`, 1,000 queries per worker count averaged across 3 iterations.

| Workers | QPS   | p50 (ms) | p95 (ms) | p99 (ms) | Recall@10 |
|---------|-------|----------|----------|----------|-----------|
| 1       |   430 | 2.32     | 2.79     | 2.99     | 0.9901    |
| 4       | 1,502 | 2.50     | 3.69     | 4.72     | 0.9901    |
| **8**   | **2,174** | **3.39** | **5.42** | **7.00** | 0.9901 |
| 16      | 2,155 | 6.72     | 11.47    | 15.41    | 0.9901    |
| 32      | 2,077 | 13.57    | 27.28    | 40.50    | 0.9901    |
| 64      | 2,046 | 27.52    | 56.07    | 75.24    | 0.9901    |

QPS scales near-linearly to 8 workers and plateaus. Past 8 workers, additional concurrency raises p99 latency by 2 to 10x while QPS stays within 6% of peak. Recall is constant across worker counts. At 64 workers (2x the core count) the server queues gracefully with no errors and no crashes.

Recommended client pool size for production: around the physical core count, capped near 8 unless your workload accepts a higher p99.

---

## Ingestion via bulk insert from path

Workload: **W1M-1536**.

SwarnDB has two ingestion paths:

1. **Streaming bulk insert** over gRPC: the client sends vectors in batches. Use this when the client process owns the data in memory and you do not want to stage a file.
2. **Bulk insert from path**: the server reads a `.npy` or flat `.f32` file from its own filesystem via memory mapping. Use this for large loads where the data is already on disk or can be staged through an object store mount.

For very large loads (hundreds of thousands of vectors and up at high dimension), bulk insert from path is the recommended approach. The server reads vectors directly from the kernel page cache without copying them through a gRPC payload, which keeps the server's working memory for the load bounded by the index it builds rather than by the input file size.

### Measured numbers, 1M load via `bulk_insert_from_path` (W1M-1536)

| Metric                                                | Value     |
|-------------------------------------------------------|-----------|
| Peak container RSS during the load (fresh container)  | 7.45 GiB  |
| Sustained ingestion rate                              | 3,607 vectors/sec |
| Persisted HNSW base file size after load              | 141 MB    |

Reading the same 1M dataset via the streaming gRPC bulk insert path on the same host hits a peak container RSS of approximately 16 GiB. The bulk-insert-from-path mmap path is the right ingestion mode when peak memory matters.

### Reproduction recipe

Stage the data as a `.npy` file in a directory listed in the server's `SWARNDB_BULK_INSERT_ALLOWED_ROOTS` (which defaults to `SWARNDB_DATA_DIR`):

```python
import numpy as np
from swarndb import SwarnDBClient

vectors = np.random.rand(1_000_000, 1536).astype(np.float32)
np.save("/data/ingest/dbpedia_1m.npy", vectors)

with SwarnDBClient(host="localhost", port=50051) as client:
    client.collections.create("dbpedia", dimension=1536, distance_metric="cosine")

    result = client.vectors.bulk_insert_from_path(
        collection="dbpedia",
        path="/data/ingest/dbpedia_1m.npy",
        dim=1536,
        expected_count=1_000_000,
        total_count_hint=1_000_000,
        index_mode="immediate",
    )

    print(result.inserted_count, len(result.assigned_ids))
```

The async client (`AsyncSwarnDBClient`) accepts the same signature.

### Tunable: chunk_size

The `chunk_size` parameter on `bulk_insert_from_path` is `0` by default, which loads the file in a single pass. Setting it to a positive value (for example, `100_000`) processes the load in chunks, snapshots between chunks, prunes the write-ahead log, and releases scratch memory between chunks. This trades wall-clock time for a lower peak resident memory footprint, and is intended for hosts where the single-pass load would not fit.

### Recommended bulk insert batch size

For the streaming bulk insert path (`BulkInsertWithOptions`), the recommended `batch_lock_size` on this reference host is `10000`, which sustained 2,921 vectors/sec on the W1M-1536 workload.

---

## Restart and recovery

Workload: any persistent collection. Numbers below are measured on the **W1M-1536** workload and on a 200,000-vector and 100,000-vector test collection.

After a server restart (planned or unplanned), plain HNSW collections become queryable within seconds of the gRPC and REST ports opening, regardless of collection size. Collections are loaded from disk in parallel during startup, governed by `SWARNDB_MAX_CONCURRENT_COLLECTION_LOADS` (default: `min(cores, 4)`).

Recovery from an unclean shutdown takes two forms, both transparent:

- **Incremental delta replay**: if the server shut down after taking a recent snapshot, only the delta log since that snapshot is replayed.
- **Full write-ahead log replay**: if no usable snapshot exists, the full write-ahead log is replayed.

### Measured numbers

| Scenario                                                | Time to `/readyz` returning 200 |
|---------------------------------------------------------|---------------------------------|
| `docker kill -s SIGKILL` then restart, 200k collection  | 5.5 seconds                     |
| Bulk-insert resume after crash, 100k collection         | 20 seconds                      |

The server exposes the recovery state through `GET /recovery_status` and through the `GetRecoveryStatus` gRPC RPC. Use this from an orchestrator or load balancer to gate traffic until recovery completes. The Kubernetes readiness and startup probes at `GET /readyz` and `GET /startupz` reflect the same state and are the recommended hook in containerized deployments.

Per-collection persistence health is exposed through `GET /api/v1/collections/{collection}/persistence_status` and through the `GetPersistenceStatus` gRPC RPC.

---

## Memory behavior

Workload: **W1M-1536**.

SwarnDB's resident memory for a loaded collection is dominated by the HNSW graph. The graph's working set scales with the vector count and dimension. Vector bytes are held once, inside the graph arena; per-row metadata is stored separately and does not carry a second copy of the vector data.

### Measured numbers, multi-collection footprint

Two 1M, 1536-dim collections coexisting on the reference host:

| Metric                                                              | Value     |
|---------------------------------------------------------------------|-----------|
| Resident container RSS, both collections loaded, idle                | 25.4 GiB  |
| Peak container RSS during a second-collection insert in flight       | 30.7 GiB  |
| Insert-time transient gap (peak minus rest)                          | 5.34 GiB  |

The 5.34 GiB transient gap is the working memory consumed by an active gRPC bulk insert that runs concurrent with already-loaded collections. The mmap path (`bulk_insert_from_path`) does not produce a transient peak of this size, since vectors are read from the kernel page cache rather than buffered in the process heap.

**Planning rule of thumb:** budget approximately 12 to 13 GiB resident per 1M, 1536-dim collection, plus an additional approximately 5 GiB transient peak per concurrent gRPC bulk insert in flight. For load-then-serve deployments, peak and resident converge once the load completes.

### Operational properties

- **Steady-state memory** for a loaded collection is bounded by the graph and stays flat across long-running workloads with continuous insert, search, and delete traffic. Memory does not drift upward with operation count.
- **Memory after a bulk load** is released back to the operating system once the load completes. The process resident size returns to the steady-state working set rather than holding onto allocator slack from the load.
- **Memory during a bulk load via `bulk_insert_from_path`** is bounded by the index being built, since vectors are read directly from the memory-mapped file rather than being copied through the gRPC ingest pipeline.

---

## Reproducing these results

### 1. Spin up SwarnDB

Default Docker run with persistent storage and the WAL fsync mode used for the measurements above:

```bash
docker run -d --name swarndb \
  -p 8080:8080 \
  -p 50051:50051 \
  -v swarndb_data:/data \
  -e SWARNDB_DATA_DIR=/data \
  -e SWARNDB_BULK_INSERT_ALLOWED_ROOTS=/data \
  -e SWARNDB_WAL_FSYNC_MODE=per_batch:10000 \
  sarthiai/swarndb:latest
```

For the ingestion run, mount the directory that holds your `.npy` file at the path you pass to `bulk_insert_from_path`, and include it in `SWARNDB_BULK_INSERT_ALLOWED_ROOTS`.

### 2. Stage the dataset

Stage the W1M-1536 dataset (DBpedia 1M at 1536 dim) as a `.npy` file inside an allowed root:

```python
import numpy as np
vectors = np.load("dbpedia_1m_1536.npy")        # or generate your own
assert vectors.shape == (1_000_000, 1536)
np.save("/data/ingest/dbpedia_1m.npy", vectors)
```

### 3. Load via `bulk_insert_from_path`

See the [Ingestion section](#ingestion-via-bulk-insert-from-path) for the full Python call.

### 4. Run the search measurement

The published numbers come from the `benchmark/qps_vs_recall.py` harness with `--workers 8`:

```bash
python benchmark/qps_vs_recall.py \
  --collection-name dbpedia_1m \
  --rest-port 8080 --grpc-port 50051 \
  --n-queries 1000 --k 10 --iterations 3 \
  --ef-search-list 25,50,100,200,400,800 \
  --workers 8
```

`--workers 8` matches the published numbers; pass `--workers 1` to reproduce the single-thread baseline. For recall, the harness computes ground truth against a brute-force search on the same query set.

### 5. Measure memory, ingestion, and restart

- Container resident memory: `docker stats swarndb` or `cat /proc/$(pidof swarndb)/status | grep VmRSS`. Note that `docker stats` includes mmap'd file pages from the kernel page cache while `VmRSS` does not, so the two differ by several GiB after recovery. For the peak-RSS planning numbers above, `docker stats` is the authoritative metric.
- Recovery state: `curl http://localhost:8080/recovery_status`.
- Per-collection persistence state: `curl http://localhost:8080/api/v1/collections/dbpedia/persistence_status`.
- Restart-to-queryable time: `docker kill -s SIGKILL swarndb`, start it, poll `GET /readyz` until it returns 200, run a search.

---

For the full API surface used in these recipes, see [Python SDK](python-sdk.md) and [API Reference](api-reference.md).
