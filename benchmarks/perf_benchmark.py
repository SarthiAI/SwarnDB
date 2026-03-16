#!/usr/bin/env python3
"""SwarnDB Lightweight Performance Benchmark.

Runs against the REST API via Docker on an 8 GB Mac.
Uses 10K vectors at 128-dim (~5 MB) -- fits easily in memory.

Usage:
    python perf_benchmark.py                  # full run (10K vectors, 1K queries)
    python perf_benchmark.py --quick          # quick run (2K vectors, 200 queries)
    python perf_benchmark.py --url http://host:port

Dependencies: requests (stdlib otherwise)
"""

# Baselines (pre-optimization, from Phase 14):
# - Bulk insert: ~110 vec/s at 1536-dim
# - Search QPS: ~1620 at 500K vectors
# - Search p50: ~0.56ms
# - Recall@10: ~0.965
# Note: These baselines were at larger scale/dimension.
# Direct comparison not valid but directional trends are meaningful.

import argparse
import json
import math
import random
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("ERROR: requests is required.  pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = "http://localhost:8080"
NUM_VECTORS = 10000
DIMENSION = 128
NUM_SEARCH_QUERIES = 1000
CONCURRENT_THREADS = 4
EF_SEARCH = 200  # Higher ef_search for meaningful recall measurement

# ---------------------------------------------------------------------------
# Global session (connection pooling for 2-3x QPS improvement)
# ---------------------------------------------------------------------------

SESSION = requests.Session()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_url():
    return f"{SERVER_URL}/api/v1"


def rand_vec(dim=DIMENSION, rng=None):
    """Generate a random vector using stdlib random."""
    r = rng or random
    return [r.uniform(-1.0, 1.0) for _ in range(dim)]


def normalize(vec):
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def rand_gaussian_vec(dim=DIMENSION, rng=None):
    """Generate unnormalized Gaussian random vector.

    Gaussian vectors produce more realistic distance distributions than
    normalized uniform vectors, giving better recall measurement because
    distances are more spread out (not clustered near a narrow band).
    """
    r = rng or random
    return [r.gauss(0.0, 1.0) for _ in range(dim)]


def fresh_name(prefix="bench"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_collection(name, dim=DIMENSION, session=None):
    s = session or SESSION
    resp = s.post(f"{api_url()}/collections", json={
        "name": name,
        "dimension": dim,
        "distance_metric": "cosine",
        "ef_search": EF_SEARCH,
    })
    if resp.status_code != 200:
        raise RuntimeError(f"create_collection failed: {resp.status_code} {resp.text}")
    return resp.json()


def delete_collection(name, session=None):
    s = session or SESSION
    try:
        s.delete(f"{api_url()}/collections/{name}")
    except Exception:
        pass


def insert_vector(collection, vid, values, metadata=None, session=None):
    s = session or SESSION
    payload = {"id": vid, "values": values}
    if metadata:
        payload["metadata"] = metadata
    resp = s.post(f"{api_url()}/collections/{collection}/vectors", json=payload)
    return resp


def bulk_insert(collection, vectors, session=None):
    s = session or SESSION
    resp = s.post(
        f"{api_url()}/collections/{collection}/vectors/bulk",
        json={"vectors": vectors},
    )
    return resp


def search(collection, query, k=10, session=None, **kwargs):
    s = session or SESSION
    payload = {"query": query, "k": k}
    payload.update(kwargs)
    resp = s.post(f"{api_url()}/collections/{collection}/search", json=payload)
    return resp


def set_threshold(collection, threshold, session=None):
    s = session or SESSION
    resp = s.post(
        f"{api_url()}/collections/{collection}/graph/threshold",
        json={"threshold": threshold},
    )
    return resp


def optimize_collection(collection, session=None):
    s = session or SESSION
    resp = s.post(f"{api_url()}/collections/{collection}/optimize")
    return resp


def percentile(sorted_list, pct):
    """Return the p-th percentile from a sorted list."""
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * pct / 100.0)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


def docker_memory_mb(container="swarndb"):
    """Query Docker stats for memory usage of the container (MB)."""
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().strip()
        # Format: "123.4MiB / 8GiB" or "1.23GiB / 8GiB"
        mem_str = out.split("/")[0].strip()
        if "GiB" in mem_str:
            return float(mem_str.replace("GiB", "").strip()) * 1024
        elif "MiB" in mem_str:
            return float(mem_str.replace("MiB", "").strip())
        elif "KiB" in mem_str:
            return float(mem_str.replace("KiB", "").strip()) / 1024
        return 0.0
    except Exception:
        return 0.0


def warmup_connections(coll, num_warmup=10):
    """Send warmup queries to prime TCP connections and server caches.

    This avoids cold-start effects (TCP handshake, server JIT, cache misses)
    from polluting actual benchmark measurements.
    """
    rng = random.Random(9999)
    for _ in range(num_warmup):
        q = rand_gaussian_vec(DIMENSION, rng)
        search(coll, q, k=1)


# ---------------------------------------------------------------------------
# Benchmark: Single Insert Throughput
# ---------------------------------------------------------------------------

def bench_single_insert(num_vectors_single=1000):
    """Insert vectors one by one and measure ops/sec."""
    coll = fresh_name("bench_single")
    create_collection(coll)
    try:
        rng = random.Random(42)
        vectors = [rand_gaussian_vec(DIMENSION, rng) for _ in range(num_vectors_single)]

        start = time.perf_counter()
        for i, vec in enumerate(vectors):
            resp = insert_vector(coll, 0, vec, metadata={"idx": i})
            if resp.status_code != 200:
                print(f"  [WARN] single insert {i} failed: {resp.status_code}")
        elapsed = time.perf_counter() - start

        ops_per_sec = num_vectors_single / elapsed if elapsed > 0 else 0
        return ops_per_sec
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Benchmark: Bulk Insert Throughput
# ---------------------------------------------------------------------------

def bench_bulk_insert(num_vectors_bulk=9000, batch_size=500):
    """Bulk insert vectors in batches and measure vec/sec."""
    coll = fresh_name("bench_bulk")
    create_collection(coll)
    try:
        rng = random.Random(123)
        total_inserted = 0

        start = time.perf_counter()
        for batch_start in range(0, num_vectors_bulk, batch_size):
            batch_end = min(batch_start + batch_size, num_vectors_bulk)
            batch = []
            for i in range(batch_start, batch_end):
                vec = rand_gaussian_vec(DIMENSION, rng)
                batch.append({
                    "id": 0,
                    "values": vec,
                    "metadata": {"group": i % 10, "score": round(rng.random(), 4)},
                })
            resp = bulk_insert(coll, batch)
            if resp.status_code == 200:
                total_inserted += resp.json().get("inserted_count", 0)
            else:
                print(f"  [WARN] bulk insert batch {batch_start} failed: {resp.status_code}")
        elapsed = time.perf_counter() - start

        vec_per_sec = total_inserted / elapsed if elapsed > 0 else 0
        return vec_per_sec
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Benchmark: Search QPS + Latency (sequential)
# ---------------------------------------------------------------------------

def bench_search(num_queries=1000, num_vectors_search=5000):
    """Insert vectors, then run sequential search queries.

    QPS is measured using wall-clock time of the entire batch
    (not sum of individual latencies) for accurate throughput.
    """
    coll = fresh_name("bench_search")
    create_collection(coll)
    try:
        # Insert vectors
        rng = random.Random(200)
        for batch_start in range(0, num_vectors_search, 500):
            batch_end = min(batch_start + 500, num_vectors_search)
            batch = []
            for i in range(batch_start, batch_end):
                vec = rand_gaussian_vec(DIMENSION, rng)
                batch.append({"id": 0, "values": vec})
            bulk_insert(coll, batch)

        # Generate queries
        qrng = random.Random(300)
        queries = [rand_gaussian_vec(DIMENSION, qrng) for _ in range(num_queries)]

        # Warmup: prime TCP connections and server caches
        warmup_connections(coll)

        # Run searches -- measure wall-clock for QPS, individual for latency
        latencies = []
        wall_start = time.perf_counter()
        for q in queries:
            t0 = time.perf_counter()
            resp = search(coll, q, k=10)
            t1 = time.perf_counter()
            if resp.status_code == 200:
                latencies.append((t1 - t0) * 1000)  # ms
        wall_end = time.perf_counter()

        wall_elapsed = wall_end - wall_start
        qps = len(latencies) / wall_elapsed if wall_elapsed > 0 else 0

        latencies.sort()
        p50 = percentile(latencies, 50)
        p99 = percentile(latencies, 99)

        return qps, p50, p99
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Benchmark: Concurrent Search QPS (ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def bench_concurrent_search_qps(num_queries=1000, num_vectors_search=5000,
                                 num_threads=None):
    """Measure peak QPS using concurrent search threads.

    Each thread gets its own requests.Session for independent connection
    pooling. Wall-clock time of the entire batch determines QPS.
    """
    if num_threads is None:
        num_threads = CONCURRENT_THREADS
    coll = fresh_name("bench_conc_qps")
    create_collection(coll)
    try:
        # Insert vectors
        rng = random.Random(250)
        for batch_start in range(0, num_vectors_search, 500):
            batch_end = min(batch_start + 500, num_vectors_search)
            batch = []
            for i in range(batch_start, batch_end):
                vec = rand_gaussian_vec(DIMENSION, rng)
                batch.append({"id": 0, "values": vec})
            bulk_insert(coll, batch)

        # Generate all queries upfront
        qrng = random.Random(350)
        queries = [rand_gaussian_vec(DIMENSION, qrng) for _ in range(num_queries)]

        # Warmup
        warmup_connections(coll)

        # Partition queries across threads
        per_thread = num_queries // num_threads
        query_chunks = []
        for t in range(num_threads):
            start_idx = t * per_thread
            end_idx = start_idx + per_thread if t < num_threads - 1 else num_queries
            query_chunks.append(queries[start_idx:end_idx])

        def worker(chunk):
            """Run a chunk of queries using a thread-local session."""
            thread_session = requests.Session()
            completed = 0
            for q in chunk:
                resp = search(coll, q, k=10, session=thread_session)
                if resp.status_code == 200:
                    completed += 1
            thread_session.close()
            return completed

        # Measure wall-clock across all threads
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, chunk) for chunk in query_chunks]
            total_completed = sum(f.result() for f in as_completed(futures))
        wall_end = time.perf_counter()

        wall_elapsed = wall_end - wall_start
        concurrent_qps = total_completed / wall_elapsed if wall_elapsed > 0 else 0
        return concurrent_qps
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Benchmark: Filtered Search QPS
# ---------------------------------------------------------------------------

def bench_filtered_search(num_queries=1000, num_vectors_filt=5000):
    """Search with metadata filter and measure QPS."""
    coll = fresh_name("bench_filt")
    create_collection(coll)
    try:
        # Insert vectors with metadata
        rng = random.Random(400)
        categories = ["alpha", "beta", "gamma", "delta", "epsilon"]
        for batch_start in range(0, num_vectors_filt, 500):
            batch_end = min(batch_start + 500, num_vectors_filt)
            batch = []
            for i in range(batch_start, batch_end):
                vec = rand_gaussian_vec(DIMENSION, rng)
                batch.append({
                    "id": 0,
                    "values": vec,
                    "metadata": {
                        "category": categories[i % len(categories)],
                        "price": round(rng.uniform(1.0, 1000.0), 2),
                    },
                })
            bulk_insert(coll, batch)

        # Generate queries with filter
        qrng = random.Random(500)
        queries = [rand_gaussian_vec(DIMENSION, qrng) for _ in range(num_queries)]

        # Warmup
        warmup_connections(coll)

        latencies = []
        wall_start = time.perf_counter()
        for q in queries:
            t0 = time.perf_counter()
            resp = search(coll, q, k=10,
                          filter={"field": "category", "op": "eq", "value": "alpha"},
                          strategy="pre_filter",
                          include_metadata=True)
            t1 = time.perf_counter()
            if resp.status_code == 200:
                latencies.append((t1 - t0) * 1000)
        wall_end = time.perf_counter()

        wall_elapsed = wall_end - wall_start
        qps = len(latencies) / wall_elapsed if wall_elapsed > 0 else 0
        return qps
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Benchmark: Concurrent Search During Write
# ---------------------------------------------------------------------------

def bench_concurrent_search_write(num_search=500, num_write=2000):
    """4 threads searching while 1 thread bulk inserts."""
    coll = fresh_name("bench_conc")
    create_collection(coll)
    try:
        # Seed with some vectors first
        rng = random.Random(600)
        for batch_start in range(0, 3000, 500):
            batch = []
            for i in range(500):
                vec = rand_gaussian_vec(DIMENSION, rng)
                batch.append({"id": 0, "values": vec})
            bulk_insert(coll, batch)

        # Warmup
        warmup_connections(coll)

        search_latencies = []

        def search_worker(worker_id):
            """Run searches and collect latencies."""
            local_lats = []
            wrng = random.Random(700 + worker_id)
            thread_session = requests.Session()
            per_worker = num_search // CONCURRENT_THREADS
            for _ in range(per_worker):
                q = rand_gaussian_vec(DIMENSION, wrng)
                t0 = time.perf_counter()
                resp = search(coll, q, k=10, session=thread_session)
                t1 = time.perf_counter()
                if resp.status_code == 200:
                    local_lats.append((t1 - t0) * 1000)
            thread_session.close()
            return local_lats

        def write_worker():
            """Bulk insert vectors concurrently."""
            wrng = random.Random(900)
            thread_session = requests.Session()
            for batch_start in range(0, num_write, 500):
                batch_end = min(batch_start + 500, num_write)
                batch = []
                for _ in range(batch_end - batch_start):
                    vec = rand_gaussian_vec(DIMENSION, wrng)
                    batch.append({"id": 0, "values": vec})
                bulk_insert(coll, batch, session=thread_session)
            thread_session.close()

        with ThreadPoolExecutor(max_workers=CONCURRENT_THREADS + 1) as executor:
            # Submit search workers
            search_futures = [executor.submit(search_worker, i) for i in range(CONCURRENT_THREADS)]
            # Submit write worker
            write_future = executor.submit(write_worker)

            for f in as_completed(search_futures):
                search_latencies.extend(f.result())
            write_future.result()  # wait for writer

        search_latencies.sort()
        p99 = percentile(search_latencies, 99)
        return p99
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Benchmark: Graph Compute + Search
# ---------------------------------------------------------------------------

def bench_graph_search(num_vectors_graph=3000, num_queries_graph=500):
    """Compute graph, then run graph-enriched search."""
    coll = fresh_name("bench_graph")
    create_collection(coll)
    try:
        # Insert clustered vectors for meaningful graph
        rng = random.Random(1000)
        for batch_start in range(0, num_vectors_graph, 500):
            batch_end = min(batch_start + 500, num_vectors_graph)
            batch = []
            for i in range(batch_start, batch_end):
                vec = rand_gaussian_vec(DIMENSION, rng)
                batch.append({"id": 0, "values": vec})
            bulk_insert(coll, batch)

        # Set threshold to trigger graph computation
        set_threshold(coll, 0.5)

        # Run graph-enriched search
        qrng = random.Random(1100)
        queries = [rand_gaussian_vec(DIMENSION, qrng) for _ in range(num_queries_graph)]

        # Warmup
        warmup_connections(coll)

        latencies = []
        wall_start = time.perf_counter()
        for q in queries:
            t0 = time.perf_counter()
            resp = search(coll, q, k=10, include_graph=True)
            t1 = time.perf_counter()
            if resp.status_code == 200:
                latencies.append((t1 - t0) * 1000)
        wall_end = time.perf_counter()

        wall_elapsed = wall_end - wall_start
        qps = len(latencies) / wall_elapsed if wall_elapsed > 0 else 0
        return qps
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Benchmark: Memory Usage
# ---------------------------------------------------------------------------

def bench_memory(container_name="swarndb", num_vectors_mem=10000):
    """Query Docker stats before and after loading vectors."""
    mem_base = docker_memory_mb(container_name)

    coll = fresh_name("bench_mem")
    create_collection(coll)
    try:
        rng = random.Random(1200)
        for batch_start in range(0, num_vectors_mem, 500):
            batch_end = min(batch_start + 500, num_vectors_mem)
            batch = []
            for i in range(batch_start, batch_end):
                vec = rand_gaussian_vec(DIMENSION, rng)
                batch.append({"id": 0, "values": vec})
            bulk_insert(coll, batch)

        # Small pause for memory to settle
        time.sleep(1)
        mem_loaded = docker_memory_mb(container_name)
        return mem_base, mem_loaded
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Benchmark: Recall@10
# ---------------------------------------------------------------------------

def bench_recall(num_vectors_recall=5000, num_queries_recall=100, k=10):
    """Insert known vectors, search, verify recall against brute-force.

    NOTE: Random (Gaussian) vectors inherently produce lower recall than
    real-world embeddings (e.g., sentence-transformers, OpenAI). Real
    embeddings cluster in meaningful subspaces, making ANN indexes more
    effective. Recall numbers here are a lower bound -- expect 5-15%
    higher recall on production data with semantic structure.
    """
    coll = fresh_name("bench_recall")
    create_collection(coll)
    try:
        # Generate vectors deterministically
        rng = random.Random(1300)
        all_vectors = [rand_gaussian_vec(DIMENSION, rng) for _ in range(num_vectors_recall)]

        # Insert in batches
        for batch_start in range(0, num_vectors_recall, 500):
            batch_end = min(batch_start + 500, num_vectors_recall)
            batch = []
            for i in range(batch_start, batch_end):
                batch.append({"id": i + 1, "values": all_vectors[i]})
            bulk_insert(coll, batch)

        # Generate query vectors
        qrng = random.Random(1400)
        query_vectors = [rand_gaussian_vec(DIMENSION, qrng) for _ in range(num_queries_recall)]

        # Compute brute-force ground truth (cosine similarity)
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        ground_truth = []
        for qvec in query_vectors:
            sims = []
            for idx, dvec in enumerate(all_vectors):
                sims.append((idx + 1, cosine_sim(qvec, dvec)))  # 1-based ID
            sims.sort(key=lambda x: -x[1])
            ground_truth.append({s[0] for s in sims[:k]})

        # Run searches and compute recall
        hits = 0
        total = 0
        for i, qvec in enumerate(query_vectors):
            resp = search(coll, qvec, k=k)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                retrieved_ids = {r["id"] for r in results}
                hits += len(retrieved_ids & ground_truth[i])
                total += k

        recall = hits / total if total > 0 else 0.0
        return recall
    finally:
        delete_collection(coll)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global SERVER_URL, NUM_VECTORS, DIMENSION, NUM_SEARCH_QUERIES, CONCURRENT_THREADS

    parser = argparse.ArgumentParser(description="SwarnDB Performance Benchmark")
    parser.add_argument("--url", default=SERVER_URL, help="Server base URL (default: http://localhost:8080)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 2K vectors, 200 queries")
    parser.add_argument("--container", default="swarndb", help="Docker container name for memory stats")
    parser.add_argument("--json", default="", help="Save results to JSON file")
    args = parser.parse_args()

    SERVER_URL = args.url

    if args.quick:
        NUM_VECTORS = 2000
        NUM_SEARCH_QUERIES = 200
        print("[QUICK MODE] 2K vectors, 200 queries\n")

    # Verify server is reachable
    try:
        resp = SESSION.get(f"{api_url()}/collections", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: Server returned {resp.status_code} at {api_url()}/collections")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {SERVER_URL}")
        sys.exit(1)

    print(f"Server: {SERVER_URL}")
    print(f"Vectors: {NUM_VECTORS}  Dimension: {DIMENSION}  Queries: {NUM_SEARCH_QUERIES}")
    print(f"ef_search: {EF_SEARCH}  Threads: {CONCURRENT_THREADS}")
    print()

    results = {}

    # --- 1. Single Insert ---
    print("[1/9] Single insert throughput ...")
    n_single = min(1000, NUM_VECTORS // 10)
    single_vps = bench_single_insert(n_single)
    results["single_insert_vps"] = round(single_vps, 1)
    print(f"       {single_vps:.1f} vec/s")

    # --- 2. Bulk Insert ---
    print("[2/9] Bulk insert throughput ...")
    n_bulk = NUM_VECTORS - n_single
    if args.quick:
        n_bulk = 1800
    bulk_vps = bench_bulk_insert(n_bulk, batch_size=500)
    results["bulk_insert_vps"] = round(bulk_vps, 1)
    print(f"       {bulk_vps:.1f} vec/s")

    # --- 3. Search QPS (sequential) ---
    print("[3/9] Search QPS + latency (sequential) ...")
    n_search_vecs = NUM_VECTORS // 2
    search_qps, search_p50, search_p99 = bench_search(NUM_SEARCH_QUERIES, n_search_vecs)
    results["search_qps"] = round(search_qps, 1)
    results["search_p50_ms"] = round(search_p50, 2)
    results["search_p99_ms"] = round(search_p99, 2)
    print(f"       {search_qps:.1f} QPS  p50={search_p50:.2f}ms  p99={search_p99:.2f}ms")

    # --- 4. Concurrent Search QPS ---
    print(f"[4/9] Concurrent search QPS ({CONCURRENT_THREADS} threads) ...")
    conc_qps = bench_concurrent_search_qps(NUM_SEARCH_QUERIES, n_search_vecs)
    results["concurrent_search_qps"] = round(conc_qps, 1)
    print(f"       {conc_qps:.1f} QPS")

    # --- 5. Filtered Search ---
    print("[5/9] Filtered search QPS ...")
    filt_qps = bench_filtered_search(NUM_SEARCH_QUERIES, n_search_vecs)
    results["filtered_search_qps"] = round(filt_qps, 1)
    print(f"       {filt_qps:.1f} queries/s")

    # --- 6. Concurrent Search During Write ---
    print("[6/9] Concurrent search during write ...")
    n_conc_search = NUM_SEARCH_QUERIES // 2
    n_conc_write = NUM_VECTORS // 5
    if args.quick:
        n_conc_search = 100
        n_conc_write = 400
    conc_p99 = bench_concurrent_search_write(n_conc_search, n_conc_write)
    results["concurrent_search_p99_ms"] = round(conc_p99, 2)
    print(f"       p99={conc_p99:.2f}ms")

    # --- 7. Graph Search ---
    print("[7/9] Graph compute + search ...")
    n_graph_vecs = min(3000, NUM_VECTORS // 3)
    n_graph_q = NUM_SEARCH_QUERIES // 2
    if args.quick:
        n_graph_vecs = 600
        n_graph_q = 100
    graph_qps = bench_graph_search(n_graph_vecs, n_graph_q)
    results["graph_search_qps"] = round(graph_qps, 1)
    print(f"       {graph_qps:.1f} queries/s")

    # --- 8. Memory Usage ---
    print("[8/9] Memory usage (Docker) ...")
    mem_base, mem_loaded = bench_memory(args.container, NUM_VECTORS)
    results["memory_base_mb"] = round(mem_base, 1)
    results["memory_loaded_mb"] = round(mem_loaded, 1)
    if mem_base > 0 or mem_loaded > 0:
        print(f"       base={mem_base:.1f} MB  loaded={mem_loaded:.1f} MB")
    else:
        print(f"       (Docker stats unavailable -- container '{args.container}' not found)")

    # --- 9. Recall@10 ---
    print("[9/9] Recall@10 ...")
    n_recall_vecs = NUM_VECTORS // 2
    n_recall_q = min(100, NUM_SEARCH_QUERIES // 10)
    if args.quick:
        n_recall_vecs = 1000
        n_recall_q = 50
    recall = bench_recall(n_recall_vecs, n_recall_q, k=10)
    results["recall_at_10"] = round(recall, 3)
    print(f"       {recall:.3f}")
    print("       (Note: random vectors give lower recall than real embeddings)")

    # --- Output ---
    print()
    print("=" * 50)
    print("  SwarnDB Performance Benchmark Results")
    print("=" * 50)
    print(f"Single Insert:         {results['single_insert_vps']:.0f} vec/s")
    print(f"Bulk Insert:           {results['bulk_insert_vps']:.0f} vec/s")
    print(f"Search QPS (seq):      {results['search_qps']:.0f} queries/s")
    print(f"Search p50:            {results['search_p50_ms']:.2f} ms")
    print(f"Search p99:            {results['search_p99_ms']:.2f} ms")
    print(f"Search QPS (conc):     {results['concurrent_search_qps']:.0f} queries/s ({CONCURRENT_THREADS}t)")
    print(f"Filtered Search:       {results['filtered_search_qps']:.0f} queries/s")
    print(f"Concurrent Search p99: {results['concurrent_search_p99_ms']:.2f} ms (during writes)")
    print(f"Graph Search QPS:      {results['graph_search_qps']:.0f} queries/s")
    if results['memory_base_mb'] > 0 or results['memory_loaded_mb'] > 0:
        print(f"Memory (base):         {results['memory_base_mb']:.0f} MB")
        print(f"Memory (loaded):       {results['memory_loaded_mb']:.0f} MB")
    else:
        print("Memory (base):         N/A (Docker stats unavailable)")
        print("Memory (loaded):       N/A (Docker stats unavailable)")
    print(f"Recall@10:             {results['recall_at_10']:.3f} (random vecs; real data ~5-15% higher)")
    print("=" * 50)

    # --- JSON output ---
    if args.json:
        output = {
            "benchmark": "SwarnDB Performance Benchmark",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": {
                "server_url": SERVER_URL,
                "num_vectors": NUM_VECTORS,
                "dimension": DIMENSION,
                "num_search_queries": NUM_SEARCH_QUERIES,
                "concurrent_threads": CONCURRENT_THREADS,
                "ef_search": EF_SEARCH,
                "quick_mode": args.quick,
            },
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json}")

    # Clean up session
    SESSION.close()


if __name__ == "__main__":
    main()
