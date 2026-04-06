#!/usr/bin/env python3
"""
Concurrent reads during snapshot persistence test.

Validates that search queries never fail while vectors are being inserted
(which triggers snapshot activity). Simulates a live-search-during-maintenance
scenario.

Usage:
    python test_persistence_concurrent.py
"""

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

BASE_URL = "http://localhost:8080/api/v1"
COLLECTION = "live_search"
DIM = 128
INITIAL_VECTORS = 10_000
ADDITIONAL_VECTORS = 1_000
BATCH_SIZE = 500
RUN_DURATION_SEC = 30
QUERY_INTERVAL_SEC = 0.01  # 1 query per 10ms

# Shared state for the background query thread
query_stats = {
    "total": 0,
    "failed": 0,
    "latencies": [],
}
stop_event = threading.Event()
stats_lock = threading.Lock()


def setup_collection():
    """Create collection and insert initial 10000 vectors."""
    # Clean up from prior run
    requests.delete(f"{BASE_URL}/collections/{COLLECTION}")

    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": COLLECTION,
        "dimension": DIM,
        "distance_metric": "cosine",
    })
    if resp.status_code != 200 or not resp.json().get("success"):
        print(f"[FAIL] Could not create collection: {resp.status_code} {resp.text[:200]}")
        return False

    rng = np.random.default_rng(seed=42)
    num_batches = INITIAL_VECTORS // BATCH_SIZE

    for batch_num in range(num_batches):
        vectors = []
        for _ in range(BATCH_SIZE):
            vec = rng.standard_normal(DIM).tolist()
            vectors.append({"values": vec, "metadata": {"phase": "initial"}})

        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/bulk",
            json={"vectors": vectors},
        )
        if resp.status_code != 200:
            print(f"[FAIL] Batch {batch_num} insert failed: {resp.status_code}")
            return False

        pct = (batch_num + 1) * 100 // num_batches
        print(f"  Setup: inserted batch {batch_num + 1}/{num_batches} ({pct}%)", end="\r")

    print()

    # Verify count
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = resp.json().get("vector_count", 0)
    if count != INITIAL_VECTORS:
        print(f"[FAIL] Expected {INITIAL_VECTORS} vectors, got {count}")
        return False

    print(f"  Setup complete: {count} vectors in '{COLLECTION}'")
    return True


def background_query_worker():
    """Continuously send search queries until stop_event is set."""
    rng = np.random.default_rng(seed=777)

    while not stop_event.is_set():
        query_vec = rng.standard_normal(DIM).tolist()
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{BASE_URL}/collections/{COLLECTION}/search",
                json={"query": query_vec, "k": 10},
                timeout=5.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            with stats_lock:
                query_stats["total"] += 1
                query_stats["latencies"].append(latency_ms)
                if resp.status_code != 200:
                    query_stats["failed"] += 1
                else:
                    results = resp.json().get("results", [])
                    if len(results) == 0:
                        query_stats["failed"] += 1

        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            with stats_lock:
                query_stats["total"] += 1
                query_stats["failed"] += 1
                query_stats["latencies"].append(latency_ms)

        # Throttle to ~1 query per 10ms
        elapsed = time.perf_counter() - t0
        sleep_time = QUERY_INTERVAL_SEC - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def insert_additional_vectors():
    """Insert additional vectors rapidly to trigger snapshot activity."""
    rng = np.random.default_rng(seed=999)
    inserted = 0
    errors = 0
    num_batches = ADDITIONAL_VECTORS // BATCH_SIZE

    for batch_num in range(num_batches):
        vectors = []
        for _ in range(BATCH_SIZE):
            vec = rng.standard_normal(DIM).tolist()
            vectors.append({"values": vec, "metadata": {"phase": "concurrent"}})

        try:
            resp = requests.post(
                f"{BASE_URL}/collections/{COLLECTION}/vectors/bulk",
                json={"vectors": vectors},
                timeout=30.0,
            )
            if resp.status_code == 200:
                inserted += resp.json().get("inserted_count", 0)
            else:
                errors += BATCH_SIZE
        except Exception:
            errors += BATCH_SIZE

    return inserted, errors


def run_test():
    """Main test orchestration."""
    print("=" * 60)
    print("SwarnDB Concurrent Reads During Snapshot Test")
    print(f"Target: {BASE_URL}")
    print(f"Initial vectors: {INITIAL_VECTORS}, Additional: {ADDITIONAL_VECTORS}")
    print(f"Run duration: {RUN_DURATION_SEC}s, Query interval: {QUERY_INTERVAL_SEC * 1000:.0f}ms")
    print("=" * 60)
    print()

    # Step 1: Setup
    print("[Step 1] Setting up collection and inserting initial vectors...")
    if not setup_collection():
        print("\n[FAIL] Setup failed. Aborting.")
        return False

    # Step 2: Start background query thread
    print(f"\n[Step 2] Starting background query thread...")
    executor = ThreadPoolExecutor(max_workers=1)
    query_future = executor.submit(background_query_worker)

    # Step 3: Insert additional vectors in the main thread
    print(f"[Step 3] Inserting {ADDITIONAL_VECTORS} additional vectors rapidly...")
    inserted, insert_errors = insert_additional_vectors()
    print(f"  Inserted {inserted} additional vectors ({insert_errors} errors)")

    # Step 4: Let background queries continue for the remaining duration
    print(f"\n[Step 4] Running background queries for {RUN_DURATION_SEC} seconds total...")
    start_time = time.time()
    while time.time() - start_time < RUN_DURATION_SEC:
        with stats_lock:
            total = query_stats["total"]
            failed = query_stats["failed"]
        elapsed = time.time() - start_time
        print(f"  Elapsed: {elapsed:.0f}s  Queries: {total}  Failed: {failed}", end="\r")
        time.sleep(1.0)

    print()

    # Step 5: Stop background thread
    print("\n[Step 5] Stopping background query thread...")
    stop_event.set()
    query_future.result(timeout=10)
    executor.shutdown(wait=True)

    # Step 6: Compute and report results
    print("\n[Step 6] Computing results...")
    with stats_lock:
        total_queries = query_stats["total"]
        failed_queries = query_stats["failed"]
        latencies = np.array(query_stats["latencies"])

    if len(latencies) == 0:
        print("[FAIL] No queries were executed.")
        return False

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))
    max_lat = float(np.max(latencies))

    print()
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"  Total queries:   {total_queries}")
    print(f"  Failed queries:  {failed_queries}")
    print(f"  Mean latency:    {mean_lat:.2f} ms")
    print(f"  p50 latency:     {p50:.2f} ms")
    print(f"  p95 latency:     {p95:.2f} ms")
    print(f"  p99 latency:     {p99:.2f} ms")
    print(f"  Max latency:     {max_lat:.2f} ms")
    print("-" * 60)

    # Step 7: Assertions
    all_pass = True

    if failed_queries == 0:
        print("[PASS] Zero failed queries")
    else:
        print(f"[FAIL] {failed_queries} queries failed (expected 0)")
        all_pass = False

    if p99 < 500.0:
        print(f"[PASS] p99 latency {p99:.2f}ms < 500ms (no extreme spikes)")
    else:
        print(f"[FAIL] p99 latency {p99:.2f}ms >= 500ms (spike during snapshot)")
        all_pass = False

    return all_pass


def cleanup():
    """Delete the test collection."""
    print("\nCleaning up...")
    resp = requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    if resp.status_code == 200:
        print(f"  Collection '{COLLECTION}' deleted.")
    else:
        print(f"  Cleanup warning: status={resp.status_code}")


def main():
    try:
        passed = run_test()
    except Exception as e:
        print(f"\n[FAIL] Unhandled exception: {e}")
        passed = False
    finally:
        cleanup()

    print()
    print("=" * 60)
    if passed:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
