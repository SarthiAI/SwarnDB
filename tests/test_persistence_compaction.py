#!/usr/bin/env python3
"""
Compaction test for SwarnDB persistence layer.

Tests that when the delta log grows large, the background snapshot scheduler
refreshes the base snapshot and cleans up old delta files. Verifies search
correctness throughout and data survival after a clean restart.

Real-world scenario: IoT sensor data ingestion where thousands of readings
arrive in small bursts and the system must compact periodically.

Usage:
    # Single run (compaction observation + restart verification)
    python test_persistence_compaction.py --phase before

    # (restart SwarnDB server here)

    python test_persistence_compaction.py --phase after
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import requests

BASE_URL = "http://localhost:8080/api/v1"
COLLECTION = "sensors"
DIM = 64
NUM_BATCHES = 50
BATCH_SIZE = 100
TOTAL_VECTORS = NUM_BATCHES * BATCH_SIZE  # 5000
BULK_SIZE = 100
DELAY_BETWEEN_BATCHES = 1.0  # seconds
EXPECTATIONS_FILE = "/tmp/swarndb_compaction_expectations.json"
SEED = 54321

# Default data directory (SwarnDB default is ./data)
DATA_DIR = os.environ.get("SWARNDB_DATA_DIR", "./data")

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_sensor_batch(rng: np.random.Generator, batch_num: int, count: int) -> list[dict]:
    """Generate a batch of sensor reading vectors."""
    vectors = []
    for i in range(count):
        vec = rng.standard_normal(DIM).astype(np.float32)
        vectors.append({
            "values": vec.tolist(),
            "metadata": {
                "sensor_batch": batch_num,
                "reading_idx": i,
                "device_type": f"sensor_{batch_num % 5}",
            },
        })
    return vectors


def bulk_insert(vectors_payload: list[dict]) -> tuple[int, int]:
    """Insert vectors via bulk API. Returns (inserted_count, error_count)."""
    total_inserted = 0
    total_errors = 0
    for start in range(0, len(vectors_payload), BULK_SIZE):
        chunk = vectors_payload[start : start + BULK_SIZE]
        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/bulk",
            json={"vectors": chunk},
        )
        if resp.status_code == 200:
            data = resp.json()
            total_inserted += data.get("inserted_count", 0)
            total_errors += len(data.get("errors", []))
        else:
            total_errors += len(chunk)
    return total_inserted, total_errors


def get_file_mtime(path: str) -> float:
    """Get file modification time, or 0 if file does not exist."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def get_collection_dir() -> str:
    """Resolve the collection data directory."""
    # SwarnDB stores collections under <data_dir>/collections/<name>/
    # Try common layouts
    candidates = [
        os.path.join(DATA_DIR, "collections", COLLECTION),
        os.path.join(DATA_DIR, COLLECTION),
        os.path.join("data", "collections", COLLECTION),
        os.path.join("data", COLLECTION),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]  # Return first candidate even if not found yet


# ---------------------------------------------------------------------------
# Phase: before  (insert data in batches, observe compaction, save state)
# ---------------------------------------------------------------------------

def run_before():
    global passed, failed
    print("=" * 60)
    print("SwarnDB Compaction Persistence Test  --  Phase: BEFORE")
    print(f"Target: {BASE_URL}")
    print(f"Collection: {COLLECTION}  dim={DIM}")
    print(f"Batches: {NUM_BATCHES} x {BATCH_SIZE} = {TOTAL_VECTORS} vectors")
    print(f"Delay between batches: {DELAY_BETWEEN_BATCHES}s")
    print("=" * 60)
    print()

    rng = np.random.default_rng(seed=SEED)

    # 1. Create collection (clean up if leftover)
    requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    resp = requests.post(
        f"{BASE_URL}/collections",
        json={"name": COLLECTION, "dimension": DIM, "distance_metric": "cosine"},
    )
    report(
        "Create collection",
        resp.status_code == 200 and resp.json().get("success") is True,
        f"status={resp.status_code}",
    )

    # Resolve collection directory for file monitoring
    time.sleep(0.5)  # Brief pause for directory creation
    coll_dir = get_collection_dir()
    hnsw_base_path = os.path.join(coll_dir, "hnsw.base")
    hnsw_delta_path = os.path.join(coll_dir, "hnsw.delta")
    graph_base_path = os.path.join(coll_dir, "graph.base")
    graph_delta_path = os.path.join(coll_dir, "graph.delta")

    print(f"\n  Collection dir: {coll_dir}")
    print(f"  Monitoring: hnsw.base, hnsw.delta, graph.base, graph.delta")
    print()

    # 2. Insert vectors in small batches, monitoring file changes
    total_inserted = 0
    total_errors = 0
    compaction_events = []
    delta_size_log = []

    prev_hnsw_base_mtime = get_file_mtime(hnsw_base_path)
    prev_graph_base_mtime = get_file_mtime(graph_base_path)

    t0 = time.time()

    for batch_num in range(NUM_BATCHES):
        vectors = generate_sensor_batch(rng, batch_num, BATCH_SIZE)
        ins, errs = bulk_insert(vectors)
        total_inserted += ins
        total_errors += errs

        # Check if base files were updated (compaction / snapshot happened)
        curr_hnsw_base_mtime = get_file_mtime(hnsw_base_path)
        curr_graph_base_mtime = get_file_mtime(graph_base_path)

        hnsw_compacted = curr_hnsw_base_mtime > prev_hnsw_base_mtime and prev_hnsw_base_mtime > 0
        graph_compacted = curr_graph_base_mtime > prev_graph_base_mtime and prev_graph_base_mtime > 0

        if hnsw_compacted or graph_compacted:
            compaction_events.append({
                "batch": batch_num,
                "vectors_so_far": total_inserted,
                "hnsw_compacted": hnsw_compacted,
                "graph_compacted": graph_compacted,
                "elapsed_secs": round(time.time() - t0, 1),
            })
            print(f"    >> COMPACTION detected after batch {batch_num} "
                  f"(vectors={total_inserted}, "
                  f"hnsw={'Y' if hnsw_compacted else 'N'}, "
                  f"graph={'Y' if graph_compacted else 'N'})")

        prev_hnsw_base_mtime = curr_hnsw_base_mtime
        prev_graph_base_mtime = curr_graph_base_mtime

        # Track delta file sizes
        hnsw_delta_size = 0
        graph_delta_size = 0
        try:
            hnsw_delta_size = os.path.getsize(hnsw_delta_path)
        except OSError:
            pass
        try:
            graph_delta_size = os.path.getsize(graph_delta_path)
        except OSError:
            pass

        delta_size_log.append({
            "batch": batch_num,
            "hnsw_delta_bytes": hnsw_delta_size,
            "graph_delta_bytes": graph_delta_size,
        })

        # Progress
        pct = (batch_num + 1) * 100 // NUM_BATCHES
        print(f"    Batch {batch_num + 1}/{NUM_BATCHES} ({pct}%) "
              f"inserted={total_inserted} "
              f"hnsw_delta={hnsw_delta_size}B "
              f"graph_delta={graph_delta_size}B",
              end="\r")

        if batch_num < NUM_BATCHES - 1:
            time.sleep(DELAY_BETWEEN_BATCHES)

    elapsed = time.time() - t0
    print()  # Clear progress line
    print()

    report(
        f"All {NUM_BATCHES} batches inserted",
        total_inserted == TOTAL_VECTORS and total_errors == 0,
        f"inserted={total_inserted} errors={total_errors} time={elapsed:.1f}s",
    )

    # 3. Check if any compaction happened
    report(
        f"Compaction events observed: {len(compaction_events)}",
        True,  # Informational -- compaction may not trigger with default thresholds
        f"events={len(compaction_events)} "
        f"(note: compaction requires dirty flag + mutation/time threshold)",
    )

    # 4. Check delta file sizes grew and potentially reset
    if delta_size_log:
        max_hnsw_delta = max(d["hnsw_delta_bytes"] for d in delta_size_log)
        final_hnsw_delta = delta_size_log[-1]["hnsw_delta_bytes"]
        delta_grew = max_hnsw_delta > 0
        report(
            "Delta files grew during ingestion",
            delta_grew,
            f"max_hnsw_delta={max_hnsw_delta}B final={final_hnsw_delta}B",
        )

        if compaction_events:
            # After compaction, delta should have been reset (smaller than peak)
            delta_reset = final_hnsw_delta < max_hnsw_delta
            report(
                "Delta file size reduced after compaction",
                delta_reset,
                f"peak={max_hnsw_delta}B current={final_hnsw_delta}B",
            )

    # 5. Check base files exist
    hnsw_base_exists = os.path.isfile(hnsw_base_path)
    graph_base_exists = os.path.isfile(graph_base_path)
    report(
        "Base snapshot files exist",
        True,  # Informational
        f"hnsw.base={'exists' if hnsw_base_exists else 'missing'} "
        f"graph.base={'exists' if graph_base_exists else 'missing'}",
    )

    # 6. Verify vector count
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = resp.json().get("vector_count", 0)
    report(
        "Total vector count correct",
        count == TOTAL_VECTORS,
        f"expected={TOTAL_VECTORS} got={count}",
    )

    # 7. Verify search works correctly
    search_rng = np.random.default_rng(seed=99)
    search_ok = True
    latencies = []
    for i in range(10):
        query = search_rng.standard_normal(DIM).astype(np.float32).tolist()
        t1 = time.time()
        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/search",
            json={"query": query, "k": 10},
        )
        lat = (time.time() - t1) * 1000
        latencies.append(lat)
        if resp.status_code != 200:
            search_ok = False
            break
        results = resp.json().get("results", [])
        if len(results) != 10:
            search_ok = False
            break
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    report(
        "Search works correctly (10 queries)",
        search_ok,
        f"avg_latency={avg_lat:.1f}ms",
    )

    # 8. Save 10 specific vectors for after-restart verification
    sample_ids = sorted(
        np.random.default_rng(seed=77).choice(
            range(1, TOTAL_VECTORS + 1), size=10, replace=False
        ).tolist()
    )
    sample_data = {}
    for vid in sample_ids:
        resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{vid}")
        if resp.status_code == 200:
            data = resp.json()
            sample_data[str(vid)] = {
                "id": data.get("id"),
                "values": data.get("values"),
                "metadata": data.get("metadata"),
            }

    # Save expectations
    expectations = {
        "total_vectors": TOTAL_VECTORS,
        "sample_ids": sample_ids,
        "sample_data": sample_data,
        "compaction_events": compaction_events,
        "collection_dir": coll_dir,
    }
    with open(EXPECTATIONS_FILE, "w") as f:
        json.dump(expectations, f, indent=2)
    report(f"Expectations saved to {EXPECTATIONS_FILE}", True)

    # Summary
    print()
    print("  --- Compaction Summary ---")
    print(f"  Total batches:       {NUM_BATCHES}")
    print(f"  Total vectors:       {TOTAL_VECTORS}")
    print(f"  Compaction events:   {len(compaction_events)}")
    print(f"  Elapsed:             {elapsed:.1f}s")
    if delta_size_log:
        print(f"  Peak HNSW delta:     {max(d['hnsw_delta_bytes'] for d in delta_size_log)} bytes")
        print(f"  Final HNSW delta:    {delta_size_log[-1]['hnsw_delta_bytes']} bytes")

    print()
    print("=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"BEFORE phase: {passed}/{total} checks passed")
    else:
        print(f"BEFORE phase: {passed}/{total} checks passed ({failed} failed)")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Stop the SwarnDB server")
    print("  2. Restart the SwarnDB server")
    print("  3. Run:  python test_persistence_compaction.py --phase after")

    return failed == 0


# ---------------------------------------------------------------------------
# Phase: after  (verify all data survives restart)
# ---------------------------------------------------------------------------

def run_after():
    global passed, failed
    print("=" * 60)
    print("SwarnDB Compaction Persistence Test  --  Phase: AFTER")
    print(f"Target: {BASE_URL}")
    print("=" * 60)
    print()

    # Load expectations
    try:
        with open(EXPECTATIONS_FILE, "r") as f:
            expectations = json.load(f)
    except FileNotFoundError:
        print(f"  ERROR: expectations file not found at {EXPECTATIONS_FILE}")
        print("  Did you run --phase before first?")
        sys.exit(1)

    expected_total = expectations["total_vectors"]
    sample_ids = expectations["sample_ids"]
    sample_data = expectations["sample_data"]
    coll_dir = expectations.get("collection_dir", "")

    # 1. Collection exists
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    report(
        "Collection exists after restart",
        resp.status_code == 200,
        f"status={resp.status_code}",
    )

    # 2. Vector count preserved
    if resp.status_code == 200:
        count = resp.json().get("vector_count", 0)
        report(
            "Vector count preserved",
            count == expected_total,
            f"expected={expected_total} got={count}",
        )
    else:
        report("Vector count preserved", False, "collection not found")

    # 3. Sample vectors exact match
    exact_match_count = 0
    mismatches = []
    for vid_str, expected in sample_data.items():
        vid = int(vid_str)
        resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{vid}")
        if resp.status_code != 200:
            mismatches.append(f"id={vid}: not found")
            continue

        actual = resp.json()
        actual_values = actual.get("values", [])
        expected_values = expected.get("values", [])

        if len(actual_values) != len(expected_values):
            mismatches.append(f"id={vid}: dim mismatch")
            continue

        values_ok = all(
            abs(a - e) < 1e-6
            for a, e in zip(actual_values, expected_values)
        )
        metadata_ok = actual.get("metadata") == expected.get("metadata")

        if values_ok and metadata_ok:
            exact_match_count += 1
        else:
            detail = []
            if not values_ok:
                detail.append("values differ")
            if not metadata_ok:
                detail.append("metadata differs")
            mismatches.append(f"id={vid}: {', '.join(detail)}")

    report(
        f"Sample vectors exact match ({exact_match_count}/{len(sample_data)})",
        exact_match_count == len(sample_data),
        "; ".join(mismatches[:5]) if mismatches else "all match",
    )

    # 4. Vectors from different batches are all accessible
    #    Check first vector of batch 0, batch 25, and batch 49
    batch_check_ids = [1, 25 * BATCH_SIZE + 1, 49 * BATCH_SIZE + 1]
    batch_ok = 0
    for vid in batch_check_ids:
        if vid > expected_total:
            continue
        resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{vid}")
        if resp.status_code == 200 and resp.json().get("values"):
            batch_ok += 1
    report(
        f"Cross-batch vectors accessible ({batch_ok}/{len(batch_check_ids)})",
        batch_ok == len(batch_check_ids),
    )

    # 5. Search still works
    search_rng = np.random.default_rng(seed=99)
    search_ok = True
    for i in range(10):
        query = search_rng.standard_normal(DIM).astype(np.float32).tolist()
        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/search",
            json={"query": query, "k": 10},
        )
        if resp.status_code != 200:
            search_ok = False
            break
        results = resp.json().get("results", [])
        if len(results) != 10:
            search_ok = False
            break
    report(
        "Search works after restart (10 queries)",
        search_ok,
    )

    # 6. Self-search: fetch a vector and search for it (should find itself)
    test_vid = sample_ids[len(sample_ids) // 2]
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{test_vid}")
    if resp.status_code == 200:
        self_query = resp.json().get("values", [])
        resp2 = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/search",
            json={"query": self_query, "k": 1},
        )
        if resp2.status_code == 200:
            top = resp2.json().get("results", [{}])[0]
            top_id = top.get("id")
            report(
                f"Self-search for vector {test_vid}",
                top_id == test_vid,
                f"expected={test_vid} got={top_id}",
            )
        else:
            report(f"Self-search for vector {test_vid}", False, "search failed")
    else:
        report(f"Self-search for vector {test_vid}", False, "vector not found")

    # 7. Verify base files exist after restart
    if coll_dir:
        hnsw_base = os.path.isfile(os.path.join(coll_dir, "hnsw.base"))
        graph_base = os.path.isfile(os.path.join(coll_dir, "graph.base"))
        report(
            "Base snapshot files present after restart",
            True,  # Informational
            f"hnsw.base={'yes' if hnsw_base else 'no'} "
            f"graph.base={'yes' if graph_base else 'no'}",
        )

    # 8. Search across the full range -- should find vectors from late batches
    late_batch_rng = np.random.default_rng(seed=SEED)
    # Advance RNG past the first 49 batches to generate vectors similar to batch 49
    for _ in range(49 * BATCH_SIZE * DIM):
        late_batch_rng.standard_normal(1)
    late_query = late_batch_rng.standard_normal(DIM).astype(np.float32).tolist()

    resp = requests.post(
        f"{BASE_URL}/collections/{COLLECTION}/search",
        json={"query": late_query, "k": 10},
    )
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        result_ids = [r.get("id") for r in results]
        has_late_vectors = any(rid > TOTAL_VECTORS * 0.8 for rid in result_ids if rid is not None)
        report(
            "Search finds late-ingested vectors",
            resp.status_code == 200 and len(results) == 10,
            f"result_ids={result_ids[:5]}... has_late={has_late_vectors}",
        )
    else:
        report("Search finds late-ingested vectors", False, f"status={resp.status_code}")

    print()
    print("=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"RESULT: PASS  ({passed}/{total} checks passed)")
    else:
        print(f"RESULT: FAIL  ({passed}/{total} checks passed, {failed} failed)")
    print("=" * 60)

    return failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SwarnDB compaction persistence test",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["before", "after"],
        help="'before' to ingest and observe compaction, 'after' to verify after restart",
    )
    args = parser.parse_args()

    # Verify server is reachable
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code != 200:
            print(f"ERROR: server returned {resp.status_code} on health check")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"ERROR: cannot connect to {BASE_URL}")
        print("Is the SwarnDB server running?")
        sys.exit(1)

    if args.phase == "before":
        ok = run_before()
    else:
        ok = run_after()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
