#!/usr/bin/env python3
"""
SwarnDB Persistence Test: Corruption Handling
==============================================
Verifies that the server gracefully falls back to a full rebuild when
snapshot files (hnsw.base, graph.base) are corrupted.

Usage (three-phase approach -- restart the server between phases):

    python test_persistence_corruption.py --phase setup
    # (restart the server cleanly so snapshots are written)
    python test_persistence_corruption.py --phase corrupt
    # (restart the server -- it should detect corruption and rebuild)
    python test_persistence_corruption.py --phase verify

REST API: http://localhost:8080
Data directory: ./data  (override with SWARNDB_DATA_DIR)
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("SWARNDB_BASE_URL", "http://localhost:8080/api/v1")
DATA_DIR = os.environ.get("SWARNDB_DATA_DIR", "./data")
COLLECTION = "corruption_test"
DIM = 128
NUM_VECTORS = 2000
BATCH_SIZE = 500
SEED = 12345
EXPECTATIONS_FILE = os.path.join(DATA_DIR, COLLECTION, "_test_expectations.json")

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


# ---------------------------------------------------------------------------
# Phase: setup
# ---------------------------------------------------------------------------

def phase_setup():
    """Create collection, insert 2000 vectors, save expectations."""
    print("\n=== PHASE: SETUP ===\n")

    # Clean up any leftover collection
    requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    time.sleep(0.5)

    # 1. Create collection
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": COLLECTION,
        "dimension": DIM,
        "distance_metric": "cosine",
    })
    ok = resp.status_code == 200 and resp.json().get("success") is True
    report("Create collection", ok, f"status={resp.status_code}")
    if not ok:
        print("  Cannot proceed without collection. Aborting.")
        return False

    # 2. Insert vectors in batches
    rng = np.random.default_rng(seed=SEED)
    all_ids = []
    total_inserted = 0

    for batch_num in range(NUM_VECTORS // BATCH_SIZE):
        vectors = []
        for i in range(BATCH_SIZE):
            vec = rng.standard_normal(DIM).tolist()
            vectors.append({
                "values": vec,
                "metadata": {"batch": batch_num, "idx": i},
            })
        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/bulk",
            json={"vectors": vectors},
        )
        if resp.status_code == 200:
            data = resp.json()
            inserted = data.get("inserted_count", 0)
            total_inserted += inserted
            ids = data.get("ids", [])
            all_ids.extend(ids)
        else:
            report(f"Bulk insert batch {batch_num}", False, resp.text[:200])
            return False
        print(f"    Batch {batch_num + 1}/{NUM_VECTORS // BATCH_SIZE} done", end="\r")

    print()
    report(f"Insert {NUM_VECTORS} vectors", total_inserted == NUM_VECTORS,
           f"inserted={total_inserted}")

    # 3. Verify count
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = resp.json().get("vector_count", -1)
    report("Verify vector count", count == NUM_VECTORS, f"count={count}")

    # 4. Save expectations (vector IDs + a few sample vectors for search verification)
    rng_search = np.random.default_rng(seed=99)
    sample_queries = [rng_search.standard_normal(DIM).tolist() for _ in range(5)]

    expectations = {
        "num_vectors": NUM_VECTORS,
        "sample_ids": all_ids[:20] if all_ids else list(range(1, 21)),
        "sample_queries": sample_queries,
    }

    collection_dir = os.path.join(DATA_DIR, COLLECTION)
    os.makedirs(collection_dir, exist_ok=True)
    with open(EXPECTATIONS_FILE, "w") as f:
        json.dump(expectations, f)
    report("Save expectations", os.path.exists(EXPECTATIONS_FILE))

    print("\n  Setup complete. Please restart the server cleanly (SIGTERM)")
    print("  so that snapshots (hnsw.base, graph.base) are written.")
    print("  Then run: python test_persistence_corruption.py --phase corrupt")
    return True


# ---------------------------------------------------------------------------
# Phase: corrupt
# ---------------------------------------------------------------------------

def phase_corrupt():
    """Corrupt the hnsw.base file by overwriting bytes 100-200 with random data."""
    print("\n=== PHASE: CORRUPT ===\n")

    collection_dir = os.path.join(DATA_DIR, COLLECTION)
    hnsw_path = os.path.join(collection_dir, "hnsw.base")

    if not os.path.exists(hnsw_path):
        report("hnsw.base exists", False,
               f"File not found at {hnsw_path}. Did you restart the server after setup?")
        return False

    file_size = os.path.getsize(hnsw_path)
    report("hnsw.base exists", True, f"size={file_size} bytes")

    if file_size < 300:
        report("hnsw.base large enough to corrupt", False,
               f"File is only {file_size} bytes, expected at least 300")
        return False

    # Corrupt bytes 100-200 with random data
    random.seed(42)
    garbage = bytes([random.randint(0, 255) for _ in range(100)])

    with open(hnsw_path, "r+b") as f:
        f.seek(100)
        f.write(garbage)
        f.flush()
        os.fsync(f.fileno())

    report("Corrupt hnsw.base (bytes 100-200)", True, "100 random bytes written")

    print("\n  hnsw.base has been corrupted.")
    print("  Please restart the server. It should detect corruption and fall back")
    print("  to a full rebuild from the WAL/vectors.")
    print("  Then run: python test_persistence_corruption.py --phase verify")
    return True


# ---------------------------------------------------------------------------
# Phase: verify
# ---------------------------------------------------------------------------

def phase_verify():
    """After restart with corrupted snapshot, verify data integrity and search."""
    print("\n=== PHASE: VERIFY (post-corruption restart) ===\n")

    # Load expectations
    if not os.path.exists(EXPECTATIONS_FILE):
        print(f"  ERROR: Expectations file not found at {EXPECTATIONS_FILE}")
        print("  Did you run --phase setup first?")
        return False

    with open(EXPECTATIONS_FILE) as f:
        expectations = json.load(f)

    expected_count = expectations["num_vectors"]
    sample_ids = expectations["sample_ids"]
    sample_queries = expectations["sample_queries"]

    # 1. Server should be healthy
    try:
        resp = requests.get(f"{BASE_URL.rsplit('/api', 1)[0]}/health", timeout=10)
        report("Server is healthy after restart", resp.status_code == 200,
               f"status={resp.status_code}")
    except requests.ConnectionError:
        report("Server is healthy after restart", False, "Connection refused")
        return False

    # 2. Collection should exist
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    ok = resp.status_code == 200
    report("Collection exists after restart", ok, f"status={resp.status_code}")
    if not ok:
        return False

    # 3. All vectors should be present
    count = resp.json().get("vector_count", -1)
    report("All vectors recovered", count == expected_count,
           f"expected={expected_count} got={count}")

    # 4. Individual vector retrieval
    vectors_ok = 0
    for vid in sample_ids[:5]:
        r = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{vid}")
        if r.status_code == 200:
            data = r.json()
            if len(data.get("values", [])) == DIM:
                vectors_ok += 1
    report("Sample vector retrieval", vectors_ok == min(5, len(sample_ids[:5])),
           f"{vectors_ok}/5 vectors OK")

    # 5. Search should work
    search_ok = 0
    for query in sample_queries:
        r = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/search",
            json={"query": query, "k": 10},
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if len(results) == 10:
                search_ok += 1
    report("Search works after corruption recovery", search_ok == len(sample_queries),
           f"{search_ok}/{len(sample_queries)} queries returned 10 results")

    # 6. Now test graph.base corruption (for completeness)
    print("\n  --- Now testing graph.base corruption ---\n")

    collection_dir = os.path.join(DATA_DIR, COLLECTION)
    graph_path = os.path.join(collection_dir, "graph.base")

    if os.path.exists(graph_path):
        file_size = os.path.getsize(graph_path)
        report("graph.base exists", True, f"size={file_size} bytes")

        if file_size >= 300:
            random.seed(99)
            garbage = bytes([random.randint(0, 255) for _ in range(100)])
            with open(graph_path, "r+b") as f:
                f.seek(100)
                f.write(garbage)
                f.flush()
                os.fsync(f.fileno())
            report("Corrupt graph.base (bytes 100-200)", True, "100 random bytes written")
            print("\n  graph.base has been corrupted.")
            print("  Restart the server one more time and re-run --phase verify")
            print("  to confirm the graph also recovers via full rebuild.")
        else:
            report("graph.base large enough to corrupt", False, f"only {file_size} bytes")
    else:
        report("graph.base exists", False,
               "File not found -- graph snapshot may not have been written")

    # 7. Clean up expectations file
    # (Keep it around for a potential second verify pass after graph corruption)

    return True


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup():
    """Delete the test collection."""
    print("\n  Cleaning up...")
    resp = requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    if resp.status_code == 200:
        print(f"  Collection '{COLLECTION}' deleted.")
    else:
        print(f"  Cleanup warning: status={resp.status_code}")

    # Remove expectations file
    if os.path.exists(EXPECTATIONS_FILE):
        os.remove(EXPECTATIONS_FILE)
        print("  Expectations file removed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SwarnDB Persistence Test: Corruption Handling"
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["setup", "corrupt", "verify", "cleanup"],
        help="Test phase to execute",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SwarnDB Persistence Test: Corruption Handling")
    print(f"Phase: {args.phase}")
    print(f"Target: {BASE_URL}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Collection: {COLLECTION} (dim={DIM}, n={NUM_VECTORS})")
    print("=" * 60)

    if args.phase == "setup":
        phase_setup()
    elif args.phase == "corrupt":
        phase_corrupt()
    elif args.phase == "verify":
        phase_verify()
    elif args.phase == "cleanup":
        cleanup()

    print()
    print("=" * 60)
    print(f"RESULT: {passed} PASSED, {failed} FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
