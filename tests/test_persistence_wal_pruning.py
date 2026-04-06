#!/usr/bin/env python3
"""
SwarnDB Persistence Test: WAL Pruning
======================================
Verifies that old WAL files (wal_*.log.old) are cleaned up after background
snapshots occur.

Usage:
    python test_persistence_wal_pruning.py

The server must be running at http://localhost:8080.
This script inserts enough vectors to trigger WAL rotations, waits for
a background snapshot, and then verifies that old WAL files are pruned.

REST API: http://localhost:8080
Data directory: ./data  (override with SWARNDB_DATA_DIR)

Tip: For faster results, start the server with lower snapshot thresholds:
    SWARNDB_SNAPSHOT_MUTATION_THRESHOLD=5000 \
    SWARNDB_SNAPSHOT_INTERVAL_SECS=30 \
    SWARNDB_SNAPSHOT_CHECK_INTERVAL_SECS=5 \
    vf-server
"""

import glob
import os
import sys
import time

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("SWARNDB_BASE_URL", "http://localhost:8080/api/v1")
DATA_DIR = os.environ.get("SWARNDB_DATA_DIR", "./data")
COLLECTION = "wal_prune_test"
DIM = 64
TOTAL_VECTORS = 110_000  # enough to trigger WAL rotations
BATCH_SIZE = 1000
SEED = 55555
MAX_SNAPSHOT_WAIT_SECS = 120  # max time to wait for background snapshot

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


def collection_dir():
    return os.path.join(DATA_DIR, COLLECTION)


def count_old_wal_files():
    """Count wal_*.log.old files in the collection directory."""
    pattern = os.path.join(collection_dir(), "wal_*.log.old")
    return len(glob.glob(pattern))


def list_old_wal_files():
    """List wal_*.log.old files in the collection directory."""
    pattern = os.path.join(collection_dir(), "wal_*.log.old")
    return sorted(glob.glob(pattern))


def server_healthy():
    """Check if the server is up and responding."""
    try:
        health_url = BASE_URL.rsplit("/api", 1)[0] + "/health"
        resp = requests.get(health_url, timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


# ---------------------------------------------------------------------------
# Main test flow
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SwarnDB Persistence Test: WAL Pruning")
    print(f"Target: {BASE_URL}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Collection: {COLLECTION} (dim={DIM}, n={TOTAL_VECTORS})")
    print("=" * 60)
    print()

    # Step 0: Check server health
    if not server_healthy():
        report("Server is running", False, "Cannot reach server")
        print("\n  Start the server first. For best results, use low snapshot thresholds.")
        sys.exit(1)
    report("Server is running", True)

    # Step 1: Create collection
    requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    time.sleep(0.5)

    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": COLLECTION,
        "dimension": DIM,
        "distance_metric": "cosine",
    })
    ok = resp.status_code == 200 and resp.json().get("success") is True
    report("Create collection", ok, f"status={resp.status_code}")
    if not ok:
        print("  Cannot proceed. Aborting.")
        sys.exit(1)

    # Step 2: Insert vectors in many small batches to cause WAL rotations
    print(f"\n  Inserting {TOTAL_VECTORS} vectors in batches of {BATCH_SIZE}...")
    rng = np.random.default_rng(seed=SEED)
    total_inserted = 0
    num_batches = TOTAL_VECTORS // BATCH_SIZE

    t0 = time.time()
    for batch_num in range(num_batches):
        vectors = []
        for i in range(BATCH_SIZE):
            vec = rng.standard_normal(DIM).tolist()
            vectors.append({
                "values": vec,
                "metadata": {"batch": batch_num},
            })
        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/bulk",
            json={"vectors": vectors},
        )
        if resp.status_code == 200:
            total_inserted += resp.json().get("inserted_count", 0)
        else:
            report(f"Batch {batch_num} insert", False, resp.text[:200])

        pct = (batch_num + 1) * 100 // num_batches
        print(f"    Batch {batch_num + 1}/{num_batches} ({pct}%)", end="\r")

    elapsed = time.time() - t0
    print()
    report(f"Insert {TOTAL_VECTORS} vectors",
           total_inserted == TOTAL_VECTORS,
           f"inserted={total_inserted} time={elapsed:.1f}s")

    # Step 3: Check for old WAL files
    time.sleep(1)  # brief pause for file system sync
    old_wal_count_initial = count_old_wal_files()
    old_wal_files = list_old_wal_files()
    report("WAL rotations occurred (old WAL files exist)",
           old_wal_count_initial > 0,
           f"count={old_wal_count_initial}")

    if old_wal_files:
        print(f"    Old WAL files found:")
        for f in old_wal_files[:10]:
            size = os.path.getsize(f)
            print(f"      {os.path.basename(f)} ({size} bytes)")
        if len(old_wal_files) > 10:
            print(f"      ... and {len(old_wal_files) - 10} more")

    if old_wal_count_initial == 0:
        print("\n  No old WAL files found. Possible reasons:")
        print("    - WAL max size is very large (no rotations happened)")
        print("    - Background snapshot already pruned them")
        print("    - Not enough data to trigger rotation")
        print("  Test may still pass if snapshot triggers pruning of future WAL files.")

    # Step 4: Wait for a background snapshot to occur
    print(f"\n  Waiting up to {MAX_SNAPSHOT_WAIT_SECS}s for background snapshot + WAL pruning...")
    print("  (The snapshot scheduler checks every SWARNDB_SNAPSHOT_CHECK_INTERVAL_SECS)")

    snapshot_occurred = False
    start_wait = time.time()
    check_interval = 5  # check every 5 seconds

    while time.time() - start_wait < MAX_SNAPSHOT_WAIT_SECS:
        # Check if hnsw.base exists (indicates a snapshot happened)
        hnsw_base = os.path.join(collection_dir(), "hnsw.base")
        if os.path.exists(hnsw_base):
            # Check if old WAL files were pruned
            current_old_count = count_old_wal_files()
            if current_old_count < old_wal_count_initial or current_old_count == 0:
                snapshot_occurred = True
                break

        elapsed_wait = time.time() - start_wait
        print(f"    Waiting... ({elapsed_wait:.0f}s / {MAX_SNAPSHOT_WAIT_SECS}s)"
              f" old_wal_files={count_old_wal_files()}", end="\r")
        time.sleep(check_interval)

    print()

    if snapshot_occurred:
        report("Background snapshot + WAL pruning detected", True,
               f"old_wal_files after pruning: {count_old_wal_files()}")
    else:
        # Even if we did not see pruning during the wait, check final state
        final_old_count = count_old_wal_files()
        hnsw_exists = os.path.exists(os.path.join(collection_dir(), "hnsw.base"))

        if hnsw_exists and final_old_count == 0:
            report("Background snapshot + WAL pruning detected", True,
                   "hnsw.base exists and no old WAL files remain")
        else:
            report("Background snapshot + WAL pruning detected", False,
                   f"hnsw.base={hnsw_exists} old_wal_files={final_old_count} "
                   f"(waited {MAX_SNAPSHOT_WAIT_SECS}s)")
            print("  Tip: Lower the snapshot thresholds:")
            print("    SWARNDB_SNAPSHOT_MUTATION_THRESHOLD=5000")
            print("    SWARNDB_SNAPSHOT_CHECK_INTERVAL_SECS=5")

    # Step 5: Verify search still works
    print()
    rng_search = np.random.default_rng(seed=99)
    search_ok = 0
    for _ in range(5):
        query = rng_search.standard_normal(DIM).tolist()
        r = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/search",
            json={"query": query, "k": 10},
        )
        if r.status_code == 200 and len(r.json().get("results", [])) == 10:
            search_ok += 1

    report("Search still works after WAL pruning", search_ok == 5,
           f"{search_ok}/5 queries returned 10 results")

    # Step 6: Verify vector count
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = resp.json().get("vector_count", -1)
    report("Vector count preserved", count == TOTAL_VECTORS,
           f"expected={TOTAL_VECTORS} got={count}")

    # Step 7: Clean up
    print("\n  Cleaning up...")
    resp = requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    if resp.status_code == 200:
        print(f"  Collection '{COLLECTION}' deleted.")
    else:
        print(f"  Cleanup warning: status={resp.status_code}")

    # Summary
    print()
    print("=" * 60)
    print(f"RESULT: {passed} PASSED, {failed} FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
