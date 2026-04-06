#!/usr/bin/env python3
"""
Recovery time measurement test for SwarnDB persistence layer.

Measures snapshot-based recovery vs full rebuild recovery, validating that
snapshots provide at least 3x faster startup.

Usage (run phases in order, restarting the server between them):

    # Phase 1: Insert 50K vectors and save expectations
    python test_persistence_recovery_time.py --phase setup

    # Phase 2: Restart server normally, then measure snapshot recovery
    python test_persistence_recovery_time.py --phase time_snapshot

    # Phase 3: Delete hnsw.base/graph.base, restart, measure rebuild
    python test_persistence_recovery_time.py --phase time_rebuild

    # Phase 4: Compare both timings
    python test_persistence_recovery_time.py --phase compare
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import requests

BASE_URL = "http://localhost:8080/api/v1"
COLLECTION = "perf_recovery"
DIM = 128
TOTAL_VECTORS = 50_000
BATCH_SIZE = 1_000

# Timing result files (stored alongside this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_TIMING_FILE = os.path.join(SCRIPT_DIR, "recovery_snapshot_timing.json")
REBUILD_TIMING_FILE = os.path.join(SCRIPT_DIR, "recovery_rebuild_timing.json")
EXPECTATIONS_FILE = os.path.join(SCRIPT_DIR, "recovery_expectations.json")


def wait_for_server(max_wait_sec=120, poll_interval_sec=0.5):
    """Poll health endpoint until server is responsive. Returns time taken."""
    start = time.perf_counter()
    deadline = start + max_wait_sec

    while time.perf_counter() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}", timeout=2.0)
            if resp.status_code == 200:
                elapsed = time.perf_counter() - start
                return elapsed
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(poll_interval_sec)

    return None


def phase_setup():
    """Create collection, insert 50K vectors, save expectations."""
    print("=" * 60)
    print("PHASE: SETUP")
    print(f"Creating '{COLLECTION}' with {TOTAL_VECTORS} vectors (dim={DIM})")
    print("=" * 60)
    print()

    # Clean up from prior run
    requests.delete(f"{BASE_URL}/collections/{COLLECTION}")

    # Create collection
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": COLLECTION,
        "dimension": DIM,
        "distance_metric": "cosine",
    })
    if resp.status_code != 200 or not resp.json().get("success"):
        print(f"[FAIL] Could not create collection: {resp.status_code} {resp.text[:200]}")
        return False

    print("  Collection created.")

    # Bulk insert
    rng = np.random.default_rng(seed=42)
    num_batches = TOTAL_VECTORS // BATCH_SIZE
    total_inserted = 0
    t0 = time.time()

    for batch_num in range(num_batches):
        vectors = []
        for _ in range(BATCH_SIZE):
            vec = rng.standard_normal(DIM).tolist()
            vectors.append({"values": vec, "metadata": {"batch": batch_num}})

        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/bulk",
            json={"vectors": vectors},
            timeout=60.0,
        )
        if resp.status_code == 200:
            total_inserted += resp.json().get("inserted_count", 0)
        else:
            print(f"  [WARN] Batch {batch_num} failed: {resp.status_code}")

        pct = (batch_num + 1) * 100 // num_batches
        print(f"  Inserting: batch {batch_num + 1}/{num_batches} ({pct}%)", end="\r")

    elapsed = time.time() - t0
    print()
    rate = total_inserted / elapsed if elapsed > 0 else 0
    print(f"  Inserted {total_inserted} vectors in {elapsed:.1f}s ({rate:.0f} vec/s)")

    if total_inserted != TOTAL_VECTORS:
        print(f"[FAIL] Expected {TOTAL_VECTORS} inserted, got {total_inserted}")
        return False

    # Verify count
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = resp.json().get("vector_count", 0)
    if count != TOTAL_VECTORS:
        print(f"[FAIL] Collection reports {count} vectors, expected {TOTAL_VECTORS}")
        return False

    # Save expectations
    expectations = {
        "collection": COLLECTION,
        "dimension": DIM,
        "vector_count": TOTAL_VECTORS,
    }
    with open(EXPECTATIONS_FILE, "w") as f:
        json.dump(expectations, f, indent=2)
    print(f"  Expectations saved to {EXPECTATIONS_FILE}")

    # Clean up old timing files
    for path in [SNAPSHOT_TIMING_FILE, REBUILD_TIMING_FILE]:
        if os.path.exists(path):
            os.remove(path)

    print()
    print("[PASS] Setup complete.")
    print()
    print(">>> Next step: Restart the SwarnDB server, then run:")
    print(">>>   python test_persistence_recovery_time.py --phase time_snapshot")
    return True


def phase_time_snapshot():
    """Measure time until server responds after a normal restart (snapshot recovery)."""
    print("=" * 60)
    print("PHASE: TIME SNAPSHOT RECOVERY")
    print("Waiting for server to come up with snapshot-based recovery...")
    print("=" * 60)
    print()

    # Load expectations
    if not os.path.exists(EXPECTATIONS_FILE):
        print("[FAIL] No expectations file found. Run --phase setup first.")
        return False

    with open(EXPECTATIONS_FILE) as f:
        expectations = json.load(f)

    expected_count = expectations["vector_count"]

    # Measure recovery time
    print("  Polling server...")
    recovery_sec = wait_for_server(max_wait_sec=120)

    if recovery_sec is None:
        print("[FAIL] Server did not respond within 120 seconds.")
        return False

    print(f"  Server responded in {recovery_sec:.3f} seconds")

    # Verify data integrity
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    if resp.status_code != 200:
        print(f"[FAIL] Collection not found after recovery: {resp.status_code}")
        return False

    count = resp.json().get("vector_count", 0)
    if count != expected_count:
        print(f"[FAIL] Expected {expected_count} vectors, got {count}")
        return False

    print(f"  Verified: {count} vectors present")

    # Run a quick search to verify functionality
    rng = np.random.default_rng(seed=123)
    query = rng.standard_normal(DIM).tolist()
    resp = requests.post(
        f"{BASE_URL}/collections/{COLLECTION}/search",
        json={"query": query, "k": 10},
        timeout=5.0,
    )
    if resp.status_code != 200 or len(resp.json().get("results", [])) == 0:
        print("[FAIL] Search does not work after snapshot recovery")
        return False

    print("  Search verified: working")

    # Save timing
    timing = {
        "phase": "snapshot",
        "recovery_seconds": recovery_sec,
        "vector_count": count,
    }
    with open(SNAPSHOT_TIMING_FILE, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"  Timing saved to {SNAPSHOT_TIMING_FILE}")

    print()
    print(f"[PASS] Snapshot recovery: {recovery_sec:.3f} seconds")
    print()
    print(">>> Next step:")
    print(">>>   1. Stop the SwarnDB server")
    print(">>>   2. Delete hnsw.base and graph.base files from the data directory")
    print(">>>   3. Restart the server")
    print(">>>   4. Run: python test_persistence_recovery_time.py --phase time_rebuild")
    return True


def phase_time_rebuild():
    """Measure time until server responds after index files are deleted (full rebuild)."""
    print("=" * 60)
    print("PHASE: TIME FULL REBUILD RECOVERY")
    print("Waiting for server to come up with full index rebuild...")
    print("=" * 60)
    print()

    # Load expectations
    if not os.path.exists(EXPECTATIONS_FILE):
        print("[FAIL] No expectations file found. Run --phase setup first.")
        return False

    with open(EXPECTATIONS_FILE) as f:
        expectations = json.load(f)

    expected_count = expectations["vector_count"]

    # Measure recovery time
    print("  Polling server (this may take longer due to index rebuild)...")
    recovery_sec = wait_for_server(max_wait_sec=300)

    if recovery_sec is None:
        print("[FAIL] Server did not respond within 300 seconds.")
        return False

    print(f"  Server responded in {recovery_sec:.3f} seconds")

    # Verify data integrity
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    if resp.status_code != 200:
        print(f"[FAIL] Collection not found after rebuild: {resp.status_code}")
        return False

    count = resp.json().get("vector_count", 0)
    if count != expected_count:
        print(f"[FAIL] Expected {expected_count} vectors, got {count}")
        return False

    print(f"  Verified: {count} vectors present")

    # Run a quick search to verify functionality
    rng = np.random.default_rng(seed=123)
    query = rng.standard_normal(DIM).tolist()
    resp = requests.post(
        f"{BASE_URL}/collections/{COLLECTION}/search",
        json={"query": query, "k": 10},
        timeout=10.0,
    )
    if resp.status_code != 200 or len(resp.json().get("results", [])) == 0:
        print("[FAIL] Search does not work after full rebuild")
        return False

    print("  Search verified: working")

    # Save timing
    timing = {
        "phase": "rebuild",
        "recovery_seconds": recovery_sec,
        "vector_count": count,
    }
    with open(REBUILD_TIMING_FILE, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"  Timing saved to {REBUILD_TIMING_FILE}")

    print()
    print(f"[PASS] Full rebuild recovery: {recovery_sec:.3f} seconds")
    print()
    print(">>> Next step: Run comparison:")
    print(">>>   python test_persistence_recovery_time.py --phase compare")
    return True


def phase_compare():
    """Load both timing files and compare snapshot vs rebuild recovery."""
    print("=" * 60)
    print("PHASE: COMPARE RECOVERY TIMES")
    print("=" * 60)
    print()

    # Load snapshot timing
    if not os.path.exists(SNAPSHOT_TIMING_FILE):
        print("[FAIL] Snapshot timing file not found. Run --phase time_snapshot first.")
        return False

    with open(SNAPSHOT_TIMING_FILE) as f:
        snapshot_data = json.load(f)

    # Load rebuild timing
    if not os.path.exists(REBUILD_TIMING_FILE):
        print("[FAIL] Rebuild timing file not found. Run --phase time_rebuild first.")
        return False

    with open(REBUILD_TIMING_FILE) as f:
        rebuild_data = json.load(f)

    snapshot_sec = snapshot_data["recovery_seconds"]
    rebuild_sec = rebuild_data["recovery_seconds"]
    vector_count = snapshot_data["vector_count"]

    # Calculate speedup
    if snapshot_sec > 0:
        speedup = rebuild_sec / snapshot_sec
    else:
        speedup = float("inf")

    print("-" * 60)
    print("RECOVERY TIME COMPARISON")
    print("-" * 60)
    print(f"  Collection:           {COLLECTION}")
    print(f"  Vector count:         {vector_count}")
    print(f"  Dimension:            {DIM}")
    print(f"  Snapshot recovery:    {snapshot_sec:.3f} seconds")
    print(f"  Full rebuild:         {rebuild_sec:.3f} seconds")
    print(f"  Speedup ratio:        {speedup:.2f}x")
    print("-" * 60)

    # Assertion: snapshot recovery at least 3x faster
    all_pass = True

    if speedup >= 3.0:
        print(f"[PASS] Snapshot recovery is {speedup:.2f}x faster (>= 3x required)")
    else:
        print(f"[FAIL] Snapshot recovery is only {speedup:.2f}x faster (< 3x required)")
        all_pass = False

    # Additional context
    time_saved = rebuild_sec - snapshot_sec
    pct_saved = (time_saved / rebuild_sec) * 100 if rebuild_sec > 0 else 0
    print(f"\n  Time saved:           {time_saved:.3f} seconds ({pct_saved:.1f}%)")

    return all_pass


def cleanup():
    """Delete test collection and timing files."""
    print("\nCleaning up...")
    resp = requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    if resp.status_code == 200:
        print(f"  Collection '{COLLECTION}' deleted.")
    else:
        print(f"  Cleanup note: collection delete returned {resp.status_code}")

    for path in [SNAPSHOT_TIMING_FILE, REBUILD_TIMING_FILE, EXPECTATIONS_FILE]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed {os.path.basename(path)}")


def main():
    parser = argparse.ArgumentParser(
        description="SwarnDB recovery time measurement test"
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["setup", "time_snapshot", "time_rebuild", "compare", "cleanup"],
        help="Which phase to run",
    )
    args = parser.parse_args()

    passed = False

    try:
        if args.phase == "setup":
            passed = phase_setup()
        elif args.phase == "time_snapshot":
            passed = phase_time_snapshot()
        elif args.phase == "time_rebuild":
            passed = phase_time_rebuild()
        elif args.phase == "compare":
            passed = phase_compare()
        elif args.phase == "cleanup":
            cleanup()
            passed = True
    except Exception as e:
        print(f"\n[FAIL] Unhandled exception: {e}")
        passed = False

    print()
    print("=" * 60)
    if passed:
        print(f"PHASE '{args.phase}': PASS")
    else:
        print(f"PHASE '{args.phase}': FAIL")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
