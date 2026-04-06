#!/usr/bin/env python3
"""
SwarnDB Persistence Test: Clean vs Dirty Shutdown Detection
============================================================
Verifies that the server writes a shutdown_clean marker on clean shutdown
(SIGTERM) and that no marker exists after a dirty shutdown (SIGKILL).

Usage (three-phase approach -- restart the server between phases):

    python test_persistence_shutdown_marker.py --phase setup
    # (send SIGTERM to the server for a clean shutdown)
    python test_persistence_shutdown_marker.py --phase check_clean
    # (now send SIGKILL to the running server for a dirty shutdown)
    python test_persistence_shutdown_marker.py --phase check_dirty

REST API: http://localhost:8080
Data directory: ./data  (override with SWARNDB_DATA_DIR)
"""

import argparse
import json
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
COLLECTION = "shutdown_test"
DIM = 64
NUM_VECTORS = 1000
BATCH_SIZE = 200
SEED = 77777

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


def marker_path():
    return os.path.join(collection_dir(), "shutdown_clean")


def server_healthy():
    """Check if the server is up and responding."""
    try:
        health_url = BASE_URL.rsplit("/api", 1)[0] + "/health"
        resp = requests.get(health_url, timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


# ---------------------------------------------------------------------------
# Phase: setup
# ---------------------------------------------------------------------------

def phase_setup():
    """Create collection, insert 1000 vectors. Server must be running."""
    print("\n=== PHASE: SETUP ===\n")

    if not server_healthy():
        report("Server is running", False, "Cannot reach server")
        return False
    report("Server is running", True)

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
        return False

    # 2. Insert vectors
    rng = np.random.default_rng(seed=SEED)
    total_inserted = 0

    for batch_num in range(NUM_VECTORS // BATCH_SIZE):
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
            report(f"Bulk insert batch {batch_num}", False, resp.text[:200])
            return False

    report(f"Insert {NUM_VECTORS} vectors", total_inserted == NUM_VECTORS,
           f"inserted={total_inserted}")

    # 3. Verify count
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = resp.json().get("vector_count", -1)
    report("Verify vector count", count == NUM_VECTORS, f"count={count}")

    # 4. Check that the marker does NOT exist yet (server is still running)
    marker_exists = os.path.exists(marker_path())
    report("No shutdown marker while running", not marker_exists,
           f"marker_path={marker_path()}")

    print("\n  Setup complete.")
    print("  Now send SIGTERM to the server for a clean shutdown:")
    print("    kill -TERM <server_pid>")
    print("    # or: docker stop swarndb")
    print("  Then run: python test_persistence_shutdown_marker.py --phase check_clean")
    return True


# ---------------------------------------------------------------------------
# Phase: check_clean (after SIGTERM)
# ---------------------------------------------------------------------------

def phase_check_clean():
    """After a clean shutdown (SIGTERM), check for the shutdown_clean marker."""
    print("\n=== PHASE: CHECK_CLEAN (after SIGTERM) ===\n")

    # Server should be DOWN at this point
    if server_healthy():
        print("  WARNING: Server is still running. This phase should be run")
        print("  AFTER the server has been stopped with SIGTERM.")
        print("  Proceeding anyway -- marker check may reflect current state.\n")

    cdir = collection_dir()
    if not os.path.isdir(cdir):
        report("Collection directory exists", False, f"dir={cdir}")
        print("  The collection directory does not exist. Did setup run correctly?")
        return False

    report("Collection directory exists", True, f"dir={cdir}")

    # 1. Check for shutdown_clean marker
    marker = marker_path()
    marker_exists = os.path.exists(marker)
    report("shutdown_clean marker exists (clean shutdown)", marker_exists,
           f"path={marker}")

    if not marker_exists:
        print("  EXPECTED: The marker should exist after a clean SIGTERM shutdown.")
        print("  Possible causes:")
        print("    - Server did not receive SIGTERM (used SIGKILL instead?)")
        print("    - Graceful shutdown code did not run")
        print("    - Data directory mismatch")

    # 2. Check that hnsw.base and graph.base were written
    hnsw_exists = os.path.exists(os.path.join(cdir, "hnsw.base"))
    graph_exists = os.path.exists(os.path.join(cdir, "graph.base"))
    report("hnsw.base snapshot exists", hnsw_exists)
    report("graph.base snapshot exists", graph_exists)

    # 3. Check wal_meta.json
    wal_meta_path = os.path.join(cdir, "wal_meta.json")
    if os.path.exists(wal_meta_path):
        with open(wal_meta_path) as f:
            meta = json.load(f)
        report("wal_meta.json exists", True,
               f"next_lsn={meta.get('next_lsn')} snapshot_lsn={meta.get('last_snapshot_lsn')}")
    else:
        report("wal_meta.json exists", False)

    print("\n  Clean shutdown check complete.")
    print("  Now start the server again, then SIGKILL it:")
    print("    kill -9 <server_pid>")
    print("    # or: docker kill swarndb")
    print("  Then run: python test_persistence_shutdown_marker.py --phase check_dirty")
    return True


# ---------------------------------------------------------------------------
# Phase: check_dirty (after SIGKILL)
# ---------------------------------------------------------------------------

def phase_check_dirty():
    """After a dirty shutdown (SIGKILL), verify no shutdown_clean marker exists."""
    print("\n=== PHASE: CHECK_DIRTY (after SIGKILL) ===\n")

    # Server should be DOWN at this point
    if server_healthy():
        print("  WARNING: Server is still running. This phase should be run")
        print("  AFTER the server has been killed with SIGKILL.")
        print("  Proceeding anyway.\n")

    cdir = collection_dir()
    if not os.path.isdir(cdir):
        report("Collection directory exists", False, f"dir={cdir}")
        return False

    report("Collection directory exists", True, f"dir={cdir}")

    # 1. On startup, the server removes the old shutdown_clean marker.
    #    After a SIGKILL, the server never wrote a new one.
    #    So the marker should NOT exist.
    marker = marker_path()
    marker_exists = os.path.exists(marker)

    # After the server restarted (post-clean-shutdown), it removes the marker.
    # Then after SIGKILL, no new marker is written.
    # So the marker should be absent.
    report("No shutdown_clean marker (dirty shutdown)", not marker_exists,
           f"path={marker}")

    if marker_exists:
        print("  UNEXPECTED: The marker should NOT exist after a SIGKILL.")
        print("  This means either:")
        print("    - The server was not killed with SIGKILL")
        print("    - The marker from the previous clean shutdown was not removed on startup")

    # 2. Verify data is still on disk (WAL, segments)
    wal_path = os.path.join(cdir, "wal.log")
    wal_exists = os.path.exists(wal_path)
    report("WAL file exists (for replay)", wal_exists)

    # 3. Now restart the server and verify data recovery
    print("\n  Dirty shutdown check complete.")
    print("  Now restart the server. It should perform WAL replay (no clean marker).")
    print("  After restart, verify with:")
    print(f"    curl {BASE_URL}/collections/{COLLECTION}")
    print(f"  Expected vector_count: {NUM_VECTORS}")
    return True


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup():
    """Delete the test collection."""
    print("\n  Cleaning up...")
    if server_healthy():
        resp = requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
        if resp.status_code == 200:
            print(f"  Collection '{COLLECTION}' deleted.")
        else:
            print(f"  Cleanup warning: status={resp.status_code}")
    else:
        print("  Server is not running. Cannot clean up via API.")
        print(f"  Manually remove: {collection_dir()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SwarnDB Persistence Test: Clean vs Dirty Shutdown Detection"
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["setup", "check_clean", "check_dirty", "cleanup"],
        help="Test phase to execute",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SwarnDB Persistence Test: Shutdown Marker Detection")
    print(f"Phase: {args.phase}")
    print(f"Target: {BASE_URL}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Marker path: {marker_path()}")
    print("=" * 60)

    if args.phase == "setup":
        phase_setup()
    elif args.phase == "check_clean":
        phase_check_clean()
    elif args.phase == "check_dirty":
        phase_check_dirty()
    elif args.phase == "cleanup":
        cleanup()

    print()
    print("=" * 60)
    print(f"RESULT: {passed} PASSED, {failed} FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
