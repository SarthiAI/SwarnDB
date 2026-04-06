#!/usr/bin/env python3
"""
SwarnDB Crash Recovery Test (SIGKILL simulation).

Simulates a recommendation engine crash scenario:
  --phase=before : Insert 3000 vectors, save expectations
  --phase=crash  : Find server PID, send SIGKILL (kill -9)
  --phase=after  : Verify data integrity after restart

Usage:
  python test_persistence_crash.py --phase=before
  python test_persistence_crash.py --phase=crash
  # ... manually restart server ...
  python test_persistence_crash.py --phase=after
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import requests

BASE_URL = "http://localhost:8080/api/v1"
COLLECTION = "recommendations"
DIM = 128
TOTAL_VECTORS = 3000
BATCH_SIZE = 100
EXPECTATIONS_FILE = "/tmp/swarndb_crash_expectations.json"
DATA_DIR = os.environ.get("SWARNDB_DATA_DIR", "./data")

# Fixed seed for reproducible vectors
RNG_SEED = 7777


# ── Helpers ─────────────────────────────────────────────────────────────────

def generate_vector(rng, dim):
    """Generate a single random vector."""
    return rng.standard_normal(dim).astype(np.float32).tolist()


def generate_metadata(idx):
    """Generate recommendation-engine-style metadata for a vector."""
    categories = ["electronics", "books", "clothing", "food", "sports",
                  "music", "movies", "games", "travel", "health"]
    return {
        "user_id": f"user_{idx:05d}",
        "category": categories[idx % len(categories)],
        "preference_score": round(float((idx % 100) / 100.0), 2),
        "active": idx % 3 != 0,
    }


def report(label, ok, detail=""):
    """Print a PASS/FAIL line."""
    tag = "PASS" if ok else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    return ok


# ── Phase: before ───────────────────────────────────────────────────────────

def phase_before():
    """Insert 3000 vectors and save expectations to disk."""
    print("=" * 60)
    print("PHASE: before (insert data)")
    print(f"Target: {BASE_URL}")
    print(f"Collection: {COLLECTION} ({DIM}-dim, {TOTAL_VECTORS} vectors)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # Step 1: Clean up and create collection
    requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": COLLECTION,
        "dimension": DIM,
        "distance_metric": "cosine",
    })
    data = resp.json()
    ok = resp.status_code == 200 and data.get("success") is True
    if report("Create collection", ok,
              f"status={resp.status_code} success={data.get('success')}"):
        passed += 1
    else:
        failed += 1
        print("  Cannot proceed without collection. Aborting.")
        sys.exit(1)

    # Step 2: Insert 3000 vectors in batches of 100
    rng = np.random.default_rng(seed=RNG_SEED)
    total_inserted = 0
    total_errors = 0
    all_ids = []
    t0 = time.time()

    num_batches = TOTAL_VECTORS // BATCH_SIZE
    for batch_num in range(num_batches):
        vectors = []
        for i in range(BATCH_SIZE):
            global_idx = batch_num * BATCH_SIZE + i
            vec = generate_vector(rng, DIM)
            meta = generate_metadata(global_idx)
            vectors.append({"values": vec, "metadata": meta})

        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/bulk",
            json={"vectors": vectors},
        )
        if resp.status_code == 200:
            rdata = resp.json()
            inserted = rdata.get("inserted_count", 0)
            total_inserted += inserted
            total_errors += len(rdata.get("errors", []))
            ids = rdata.get("ids", [])
            all_ids.extend(ids)
        else:
            total_errors += BATCH_SIZE
            print(f"    Batch {batch_num} failed: {resp.status_code} "
                  f"{resp.text[:200]}")

        pct = (batch_num + 1) * 100 // num_batches
        print(f"    Batch {batch_num + 1}/{num_batches} ({pct}%)", end="\r")

    elapsed = time.time() - t0
    print()
    rate = total_inserted / elapsed if elapsed > 0 else 0
    ok = total_inserted == TOTAL_VECTORS and total_errors == 0
    if report("Bulk insert 3000 vectors", ok,
              f"inserted={total_inserted} errors={total_errors} "
              f"time={elapsed:.1f}s rate={rate:.0f} vec/s"):
        passed += 1
    else:
        failed += 1

    # Step 3: Verify count
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = resp.json().get("vector_count", -1)
    ok = resp.status_code == 200 and count == TOTAL_VECTORS
    if report("Verify vector count", ok,
              f"expected={TOTAL_VECTORS} got={count}"):
        passed += 1
    else:
        failed += 1

    # Step 4: Record 10 specific vectors for later verification
    saved_vectors = {}
    sample_ids = []
    if len(all_ids) >= 10:
        # Pick 10 evenly spaced IDs
        step = len(all_ids) // 10
        sample_ids = [all_ids[i * step] for i in range(10)]
    elif len(all_ids) > 0:
        sample_ids = all_ids[:min(10, len(all_ids))]

    fetch_ok = True
    for vid in sample_ids:
        resp = requests.get(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/{vid}"
        )
        if resp.status_code == 200:
            vdata = resp.json()
            saved_vectors[str(vid)] = {
                "id": vdata.get("id"),
                "values": vdata.get("values"),
                "metadata": vdata.get("metadata"),
            }
        else:
            fetch_ok = False

    ok = fetch_ok and len(saved_vectors) == len(sample_ids)
    if report("Record 10 sample vectors", ok,
              f"saved={len(saved_vectors)}/{len(sample_ids)}"):
        passed += 1
    else:
        failed += 1

    # Step 5: Save expectations
    expectations = {
        "collection": COLLECTION,
        "dimension": DIM,
        "total_vectors": TOTAL_VECTORS,
        "total_inserted": total_inserted,
        "saved_vectors": saved_vectors,
        "sample_ids": sample_ids,
        "rng_seed": RNG_SEED,
        "timestamp": time.time(),
    }
    with open(EXPECTATIONS_FILE, "w") as f:
        json.dump(expectations, f, indent=2)
    ok = os.path.exists(EXPECTATIONS_FILE)
    if report("Save expectations", ok, f"file={EXPECTATIONS_FILE}"):
        passed += 1
    else:
        failed += 1

    # Summary
    print()
    print("-" * 60)
    print(f"BEFORE phase: {passed}/{passed + failed} passed")
    print(f"Expectations saved to: {EXPECTATIONS_FILE}")
    print(f"Sample vector IDs: {sample_ids}")
    print()
    print("Next step: python test_persistence_crash.py --phase=crash")
    print("-" * 60)

    return failed == 0


# ── Phase: crash ────────────────────────────────────────────────────────────

def phase_crash():
    """Find server PID and send SIGKILL to simulate a crash."""
    print("=" * 60)
    print("PHASE: crash (SIGKILL simulation)")
    print("=" * 60)
    print()

    # Step 1: Find server PID
    print("  Looking for SwarnDB server process (vf-server)...")
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vf-server"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split("\n")
        pids = [p.strip() for p in pids if p.strip()]
    except Exception as e:
        print(f"  [FAIL] Could not run pgrep: {e}")
        sys.exit(1)

    if not pids:
        print("  [FAIL] No vf-server process found.")
        print("  Make sure the server is running before executing --phase=crash")
        sys.exit(1)

    pid = int(pids[0])
    print(f"  Found server PID: {pid}")

    # Confirm the expectations file exists
    if not os.path.exists(EXPECTATIONS_FILE):
        print(f"  [WARN] Expectations file not found at {EXPECTATIONS_FILE}")
        print("  Run --phase=before first!")
        sys.exit(1)

    # Step 2: Send SIGKILL
    print(f"  Sending SIGKILL (kill -9) to PID {pid}...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        print(f"  [WARN] Process {pid} already dead.")
    except PermissionError:
        print(f"  [FAIL] Permission denied sending SIGKILL to PID {pid}.")
        print("  Try running with sudo or check process ownership.")
        sys.exit(1)

    # Step 3: Verify process is dead
    time.sleep(1)
    dead = False
    for attempt in range(5):
        try:
            os.kill(pid, 0)  # signal 0 = check existence
            time.sleep(0.5)
        except ProcessLookupError:
            dead = True
            break
        except PermissionError:
            # Process exists but we can't signal it
            time.sleep(0.5)

    if dead:
        report("Server process killed", True, f"PID {pid} is dead")
    else:
        report("Server process killed", False,
               f"PID {pid} may still be alive")

    # Step 4: Print instructions
    print()
    print("-" * 60)
    print("CRASH phase complete. The server was killed without graceful")
    print("shutdown (no flush, no shutdown_clean marker).")
    print()
    print("Next steps:")
    print("  1. Restart the SwarnDB server")
    print("  2. Wait for it to become healthy")
    print("  3. Run: python test_persistence_crash.py --phase=after")
    print("-" * 60)

    return dead


# ── Phase: after ────────────────────────────────────────────────────────────

def phase_after():
    """Verify data integrity after crash recovery."""
    print("=" * 60)
    print("PHASE: after (verify crash recovery)")
    print("=" * 60)
    print()

    # Load expectations
    if not os.path.exists(EXPECTATIONS_FILE):
        print(f"  [FAIL] Expectations file not found: {EXPECTATIONS_FILE}")
        print("  Run --phase=before and --phase=crash first.")
        sys.exit(1)

    with open(EXPECTATIONS_FILE, "r") as f:
        exp = json.load(f)

    collection = exp["collection"]
    dimension = exp["dimension"]
    total_vectors = exp["total_vectors"]
    total_inserted = exp["total_inserted"]
    saved_vectors = exp["saved_vectors"]
    sample_ids = exp["sample_ids"]

    passed = 0
    failed = 0

    # Check 1: Collection exists
    resp = requests.get(f"{BASE_URL}/collections/{collection}")
    ok = resp.status_code == 200
    if ok:
        cdata = resp.json()
        name_ok = cdata.get("name") == collection
        dim_ok = cdata.get("dimension") == dimension
        ok = name_ok and dim_ok
    if report("Collection exists", ok,
              f"name={collection} dim={dimension}"):
        passed += 1
    else:
        failed += 1
        print("  Collection missing after crash recovery. Cannot continue.")
        _print_summary(passed, failed)
        sys.exit(1)

    # Check 2: Vector count (allow some loss from unflushed batches)
    count = cdata.get("vector_count", 0)
    # After a crash, some vectors from the last unflushed WAL batch may be
    # lost. We accept >= 90% recovery as valid for a hard kill scenario.
    min_acceptable = int(total_inserted * 0.90)
    count_ok = count >= min_acceptable
    count_exact = count == total_inserted
    detail = (f"expected={total_inserted} got={count} "
              f"min_acceptable={min_acceptable}")
    if count_exact:
        detail += " (exact match)"
    elif count_ok:
        loss = total_inserted - count
        detail += f" (lost {loss} vectors from unflushed batch -- acceptable)"
    if report("Vector count after recovery", count_ok, detail):
        passed += 1
    else:
        failed += 1

    # Check 3: Verify the 10 saved vectors
    vectors_ok = 0
    vectors_checked = 0
    for vid_str, expected in saved_vectors.items():
        vid = expected["id"]
        resp = requests.get(
            f"{BASE_URL}/collections/{collection}/vectors/{vid}"
        )
        vectors_checked += 1

        if resp.status_code != 200:
            report(f"Vector {vid} retrieval", False,
                   f"status={resp.status_code}")
            continue

        actual = resp.json()

        # Compare values (float tolerance)
        exp_values = expected["values"]
        act_values = actual.get("values", [])
        if len(exp_values) == len(act_values):
            values_match = all(
                abs(a - b) < 1e-5
                for a, b in zip(exp_values, act_values)
            )
        else:
            values_match = False

        # Compare metadata
        exp_meta = expected["metadata"]
        act_meta = actual.get("metadata", {})
        meta_match = True
        for key, val in exp_meta.items():
            if act_meta.get(key) != val:
                meta_match = False
                break

        if values_match and meta_match:
            vectors_ok += 1
        else:
            report(f"Vector {vid} data mismatch", False,
                   f"values_match={values_match} meta_match={meta_match}")

    all_vectors_ok = vectors_ok == vectors_checked and vectors_checked > 0
    if report("Saved vectors integrity", all_vectors_ok,
              f"{vectors_ok}/{vectors_checked} vectors match"):
        passed += 1
    else:
        failed += 1

    # Check 4: Search queries return reasonable results
    rng = np.random.default_rng(seed=42)
    search_ok = True
    search_count = 5
    for i in range(search_count):
        query = generate_vector(rng, dimension)
        resp = requests.post(
            f"{BASE_URL}/collections/{collection}/search",
            json={"query": query, "k": 10},
        )
        if resp.status_code != 200:
            search_ok = False
            continue

        results = resp.json().get("results", [])
        if len(results) == 0:
            search_ok = False
            continue

        # Verify results have valid scores
        for r in results:
            score = r.get("score")
            if score is None or not isinstance(score, (int, float)):
                search_ok = False
                break

    if report("Search queries after recovery", search_ok,
              f"{search_count} queries, k=10"):
        passed += 1
    else:
        failed += 1

    # Check 5: No shutdown_clean marker (dirty recovery path)
    # The marker lives at <data_dir>/<collection>/shutdown_clean
    data_dir = os.environ.get("SWARNDB_DATA_DIR", "./data")
    marker_path = os.path.join(data_dir, collection, "shutdown_clean")
    marker_exists = os.path.exists(marker_path)
    # After a crash, the marker should NOT exist (it was never written)
    # AND after recovery the server removes it anyway.
    # So we verify it is absent.
    ok = not marker_exists
    if report("No shutdown_clean marker (dirty recovery)", ok,
              f"marker_path={marker_path} exists={marker_exists}"):
        passed += 1
    else:
        failed += 1

    # Summary
    _print_summary(passed, failed)
    return failed == 0


def _print_summary(passed, failed):
    """Print the final PASS/FAIL summary."""
    total = passed + failed
    print()
    print("=" * 60)
    if failed == 0:
        print(f"CRASH RECOVERY TEST: PASS ({passed}/{total} checks passed)")
    else:
        print(f"CRASH RECOVERY TEST: FAIL ({passed}/{total} checks passed, "
              f"{failed} failed)")
    print("=" * 60)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SwarnDB crash recovery test (SIGKILL simulation)"
    )
    parser.add_argument(
        "--phase", required=True,
        choices=["before", "crash", "after"],
        help="Test phase to execute",
    )
    args = parser.parse_args()

    if args.phase == "before":
        ok = phase_before()
    elif args.phase == "crash":
        ok = phase_crash()
    elif args.phase == "after":
        ok = phase_after()
    else:
        print(f"Unknown phase: {args.phase}")
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
