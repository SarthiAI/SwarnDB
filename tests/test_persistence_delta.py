#!/usr/bin/env python3
"""
Delta log correctness test for SwarnDB persistence layer.

Tests that vectors inserted after a base snapshot are correctly recovered
via delta replay after a server restart.

Real-world scenario: document embeddings pipeline where new documents arrive
continuously and must survive restarts without loss.

Usage:
    # Phase 1: insert data and save expectations
    python test_persistence_delta.py --phase before

    # (restart SwarnDB server here)

    # Phase 2: verify all data survived the restart
    python test_persistence_delta.py --phase after
"""

import argparse
import json
import sys
import time

import numpy as np
import requests

BASE_URL = "http://localhost:8080/api/v1"
COLLECTION = "documents"
DIM = 128
INITIAL_BATCH = 1000
DELTA_BATCH = 500
TOTAL_VECTORS = INITIAL_BATCH + DELTA_BATCH
BULK_SIZE = 250
EXPECTATIONS_FILE = "/tmp/swarndb_delta_expectations.json"
SEED = 12345
SAMPLE_COUNT = 20

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

def bulk_insert(vectors_payload: list[dict]) -> tuple[int, int]:
    """Insert vectors in chunks. Returns (inserted_count, error_count)."""
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


def generate_vectors(rng: np.random.Generator, count: int, metadata_prefix: str) -> list[dict]:
    """Generate a batch of vectors with metadata tags."""
    vectors = []
    for i in range(count):
        vec = rng.standard_normal(DIM).astype(np.float32)
        vectors.append({
            "values": vec.tolist(),
            "metadata": {"source": metadata_prefix, "seq": i},
        })
    return vectors


# ---------------------------------------------------------------------------
# Phase: before  (insert data, save expectations)
# ---------------------------------------------------------------------------

def run_before():
    global passed, failed
    print("=" * 60)
    print("SwarnDB Delta Persistence Test  --  Phase: BEFORE")
    print(f"Target: {BASE_URL}")
    print(f"Collection: {COLLECTION}  dim={DIM}")
    print(f"Initial batch: {INITIAL_BATCH}  Delta batch: {DELTA_BATCH}")
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

    # 2. Insert initial batch (these will land in the base snapshot)
    print(f"\n  Inserting initial batch ({INITIAL_BATCH} vectors)...")
    initial_vectors = generate_vectors(rng, INITIAL_BATCH, "initial")
    ins, errs = bulk_insert(initial_vectors)
    report(
        f"Insert initial batch ({INITIAL_BATCH})",
        ins == INITIAL_BATCH and errs == 0,
        f"inserted={ins} errors={errs}",
    )

    # 3. Wait for background snapshot to potentially occur
    wait_secs = 5
    print(f"\n  Waiting {wait_secs}s for background snapshot opportunity...")
    time.sleep(wait_secs)

    # 4. Insert delta batch (these go into the delta log only)
    print(f"\n  Inserting delta batch ({DELTA_BATCH} vectors)...")
    delta_vectors = generate_vectors(rng, DELTA_BATCH, "delta")
    ins2, errs2 = bulk_insert(delta_vectors)
    report(
        f"Insert delta batch ({DELTA_BATCH})",
        ins2 == DELTA_BATCH and errs2 == 0,
        f"inserted={ins2} errors={errs2}",
    )

    # 5. Collect all vector IDs
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    total_count = resp.json().get("vector_count", 0)
    report(
        "Total vector count matches",
        total_count == TOTAL_VECTORS,
        f"expected={TOTAL_VECTORS} got={total_count}",
    )

    # 6. Save sample vectors for exact-match verification after restart.
    #    Sample from both initial and delta ranges.
    sample_ids = []
    sample_data = {}

    # Pick 10 from the initial range and 10 from the delta range
    initial_sample_ids = sorted(rng.choice(range(1, INITIAL_BATCH + 1), size=min(10, INITIAL_BATCH), replace=False).tolist())
    delta_sample_ids = sorted(rng.choice(range(INITIAL_BATCH + 1, TOTAL_VECTORS + 1), size=min(10, DELTA_BATCH), replace=False).tolist())
    sample_ids = initial_sample_ids + delta_sample_ids

    all_ok = True
    for vid in sample_ids:
        resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{vid}")
        if resp.status_code == 200:
            data = resp.json()
            sample_data[str(vid)] = {
                "id": data.get("id"),
                "values": data.get("values"),
                "metadata": data.get("metadata"),
            }
        else:
            all_ok = False
            print(f"    WARNING: could not fetch vector {vid} (status={resp.status_code})")

    report(
        f"Saved {len(sample_data)} sample vectors ({SAMPLE_COUNT} target)",
        len(sample_data) == SAMPLE_COUNT and all_ok,
        f"saved={len(sample_data)}",
    )

    # 7. Run a search specifically looking for delta-batch vectors
    delta_rng = np.random.default_rng(seed=SEED)
    # Advance past initial batch to recreate a delta vector for searching
    for _ in range(INITIAL_BATCH):
        delta_rng.standard_normal(DIM)
    # Generate the first delta vector as a query
    delta_query = delta_rng.standard_normal(DIM).astype(np.float32).tolist()

    resp = requests.post(
        f"{BASE_URL}/collections/{COLLECTION}/search",
        json={"query": delta_query, "k": 5},
    )
    search_results_before = []
    if resp.status_code == 200:
        search_results_before = resp.json().get("results", [])
    report(
        "Search for delta vectors works",
        resp.status_code == 200 and len(search_results_before) == 5,
        f"got {len(search_results_before)} results",
    )

    # 8. Write expectations to disk
    expectations = {
        "total_vectors": TOTAL_VECTORS,
        "initial_batch": INITIAL_BATCH,
        "delta_batch": DELTA_BATCH,
        "sample_ids": sample_ids,
        "sample_data": sample_data,
        "delta_search_query": delta_query,
        "delta_search_result_ids": [r.get("id") for r in search_results_before],
    }
    with open(EXPECTATIONS_FILE, "w") as f:
        json.dump(expectations, f, indent=2)
    report(
        f"Expectations saved to {EXPECTATIONS_FILE}",
        True,
    )

    print()
    print("=" * 60)
    total = passed + failed
    print(f"BEFORE phase: {passed}/{total} checks passed")
    if failed > 0:
        print("WARNING: Some checks failed. 'after' phase may not pass.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Stop the SwarnDB server")
    print("  2. Restart the SwarnDB server")
    print("  3. Run:  python test_persistence_delta.py --phase after")

    return failed == 0


# ---------------------------------------------------------------------------
# Phase: after  (verify data survived restart)
# ---------------------------------------------------------------------------

def run_after():
    global passed, failed
    print("=" * 60)
    print("SwarnDB Delta Persistence Test  --  Phase: AFTER")
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
    delta_search_query = expectations["delta_search_query"]
    delta_search_result_ids = expectations["delta_search_result_ids"]
    initial_batch = expectations["initial_batch"]
    delta_batch = expectations["delta_batch"]

    # 1. Verify collection exists
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    report(
        "Collection exists after restart",
        resp.status_code == 200,
        f"status={resp.status_code}",
    )

    # 2. Verify total vector count
    if resp.status_code == 200:
        count = resp.json().get("vector_count", 0)
        report(
            "Total vector count preserved",
            count == expected_total,
            f"expected={expected_total} got={count}",
        )
    else:
        report("Total vector count preserved", False, "collection not found")

    # 3. Verify all sample vectors have exact matching data
    exact_match_count = 0
    mismatch_details = []
    for vid_str, expected in sample_data.items():
        vid = int(vid_str)
        resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{vid}")
        if resp.status_code != 200:
            mismatch_details.append(f"id={vid}: not found (status={resp.status_code})")
            continue

        actual = resp.json()
        actual_values = actual.get("values", [])
        expected_values = expected["values"]

        # Compare values with floating point tolerance
        if len(actual_values) != len(expected_values):
            mismatch_details.append(f"id={vid}: dim mismatch {len(actual_values)} vs {len(expected_values)}")
            continue

        values_match = all(
            abs(a - e) < 1e-6
            for a, e in zip(actual_values, expected_values)
        )

        metadata_match = actual.get("metadata") == expected.get("metadata")

        if values_match and metadata_match:
            exact_match_count += 1
        else:
            if not values_match:
                mismatch_details.append(f"id={vid}: values differ")
            if not metadata_match:
                mismatch_details.append(f"id={vid}: metadata differs (got={actual.get('metadata')} expected={expected.get('metadata')})")

    report(
        f"Sample vectors exact match ({exact_match_count}/{len(sample_data)})",
        exact_match_count == len(sample_data),
        "; ".join(mismatch_details[:5]) if mismatch_details else "all match",
    )

    # 4. Verify initial-batch sample vectors specifically
    initial_ok = 0
    for vid_str in [str(v) for v in sample_ids if v <= initial_batch]:
        resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{int(vid_str)}")
        if resp.status_code == 200 and resp.json().get("values"):
            initial_ok += 1
    initial_count = len([v for v in sample_ids if v <= initial_batch])
    report(
        f"Initial-batch samples accessible ({initial_ok}/{initial_count})",
        initial_ok == initial_count,
    )

    # 5. Verify delta-batch sample vectors specifically (these are the critical ones)
    delta_ok = 0
    for vid_str in [str(v) for v in sample_ids if v > initial_batch]:
        resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{int(vid_str)}")
        if resp.status_code == 200 and resp.json().get("values"):
            delta_ok += 1
    delta_count = len([v for v in sample_ids if v > initial_batch])
    report(
        f"Delta-batch samples accessible ({delta_ok}/{delta_count})",
        delta_ok == delta_count,
    )

    # 6. Search that targets delta-batch vectors
    resp = requests.post(
        f"{BASE_URL}/collections/{COLLECTION}/search",
        json={"query": delta_search_query, "k": 5},
    )
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        result_ids = [r.get("id") for r in results]
        # The same query should return the same (or very similar) top results
        overlap = len(set(result_ids) & set(delta_search_result_ids))
        report(
            "Delta search results consistent",
            overlap >= 3,
            f"overlap={overlap}/5 (before={delta_search_result_ids[:5]} after={result_ids[:5]})",
        )
    else:
        report("Delta search results consistent", False, f"search failed status={resp.status_code}")

    # 7. Run a broader search to confirm HNSW index is functional
    rng = np.random.default_rng(seed=99999)
    search_ok = True
    for i in range(5):
        query = rng.standard_normal(DIM).astype(np.float32).tolist()
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
        "General search functional (5 queries, k=10)",
        search_ok,
    )

    # 8. Verify a delta-batch vector by searching for it as its own query
    #    (nearest neighbor of itself should be itself)
    delta_sample_vid = sample_ids[-1]  # Last sample should be from delta batch
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/{delta_sample_vid}")
    if resp.status_code == 200:
        self_query = resp.json().get("values", [])
        resp2 = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/search",
            json={"query": self_query, "k": 1},
        )
        if resp2.status_code == 200:
            top_result = resp2.json().get("results", [{}])[0]
            top_id = top_result.get("id")
            report(
                f"Self-search for delta vector {delta_sample_vid}",
                top_id == delta_sample_vid,
                f"expected={delta_sample_vid} got={top_id}",
            )
        else:
            report(f"Self-search for delta vector {delta_sample_vid}", False, "search failed")
    else:
        report(f"Self-search for delta vector {delta_sample_vid}", False, "vector not found")

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
        description="SwarnDB delta persistence test",
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["before", "after"],
        help="'before' to insert data, 'after' to verify after restart",
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
