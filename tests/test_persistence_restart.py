#!/usr/bin/env python3
"""
Test persistence across server restart (clean shutdown).

Usage:
    Phase 1 - Insert data and save expectations:
        python test_persistence_restart.py --phase=before

    (Restart the SwarnDB server now)

    Phase 2 - Verify data survived restart:
        python test_persistence_restart.py --phase=after

E-commerce product search scenario: 5000 product vectors with metadata,
verifying data, HNSW index, and virtual graph survive a clean restart.
"""

import argparse
import json
import sys
import time

import numpy as np
import requests

# ── Configuration ───────────────────────────────────────────────────────

BASE_URL = "http://localhost:8080/api/v1"
COLLECTION = "persist_restart_test"
DIM = 128
NUM_VECTORS = 30000
BATCH_SIZE = 500
NUM_SAMPLE_VECTORS = 10
NUM_SEARCH_QUERIES = 5
SEARCH_K = 10
GRAPH_THRESHOLD = 0.3
SEED = 12345
EXPECTATIONS_FILE = "/tmp/swarndb_restart_expectations.json"

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Books", "Sports",
    "Toys", "Beauty", "Automotive", "Garden", "Health",
]

# ── Helpers ─────────────────────────────────────────────────────────────

passed_count = 0
failed_count = 0


def check(name, condition, detail=""):
    """Record a PASS/FAIL check."""
    global passed_count, failed_count
    tag = "PASS" if condition else "FAIL"
    if condition:
        passed_count += 1
    else:
        failed_count += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return condition


def generate_product_metadata(idx, rng):
    """Generate e-commerce product metadata for a vector."""
    category = CATEGORIES[idx % len(CATEGORIES)]
    price = round(float(rng.uniform(5.0, 500.0)), 2)
    return {
        "name": f"Product-{idx:05d}",
        "category": category,
        "price": price,
        "in_stock": bool(rng.integers(0, 2)),
        "rating": round(float(rng.uniform(1.0, 5.0)), 1),
    }


# ── Phase: BEFORE (insert data, save expectations) ─────────────────────

def phase_before():
    """Insert 5000 vectors and save expectations to a JSON file."""
    print("=" * 64)
    print("PHASE: BEFORE RESTART")
    print(f"  Server:     {BASE_URL}")
    print(f"  Collection: {COLLECTION}")
    print(f"  Vectors:    {NUM_VECTORS} x {DIM}-dim")
    print("=" * 64)
    print()

    rng = np.random.default_rng(seed=SEED)

    # ── 1. Clean up any leftover collection ─────────────────────────────
    print("[Step 1] Cleanup + create collection")
    requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": COLLECTION,
        "dimension": DIM,
        "distance_metric": "cosine",
    })
    ok = resp.status_code == 200 and resp.json().get("success") is True
    check("Create collection", ok, f"status={resp.status_code}")
    if not ok:
        print("  FATAL: Cannot create collection. Aborting.")
        sys.exit(1)
    print()

    # ── 2. Set graph threshold so edges get created during insert ───────
    print("[Step 2] Set graph threshold")
    resp = requests.post(
        f"{BASE_URL}/collections/{COLLECTION}/graph/threshold",
        json={"threshold": GRAPH_THRESHOLD},
    )
    threshold_ok = resp.status_code == 200
    check("Set graph threshold", threshold_ok,
          f"threshold={GRAPH_THRESHOLD} status={resp.status_code}")
    print()

    # ── 3. Bulk insert 5000 vectors ─────────────────────────────────────
    print(f"[Step 3] Bulk insert {NUM_VECTORS} vectors")

    # Pre-generate ALL vectors and metadata so we can sample from them
    all_vectors = []
    all_metadata = []
    for i in range(NUM_VECTORS):
        vec = rng.standard_normal(DIM).astype(np.float32).tolist()
        meta = generate_product_metadata(i, rng)
        all_vectors.append(vec)
        all_metadata.append(meta)

    total_inserted = 0
    insert_errors = 0
    t0 = time.time()
    num_batches = NUM_VECTORS // BATCH_SIZE

    for batch_num in range(num_batches):
        start = batch_num * BATCH_SIZE
        end = start + BATCH_SIZE
        vectors_payload = []
        for i in range(start, end):
            vectors_payload.append({
                "values": all_vectors[i],
                "metadata": all_metadata[i],
            })

        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/bulk",
            json={"vectors": vectors_payload},
        )
        if resp.status_code == 200:
            data = resp.json()
            total_inserted += data.get("inserted_count", 0)
            insert_errors += len(data.get("errors", []))
        else:
            insert_errors += BATCH_SIZE
            print(f"    Batch {batch_num} FAILED: {resp.status_code} "
                  f"{resp.text[:200]}")

        pct = (batch_num + 1) * 100 // num_batches
        print(f"    Batch {batch_num + 1}/{num_batches} ({pct}%)", end="\r")

    elapsed = time.time() - t0
    print()
    rate = total_inserted / elapsed if elapsed > 0 else 0
    check("Bulk insert complete", total_inserted == NUM_VECTORS,
          f"inserted={total_inserted}/{NUM_VECTORS} errors={insert_errors} "
          f"time={elapsed:.1f}s rate={rate:.0f} vec/s")
    print()

    # ── 4. Verify count ─────────────────────────────────────────────────
    print("[Step 4] Verify collection count")
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = resp.json().get("vector_count", -1)
    check("Vector count matches", count == NUM_VECTORS,
          f"expected={NUM_VECTORS} got={count}")
    print()

    # ── 5. Pick 10 sample vector IDs and fetch their exact data ─────────
    print(f"[Step 5] Save {NUM_SAMPLE_VECTORS} sample vectors")
    # Use deterministic sampling: evenly spaced across the ID range
    # IDs are 1-based (server auto-assigns starting from 1)
    sample_ids = [1 + i * (NUM_VECTORS // NUM_SAMPLE_VECTORS)
                  for i in range(NUM_SAMPLE_VECTORS)]

    sample_vectors = {}
    for vid in sample_ids:
        resp = requests.get(
            f"{BASE_URL}/collections/{COLLECTION}/vectors/{vid}")
        if resp.status_code == 200:
            data = resp.json()
            sample_vectors[str(vid)] = {
                "id": data.get("id"),
                "values": data.get("values"),
                "metadata": data.get("metadata"),
            }
        else:
            print(f"    WARNING: Could not fetch vector {vid}: "
                  f"status={resp.status_code}")

    check("Sample vectors fetched",
          len(sample_vectors) == NUM_SAMPLE_VECTORS,
          f"fetched={len(sample_vectors)}/{NUM_SAMPLE_VECTORS}")
    print()

    # ── 6. Run 5 search queries and save results ───────────────────────
    print(f"[Step 6] Run {NUM_SEARCH_QUERIES} search queries")
    search_rng = np.random.default_rng(seed=SEED + 1000)
    search_queries = []
    search_results = []

    for i in range(NUM_SEARCH_QUERIES):
        query_vec = search_rng.standard_normal(DIM).astype(np.float32).tolist()
        search_queries.append(query_vec)

        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/search",
            json={"query": query_vec, "k": SEARCH_K},
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            search_results.append([
                {"id": r["id"], "score": r["score"]} for r in results
            ])
        else:
            search_results.append([])
            print(f"    WARNING: Search {i} failed: status={resp.status_code}")

    all_search_ok = all(len(r) == SEARCH_K for r in search_results)
    check("Search queries complete", all_search_ok,
          f"queries={NUM_SEARCH_QUERIES} "
          f"all_returned_k={all_search_ok}")
    print()

    # ── 7. Check virtual graph edges ────────────────────────────────────
    print("[Step 7] Check virtual graph edges")
    graph_samples = {}
    graph_edge_count = 0
    for vid in sample_ids[:5]:
        resp = requests.get(
            f"{BASE_URL}/collections/{COLLECTION}/graph/related/{vid}",
            params={"threshold": GRAPH_THRESHOLD},
        )
        if resp.status_code == 200:
            edges = resp.json().get("edges", [])
            graph_samples[str(vid)] = [
                {"target_id": e["target_id"], "similarity": e["similarity"]}
                for e in edges
            ]
            graph_edge_count += len(edges)
        else:
            graph_samples[str(vid)] = []

    check("Graph edges exist", graph_edge_count > 0,
          f"total_edges_across_5_samples={graph_edge_count}")
    print()

    # ── 8. Save all expectations to JSON ────────────────────────────────
    print(f"[Step 8] Save expectations to {EXPECTATIONS_FILE}")
    expectations = {
        "collection": COLLECTION,
        "num_vectors": NUM_VECTORS,
        "dimension": DIM,
        "sample_vectors": sample_vectors,
        "search_queries": search_queries,
        "search_results": search_results,
        "graph_samples": graph_samples,
        "graph_threshold": GRAPH_THRESHOLD,
        "sample_ids": sample_ids,
        "timestamp": time.time(),
    }

    with open(EXPECTATIONS_FILE, "w") as f:
        json.dump(expectations, f, indent=2)

    check("Expectations saved", True,
          f"file={EXPECTATIONS_FILE}")
    print()

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"BEFORE phase complete: {passed_count} PASS, {failed_count} FAIL")
    print()
    print("Next steps:")
    print("  1. Restart the SwarnDB server (docker compose restart, SIGTERM, etc.)")
    print("  2. Run: python test_persistence_restart.py --phase=after")
    print("=" * 64)

    return failed_count == 0


# ── Phase: AFTER (verify data survived restart) ────────────────────────

def phase_after():
    """Load expectations and verify everything survived the restart."""
    print("=" * 64)
    print("PHASE: AFTER RESTART")
    print(f"  Server:     {BASE_URL}")
    print(f"  Expectations: {EXPECTATIONS_FILE}")
    print("=" * 64)
    print()

    # ── 1. Load expectations ────────────────────────────────────────────
    print("[Step 1] Load expectations")
    try:
        with open(EXPECTATIONS_FILE, "r") as f:
            exp = json.load(f)
    except FileNotFoundError:
        print(f"  FATAL: {EXPECTATIONS_FILE} not found. Run --phase=before first.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"  FATAL: Invalid JSON in expectations file: {e}")
        sys.exit(1)

    check("Expectations loaded", True,
          f"collection={exp['collection']} vectors={exp['num_vectors']}")
    print()

    # ── 2. Wait for server to be ready ──────────────────────────────────
    print("[Step 2] Wait for server readiness")
    server_ready = False
    for attempt in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/collections/{exp['collection']}",
                                timeout=2)
            if resp.status_code == 200:
                server_ready = True
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
        print(f"    Waiting for server... attempt {attempt + 1}/30", end="\r")

    print()
    check("Server is ready", server_ready)
    if not server_ready:
        print("  FATAL: Server did not become ready within 30 seconds.")
        sys.exit(1)
    print()

    # ── 3. Verify collection exists and vector count ────────────────────
    print("[Step 3] Verify collection and vector count")
    resp = requests.get(f"{BASE_URL}/collections/{exp['collection']}")
    collection_exists = resp.status_code == 200
    check("Collection exists after restart", collection_exists)

    if collection_exists:
        data = resp.json()
        count = data.get("vector_count", -1)
        check("Vector count matches",
              count == exp["num_vectors"],
              f"expected={exp['num_vectors']} got={count}")
    else:
        check("Vector count matches", False, "collection not found")
    print()

    # ── 4. Verify sample vectors have exact matching data ───────────────
    print(f"[Step 4] Verify {len(exp['sample_vectors'])} sample vectors")
    vectors_ok = 0
    vectors_checked = 0

    for vid_str, expected in exp["sample_vectors"].items():
        vid = int(vid_str)
        vectors_checked += 1
        resp = requests.get(
            f"{BASE_URL}/collections/{exp['collection']}/vectors/{vid}")

        if resp.status_code != 200:
            check(f"Vector {vid} exists", False,
                  f"status={resp.status_code}")
            continue

        actual = resp.json()

        # Check ID
        id_match = actual.get("id") == expected["id"]

        # Check values (float comparison with tolerance)
        actual_values = actual.get("values", [])
        expected_values = expected["values"]
        values_match = False
        if len(actual_values) == len(expected_values):
            max_diff = max(
                abs(a - e)
                for a, e in zip(actual_values, expected_values)
            ) if actual_values else 0.0
            values_match = max_diff < 1e-5

        # Check metadata
        actual_meta = actual.get("metadata", {})
        expected_meta = expected["metadata"]
        meta_match = True
        for key, val in expected_meta.items():
            actual_val = actual_meta.get(key)
            if isinstance(val, float):
                meta_match = meta_match and (
                    actual_val is not None
                    and abs(float(actual_val) - val) < 1e-3
                )
            else:
                meta_match = meta_match and (actual_val == val)

        all_match = id_match and values_match and meta_match
        if all_match:
            vectors_ok += 1

        if not all_match:
            detail_parts = []
            if not id_match:
                detail_parts.append(f"id mismatch ({actual.get('id')} != {expected['id']})")
            if not values_match:
                detail_parts.append(f"values mismatch (max_diff={max_diff:.2e})")
            if not meta_match:
                detail_parts.append("metadata mismatch")
            check(f"Vector {vid} data integrity", False,
                  "; ".join(detail_parts))

    check("All sample vectors match",
          vectors_ok == vectors_checked,
          f"{vectors_ok}/{vectors_checked} vectors intact")
    print()

    # ── 5. Re-run search queries and compare results ────────────────────
    print(f"[Step 5] Verify {len(exp['search_queries'])} search queries")
    search_match_count = 0

    for i, query_vec in enumerate(exp["search_queries"]):
        resp = requests.post(
            f"{BASE_URL}/collections/{exp['collection']}/search",
            json={"query": query_vec, "k": SEARCH_K},
        )
        if resp.status_code != 200:
            check(f"Search query {i} succeeded", False,
                  f"status={resp.status_code}")
            continue

        actual_results = resp.json().get("results", [])
        expected_results = exp["search_results"][i]

        # Compare result IDs in order
        actual_ids = [r["id"] for r in actual_results]
        expected_ids = [r["id"] for r in expected_results]
        ids_match = actual_ids == expected_ids

        # Compare scores with tolerance
        scores_match = True
        if len(actual_results) == len(expected_results):
            for ar, er in zip(actual_results, expected_results):
                if abs(ar["score"] - er["score"]) > 1e-4:
                    scores_match = False
                    break
        else:
            scores_match = False

        query_ok = ids_match and scores_match
        if query_ok:
            search_match_count += 1
        else:
            detail_parts = []
            if not ids_match:
                detail_parts.append(
                    f"IDs differ: got {actual_ids[:3]}... "
                    f"expected {expected_ids[:3]}...")
            if not scores_match:
                detail_parts.append("scores differ beyond tolerance")
            check(f"Search query {i} results match", False,
                  "; ".join(detail_parts))

    check("All search results match",
          search_match_count == len(exp["search_queries"]),
          f"{search_match_count}/{len(exp['search_queries'])} queries match")
    print()

    # ── 6. Verify virtual graph edges ───────────────────────────────────
    print("[Step 6] Verify virtual graph edges")
    graph_ok_count = 0
    graph_checked = 0

    for vid_str, expected_edges in exp["graph_samples"].items():
        vid = int(vid_str)
        graph_checked += 1

        resp = requests.get(
            f"{BASE_URL}/collections/{exp['collection']}/graph/related/{vid}",
            params={"threshold": exp["graph_threshold"]},
        )

        if resp.status_code != 200:
            check(f"Graph edges for vector {vid}", False,
                  f"status={resp.status_code}")
            continue

        actual_edges = resp.json().get("edges", [])

        # Compare edge target IDs (order may vary, compare as sets)
        actual_targets = set(e["target_id"] for e in actual_edges)
        expected_targets = set(e["target_id"] for e in expected_edges)

        targets_match = actual_targets == expected_targets

        # Compare similarities with tolerance
        sims_match = True
        if targets_match and expected_edges:
            actual_sim_map = {e["target_id"]: e["similarity"]
                              for e in actual_edges}
            for ee in expected_edges:
                actual_sim = actual_sim_map.get(ee["target_id"])
                if actual_sim is None or abs(actual_sim - ee["similarity"]) > 1e-4:
                    sims_match = False
                    break

        edge_ok = targets_match and sims_match
        if edge_ok:
            graph_ok_count += 1
        else:
            detail_parts = []
            if not targets_match:
                missing = expected_targets - actual_targets
                extra = actual_targets - expected_targets
                if missing:
                    detail_parts.append(f"missing targets: {missing}")
                if extra:
                    detail_parts.append(f"extra targets: {extra}")
            if not sims_match:
                detail_parts.append("similarity values differ")
            check(f"Graph edges for vector {vid}", False,
                  "; ".join(detail_parts))

    check("All graph edges match",
          graph_ok_count == graph_checked,
          f"{graph_ok_count}/{graph_checked} graph samples intact")
    print()

    # ── 7. Cleanup ──────────────────────────────────────────────────────
    print("[Step 7] Cleanup")
    resp = requests.delete(f"{BASE_URL}/collections/{exp['collection']}")
    cleanup_ok = resp.status_code == 200
    check("Collection deleted", cleanup_ok,
          f"status={resp.status_code}")
    print()

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"AFTER phase complete: {passed_count} PASS, {failed_count} FAIL")
    print("=" * 64)

    return failed_count == 0


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test SwarnDB persistence across server restart.",
        epilog=(
            "Example workflow:\n"
            "  1. python test_persistence_restart.py --phase=before\n"
            "  2. Restart SwarnDB server\n"
            "  3. python test_persistence_restart.py --phase=after\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["before", "after"],
        required=True,
        help="'before' inserts data and saves expectations; "
             "'after' verifies data survived restart.",
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.phase == "before":
        success = phase_before()
    else:
        success = phase_after()

    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.1f}s")
    sys.exit(0 if success else 1)
