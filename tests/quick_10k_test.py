#!/usr/bin/env python3
"""Quick 10K vector validation test against SwarnDB REST API."""

import time
import subprocess
import sys

import numpy as np
import requests

BASE_URL = "http://localhost:8080/api/v1"
COLLECTION = "test_10k"
DIM = 128
TOTAL_VECTORS = 10_000
BATCH_SIZE = 1_000

passed_count = 0
failed_count = 0
total_steps = 8


def report(step: int, name: str, ok: bool, detail: str = ""):
    global passed_count, failed_count
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed_count += 1
    else:
        failed_count += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] Step {step}: {name}{suffix}")


# ── Step 1: Create collection ───────────────────────────────────────────

def step1_create_collection():
    # Clean up if leftover from previous run
    requests.delete(f"{BASE_URL}/collections/{COLLECTION}")

    resp = requests.post(f"{BASE_URL}/collections", json={
        "name": COLLECTION,
        "dimension": DIM,
        "distance_metric": "cosine",
    })
    data = resp.json()
    ok = resp.status_code == 200 and data.get("success") is True
    report(1, "Create collection", ok,
           f"status={resp.status_code} success={data.get('success')}")
    return ok


# ── Step 2: Bulk insert 10K vectors ────────────────────────────────────

def step2_bulk_insert():
    rng = np.random.default_rng(seed=42)
    total_inserted = 0
    total_errors = 0
    t0 = time.time()

    num_batches = TOTAL_VECTORS // BATCH_SIZE
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
            data = resp.json()
            total_inserted += data.get("inserted_count", 0)
            total_errors += len(data.get("errors", []))
        else:
            total_errors += BATCH_SIZE
            print(f"    Batch {batch_num} failed: {resp.status_code} {resp.text[:200]}")

        # Progress indicator
        pct = (batch_num + 1) * 100 // num_batches
        print(f"    Batch {batch_num + 1}/{num_batches} done ({pct}%)", end="\r")

    elapsed = time.time() - t0
    print()  # newline after progress
    ok = total_inserted == TOTAL_VECTORS and total_errors == 0
    rate = total_inserted / elapsed if elapsed > 0 else 0
    report(2, "Bulk insert 10K vectors", ok,
           f"inserted={total_inserted} errors={total_errors} "
           f"time={elapsed:.1f}s rate={rate:.0f} vec/s")
    return ok


# ── Step 3: Verify count ───────────────────────────────────────────────

def step3_verify_count():
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    data = resp.json()
    count = data.get("vector_count", -1)
    ok = resp.status_code == 200 and count == TOTAL_VECTORS
    report(3, "Verify vector count", ok,
           f"expected={TOTAL_VECTORS} got={count}")
    return ok


# ── Step 4: Search test (10 queries) ──────────────────────────────────

def step4_search_test():
    rng = np.random.default_rng(seed=99)
    all_ok = True
    latencies = []

    for i in range(10):
        query = rng.standard_normal(DIM).tolist()
        t0 = time.time()
        resp = requests.post(
            f"{BASE_URL}/collections/{COLLECTION}/search",
            json={"query": query, "k": 10},
        )
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        if resp.status_code != 200:
            all_ok = False
            continue

        data = resp.json()
        results = data.get("results", [])
        if len(results) != 10:
            all_ok = False
            continue

        # Verify each result has a valid score (finite number)
        for r in results:
            score = r.get("score")
            if score is None or not isinstance(score, (int, float)):
                all_ok = False
                break

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p99_lat = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
    report(4, "Search test (10 queries, k=10)", all_ok,
           f"avg={avg_lat:.1f}ms p99={p99_lat:.1f}ms")
    return all_ok


# ── Step 5: Get single vector ─────────────────────────────────────────

def step5_get_vector():
    resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/1")
    if resp.status_code != 200:
        report(5, "Get single vector", False,
               f"status={resp.status_code}")
        return False

    data = resp.json()
    values = data.get("values", [])
    vec_id = data.get("id", -1)
    has_metadata = data.get("metadata") is not None

    ok = (vec_id == 1
          and len(values) == DIM
          and all(isinstance(v, (int, float)) for v in values)
          and has_metadata)
    report(5, "Get single vector", ok,
           f"id={vec_id} dim={len(values)} has_metadata={has_metadata}")
    return ok


# ── Step 6: Update vector metadata ───────────────────────────────────

def step6_update_metadata():
    resp = requests.put(
        f"{BASE_URL}/collections/{COLLECTION}/vectors/1",
        json={"metadata": {"updated": True}},
    )
    if resp.status_code != 200:
        report(6, "Update vector metadata", False,
               f"status={resp.status_code} body={resp.text[:200]}")
        return False

    data = resp.json()
    ok = data.get("success") is True

    # Verify the update stuck
    verify = requests.get(f"{BASE_URL}/collections/{COLLECTION}/vectors/1")
    if verify.status_code == 200:
        meta = verify.json().get("metadata", {})
        ok = ok and meta.get("updated") is True

    report(6, "Update vector metadata", ok,
           f"success={data.get('success')}")
    return ok


# ── Step 7: Delete a vector ──────────────────────────────────────────

def step7_delete_vector():
    resp = requests.delete(f"{BASE_URL}/collections/{COLLECTION}/vectors/2")
    if resp.status_code != 200:
        report(7, "Delete vector", False,
               f"status={resp.status_code}")
        return False

    data = resp.json()
    ok = data.get("success") is True

    # Verify count is now 9999
    count_resp = requests.get(f"{BASE_URL}/collections/{COLLECTION}")
    count = count_resp.json().get("vector_count", -1)
    ok = ok and count == TOTAL_VECTORS - 1

    report(7, "Delete vector and verify count", ok,
           f"success={data.get('success')} count={count} expected={TOTAL_VECTORS - 1}")
    return ok


# ── Step 8: Memory check ─────────────────────────────────────────────

def step8_memory_check():
    try:
        result = subprocess.run(
            ["docker", "stats", "swarndb", "--no-stream",
             "--format", "{{.MemUsage}} / {{.MemPerc}}"],
            capture_output=True, text=True, timeout=10,
        )
        mem_info = result.stdout.strip()
        ok = len(mem_info) > 0
        report(8, "Docker memory check", ok, f"memory={mem_info}")
        return ok
    except Exception as e:
        report(8, "Docker memory check", False, f"error={e}")
        return False


# ── Cleanup ──────────────────────────────────────────────────────────

def cleanup():
    print("\n  Cleaning up...")
    resp = requests.delete(f"{BASE_URL}/collections/{COLLECTION}")
    if resp.status_code == 200:
        print(f"  Collection '{COLLECTION}' deleted.")
    else:
        print(f"  Cleanup warning: status={resp.status_code}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SwarnDB 10K Vector Validation Test")
    print(f"Target: {BASE_URL}")
    print(f"Vectors: {TOTAL_VECTORS} x {DIM}-dim, batches of {BATCH_SIZE}")
    print("=" * 60)
    print()

    t_start = time.time()

    step1_create_collection()
    step2_bulk_insert()
    step3_verify_count()
    step4_search_test()
    step5_get_vector()
    step6_update_metadata()
    step7_delete_vector()
    step8_memory_check()

    cleanup()

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"RESULT: {passed_count}/{total_steps} PASSED")
    print(f"Total time: {elapsed:.1f}s")
    print("=" * 60)

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
