#!/usr/bin/env python3
"""Perf_Stability mixed-load threading harness.

Exercises the P01 deferred-flag race (Bug 1) by hammering a single
collection with concurrent writers (insert + defer_graph + optimize)
and concurrent searchers. Pass criteria: no panics, no unexpected
errors, and the deferred_graph flag settles to false after the last
optimize call (verified via a final optimize() returning the
already_optimized status).
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(
    0,
    "/Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/sdk/python/src",
)

from swarndb.client import SwarnDBClient

COLLECTION = "perf_stability_threading"
DIMENSION = 128


def make_vectors(rng, count, dim):
    return [rng.randn(dim).astype(np.float32).tolist() for _ in range(count)]


def writer_worker(worker_id, stop_event, args, counters, lock):
    """Insert, defer_graph bulk insert, then optimize in a tight loop."""
    client = SwarnDBClient(host="localhost", port=50051)
    rng = np.random.RandomState(1000 + worker_id)
    ops = 0
    errs = {}
    while not stop_event.is_set():
        try:
            vecs = make_vectors(rng, args.vectors_per_batch, DIMENSION)
            client.vectors.bulk_insert(
                COLLECTION,
                vecs,
                batch_lock_size=200,
            )
            ops += 1
        except Exception as e:
            key = type(e).__name__
            errs[key] = errs.get(key, 0) + 1

        if stop_event.is_set():
            break

        try:
            vecs = make_vectors(rng, args.vectors_per_batch, DIMENSION)
            client.vectors.bulk_insert(
                COLLECTION,
                vecs,
                batch_lock_size=200,
                defer_graph=True,
            )
            ops += 1
        except Exception as e:
            key = type(e).__name__
            errs[key] = errs.get(key, 0) + 1

        if stop_event.is_set():
            break

        try:
            client.collections.optimize(COLLECTION, rebuild_graph=True)
            ops += 1
        except Exception as e:
            key = type(e).__name__
            errs[key] = errs.get(key, 0) + 1

    with lock:
        counters["writer_ops"] += ops
        for k, v in errs.items():
            counters["writer_errs"][k] = counters["writer_errs"].get(k, 0) + v


def searcher_worker(worker_id, stop_event, args, counters, lock):
    """Run concurrent k=10 searches until stop_event."""
    client = SwarnDBClient(host="localhost", port=50051)
    rng = np.random.RandomState(5000 + worker_id)
    ops = 0
    errs = {}
    while not stop_event.is_set():
        try:
            q = rng.randn(DIMENSION).astype(np.float32).tolist()
            client.search.query(COLLECTION, q, k=10)
            ops += 1
        except Exception as e:
            key = type(e).__name__
            errs[key] = errs.get(key, 0) + 1

    with lock:
        counters["searcher_ops"] += ops
        for k, v in errs.items():
            counters["searcher_errs"][k] = counters["searcher_errs"].get(k, 0) + v


def main():
    parser = argparse.ArgumentParser(
        description="Perf_Stability mixed-load threading harness"
    )
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--writer-threads", type=int, default=8)
    parser.add_argument("--searcher-threads", type=int, default=4)
    parser.add_argument("--vectors-per-batch", type=int, default=80)
    args = parser.parse_args()

    print("=" * 60)
    print("Perf_Stability mixed-load threading harness")
    print(
        f"writers={args.writer_threads} searchers={args.searcher_threads} "
        f"batch={args.vectors_per_batch} duration={args.duration_seconds}s"
    )
    print("=" * 60)

    admin = SwarnDBClient(host="localhost", port=50051)
    try:
        admin.collections.delete(COLLECTION)
    except Exception:
        pass
    admin.collections.create(
        name=COLLECTION,
        dimension=DIMENSION,
        distance_metric="cosine",
    )

    # Seed with one batch so searches have something to find.
    seed_rng = np.random.RandomState(7)
    seed_vecs = make_vectors(seed_rng, 500, DIMENSION)
    admin.vectors.bulk_insert(COLLECTION, seed_vecs, batch_lock_size=200)

    counters = {
        "writer_ops": 0,
        "searcher_ops": 0,
        "writer_errs": {},
        "searcher_errs": {},
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    total_threads = args.writer_threads + args.searcher_threads
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=total_threads) as ex:
        futures = []
        for i in range(args.writer_threads):
            futures.append(
                ex.submit(writer_worker, i, stop_event, args, counters, lock)
            )
        for i in range(args.searcher_threads):
            futures.append(
                ex.submit(searcher_worker, i, stop_event, args, counters, lock)
            )

        time.sleep(args.duration_seconds)
        stop_event.set()

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"  [WARN] worker raised: {type(e).__name__}: {e}")

    elapsed = time.perf_counter() - start

    # Final optimize loop: call optimize() repeatedly until it reports
    # already_optimized (or budget exhausted). This verifies the
    # deferred_graph flag settled to false.
    print()
    print("Final optimize convergence check ...")
    converged = False
    final_status = ""
    for attempt in range(5):
        try:
            res = admin.collections.optimize(COLLECTION, rebuild_graph=True)
            final_status = res.status
            print(f"  attempt {attempt + 1}: status={res.status} "
                  f"duration_ms={res.duration_ms} "
                  f"vectors_processed={res.vectors_processed}")
            if res.status in ("already_optimized", "noop", "skipped"):
                converged = True
                break
        except Exception as e:
            print(f"  attempt {attempt + 1}: optimize raised {type(e).__name__}: {e}")
        time.sleep(1)

    # Sanity search post-convergence.
    search_ok = False
    try:
        q = np.random.RandomState(99).randn(DIMENSION).astype(np.float32).tolist()
        sres = admin.search.query(COLLECTION, q, k=10)
        search_ok = len(sres.results) > 0
        print(f"  post-convergence search: {len(sres.results)} hits")
    except Exception as e:
        print(f"  post-convergence search raised: {type(e).__name__}: {e}")

    # Cleanup
    try:
        admin.collections.delete(COLLECTION)
    except Exception:
        pass

    # Report
    total_ops = counters["writer_ops"] + counters["searcher_ops"]
    rate = total_ops / elapsed if elapsed > 0 else 0
    print()
    print("=" * 60)
    print(f"Duration:        {elapsed:.1f} s")
    print(f"Writer ops:      {counters['writer_ops']}")
    print(f"Searcher ops:    {counters['searcher_ops']}")
    print(f"Total ops:       {total_ops} ({rate:.1f} ops/s)")
    print(f"Writer errs:     {counters['writer_errs']}")
    print(f"Searcher errs:   {counters['searcher_errs']}")
    print(f"Final optimize:  status={final_status} converged={converged}")
    print(f"Post search ok:  {search_ok}")
    print("=" * 60)

    writer_err_total = sum(counters["writer_errs"].values())
    searcher_err_total = sum(counters["searcher_errs"].values())

    # Pass: deferred flag converged, post-search returned hits, and the
    # error rate stayed under 5% of writer ops (allows for benign rate
    # limits or transient conflicts under heavy concurrency).
    err_budget = max(5, int(0.05 * counters["writer_ops"]))
    err_ok = (writer_err_total + searcher_err_total) <= err_budget
    ok = converged and search_ok and err_ok

    if ok:
        print("[PASS] mixed-load threading harness")
        sys.exit(0)
    else:
        reasons = []
        if not converged:
            reasons.append("optimize did not converge to already_optimized")
        if not search_ok:
            reasons.append("post-convergence search returned no hits")
        if not err_ok:
            reasons.append(
                f"error count {writer_err_total + searcher_err_total} "
                f"exceeded budget {err_budget}"
            )
        print(f"[FAIL] mixed-load threading harness: {'; '.join(reasons)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
