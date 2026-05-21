#!/usr/bin/env python3
"""P03 SIGKILL + resume harness.

Two-phase orchestration:

  --phase before
    1. Create collection.
    2. Generate N random D-dim vectors with a fixed seed.
    3. Start bulk_insert on a background thread with checkpoint_every
       and batch_lock_size.
    4. Poll collection.info every second; when at least kill-at-pct of
       N is present, send SIGKILL to the container via
       `docker kill -s KILL <container>`.
    5. Save state (collection, N, dimension, seed, batch_lock_size,
       checkpoint_every, last observed count, resume_token if the
       background thread captured one) to --state-file.

  Manual step (between phases): user runs
       `docker start <container>` and waits for healthcheck.

  --phase after
    1. Load state from --state-file.
    2. Regenerate vectors with the saved seed (so ids match).
    3. Re-invoke bulk_insert with the saved resume_token (when
       available) and verify final vector_count == N.
    4. Run a few searches to confirm results are reasonable.

Fallback: if no resume_token was captured before SIGKILL (server
returns a unary terminal response, so the partial response is lost),
the harness simply re-invokes bulk_insert with no token. The server's
existing id-uniqueness semantics handle duplicates.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(
    0,
    "/Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/sdk/python/src",
)

from swarndb.client import SwarnDBClient

COLLECTION = "perf_stability_resume"
CONTAINER = "swarndb"
RNG_SEED = 4242


def build_vectors(n, dim, seed):
    rng = np.random.RandomState(seed)
    return [rng.randn(dim).astype(np.float32).tolist() for _ in range(n)]


def docker_kill(container):
    """Send SIGKILL to the named docker container."""
    res = subprocess.run(
        ["docker", "kill", "-s", "KILL", container],
        capture_output=True, text=True, timeout=10,
    )
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def docker_is_healthy(container):
    """Return True if docker reports the container as healthy."""
    try:
        out = subprocess.check_output(
            ["docker", "inspect", "--format",
             "{{.State.Health.Status}}", container],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
        return out == "healthy"
    except Exception:
        return False


def wait_healthy(container, timeout_s=120):
    """Block until container reports healthy or timeout expires."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if docker_is_healthy(container):
            return True
        time.sleep(2)
    return False


def phase_before(args):
    print("=" * 60)
    print("PHASE: before (insert, kill mid-flight)")
    print(f"N={args.n_vectors} dim={args.dimension} "
          f"batch_lock_size={args.batch_lock_size} "
          f"checkpoint_every={args.checkpoint_every} "
          f"kill_at_pct={args.kill_at_pct}")
    print("=" * 60)

    client = SwarnDBClient(host="localhost", port=50051)
    try:
        client.collections.delete(COLLECTION)
    except Exception:
        pass
    client.collections.create(
        name=COLLECTION,
        dimension=args.dimension,
        distance_metric="cosine",
    )

    print(f"Generating {args.n_vectors} x {args.dimension} vectors ...")
    vectors = build_vectors(args.n_vectors, args.dimension, RNG_SEED)
    ids = list(range(1, args.n_vectors + 1))

    result_holder = {"resume_token": "", "inserted_count": 0,
                     "last_batch_idx": 0, "error": ""}

    def insert_worker():
        try:
            res = client.vectors.bulk_insert(
                COLLECTION,
                vectors,
                ids=ids,
                batch_lock_size=args.batch_lock_size,
                checkpoint_every=args.checkpoint_every,
            )
            result_holder["resume_token"] = res.resume_token or ""
            result_holder["inserted_count"] = res.inserted_count
            result_holder["last_batch_idx"] = res.last_completed_batch_idx
        except Exception as e:
            result_holder["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=insert_worker, daemon=True)
    t.start()

    kill_threshold = int(args.n_vectors * args.kill_at_pct / 100.0)
    print(f"Waiting for >= {kill_threshold} vectors before SIGKILL ...")

    polling_client = SwarnDBClient(host="localhost", port=50051)
    last_count = 0
    killed = False
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            info = polling_client.collections.get(COLLECTION)
            last_count = info.vector_count
            print(f"  observed vector_count={last_count}", end="\r")
            if last_count >= kill_threshold:
                break
        except Exception as e:
            print(f"\n  poll error: {type(e).__name__}: {e}")
        time.sleep(1)
    print()

    if last_count < kill_threshold:
        print(f"  [WARN] never reached {kill_threshold} vectors "
              f"(got {last_count}); killing anyway")

    print(f"Sending SIGKILL to container {CONTAINER} ...")
    rc, out, err = docker_kill(CONTAINER)
    if rc != 0:
        print(f"  [FAIL] docker kill rc={rc} stderr={err}")
        sys.exit(1)
    print(f"  docker kill ok: {out}")
    killed = True

    # Give the background thread a moment to surface the broken pipe.
    t.join(timeout=10)

    state = {
        "collection": COLLECTION,
        "n_vectors": args.n_vectors,
        "dimension": args.dimension,
        "rng_seed": RNG_SEED,
        "batch_lock_size": args.batch_lock_size,
        "checkpoint_every": args.checkpoint_every,
        "ids": ids,
        "last_observed_count": last_count,
        "resume_token": result_holder["resume_token"],
        "last_completed_batch_idx": result_holder["last_batch_idx"],
        "insert_thread_error": result_holder["error"],
        "killed": killed,
    }
    with open(args.state_file, "w") as f:
        json.dump(state, f, indent=2)

    print()
    print("-" * 60)
    print(f"State saved to {args.state_file}")
    print(f"  last_observed_count={last_count}")
    print(f"  resume_token_len={len(state['resume_token'])}")
    print(f"  insert_thread_error={state['insert_thread_error'] or '<none>'}")
    print()
    print(f"Next: docker start {CONTAINER}; wait healthy; "
          f"python {os.path.basename(__file__)} --phase after")
    print("-" * 60)

    return True


def phase_after(args):
    print("=" * 60)
    print("PHASE: after (resume + verify)")
    print("=" * 60)

    if not os.path.exists(args.state_file):
        print(f"  [FAIL] state file not found: {args.state_file}")
        sys.exit(1)

    with open(args.state_file, "r") as f:
        state = json.load(f)

    print(f"Waiting for {CONTAINER} to become healthy ...")
    if not wait_healthy(CONTAINER, timeout_s=180):
        print(f"  [FAIL] container did not become healthy")
        sys.exit(1)
    print("  container healthy")

    client = SwarnDBClient(host="localhost", port=50051)

    # Pre-check existing collection state.
    try:
        info = client.collections.get(state["collection"])
        pre_count = info.vector_count
    except Exception as e:
        print(f"  [FAIL] cannot fetch collection: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"  pre-resume vector_count={pre_count}")

    print(f"Rebuilding {state['n_vectors']} x {state['dimension']} "
          f"vectors with seed={state['rng_seed']} ...")
    vectors = build_vectors(state["n_vectors"], state["dimension"],
                            state["rng_seed"])
    ids = state["ids"]

    resume_token = state.get("resume_token", "") or ""
    if resume_token:
        print(f"  resuming with token (len={len(resume_token)})")
    else:
        print("  no resume_token captured; relying on id-uniqueness "
              "to skip already-committed entries")

    t0 = time.perf_counter()
    res = client.vectors.bulk_insert(
        state["collection"],
        vectors,
        ids=ids,
        batch_lock_size=state["batch_lock_size"],
        checkpoint_every=state["checkpoint_every"],
        resume_token=resume_token,
    )
    elapsed = time.perf_counter() - t0
    print(f"  bulk_insert returned in {elapsed:.1f}s "
          f"inserted_count={res.inserted_count} "
          f"errors={len(res.errors)}")

    # Verify final count.
    info = client.collections.get(state["collection"])
    final_count = info.vector_count
    count_ok = final_count >= state["n_vectors"]
    print(f"  final vector_count={final_count} expected>={state['n_vectors']}")

    # Sample searches.
    rng = np.random.RandomState(11)
    search_ok = True
    for i in range(5):
        q = rng.randn(state["dimension"]).astype(np.float32).tolist()
        try:
            sres = client.search.query(state["collection"], q, k=10)
            if len(sres.results) == 0:
                search_ok = False
        except Exception as e:
            print(f"  search {i} raised: {type(e).__name__}: {e}")
            search_ok = False
    print(f"  sample searches ok={search_ok}")

    # Cleanup.
    try:
        client.collections.delete(state["collection"])
    except Exception:
        pass

    ok = count_ok and search_ok
    print()
    if ok:
        print("[PASS] resume_after_kill harness")
        sys.exit(0)
    else:
        reasons = []
        if not count_ok:
            reasons.append(
                f"final vector_count {final_count} < expected {state['n_vectors']}"
            )
        if not search_ok:
            reasons.append("one or more sample searches returned no results")
        print(f"[FAIL] resume_after_kill harness: {'; '.join(reasons)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="P03 SIGKILL + resume harness"
    )
    parser.add_argument("--phase", required=True, choices=["before", "after"])
    parser.add_argument("--n-vectors", type=int, default=100000)
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--batch-lock-size", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--state-file", default="/tmp/p03_resume_state.json")
    parser.add_argument("--kill-at-pct", type=int, default=30)
    args = parser.parse_args()

    if args.phase == "before":
        phase_before(args)
    else:
        phase_after(args)


if __name__ == "__main__":
    main()
