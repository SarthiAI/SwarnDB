#!/usr/bin/env python3
"""
SwarnDB recall-against-ground-truth harness (Perf_Stability P10.8).

Purpose
=======

This harness measures search recall@k of a running SwarnDB collection
against brute-force ground truth computed offline from the raw dataset.
It is the final P10.8 validation harness slot, complementing the
existing memory-release, concurrent-search-under-write, and D-1
collision harnesses with a recall-correctness gate.

The harness has two callable surfaces:

A. Library mode
---------------

    from test_recall_ground_truth import compute_or_load_ground_truth

    query_vectors, query_indices, ground_truth_top_k = (
        compute_or_load_ground_truth(
            dataset_path="/path/to/dbpedia.npy",
            n_queries=100,
            k=10,
            cache_path="/tmp/swarndb_recall_gt_100_10.npz",
            seed=42,
            distance_metric="cosine",
        )
    )

The function loads a numpy `.npy` dataset of shape (N, dim), picks
n_queries deterministic indices via numpy.random.default_rng(seed),
computes brute-force top-k nearest neighbors per query (cosine via
dot product on L2-normalized vectors, or L2 via squared Euclidean),
and caches results. Subsequent runs with the same metadata reload
from cache instead of recomputing.

B. CLI mode
-----------

Run against a collection that has already been loaded with the same
dataset (this harness does NOT create or insert):

    python3 test_recall_ground_truth.py \\
        --dataset-path /data/dbpedia_1m_1536.npy \\
        --collection-name dbpedia_1m \\
        --n-queries 100 --k 10 \\
        --rest-port 18098 --grpc-port 18099 \\
        --distance-metric cosine \\
        --recall-ceiling 0.9

Exits 0 on PASS (mean recall@k >= ceiling), 1 on FAIL, 2 on setup
error (bad args, dataset missing, server not reachable, etc.).

ID-mapping assumption
=====================

The dataset `.npy` rows are 0-based positions, but SwarnDB assigns its
own vector ids when the collection is loaded. The simplest mapping the
harness uses is:

    swarndb_id = dataset_position + id_offset

with `--id-offset` defaulting to 1 (matches the convention used by the
P10 and P10.5 load scripts which insert dataset row i as id i+1).
Adjust `--id-offset` if your loader used a different base. This
assumption is the recall correctness contract: if the offset is wrong,
recall will read as near-zero even on a perfect index.

Distance metric
===============

For cosine the harness L2-normalizes both the dataset rows and each
query vector before computing dot products; brute-force top-k is then
argpartition over the negative similarity. For L2 the harness computes
squared Euclidean distance and argpartitions over the positive
distance. The collection on the server side must have been created
with the matching distance metric; the harness does not (and cannot)
query the server's distance metric, so this is a caller invariant.

Server discovery
================

Mirrors the P10.6 `find_or_spawn_server` pattern from
`test_d1_collision_concurrent.py`: if /readyz returns 200 on the
configured REST port, the harness reuses that server. Otherwise it
spawns one from --binary-path. If neither path works, the harness
errors out clearly and exits 2.

Hard rules followed (per P10.8 spec)
====================================

- No git commit, no push, no git mutation.
- No Civo / no SSH.
- No em-dashes anywhere in this file.
- No new pyproject dependencies (numpy is already in the venv per
  existing harnesses).
- Only this one new file is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

# Make the in-tree SDK importable so the harness runs from a checkout
# without a wheel install. Mirrors the pattern used by every other
# perf-stability harness in this folder.
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("recall_ground_truth_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------

# Port assignments per P10.8 spec (matches the P10.3B / P10.8 matrix).
DEFAULT_REST_PORT = 18098
DEFAULT_GRPC_PORT = 18099

# /readyz contract.
READYZ_DEADLINE_SECONDS = 60.0
READYZ_POLL_INTERVAL_SECONDS = 0.5
EXTERNAL_PROBE_TIMEOUT_SECONDS = 2.0

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

DEFAULT_N_QUERIES = 100
DEFAULT_K = 10
DEFAULT_SEED = 42
DEFAULT_DISTANCE_METRIC = "cosine"
DEFAULT_RECALL_CEILING = 0.9
DEFAULT_ID_OFFSET = 1

# Brute-force batching: rows per chunk while computing similarity. Tuned
# for memory: at dim=1536, float32, 50000 rows per chunk is roughly
# 300 MB plus the queries, well within a dev box budget.
BRUTE_FORCE_DATASET_CHUNK = 50_000


# ---------------------------------------------------------------------------
# Cache metadata helpers
# ---------------------------------------------------------------------------


def _dataset_fingerprint(dataset_path: str) -> str:
    """Stable fingerprint of the dataset file.

    Uses (absolute_path, size, mtime_ns) hashed with sha256. This
    avoids hashing multi-GB files while still invalidating the cache
    if the dataset file changes underneath the harness.
    """
    p = Path(dataset_path).resolve()
    stat = p.stat()
    raw = f"{p}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_cache_meta(
    dataset_path: str,
    n_queries: int,
    k: int,
    seed: int,
    distance_metric: str,
) -> Dict[str, Any]:
    return {
        "dataset_fingerprint": _dataset_fingerprint(dataset_path),
        "n_queries": int(n_queries),
        "k": int(k),
        "seed": int(seed),
        "distance_metric": str(distance_metric),
        "harness_version": 1,
    }


def _meta_sidecar_path(cache_path: str) -> Path:
    return Path(str(cache_path) + ".meta.json")


def _read_cache_meta(cache_path: str) -> Optional[Dict[str, Any]]:
    sidecar = _meta_sidecar_path(cache_path)
    if not sidecar.exists():
        return None
    try:
        with sidecar.open("r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cache meta unreadable at %s: %s", sidecar, exc)
        return None


def _write_cache_meta(cache_path: str, meta: Dict[str, Any]) -> None:
    sidecar = _meta_sidecar_path(cache_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Brute-force ground truth
# ---------------------------------------------------------------------------


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D float32 array.

    Zero-norm rows are left as zeros (cosine similarity with a zero
    vector is undefined; the dataset should not contain them in
    practice, but the harness must not crash if it does).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return (matrix / safe).astype(np.float32, copy=False)


def _brute_force_top_k_cosine(
    dataset: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
) -> np.ndarray:
    """Per-query brute-force top-k by cosine similarity.

    Both inputs must be L2-normalized. Similarity is dataset @ query.T;
    the top-k are the indices of the k largest similarities. Returns
    int64 array shape (n_queries, k), sorted by descending similarity.

    Uses chunked argpartition over the dataset to keep peak memory
    bounded on large corpora.
    """
    n_queries = query_vectors.shape[0]
    n_rows = dataset.shape[0]
    if k > n_rows:
        raise ValueError(
            f"k={k} exceeds dataset size {n_rows}; cannot compute top-k"
        )

    # Per-query running top-k via heap-style accumulation across chunks.
    best_idx = np.full((n_queries, k), -1, dtype=np.int64)
    best_sim = np.full((n_queries, k), -np.inf, dtype=np.float32)

    for chunk_lo in range(0, n_rows, BRUTE_FORCE_DATASET_CHUNK):
        chunk_hi = min(chunk_lo + BRUTE_FORCE_DATASET_CHUNK, n_rows)
        chunk = dataset[chunk_lo:chunk_hi]
        # shape (chunk_size, n_queries)
        sims = chunk @ query_vectors.T

        for qi in range(n_queries):
            chunk_sims = sims[:, qi]
            # Merge current best with chunk candidates.
            merged_sims = np.concatenate([best_sim[qi], chunk_sims])
            merged_idx = np.concatenate(
                [best_idx[qi], np.arange(chunk_lo, chunk_hi, dtype=np.int64)]
            )
            if merged_sims.shape[0] <= k:
                top_local = np.argsort(-merged_sims)
            else:
                part = np.argpartition(-merged_sims, k)[:k]
                top_local = part[np.argsort(-merged_sims[part])]
            best_sim[qi] = merged_sims[top_local][:k]
            best_idx[qi] = merged_idx[top_local][:k]

    return best_idx


def _brute_force_top_k_l2(
    dataset: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
) -> np.ndarray:
    """Per-query brute-force top-k by squared L2 distance (smallest k).

    Uses the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, so the
    ordering can be done on negative similarity once the per-row norm
    is folded in.
    """
    n_queries = query_vectors.shape[0]
    n_rows = dataset.shape[0]
    if k > n_rows:
        raise ValueError(
            f"k={k} exceeds dataset size {n_rows}; cannot compute top-k"
        )

    query_sq = np.sum(query_vectors * query_vectors, axis=1)  # (n_queries,)

    best_idx = np.full((n_queries, k), -1, dtype=np.int64)
    best_dist = np.full((n_queries, k), np.inf, dtype=np.float32)

    for chunk_lo in range(0, n_rows, BRUTE_FORCE_DATASET_CHUNK):
        chunk_hi = min(chunk_lo + BRUTE_FORCE_DATASET_CHUNK, n_rows)
        chunk = dataset[chunk_lo:chunk_hi]
        chunk_sq = np.sum(chunk * chunk, axis=1)  # (chunk_size,)
        # squared distance per (chunk_row, query)
        dots = chunk @ query_vectors.T  # (chunk_size, n_queries)
        dist = (
            chunk_sq[:, None]
            + query_sq[None, :]
            - 2.0 * dots
        ).astype(np.float32, copy=False)
        # Negative values can appear from float rounding; clamp.
        np.maximum(dist, 0.0, out=dist)

        for qi in range(n_queries):
            chunk_dist = dist[:, qi]
            merged_dist = np.concatenate([best_dist[qi], chunk_dist])
            merged_idx = np.concatenate(
                [best_idx[qi], np.arange(chunk_lo, chunk_hi, dtype=np.int64)]
            )
            if merged_dist.shape[0] <= k:
                top_local = np.argsort(merged_dist)
            else:
                part = np.argpartition(merged_dist, k)[:k]
                top_local = part[np.argsort(merged_dist[part])]
            best_dist[qi] = merged_dist[top_local][:k]
            best_idx[qi] = merged_idx[top_local][:k]

    return best_idx


# ---------------------------------------------------------------------------
# Public library entry point
# ---------------------------------------------------------------------------


def compute_or_load_ground_truth(
    dataset_path: str,
    n_queries: int,
    k: int,
    cache_path: str,
    seed: int = 42,
    distance_metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute or load brute-force top-k nearest neighbors for n_queries.

    The function loads a numpy `.npy` dataset of shape (N, dim), picks
    n_queries deterministic indices via numpy.random.default_rng(seed),
    computes brute-force top-k per query, and caches the result in
    `cache_path` (a .npz) plus a sidecar `<cache_path>.meta.json` whose
    contents act as the cache key. Cache hit means the metadata at
    `<cache_path>.meta.json` matches exactly; otherwise the cache is
    rebuilt.

    Args:
        dataset_path: Path to a numpy `.npy` file shape (N, dim), float.
        n_queries: How many query vectors to draw from the dataset.
        k: Top-k nearest neighbors to compute per query.
        cache_path: Path to the `.npz` cache file. Created if missing.
        seed: RNG seed for query index selection (default 42).
        distance_metric: "cosine" or "l2" (default "cosine").

    Returns:
        Tuple of:
            query_vectors:        shape (n_queries, dim), float32.
            query_indices:        shape (n_queries,), int64 (positions
                                  in the dataset).
            ground_truth_top_k:   shape (n_queries, k), int64 (positions
                                  in the dataset). Sorted descending by
                                  similarity for cosine; ascending by
                                  distance for L2.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If distance_metric is not one of the supported set,
            or if k > N.
    """
    if distance_metric not in ("cosine", "l2"):
        raise ValueError(
            f"distance_metric must be 'cosine' or 'l2', got {distance_metric!r}"
        )

    dataset_p = Path(dataset_path)
    if not dataset_p.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    desired_meta = _build_cache_meta(
        dataset_path, n_queries, k, seed, distance_metric,
    )

    cache_p = Path(cache_path)
    cached_meta = _read_cache_meta(cache_path)
    cache_hit = (
        cache_p.exists()
        and cached_meta is not None
        and cached_meta == desired_meta
    )

    # The dataset always has to be memory-mapped at least for the query
    # extraction step. The brute-force pass also needs it. Use mmap so
    # the OS pages it in lazily for large corpora.
    logger.info("loading dataset (mmap) from %s", dataset_path)
    dataset_full = np.load(str(dataset_p), mmap_mode="r")
    if dataset_full.ndim != 2:
        raise ValueError(
            f"dataset must be 2D (N, dim), got shape {dataset_full.shape}"
        )
    n_rows, dim = dataset_full.shape
    if n_queries > n_rows:
        raise ValueError(
            f"n_queries={n_queries} exceeds dataset rows {n_rows}"
        )
    if k > n_rows:
        raise ValueError(
            f"k={k} exceeds dataset rows {n_rows}"
        )

    rng = np.random.default_rng(seed)
    query_indices = np.sort(
        rng.choice(n_rows, size=n_queries, replace=False)
    ).astype(np.int64)
    # Materialize the queries as a contiguous float32 array.
    query_vectors = np.ascontiguousarray(
        dataset_full[query_indices], dtype=np.float32,
    )

    if cache_hit:
        logger.info("ground-truth cache HIT at %s", cache_path)
        loaded = np.load(str(cache_p))
        cached_query_indices = loaded["query_indices"].astype(np.int64)
        cached_top_k = loaded["ground_truth_top_k"].astype(np.int64)
        # Sanity-check the cache: indices must match the selection we
        # just made from the same seed.
        if (
            cached_query_indices.shape == query_indices.shape
            and np.array_equal(cached_query_indices, query_indices)
            and cached_top_k.shape == (n_queries, k)
        ):
            return query_vectors, query_indices, cached_top_k
        logger.warning(
            "ground-truth cache shape mismatch (cached %s vs requested "
            "(%d, %d)); rebuilding.",
            cached_top_k.shape, n_queries, k,
        )

    logger.info(
        "computing ground truth: n_queries=%d k=%d metric=%s dim=%d N=%d",
        n_queries, k, distance_metric, dim, n_rows,
    )
    start = time.perf_counter()

    if distance_metric == "cosine":
        # Materialize the full dataset as float32 and L2-normalize.
        # For very large N this can be heavy; the chunked routine
        # already keeps the dot-product side memory-bounded, so the
        # full normalize is the only one-time cost.
        logger.info("normalizing dataset rows (cosine path)")
        dataset_f32 = np.ascontiguousarray(dataset_full, dtype=np.float32)
        dataset_norm = _l2_normalize_rows(dataset_f32)
        queries_norm = _l2_normalize_rows(query_vectors)
        ground_truth_top_k = _brute_force_top_k_cosine(
            dataset_norm, queries_norm, k,
        )
    else:
        dataset_f32 = np.ascontiguousarray(dataset_full, dtype=np.float32)
        ground_truth_top_k = _brute_force_top_k_l2(
            dataset_f32, query_vectors, k,
        )

    elapsed = time.perf_counter() - start
    logger.info(
        "ground truth computed in %.2fs; writing cache to %s",
        elapsed, cache_path,
    )

    cache_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(cache_p),
        query_indices=query_indices,
        ground_truth_top_k=ground_truth_top_k.astype(np.int64),
    )
    _write_cache_meta(cache_path, desired_meta)

    return query_vectors, query_indices, ground_truth_top_k.astype(np.int64)


# ---------------------------------------------------------------------------
# Process management (mirrors P10.6 pattern)
# ---------------------------------------------------------------------------


class SwarndbProcess:
    """Spawn-and-supervise wrapper around the swarndb binary.

    Copied verbatim in shape from `test_d1_collision_concurrent.py` so
    every P10.x harness behaves the same way on spawn and teardown.
    """

    def __init__(
        self,
        binary: Path,
        data_dir: Path,
        rest_port: int,
        grpc_port: int,
        log_path: Path,
        extra_env: Optional[dict] = None,
    ) -> None:
        self.binary = binary
        self.data_dir = data_dir
        self.rest_port = rest_port
        self.grpc_port = grpc_port
        self.log_path = log_path
        self.extra_env = extra_env or {}
        self.proc: Optional[subprocess.Popen] = None
        self._log_fh = None

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("swarndb process already running")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["SWARNDB_DATA_DIR"] = str(self.data_dir)
        env["SWARNDB_REST_PORT"] = str(self.rest_port)
        env["SWARNDB_GRPC_PORT"] = str(self.grpc_port)
        env["SWARNDB_HOST"] = "127.0.0.1"
        env.update(self.extra_env)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self.log_path.open("ab")
        logger.info(
            "spawning swarndb: binary=%s data_dir=%s rest=%d grpc=%d log=%s",
            self.binary, self.data_dir, self.rest_port, self.grpc_port,
            self.log_path,
        )
        self.proc = subprocess.Popen(
            [str(self.binary)],
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def terminate(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            self._close_log()
            return
        try:
            self.proc.terminate()
        except ProcessLookupError:
            self._close_log()
            return
        deadline = time.time() + PROCESS_TERMINATE_GRACE_SECONDS
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self._close_log()
                return
            time.sleep(0.1)
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=PROCESS_KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("swarndb did not exit after SIGKILL fallback")
        self._close_log()

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None


def wait_for_readyz(
    rest_port: int,
    deadline_seconds: float,
    started_at: float,
) -> Tuple[bool, float]:
    """Poll /readyz until 200 OK or the deadline elapses."""
    url = f"http://127.0.0.1:{rest_port}/readyz"
    deadline = started_at + deadline_seconds
    last_status: Optional[int] = None
    last_body: str = ""

    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2.0)
            last_status = resp.status_code
            last_body = resp.text[:400]
            if resp.status_code == 200:
                return True, time.time() - started_at
        except (requests.ConnectionError, requests.Timeout):
            last_status = None
            last_body = ""
        time.sleep(READYZ_POLL_INTERVAL_SECONDS)

    elapsed = time.time() - started_at
    logger.error(
        "readyz never returned 200 within %.1fs; last_status=%s last_body=%s",
        deadline_seconds, last_status, last_body,
    )
    return False, elapsed


def find_or_spawn_server(
    args: argparse.Namespace,
    log: logging.Logger,
    proc_factory: Callable[[], SwarndbProcess],
) -> Tuple[str, str, Optional[SwarndbProcess]]:
    """Auto-detect a swarndb-server on args.rest_port; else spawn one.

    Returns (rest_url, grpc_url, spawned_process_or_None). When the
    harness reuses an external server it returns None for the process
    handle so teardown skips a process it does not own. When the harness
    spawns the server it returns the wrapper so the caller can
    terminate it on exit.
    """
    rest_url = f"http://localhost:{args.rest_port}"
    grpc_url = f"localhost:{args.grpc_port}"
    external_detected = False
    try:
        req = urllib.request.Request(f"{rest_url}/readyz")
        with urllib.request.urlopen(req, timeout=EXTERNAL_PROBE_TIMEOUT_SECONDS) as resp:
            if resp.status == 200:
                external_detected = True
    except (urllib.error.URLError, OSError, ConnectionError, TimeoutError):
        external_detected = False

    if external_detected:
        log.info(
            "Found existing swarndb-server on %s; using it "
            "(skipping spawn and teardown).",
            rest_url,
        )
        return rest_url, grpc_url, None

    if not args.binary_path:
        log.error(
            "No swarndb-server detected on port %d and --binary-path "
            "not provided. Either start a swarndb-server on the "
            "configured port OR pass --binary-path to spawn one.",
            args.rest_port,
        )
        sys.exit(2)

    log.info(
        "No swarndb-server on %s; spawning a fresh one from %s.",
        rest_url, args.binary_path,
    )
    proc = proc_factory()
    proc.start()
    ok, elapsed = wait_for_readyz(
        args.rest_port, READYZ_DEADLINE_SECONDS, time.time(),
    )
    if not ok:
        log.error(
            "Spawned swarndb-server did not return /readyz=200 within "
            "%.1fs (elapsed=%.1fs); aborting.",
            READYZ_DEADLINE_SECONDS, elapsed,
        )
        proc.terminate()
        sys.exit(2)
    return rest_url, grpc_url, proc


# ---------------------------------------------------------------------------
# SDK helpers
# ---------------------------------------------------------------------------


def make_client(grpc_port: int) -> SwarnDBClient:
    """Build an SDK client.

    Generous timeout and retries because the harness may run while the
    server is also handling concurrent writes from a sibling harness on
    the same box.
    """
    return SwarnDBClient(
        host="127.0.0.1",
        port=grpc_port,
        timeout=600.0,
        max_retries=5,
        retry_delay=1.0,
    )


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------


def _normalize_query_for_search(
    query: np.ndarray,
    distance_metric: str,
) -> List[float]:
    """Prepare a single query vector for the SDK search call.

    For cosine the harness L2-normalizes the query so that the
    server-side cosine path sees a unit vector (the server normalizes
    internally too; pre-normalizing is harmless and keeps the recall
    math symmetric with the brute-force side). For L2 the query is
    passed through unchanged.
    """
    if distance_metric == "cosine":
        norm = np.linalg.norm(query)
        if norm > 0.0:
            return (query / norm).astype(np.float32).tolist()
        return query.astype(np.float32).tolist()
    return query.astype(np.float32).tolist()


def _dataset_position_to_swarndb_id(pos: int, id_offset: int) -> int:
    """Map a 0-based dataset position to a SwarnDB id."""
    return int(pos) + int(id_offset)


def measure_recall(
    client: SwarnDBClient,
    collection: str,
    query_vectors: np.ndarray,
    ground_truth_top_k: np.ndarray,
    k: int,
    distance_metric: str,
    id_offset: int,
) -> Dict[str, Any]:
    """Run each query through the SDK and compute recall@k.

    For each query qi the harness:
        1. Calls client.search.query(collection, vec, k=k).
        2. Extracts the returned ids (set).
        3. Maps the ground-truth dataset positions to expected
           SwarnDB ids via id_offset.
        4. Computes recall@k = |returned ids intersect expected ids| / k.

    Returns a dict with aggregate stats (mean, min, max, std, count)
    plus the per-query recall vector under "per_query_recall" so a
    failing run can be triaged.
    """
    n_queries = query_vectors.shape[0]
    per_query_recall = np.zeros(n_queries, dtype=np.float64)
    search_times_us: List[float] = []
    empty_results = 0
    last_warning: Optional[str] = None

    for qi in range(n_queries):
        query_list = _normalize_query_for_search(
            query_vectors[qi], distance_metric,
        )
        try:
            result = client.search.query(collection, query_list, k=k)
        except SwarnDBError as exc:
            logger.error(
                "search.query failed at qi=%d: %s; treating as zero recall",
                qi, exc,
            )
            per_query_recall[qi] = 0.0
            continue

        scored = result.results or []
        if not scored:
            empty_results += 1

        if getattr(result, "warning", "") and result.warning != last_warning:
            last_warning = result.warning
            logger.warning("server warning at qi=%d: %s", qi, result.warning)

        returned_ids = {int(r.id) for r in scored}
        expected_ids = {
            _dataset_position_to_swarndb_id(int(pos), id_offset)
            for pos in ground_truth_top_k[qi].tolist()
        }
        hits = len(returned_ids & expected_ids)
        per_query_recall[qi] = hits / float(k)

        if getattr(result, "search_time_us", 0):
            search_times_us.append(float(result.search_time_us))

        if (qi + 1) % 25 == 0 or qi == n_queries - 1:
            logger.info(
                "progress: %d/%d queries; running mean recall@%d=%.4f",
                qi + 1, n_queries, k,
                float(per_query_recall[: qi + 1].mean()),
            )

    stats: Dict[str, Any] = {
        "count": int(n_queries),
        "k": int(k),
        "mean_recall": float(per_query_recall.mean()) if n_queries else 0.0,
        "min_recall": float(per_query_recall.min()) if n_queries else 0.0,
        "max_recall": float(per_query_recall.max()) if n_queries else 0.0,
        "std_recall": float(per_query_recall.std()) if n_queries else 0.0,
        "empty_results": int(empty_results),
        "per_query_recall": per_query_recall.tolist(),
    }
    if search_times_us:
        stats["mean_search_time_us"] = float(np.mean(search_times_us))
        stats["p50_search_time_us"] = float(np.percentile(search_times_us, 50))
        stats["p95_search_time_us"] = float(np.percentile(search_times_us, 95))
        stats["p99_search_time_us"] = float(np.percentile(search_times_us, 99))
    return stats


# ---------------------------------------------------------------------------
# Data directory plumbing (only used on spawn path)
# ---------------------------------------------------------------------------


def _resolve_data_dir(args: argparse.Namespace) -> Tuple[Path, bool]:
    """Pick the data directory for a spawned server.

    External-mode runs never reach this path. If --data-dir is given,
    the harness uses it and does not delete it on exit. Otherwise a
    tempdir is created and cleaned up.
    """
    if args.data_dir:
        base = Path(args.data_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base, False
    base = Path(tempfile.mkdtemp(prefix="swarndb_p10_8_recall_"))
    return base, True


def _cleanup_data_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        logger.warning("cleanup of %s failed: %s", path, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SwarnDB recall-against-ground-truth harness "
            "(Perf_Stability P10.8). Computes brute-force top-k for "
            "n_queries sampled from the dataset, runs the same queries "
            "through the SDK against a pre-loaded collection, and "
            "asserts mean recall@k is at or above the configured "
            "ceiling."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help=(
            "Path to the numpy .npy dataset of shape (N, dim). The "
            "collection on the server must have been loaded with the "
            "same dataset in row order so the id-offset mapping holds."
        ),
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help=(
            "Collection to query. The harness does NOT create or "
            "insert into the collection; it must already exist."
        ),
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=DEFAULT_N_QUERIES,
        help=f"Number of query vectors to sample (default {DEFAULT_N_QUERIES}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-k neighbors per query (default {DEFAULT_K}).",
    )
    parser.add_argument(
        "--ground-truth-cache",
        default=None,
        help=(
            "Path to the .npz ground-truth cache. Default depends on "
            "--n-queries and --k: /tmp/swarndb_recall_gt_{n}_{k}.npz."
        ),
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=DEFAULT_REST_PORT,
        help=f"REST port for /readyz polling (default {DEFAULT_REST_PORT}).",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help=f"gRPC port for the SDK client (default {DEFAULT_GRPC_PORT}).",
    )
    parser.add_argument(
        "--binary-path",
        default=None,
        help=(
            "Optional path to the swarndb binary. Used only on the "
            "spawn path (when no external server is found on "
            "--rest-port). If unset and no server is up, the harness "
            "exits with code 2."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SWARNDB_HARNESS_DATA_DIR"),
        help=(
            "Optional base data directory used only on the spawn path. "
            "If omitted, a tempdir is created and removed on exit."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for query selection (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["cosine", "l2"],
        default=DEFAULT_DISTANCE_METRIC,
        help=(
            "Distance metric used both for brute-force ground truth "
            f"and for query normalization (default {DEFAULT_DISTANCE_METRIC}). "
            "The server-side collection must have been created with "
            "the matching metric."
        ),
    )
    parser.add_argument(
        "--recall-ceiling",
        type=float,
        default=DEFAULT_RECALL_CEILING,
        help=(
            f"Minimum acceptable mean recall@k (default {DEFAULT_RECALL_CEILING}). "
            "The harness exits 0 iff observed mean recall is at or "
            "above this value."
        ),
    )
    parser.add_argument(
        "--id-offset",
        type=int,
        default=DEFAULT_ID_OFFSET,
        help=(
            f"Offset added to a 0-based dataset position to obtain its "
            f"SwarnDB id (default {DEFAULT_ID_OFFSET}). Set to 0 if the "
            "loader inserted dataset row i as id i; set to 1 if the "
            "loader inserted dataset row i as id i+1 (which is the "
            "default convention used by the perf-stability loaders)."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to dump the per-run stats as JSON.",
    )

    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.n_queries <= 0:
        return f"--n-queries must be positive (got {args.n_queries})"
    if args.k <= 0:
        return f"--k must be positive (got {args.k})"
    if not (0.0 <= args.recall_ceiling <= 1.0):
        return (
            f"--recall-ceiling must be in [0.0, 1.0] "
            f"(got {args.recall_ceiling})"
        )
    if args.id_offset < 0:
        return f"--id-offset must be non-negative (got {args.id_offset})"
    if not Path(args.dataset_path).exists():
        return f"--dataset-path does not exist: {args.dataset_path}"
    return None


def _default_cache_path(n_queries: int, k: int) -> str:
    return f"/tmp/swarndb_recall_gt_{n_queries}_{k}.npz"


def _write_output_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("wrote results to %s", out)


def main() -> int:
    args = parse_args()

    err = _validate_args(args)
    if err is not None:
        logger.error("[FAIL] %s", err)
        return 2

    if args.binary_path:
        bin_p = Path(args.binary_path)
        if not bin_p.exists():
            logger.error("[FAIL] binary not found at %s", bin_p)
            return 2

    cache_path = (
        args.ground_truth_cache
        if args.ground_truth_cache
        else _default_cache_path(args.n_queries, args.k)
    )

    logger.info(
        "P10.8 recall harness: dataset=%s collection=%s n_queries=%d k=%d "
        "metric=%s ceiling=%.3f id_offset=%d cache=%s",
        args.dataset_path, args.collection_name, args.n_queries, args.k,
        args.distance_metric, args.recall_ceiling, args.id_offset, cache_path,
    )

    # Compute or load ground truth FIRST. This step does not need a
    # running server and is the most expensive piece on a cold cache;
    # if it fails the harness exits before touching the server.
    try:
        query_vectors, query_indices, ground_truth_top_k = (
            compute_or_load_ground_truth(
                dataset_path=args.dataset_path,
                n_queries=args.n_queries,
                k=args.k,
                cache_path=cache_path,
                seed=args.seed,
                distance_metric=args.distance_metric,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("[FAIL] ground-truth setup error: %s", exc)
        return 2

    # Server lifecycle.
    data_dir, owns_data_dir = _resolve_data_dir(args)
    log_path = data_dir.parent / "swarndb_p10_8_recall.log"

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc = find_or_spawn_server(args, logger, _build_proc)

    stats: Dict[str, Any] = {}
    overall_ok = False
    try:
        client = make_client(args.grpc_port)
        try:
            # Sanity-check the collection exists before measuring recall.
            if not client.collections.exists(args.collection_name):
                logger.error(
                    "[FAIL] collection '%s' does not exist on the server. "
                    "The harness does not create or insert; load the "
                    "collection first.",
                    args.collection_name,
                )
                return 2

            stats = measure_recall(
                client=client,
                collection=args.collection_name,
                query_vectors=query_vectors,
                ground_truth_top_k=ground_truth_top_k,
                k=args.k,
                distance_metric=args.distance_metric,
                id_offset=args.id_offset,
            )
        finally:
            try:
                client.close()
            except Exception:
                pass
    except SwarnDBError as exc:
        logger.error("[FAIL] SDK error during recall measurement: %s", exc)
        return 2
    finally:
        if proc is not None:
            proc.terminate()
        else:
            logger.info(
                "External server in use; teardown skipped (harness did "
                "not own the process)."
            )
        if owns_data_dir:
            _cleanup_data_dir(data_dir)

    mean_recall = stats.get("mean_recall", 0.0)
    overall_ok = mean_recall >= args.recall_ceiling

    verdict = "PASS" if overall_ok else "FAIL"
    border = "=" * 70
    logger.info("\n%s", border)
    logger.info(
        "RECALL @ k=%d: %s "
        "(mean=%.4f, min=%.4f, max=%.4f, std=%.4f, n=%d, ceiling=%.4f)",
        args.k, verdict,
        stats.get("mean_recall", 0.0),
        stats.get("min_recall", 0.0),
        stats.get("max_recall", 0.0),
        stats.get("std_recall", 0.0),
        stats.get("count", 0),
        args.recall_ceiling,
    )
    if "mean_search_time_us" in stats:
        logger.info(
            "search latency: mean=%.1fus p50=%.1fus p95=%.1fus p99=%.1fus",
            stats.get("mean_search_time_us", 0.0),
            stats.get("p50_search_time_us", 0.0),
            stats.get("p95_search_time_us", 0.0),
            stats.get("p99_search_time_us", 0.0),
        )
    if stats.get("empty_results", 0) > 0:
        logger.warning(
            "empty-result queries: %d / %d",
            stats["empty_results"], stats.get("count", 0),
        )
    logger.info("%s", border)

    payload = {
        "verdict": verdict,
        "config": {
            "dataset_path": args.dataset_path,
            "collection_name": args.collection_name,
            "n_queries": args.n_queries,
            "k": args.k,
            "seed": args.seed,
            "distance_metric": args.distance_metric,
            "recall_ceiling": args.recall_ceiling,
            "id_offset": args.id_offset,
            "ground_truth_cache": cache_path,
            "rest_port": args.rest_port,
            "grpc_port": args.grpc_port,
        },
        "stats": stats,
        "query_indices": query_indices.tolist(),
    }
    _write_output_json(args.output_json, payload)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
