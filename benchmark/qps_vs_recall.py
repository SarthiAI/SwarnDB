#!/usr/bin/env python3
"""
SwarnDB QPS vs Recall sweep benchmark.

Sweeps the HNSW ``ef_search`` parameter against a live SwarnDB
collection, measures QPS and mean recall@k for each setting, and emits
a CSV plus a printed Pareto-frontier summary table.

This is a benchmark, not a correctness test. It does not create or
insert into the collection; the collection must already be loaded with
the same dataset that drives ground truth. The benchmark exits 0 on a
clean run, even if recall is low, unless ``--recall-floor`` is set; in
that case the largest ``ef_search`` sweep point must reach the floor or
the benchmark exits 1.

The brute-force ground-truth routine is shared with the recall test
harness at ``swarndb/tests/test_recall_ground_truth.py`` and imported
via a relative path, so no wheel install is required. See
``_import_ground_truth`` below for the exact import line.

CLI summary
===========

Run against a collection that has already been loaded::

    python3 qps_vs_recall.py \\
        --dataset-path /data/dbpedia_1m_1536.npy \\
        --collection-name dbpedia_1m \\
        --rest-port 8080 --grpc-port 50051 \\
        --n-queries 1000 --k 10 \\
        --iterations 3 \\
        --ef-search-list 25,50,100,200,400,800 \\
        --workers 8 \\
        --distance-metric cosine \\
        --recall-floor 0.95
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests

# Make the in-tree SDK importable so the benchmark runs from a checkout
# without a wheel install. Mirrors the pattern used by every other
# perf-stability harness in this folder.
_BENCHMARK_DIR = Path(__file__).resolve().parent
_SDK_SRC = _BENCHMARK_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

# Relative-path import of the ground-truth routine shared with the
# recall test harness. Keeps the benchmark coupled to the harness
# module without requiring a Python
# package install. The harness lives at swarndb/tests/.
sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parents[1] / "tests"),
)
from test_recall_ground_truth import compute_or_load_ground_truth  # noqa: E402

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("qps_vs_recall_benchmark")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------

# Default port assignments used when the benchmark spawns its own
# swarndb process. Override with --rest-port and --grpc-port for an
# already-running server.
DEFAULT_REST_PORT = 18102
DEFAULT_GRPC_PORT = 18103

# /readyz contract.
READYZ_DEADLINE_SECONDS = 60.0
READYZ_POLL_INTERVAL_SECONDS = 0.5
EXTERNAL_PROBE_TIMEOUT_SECONDS = 2.0

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

DEFAULT_N_QUERIES = 100
DEFAULT_K = 10
DEFAULT_ITERATIONS = 3
DEFAULT_EF_SEARCH_LIST = "10,25,50,100,200,400,800"
DEFAULT_SEED = 42
DEFAULT_DISTANCE_METRIC = "cosine"
DEFAULT_ID_OFFSET = 1

# A high-recall threshold used in the Pareto summary line. Tunable via
# CLI for downstream consumers if needed.
DEFAULT_RECALL_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Process management: optional spawn-and-supervise wrapper around the
# swarndb binary for self-contained benchmark runs. Use the --rest-port
# and --grpc-port flags to target an already-running server instead.
# ---------------------------------------------------------------------------


class SwarndbProcess:
    """Spawn-and-supervise wrapper around the swarndb binary."""

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
    benchmark reuses an external server it returns None for the process
    handle so teardown skips a process it does not own.
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
    """Build an SDK client with a generous timeout suitable for sweeps."""
    return SwarnDBClient(
        host="127.0.0.1",
        port=grpc_port,
        timeout=600.0,
        max_retries=5,
        retry_delay=1.0,
    )


def _normalize_query_for_search(
    query: np.ndarray,
    distance_metric: str,
) -> List[float]:
    """Prepare a query vector for the SDK search call.

    Matches the convention used by ``test_recall_ground_truth.py``:
    cosine queries are L2-normalized client-side so the recall math is
    symmetric with the brute-force side; L2 queries are passed through.
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


# ---------------------------------------------------------------------------
# Sweep core
# ---------------------------------------------------------------------------


def _parse_ef_search_list(raw: str) -> List[int]:
    """Parse the --ef-search-list CLI string into ints."""
    try:
        values = [int(s) for s in raw.split(",") if s.strip()]
    except ValueError as exc:
        raise ValueError(
            f"--ef-search-list must be comma-separated ints, got {raw!r}: {exc}"
        ) from exc
    if not values:
        raise ValueError("--ef-search-list must contain at least one value")
    if any(v <= 0 for v in values):
        raise ValueError(
            f"--ef-search-list values must be positive, got {values}"
        )
    return values


def _run_one_pass(
    client: SwarnDBClient,
    collection: str,
    query_lists: List[List[float]],
    k: int,
    ef_search: int,
    record_results: bool,
    workers: int = 1,
) -> Tuple[float, List[float], List[List[int]]]:
    """Run all queries once for a given ef_search value.

    Returns (elapsed_seconds, per_query_latency_seconds, per_query_top_ids).
    ``per_query_top_ids`` is empty when ``record_results`` is False to save
    memory and Python work in latency-only passes.

    When ``workers == 1`` (default), runs queries serially. When
    ``workers > 1``, dispatches queries across a thread pool; wall-clock
    QPS is measured from first submit to last result, and per-query
    latency is still captured around each individual search.query call.
    """
    per_query_latency: List[float] = []
    per_query_top_ids: List[List[int]] = []

    if workers <= 1:
        pass_start = time.perf_counter()
        for q in query_lists:
            q_start = time.perf_counter_ns()
            try:
                result = client.search.query(
                    collection, q, k=k, ef_search=ef_search,
                )
            except SwarnDBError as exc:
                logger.error(
                    "search.query failed at ef_search=%d: %s; "
                    "recording empty result for this query",
                    ef_search, exc,
                )
                q_end = time.perf_counter_ns()
                per_query_latency.append((q_end - q_start) / 1e9)
                if record_results:
                    per_query_top_ids.append([])
                continue
            q_end = time.perf_counter_ns()
            per_query_latency.append((q_end - q_start) / 1e9)
            if record_results:
                scored = result.results or []
                per_query_top_ids.append([int(r.id) for r in scored])
        pass_elapsed = time.perf_counter() - pass_start

        return pass_elapsed, per_query_latency, per_query_top_ids

    # Concurrent path: workers > 1. Capture ef_search and the query
    # vector by value via default arguments so each task carries its
    # own snapshot rather than sharing the loop variable by reference.
    n = len(query_lists)
    latencies_arr: List[float] = [0.0] * n
    top_ids_arr: List[List[int]] = [[] for _ in range(n)] if record_results else []

    def _do_query(
        idx: int,
        q_vec: List[float] = None,
        ef: int = ef_search,
    ) -> Tuple[int, float, List[int]]:
        q_start = time.perf_counter_ns()
        try:
            result = client.search.query(
                collection, q_vec, k=k, ef_search=ef,
            )
        except SwarnDBError as exc:
            logger.error(
                "search.query failed at ef_search=%d: %s; "
                "recording empty result for this query",
                ef, exc,
            )
            q_end = time.perf_counter_ns()
            return idx, (q_end - q_start) / 1e9, []
        q_end = time.perf_counter_ns()
        if record_results:
            scored = result.results or []
            ids = [int(r.id) for r in scored]
        else:
            ids = []
        return idx, (q_end - q_start) / 1e9, ids

    pass_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_do_query, i, query_lists[i]) for i in range(n)
        ]
        for fut in futures:
            idx, lat, ids = fut.result()
            latencies_arr[idx] = lat
            if record_results:
                top_ids_arr[idx] = ids
    pass_elapsed = time.perf_counter() - pass_start

    per_query_latency = latencies_arr
    per_query_top_ids = top_ids_arr if record_results else []

    return pass_elapsed, per_query_latency, per_query_top_ids


def _compute_recall_for_pass(
    per_query_top_ids: List[List[int]],
    ground_truth_top_k: np.ndarray,
    k: int,
    id_offset: int,
) -> Tuple[float, float]:
    """Compute (mean_recall, p10_recall) over a recorded pass."""
    n = len(per_query_top_ids)
    if n == 0:
        return 0.0, 0.0
    per_query_recall = np.zeros(n, dtype=np.float64)
    for qi, returned_ids in enumerate(per_query_top_ids):
        expected_ids = {
            _dataset_position_to_swarndb_id(int(pos), id_offset)
            for pos in ground_truth_top_k[qi].tolist()
        }
        returned_set = set(returned_ids)
        hits = len(returned_set & expected_ids)
        per_query_recall[qi] = hits / float(k)
    mean_recall = float(per_query_recall.mean())
    p10_recall = float(np.percentile(per_query_recall, 10))
    return mean_recall, p10_recall


def sweep_ef_search(
    client: SwarnDBClient,
    collection: str,
    query_vectors: np.ndarray,
    ground_truth_top_k: np.ndarray,
    k: int,
    iterations: int,
    ef_search_list: List[int],
    distance_metric: str,
    id_offset: int,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """Run the full ef_search sweep.

    For each ``ef_search`` value the sweep runs ``iterations`` complete
    passes of all queries to average QPS. Recall is measured on the
    FIRST pass only (per spec): the per-query top-k ids are recorded
    once and compared against the cached ground truth.
    """
    n_queries = query_vectors.shape[0]
    query_lists = [
        _normalize_query_for_search(query_vectors[qi], distance_metric)
        for qi in range(n_queries)
    ]

    rows: List[Dict[str, Any]] = []
    for ef in ef_search_list:
        logger.info(
            "sweep ef_search=%d: %d iterations of %d queries (recall on pass 1)",
            ef, iterations, n_queries,
        )
        pass_times: List[float] = []
        all_latencies: List[float] = []
        first_pass_top_ids: List[List[int]] = []
        for it in range(iterations):
            elapsed, latencies, top_ids = _run_one_pass(
                client=client,
                collection=collection,
                query_lists=query_lists,
                k=k,
                ef_search=ef,
                record_results=(it == 0),
                workers=workers,
            )
            pass_times.append(elapsed)
            all_latencies.extend(latencies)
            if it == 0:
                first_pass_top_ids = top_ids
            logger.info(
                "  ef_search=%d pass %d/%d: %.3fs (%.1f q/s)",
                ef, it + 1, iterations, elapsed,
                (n_queries / elapsed) if elapsed > 0 else 0.0,
            )

        # QPS aggregate: total queries across all passes over total time.
        total_time = sum(pass_times)
        total_queries = n_queries * iterations
        qps_overall = (total_queries / total_time) if total_time > 0 else 0.0

        # Per-pass QPS distribution for stddev.
        per_pass_qps = np.array(
            [(n_queries / t) if t > 0 else 0.0 for t in pass_times],
            dtype=np.float64,
        )
        qps_stddev = float(per_pass_qps.std(ddof=0)) if iterations > 0 else 0.0

        # Recall on the first pass only.
        recall_mean, recall_p10 = _compute_recall_for_pass(
            first_pass_top_ids, ground_truth_top_k, k, id_offset,
        )

        # Latency percentiles across every query in every pass.
        if all_latencies:
            lat_array_ms = np.array(all_latencies, dtype=np.float64) * 1000.0
            lat_p50 = float(np.percentile(lat_array_ms, 50))
            lat_p95 = float(np.percentile(lat_array_ms, 95))
            lat_p99 = float(np.percentile(lat_array_ms, 99))
        else:
            lat_p50 = lat_p95 = lat_p99 = 0.0

        row = {
            "ef_search": ef,
            "qps_mean": qps_overall,
            "qps_stddev": qps_stddev,
            "recall_mean": recall_mean,
            "recall_p10": recall_p10,
            "latency_p50_ms": lat_p50,
            "latency_p95_ms": lat_p95,
            "latency_p99_ms": lat_p99,
        }
        logger.info(
            "  ef_search=%d summary: qps=%.1f recall@%d=%.4f p99=%.2fms",
            ef, qps_overall, k, recall_mean, lat_p99,
        )
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


CSV_COLUMNS = [
    "ef_search",
    "qps_mean",
    "qps_stddev",
    "recall_mean",
    "recall_p10",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "n_queries",
    "k",
    "iterations",
    "timestamp",
    "workers",
]


def _write_csv(
    output_csv: Path,
    rows: List[Dict[str, Any]],
    n_queries: int,
    k: int,
    iterations: int,
    timestamp_iso: str,
    notes_header_lines: Optional[List[str]] = None,
    workers: int = 1,
) -> None:
    """Write the sweep CSV with optional comment header lines.

    Comment lines start with ``#`` and document any operational notes
    such as SDK gap fallbacks. The CSV body itself remains parseable by
    standard tools when those header lines are filtered.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        if notes_header_lines:
            for line in notes_header_lines:
                f.write(f"# {line}\n")
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "ef_search": r["ef_search"],
                "qps_mean": f"{r['qps_mean']:.4f}",
                "qps_stddev": f"{r['qps_stddev']:.4f}",
                "recall_mean": f"{r['recall_mean']:.6f}",
                "recall_p10": f"{r['recall_p10']:.6f}",
                "latency_p50_ms": f"{r['latency_p50_ms']:.4f}",
                "latency_p95_ms": f"{r['latency_p95_ms']:.4f}",
                "latency_p99_ms": f"{r['latency_p99_ms']:.4f}",
                "n_queries": n_queries,
                "k": k,
                "iterations": iterations,
                "timestamp": timestamp_iso,
                "workers": workers,
            })
    logger.info("wrote CSV to %s", output_csv)


# ---------------------------------------------------------------------------
# Pareto-frontier summary
# ---------------------------------------------------------------------------


def _pareto_frontier_indices(rows: List[Dict[str, Any]]) -> List[int]:
    """Return indices of rows on the Pareto frontier of (qps, recall).

    A row is dominated when another row has both higher (or equal) QPS
    and higher (or equal) recall, with at least one strictly higher.
    The remaining rows form the frontier.
    """
    frontier: List[int] = []
    for i, ri in enumerate(rows):
        dominated = False
        for j, rj in enumerate(rows):
            if i == j:
                continue
            if (
                rj["qps_mean"] >= ri["qps_mean"]
                and rj["recall_mean"] >= ri["recall_mean"]
                and (
                    rj["qps_mean"] > ri["qps_mean"]
                    or rj["recall_mean"] > ri["recall_mean"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def _print_summary_table(
    rows: List[Dict[str, Any]],
    k: int,
    recall_threshold: float,
    workers: int = 1,
) -> None:
    """Print a plain-text Pareto table plus a one-line summary."""
    if not rows:
        print("(no sweep rows; nothing to summarize)")
        return

    frontier_set = set(_pareto_frontier_indices(rows))

    print()
    print(f"QPS-vs-Recall sweep (workers={workers})")
    header = (
        f"ef_search | QPS       | recall@{k:<2} | "
        f"p50 latency | p95 latency | p99 latency | pareto"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for i, r in enumerate(rows):
        mark = "*" if i in frontier_set else " "
        print(
            f"{r['ef_search']:<9} | "
            f"{r['qps_mean']:>9.1f} | "
            f"{r['recall_mean']:>8.4f} | "
            f"{r['latency_p50_ms']:>8.2f} ms | "
            f"{r['latency_p95_ms']:>8.2f} ms | "
            f"{r['latency_p99_ms']:>8.2f} ms | "
            f"  {mark}"
        )
    print(sep)
    print("(* indicates a point on the Pareto frontier of QPS vs recall)")

    # Best QPS at recall >= threshold.
    eligible = [r for r in rows if r["recall_mean"] >= recall_threshold]
    if eligible:
        best = max(eligible, key=lambda r: r["qps_mean"])
        print(
            f"Pareto summary: best QPS at recall@{k} >= {recall_threshold:.2f} "
            f"is {best['qps_mean']:.1f} q/s at ef_search={best['ef_search']} "
            f"(recall={best['recall_mean']:.4f}, p99={best['latency_p99_ms']:.2f}ms)."
        )
    else:
        max_recall_row = max(rows, key=lambda r: r["recall_mean"])
        print(
            f"Pareto summary: no sweep point reached recall@{k} >= "
            f"{recall_threshold:.2f}; highest observed recall was "
            f"{max_recall_row['recall_mean']:.4f} at "
            f"ef_search={max_recall_row['ef_search']} "
            f"(qps={max_recall_row['qps_mean']:.1f})."
        )


# ---------------------------------------------------------------------------
# Data directory plumbing (only used on spawn path)
# ---------------------------------------------------------------------------


def _resolve_data_dir(args: argparse.Namespace) -> Tuple[Path, bool]:
    """Pick the data directory for a spawned server.

    External-mode runs never reach this path. If --data-dir is given,
    the benchmark uses it and does not delete it on exit. Otherwise a
    tempdir is created and cleaned up.
    """
    if args.data_dir:
        base = Path(args.data_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base, False
    base = Path(tempfile.mkdtemp(prefix="swarndb_p10_8_qps_recall_"))
    return base, True


def _cleanup_data_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        logger.warning("cleanup of %s failed: %s", path, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_output_csv() -> str:
    """Default CSV path: ``benchmark/results/qps_vs_recall_<timestamp>.csv``.

    The directory is created if it does not exist when the file is
    written. Override with ``--output-csv``.
    """
    results_dir = Path(__file__).resolve().parent / "results"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(results_dir / f"qps_vs_recall_{ts}.csv")


def _default_ground_truth_cache(n_queries: int, k: int) -> str:
    return f"/tmp/swarndb_recall_gt_{n_queries}_{k}.npz"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SwarnDB QPS vs Recall sweep benchmark. Sweeps ef_search "
            "against a live collection and emits a CSV plus a "
            "Pareto-frontier summary."
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
        help="Collection to query. Must already exist on the server.",
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
            "Optional path to the swarndb binary. Used only when no "
            "external server is found on --rest-port."
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
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=(
            f"Number of full sweep passes per ef_search to average QPS "
            f"(default {DEFAULT_ITERATIONS})."
        ),
    )
    parser.add_argument(
        "--ef-search-list",
        default=DEFAULT_EF_SEARCH_LIST,
        help=(
            "Comma-separated ef_search values to sweep "
            f"(default {DEFAULT_EF_SEARCH_LIST})."
        ),
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
            "Distance metric used for ground truth and query "
            f"normalization (default {DEFAULT_DISTANCE_METRIC}). The "
            "server-side collection must have been created with the "
            "matching metric."
        ),
    )
    parser.add_argument(
        "--id-offset",
        type=int,
        default=DEFAULT_ID_OFFSET,
        help=(
            f"Offset added to a 0-based dataset position to obtain its "
            f"SwarnDB id (default {DEFAULT_ID_OFFSET})."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Path to write the sweep CSV. Default is "
            "benchmark/results/qps_vs_recall_{timestamp}.csv "
            "(created if needed)."
        ),
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=None,
        help=(
            "Optional floor for mean recall@k at the LARGEST ef_search. "
            "If set, the benchmark exits 1 when the observed mean "
            "recall at the largest ef_search is below this value. "
            "When unset, the benchmark is report-only and always "
            "exits 0 on a clean run."
        ),
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=DEFAULT_RECALL_THRESHOLD,
        help=(
            "Threshold used by the Pareto summary line for the "
            "best-QPS-at-recall callout "
            f"(default {DEFAULT_RECALL_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent searcher threads (default 1 for the "
            "published per-thread numbers; pass 4-8 to reproduce the "
            "README's concurrent QPS table)."
        ),
    )

    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.n_queries <= 0:
        return f"--n-queries must be positive (got {args.n_queries})"
    if args.k <= 0:
        return f"--k must be positive (got {args.k})"
    if args.iterations <= 0:
        return f"--iterations must be positive (got {args.iterations})"
    if args.workers <= 0:
        return f"--workers must be positive (got {args.workers})"
    if args.id_offset < 0:
        return f"--id-offset must be non-negative (got {args.id_offset})"
    if args.recall_floor is not None and not (
        0.0 <= args.recall_floor <= 1.0
    ):
        return (
            f"--recall-floor must be in [0.0, 1.0] "
            f"(got {args.recall_floor})"
        )
    if not (0.0 <= args.recall_threshold <= 1.0):
        return (
            f"--recall-threshold must be in [0.0, 1.0] "
            f"(got {args.recall_threshold})"
        )
    if not Path(args.dataset_path).exists():
        return f"--dataset-path does not exist: {args.dataset_path}"
    try:
        _parse_ef_search_list(args.ef_search_list)
    except ValueError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

    ef_search_list = _parse_ef_search_list(args.ef_search_list)
    cache_path = (
        args.ground_truth_cache
        if args.ground_truth_cache
        else _default_ground_truth_cache(args.n_queries, args.k)
    )
    output_csv_path = Path(
        args.output_csv if args.output_csv else _default_output_csv()
    )
    timestamp_iso = datetime.now(timezone.utc).isoformat()

    logger.info(
        "QPS vs Recall benchmark: dataset=%s collection=%s "
        "n_queries=%d k=%d iterations=%d ef_search_list=%s metric=%s "
        "id_offset=%d cache=%s output_csv=%s",
        args.dataset_path, args.collection_name, args.n_queries, args.k,
        args.iterations, ef_search_list, args.distance_metric,
        args.id_offset, cache_path, output_csv_path,
    )

    # Compute or load ground truth FIRST. This is the slow one-time
    # step on a cold cache; if it fails the benchmark exits before
    # touching the server.
    try:
        query_vectors, _query_indices, ground_truth_top_k = (
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
    log_path = data_dir.parent / "swarndb_p10_8_qps_recall.log"

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc = find_or_spawn_server(args, logger, _build_proc)

    rows: List[Dict[str, Any]] = []
    try:
        client = make_client(args.grpc_port)
        try:
            if not client.collections.exists(args.collection_name):
                logger.error(
                    "[FAIL] collection '%s' does not exist on the server. "
                    "Load the collection first; the benchmark does not "
                    "create or insert.",
                    args.collection_name,
                )
                return 2

            rows = sweep_ef_search(
                client=client,
                collection=args.collection_name,
                query_vectors=query_vectors,
                ground_truth_top_k=ground_truth_top_k,
                k=args.k,
                iterations=args.iterations,
                ef_search_list=ef_search_list,
                distance_metric=args.distance_metric,
                id_offset=args.id_offset,
                workers=args.workers,
            )
        finally:
            try:
                client.close()
            except Exception:
                pass
    except SwarnDBError as exc:
        logger.error("[FAIL] SDK error during sweep: %s", exc)
        return 2
    finally:
        if proc is not None:
            proc.terminate()
        else:
            logger.info(
                "External server in use; teardown skipped (benchmark did "
                "not own the process)."
            )
        if owns_data_dir:
            _cleanup_data_dir(data_dir)

    # Write the CSV with operational notes captured in the header.
    notes_header_lines = [
        f"benchmark=qps_vs_recall.py timestamp={timestamp_iso}",
        f"collection={args.collection_name} dataset={args.dataset_path}",
        (
            f"n_queries={args.n_queries} k={args.k} "
            f"iterations={args.iterations} metric={args.distance_metric} "
            f"id_offset={args.id_offset}"
        ),
        f"ef_search_list={ef_search_list}",
        "ef_search_exposure=sdk_kwarg (search.query supports ef_search=int)",
    ]
    _write_csv(
        output_csv=output_csv_path,
        rows=rows,
        n_queries=args.n_queries,
        k=args.k,
        iterations=args.iterations,
        timestamp_iso=timestamp_iso,
        notes_header_lines=notes_header_lines,
        workers=args.workers,
    )

    _print_summary_table(rows, args.k, args.recall_threshold, args.workers)

    # Recall-floor gate (optional). Applied to the LARGEST ef_search
    # value, which is the highest-recall point in the sweep.
    if args.recall_floor is not None and rows:
        largest_ef = max(ef_search_list)
        anchor_row = next(
            (r for r in rows if r["ef_search"] == largest_ef),
            rows[-1],
        )
        anchor_recall = anchor_row["recall_mean"]
        if anchor_recall < args.recall_floor:
            logger.error(
                "[FAIL] recall floor not met: at ef_search=%d "
                "mean recall@%d=%.4f < floor %.4f",
                anchor_row["ef_search"], args.k, anchor_recall,
                args.recall_floor,
            )
            return 1
        logger.info(
            "[PASS] recall floor met: at ef_search=%d "
            "mean recall@%d=%.4f >= floor %.4f",
            anchor_row["ef_search"], args.k, anchor_recall,
            args.recall_floor,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
