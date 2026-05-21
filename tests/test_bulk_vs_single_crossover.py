#!/usr/bin/env python3
"""
SwarnDB bulk-vs-single crossover harness (Perf_Stability P10 Investigation B).

P10 Investigation B observed that the bulk_insert path was about 27%
slower than the single-insert path at dim=128 with chunks of 100,
which inverted the expected throughput relationship. The P10 fixes
attack the per-row hidden overhead on the bulk path:

    1. The vector signature change to share Arc<Vec<f32>> end to end
       so the per-row clone goes away.
    2. Elimination of the double clone in the bulk dispatch loop
       (one for the per-row task spawn, one for the WAL write path).
    3. Merging the per-row metadata iteration and the per-row vector
       iteration into a single pass so the bulk path stops paying
       2x iteration cost.

This harness measures the crossover batch size after those fixes
land: the smallest batch size at which the bulk path matches or
beats the single-insert baseline within a 5% tolerance. Investigation
B's target is crossover at batch_size <= 100 for the 1536-dim path
post-fix; the 128-dim path is expected to cross over later (1000 or
beyond) because per-row work is a larger share of the total at small
dim.

Modes:

    single      Insert n_vectors one at a time per dim; record QPS.
    bulk        Insert n_vectors in chunks of each batch_size per dim;
                record QPS per (dim, batch_size).
    crossover   Run single + bulk and report the smallest batch_size
                where bulk_qps >= single_qps * 0.95 per dim.
    all         crossover plus per-test detail in the output.

Usage:

    python test_bulk_vs_single_crossover.py \\
        --binary-path /usr/local/bin/swarndb \\
        --n-vectors 10000 \\
        --dimensions 128,1536 \\
        --batch-sizes 1,10,100,1000,5000 \\
        --mode all \\
        --rest-port 18092 --grpc-port 18093 \\
        --output-json /tmp/p10_crossover.json

The P11 regression gate invokes this harness on the Civo binary with
the default n_vectors=10000 and the default dim and batch sweep.
Local P10 Step 3 does NOT execute this harness; it ships ready for
P11. The 10000 default tracks the perf_benchmark signal that surfaced
the Investigation B finding; P11 may scale up if the signal needs
more resolution.

Exit codes:

    0   At least one dim reached crossover (or mode is single or
        bulk, both of which always pass).
    1   No dim reached crossover in crossover or all modes; the
        bulk path is still slower than single across the sweep.
    2   Setup error (binary missing, /readyz timeout, SDK error).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# Make the in-tree SDK importable so the harness runs from a checkout
# without a wheel install. Mirrors the pattern used by the P07, P08,
# and P09 harnesses.
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("bulk_vs_single_crossover_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and tunables
# ---------------------------------------------------------------------------

# /readyz contract: generous to cover cold start on a fresh data dir.
READYZ_DEADLINE_SECONDS = 60.0
READYZ_POLL_INTERVAL_SECONDS = 0.5

# Ports kept distinct from the P07, P08, and P09 harnesses so the
# four can coexist on the same dev box.
DEFAULT_REST_PORT = 18092
DEFAULT_GRPC_PORT = 18093

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

# Sweep defaults.
DEFAULT_BATCH_SIZES: List[int] = [1, 10, 100, 1000, 5000]
DEFAULT_DIMENSIONS: List[int] = [128, 1536]
DEFAULT_N_VECTORS = 10_000
DEFAULT_RPS_WARMUP_ROWS = 200
DEFAULT_SEED = 42

# Crossover tolerance: bulk wins if its QPS is at least this fraction
# of the single-insert baseline.
CROSSOVER_TOLERANCE = 0.95

# Collection plumbing.
DEFAULT_COLLECTION_PREFIX = "p10_crossover"
DEFAULT_DISTANCE_METRIC = "cosine"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class AssertionRecord:
    name: str
    passed: bool
    detail: str

    def render(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] {self.name}: {self.detail}"


@dataclass
class HarnessReport:
    mode: str
    records: List[AssertionRecord] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        rec = AssertionRecord(name=name, passed=passed, detail=detail)
        self.records.append(rec)
        logger.info(rec.render())

    def all_passed(self) -> bool:
        return all(r.passed for r in self.records)

    def summary(self) -> str:
        total = len(self.records)
        passed = sum(1 for r in self.records if r.passed)
        verdict = "PASS" if self.all_passed() else "FAIL"
        return (
            f"\n{'=' * 70}\n"
            f"MODE '{self.mode}': {verdict} ({passed}/{total} assertions passed)\n"
            f"{'=' * 70}"
        )


@dataclass
class SingleInsertResult:
    dim: int
    n_vectors: int
    elapsed_seconds: float
    qps: float


@dataclass
class BulkInsertResultRow:
    dim: int
    batch_size: int
    n_vectors: int
    n_chunks: int
    elapsed_seconds: float
    qps: float
    ratio_to_single: Optional[float] = None


@dataclass
class CrossoverResult:
    dim: int
    single_qps: float
    crossover_batch_size: Optional[int]
    crossover_qps: Optional[float]
    crossover_ratio: Optional[float]
    tested_batch_sizes: List[int]


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


class SwarndbProcess:
    """Spawn-and-supervise wrapper around the swarndb binary.

    Mirrors the SwarndbProcess class used by the P07, P08, and P09
    harnesses. Kept local so each harness owns its lifecycle without
    a shared utility module.
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

    def pid(self) -> int:
        if self.proc is None:
            raise RuntimeError("swarndb process not started")
        return self.proc.pid

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def terminate(self) -> None:
        """Graceful stop with SIGTERM fallback to SIGKILL."""
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


# ---------------------------------------------------------------------------
# Auto-detect existing server (P10.6)
# ---------------------------------------------------------------------------


EXTERNAL_PROBE_TIMEOUT_SECONDS = 2.0


def find_or_spawn_server(
    args: argparse.Namespace,
    log: logging.Logger,
    proc_factory,
) -> Tuple[str, str, Optional[SwarndbProcess]]:
    """Auto-detect a swarndb-server on args.rest_port; else spawn one.

    Returns (rest_url, grpc_url, spawned_process_or_None).

    Auto-detect path: if GET /readyz returns 200 on args.rest_port, the
    harness reuses that server and returns spawned_process_or_None=None
    so teardown skips a process the harness does not own.

    Spawn path: if /readyz is unreachable AND --binary-path is provided,
    the harness calls proc_factory() to build a SwarndbProcess with the
    mode's resolved data_dir, starts it, waits for /readyz, and returns
    the wrapper. If --binary-path is missing AND no external server is
    up, the harness emits a clear error and exits non-zero.
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
# /readyz polling
# ---------------------------------------------------------------------------


def wait_for_readyz(
    rest_port: int,
    deadline_seconds: float,
    started_at: float,
) -> Tuple[bool, float]:
    """Poll /readyz until 200 OK or the deadline elapses.

    Returns (ok, elapsed_from_started_at). Mirrors the helper used by
    the P07, P08, and P09 harnesses.
    """
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


# ---------------------------------------------------------------------------
# SDK helpers
# ---------------------------------------------------------------------------


def make_client(grpc_port: int) -> SwarnDBClient:
    return SwarnDBClient(
        host="127.0.0.1",
        port=grpc_port,
        timeout=600.0,
        max_retries=5,
        retry_delay=1.0,
    )


def collection_name(
    prefix: str,
    dim: int,
    batch_size: Optional[int],
    mode_label: str,
    run_stamp: int,
) -> str:
    """Build a unique collection name per (dim, batch, mode) run.

    The run_stamp keeps repeated invocations on the same data dir from
    clashing on a stale collection name.
    """
    if batch_size is None:
        return f"{prefix}_{dim}_single_{mode_label}_{run_stamp}"
    return f"{prefix}_{dim}_b{batch_size}_{mode_label}_{run_stamp}"


def ensure_collection(client: SwarnDBClient, name: str, dimension: int) -> None:
    """Drop-and-create the collection so each test starts clean."""
    if client.collections.exists(name):
        client.collections.delete(name)
    client.collections.create(
        name,
        dimension=dimension,
        distance_metric=DEFAULT_DISTANCE_METRIC,
    )
    logger.info("created collection '%s' dim=%d", name, dimension)


def drop_collection_quiet(client: SwarnDBClient, name: str) -> None:
    """Best-effort drop. Swallows errors so a test failure does not
    leak a half-cleaned collection into the next test.
    """
    try:
        if client.collections.exists(name):
            client.collections.delete(name)
    except Exception as exc:
        logger.warning("cleanup of collection '%s' failed: %s", name, exc)


# ---------------------------------------------------------------------------
# Synthetic vectors
# ---------------------------------------------------------------------------


def generate_vectors(count: int, dimension: int, seed: int) -> List[List[float]]:
    """Deterministic synthetic vectors. The harness measures
    throughput, not recall, so randn / float32 is sufficient. Returned
    as list-of-list so the SDK can stream it directly.
    """
    rng = np.random.RandomState(seed)
    arr = rng.randn(count, dimension).astype(np.float32)
    return arr.tolist()


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def warm_up(
    client: SwarnDBClient,
    collection: str,
    dimension: int,
    rows: int,
    seed: int,
) -> None:
    """Insert `rows` rows in a single bulk call so the index is past
    its zero-vector cold state when the measurement starts.

    Required because the smallest batch sizes (1 and 10) finish in a
    handful of seconds, which would otherwise be dominated by index
    init costs rather than per-insert cost.
    """
    if rows <= 0:
        return
    vectors = generate_vectors(rows, dimension, seed=seed)
    metadata = [{"row_idx": -1 - i} for i in range(rows)]
    logger.info(
        "warm-up: bulk_insert %d rows into '%s' before measurement",
        rows, collection,
    )
    result = client.vectors.bulk_insert(
        collection,
        vectors,
        metadata_list=metadata,
        batch_size=min(rows, 1000),
    )
    if result.inserted_count != rows or result.errors:
        raise RuntimeError(
            f"warm-up short: inserted={result.inserted_count}/{rows}, "
            f"errors={result.errors[:3]}"
        )


# ---------------------------------------------------------------------------
# Benchmark primitives
# ---------------------------------------------------------------------------


def bench_single(
    client: SwarnDBClient,
    collection: str,
    vectors: List[List[float]],
    id_offset: int,
) -> Tuple[float, float]:
    """Insert vectors one at a time via the single-insert SDK call.

    Returns (elapsed_seconds, qps).
    """
    n = len(vectors)
    if n == 0:
        return 0.0, 0.0
    start = time.perf_counter()
    for i, vec in enumerate(vectors):
        client.vectors.insert(
            collection,
            vec,
            metadata={"row_idx": id_offset + i},
        )
    elapsed = time.perf_counter() - start
    qps = n / elapsed if elapsed > 0 else 0.0
    return elapsed, qps


def bench_bulk(
    client: SwarnDBClient,
    collection: str,
    vectors: List[List[float]],
    batch_size: int,
    id_offset: int,
) -> Tuple[float, float, int]:
    """Insert vectors in chunks of `batch_size` via bulk_insert.

    The harness drives the chunking itself (one bulk_insert call per
    chunk) so the chunk count matches the spec exactly. A single
    bulk_insert with internal batching would conflate two different
    knobs.

    Returns (elapsed_seconds, qps, n_chunks).
    """
    n = len(vectors)
    if n == 0:
        return 0.0, 0.0, 0
    n_chunks = math.ceil(n / batch_size)
    start = time.perf_counter()
    inserted_total = 0
    for chunk_idx in range(n_chunks):
        lo = chunk_idx * batch_size
        hi = min(lo + batch_size, n)
        chunk = vectors[lo:hi]
        chunk_meta = [{"row_idx": id_offset + lo + i} for i in range(len(chunk))]
        result = client.vectors.bulk_insert(
            collection,
            chunk,
            metadata_list=chunk_meta,
            batch_size=batch_size,
        )
        if result.inserted_count != len(chunk) or result.errors:
            raise RuntimeError(
                f"bulk chunk {chunk_idx + 1}/{n_chunks} short: "
                f"inserted={result.inserted_count}/{len(chunk)}, "
                f"errors={result.errors[:3]}"
            )
        inserted_total += result.inserted_count
    elapsed = time.perf_counter() - start
    qps = inserted_total / elapsed if elapsed > 0 else 0.0
    return elapsed, qps, n_chunks


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def run_single_for_dim(
    client: SwarnDBClient,
    dim: int,
    n_vectors: int,
    warmup_rows: int,
    seed: int,
    prefix: str,
    run_stamp: int,
) -> SingleInsertResult:
    """Run the single-insert baseline for one dim and return the
    measured QPS. Fresh collection per call.
    """
    name = collection_name(prefix, dim, None, "single", run_stamp)
    ensure_collection(client, name, dim)
    try:
        warm_up(client, name, dim, warmup_rows, seed=seed)
        logger.info(
            "[BENCH dim=%d mode=single] inserting %d rows one-at-a-time",
            dim, n_vectors,
        )
        vectors = generate_vectors(n_vectors, dim, seed=seed + 1)
        elapsed, qps = bench_single(client, name, vectors, id_offset=0)
        logger.info(
            "[RESULT dim=%d mode=single] %d rows in %.2fs, %.1f vec/s",
            dim, n_vectors, elapsed, qps,
        )
        return SingleInsertResult(
            dim=dim,
            n_vectors=n_vectors,
            elapsed_seconds=elapsed,
            qps=qps,
        )
    finally:
        drop_collection_quiet(client, name)


def run_bulk_for_dim_batch(
    client: SwarnDBClient,
    dim: int,
    batch_size: int,
    n_vectors: int,
    warmup_rows: int,
    seed: int,
    prefix: str,
    run_stamp: int,
    single_qps: Optional[float],
) -> BulkInsertResultRow:
    """Run the bulk-insert sweep for one (dim, batch_size) pair and
    return the measured QPS plus the ratio to the single baseline if
    one was provided.
    """
    name = collection_name(prefix, dim, batch_size, "bulk", run_stamp)
    ensure_collection(client, name, dim)
    try:
        warm_up(client, name, dim, warmup_rows, seed=seed)
        n_chunks = math.ceil(n_vectors / batch_size)
        logger.info(
            "[BENCH dim=%d batch=%d mode=bulk] inserting %d rows in %d chunks of %d",
            dim, batch_size, n_vectors, n_chunks, batch_size,
        )
        vectors = generate_vectors(n_vectors, dim, seed=seed + 1)
        elapsed, qps, observed_chunks = bench_bulk(
            client, name, vectors, batch_size, id_offset=0,
        )
        ratio = (qps / single_qps) if (single_qps and single_qps > 0) else None
        if ratio is not None:
            logger.info(
                "[RESULT dim=%d batch=%d] %d rows in %.2fs, %.1f vec/s, ratio=%.2f",
                dim, batch_size, n_vectors, elapsed, qps, ratio,
            )
        else:
            logger.info(
                "[RESULT dim=%d batch=%d] %d rows in %.2fs, %.1f vec/s",
                dim, batch_size, n_vectors, elapsed, qps,
            )
        return BulkInsertResultRow(
            dim=dim,
            batch_size=batch_size,
            n_vectors=n_vectors,
            n_chunks=observed_chunks,
            elapsed_seconds=elapsed,
            qps=qps,
            ratio_to_single=ratio,
        )
    finally:
        drop_collection_quiet(client, name)


def analyze_crossover(
    dim: int,
    single_qps: float,
    bulk_rows: List[BulkInsertResultRow],
    tested_batch_sizes: List[int],
) -> CrossoverResult:
    """Identify the smallest batch_size where bulk QPS reaches
    single_qps * CROSSOVER_TOLERANCE for this dim.
    """
    rows_sorted = sorted(bulk_rows, key=lambda r: r.batch_size)
    threshold = single_qps * CROSSOVER_TOLERANCE if single_qps > 0 else float("inf")
    for row in rows_sorted:
        if row.qps >= threshold:
            return CrossoverResult(
                dim=dim,
                single_qps=single_qps,
                crossover_batch_size=row.batch_size,
                crossover_qps=row.qps,
                crossover_ratio=(row.qps / single_qps) if single_qps > 0 else None,
                tested_batch_sizes=tested_batch_sizes,
            )
    return CrossoverResult(
        dim=dim,
        single_qps=single_qps,
        crossover_batch_size=None,
        crossover_qps=None,
        crossover_ratio=None,
        tested_batch_sizes=tested_batch_sizes,
    )


# ---------------------------------------------------------------------------
# Top-level mode dispatch
# ---------------------------------------------------------------------------


def _run_with_server(
    args: argparse.Namespace,
    work,
) -> Tuple[bool, Dict]:
    """Spin up the SwarnDB process, hand a connected client to `work`,
    and tear everything down on exit.

    `work` is a callable taking (client, args) and returning
    (ok: bool, payload: dict). The payload is propagated up so the
    caller can dump it to JSON.
    """
    data_dir, owns_data_dir = _resolve_data_dir(args)
    log_path = data_dir.parent / "swarndb_crossover.log"

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc = find_or_spawn_server(args, logger, _build_proc)

    payload: Dict = {}
    try:
        client = make_client(args.grpc_port)
        try:
            return work(client, args)
        finally:
            try:
                client.close()
            except Exception:
                pass
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


def run_single_mode(args: argparse.Namespace) -> Tuple[bool, Dict]:
    """Single-insert baseline only. PASS always (records baseline)."""

    def _work(client: SwarnDBClient, a: argparse.Namespace) -> Tuple[bool, Dict]:
        report = HarnessReport("single")
        run_stamp = int(time.time())
        results: List[SingleInsertResult] = []
        for dim in a.dimensions:
            res = run_single_for_dim(
                client=client,
                dim=dim,
                n_vectors=a.n_vectors,
                warmup_rows=a.rps_warmup_rows,
                seed=a.seed,
                prefix=a.collection_name_prefix,
                run_stamp=run_stamp,
            )
            results.append(res)
            report.add(
                f"single_baseline_dim{dim}",
                True,
                f"qps={res.qps:.1f} elapsed={res.elapsed_seconds:.2f}s n={res.n_vectors}",
            )
        logger.info(report.summary())
        return True, {
            "mode": "single",
            "single_results": [asdict(r) for r in results],
        }

    return _run_with_server(args, _work)


def run_bulk_mode(args: argparse.Namespace) -> Tuple[bool, Dict]:
    """Bulk-insert sweep only. PASS always."""

    def _work(client: SwarnDBClient, a: argparse.Namespace) -> Tuple[bool, Dict]:
        report = HarnessReport("bulk")
        run_stamp = int(time.time())
        bulk_rows: List[BulkInsertResultRow] = []
        for dim in a.dimensions:
            for batch_size in a.batch_sizes:
                row = run_bulk_for_dim_batch(
                    client=client,
                    dim=dim,
                    batch_size=batch_size,
                    n_vectors=a.n_vectors,
                    warmup_rows=a.rps_warmup_rows,
                    seed=a.seed,
                    prefix=a.collection_name_prefix,
                    run_stamp=run_stamp,
                    single_qps=None,
                )
                bulk_rows.append(row)
                report.add(
                    f"bulk_dim{dim}_b{batch_size}",
                    True,
                    f"qps={row.qps:.1f} elapsed={row.elapsed_seconds:.2f}s "
                    f"chunks={row.n_chunks} n={row.n_vectors}",
                )
        logger.info(report.summary())
        return True, {
            "mode": "bulk",
            "bulk_results": [asdict(r) for r in bulk_rows],
        }

    return _run_with_server(args, _work)


def run_crossover_mode(
    args: argparse.Namespace,
    emit_detail: bool,
) -> Tuple[bool, Dict]:
    """Single baseline plus bulk sweep plus crossover analysis. PASS
    if at least one dim crosses over.
    """

    def _work(client: SwarnDBClient, a: argparse.Namespace) -> Tuple[bool, Dict]:
        mode_label = "all" if emit_detail else "crossover"
        report = HarnessReport(mode_label)
        run_stamp = int(time.time())

        # Step 1: single baseline per dim.
        singles: Dict[int, SingleInsertResult] = {}
        for dim in a.dimensions:
            res = run_single_for_dim(
                client=client,
                dim=dim,
                n_vectors=a.n_vectors,
                warmup_rows=a.rps_warmup_rows,
                seed=a.seed,
                prefix=a.collection_name_prefix,
                run_stamp=run_stamp,
            )
            singles[dim] = res
            if emit_detail:
                report.add(
                    f"single_baseline_dim{dim}",
                    True,
                    f"qps={res.qps:.1f} elapsed={res.elapsed_seconds:.2f}s",
                )

        # Step 2: bulk sweep per dim.
        bulk_rows: List[BulkInsertResultRow] = []
        for dim in a.dimensions:
            single_qps = singles[dim].qps
            for batch_size in a.batch_sizes:
                row = run_bulk_for_dim_batch(
                    client=client,
                    dim=dim,
                    batch_size=batch_size,
                    n_vectors=a.n_vectors,
                    warmup_rows=a.rps_warmup_rows,
                    seed=a.seed,
                    prefix=a.collection_name_prefix,
                    run_stamp=run_stamp,
                    single_qps=single_qps,
                )
                bulk_rows.append(row)
                if emit_detail:
                    ratio_txt = (
                        f" ratio={row.ratio_to_single:.2f}"
                        if row.ratio_to_single is not None else ""
                    )
                    report.add(
                        f"bulk_dim{dim}_b{batch_size}",
                        True,
                        f"qps={row.qps:.1f} elapsed={row.elapsed_seconds:.2f}s"
                        f"{ratio_txt}",
                    )

        # Step 3: crossover analysis per dim.
        crossovers: List[CrossoverResult] = []
        any_crossover = False
        for dim in a.dimensions:
            dim_rows = [r for r in bulk_rows if r.dim == dim]
            xo = analyze_crossover(
                dim=dim,
                single_qps=singles[dim].qps,
                bulk_rows=dim_rows,
                tested_batch_sizes=list(a.batch_sizes),
            )
            crossovers.append(xo)
            if xo.crossover_batch_size is not None:
                any_crossover = True
                logger.info(
                    "[CROSSOVER dim=%d] crossover at batch_size=%d "
                    "(bulk_qps=%.1f, single_qps=%.1f, ratio=%.2f)",
                    dim, xo.crossover_batch_size, xo.crossover_qps,
                    xo.single_qps, xo.crossover_ratio,
                )
                report.add(
                    f"crossover_dim{dim}",
                    True,
                    f"reached at batch_size={xo.crossover_batch_size} "
                    f"(ratio={xo.crossover_ratio:.2f})",
                )
            else:
                logger.info(
                    "[CROSSOVER dim=%d] not found; bulk never exceeds "
                    "single*%.2f within tested batch sizes %s",
                    dim, CROSSOVER_TOLERANCE, xo.tested_batch_sizes,
                )
                report.add(
                    f"crossover_dim{dim}",
                    False,
                    "no batch_size in sweep reached "
                    f"single_qps * {CROSSOVER_TOLERANCE:.2f}",
                )

        logger.info(report.summary())
        payload = {
            "mode": mode_label,
            "tolerance": CROSSOVER_TOLERANCE,
            "n_vectors": a.n_vectors,
            "single_results": {
                str(dim): asdict(singles[dim]) for dim in a.dimensions
            },
            "bulk_results": [asdict(r) for r in bulk_rows],
            "crossovers": [asdict(x) for x in crossovers],
            "any_crossover": any_crossover,
        }
        return any_crossover, payload

    return _run_with_server(args, _work)


# ---------------------------------------------------------------------------
# Data directory plumbing
# ---------------------------------------------------------------------------


def _resolve_data_dir(args: argparse.Namespace) -> Tuple[Path, bool]:
    """Pick the data directory for the run.

    If the user passed --data-dir, the harness uses it directly and
    does NOT delete it on exit. Otherwise the harness creates a fresh
    tmpdir and removes it on exit.

    Returns (data_dir, owns_data_dir).
    """
    if args.data_dir:
        base = Path(args.data_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base, False
    base = Path(tempfile.mkdtemp(prefix="swarndb_p10_crossover_"))
    return base, True


def _cleanup_data_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        logger.warning("cleanup of %s failed: %s", path, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_csv_ints(raw: str, flag: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{flag}: '{token}' is not an integer"
            )
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"{flag}: values must be positive (got {value})"
            )
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError(f"{flag}: at least one value required")
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SwarnDB bulk-vs-single crossover harness "
            "(Perf_Stability P10 Investigation B / P11 execution)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["single", "bulk", "crossover", "all"],
        default="all",
        help=(
            "Which mode to run. 'single' runs the baseline only; "
            "'bulk' runs the sweep only; 'crossover' runs both plus "
            "the crossover report; 'all' is crossover plus per-test "
            "detail."
        ),
    )
    parser.add_argument(
        "--binary-path",
        default=None,
        help=(
            "Path to the pre-built swarndb binary. The harness does NOT "
            "build. Optional: if a swarndb-server is already up on "
            "--rest-port (auto-detected via /readyz), the harness uses "
            "it and skips spawn. Required only when no external server "
            "is up on the configured port."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SWARNDB_HARNESS_DATA_DIR"),
        help=(
            "Optional base data directory. If omitted, the harness "
            "creates a tempdir and removes it on exit."
        ),
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=DEFAULT_N_VECTORS,
        help=(
            "Vector count per (dim, batch_size) test. Default 10000 "
            "to match the perf_benchmark signal that surfaced "
            "Investigation B."
        ),
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default=",".join(str(d) for d in DEFAULT_DIMENSIONS),
        help=(
            "Comma-separated vector dimensions to sweep. Default "
            "'128,1536'."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
        help=(
            "Comma-separated bulk batch sizes to sweep. Default "
            "'1,10,100,1000,5000'."
        ),
    )
    parser.add_argument(
        "--collection-name-prefix",
        default=DEFAULT_COLLECTION_PREFIX,
        help="Collection-name prefix (default 'p10_crossover').",
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=DEFAULT_REST_PORT,
        help="REST port for /readyz polling (default 18092).",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help="gRPC port for the SDK client (default 18093).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to dump full results as JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for vector generation (default 42).",
    )
    parser.add_argument(
        "--rps-warmup-rows",
        type=int,
        default=DEFAULT_RPS_WARMUP_ROWS,
        help=(
            "Rows to bulk-insert as a warm-up before each measured "
            "test (default 200). Keeps sub-100-row tests from being "
            "dominated by index init costs."
        ),
    )

    parsed = parser.parse_args(argv)

    # Translate CSV flags into typed lists.
    parsed.dimensions = _parse_csv_ints(parsed.dimensions, "--dimensions")
    parsed.batch_sizes = _parse_csv_ints(parsed.batch_sizes, "--batch-sizes")

    return parsed


def _write_output_json(path: Optional[str], payload: Dict) -> None:
    if not path:
        return
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("wrote results to %s", out)


def main() -> int:
    args = parse_args()

    if args.n_vectors <= 0:
        logger.error("[FAIL] --n-vectors must be positive (got %d)", args.n_vectors)
        return 2

    # --binary-path is optional in external mode (auto-detect picks up
    # a server already running on --rest-port). Validate the path only
    # when it was provided.
    if args.binary_path:
        binary_path = Path(args.binary_path)
        if not binary_path.exists():
            logger.error("[FAIL] binary not found at %s", binary_path)
            return 2

    try:
        if args.mode == "single":
            ok, payload = run_single_mode(args)
        elif args.mode == "bulk":
            ok, payload = run_bulk_mode(args)
        elif args.mode == "crossover":
            ok, payload = run_crossover_mode(args, emit_detail=False)
        else:
            ok, payload = run_crossover_mode(args, emit_detail=True)
    except SwarnDBError as exc:
        logger.error("[FAIL] SDK error: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("[FAIL] runtime error: %s", exc)
        return 2

    _write_output_json(args.output_json, payload)

    # In single / bulk modes, success is recording the data; no
    # crossover assertion applies.
    if args.mode in ("single", "bulk"):
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
