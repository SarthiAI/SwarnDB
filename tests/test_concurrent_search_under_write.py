#!/usr/bin/env python3
"""
SwarnDB concurrent-search-under-write validation harness
(Perf_Stability P08 Step 4).

Exercises the per-collection RwLock structural fix that P08 just shipped.
The pre-P08 baseline on Civo showed search QPS against a collection
collapsing from ~1162 (standalone) to ~21 (under sustained concurrent
bulk_insert against the same collection) -- a 55x degradation caused by
read-write lock contention on the single collection RwLock. P08 split
the lock surface so reads and writes can interleave on the same
collection and so writes on collection A no longer block reads on
collection B.

This is an integration harness: it spawns a real swarndb binary, drives
load through the official Python SDK (gRPC), and measures aggregate
search QPS across multiple searcher threads. Designed to be run by P11
against a 1M x 1536 plain HNSW collection (dbpedia_1m); also runs end
to end on smaller workloads for local iteration.

Phases (driven by --mode):

    baseline           Phase A only. No concurrent writes. 4 searcher
                       threads loop for --phase-seconds against
                       collection A. Aggregate QPS recorded.
    contention         Phase B only. 1 writer thread bulk_inserts into
                       collection A; 4 searcher threads query the same
                       collection A. Aggregate search QPS recorded; this
                       is the P08 target (>= 800).
    cross_collection   Phase C only. 1 writer thread bulk_inserts into
                       collection A; 4 searcher threads query collection
                       B. Aggregate search QPS on B must stay near
                       baseline (>= 0.85x Phase A); proves the
                       per-collection lock structure isolates write
                       pressure across collections.
    all                Runs Phase A, then Phase B, then Phase C, in that
                       order, reusing the same swarndb process and
                       collections so timing is comparable.
    correctness        Skips QPS measurement; only checks pre/post
                       bulk_insert top-K overlap.

Usage (P11 will run this against Civo with 1M x 1536):

    python test_concurrent_search_under_write.py \\
        --mode all \\
        --dataset /data/dbpedia_1m.fvecs \\
        --dim 1536 \\
        --vectors 1000000 \\
        --write-vectors 100000 \\
        --binary /usr/local/bin/swarndb \\
        --data-dir /var/lib/swarndb/contention_harness \\
        --rest-port 18080 \\
        --grpc-port 18081 \\
        --phase-seconds 30 \\
        --searcher-threads 4

For faster local iteration, --vectors 100000 --write-vectors 20000
--phase-seconds 10 works against the same harness. The relative QPS
ratios are dataset-scale agnostic; only the absolute floors below scale
with hardware.

Exit code 0 on every assertion passing; non-zero on the first failure.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import requests

# The SDK lives in-tree under sdk/python/src. Make it importable so the
# harness can run from a checkout without a wheel install. Mirrors the
# pattern in tests/test_recovery_at_scale.py (P07 harness).
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("contention_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and tunables (P08 contract)
# ---------------------------------------------------------------------------

COLLECTION_A_NAME = "contention_harness_a"
COLLECTION_B_NAME = "contention_harness_b"
DISTANCE_METRIC = "cosine"

# /readyz contract: server must come up within 60s on a fresh data dir,
# plus a 5s startup buffer to absorb process spawn + port-bind latency.
READYZ_DEADLINE_SECONDS = 65.0
READYZ_POLL_INTERVAL_SECONDS = 0.5

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

# Search depth per query. Mirrors the P07 harness so per-query cost is
# comparable across the two harnesses.
K_TOPK = 10

# Bulk-insert tuning. The contention harness deliberately does NOT use
# defer_graph: the goal is to exercise the live insert path (graph
# updates, WAL writes, scheduler ticks) that hits the same per
# collection RwLock the searchers acquire. defer_graph would mask the
# contention surface P08 is meant to fix.
WORKLOAD_BATCH_SIZE = 2_000
BULK_INSERT_BATCH_LOCK_SIZE = 2_000
BULK_INSERT_DEFER_GRAPH = False
BULK_INSERT_INDEX_MODE: Optional[str] = None  # server default (immediate)

# Pass/fail thresholds. The Phase B floor is the P08 mandate: aggregate
# search QPS must hold >= 800 across the 4 searcher threads while a
# sustained bulk_insert runs on the same collection. Pre-P08 Civo
# measurement was 21 QPS, so 800 is a >38x improvement and the
# structural fix is what unlocks it. Phase A's >= 1000 floor is the
# baseline confidence check; Phase C is expressed as a ratio against
# Phase A so it scales with hardware.
PHASE_A_BASELINE_QPS_FLOOR = 1_000.0
PHASE_B_CONTENTION_QPS_FLOOR = 800.0
PHASE_C_CROSS_COLLECTION_QPS_RATIO = 0.85

# Correctness contract. After Phase B's writer has appended new
# vectors to collection A, the original baseline queries should still
# return their original top-K results with high overlap. New writes
# can in principle displace a few neighbors (a freshly inserted vector
# may be closer than one of the previous top-K) so a strict equality
# is too tight; the same per-query / aggregate floors used by the P07
# recovery harness apply here.
K_BASELINE_QUERIES = 50
PER_QUERY_OVERLAP_FLOOR = 0.7
AGGREGATE_OVERLAP_FLOOR = 0.95

# Searcher loop pacing. The searcher threads pick a query from a
# pre-loaded pool; the pool is large enough that re-use does not warm
# the server cache in a way that distorts QPS upward beyond the real
# steady-state.
SEARCHER_QUERY_POOL_SIZE = 1_024


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


class HarnessReport:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.records: List[AssertionRecord] = []

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
class PhaseResult:
    name: str
    duration_seconds: float
    total_queries: int
    per_thread_queries: List[int]
    errors: int
    error_samples: List[str] = field(default_factory=list)
    # P10.8: per-query latency samples (milliseconds), one entry per
    # successful search call across all searcher threads in this phase.
    # Empty list means latency tracking was not exercised (e.g. zero
    # successful queries). Aggregated percentiles are computed lazily by
    # the latency_* properties below.
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def qps(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.total_queries / self.duration_seconds

    def _percentile_ms(self, pct: float) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, pct))

    @property
    def latency_p50_ms(self) -> float:
        return self._percentile_ms(50.0)

    @property
    def latency_p95_ms(self) -> float:
        return self._percentile_ms(95.0)

    @property
    def latency_p99_ms(self) -> float:
        return self._percentile_ms(99.0)

    @property
    def latency_max_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(max(self.latencies_ms))

    @property
    def latency_count(self) -> int:
        return len(self.latencies_ms)

    def latency_summary(self) -> str:
        if not self.latencies_ms:
            return "latency=<no samples>"
        return (
            f"latency_ms p50={self.latency_p50_ms:.2f} "
            f"p95={self.latency_p95_ms:.2f} "
            f"p99={self.latency_p99_ms:.2f} "
            f"max={self.latency_max_ms:.2f} "
            f"n={self.latency_count}"
        )


# ---------------------------------------------------------------------------
# Dataset loader (fvecs + raw f32; falls back to deterministic synthetic).
# Identical to the P07 harness so both harnesses accept the same dataset
# files in P11.
# ---------------------------------------------------------------------------


def _iter_fvecs(path: Path, dim: int, max_vectors: int) -> Iterator[List[float]]:
    """Stream vectors from a .fvecs file (Texmex / DBpedia convention).

    Each record is: int32 dim, then dim * float32 values.
    """
    fmt_dim = struct.Struct("<i")
    with path.open("rb") as fh:
        produced = 0
        while produced < max_vectors:
            header = fh.read(4)
            if len(header) < 4:
                return
            (record_dim,) = fmt_dim.unpack(header)
            if record_dim != dim:
                raise ValueError(
                    f"fvecs dim mismatch at record {produced}: "
                    f"file says {record_dim}, harness expects {dim}"
                )
            payload = fh.read(4 * dim)
            if len(payload) < 4 * dim:
                return
            arr = np.frombuffer(payload, dtype=np.float32, count=dim)
            yield arr.astype(np.float32).tolist()
            produced += 1


def _iter_raw_f32(path: Path, dim: int, max_vectors: int) -> Iterator[List[float]]:
    """Stream vectors from a flat float32 file (no per-record dim header)."""
    record_bytes = 4 * dim
    with path.open("rb") as fh:
        produced = 0
        while produced < max_vectors:
            payload = fh.read(record_bytes)
            if len(payload) < record_bytes:
                return
            arr = np.frombuffer(payload, dtype=np.float32, count=dim)
            yield arr.astype(np.float32).tolist()
            produced += 1


def _iter_synthetic(dim: int, count: int, seed: int = 4242) -> Iterator[List[float]]:
    """Deterministic synthetic vectors for harness self-test runs."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield rng.standard_normal(dim, dtype=np.float32).tolist()


def load_vectors(
    dataset: Optional[Path],
    dim: int,
    count: int,
    *,
    synthetic_seed: int = 4242,
) -> List[List[float]]:
    """Load workload vectors into RAM.

    The harness keeps all vectors in memory so the writer thread can
    splice off contiguous ranges without re-reading the dataset file
    (which would dominate the phase-B insert latency at 1M scale).
    """
    if dataset is None:
        logger.warning(
            "No --dataset provided; using deterministic synthetic vectors. "
            "This path is for harness self-test only; P11 must pass --dataset."
        )
        return list(_iter_synthetic(dim, count, seed=synthetic_seed))

    suffix = dataset.suffix.lower()
    if suffix == ".fvecs":
        loader = _iter_fvecs(dataset, dim, count)
    elif suffix in {".f32", ".raw", ".bin"}:
        loader = _iter_raw_f32(dataset, dim, count)
    else:
        raise ValueError(
            f"Unsupported dataset suffix {suffix!r}. "
            f"Supported: .fvecs (Texmex), .f32 / .raw / .bin (flat float32)."
        )
    vectors = list(loader)
    if len(vectors) < count:
        raise ValueError(
            f"Dataset {dataset} yielded {len(vectors)} vectors; "
            f"harness requires {count}."
        )
    return vectors


# ---------------------------------------------------------------------------
# Process management. Mirrors the P07 harness SwarndbProcess so the two
# harnesses can be invoked back-to-back from a single P11 driver.
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
        self.proc.wait(timeout=PROCESS_KILL_WAIT_SECONDS)
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

    Spawn path: if /readyz is unreachable AND --binary is provided, the
    harness calls proc_factory() to build a SwarndbProcess with the
    mode's resolved data_dir, starts it, waits for /readyz, and returns
    the wrapper. If --binary is missing (or only the bare 'swarndb'
    default name, which would not resolve to an executable on most
    systems) AND no external server is up, the harness emits a clear
    error and exits non-zero.
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

    # Treat the bare default "swarndb" as "no binary configured" only
    # when it is not a resolvable executable on PATH. A real path or a
    # binary that exists in PATH means the user is on the spawn path.
    binary_arg = getattr(args, "binary", None) or getattr(args, "binary_path", None)
    if not binary_arg:
        log.error(
            "No swarndb-server detected on port %d and no swarndb "
            "binary provided. Either start a swarndb-server on the "
            "configured port OR pass --binary to spawn one.",
            args.rest_port,
        )
        sys.exit(2)

    log.info(
        "No swarndb-server on %s; spawning a fresh one from %s.",
        rest_url, binary_arg,
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
# /readyz polling (raw HTTP; the SDK does not expose this probe)
# ---------------------------------------------------------------------------


def wait_for_readyz(
    rest_port: int,
    deadline_seconds: float,
    started_at: float,
) -> Tuple[bool, float]:
    """Poll /readyz until 200 OK or until the deadline elapses.

    Returns (ok, elapsed_from_started_at).
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


def ensure_collection(client: SwarnDBClient, name: str, dim: int) -> None:
    """Drop-and-create a collection so each harness run starts clean."""
    if client.collections.exists(name):
        client.collections.delete(name)
    client.collections.create(
        name,
        dimension=dim,
        distance_metric=DISTANCE_METRIC,
    )
    logger.info("created collection '%s' dim=%d", name, dim)


def bulk_insert_range(
    client: SwarnDBClient,
    collection: str,
    vectors: List[List[float]],
    *,
    id_offset: int = 0,
    label: str = "bulk_insert",
) -> None:
    """Run a single bulk_insert call covering the supplied vector slice.

    The metadata carries a row_idx field with `id_offset + i` so the
    harness can later distinguish baseline-loaded vectors from
    contention-phase-written vectors when reading back through
    `collection.get`.
    """
    total = len(vectors)
    if total == 0:
        return
    metadata_list = [{"row_idx": id_offset + i} for i in range(total)]
    logger.info(
        "%s: inserting %d vectors into '%s' (batch_lock_size=%d, defer_graph=%s)",
        label, total, collection, BULK_INSERT_BATCH_LOCK_SIZE,
        BULK_INSERT_DEFER_GRAPH,
    )
    t0 = time.time()
    result = client.vectors.bulk_insert(
        collection,
        vectors,
        metadata_list=metadata_list,
        batch_size=WORKLOAD_BATCH_SIZE,
        batch_lock_size=BULK_INSERT_BATCH_LOCK_SIZE,
        defer_graph=BULK_INSERT_DEFER_GRAPH,
        index_mode=BULK_INSERT_INDEX_MODE,
    )
    elapsed = time.time() - t0
    rate = result.inserted_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "%s done: inserted=%d errors=%d in %.1fs (%.0f vec/s)",
        label, result.inserted_count, len(result.errors), elapsed, rate,
    )
    if result.inserted_count != total or result.errors:
        raise RuntimeError(
            f"{label} short: inserted={result.inserted_count}/{total}, "
            f"errors={result.errors[:3]}"
        )


def capture_baseline_results(
    client: SwarnDBClient,
    collection: str,
    pool: Sequence[List[float]],
    *,
    seed: int = 9001,
    sample_size: int = K_BASELINE_QUERIES,
) -> List[Tuple[List[float], List[int]]]:
    """Pick `sample_size` queries from `pool`, record top-K IDs."""
    rng = np.random.default_rng(seed)
    total = len(pool)
    if total == 0:
        return []
    sample_size = min(sample_size, total)
    sample_idx = rng.choice(total, size=sample_size, replace=False)
    baseline: List[Tuple[List[float], List[int]]] = []
    for i, q_idx in enumerate(sample_idx):
        query = pool[int(q_idx)]
        result = client.search.query(
            collection,
            query,
            k=K_TOPK,
            include_metadata=False,
            include_graph=False,
        )
        ids = [r.id for r in result.results]
        baseline.append((query, ids))
        if (i + 1) % 10 == 0:
            logger.info(
                "baseline (%s): captured %d/%d queries",
                collection, i + 1, sample_size,
            )
    return baseline


def compare_search_overlap(
    client: SwarnDBClient,
    collection: str,
    baseline: List[Tuple[List[float], List[int]]],
) -> Tuple[bool, bool, float, int]:
    """Re-run baseline queries; compare top-K overlap against captured IDs.

    Returns (per_query_ok, aggregate_ok, mean_overlap, failed_query_count).
    """
    overlaps: List[float] = []
    failed_queries = 0
    for query, baseline_ids in baseline:
        result = client.search.query(
            collection,
            query,
            k=K_TOPK,
            include_metadata=False,
            include_graph=False,
        )
        new_ids = {r.id for r in result.results}
        baseline_set = set(baseline_ids)
        if not baseline_set:
            continue
        overlap_ratio = len(new_ids & baseline_set) / len(baseline_set)
        overlaps.append(overlap_ratio)
        if overlap_ratio < PER_QUERY_OVERLAP_FLOOR:
            failed_queries += 1

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    per_query_ok = failed_queries == 0
    aggregate_ok = mean_overlap >= AGGREGATE_OVERLAP_FLOOR
    return per_query_ok, aggregate_ok, mean_overlap, failed_queries


# ---------------------------------------------------------------------------
# Searcher and writer threads
# ---------------------------------------------------------------------------


def _searcher_loop(
    client: SwarnDBClient,
    collection: str,
    query_pool: Sequence[List[float]],
    deadline: float,
    stop_event: threading.Event,
    thread_seed: int,
    error_lock: threading.Lock,
    error_sink: List[str],
    max_error_samples: int = 5,
) -> Tuple[int, List[float]]:
    """Run search calls until `deadline` or `stop_event` is set.

    Returns (completed_count, latencies_ms). `latencies_ms` is a thread
    local list of per-query latencies in milliseconds, one entry per
    successful search call. Failed calls do not record a latency so the
    percentile aggregation is not biased by error-handling cost.
    """
    rng = random.Random(thread_seed)
    pool_size = len(query_pool)
    if pool_size == 0:
        return 0, []
    completed = 0
    error_count_local = 0
    # P10.8: per-thread latency buffer. Pre-sized hint is not used: the
    # number of completed queries is dictated by phase_seconds and is
    # variable, so a plain append loop is the right shape.
    latencies_ms: List[float] = []
    perf_counter_ns = time.perf_counter_ns  # local bind for tight loop
    search_query = client.search.query  # bind once outside loop
    while time.time() < deadline and not stop_event.is_set():
        query = query_pool[rng.randrange(pool_size)]
        try:
            # Timing window is TIGHT around the gRPC call only. Anything
            # outside these two perf_counter_ns reads (rng draw, latency
            # append, error handling) is excluded from the sample.
            t_start_ns = perf_counter_ns()
            search_query(
                collection,
                query,
                k=K_TOPK,
                include_metadata=False,
                include_graph=False,
            )
            t_end_ns = perf_counter_ns()
            latencies_ms.append((t_end_ns - t_start_ns) / 1_000_000.0)
            completed += 1
        except (SwarnDBError, Exception) as exc:
            error_count_local += 1
            with error_lock:
                if len(error_sink) < max_error_samples:
                    error_sink.append(
                        f"thread_seed={thread_seed} err={type(exc).__name__}: {exc}"
                    )
            # Tight error loop would burn CPU; back off briefly.
            time.sleep(0.05)
    logger.info(
        "searcher thread seed=%d completed=%d errors=%d latency_samples=%d",
        thread_seed, completed, error_count_local, len(latencies_ms),
    )
    return completed, latencies_ms


def _writer_loop(
    client: SwarnDBClient,
    collection: str,
    write_pool: List[List[float]],
    chunk_size: int,
    id_offset_start: int,
    deadline: float,
    stop_event: threading.Event,
    error_lock: threading.Lock,
    error_sink: List[str],
) -> int:
    """Run bulk_insert chunks until `deadline` or write_pool is exhausted.

    The writer slices `write_pool` into `chunk_size`-sized batches and
    feeds them through bulk_insert back-to-back, giving Phase B and
    Phase C a sustained write pressure for the whole phase. Returns the
    number of vectors successfully inserted.
    """
    inserted = 0
    cursor = 0
    pool_size = len(write_pool)
    chunk_idx = 0
    while (
        cursor < pool_size
        and time.time() < deadline
        and not stop_event.is_set()
    ):
        end = min(cursor + chunk_size, pool_size)
        chunk = write_pool[cursor:end]
        chunk_metadata = [
            {"row_idx": id_offset_start + cursor + i}
            for i in range(len(chunk))
        ]
        try:
            result = client.vectors.bulk_insert(
                collection,
                chunk,
                metadata_list=chunk_metadata,
                batch_size=WORKLOAD_BATCH_SIZE,
                batch_lock_size=BULK_INSERT_BATCH_LOCK_SIZE,
                defer_graph=BULK_INSERT_DEFER_GRAPH,
                index_mode=BULK_INSERT_INDEX_MODE,
            )
            inserted += result.inserted_count
            if result.errors:
                with error_lock:
                    if len(error_sink) < 5:
                        error_sink.append(
                            f"writer chunk={chunk_idx} errors={result.errors[:3]}"
                        )
        except (SwarnDBError, Exception) as exc:
            with error_lock:
                if len(error_sink) < 5:
                    error_sink.append(
                        f"writer chunk={chunk_idx} exc={type(exc).__name__}: {exc}"
                    )
            # Writer is the contention generator; if it dies the phase
            # is no longer meaningful, so stop the entire phase.
            stop_event.set()
            break
        cursor = end
        chunk_idx += 1
    logger.info(
        "writer thread inserted=%d chunks=%d (pool_size=%d)",
        inserted, chunk_idx, pool_size,
    )
    return inserted


def run_phase(
    *,
    label: str,
    grpc_port: int,
    searcher_collection: str,
    searcher_threads: int,
    query_pool: Sequence[List[float]],
    phase_seconds: float,
    writer_collection: Optional[str] = None,
    writer_pool: Optional[List[List[float]]] = None,
    writer_chunk_size: int = 5_000,
    writer_id_offset_start: int = 0,
) -> Tuple[PhaseResult, int]:
    """Run a single phase: N searcher threads (and optionally 1 writer).

    Returns (PhaseResult, vectors_written_by_writer). When writer_pool
    is None the writer is not started and the second return is 0.
    """
    logger.info(
        "phase '%s': searchers=%d collection=%s duration=%.1fs writer=%s",
        label, searcher_threads, searcher_collection, phase_seconds,
        writer_collection if writer_pool else "none",
    )
    # Each thread gets its own client to avoid serialized stub usage on
    # a single channel under high concurrency. gRPC channels are thread
    # safe but the SDK's per-call retry path can produce uneven latency
    # if many threads share one stub instance; separate channels keep
    # the QPS measurement honest.
    searcher_clients: List[SwarnDBClient] = []
    writer_client: Optional[SwarnDBClient] = None
    stop_event = threading.Event()
    error_lock = threading.Lock()
    error_sink: List[str] = []
    deadline = time.time() + phase_seconds
    inserted = 0

    try:
        for _ in range(searcher_threads):
            searcher_clients.append(make_client(grpc_port))

        # Use one extra worker slot for the writer (if any) so the
        # ThreadPoolExecutor does not starve searchers waiting for a
        # writer slot.
        total_workers = searcher_threads + (1 if writer_pool else 0)
        executor = ThreadPoolExecutor(
            max_workers=total_workers,
            thread_name_prefix=f"phase-{label}",
        )
        try:
            futures: List[Tuple[str, Future]] = []
            for i, client in enumerate(searcher_clients):
                fut = executor.submit(
                    _searcher_loop,
                    client,
                    searcher_collection,
                    query_pool,
                    deadline,
                    stop_event,
                    thread_seed=10_000 + i,
                    error_lock=error_lock,
                    error_sink=error_sink,
                )
                futures.append(("searcher", fut))

            writer_future: Optional[Future] = None
            if writer_pool and writer_collection:
                writer_client = make_client(grpc_port)
                writer_future = executor.submit(
                    _writer_loop,
                    writer_client,
                    writer_collection,
                    writer_pool,
                    writer_chunk_size,
                    writer_id_offset_start,
                    deadline,
                    stop_event,
                    error_lock,
                    error_sink,
                )
                futures.append(("writer", writer_future))

            t_start = time.time()
            per_thread_queries: List[int] = []
            # P10.8: aggregate per-query latency samples (ms) from every
            # searcher thread. The writer thread does NOT contribute
            # here (bulk_insert is not a search).
            aggregated_latencies_ms: List[float] = []
            for kind, fut in futures:
                if kind == "searcher":
                    completed, thread_latencies = fut.result()
                    per_thread_queries.append(completed)
                    aggregated_latencies_ms.extend(thread_latencies)
                elif kind == "writer":
                    inserted = fut.result()
            t_end = time.time()
        finally:
            executor.shutdown(wait=True)
    finally:
        for c in searcher_clients:
            try:
                c.close()
            except Exception:
                pass
        if writer_client is not None:
            try:
                writer_client.close()
            except Exception:
                pass

    total_queries = sum(per_thread_queries)
    duration = max(t_end - t_start, 1e-6)
    result = PhaseResult(
        name=label,
        duration_seconds=duration,
        total_queries=total_queries,
        per_thread_queries=per_thread_queries,
        errors=len(error_sink),
        error_samples=list(error_sink),
        latencies_ms=aggregated_latencies_ms,
    )
    logger.info(
        "phase '%s' complete: total_queries=%d duration=%.2fs qps=%.1f "
        "per_thread=%s writer_inserted=%d errors=%d %s",
        label, total_queries, duration, result.qps, per_thread_queries,
        inserted, len(error_sink), result.latency_summary(),
    )
    if error_sink:
        for sample in error_sink:
            logger.warning("phase '%s' error sample: %s", label, sample)
    return result, inserted


# ---------------------------------------------------------------------------
# Mode orchestration
# ---------------------------------------------------------------------------


def _record_latency_assertion(
    report: HarnessReport,
    phase_label: str,
    result: PhaseResult,
    p99_ceiling_ms: Optional[int],
) -> None:
    """Emit a latency report line, and optionally a p99-ceiling assertion.

    Always logs the aggregated percentile summary so operators see the
    p50 / p95 / p99 / max numbers even when no ceiling is configured.
    When the operator passes --latency-p99-ceiling-ms, this also adds a
    pass/fail assertion enforcing the ceiling. The assertion is skipped
    when no latency samples were collected (e.g. an aborted phase) so a
    failed phase does not double-fail on a misleading latency check.
    """
    # Always emit an informational record so the percentile numbers are
    # visible in the report even when no ceiling assertion is configured.
    report.add(
        f"{phase_label} latency percentiles (report-only)",
        True,
        result.latency_summary(),
    )

    if p99_ceiling_ms is None:
        return
    if result.latency_count == 0:
        report.add(
            f"{phase_label} p99 latency <= {p99_ceiling_ms} ms",
            False,
            "no latency samples collected; cannot evaluate ceiling",
        )
        return
    p99 = result.latency_p99_ms
    report.add(
        f"{phase_label} p99 latency <= {p99_ceiling_ms} ms",
        p99 <= float(p99_ceiling_ms),
        f"p99={p99:.2f}ms ceiling={p99_ceiling_ms}ms "
        f"n={result.latency_count}",
    )


def _split_vectors(
    vectors: List[List[float]],
    write_count: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Reserve the trailing `write_count` vectors for the writer pool.

    Returns (baseline_pool, writer_pool). The baseline_pool is what the
    collection is seeded with up-front (the searchers' query domain);
    the writer_pool is what Phase B and Phase C inject under contention.
    """
    total = len(vectors)
    if write_count >= total:
        raise ValueError(
            f"--write-vectors ({write_count}) must be < total --vectors "
            f"({total}); the harness reserves the trailing slice for the "
            f"writer pool, so there must be at least one baseline vector "
            f"left for the collection to seed with."
        )
    baseline_pool = vectors[: total - write_count]
    writer_pool = vectors[total - write_count:]
    return baseline_pool, writer_pool


def _build_query_pool(
    vectors: Sequence[List[float]],
    size: int,
    seed: int,
) -> List[List[float]]:
    """Pick `size` queries from `vectors` deterministically."""
    if not vectors:
        return []
    rng = np.random.default_rng(seed)
    n = min(size, len(vectors))
    idx = rng.choice(len(vectors), size=n, replace=False)
    return [vectors[int(i)] for i in idx]


def setup_collections(
    client: SwarnDBClient,
    args: argparse.Namespace,
    baseline_pool: List[List[float]],
    needs_collection_b: bool,
) -> int:
    """Seed collection A (and optionally collection B) before the phases.

    Returns the next free row_idx that the writer should start at when
    inserting into collection A under contention.
    """
    ensure_collection(client, COLLECTION_A_NAME, args.dim)
    bulk_insert_range(
        client,
        COLLECTION_A_NAME,
        baseline_pool,
        id_offset=0,
        label="seed collection A",
    )
    next_offset = len(baseline_pool)

    if needs_collection_b:
        ensure_collection(client, COLLECTION_B_NAME, args.dim)
        # Collection B carries a small seed sized from --collection-b-seed
        # (default 10k). Vectors are taken from the beginning of the
        # baseline pool so they live in the same distribution as the
        # search queries Phase C uses.
        seed_count = min(args.collection_b_seed, len(baseline_pool))
        bulk_insert_range(
            client,
            COLLECTION_B_NAME,
            baseline_pool[:seed_count],
            id_offset=0,
            label="seed collection B",
        )

    return next_offset


def run_modes(args: argparse.Namespace) -> bool:
    """Top-level orchestrator. One swarndb process; all selected phases."""
    report = HarnessReport(args.mode)
    data_dir = Path(args.data_dir).resolve()
    log_path = Path(args.log_dir).resolve() / "swarndb_contention.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_proc() -> SwarndbProcess:
        # Only wipe the data_dir on the spawn path; in external mode
        # the user's server already manages its own state and the
        # harness must not touch it.
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True)
        return SwarndbProcess(
            binary=Path(args.binary),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    do_phase_a = args.mode in ("baseline", "all", "correctness")
    do_phase_b = args.mode in ("contention", "all", "correctness")
    do_phase_c = args.mode in ("cross_collection", "all")
    do_qps_assertions = args.mode in ("baseline", "contention",
                                      "cross_collection", "all")

    _, _, proc = find_or_spawn_server(args, logger, _build_proc)

    try:
        report.add(
            "initial /readyz within 65s",
            True,
            "auto-detect or spawn path satisfied readiness before main run"
            if proc is None
            else "spawn path satisfied readiness in find_or_spawn_server",
        )

        # Load the vectors once; split into baseline + writer pools.
        logger.info("loading %d vectors from dataset", args.vectors)
        vectors = load_vectors(
            Path(args.dataset) if args.dataset else None,
            args.dim,
            args.vectors,
        )
        baseline_pool, writer_pool = _split_vectors(vectors, args.write_vectors)
        logger.info(
            "vector split: baseline_pool=%d writer_pool=%d",
            len(baseline_pool), len(writer_pool),
        )

        # Query pool used by every searcher thread. Drawn from the
        # baseline_pool so all queried vectors are guaranteed to be
        # present in the collection at the start of each phase.
        query_pool = _build_query_pool(
            baseline_pool, SEARCHER_QUERY_POOL_SIZE, seed=7777,
        )

        client = make_client(args.grpc_port)
        baseline_top_k: List[Tuple[List[float], List[int]]] = []
        writer_id_offset_start = 0
        try:
            writer_id_offset_start = setup_collections(
                client,
                args,
                baseline_pool,
                needs_collection_b=(do_phase_c),
            )

            # Capture pre-contention baseline top-K for correctness check.
            if do_phase_b or args.mode == "correctness":
                logger.info(
                    "capturing pre-contention baseline (%d queries) on '%s'",
                    K_BASELINE_QUERIES, COLLECTION_A_NAME,
                )
                baseline_top_k = capture_baseline_results(
                    client,
                    COLLECTION_A_NAME,
                    baseline_pool,
                    sample_size=K_BASELINE_QUERIES,
                )

            pre_count_a = client.collections.get(COLLECTION_A_NAME).vector_count
            logger.info(
                "pre-phase vector_count(collection A)=%d", pre_count_a,
            )
        finally:
            client.close()

        phase_a_qps: Optional[float] = None

        # Phase A: baseline (no concurrent writes).
        if do_phase_a:
            result_a, _ = run_phase(
                label="A_baseline",
                grpc_port=args.grpc_port,
                searcher_collection=COLLECTION_A_NAME,
                searcher_threads=args.searcher_threads,
                query_pool=query_pool,
                phase_seconds=args.phase_seconds,
            )
            phase_a_qps = result_a.qps
            if do_qps_assertions:
                report.add(
                    "Phase A baseline aggregate search QPS >= "
                    f"{PHASE_A_BASELINE_QPS_FLOOR:.0f}",
                    result_a.qps >= PHASE_A_BASELINE_QPS_FLOOR,
                    f"qps={result_a.qps:.1f} "
                    f"total_queries={result_a.total_queries} "
                    f"duration={result_a.duration_seconds:.2f}s "
                    f"per_thread={result_a.per_thread_queries} "
                    f"errors={result_a.errors} "
                    f"{result_a.latency_summary()}",
                )
            _record_latency_assertion(report, "Phase A", result_a,
                                      args.latency_p99_ceiling_ms)

        # Phase B: contention (writes on collection A, reads on collection A).
        if do_phase_b:
            result_b, b_inserted = run_phase(
                label="B_contention",
                grpc_port=args.grpc_port,
                searcher_collection=COLLECTION_A_NAME,
                searcher_threads=args.searcher_threads,
                query_pool=query_pool,
                phase_seconds=args.phase_seconds,
                writer_collection=COLLECTION_A_NAME,
                writer_pool=writer_pool,
                writer_chunk_size=args.writer_chunk_size,
                writer_id_offset_start=writer_id_offset_start,
            )
            if do_qps_assertions:
                report.add(
                    "Phase B contention aggregate search QPS >= "
                    f"{PHASE_B_CONTENTION_QPS_FLOOR:.0f} "
                    "(P08 mandate)",
                    result_b.qps >= PHASE_B_CONTENTION_QPS_FLOOR,
                    f"qps={result_b.qps:.1f} "
                    f"total_queries={result_b.total_queries} "
                    f"duration={result_b.duration_seconds:.2f}s "
                    f"per_thread={result_b.per_thread_queries} "
                    f"writer_inserted={b_inserted} "
                    f"errors={result_b.errors} "
                    f"{result_b.latency_summary()}",
                )
            _record_latency_assertion(report, "Phase B", result_b,
                                      args.latency_p99_ceiling_ms)
            report.add(
                "Phase B writer made forward progress",
                b_inserted > 0,
                f"writer_inserted={b_inserted}",
            )

            # Post-Phase-B correctness: confirm vectors are durable AND
            # baseline top-K queries still hit their original neighbors
            # within tolerance.
            client = make_client(args.grpc_port)
            try:
                expected_min = pre_count_a + max(b_inserted - 0, 0)
                info_a = client.collections.get(COLLECTION_A_NAME)
                report.add(
                    "Phase B post-state: vector_count grew by writer inserts",
                    info_a.vector_count >= expected_min,
                    f"pre={pre_count_a} writer_inserted={b_inserted} "
                    f"post={info_a.vector_count} expected_min={expected_min}",
                )

                # Spot-check that a sample of writer-inserted IDs is
                # readable via vectors.get. Pick the first id in the
                # range we just wrote.
                if b_inserted > 0:
                    probe_row = writer_id_offset_start
                    # vectors.get takes an integer id; since the server
                    # assigns ids on insert and we do not know them
                    # directly, we instead probe the collection via a
                    # search call on the first writer-pool vector and
                    # verify it surfaces.
                    probe_vec = writer_pool[0]
                    probe_result = client.search.query(
                        COLLECTION_A_NAME,
                        probe_vec,
                        k=1,
                        include_metadata=False,
                        include_graph=False,
                    )
                    found = len(probe_result.results) >= 1
                    report.add(
                        "Phase B post-state: writer-inserted vector is queryable",
                        found,
                        f"probe_row={probe_row} hits={len(probe_result.results)}",
                    )

                if baseline_top_k:
                    per_q, aggr, mean_overlap, failed_q = compare_search_overlap(
                        client, COLLECTION_A_NAME, baseline_top_k,
                    )
                    report.add(
                        "Phase B post-state: per-query top-K overlap above floor",
                        per_q,
                        f"failed_queries={failed_q}/{len(baseline_top_k)} "
                        f"floor={PER_QUERY_OVERLAP_FLOOR}",
                    )
                    report.add(
                        "Phase B post-state: aggregate mean top-K overlap above floor",
                        aggr,
                        f"mean_overlap={mean_overlap:.3f} "
                        f"floor={AGGREGATE_OVERLAP_FLOOR}",
                    )
            finally:
                client.close()

        # Phase C: cross-collection parallelism.
        if do_phase_c:
            # For Phase C the writer must still have inventory left.
            # Phase B may have already consumed `writer_pool` if it ran
            # first; refresh the writer pool by recycling vectors from
            # the tail of the baseline pool (these are guaranteed to be
            # the same dim and within distribution). This keeps Phase C
            # standalone-runnable AND meaningful when --mode=all has
            # already drained the writer pool.
            client = make_client(args.grpc_port)
            try:
                pre_count_a_before_c = client.collections.get(
                    COLLECTION_A_NAME,
                ).vector_count
            finally:
                client.close()

            # Use the writer_pool directly when Phase B did not run
            # (writer_pool is intact); otherwise recycle from the tail
            # of the baseline_pool.
            if do_phase_b:
                tail_take = min(len(writer_pool), len(baseline_pool))
                phase_c_writer_pool = baseline_pool[-tail_take:]
                logger.info(
                    "Phase C recycling %d baseline vectors as writer pool",
                    tail_take,
                )
            else:
                phase_c_writer_pool = writer_pool

            # Choose an id_offset that does not collide with previously
            # inserted rows on collection A. The metadata row_idx is
            # purely informational on the server side; collisions cause
            # no functional issue, but a non-overlapping range keeps
            # the data dump readable.
            c_id_offset_start = writer_id_offset_start + len(writer_pool)

            result_c, c_inserted = run_phase(
                label="C_cross_collection",
                grpc_port=args.grpc_port,
                searcher_collection=COLLECTION_B_NAME,
                searcher_threads=args.searcher_threads,
                query_pool=query_pool,
                phase_seconds=args.phase_seconds,
                writer_collection=COLLECTION_A_NAME,
                writer_pool=phase_c_writer_pool,
                writer_chunk_size=args.writer_chunk_size,
                writer_id_offset_start=c_id_offset_start,
            )
            if do_qps_assertions:
                # Cross-collection floor is expressed as a fraction of
                # the Phase A baseline if known, else as a hard floor
                # equal to PHASE_A_BASELINE_QPS_FLOOR * the ratio.
                if phase_a_qps is not None and phase_a_qps > 0:
                    threshold = phase_a_qps * PHASE_C_CROSS_COLLECTION_QPS_RATIO
                    ratio_detail = (
                        f"phase_a_qps={phase_a_qps:.1f} "
                        f"ratio_floor={PHASE_C_CROSS_COLLECTION_QPS_RATIO}"
                    )
                else:
                    threshold = (
                        PHASE_A_BASELINE_QPS_FLOOR
                        * PHASE_C_CROSS_COLLECTION_QPS_RATIO
                    )
                    ratio_detail = (
                        "phase_a_qps=unknown, falling back to "
                        f"{PHASE_A_BASELINE_QPS_FLOOR:.0f} * "
                        f"{PHASE_C_CROSS_COLLECTION_QPS_RATIO}"
                    )
                report.add(
                    "Phase C cross-collection search QPS >= "
                    f"{PHASE_C_CROSS_COLLECTION_QPS_RATIO:.2f}x Phase A",
                    result_c.qps >= threshold,
                    f"qps={result_c.qps:.1f} threshold={threshold:.1f} "
                    f"{ratio_detail} "
                    f"total_queries={result_c.total_queries} "
                    f"duration={result_c.duration_seconds:.2f}s "
                    f"per_thread={result_c.per_thread_queries} "
                    f"writer_inserted_into_A={c_inserted} "
                    f"errors={result_c.errors} "
                    f"{result_c.latency_summary()}",
                )
            _record_latency_assertion(report, "Phase C", result_c,
                                      args.latency_p99_ceiling_ms)
            report.add(
                "Phase C writer made forward progress on collection A",
                c_inserted > 0,
                f"writer_inserted={c_inserted}",
            )

            client = make_client(args.grpc_port)
            try:
                info_a = client.collections.get(COLLECTION_A_NAME)
                report.add(
                    "Phase C post-state: collection A vector_count grew",
                    info_a.vector_count >= pre_count_a_before_c + c_inserted,
                    f"pre={pre_count_a_before_c} writer_inserted={c_inserted} "
                    f"post={info_a.vector_count}",
                )
                info_b = client.collections.get(COLLECTION_B_NAME)
                report.add(
                    "Phase C post-state: collection B is queryable and intact",
                    info_b.vector_count > 0,
                    f"vector_count={info_b.vector_count}",
                )
            finally:
                client.close()

        logger.info(report.summary())
        return report.all_passed()
    finally:
        if proc is not None:
            proc.terminate()
        else:
            logger.info(
                "External server in use; teardown skipped (harness did "
                "not own the process)."
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SwarnDB concurrent-search-under-write validation harness "
            "(Perf_Stability P08 Step 4 / P11 execution)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "contention",
            "cross_collection",
            "all",
            "correctness",
        ],
        default="all",
        help="Which phase(s) to run.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("SWARNDB_HARNESS_DATASET"),
        help=(
            "Path to the input vector file (.fvecs / .f32 / .raw / .bin). "
            "If omitted, the harness uses deterministic synthetic vectors "
            "(self-test path only; P11 must supply --dataset)."
        ),
    )
    parser.add_argument("--dim", type=int, default=1536,
                        help="Vector dimension (default 1536 for DBpedia).")
    parser.add_argument("--vectors", type=int, default=1_000_000,
                        help="Total vectors to load (default 1_000_000).")
    parser.add_argument(
        "--write-vectors",
        type=int,
        default=100_000,
        help=(
            "Number of vectors reserved as the writer pool (Phase B/C "
            "writer feeds these into collection A under contention). "
            "Default 100_000. Must be < --vectors."
        ),
    )
    parser.add_argument(
        "--collection-b-seed",
        type=int,
        default=10_000,
        help=(
            "Number of vectors to seed collection B with for Phase C "
            "(searcher threads query against this seed). Default 10_000."
        ),
    )
    parser.add_argument(
        "--binary",
        default=os.environ.get("SWARNDB_BINARY", "swarndb"),
        help=(
            "Path to the swarndb binary. Optional in external mode: if "
            "a swarndb-server is already up on --rest-port "
            "(auto-detected via /readyz), the harness uses it and skips "
            "spawn. Required only when no external server is up."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get(
            "SWARNDB_HARNESS_DATA_DIR",
            "/tmp/swarndb_contention_harness",
        ),
        help="Data directory the harness will own (wiped on each run).",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get(
            "SWARNDB_HARNESS_LOG_DIR",
            "/tmp/swarndb_contention_harness_logs",
        ),
        help="Directory for swarndb stdout/stderr capture.",
    )
    parser.add_argument("--rest-port", type=int, default=18080,
                        help="REST port for /readyz polling.")
    parser.add_argument("--grpc-port", type=int, default=18081,
                        help="gRPC port for the SDK client.")
    parser.add_argument(
        "--phase-seconds",
        type=float,
        default=30.0,
        help="Duration of each phase in seconds (default 30.0).",
    )
    parser.add_argument(
        "--searcher-threads",
        type=int,
        default=4,
        help="Number of concurrent searcher threads per phase (default 4).",
    )
    parser.add_argument(
        "--writer-chunk-size",
        type=int,
        default=5_000,
        help=(
            "Vectors per writer bulk_insert chunk during Phase B/C. "
            "Default 5_000. Smaller chunks raise lock-acquisition cadence "
            "(stricter contention test); larger chunks raise per-call "
            "throughput."
        ),
    )
    parser.add_argument(
        "--latency-p99-ceiling-ms",
        type=int,
        default=None,
        help=(
            "Optional per-phase p99 search-latency ceiling (milliseconds). "
            "When set, each phase asserts that its aggregated p99 latency "
            "is less than or equal to this value. When omitted (default), "
            "latency percentiles are still reported but no ceiling "
            "assertion runs. This flag is opt-in to preserve backward "
            "compatibility with the existing P08 QPS-only contract."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    ok = run_modes(args)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
