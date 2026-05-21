#!/usr/bin/env python3
"""
SwarnDB recovery-at-scale validation harness (Perf_Stability P07 Step 5).

NOTE: This harness spawns and manages its own swarndb-server because
the test inherently kills the server with SIGKILL and restarts it to
measure recovery. External-server mode is not supported. Pass
--binary so the harness can own the process; auto-detect against an
already-running server would be incoherent with the SIGKILL + restart
cycle.

Exercises the Option B recovery path that P07 just shipped: incremental
topology persistence (hnsw.base + hnsw.delta + graph.base + graph.delta +
wal_meta.json), tightened snapshot cadence (25k mutations / 120s), G1
fsync ordering, G2 delta/base coherence demotion, and post-restart
correctness preservation under a SIGKILL crash.

This is an integration harness: it spawns a real swarndb binary, drives
load through the official Python SDK (gRPC), kills the process abruptly,
restarts it against the same data directory, and asserts the Option B
contract end-to-end.

Phases (driven by --mode):

    happy_path       Full sequence: workload, baseline, SIGKILL, restart,
                     /readyz < 60s, search correctness, scheduler invariants,
                     wal_meta coherence.
    g2_demotion      Same workload + SIGKILL, but graph.delta is removed
                     while hnsw.delta is preserved before restart; the
                     server must demote to FullRebuild (and /readyz still
                     comes up; correctness checks run against the rebuilt
                     index).
    all              Run both modes back-to-back (each with its own data
                     directory, so the runs are independent).

Usage (P11 will run this against Civo with 1M x 1536):

    python test_recovery_at_scale.py \\
        --mode happy_path \\
        --dataset /data/dbpedia_1m.fvecs \\
        --dim 1536 \\
        --vectors 1000000 \\
        --binary /usr/local/bin/swarndb \\
        --data-dir /var/lib/swarndb/recovery_harness \\
        --rest-port 18080 \\
        --grpc-port 18081

For faster local iteration, --vectors 100000 works against the same
harness (Option B's contract is dataset-size agnostic; only the timing
thresholds tighten at 1M).

Exit code 0 on every assertion passing; non-zero on the first failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import psutil
import requests

# The SDK lives in-tree under sdk/python/src. Make it importable so the
# harness can run from a checkout without a wheel install.
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("recovery_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and tunables (Option B contract)
# ---------------------------------------------------------------------------

COLLECTION_NAME = "recovery_harness"
DISTANCE_METRIC = "cosine"

# /readyz contract: 60s SLA at 1M, plus a 5s startup buffer to absorb
# process spawn + port-bind latency.
READYZ_DEADLINE_SECONDS = 65.0
READYZ_POLL_INTERVAL_SECONDS = 0.5

# Baseline search sample. K_BASELINE_QUERIES is large enough that a
# silent regression in graph topology shows up as an aggregate top-K
# mismatch; K_TOPK is the search depth per query.
K_BASELINE_QUERIES = 50
K_TOPK = 10

# Recovery-path discrimination by wall-clock timing.
# Option B Step-4 contract: IncrementalReplay completes in <= ~40s at 1M.
# FullRebuild at 1M takes >= ~70 min (~4200s). The mid-band 65..120s window
# is the failure surface we want to flag: anything in that window means
# Option B is silently degrading toward rebuild behavior.
INCREMENTAL_REPLAY_MAX_SECONDS = 65.0   # readyz deadline (inclusive)
FULL_REBUILD_MIN_SECONDS = 1200.0       # 20 min, conservative lower bound

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

# Workload pacing. The bulk_insert RPC is streaming, so a single call
# carries the entire payload; BATCH_SIZE controls the in-memory slice
# size when generating vectors so the harness does not blow up RAM on
# the test box.
WORKLOAD_BATCH_SIZE = 2_000

# Scheduler cadence threshold (mirrors Option B config defaults):
# the scheduler fires at 25k mutations or 120s, whichever comes first.
# The harness asserts that hnsw.base mtime advances at least once before
# the SIGKILL, which requires the workload to either exceed 25k inserts
# OR remain live for more than 120s.
SNAPSHOT_MUTATION_THRESHOLD = 25_000
SNAPSHOT_TIME_INTERVAL_SECONDS = 120.0

# Bulk-insert tuning: defer_graph + parallel_build + skip_metadata_index
# matches the P03 "fast ingest" path; the harness flushes everything via
# optimize() before the kill so the on-disk state is realistic.
BULK_INSERT_BATCH_LOCK_SIZE = 2_000
BULK_INSERT_DEFER_GRAPH = False
BULK_INSERT_INDEX_MODE: Optional[str] = None  # use server default

# Tolerance for post-recovery top-K comparison. HNSW is approximate, so a
# strict equality check is too tight; recall@K_TOPK >= 0.95 across the
# baseline query set is the Option B correctness contract. The tolerance
# is applied per query (overlap_ratio) and aggregated (mean overlap).
PER_QUERY_OVERLAP_FLOOR = 0.7
AGGREGATE_OVERLAP_FLOOR = 0.95

# Post-restart RSS sampling (P10.8 M3). After /readyz returns 200, the
# runtime needs a settle window to release transient restart-time
# buffers before the rest-state working set is observable. The default
# settle window matches the phase-file spec (30s). The sampling cost
# coefficient matches the P09 rest-state formula (~9 KB/vec), floored
# to avoid pathological small-N runs.
POST_RESTART_SETTLE_SECONDS = 30.0
POST_RESTART_SAMPLE_COUNT = 5
POST_RESTART_SAMPLE_INTERVAL_SECONDS = 1.0
POST_RESTART_COST_GIB_PER_VEC = 9.0e-6  # 9 microGiB / vec = ~9 KB/vec
POST_RESTART_CEILING_FLOOR_GIB = 0.5


# ---------------------------------------------------------------------------
# Result container
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


# ---------------------------------------------------------------------------
# Dataset loader (fvecs + raw f32; falls back to deterministic synthetic)
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
) -> List[List[float]]:
    """Load the workload vectors into RAM.

    The harness keeps all vectors in memory so the SIGKILL/restart cycle
    does not have to re-read the dataset; the dataset path is the
    bottleneck for 1M loads and we only want to pay it once.
    """
    if dataset is None:
        logger.warning(
            "No --dataset provided; using deterministic synthetic vectors. "
            "This path is for harness self-test only; P11 must pass --dataset."
        )
        return list(_iter_synthetic(dim, count))

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
# Process management
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

    def sigkill(self) -> int:
        """SIGKILL the process; returns the PID that was killed."""
        if self.proc is None:
            raise RuntimeError("no swarndb process to kill")
        pid = self.proc.pid
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            logger.warning("process %d already gone before SIGKILL", pid)
        deadline = time.time() + PROCESS_KILL_WAIT_SECONDS
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        if self.proc.poll() is None:
            raise RuntimeError(
                f"swarndb pid {pid} did not exit within "
                f"{PROCESS_KILL_WAIT_SECONDS}s of SIGKILL"
            )
        self._close_log()
        return pid

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
# Post-restart RSS sampling (P10.8 M3)
# ---------------------------------------------------------------------------


def compute_post_restart_ceiling_gib(n_vectors: int) -> float:
    """Pick the post-restart RSS ceiling for n_vectors.

    Mirrors the P09 rest-state formula (~9 KB/vec, floored). At 200k
    the ceiling is 1.8 GiB; at 1M it is 9.0 GiB. The post-restart RSS
    should land near the rest-state ceiling because the restart-settle
    window covers transient replay buffers and the working set
    converges to the same per-vector cost as a steady-state ingest.
    """
    raw = POST_RESTART_COST_GIB_PER_VEC * float(n_vectors)
    ceiling = max(POST_RESTART_CEILING_FLOOR_GIB, raw)
    return round(ceiling, 1)


def sample_post_restart_rss_gib(
    pid: int,
    n_samples: int,
    interval_seconds: float,
) -> float:
    """Sample RSS via psutil on the restarted process; return the median.

    Median over mean discounts transient spikes from scheduler-driven
    snapshot writes that may fire shortly after restart. Raises
    RuntimeError if the PID is gone (the harness then reports a clear
    failure instead of silently emitting a misleading zero).
    """
    samples: List[float] = []
    proc = psutil.Process(pid)
    for i in range(n_samples):
        try:
            rss_bytes = proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            raise RuntimeError(f"psutil RSS sample failed: {exc}")
        rss_gib = rss_bytes / (1024.0 ** 3)
        samples.append(rss_gib)
        logger.info(
            "  post-restart rss sample %d/%d: %.3f GiB",
            i + 1, n_samples, rss_gib,
        )
        if i + 1 < n_samples:
            time.sleep(interval_seconds)
    samples.sort()
    mid = len(samples) // 2
    if len(samples) % 2 == 1:
        return samples[mid]
    return 0.5 * (samples[mid - 1] + samples[mid])


def measure_post_restart_rss(
    proc: SwarndbProcess,
    args: argparse.Namespace,
    report: HarnessReport,
    label: str,
) -> None:
    """Settle, sample, and assert post-restart RSS on the restarted process.

    Always runs as an ADDITIONAL assertion (it does not replace any of
    the existing recovery-time or correctness checks). If the user
    passed --post-restart-ceiling-gib, that value is the ceiling;
    otherwise the harness derives the ceiling from --vectors via the
    same 9 KB/vec formula the P09 rest-state contract uses.
    """
    if proc.proc is None or proc.proc.poll() is not None:
        report.add(
            f"{label}: post-restart RSS",
            False,
            "swarndb process is not alive; cannot sample RSS",
        )
        return

    pid = proc.proc.pid
    settle_seconds = float(args.post_restart_settle_seconds)
    logger.info(
        "[POST_RESTART] settling for %.1fs before RSS sample (pid=%d)",
        settle_seconds, pid,
    )
    time.sleep(settle_seconds)

    try:
        observed_gib = sample_post_restart_rss_gib(
            pid,
            POST_RESTART_SAMPLE_COUNT,
            POST_RESTART_SAMPLE_INTERVAL_SECONDS,
        )
    except RuntimeError as exc:
        report.add(
            f"{label}: post-restart RSS",
            False,
            f"sampler error: {exc}",
        )
        return

    if args.post_restart_ceiling_gib is not None:
        ceiling_gib = float(args.post_restart_ceiling_gib)
        ceiling_source = "cli_override"
    else:
        ceiling_gib = compute_post_restart_ceiling_gib(args.vectors)
        ceiling_source = "proportional_formula"

    passed = observed_gib <= ceiling_gib
    report.add(
        f"{label}: post-restart RSS",
        passed,
        (
            f"N={args.vectors} "
            f"ceiling={ceiling_gib:.2f} GiB ({ceiling_source}) "
            f"observed={observed_gib:.3f} GiB "
            f"settle={settle_seconds:.1f}s pid={pid} "
            f"verdict={'PASS' if passed else 'FAIL'}"
        ),
    )
    logger.info(
        "[POST_RESTART] observed=%.3f GiB ceiling=%.2f GiB pid=%d",
        observed_gib, ceiling_gib, pid,
    )


# ---------------------------------------------------------------------------
# /readyz polling (raw HTTP; the SDK does not expose this probe)
# ---------------------------------------------------------------------------


def wait_for_readyz(
    rest_port: int,
    deadline_seconds: float,
    started_at: float,
) -> Tuple[bool, float]:
    """Poll /readyz until 200 OK or until the deadline elapses.

    Returns (ok, elapsed_from_started_at). The deadline is measured from
    the caller-supplied start time, NOT from the first poll, so the
    elapsed value covers the full process-restart window.
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


def create_collection(client: SwarnDBClient, dim: int) -> None:
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    client.collections.create(
        COLLECTION_NAME,
        dimension=dim,
        distance_metric=DISTANCE_METRIC,
    )
    logger.info("created collection '%s' dim=%d", COLLECTION_NAME, dim)


def insert_workload(
    client: SwarnDBClient,
    vectors: List[List[float]],
) -> None:
    """Stream the entire workload through bulk_insert with metadata.

    The metadata carries a row index so the harness can correlate post
    recovery search hits back to the original input order if needed.
    """
    total = len(vectors)
    metadata_list = [{"row_idx": i} for i in range(total)]
    logger.info("bulk_insert: %d vectors (batch_lock_size=%d)",
                total, BULK_INSERT_BATCH_LOCK_SIZE)
    t0 = time.time()
    result = client.vectors.bulk_insert(
        COLLECTION_NAME,
        vectors,
        metadata_list=metadata_list,
        batch_size=WORKLOAD_BATCH_SIZE,
        batch_lock_size=BULK_INSERT_BATCH_LOCK_SIZE,
        defer_graph=BULK_INSERT_DEFER_GRAPH,
        index_mode=BULK_INSERT_INDEX_MODE,
    )
    elapsed = time.time() - t0
    rate = result.inserted_count / elapsed if elapsed > 0 else 0
    logger.info(
        "bulk_insert done: inserted=%d errors=%d in %.1fs (%.0f vec/s)",
        result.inserted_count, len(result.errors), elapsed, rate,
    )
    if result.inserted_count != total or result.errors:
        raise RuntimeError(
            f"bulk_insert short: inserted={result.inserted_count}/{total}, "
            f"errors={result.errors[:3]}"
        )


def capture_baseline_results(
    client: SwarnDBClient,
    vectors: List[List[float]],
    seed: int = 9001,
) -> List[Tuple[List[float], List[int]]]:
    """Pick K_BASELINE_QUERIES queries from the workload, record top-K IDs."""
    rng = np.random.default_rng(seed)
    total = len(vectors)
    sample_idx = rng.choice(total, size=K_BASELINE_QUERIES, replace=False)
    baseline: List[Tuple[List[float], List[int]]] = []
    for i, q_idx in enumerate(sample_idx):
        query = vectors[int(q_idx)]
        result = client.search.query(
            COLLECTION_NAME,
            query,
            k=K_TOPK,
            include_metadata=False,
            include_graph=False,
        )
        ids = [r.id for r in result.results]
        baseline.append((query, ids))
        if (i + 1) % 10 == 0:
            logger.info("baseline: captured %d/%d queries",
                        i + 1, K_BASELINE_QUERIES)
    return baseline


def compare_post_recovery_search(
    client: SwarnDBClient,
    baseline: List[Tuple[List[float], List[int]]],
) -> Tuple[bool, bool, float, int]:
    """Re-run baseline queries; compare top-K overlap.

    Returns (per_query_ok, aggregate_ok, mean_overlap, failed_query_count).
    A per-query failure is overlap < PER_QUERY_OVERLAP_FLOOR.
    An aggregate failure is mean overlap < AGGREGATE_OVERLAP_FLOOR.
    """
    overlaps: List[float] = []
    failed_queries = 0
    for query, baseline_ids in baseline:
        result = client.search.query(
            COLLECTION_NAME,
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
# Filesystem inspection (scheduler + wal_meta invariants)
# ---------------------------------------------------------------------------


def collection_dir(data_dir: Path) -> Path:
    return data_dir / COLLECTION_NAME


def wait_for_scheduler_snapshot(
    data_dir: Path,
    workload_started_at: float,
    deadline_seconds: float,
) -> Tuple[bool, Optional[float]]:
    """Block until hnsw.base mtime advances past workload_started_at.

    Option B contract: the scheduler must fire at least once during the
    workload window (25k mutations / 120s, whichever comes first). The
    harness waits up to deadline_seconds for hnsw.base to exist with an
    mtime AFTER the workload start; this proves the scheduler ran end
    to end, not just a stale snapshot from a prior run.

    Returns (ok, mtime_seen_or_None).
    """
    hnsw_base = collection_dir(data_dir) / "hnsw.base"
    deadline = time.time() + deadline_seconds
    seen_mtime: Optional[float] = None
    while time.time() < deadline:
        if hnsw_base.is_file():
            mtime = hnsw_base.stat().st_mtime
            if mtime >= workload_started_at:
                return True, mtime
            seen_mtime = mtime
        time.sleep(1.0)
    return False, seen_mtime


def read_wal_meta(data_dir: Path) -> Optional[dict]:
    path = collection_dir(data_dir) / "wal_meta.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("failed to parse wal_meta.json: %s", exc)
        return None


def assert_no_shutdown_marker(data_dir: Path, report: HarnessReport) -> None:
    """After SIGKILL there must be no shutdown_clean marker on disk.

    This catches the failure case where some path graceful-shut despite
    the kill and the recovery would take the wrong arm.
    """
    marker = collection_dir(data_dir) / "shutdown_clean"
    exists = marker.is_file()
    report.add(
        "no shutdown_clean marker post-SIGKILL",
        not exists,
        f"marker_path={marker} exists={exists}",
    )


def delete_graph_delta_preserving_hnsw_delta(
    data_dir: Path, report: HarnessReport,
) -> None:
    """G2 demotion setup: remove graph.delta while keeping hnsw.delta.

    Option B contract: on restart, the planner must detect the delta
    skew (hnsw.delta present, graph.delta missing, graph.base present)
    and demote IncrementalReplay to FullRebuild rather than silently
    loading a stale graph.
    """
    cdir = collection_dir(data_dir)
    hnsw_delta = cdir / "hnsw.delta"
    graph_delta = cdir / "graph.delta"
    graph_base = cdir / "graph.base"

    pre_state = {
        "hnsw.delta": hnsw_delta.is_file(),
        "graph.delta": graph_delta.is_file(),
        "graph.base": graph_base.is_file(),
    }
    logger.info("G2 setup pre-state: %s", pre_state)

    if not hnsw_delta.is_file():
        report.add(
            "G2 precondition: hnsw.delta present before kill",
            False,
            "hnsw.delta missing; cannot exercise G2 skew",
        )
        return

    if graph_delta.is_file():
        try:
            graph_delta.unlink()
        except OSError as exc:
            report.add(
                "G2 setup: delete graph.delta",
                False,
                f"unlink failed: {exc}",
            )
            return

    report.add(
        "G2 setup: graph.delta deleted, hnsw.delta preserved",
        not graph_delta.is_file() and hnsw_delta.is_file(),
        f"hnsw.delta={hnsw_delta.is_file()} graph.delta={graph_delta.is_file()}",
    )


# ---------------------------------------------------------------------------
# Top-level modes
# ---------------------------------------------------------------------------


def run_happy_path(args: argparse.Namespace) -> bool:
    """Workload, baseline, SIGKILL, restart, IncrementalReplay assertions."""
    report = HarnessReport("happy_path")
    data_dir = Path(args.data_dir).resolve()
    log_path = Path(args.log_dir).resolve() / "swarndb_happy_path.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean slate: the harness owns this data dir.
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    proc = SwarndbProcess(
        binary=Path(args.binary),
        data_dir=data_dir,
        rest_port=args.rest_port,
        grpc_port=args.grpc_port,
        log_path=log_path,
    )

    try:
        proc.start()
        ok, elapsed = wait_for_readyz(args.rest_port, 60.0, time.time())
        if not ok:
            report.add(
                "initial /readyz",
                False,
                f"server failed to come up within 60s (elapsed={elapsed:.1f}s)",
            )
            logger.error(report.summary())
            return False

        client = make_client(args.grpc_port)
        try:
            create_collection(client, args.dim)

            # Workload phase.
            logger.info("loading %d vectors from dataset", args.vectors)
            vectors = load_vectors(
                Path(args.dataset) if args.dataset else None,
                args.dim,
                args.vectors,
            )
            workload_started_at = time.time()
            insert_workload(client, vectors)

            # Wait for the scheduler to fire at least once.
            scheduler_deadline = max(
                SNAPSHOT_TIME_INTERVAL_SECONDS * 2.0,
                60.0,
            )
            logger.info(
                "waiting up to %.0fs for scheduler-written hnsw.base",
                scheduler_deadline,
            )
            sched_ok, sched_mtime = wait_for_scheduler_snapshot(
                data_dir, workload_started_at, scheduler_deadline,
            )
            report.add(
                "scheduler wrote hnsw.base during workload",
                sched_ok,
                f"hnsw.base mtime={sched_mtime} workload_started_at={workload_started_at}",
            )

            # Sample wal_meta pre-kill so we can assert post-kill coherence.
            pre_kill_meta = read_wal_meta(data_dir)
            report.add(
                "wal_meta.json present pre-kill with non-zero last_snapshot_lsn",
                pre_kill_meta is not None
                and pre_kill_meta.get("last_snapshot_lsn", 0) > 0,
                f"pre_kill_meta={pre_kill_meta}",
            )

            # Capture baseline search results.
            logger.info("capturing %d baseline queries", K_BASELINE_QUERIES)
            baseline = capture_baseline_results(client, vectors)
        finally:
            client.close()

        # SIGKILL phase.
        pid = proc.sigkill()
        logger.info("SIGKILLed swarndb pid=%d", pid)
        assert_no_shutdown_marker(data_dir, report)

        # Restart phase + readyz timing.
        restart_started_at = time.time()
        proc.start()
        ok, ready_elapsed = wait_for_readyz(
            args.rest_port,
            READYZ_DEADLINE_SECONDS,
            restart_started_at,
        )
        report.add(
            "post-SIGKILL /readyz within 65s",
            ok,
            f"elapsed={ready_elapsed:.1f}s deadline={READYZ_DEADLINE_SECONDS}s",
        )

        # Recovery path discrimination by wall-clock.
        # Option B IncrementalReplay must finish well under FullRebuild
        # time at 1M; anything above INCREMENTAL_REPLAY_MAX_SECONDS but
        # still under FullRebuild lower bound is the failure surface.
        if ok and ready_elapsed <= INCREMENTAL_REPLAY_MAX_SECONDS:
            recovery_path = "IncrementalReplay"
            recovery_ok = True
            recovery_detail = (
                f"elapsed={ready_elapsed:.1f}s (<= {INCREMENTAL_REPLAY_MAX_SECONDS}s "
                f"=> IncrementalReplay)"
            )
        elif ready_elapsed >= FULL_REBUILD_MIN_SECONDS:
            recovery_path = "FullRebuild"
            recovery_ok = False
            recovery_detail = (
                f"elapsed={ready_elapsed:.1f}s (>= {FULL_REBUILD_MIN_SECONDS}s "
                f"=> FullRebuild; Option B regressed)"
            )
        else:
            recovery_path = "Indeterminate"
            recovery_ok = False
            recovery_detail = (
                f"elapsed={ready_elapsed:.1f}s sits in the mid-band "
                f"({INCREMENTAL_REPLAY_MAX_SECONDS}s..{FULL_REBUILD_MIN_SECONDS}s); "
                f"Option B silently degrading"
            )
        report.add(
            "recovery path classified as IncrementalReplay",
            recovery_ok,
            f"path={recovery_path} {recovery_detail}",
        )

        if not ok:
            logger.error(report.summary())
            return False

        # Post-recovery correctness assertion.
        client = make_client(args.grpc_port)
        try:
            info = client.collections.get(COLLECTION_NAME)
            report.add(
                "post-recovery vector count matches workload",
                info.vector_count == args.vectors,
                f"expected={args.vectors} got={info.vector_count}",
            )

            per_q, aggr, mean_overlap, failed_q = compare_post_recovery_search(
                client, baseline,
            )
            report.add(
                "per-query top-K overlap above floor",
                per_q,
                f"failed_queries={failed_q}/{K_BASELINE_QUERIES} "
                f"floor={PER_QUERY_OVERLAP_FLOOR}",
            )
            report.add(
                "aggregate mean top-K overlap above floor",
                aggr,
                f"mean_overlap={mean_overlap:.3f} floor={AGGREGATE_OVERLAP_FLOOR}",
            )
        finally:
            client.close()

        # wal_meta coherence: last_snapshot_lsn must survive the cycle
        # AND match (or exceed) the pre-kill value (scheduler may have
        # fired again post-restart; equality or monotonic-increase is OK).
        post_kill_meta = read_wal_meta(data_dir)
        pre_lsn = (pre_kill_meta or {}).get("last_snapshot_lsn", 0)
        post_lsn = (post_kill_meta or {}).get("last_snapshot_lsn", 0)
        report.add(
            "wal_meta.last_snapshot_lsn survived SIGKILL + restart",
            post_kill_meta is not None and post_lsn >= pre_lsn and post_lsn > 0,
            f"pre_lsn={pre_lsn} post_lsn={post_lsn}",
        )

        # Scheduler invariant: hnsw.base still present, mtime within
        # the workload window (not a stale file from a prior run).
        hnsw_base = collection_dir(data_dir) / "hnsw.base"
        if hnsw_base.is_file():
            mtime = hnsw_base.stat().st_mtime
            report.add(
                "hnsw.base survived SIGKILL with workload-window mtime",
                mtime >= workload_started_at,
                f"mtime={mtime} workload_started_at={workload_started_at}",
            )
        else:
            report.add(
                "hnsw.base survived SIGKILL with workload-window mtime",
                False,
                "hnsw.base missing post-recovery",
            )

        # Post-restart RSS (P10.8 M3). Sampled on the restarted process
        # PID after a settle window. Reported as an ADDITIONAL assertion
        # alongside the existing recovery-time and correctness checks.
        measure_post_restart_rss(proc, args, report, "happy_path")

        logger.info(report.summary())
        return report.all_passed()

    finally:
        proc.terminate()


def run_g2_demotion(args: argparse.Namespace) -> bool:
    """Same flow as happy_path, but graph.delta is removed before restart.

    Option B contract: planner detects the delta skew and demotes to
    FullRebuild. The harness asserts both that the recovery eventually
    succeeds (correctness restored) AND that the elapsed time crossed
    the IncrementalReplay band (or, on smaller workloads where rebuild
    is fast, simply that the server came up and search works).
    """
    report = HarnessReport("g2_demotion")
    data_dir = Path(args.data_dir).resolve()
    log_path = Path(args.log_dir).resolve() / "swarndb_g2_demotion.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    proc = SwarndbProcess(
        binary=Path(args.binary),
        data_dir=data_dir,
        rest_port=args.rest_port,
        grpc_port=args.grpc_port,
        log_path=log_path,
    )

    try:
        proc.start()
        ok, _ = wait_for_readyz(args.rest_port, 60.0, time.time())
        if not ok:
            report.add(
                "initial /readyz",
                False,
                "server failed to come up within 60s",
            )
            logger.error(report.summary())
            return False

        client = make_client(args.grpc_port)
        try:
            create_collection(client, args.dim)
            logger.info("loading %d vectors from dataset", args.vectors)
            vectors = load_vectors(
                Path(args.dataset) if args.dataset else None,
                args.dim,
                args.vectors,
            )
            workload_started_at = time.time()
            insert_workload(client, vectors)

            sched_deadline = max(SNAPSHOT_TIME_INTERVAL_SECONDS * 2.0, 60.0)
            sched_ok, _ = wait_for_scheduler_snapshot(
                data_dir, workload_started_at, sched_deadline,
            )
            report.add(
                "G2 precondition: scheduler wrote hnsw.base + graph.base",
                sched_ok and (collection_dir(data_dir) / "graph.base").is_file(),
                f"sched_ok={sched_ok} graph.base="
                f"{(collection_dir(data_dir) / 'graph.base').is_file()}",
            )

            baseline = capture_baseline_results(client, vectors)
        finally:
            client.close()

        # SIGKILL, then manipulate disk state to engineer the G2 skew.
        proc.sigkill()
        assert_no_shutdown_marker(data_dir, report)
        delete_graph_delta_preserving_hnsw_delta(data_dir, report)

        # Restart with no upper-bound deadline; FullRebuild on a real
        # 1M workload can take an hour. The harness allows that long
        # window deliberately, because the assertion under test is "did
        # the planner demote and recover correctly," not "fast restart."
        restart_started_at = time.time()
        proc.start()
        # Use a generous deadline (15 min) for the smaller-scale runs;
        # P11 at 1M may exceed this and that is intentional skip-able.
        ok, elapsed = wait_for_readyz(
            args.rest_port,
            float(args.g2_readyz_deadline_seconds),
            restart_started_at,
        )
        report.add(
            "G2 path: /readyz returns 200 after demoted recovery",
            ok,
            f"elapsed={elapsed:.1f}s deadline="
            f"{args.g2_readyz_deadline_seconds}s",
        )
        if not ok:
            logger.error(report.summary())
            return False

        # The recovery elapsed time should NOT sit in the IncrementalReplay
        # band: a fast restart here would mean the demotion did not happen
        # and the server silently loaded a stale graph.
        report.add(
            "G2 path: elapsed exceeded IncrementalReplay band (demotion observed)",
            elapsed > INCREMENTAL_REPLAY_MAX_SECONDS,
            f"elapsed={elapsed:.1f}s threshold={INCREMENTAL_REPLAY_MAX_SECONDS}s",
        )

        # Correctness assertion: top-K must still recover within tolerance
        # against the baseline (rebuild produces a topologically distinct
        # graph but the nearest neighbors of the same query vectors are
        # the same vectors, so overlap should remain high).
        client = make_client(args.grpc_port)
        try:
            info = client.collections.get(COLLECTION_NAME)
            report.add(
                "G2 path: vector count matches workload",
                info.vector_count == args.vectors,
                f"expected={args.vectors} got={info.vector_count}",
            )
            per_q, aggr, mean_overlap, failed_q = compare_post_recovery_search(
                client, baseline,
            )
            report.add(
                "G2 path: per-query top-K overlap above floor",
                per_q,
                f"failed_queries={failed_q}/{K_BASELINE_QUERIES} "
                f"floor={PER_QUERY_OVERLAP_FLOOR}",
            )
            report.add(
                "G2 path: aggregate mean top-K overlap above floor",
                aggr,
                f"mean_overlap={mean_overlap:.3f} floor={AGGREGATE_OVERLAP_FLOOR}",
            )
        finally:
            client.close()

        # Post-restart RSS (P10.8 M3) on the FullRebuild path. The
        # demoted recovery still settles to a working set proportional
        # to vector count, so the same ceiling formula applies.
        measure_post_restart_rss(proc, args, report, "g2_demotion")

        logger.info(report.summary())
        return report.all_passed()

    finally:
        proc.terminate()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SwarnDB recovery-at-scale validation harness "
            "(Perf_Stability P07 Step 5 / P11 execution). "
            "NOTE: This harness must spawn its own swarndb-server "
            "because the test inherently kills+restarts the process. "
            "External-server mode is not supported. Always pass "
            "--binary."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["happy_path", "g2_demotion", "all"],
        default="happy_path",
        help="Which assertion harness to run.",
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
                        help="Number of vectors to load (default 1_000_000).")
    parser.add_argument(
        "--binary",
        default=os.environ.get("SWARNDB_BINARY"),
        help=(
            "Path to the swarndb binary. Required: recovery testing "
            "requires the harness to own the swarndb-server process "
            "(kill + restart). External-server mode is not supported "
            "for this harness."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get(
            "SWARNDB_HARNESS_DATA_DIR",
            "/tmp/swarndb_recovery_harness",
        ),
        help="Data directory the harness will own (wiped on each run).",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get(
            "SWARNDB_HARNESS_LOG_DIR",
            "/tmp/swarndb_recovery_harness_logs",
        ),
        help="Directory for swarndb stdout/stderr capture.",
    )
    parser.add_argument("--rest-port", type=int, default=18080,
                        help="REST port for /readyz polling.")
    parser.add_argument("--grpc-port", type=int, default=18081,
                        help="gRPC port for the SDK client.")
    parser.add_argument(
        "--g2-readyz-deadline-seconds",
        type=int,
        default=900,
        help=(
            "Upper bound for the FullRebuild recovery on the G2 mode. "
            "Default 900s (15 min) suits up to ~250k; bump for 1M runs."
        ),
    )
    parser.add_argument(
        "--post-restart-ceiling-gib",
        type=float,
        default=None,
        help=(
            "Override the post-restart RSS ceiling in GiB. If omitted, "
            "the harness derives the ceiling from --vectors using the "
            "P09 proportional formula (9 KB/vec); 1.8 GiB at 200k and "
            "9.0 GiB at 1M."
        ),
    )
    parser.add_argument(
        "--post-restart-settle-seconds",
        type=float,
        default=POST_RESTART_SETTLE_SECONDS,
        help=(
            "Settle window after /readyz=200 before the post-restart "
            "RSS sample. Default 30s so the runtime can release "
            "transient replay buffers before measurement."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if not args.binary:
        logger.error(
            "test_recovery_at_scale.py requires --binary. Recovery "
            "testing requires the harness to own the swarndb-server "
            "process (kill + restart). External-server mode is not "
            "supported for this harness."
        )
        return 2

    if args.mode == "happy_path":
        ok = run_happy_path(args)
    elif args.mode == "g2_demotion":
        ok = run_g2_demotion(args)
    else:
        # 'all' runs both with independent data dirs to keep state isolated.
        base_data_dir = Path(args.data_dir)
        base_log_dir = Path(args.log_dir)
        args.data_dir = str(base_data_dir / "happy_path")
        args.log_dir = str(base_log_dir / "happy_path")
        hp_ok = run_happy_path(args)

        args.data_dir = str(base_data_dir / "g2_demotion")
        args.log_dir = str(base_log_dir / "g2_demotion")
        g2_ok = run_g2_demotion(args)
        ok = hp_ok and g2_ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
