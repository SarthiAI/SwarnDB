#!/usr/bin/env python3
"""
SwarnDB memory-release validation harness (Perf_Stability P09 Step 4).

Exercises the rest-state memory release contract that the P06..P09
work enforces: after a heavy ingest burst, the SwarnDB process must
shed the transient build-time RSS spike and settle to a rest-state
working set that is proportional to vector count, not to the peak.

This is an integration harness: it spawns a real swarndb binary,
drives load through the official Python SDK (gRPC), measures resident
set size via psutil, and asserts:

    1. Rest-state RSS after a full ingest settles below a budget that
       scales linearly with n_vectors (about 9 KB/vec; lands on 1.8
       GiB at 200k and 9 GiB at 1M).
    2. The per-vector slope of RSS-versus-vector-count is monotone
       non-increasing across checkpoints, with a small noise slack.
       Translation: each successive batch must not cost more memory
       per vector than the prior batch did. This catches a quiet
       regression where the rest-state cost creeps up with scale.

Modes (driven by --mode):

    rest_state         Insert n_vectors in one stretch, wait for the
                       rest state to settle, sample RSS, compare
                       against the proportional ceiling.
    per_vector_slope   Insert in increasing checkpoints; at each
                       checkpoint sample RSS; assert per-vector slope
                       is monotone non-increasing.
    all                Run rest_state, then per_vector_slope, each
                       against its own swarndb process and data dir.

Usage (local Step 4 development uses 200k):

    python test_memory_release.py \\
        --binary-path /usr/local/bin/swarndb \\
        --n-vectors 200000 \\
        --dimension 1536 \\
        --mode all

The P11 regression gate invokes the harness with
`--n-vectors 1000000` against the Civo dbpedia_1m collection so the
rest-state assertion runs at the real production size, not
extrapolated. Local Step 4 does NOT run at 1M; the 1M invocation is
P11-deferred.

Exit code 0 on every assertion passing; non-zero on the first
failure.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import psutil
import requests

# The SDK lives in-tree under sdk/python/src. Make it importable so the
# harness can run from a checkout without a wheel install. Mirrors the
# pattern used in tests/test_recovery_at_scale.py (P07 harness) and
# tests/test_concurrent_search_under_write.py (P08 harness).
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("memory_release_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and tunables (P09 contract)
# ---------------------------------------------------------------------------

# /readyz contract: 60s SLA at 1M plus a 5s startup buffer.
READYZ_DEADLINE_SECONDS = 65.0
READYZ_POLL_INTERVAL_SECONDS = 0.5

# Default ports kept distinct from the P07 / P08 harnesses so the three
# can coexist on the same dev box.
DEFAULT_REST_PORT = 18090
DEFAULT_GRPC_PORT = 18091

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

# Rest-state settle window. After bulk_insert returns, the runtime
# still needs a beat to release transient build buffers; sample the
# rest state only after this settle period.
REST_STATE_SETTLE_SECONDS = 5.0
REST_STATE_SAMPLE_COUNT = 5
REST_STATE_SAMPLE_INTERVAL_SECONDS = 1.0

# Slope-mode sampling. Smaller settle than rest_state because we are
# looking for a relative trend across checkpoints, not an absolute
# rest-state ceiling.
SLOPE_SETTLE_SECONDS = 3.0
SLOPE_SAMPLE_COUNT = 3
SLOPE_SAMPLE_INTERVAL_SECONDS = 1.0

# Slope-monotonicity noise slack. RSS sampling carries measurement
# noise, especially across checkpoints that include scheduler-driven
# snapshot writes. A later slope is allowed to exceed the prior one by
# up to SLOPE_NOISE_SLACK before flagging a regression.
SLOPE_NOISE_SLACK = 0.10  # 10%

# Per-vector cost coefficient for the rest-state ceiling. Derived
# from the P09 contract: ~9 KB/vec rest overhead lands on 1.8 GiB at
# 200k and 9 GiB at 1M, giving a clean proportional ceiling.
REST_STATE_COST_GIB_PER_VEC = 9.0e-6  # 9 microGiB / vec = ~9 KB/vec
REST_STATE_CEILING_FLOOR_GIB = 0.5    # avoid pathological small-N runs

# Peak-RSS sampler (P10.8 M1). Polls RSS at a coarse 200ms interval
# during the bulk insert so we capture the load-time spike without
# adding measurable CPU overhead to the insert path. The peak ceiling
# defaults to a multiple of the rest-state ceiling: the load-time
# spike is bounded but is allowed to be substantially larger than the
# rest-state working set (the spike covers transient build buffers,
# graph construction scratch, and intermediate batches).
PEAK_RSS_POLL_INTERVAL_SECONDS = 0.2
PEAK_RSS_DEFAULT_CEILING_MULTIPLIER = 3.0

# Collection / workload defaults.
DEFAULT_COLLECTION_NAME = "p09_memrelease_test"
DEFAULT_DISTANCE_METRIC = "cosine"
DEFAULT_BATCH_SIZE = 10_000

# Bulk-insert tuning matching the other harnesses.
BULK_INSERT_BATCH_LOCK_SIZE = 2_000
BULK_INSERT_DEFER_GRAPH = False
BULK_INSERT_INDEX_MODE: Optional[str] = None  # server default


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


# ---------------------------------------------------------------------------
# Rest-state ceiling logic
# ---------------------------------------------------------------------------


def compute_rest_state_ceiling_gib(n_vectors: int) -> float:
    """Pick the rest-state RSS ceiling for n_vectors.

    Linear in n_vectors at ~9 KB/vec, floored to avoid pathological
    small-N runs. Verification: at 200_000 vectors the ceiling lands at
    1.8 GiB; at 1_000_000 vectors it lands at 9.0 GiB. Both match the
    P09 phase-file contract (5.86 GiB raw + 3 GiB graph + slack at 1M,
    proportional scaling below).
    """
    raw = REST_STATE_COST_GIB_PER_VEC * float(n_vectors)
    ceiling = max(REST_STATE_CEILING_FLOOR_GIB, raw)
    return round(ceiling, 1)


def default_slope_checkpoints(n_vectors: int) -> List[int]:
    """Pick checkpoints for the slope-mode insertion sequence.

    At 200k: [100k, 150k, 200k].
    At 1M:   [100k, 150k, 200k, 250k, 500k, 750k, 1M].
    Otherwise: scale the 1M pattern proportionally and clamp to the
    requested n_vectors so the last checkpoint always equals n_vectors.
    """
    if n_vectors <= 200_000:
        base = [100_000, 150_000, 200_000]
    elif n_vectors >= 1_000_000:
        base = [100_000, 150_000, 200_000, 250_000, 500_000, 750_000, 1_000_000]
    else:
        # Scale the 1M pattern proportionally for arbitrary mid-range n.
        scale = n_vectors / 1_000_000.0
        base = [int(round(v * scale)) for v in
                (100_000, 150_000, 200_000, 250_000, 500_000, 750_000, 1_000_000)]

    # Clamp to n_vectors, drop dups, drop anything above n_vectors,
    # ensure the last checkpoint equals n_vectors exactly.
    clamped = sorted({min(c, n_vectors) for c in base if c > 0})
    if not clamped or clamped[-1] != n_vectors:
        clamped.append(n_vectors)
    return sorted(set(clamped))


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


class SwarndbProcess:
    """Spawn-and-supervise wrapper around the swarndb binary.

    Mirrors the SwarndbProcess class in test_recovery_at_scale.py
    (P07). Kept local so each harness owns its lifecycle without a
    shared utility module.
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


# Short timeout for the auto-detect probe; an existing local server
# answers /readyz in milliseconds. A long timeout would slow spawn-mode
# users who have no server up on the port (the connect refusal is
# instant; only DNS-style stalls are bounded by this).
EXTERNAL_PROBE_TIMEOUT_SECONDS = 2.0


def find_or_spawn_server(
    args: argparse.Namespace,
    log: logging.Logger,
    proc_factory,
) -> Tuple[str, str, Optional[SwarndbProcess], Optional[int]]:
    """Auto-detect a swarndb-server on args.rest_port; else spawn one.

    Returns (rest_url, grpc_url, spawned_process_or_None, pid_for_rss).

    Auto-detect path: if GET /readyz returns 200 on args.rest_port, the
    harness reuses that server and returns spawned_process_or_None=None
    so teardown skips a process the harness does not own. The PID used
    for RSS sampling comes from args.external_pid; if the user did not
    pass --external-pid the harness errors out before any test work
    starts (host-PID of a docker container is the containerd-shim, not
    swarndb).

    Spawn path: if /readyz is unreachable AND --binary-path is provided,
    the harness calls proc_factory() to build a SwarndbProcess against
    the mode's resolved data_dir, starts it, waits for /readyz, and
    returns the Popen wrapper plus its PID for RSS sampling.

    Failure modes (all exit non-zero):
      * external detected but --external-pid missing
      * external detected but --external-pid points at a dead process
      * --external-pid provided but no external server up
      * no external server up and --binary-path missing
      * spawned server does not pass /readyz inside the deadline
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
        if args.external_pid is None:
            log.error(
                "External swarndb-server detected on port %d, but "
                "--external-pid was not provided. RSS measurement "
                "requires the swarndb-server PID. In Docker, run "
                "`docker top swarndb` to find it; in docker-compose, "
                "run `docker inspect --format '{{.State.Pid}}' swarndb`.",
                args.rest_port,
            )
            sys.exit(2)
        try:
            psutil.Process(args.external_pid)
        except psutil.NoSuchProcess:
            log.error(
                "External swarndb-server detected on port %d, but "
                "--external-pid=%d does not refer to a live process. "
                "Re-check the PID via `docker top swarndb` or "
                "`docker inspect --format '{{.State.Pid}}' swarndb`.",
                args.rest_port, args.external_pid,
            )
            sys.exit(2)
        log.info(
            "Found existing swarndb-server on %s; using it "
            "(skipping spawn and teardown). RSS sampled from "
            "--external-pid=%d.",
            rest_url, args.external_pid,
        )
        return rest_url, grpc_url, None, args.external_pid

    if args.external_pid is not None:
        log.error(
            "--external-pid=%d was provided, but no swarndb-server "
            "responded on %s/readyz. Either start the external server "
            "first OR drop --external-pid to spawn a fresh server.",
            args.external_pid, rest_url,
        )
        sys.exit(2)

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
    return rest_url, grpc_url, proc, proc.pid()


# ---------------------------------------------------------------------------
# /readyz polling
# ---------------------------------------------------------------------------


def wait_for_readyz(
    rest_port: int,
    deadline_seconds: float,
    started_at: float,
) -> Tuple[bool, float]:
    """Poll /readyz until 200 OK or until the deadline elapses.

    Returns (ok, elapsed_from_started_at). Mirrors the helper used in
    the P07 and P08 harnesses.
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


def ensure_collection(
    client: SwarnDBClient,
    name: str,
    dimension: int,
) -> None:
    """Drop-and-create the collection so each run starts clean."""
    if client.collections.exists(name):
        client.collections.delete(name)
    client.collections.create(
        name,
        dimension=dimension,
        distance_metric=DEFAULT_DISTANCE_METRIC,
    )
    logger.info("created collection '%s' dim=%d", name, dimension)


def bulk_insert_slice(
    client: SwarnDBClient,
    collection: str,
    vectors,
    *,
    id_offset: int,
    batch_size: int,
    label: str,
) -> None:
    """Run a bulk_insert covering the supplied vector slice.

    Accepts either a numpy ndarray or a list-of-list. When numpy is passed,
    rows are streamed to bulk_insert in CHUNK_ROWS-sized slices so the
    harness process never holds the entire workload as Python objects
    (which costs ~10x the raw float32 footprint at 1M scale and OOMs the
    box). The numpy path is the production path; list is kept for
    backward compatibility with smaller-scale callers.
    """
    if isinstance(vectors, np.ndarray):
        total = int(vectors.shape[0])
    else:
        total = len(vectors)
    if total == 0:
        return

    # ~60 MiB per chunk at 1536 dim float32, ~600 MiB in Python list form;
    # well under any sane memory budget while still amortising the gRPC
    # roundtrip cost.
    CHUNK_ROWS = 10_000

    logger.info(
        "%s: inserting %d vectors into '%s' (offset=%d batch_size=%d)",
        label, total, collection, id_offset, batch_size,
    )
    t0 = time.time()
    inserted_count_total = 0
    errors_total = 0
    first_errors: List[str] = []

    for start in range(0, total, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, total)
        if isinstance(vectors, np.ndarray):
            # tolist() on a 10k slice converts ~60 MiB float32 to a Python
            # list-of-list; transient peak ~600 MiB which is bounded.
            chunk = vectors[start:end].tolist()
        else:
            chunk = vectors[start:end]
        chunk_metadata = [
            {"row_idx": id_offset + start + i} for i in range(end - start)
        ]
        result = client.vectors.bulk_insert(
            collection,
            chunk,
            metadata_list=chunk_metadata,
            batch_size=batch_size,
            batch_lock_size=BULK_INSERT_BATCH_LOCK_SIZE,
            defer_graph=BULK_INSERT_DEFER_GRAPH,
            index_mode=BULK_INSERT_INDEX_MODE,
        )
        inserted_count_total += result.inserted_count
        errors_total += len(result.errors)
        if not first_errors and result.errors:
            first_errors = list(result.errors[:3])
        # Release the Python-side chunk before the next iteration.
        del chunk
        del chunk_metadata

    elapsed = time.time() - t0
    rate = inserted_count_total / elapsed if elapsed > 0 else 0.0
    logger.info(
        "%s done: inserted=%d errors=%d in %.1fs (%.0f vec/s)",
        label, inserted_count_total, errors_total, elapsed, rate,
    )
    if inserted_count_total != total or errors_total > 0:
        raise RuntimeError(
            f"{label} short: inserted={inserted_count_total}/{total}, "
            f"errors={first_errors}"
        )


# ---------------------------------------------------------------------------
# Synthetic vectors
# ---------------------------------------------------------------------------


def generate_vectors(count: int, dimension: int, seed: int = 42) -> np.ndarray:
    """Deterministic synthetic vectors as a numpy float32 array.

    The harness measures RSS, not recall, so randn / float32 is enough.
    Returning numpy (not list-of-list) keeps the harness's own footprint
    near the raw float32 cost (~6 GiB at 1M x 1536); the legacy
    .tolist() conversion ballooned to ~50 GiB at the same scale and
    OOM-killed the box. bulk_insert_slice now consumes numpy directly
    and slices it into chunks before crossing the SDK boundary.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(count, dimension).astype(np.float32)


# ---------------------------------------------------------------------------
# RSS sampling via psutil
# ---------------------------------------------------------------------------


def sample_rss_gib(pid: int, n_samples: int, interval_seconds: float) -> float:
    """Take `n_samples` RSS samples `interval_seconds` apart; return the
    median in GiB. Median over mean to discount transient spikes from
    scheduler-driven snapshot writes.
    """
    samples: List[float] = []
    proc = psutil.Process(pid)
    for i in range(n_samples):
        try:
            # Single-process RSS only. If SwarnDB ever forks worker processes, switch to
            # psutil.Process(pid).memory_info().rss + children(recursive=True) aggregation.
            rss_bytes = proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            raise RuntimeError(f"psutil RSS sample failed: {exc}")
        rss_gib = rss_bytes / (1024.0 ** 3)
        samples.append(rss_gib)
        logger.info(
            "  rss sample %d/%d: %.3f GiB",
            i + 1, n_samples, rss_gib,
        )
        if i + 1 < n_samples:
            time.sleep(interval_seconds)
    samples.sort()
    mid = len(samples) // 2
    if len(samples) % 2 == 1:
        return samples[mid]
    return 0.5 * (samples[mid - 1] + samples[mid])


# ---------------------------------------------------------------------------
# Peak-RSS sampler thread (P10.8 M1)
# ---------------------------------------------------------------------------


class PeakRssSampler:
    """Background thread that polls RSS during a workload phase.

    Polls psutil.Process(pid).memory_info().rss at a coarse interval
    (default 200ms) so the load-time peak is captured without biasing
    the insert path with measurable CPU overhead. The thread is a
    daemon so an unhandled exception in the caller never strands it.

    Lifecycle:
        sampler = PeakRssSampler(pid, interval_seconds)
        sampler.start()
        ...workload runs...
        sampler.stop()
        peak_gib = sampler.peak_gib()
        n_samples = sampler.sample_count()

    Concurrency primitives:
        * threading.Event signals the sampler to stop (avoids busy
          spin and lets the thread exit promptly at end-of-phase).
        * threading.Thread (daemon=True) runs the poll loop.
        * No lock: the sampler thread is the only writer to its
          internal sample list and the peak; the caller only reads
          after .stop() returns.
    """

    def __init__(self, pid: int, interval_seconds: float) -> None:
        self._pid = pid
        self._interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_bytes: int = 0
        self._sample_count: int = 0
        self._error: Optional[BaseException] = None

    def _run(self) -> None:
        try:
            proc = psutil.Process(self._pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            self._error = exc
            return
        while not self._stop_event.is_set():
            try:
                rss = proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                self._error = exc
                return
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._sample_count += 1
            # Event.wait honours the stop signal mid-interval, so
            # tear-down latency is bounded by one poll interval.
            if self._stop_event.wait(timeout=self._interval):
                break

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("PeakRssSampler already started")
        self._thread = threading.Thread(
            target=self._run,
            name="peak-rss-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def peak_gib(self) -> float:
        return self._peak_bytes / (1024.0 ** 3)

    def sample_count(self) -> int:
        return self._sample_count

    def error(self) -> Optional[BaseException]:
        return self._error


# ---------------------------------------------------------------------------
# Mode: rest_state
# ---------------------------------------------------------------------------


def run_rest_state(args: argparse.Namespace) -> bool:
    """Insert n_vectors in one stretch; assert rest-state RSS ceiling."""
    report = HarnessReport("rest_state")

    data_dir, owns_data_dir = _resolve_data_dir(args, "rest_state")
    log_path = data_dir.parent / "swarndb_rest_state.log"

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc, pid = find_or_spawn_server(args, logger, _build_proc)

    try:
        client = make_client(args.grpc_port)
        try:
            ensure_collection(client, args.collection_name, args.dimension)

            logger.info(
                "[PHASE rest_state] generating %d vectors at dim=%d",
                args.n_vectors, args.dimension,
            )
            vectors = generate_vectors(args.n_vectors, args.dimension)

            logger.info(
                "[PHASE rest_state] bulk_insert %d vectors (batch_size=%d)",
                args.n_vectors, args.batch_size,
            )

            peak_sampling_enabled = not args.no_peak_sampling
            peak_sampler: Optional[PeakRssSampler] = None
            if peak_sampling_enabled:
                peak_sampler = PeakRssSampler(
                    pid=pid,
                    interval_seconds=PEAK_RSS_POLL_INTERVAL_SECONDS,
                )
                logger.info(
                    "[PHASE rest_state] starting peak-RSS sampler "
                    "(pid=%d interval=%.3fs)",
                    pid, PEAK_RSS_POLL_INTERVAL_SECONDS,
                )
                peak_sampler.start()

            try:
                bulk_insert_slice(
                    client,
                    args.collection_name,
                    vectors,
                    id_offset=0,
                    batch_size=args.batch_size,
                    label="rest_state_insert",
                )
            finally:
                if peak_sampler is not None:
                    peak_sampler.stop()
            # Drop the in-memory workload eagerly so the harness's own
            # process does not skew RSS sampling pressure on the box.
            del vectors

            logger.info(
                "[PHASE rest_state] settling for %.1fs before RSS sample",
                REST_STATE_SETTLE_SECONDS,
            )
            time.sleep(REST_STATE_SETTLE_SECONDS)

            try:
                observed_gib = sample_rss_gib(
                    pid,
                    REST_STATE_SAMPLE_COUNT,
                    REST_STATE_SAMPLE_INTERVAL_SECONDS,
                )
            except (psutil.NoSuchProcess, RuntimeError) as exc:
                logger.error(
                    "RSS sample failed: PID %d is gone (%s). If running "
                    "in external mode, the swarndb container likely "
                    "stopped; re-run after starting it back up.",
                    pid, exc,
                )
                return False

            if args.rest_state_budget_gib is not None:
                ceiling_gib = float(args.rest_state_budget_gib)
                ceiling_source = "cli_override"
            else:
                ceiling_gib = compute_rest_state_ceiling_gib(args.n_vectors)
                ceiling_source = "proportional_formula"

            passed = observed_gib <= ceiling_gib
            report.add(
                "rest_state",
                passed,
                (
                    f"N={args.n_vectors} "
                    f"ceiling={ceiling_gib:.2f} GiB ({ceiling_source}) "
                    f"observed={observed_gib:.3f} GiB "
                    f"verdict={'PASS' if passed else 'FAIL'}"
                ),
            )

            # Peak-RSS assertion (P10.8 M1). Only fires when the sampler
            # actually ran. If --no-peak-sampling was passed, the section
            # is skipped entirely, preserving the legacy rest-state-only
            # behaviour.
            if peak_sampler is not None:
                sampler_err = peak_sampler.error()
                peak_observed_gib = peak_sampler.peak_gib()
                peak_sample_count = peak_sampler.sample_count()

                if args.peak_ceiling_gib is not None:
                    peak_ceiling_gib = float(args.peak_ceiling_gib)
                    peak_ceiling_source = "cli_override"
                else:
                    peak_ceiling_gib = (
                        ceiling_gib * PEAK_RSS_DEFAULT_CEILING_MULTIPLIER
                    )
                    peak_ceiling_source = (
                        f"{PEAK_RSS_DEFAULT_CEILING_MULTIPLIER:.1f}x_rest_ceiling"
                    )

                retention_gap_gib = max(0.0, peak_observed_gib - observed_gib)

                if sampler_err is not None:
                    report.add(
                        "peak_rss",
                        False,
                        (
                            f"sampler error: {sampler_err}; "
                            f"samples_collected={peak_sample_count}"
                        ),
                    )
                elif peak_sample_count == 0:
                    report.add(
                        "peak_rss",
                        False,
                        "sampler collected zero samples (interval too long?)",
                    )
                else:
                    peak_passed = peak_observed_gib <= peak_ceiling_gib
                    report.add(
                        "peak_rss",
                        peak_passed,
                        (
                            f"N={args.n_vectors} "
                            f"peak_ceiling={peak_ceiling_gib:.2f} GiB "
                            f"({peak_ceiling_source}) "
                            f"peak_observed={peak_observed_gib:.3f} GiB "
                            f"rest_observed={observed_gib:.3f} GiB "
                            f"retention_gap={retention_gap_gib:.3f} GiB "
                            f"samples={peak_sample_count} "
                            f"verdict={'PASS' if peak_passed else 'FAIL'}"
                        ),
                    )
                    logger.info(
                        "[PEAK_RSS] peak=%.3f GiB rest=%.3f GiB "
                        "retention_gap=%.3f GiB samples=%d",
                        peak_observed_gib, observed_gib,
                        retention_gap_gib, peak_sample_count,
                    )
            else:
                logger.info(
                    "[PHASE rest_state] peak-RSS sampling disabled via "
                    "--no-peak-sampling; skipping peak assertion."
                )
        finally:
            try:
                client.close()
            except Exception:
                pass

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
        if owns_data_dir:
            _cleanup_data_dir(data_dir)


# ---------------------------------------------------------------------------
# Mode: per_vector_slope
# ---------------------------------------------------------------------------


def run_per_vector_slope(args: argparse.Namespace) -> bool:
    """Insert in checkpoints; assert per-vector slope is monotone non-increasing."""
    report = HarnessReport("per_vector_slope")

    data_dir, owns_data_dir = _resolve_data_dir(args, "per_vector_slope")
    log_path = data_dir.parent / "swarndb_per_vector_slope.log"

    checkpoints = default_slope_checkpoints(args.n_vectors)
    logger.info("[PHASE per_vector_slope] checkpoints=%s", checkpoints)

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc, pid = find_or_spawn_server(args, logger, _build_proc)

    try:
        client = make_client(args.grpc_port)
        samples: List[Tuple[int, float]] = []  # (n_so_far, rss_gib)
        try:
            ensure_collection(client, args.collection_name, args.dimension)

            prev_total = 0
            for idx, checkpoint in enumerate(checkpoints):
                slice_count = checkpoint - prev_total
                if slice_count <= 0:
                    continue

                logger.info(
                    "[PHASE per_vector_slope] checkpoint %d/%d: inserting %d "
                    "to reach n=%d",
                    idx + 1, len(checkpoints), slice_count, checkpoint,
                )
                # Seed per slice so memory in the harness process stays
                # bounded; the SwarnDB process is the measurement target.
                slice_vectors = generate_vectors(
                    slice_count,
                    args.dimension,
                    seed=42 + idx,
                )
                bulk_insert_slice(
                    client,
                    args.collection_name,
                    slice_vectors,
                    id_offset=prev_total,
                    batch_size=args.batch_size,
                    label=f"slope_checkpoint_{idx + 1}",
                )
                del slice_vectors

                logger.info(
                    "  settling %.1fs before RSS sample at n=%d",
                    SLOPE_SETTLE_SECONDS, checkpoint,
                )
                time.sleep(SLOPE_SETTLE_SECONDS)
                try:
                    rss_gib = sample_rss_gib(
                        pid,
                        SLOPE_SAMPLE_COUNT,
                        SLOPE_SAMPLE_INTERVAL_SECONDS,
                    )
                except (psutil.NoSuchProcess, RuntimeError) as exc:
                    logger.error(
                        "RSS sample failed: PID %d is gone (%s). If "
                        "running in external mode, the swarndb container "
                        "likely stopped; re-run after starting it back up.",
                        pid, exc,
                    )
                    return False
                samples.append((checkpoint, rss_gib))
                logger.info(
                    "[CHECKPOINT] n=%d rss=%.3f GiB",
                    checkpoint, rss_gib,
                )
                prev_total = checkpoint
        finally:
            try:
                client.close()
            except Exception:
                pass

        # Need at least two checkpoints to define a slope, and at least
        # three to test monotonicity of the slope sequence.
        if len(samples) < 2:
            report.add(
                "per_vector_slope: enough checkpoints",
                False,
                f"only {len(samples)} checkpoint(s) collected; need >= 2",
            )
            logger.error(report.summary())
            return False

        # Compute per-vector slopes (bytes per vector between consecutive
        # checkpoints). Negative values are clamped to 0 since RSS can
        # dip briefly when a snapshot completes and buffers release.
        slopes_bytes_per_vec: List[float] = []
        for i in range(1, len(samples)):
            n_prev, rss_prev = samples[i - 1]
            n_curr, rss_curr = samples[i]
            delta_n = n_curr - n_prev
            delta_rss_bytes = (rss_curr - rss_prev) * (1024.0 ** 3)
            slope = delta_rss_bytes / max(1, delta_n)
            slopes_bytes_per_vec.append(slope)
            logger.info(
                "[SLOPE] segment %d->%d (n %d->%d): %.1f bytes/vec",
                i, i + 1, n_prev, n_curr, slope,
            )

        # Monotone non-increasing with SLOPE_NOISE_SLACK absorbed. A
        # later slope is allowed to exceed the prior one by up to
        # (1 + SLOPE_NOISE_SLACK)x before flagging. Slopes are compared
        # in their natural sign: a larger positive slope means more
        # bytes per added vector, which is the regression we catch.
        violations: List[str] = []
        for i in range(1, len(slopes_bytes_per_vec)):
            prior = slopes_bytes_per_vec[i - 1]
            curr = slopes_bytes_per_vec[i]
            allowed_max = prior * (1.0 + SLOPE_NOISE_SLACK) if prior > 0 else \
                abs(prior) * SLOPE_NOISE_SLACK + 1.0
            if curr > allowed_max:
                violations.append(
                    f"segment {i + 1}: curr={curr:.1f} > "
                    f"allowed_max={allowed_max:.1f} (prior={prior:.1f}, "
                    f"slack={SLOPE_NOISE_SLACK:.0%})"
                )

        passed = not violations
        report.add(
            "per_vector_slope_monotone_non_increasing",
            passed,
            (
                "monotone non-increasing within "
                f"{SLOPE_NOISE_SLACK:.0%} slack"
                if passed
                else "; ".join(violations)
            ),
        )

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
        if owns_data_dir:
            _cleanup_data_dir(data_dir)


# ---------------------------------------------------------------------------
# Data directory plumbing
# ---------------------------------------------------------------------------


def _resolve_data_dir(args: argparse.Namespace, suffix: str) -> Tuple[Path, bool]:
    """Pick the data directory for a mode.

    If the user passed --data-dir, the harness uses <data_dir>/<suffix>
    and does NOT delete it on exit (the user owns persistence). If the
    user did not pass --data-dir, the harness creates a fresh tmpdir
    under tempfile.gettempdir() and removes it on exit.

    Returns (data_dir, owns_data_dir).
    """
    if args.data_dir:
        base = Path(args.data_dir).resolve()
        sub = base / suffix
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True, exist_ok=True)
        return sub, False
    base = Path(tempfile.mkdtemp(prefix=f"swarndb_p09_{suffix}_"))
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
            "SwarnDB memory-release validation harness "
            "(Perf_Stability P09 Step 4 / P11 execution)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["rest_state", "per_vector_slope", "all"],
        default="all",
        help="Which assertion mode to run.",
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
        "--external-pid",
        type=int,
        default=None,
        help=(
            "PID of the externally-running swarndb-server, used as the "
            "RSS measurement target when auto-detect picks up an "
            "existing server. In Docker, find this via "
            "`docker top swarndb`; in docker-compose, via "
            "`docker inspect --format '{{.State.Pid}}' swarndb`. "
            "Ignored on the spawn path (the spawned PID wins)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SWARNDB_HARNESS_DATA_DIR"),
        help=(
            "Optional base data directory. If omitted, the harness creates "
            "a tempdir per mode and removes it on exit."
        ),
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=200_000,
        help=(
            "Vector count. Default 200_000 for developer machines. "
            "P11 invokes with --n-vectors 1000000 against the Civo "
            "dbpedia_1m collection."
        ),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help="Vector dimension (default 1536 for DBpedia).",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Collection name.",
    )
    parser.add_argument(
        "--rest-state-budget-gib",
        type=float,
        default=None,
        help=(
            "Override the rest-state ceiling in GiB. If omitted, the "
            "harness computes the ceiling from --n-vectors using the "
            "proportional formula (9 KB/vec)."
        ),
    )
    parser.add_argument(
        "--peak-ceiling-gib",
        type=float,
        default=None,
        help=(
            "Override the peak-RSS ceiling in GiB for the rest_state "
            "mode. If omitted, the harness derives the ceiling as "
            f"{PEAK_RSS_DEFAULT_CEILING_MULTIPLIER:.1f}x the rest-state "
            "ceiling (so 27 GiB at 1M when the rest-state ceiling is "
            "9 GiB)."
        ),
    )
    parser.add_argument(
        "--no-peak-sampling",
        action="store_true",
        help=(
            "Disable the peak-RSS sampler thread in rest_state mode. "
            "Restores the legacy rest-state-only assertion; useful when "
            "the user only wants the post-settle rest-state number and "
            "wants to skip the load-time peak entirely."
        ),
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help=(
            "HNSW ef_construction. Reserved for future collection-create "
            "wiring; the current SDK uses server defaults."
        ),
    )
    parser.add_argument(
        "--m",
        type=int,
        default=16,
        help=(
            "HNSW M parameter. Reserved for future collection-create "
            "wiring; the current SDK uses server defaults."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="bulk_insert batch_size (default 10_000).",
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=DEFAULT_REST_PORT,
        help="REST port for /readyz polling.",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help="gRPC port for the SDK client.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if args.n_vectors <= 0:
        logger.error("[FAIL] --n-vectors must be positive (got %d)", args.n_vectors)
        return 2

    # --binary-path is optional in external mode (auto-detect picks up
    # a server already running on --rest-port). Validate the path only
    # when it was provided; find_or_spawn_server handles the
    # "no server AND no binary" failure with a clearer error.
    if args.binary_path:
        binary_path = Path(args.binary_path)
        if not binary_path.exists():
            logger.error("[FAIL] binary not found at %s", binary_path)
            return 2

    if args.mode == "rest_state":
        ok = run_rest_state(args)
    elif args.mode == "per_vector_slope":
        ok = run_per_vector_slope(args)
    else:
        rs_ok = run_rest_state(args)
        sl_ok = run_per_vector_slope(args)
        ok = rs_ok and sl_ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
