#!/usr/bin/env python3
"""
SwarnDB chunked compact-insert validation harness (memory-peak-reduction
initiative, Phase P03 Step 2).

What this harness verifies
--------------------------
The P03 work threaded a new ``chunk_size`` parameter through
``bulk_insert_from_path`` so that a single very large mmap'd load can
be processed by the server in fixed-size chunks. Between chunks the
server snapshots the collection, prunes the WAL, and releases the
transient build-time scratch buffers, trading insert wall-clock for a
materially lower peak resident set size. The default ``chunk_size=0``
is a no-op and preserves the existing single-pass behavior.

This harness exercises that surface end-to-end:

    * mode ``basic``           Sanity-check the chunked path with a
                               100k x 1536 load at ``chunk_size=50000``.
                               Asserts no SDK exception, the response
                               reports the expected row count, and
                               ``assigned_ids`` is fully populated.

    * mode ``peak_rss``        Drive a 1M x 1536 chunked load and
                               sample peak RSS at a 200ms cadence via
                               the P09 ``PeakRssSampler`` thread.
                               Asserts the observed peak stays under
                               a proportional ceiling formula derived
                               from the P09 rest-state slope plus a
                               1 GiB transient bulk-insert slack.

    * mode ``wall_clock_cost`` Run two 100k x 1536 loads back-to-back:
                               one at ``chunk_size=50000`` (chunked
                               compact path) and one at ``chunk_size=0``
                               (single-pass, current behavior). Asserts
                               the chunked path is slower than the
                               single-pass path (the snapshot + prune +
                               purge per chunk has a real cost) but
                               not by more than a 2.5x multiplier, so
                               the chunked path stays usable as a
                               default for memory-tight boxes.

    * mode ``all``             Runs ``basic``, ``peak_rss``, and
                               ``wall_clock_cost`` in sequence, each
                               against its own collection.

Which P03 surface this exercises
--------------------------------
``client.vectors.bulk_insert_from_path(..., chunk_size=...)`` (the sync
SDK; the async wrapper carries an identical surface). The harness
writes a temp ``.f32`` file (raw little-endian float32, no header) and
asks the server to mmap it. The server endpoint also accepts ``.npy``
inputs (introduced in P00); a future iteration of this harness can
swap the synthetic generator for real DBPedia ``.npy`` files without
any change to the assertion logic.

Gating
------
The Phase P03 Step 5 compile gate runs the cargo build. The live
chunked-load measurement is deferred to Phase P04, which runs this
harness on Civo against 1M x 1536 DBPedia. The file is shipped here
so that P04 can invoke it as-is; P03 only confirms the file parses
and the SDK surface plumbs the new kwarg through.

Usage
-----
    python test_chunk_compact_insert.py \\
        --binary-path /usr/local/bin/swarndb \\
        --mode basic \\
        --n-vectors 100000 \\
        --dimension 1536 \\
        --chunk-size 50000

Exits 0 if every assertion passes; non-zero on the first failure.
"""

# Mirrors patterns from tests/test_memory_release.py (the P09 PeakRss
# harness): process supervision, /readyz polling, find_or_spawn_server,
# RSS sampling, and the PeakRssSampler thread are intentionally
# duplicated rather than imported so each harness owns its lifecycle
# end-to-end and can be invoked stand-alone.

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil
import requests

# The SDK lives in-tree under sdk/python/src. Make it importable so the
# harness can run from a checkout without a wheel install. Mirrors the
# pattern used in tests/test_memory_release.py and other P0x harnesses.
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("chunk_compact_insert_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and tunables
# ---------------------------------------------------------------------------

# /readyz polling. The 60s deadline matches the P09 contract; the
# spawned server has to come up before any insert work begins.
READYZ_DEADLINE_SECONDS = 65.0
READYZ_POLL_INTERVAL_SECONDS = 0.5

# Default ports kept distinct from the P07/P08/P09 harnesses so the
# chunked-compact harness can coexist with them on the same dev box.
# 18094/18095 = recovery_at_scale, 18090/18091 = memory_release,
# 18098..18103 = sdk_ops_at_scale + threading. Pick 18096/18097.
DEFAULT_REST_PORT = 18096
DEFAULT_GRPC_PORT = 18097

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

# Settle window between bulk_insert_from_path returning and the
# rest-state sample. The chunked compact path snapshots and prunes
# between chunks so most release work has already happened, but the
# runtime still needs a beat to settle.
SETTLE_SECONDS = 5.0
SETTLE_SAMPLE_COUNT = 5
SETTLE_SAMPLE_INTERVAL_SECONDS = 1.0

# Peak-RSS sampler cadence. 200ms matches the P09 PeakRssSampler so
# the load-time spike is captured without biasing the insert path.
PEAK_RSS_POLL_INTERVAL_SECONDS = 0.2

# Per-vector cost coefficient for the proportional ceiling. Same slope
# as the P09 rest-state contract: ~9 KB/vec. The chunked path holds
# the rest state down at this slope across the entire load, with a
# transient bulk-insert slack of 1.0 GiB layered on top for the
# in-flight chunk's working memory.
COST_GIB_PER_VEC = 9.0e-6  # 9 microGiB / vec = ~9 KB/vec
CEILING_FLOOR_GIB = 0.5
BULK_INSERT_TRANSIENT_SLACK_GIB = 1.0

# Wall-clock cost guardrail. Chunked compact mode is allowed to be
# slower than single-pass (it should be, by construction; snapshot +
# prune + purge per chunk has a real cost) but not by more than this
# multiplier. Past this point the chunked path stops being usable as
# a default for memory-tight boxes.
WALL_CLOCK_COST_MAX_MULTIPLIER = 2.5

# Collection / workload defaults.
DEFAULT_COLLECTION_NAME = "p03_chunk_compact_test"
DEFAULT_DISTANCE_METRIC = "cosine"

# Default n_vectors per mode. peak_rss runs at 1M to exercise the real
# memory budget; basic + wall_clock_cost run at 100k to keep iteration
# fast on the dev box.
DEFAULT_N_VECTORS_BASIC = 100_000
DEFAULT_N_VECTORS_PEAK_RSS = 1_000_000
DEFAULT_N_VECTORS_WALL_CLOCK = 100_000

# Chunk size default. 50k is small enough that a 1M load goes through
# 20 compact passes (so the chunked-path effect is visible in RSS) and
# large enough that the per-chunk overhead does not dominate the
# wall-clock at 100k either.
DEFAULT_CHUNK_SIZE = 50_000

# Vector dimension default. 1536 = DBPedia shape; the P04 live run
# uses this dimension end-to-end.
DEFAULT_DIMENSION = 1536


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

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class HarnessReport:
    mode: str
    records: List[AssertionRecord] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str) -> None:
        rec = AssertionRecord(name=name, passed=passed, detail=detail)
        self.records.append(rec)
        logger.info(rec.render())

    def record(self, key: str, value: Any) -> None:
        """Stash a non-assertion measurement (timings, observed GiB,
        sample counts) so --output-json captures the full picture, not
        just the pass/fail booleans.
        """
        self.measurements[key] = value

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "all_passed": self.all_passed(),
            "records": [r.to_dict() for r in self.records],
            "measurements": self.measurements,
        }


# ---------------------------------------------------------------------------
# Peak-RSS ceiling logic
# ---------------------------------------------------------------------------


def compute_peak_ceiling_gib(n_vectors: int) -> float:
    """Pick the peak-RSS ceiling for n_vectors under chunked compact
    mode.

    The chunked compact path holds the rest state on the same ~9 KB/vec
    slope as the P09 rest-state contract; the only extra budget needed
    is the transient working memory of the chunk currently in flight.
    Formula:

        ceiling_gib = max(CEILING_FLOOR_GIB, n_vectors * 9.0e-6)
                      + BULK_INSERT_TRANSIENT_SLACK_GIB

    Verification at the canonical sizes:
        * 100_000  -> max(0.5, 0.9) + 1.0 = 1.9 GiB
        * 1_000_000 -> max(0.5, 9.0) + 1.0 = 10.0 GiB
    """
    raw = COST_GIB_PER_VEC * float(n_vectors)
    floored = max(CEILING_FLOOR_GIB, raw)
    return round(floored + BULK_INSERT_TRANSIENT_SLACK_GIB, 2)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


class SwarndbProcess:
    """Spawn-and-supervise wrapper around the swarndb binary.

    Mirrors the SwarndbProcess class in test_memory_release.py (P09).
    Kept local so each harness owns its lifecycle without a shared
    utility module.
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
# Auto-detect existing server
# ---------------------------------------------------------------------------


# Short timeout for the auto-detect probe; an existing local server
# answers /readyz in milliseconds. Mirrors the P09 pattern.
EXTERNAL_PROBE_TIMEOUT_SECONDS = 2.0


def find_or_spawn_server(
    args: argparse.Namespace,
    log: logging.Logger,
    proc_factory: Callable[[], SwarndbProcess],
) -> Tuple[str, str, Optional[SwarndbProcess], Optional[int]]:
    """Auto-detect a swarndb-server on args.rest_port; else spawn one.

    Mirrors the P09 helper of the same name. Returns
    (rest_url, grpc_url, spawned_process_or_None, pid_for_rss).

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
    test_memory_release.py.
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


# ---------------------------------------------------------------------------
# Synthetic vector file generation
# ---------------------------------------------------------------------------


def generate_vectors(count: int, dimension: int, seed: int = 42) -> np.ndarray:
    """Deterministic synthetic vectors as a (count, dimension) float32
    ndarray. The harness measures RSS and wall-clock, not recall, so
    standard normal noise is sufficient. C-contiguous so the .f32
    writer can tobytes() without an extra copy.
    """
    rng = np.random.RandomState(seed)
    arr = rng.randn(count, dimension).astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def write_f32_file(arr: np.ndarray, path: Path) -> None:
    """Write the dataset as a flat little-endian float32 buffer.

    No header, no magic; the server is told the dim out of band via
    the bulk_insert_from_path request. tobytes() preserves C-order and
    the native byte layout of the float32 dtype, which on every
    supported platform is little-endian.

    Note: the server endpoint (introduced in P00) also accepts ``.npy``
    inputs. A future iteration of this harness that wants real DBPedia
    vectors can swap this writer for numpy.save without changing the
    assertion logic; the file path passed to bulk_insert_from_path
    carries the format via its extension.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    assert arr.dtype == np.float32, "f32 writer expects float32 dtype"
    assert arr.flags["C_CONTIGUOUS"], "f32 writer expects C-contiguous"
    payload = arr.tobytes()
    expected_bytes = arr.shape[0] * arr.shape[1] * 4
    if len(payload) != expected_bytes:
        raise RuntimeError(
            f"f32 payload size {len(payload)} mismatches expected "
            f"{expected_bytes} bytes for shape {arr.shape}"
        )
    with path.open("wb") as f:
        f.write(payload)
    logger.info(
        "wrote .f32 file: %s size=%d bytes shape=%s",
        path, path.stat().st_size, arr.shape,
    )


# ---------------------------------------------------------------------------
# RSS sampling via psutil
# ---------------------------------------------------------------------------


def sample_rss_gib(pid: int, n_samples: int, interval_seconds: float) -> float:
    """Take ``n_samples`` RSS samples ``interval_seconds`` apart; return
    the median in GiB. Median over mean to discount transient spikes
    from scheduler-driven snapshot writes.
    """
    samples: List[float] = []
    proc = psutil.Process(pid)
    for i in range(n_samples):
        try:
            # Single-process RSS only. If SwarnDB ever forks worker
            # processes, switch to psutil.Process(pid).memory_info().rss
            # plus children(recursive=True) aggregation.
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
# Peak-RSS sampler thread
# ---------------------------------------------------------------------------


class PeakRssSampler:
    """Background thread that polls RSS during a workload phase.

    Mirrors PeakRssSampler from tests/test_memory_release.py (P09).
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
# Shared bulk-insert helper
# ---------------------------------------------------------------------------


def _do_bulk_insert_from_path(
    client: SwarnDBClient,
    collection: str,
    f32_path: Path,
    n_vectors: int,
    dimension: int,
    chunk_size: int,
    label: str,
) -> Tuple[Any, float]:
    """Invoke bulk_insert_from_path and return (result, elapsed_seconds).

    ``chunk_size=0`` exercises the single-pass path; ``> 0`` exercises
    the chunked compact path added in P03 Step 2.
    """
    logger.info(
        "[%s] bulk_insert_from_path: path=%s dim=%d expected=%d chunk_size=%d",
        label, f32_path, dimension, n_vectors, chunk_size,
    )
    t0 = time.time()
    result = client.vectors.bulk_insert_from_path(
        collection,
        str(f32_path),
        dim=dimension,
        expected_count=n_vectors,
        id_start=1,
        chunk_size=chunk_size,
    )
    elapsed = time.time() - t0
    rate = result.inserted_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "[%s] done: inserted=%d errors=%d in %.1fs (%.0f vec/s)",
        label, result.inserted_count, len(result.errors), elapsed, rate,
    )
    return result, elapsed


def _assert_bulk_response(
    report: HarnessReport,
    label: str,
    result: Any,
    expected_count: int,
) -> None:
    """Standard response-shape assertions for a chunked compact insert."""
    inserted = int(getattr(result, "inserted_count", 0))
    errors = list(getattr(result, "errors", []) or [])
    assigned_ids = list(getattr(result, "assigned_ids", []) or [])

    report.add(
        f"{label}.inserted_count",
        inserted == expected_count,
        f"inserted={inserted} expected={expected_count}",
    )
    report.add(
        f"{label}.no_errors",
        len(errors) == 0,
        f"errors={errors[:3]}" if errors else "no errors",
    )
    report.add(
        f"{label}.assigned_ids_length",
        len(assigned_ids) == expected_count,
        f"len(assigned_ids)={len(assigned_ids)} expected={expected_count}",
    )


# ---------------------------------------------------------------------------
# Data directory + file plumbing
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
    base = Path(tempfile.mkdtemp(prefix=f"swarndb_p03_chunk_{suffix}_"))
    return base, True


def _cleanup_data_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        logger.warning("cleanup of %s failed: %s", path, exc)


def _resolve_file_dir(args: argparse.Namespace, suffix: str) -> Tuple[Path, bool]:
    """Pick a directory to hold the synthetic .f32 input file(s).

    Always a tempdir under tempfile.gettempdir() unless --file-dir is
    passed (rare; mostly for debugging with stable paths). Returns
    (file_dir, owns_file_dir).
    """
    if getattr(args, "file_dir", None):
        base = Path(args.file_dir).resolve()
        sub = base / suffix
        sub.mkdir(parents=True, exist_ok=True)
        return sub, False
    base = Path(tempfile.mkdtemp(prefix=f"swarndb_p03_chunk_input_{suffix}_"))
    return base, True


# ---------------------------------------------------------------------------
# Mode: basic
# ---------------------------------------------------------------------------


def run_basic(args: argparse.Namespace) -> HarnessReport:
    """Load n_vectors (default 100k) x dimension at chunk_size=50000.

    Asserts:
      * No SDK exception.
      * Response.inserted_count == n_vectors.
      * Response.errors is empty.
      * Response.assigned_ids has length n_vectors.
    """
    report = HarnessReport("basic")
    n_vectors = args.n_vectors if args.n_vectors > 0 else DEFAULT_N_VECTORS_BASIC

    data_dir, owns_data_dir = _resolve_data_dir(args, "basic")
    file_dir, owns_file_dir = _resolve_file_dir(args, "basic")
    log_path = data_dir.parent / "swarndb_chunk_basic.log"

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc, _ = find_or_spawn_server(args, logger, _build_proc)

    try:
        client = make_client(args.grpc_port)
        try:
            collection = f"{args.collection_name}_basic"
            ensure_collection(client, collection, args.dimension)

            logger.info(
                "[PHASE basic] generating %d vectors at dim=%d",
                n_vectors, args.dimension,
            )
            arr = generate_vectors(n_vectors, args.dimension)
            f32_path = file_dir / f"{collection}.f32"
            write_f32_file(arr, f32_path)
            # Drop the harness-side copy; the file is what the server
            # mmaps, and the harness process's RSS is not the target.
            del arr

            try:
                result, elapsed = _do_bulk_insert_from_path(
                    client,
                    collection,
                    f32_path,
                    n_vectors,
                    args.dimension,
                    args.chunk_size,
                    label="basic_chunked",
                )
            except SwarnDBError as exc:
                report.add(
                    "basic.call_succeeded",
                    False,
                    f"SDK raised: {exc}",
                )
                return report

            report.add("basic.call_succeeded", True, "no SDK exception")
            report.record("basic.elapsed_seconds", elapsed)
            report.record("basic.n_vectors", n_vectors)
            report.record("basic.chunk_size", args.chunk_size)
            _assert_bulk_response(report, "basic", result, n_vectors)
        finally:
            try:
                client.close()
            except Exception:
                pass

        logger.info(report.summary())
        return report
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
        if owns_file_dir:
            _cleanup_data_dir(file_dir)


# ---------------------------------------------------------------------------
# Mode: peak_rss
# ---------------------------------------------------------------------------


def run_peak_rss(args: argparse.Namespace) -> HarnessReport:
    """Load n_vectors (default 1M) x dimension at chunk_size=50000 and
    assert peak RSS stays under the chunked-compact ceiling.

    Samples RSS via PeakRssSampler at 200ms cadence throughout the
    insert. Computes the ceiling from compute_peak_ceiling_gib unless
    the user passes --peak-ceiling-gib.
    """
    report = HarnessReport("peak_rss")
    n_vectors = args.n_vectors if args.n_vectors > 0 else DEFAULT_N_VECTORS_PEAK_RSS

    data_dir, owns_data_dir = _resolve_data_dir(args, "peak_rss")
    file_dir, owns_file_dir = _resolve_file_dir(args, "peak_rss")
    log_path = data_dir.parent / "swarndb_chunk_peak_rss.log"

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
            collection = f"{args.collection_name}_peak_rss"
            ensure_collection(client, collection, args.dimension)

            logger.info(
                "[PHASE peak_rss] generating %d vectors at dim=%d",
                n_vectors, args.dimension,
            )
            arr = generate_vectors(n_vectors, args.dimension)
            f32_path = file_dir / f"{collection}.f32"
            write_f32_file(arr, f32_path)
            del arr

            sampler = PeakRssSampler(
                pid=pid,
                interval_seconds=PEAK_RSS_POLL_INTERVAL_SECONDS,
            )
            logger.info(
                "[PHASE peak_rss] starting peak-RSS sampler "
                "(pid=%d interval=%.3fs)",
                pid, PEAK_RSS_POLL_INTERVAL_SECONDS,
            )
            sampler.start()

            insert_failed = False
            try:
                result, elapsed = _do_bulk_insert_from_path(
                    client,
                    collection,
                    f32_path,
                    n_vectors,
                    args.dimension,
                    args.chunk_size,
                    label="peak_rss_chunked",
                )
            except SwarnDBError as exc:
                report.add(
                    "peak_rss.call_succeeded",
                    False,
                    f"SDK raised: {exc}",
                )
                insert_failed = True
                result, elapsed = None, 0.0
            finally:
                sampler.stop()

            if insert_failed:
                return report

            report.add("peak_rss.call_succeeded", True, "no SDK exception")
            report.record("peak_rss.elapsed_seconds", elapsed)
            report.record("peak_rss.n_vectors", n_vectors)
            report.record("peak_rss.chunk_size", args.chunk_size)
            _assert_bulk_response(report, "peak_rss", result, n_vectors)

            # Settle window then a rest-state RSS sample, alongside
            # the peak. The two together tell the full story: how high
            # the chunked path peaks AND where it settles after.
            logger.info(
                "[PHASE peak_rss] settling %.1fs before rest-state sample",
                SETTLE_SECONDS,
            )
            time.sleep(SETTLE_SECONDS)
            try:
                rest_observed_gib = sample_rss_gib(
                    pid,
                    SETTLE_SAMPLE_COUNT,
                    SETTLE_SAMPLE_INTERVAL_SECONDS,
                )
            except (psutil.NoSuchProcess, RuntimeError) as exc:
                logger.error(
                    "RSS sample failed: PID %d is gone (%s). If running "
                    "in external mode, the swarndb container likely "
                    "stopped; re-run after starting it back up.",
                    pid, exc,
                )
                report.add(
                    "peak_rss.rest_state_sample",
                    False,
                    f"sample failed: {exc}",
                )
                return report
            report.record("peak_rss.rest_observed_gib", rest_observed_gib)

            # Resolve the peak ceiling: explicit override wins; else
            # derived from the proportional formula.
            if args.peak_ceiling_gib is not None:
                ceiling_gib = float(args.peak_ceiling_gib)
                ceiling_source = "cli_override"
            else:
                ceiling_gib = compute_peak_ceiling_gib(n_vectors)
                ceiling_source = "proportional_formula"
            report.record("peak_rss.ceiling_gib", ceiling_gib)
            report.record("peak_rss.ceiling_source", ceiling_source)

            sampler_err = sampler.error()
            peak_observed_gib = sampler.peak_gib()
            peak_sample_count = sampler.sample_count()
            report.record("peak_rss.peak_observed_gib", peak_observed_gib)
            report.record("peak_rss.sample_count", peak_sample_count)

            retention_gap_gib = max(0.0, peak_observed_gib - rest_observed_gib)
            report.record("peak_rss.retention_gap_gib", retention_gap_gib)

            if sampler_err is not None:
                report.add(
                    "peak_rss.sampler_ok",
                    False,
                    f"sampler error: {sampler_err}; "
                    f"samples_collected={peak_sample_count}",
                )
            elif peak_sample_count == 0:
                report.add(
                    "peak_rss.sampler_ok",
                    False,
                    "sampler collected zero samples (interval too long?)",
                )
            else:
                report.add(
                    "peak_rss.sampler_ok",
                    True,
                    f"samples={peak_sample_count}",
                )
                peak_passed = peak_observed_gib <= ceiling_gib
                report.add(
                    "peak_rss.under_ceiling",
                    peak_passed,
                    (
                        f"N={n_vectors} chunk_size={args.chunk_size} "
                        f"peak_ceiling={ceiling_gib:.2f} GiB "
                        f"({ceiling_source}) "
                        f"peak_observed={peak_observed_gib:.3f} GiB "
                        f"rest_observed={rest_observed_gib:.3f} GiB "
                        f"retention_gap={retention_gap_gib:.3f} GiB "
                        f"verdict={'PASS' if peak_passed else 'FAIL'}"
                    ),
                )
                logger.info(
                    "[PEAK_RSS] peak=%.3f GiB rest=%.3f GiB "
                    "retention_gap=%.3f GiB samples=%d ceiling=%.2f GiB",
                    peak_observed_gib, rest_observed_gib,
                    retention_gap_gib, peak_sample_count, ceiling_gib,
                )
        finally:
            try:
                client.close()
            except Exception:
                pass

        logger.info(report.summary())
        return report
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
        if owns_file_dir:
            _cleanup_data_dir(file_dir)


# ---------------------------------------------------------------------------
# Mode: wall_clock_cost
# ---------------------------------------------------------------------------


def run_wall_clock_cost(args: argparse.Namespace) -> HarnessReport:
    """Compare wall-clock of chunked vs single-pass on the same load.

    Runs two back-to-back inserts of n_vectors (default 100k) x
    dimension into distinct collections:
        * collection_A: chunk_size=args.chunk_size  (chunked compact)
        * collection_B: chunk_size=0                 (single-pass)

    Asserts:
        * Both calls succeed and report the expected row count.
        * Single-pass wall-clock < chunked wall-clock (sanity:
          chunked path SHOULD be slower because of the snapshot +
          prune + purge between chunks).
        * The chunked/single-pass ratio stays under
          WALL_CLOCK_COST_MAX_MULTIPLIER (default 2.5x). If it
          balloons past that, the chunked path is no longer usable
          as a default for memory-tight boxes and someone needs to
          look at why.
    """
    report = HarnessReport("wall_clock_cost")
    n_vectors = (
        args.n_vectors if args.n_vectors > 0 else DEFAULT_N_VECTORS_WALL_CLOCK
    )

    data_dir, owns_data_dir = _resolve_data_dir(args, "wall_clock_cost")
    file_dir, owns_file_dir = _resolve_file_dir(args, "wall_clock_cost")
    log_path = data_dir.parent / "swarndb_chunk_wall_clock.log"

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc, _ = find_or_spawn_server(args, logger, _build_proc)

    try:
        client = make_client(args.grpc_port)
        try:
            # Generate the dataset once and reuse the same .f32 file
            # for both loads. Identical input means the only variable
            # under test is chunk_size.
            logger.info(
                "[PHASE wall_clock_cost] generating %d vectors at dim=%d",
                n_vectors, args.dimension,
            )
            arr = generate_vectors(n_vectors, args.dimension)
            shared_path = file_dir / "wall_clock_input.f32"
            write_f32_file(arr, shared_path)
            del arr

            # Load A: chunked compact.
            collection_a = f"{args.collection_name}_wallclock_chunked"
            ensure_collection(client, collection_a, args.dimension)
            try:
                result_a, elapsed_chunked = _do_bulk_insert_from_path(
                    client,
                    collection_a,
                    shared_path,
                    n_vectors,
                    args.dimension,
                    args.chunk_size,
                    label="wall_clock_chunked",
                )
            except SwarnDBError as exc:
                report.add(
                    "wall_clock_cost.chunked_call_succeeded",
                    False,
                    f"SDK raised: {exc}",
                )
                return report
            report.add(
                "wall_clock_cost.chunked_call_succeeded",
                True,
                "no SDK exception (chunked)",
            )
            _assert_bulk_response(report, "wall_clock_cost.chunked",
                                  result_a, n_vectors)

            # Load B: single-pass.
            collection_b = f"{args.collection_name}_wallclock_singlepass"
            ensure_collection(client, collection_b, args.dimension)
            try:
                result_b, elapsed_single = _do_bulk_insert_from_path(
                    client,
                    collection_b,
                    shared_path,
                    n_vectors,
                    args.dimension,
                    0,  # single-pass
                    label="wall_clock_single_pass",
                )
            except SwarnDBError as exc:
                report.add(
                    "wall_clock_cost.single_pass_call_succeeded",
                    False,
                    f"SDK raised: {exc}",
                )
                return report
            report.add(
                "wall_clock_cost.single_pass_call_succeeded",
                True,
                "no SDK exception (single-pass)",
            )
            _assert_bulk_response(report, "wall_clock_cost.single_pass",
                                  result_b, n_vectors)

            # Comparison assertions. Guard against the pathological
            # case where one of the wall-clocks is zero (would happen
            # only if the harness clock has sub-second granularity
            # below the insert wall-clock, which it does not on any
            # supported platform).
            report.record("wall_clock_cost.chunked_seconds", elapsed_chunked)
            report.record("wall_clock_cost.single_pass_seconds", elapsed_single)
            ratio = (
                elapsed_chunked / elapsed_single
                if elapsed_single > 0
                else float("inf")
            )
            report.record("wall_clock_cost.ratio", ratio)

            single_faster = elapsed_single < elapsed_chunked
            report.add(
                "wall_clock_cost.single_pass_is_faster",
                single_faster,
                (
                    f"single_pass={elapsed_single:.2f}s "
                    f"chunked={elapsed_chunked:.2f}s "
                    f"(single faster? {single_faster})"
                ),
            )

            under_cap = ratio <= WALL_CLOCK_COST_MAX_MULTIPLIER
            report.add(
                "wall_clock_cost.under_multiplier_cap",
                under_cap,
                (
                    f"ratio={ratio:.2f}x "
                    f"cap={WALL_CLOCK_COST_MAX_MULTIPLIER:.1f}x "
                    f"(chunked={elapsed_chunked:.2f}s "
                    f"single_pass={elapsed_single:.2f}s)"
                ),
            )
        finally:
            try:
                client.close()
            except Exception:
                pass

        logger.info(report.summary())
        return report
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
        if owns_file_dir:
            _cleanup_data_dir(file_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SwarnDB chunked compact-insert validation harness "
            "(memory-peak-reduction P03 Step 2 / P04 execution)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "peak_rss", "wall_clock_cost", "all"],
        default="all",
        help=(
            "Which assertion mode to run. 'all' runs basic, peak_rss, "
            "and wall_clock_cost in sequence against fresh data dirs."
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
        "--file-dir",
        default=None,
        help=(
            "Optional base directory for the synthetic .f32 input "
            "file(s). If omitted, the harness creates a tempdir per "
            "mode and removes it on exit. Mostly useful for debugging "
            "with stable paths."
        ),
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=0,
        help=(
            "Vector count. Default is per-mode: 100_000 for basic and "
            "wall_clock_cost, 1_000_000 for peak_rss. Passing > 0 "
            "overrides all modes."
        ),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=DEFAULT_DIMENSION,
        help=f"Vector dimension (default {DEFAULT_DIMENSION} for DBPedia).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "chunk_size passed to bulk_insert_from_path for the chunked "
            f"loads (default {DEFAULT_CHUNK_SIZE}). The wall_clock_cost "
            "mode always runs the second load at chunk_size=0 for "
            "comparison; this flag controls only the chunked-path load."
        ),
    )
    parser.add_argument(
        "--peak-ceiling-gib",
        type=float,
        default=None,
        help=(
            "Override the peak-RSS ceiling in GiB for the peak_rss "
            "mode. If omitted, the harness computes the ceiling from "
            "the proportional formula "
            "(max(0.5, n * 9.0e-6) + 1.0 GiB)."
        ),
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Collection name prefix.",
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
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Optional path. If set, the harness writes a JSON file "
            "with each mode's records and measurements after the run. "
            "Useful for P04 dashboards."
        ),
    )
    return parser.parse_args(argv)


def _resolve_n_vectors(args: argparse.Namespace, mode: str) -> int:
    """The CLI uses --n-vectors=0 as 'fall back to the mode default'.
    Resolve the final count here so each mode picks up its own default
    when the user did not override.
    """
    if args.n_vectors > 0:
        return args.n_vectors
    if mode == "peak_rss":
        return DEFAULT_N_VECTORS_PEAK_RSS
    return DEFAULT_N_VECTORS_BASIC


def _write_output_json(
    path: Path,
    reports: List[HarnessReport],
    args: argparse.Namespace,
) -> None:
    payload = {
        "harness": "test_chunk_compact_insert",
        "args": {
            "mode": args.mode,
            "dimension": args.dimension,
            "chunk_size": args.chunk_size,
            "rest_port": args.rest_port,
            "grpc_port": args.grpc_port,
        },
        "reports": [r.to_dict() for r in reports],
        "all_passed": all(r.all_passed() for r in reports),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote results JSON: %s", path)


def main() -> int:
    args = parse_args()

    if args.dimension <= 0:
        logger.error("[FAIL] --dimension must be positive (got %d)", args.dimension)
        return 2
    if args.chunk_size < 0:
        logger.error(
            "[FAIL] --chunk-size must be >= 0 (got %d)", args.chunk_size,
        )
        return 2
    if args.n_vectors < 0:
        logger.error(
            "[FAIL] --n-vectors must be >= 0 (got %d)", args.n_vectors,
        )
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

    reports: List[HarnessReport] = []

    # Mode dispatcher. Each handler returns its own HarnessReport so
    # the orchestrator can aggregate exit status and JSON output.
    mode = args.mode
    if mode in ("basic", "all"):
        args_basic = argparse.Namespace(**vars(args))
        args_basic.n_vectors = _resolve_n_vectors(args, "basic")
        reports.append(run_basic(args_basic))
    if mode in ("peak_rss", "all"):
        args_peak = argparse.Namespace(**vars(args))
        args_peak.n_vectors = _resolve_n_vectors(args, "peak_rss")
        reports.append(run_peak_rss(args_peak))
    if mode in ("wall_clock_cost", "all"):
        args_wc = argparse.Namespace(**vars(args))
        args_wc.n_vectors = _resolve_n_vectors(args, "wall_clock_cost")
        reports.append(run_wall_clock_cost(args_wc))

    if args.output_json:
        try:
            _write_output_json(Path(args.output_json), reports, args)
        except Exception as exc:
            logger.warning("failed to write --output-json: %s", exc)

    ok = all(r.all_passed() for r in reports) and len(reports) > 0
    if not reports:
        logger.error("[FAIL] mode %r dispatched no handlers", mode)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
