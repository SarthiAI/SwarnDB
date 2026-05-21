#!/usr/bin/env python3
"""
SwarnDB SDK ops-surface harness at scale (Perf_Stability P10.8 Writer D).

P10.1 introduced six operational SDK endpoints plus the
``vectors.get(missing_id)`` None contract, and P10.7 tightened the
``VectorRecord.id`` type to ``int``. The P10.5 docker-compose smoke
exercises every surface against a tiny throwaway collection, but
nothing today validates the same surfaces against the live 1M
DBPedia collection at the size the P11 regression gate actually runs.

This harness closes that gap. Against a swarndb-server with a
populated 1M collection already loaded, the harness exercises every
P10.1 operational surface plus the two ``vectors.get`` contracts and
asserts the dataclass shape and sensible values for each. It runs
eight sub-tests in a fixed order, reports PASS/FAIL per surface, and
exits 0 only when all eight pass.

Sub-tests (in order):

    1. ``client.readyz()`` returns ``ReadinessStatus`` with ``ready=True``.
    2. ``client.healthz()`` returns ``HealthStatus`` with ``healthy=True``.
    3. ``client.recovery_status()`` returns ``RecoveryStatus`` whose
       ``path`` is one of ``{"CleanShutdown", "IncrementalReplay",
       "FullRebuild", "Unknown"}``, ``elapsed_secs >= 0``, and
       ``collections`` is a dict.
    4. ``client.collections.snapshot(name)`` returns an int >= 0.
    5. ``client.collections.persistence_status(name)`` returns a
       ``PersistenceStatus`` with three ints satisfying
       ``last_snapshot_lsn <= current_lsn <= next_lsn``.
    6. ``client.collections.metrics(name)`` returns a
       ``CollectionMetrics`` whose four counters are non-negative ints,
       and a second call after a search shows at least one counter has
       increased (proving the counters are live, not constants).
    7. ``client.vectors.get(name, missing_id)`` for a deliberately
       unused id returns ``None`` (P10.1 Step 4 contract).
    8. ``client.vectors.get(name, present_id)`` for an id discovered
       via a real search returns a ``VectorRecord`` whose ``id`` is
       ``int`` (the P10.7 fix).

The harness uses the P10.6 auto-detect pattern: if a swarndb-server
is already up on ``--rest-port`` it reuses it; if not and
``--binary-path`` is set, it spawns one and tears it down on exit.
The harness defaults to ports 18100/18101 to coexist with the other
perf-stability harnesses on the same dev box. There is no scale-only
fast path; the harness assumes the named collection is already
populated by an earlier harness or by the operator.

Usage:

    python3 test_sdk_ops_at_scale.py \\
        --collection-name dbpedia_1m \\
        --rest-port 18100 \\
        --grpc-port 18101 \\
        --n-vectors-expected 1000000 \\
        --output-json /tmp/p10_8_sdk_ops_at_scale.json
"""

from __future__ import annotations

import argparse
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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Make the in-tree SDK importable so the harness runs from a checkout
# without a wheel install. Mirrors the pattern used by every other
# Perf_Stability harness in this folder.
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import (  # noqa: E402
    CollectionMetrics,
    HealthStatus,
    PersistenceStatus,
    ReadinessStatus,
    RecoveryStatus,
    SwarnDBClient,
    VectorRecord,
)
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("sdk_ops_at_scale_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and tunables
# ---------------------------------------------------------------------------

# Distinct ports so this harness can coexist on the same dev box with
# the P07 (18086/18087), P08 (18088/18089), P09 (18090/18091), P10-3B
# (18092/18093), and P10 D-1 collision (18094/18095) harnesses. P10.8
# Writer D uses 18100/18101 (clean slot above the P10.8 matrix) so it
# can attach to a live 1M collection a prior harness left running
# without colliding with the P10 D-1 collision harness ports.
DEFAULT_REST_PORT = 18100
DEFAULT_GRPC_PORT = 18101

# /readyz contract: generous to cover cold start on a fresh data dir
# if the harness ends up spawning the server itself.
READYZ_DEADLINE_SECONDS = 60.0
READYZ_POLL_INTERVAL_SECONDS = 0.5

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

# Auto-detect probe timeout.
EXTERNAL_PROBE_TIMEOUT_SECONDS = 2.0

# A deliberately huge id that should never collide with a 1M
# collection. Chosen well above 1e12 so even billion-scale collections
# would not hit it by accident. The harness asserts get() returns
# None for this id.
DEFAULT_MISSING_ID = 99_999_999_999

# Valid RecoveryStatus.path values per the proto enum mapping in
# client.py. "Unknown" is included so that this harness does not fail
# on a server build that has not surfaced a known recovery path yet
# (e.g. very early after boot, or in a stub deployment).
VALID_RECOVERY_PATHS = {"CleanShutdown", "IncrementalReplay", "FullRebuild", "Unknown"}

# The four CollectionMetrics counter names, kept here as the
# canonical list so the per-counter delta scan stays in lock-step
# with the dataclass shape in types.py.
METRIC_COUNTER_NAMES = (
    "map_lock_acquisitions",
    "collection_read_acquisitions",
    "collection_write_acquisitions",
    "total_blocked_microseconds",
)

DEFAULT_COLLECTION_NAME = "dbpedia_1m"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SurfaceRecord:
    """Per-surface outcome captured for the final report."""

    index: int
    name: str
    passed: bool
    detail: str

    def render(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] #{self.index} {self.name}: {self.detail}"


@dataclass
class HarnessReport:
    """Ordered list of per-surface outcomes plus the final tally."""

    records: List[SurfaceRecord] = field(default_factory=list)

    def add(self, index: int, name: str, passed: bool, detail: str) -> None:
        rec = SurfaceRecord(index=index, name=name, passed=passed, detail=detail)
        self.records.append(rec)
        logger.info(rec.render())

    def all_passed(self) -> bool:
        return all(r.passed for r in self.records)

    def first_failure(self) -> Optional[SurfaceRecord]:
        for r in self.records:
            if not r.passed:
                return r
        return None

    def summary_line(self) -> str:
        total = len(self.records)
        passed = sum(1 for r in self.records if r.passed)
        if self.all_passed():
            return f"SDK OPS: PASS ({passed}/{total})"
        first = self.first_failure()
        first_name = first.name if first is not None else "<unknown>"
        return (
            f"SDK OPS: FAIL ({passed}/{total}, first failure: {first_name})"
        )


# ---------------------------------------------------------------------------
# Process management (only used when --binary-path is provided and no
# external server is detected on --rest-port).
# ---------------------------------------------------------------------------


class SwarndbProcess:
    """Spawn-and-supervise wrapper around the swarndb binary.

    Mirrors the SwarndbProcess class used by every Perf_Stability
    harness. Kept local so each harness owns its lifecycle without a
    shared utility module. Only used in the fallback spawn path; the
    common case is attaching to an already-running server.
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
# Auto-detect existing server (P10.6 pattern)
# ---------------------------------------------------------------------------


def find_or_spawn_server(
    args: argparse.Namespace,
    log: logging.Logger,
    proc_factory,
) -> Tuple[str, str, Optional[SwarndbProcess]]:
    """Auto-detect a swarndb-server on args.rest_port; else spawn one.

    Returns (rest_url, grpc_url, spawned_process_or_None).

    Auto-detect path: if GET /readyz returns 200 on args.rest_port,
    the harness reuses that server and returns None for the process
    handle so teardown skips a process the harness does not own.

    Spawn path: if /readyz is unreachable AND --binary-path is
    provided, the harness calls proc_factory() to build a
    SwarndbProcess, starts it, waits for /readyz, and returns the
    wrapper. If --binary-path is missing AND no external server is
    up, the harness emits a clear error and exits non-zero. P10.8
    Writer D expects the live 1M collection to already be loaded by
    a prior harness, so the auto-detect path is the common case.
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
            "configured port (with the live 1M collection loaded) "
            "OR pass --binary-path to spawn one.",
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
    the rest of the Perf_Stability harnesses.
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
# SDK client
# ---------------------------------------------------------------------------


def make_client(grpc_port: int) -> SwarnDBClient:
    """Build a single SDK client used by every sub-test."""
    return SwarnDBClient(
        host="127.0.0.1",
        port=grpc_port,
        timeout=600.0,
        max_retries=5,
        retry_delay=1.0,
    )


# ---------------------------------------------------------------------------
# Search helper: discover a present id for sub-test #8
# ---------------------------------------------------------------------------


def discover_present_id(
    client: SwarnDBClient,
    collection: str,
) -> Optional[int]:
    """Run a single search to harvest one id known to exist.

    Returns the id of the top search result, or None if the search
    came back empty. The harness uses this id as the input to the
    P10.7 ``vectors.get(present_id)`` assertion.

    Query vector: the harness fetches the collection's dimension via
    ``collections.get`` and submits an all-zeros vector. The
    correctness assertion is "the returned id round-trips through
    ``vectors.get`` as an ``int``", not recall, so the choice of
    query vector is irrelevant.
    """
    info = client.collections.get(collection)
    dim = int(info.dimension)
    if dim <= 0:
        logger.warning(
            "collection '%s' reports non-positive dimension %d; "
            "cannot synthesize a query vector",
            collection, dim,
        )
        return None
    query = [0.0] * dim
    result = client.search.query(
        collection,
        query,
        k=1,
        include_metadata=False,
        include_graph=False,
    )
    if not result.results:
        logger.warning(
            "search against '%s' returned 0 results; cannot harvest a present id",
            collection,
        )
        return None
    top = result.results[0]
    logger.info(
        "discovered present id via search: id=%s score=%.6f",
        top.id, top.score,
    )
    return int(top.id)


# ---------------------------------------------------------------------------
# Sub-tests (one function per surface)
# ---------------------------------------------------------------------------


def sub_test_readyz(client: SwarnDBClient, report: HarnessReport) -> None:
    """Sub-test #1: client.readyz() returns ReadinessStatus with ready=True."""
    try:
        status = client.readyz()
    except SwarnDBError as exc:
        report.add(1, "readyz", False, f"raised SwarnDBError: {exc}")
        return
    shape_ok = isinstance(status, ReadinessStatus)
    ready_ok = shape_ok and status.ready is True
    detail = (
        f"shape={'ReadinessStatus' if shape_ok else type(status).__name__} "
        f"ready={getattr(status, 'ready', None)} "
        f"status={getattr(status, 'status', None)!r}"
    )
    report.add(1, "readyz", shape_ok and ready_ok, detail)


def sub_test_healthz(client: SwarnDBClient, report: HarnessReport) -> None:
    """Sub-test #2: client.healthz() returns HealthStatus with healthy=True."""
    try:
        status = client.healthz()
    except SwarnDBError as exc:
        report.add(2, "healthz", False, f"raised SwarnDBError: {exc}")
        return
    shape_ok = isinstance(status, HealthStatus)
    healthy_ok = shape_ok and status.healthy is True
    detail = (
        f"shape={'HealthStatus' if shape_ok else type(status).__name__} "
        f"healthy={getattr(status, 'healthy', None)} "
        f"status={getattr(status, 'status', None)!r}"
    )
    report.add(2, "healthz", shape_ok and healthy_ok, detail)


def sub_test_recovery_status(client: SwarnDBClient, report: HarnessReport) -> None:
    """Sub-test #3: client.recovery_status() shape and value contract."""
    try:
        rs = client.recovery_status()
    except SwarnDBError as exc:
        report.add(3, "recovery_status", False, f"raised SwarnDBError: {exc}")
        return
    shape_ok = isinstance(rs, RecoveryStatus)
    if not shape_ok:
        report.add(
            3, "recovery_status", False,
            f"shape={type(rs).__name__} (expected RecoveryStatus)",
        )
        return
    path_ok = rs.path in VALID_RECOVERY_PATHS
    elapsed_ok = isinstance(rs.elapsed_secs, int) and rs.elapsed_secs >= 0
    collections_ok = isinstance(rs.collections, dict)
    passed = path_ok and elapsed_ok and collections_ok
    detail = (
        f"path={rs.path!r} "
        f"elapsed_secs={rs.elapsed_secs} "
        f"collections.type={type(rs.collections).__name__} "
        f"collections.len={len(rs.collections) if collections_ok else 'n/a'}"
    )
    report.add(3, "recovery_status", passed, detail)


def sub_test_snapshot(
    client: SwarnDBClient,
    collection: str,
    report: HarnessReport,
) -> None:
    """Sub-test #4: collections.snapshot(name) returns int >= 0."""
    try:
        lsn = client.collections.snapshot(collection)
    except SwarnDBError as exc:
        report.add(4, "snapshot", False, f"raised SwarnDBError: {exc}")
        return
    int_ok = isinstance(lsn, int)
    range_ok = int_ok and lsn >= 0
    detail = f"last_snapshot_lsn={lsn} type={type(lsn).__name__}"
    report.add(4, "snapshot", int_ok and range_ok, detail)


def sub_test_persistence_status(
    client: SwarnDBClient,
    collection: str,
    report: HarnessReport,
) -> None:
    """Sub-test #5: persistence_status(name) shape + LSN ordering."""
    try:
        ps = client.collections.persistence_status(collection)
    except SwarnDBError as exc:
        report.add(
            5, "persistence_status", False, f"raised SwarnDBError: {exc}",
        )
        return
    shape_ok = isinstance(ps, PersistenceStatus)
    if not shape_ok:
        report.add(
            5, "persistence_status", False,
            f"shape={type(ps).__name__} (expected PersistenceStatus)",
        )
        return
    ints_ok = (
        isinstance(ps.last_snapshot_lsn, int)
        and isinstance(ps.current_lsn, int)
        and isinstance(ps.next_lsn, int)
    )
    order_ok = (
        ints_ok
        and ps.current_lsn >= ps.last_snapshot_lsn
        and ps.next_lsn >= ps.current_lsn
    )
    detail = (
        f"last_snapshot_lsn={ps.last_snapshot_lsn} "
        f"current_lsn={ps.current_lsn} "
        f"next_lsn={ps.next_lsn}"
    )
    report.add(5, "persistence_status", ints_ok and order_ok, detail)


def sub_test_metrics_live(
    client: SwarnDBClient,
    collection: str,
    report: HarnessReport,
) -> None:
    """Sub-test #6: metrics() shape, non-negative, and live across a search.

    The harness fetches metrics, runs one search (which is expected
    to bump ``collection_read_acquisitions``), then fetches metrics
    again. The contract here is twofold:

      a) the response shape is ``CollectionMetrics`` with four
         non-negative ``int`` counters;
      b) at least one counter strictly increased between the two
         snapshots, proving the counters are live, not constant.

    If the search itself fails, the harness still records (a) so the
    failure detail points at the search rather than the metrics
    endpoint.
    """
    try:
        before = client.collections.metrics(collection)
    except SwarnDBError as exc:
        report.add(6, "metrics", False, f"first call raised: {exc}")
        return
    shape_ok = isinstance(before, CollectionMetrics)
    if not shape_ok:
        report.add(
            6, "metrics", False,
            f"shape={type(before).__name__} (expected CollectionMetrics)",
        )
        return

    counters_before = {n: getattr(before, n) for n in METRIC_COUNTER_NAMES}
    types_ok = all(isinstance(v, int) for v in counters_before.values())
    nonneg_ok = types_ok and all(v >= 0 for v in counters_before.values())
    if not (types_ok and nonneg_ok):
        report.add(
            6, "metrics", False,
            f"counters_before={counters_before} types_ok={types_ok} "
            f"nonneg_ok={nonneg_ok}",
        )
        return

    # Bump counters with one search. A search failure does not by
    # itself fail this sub-test, but it does suppress the delta
    # check so the failure surfaces in the detail string.
    search_failed: Optional[str] = None
    try:
        info = client.collections.get(collection)
        dim = int(info.dimension)
        query = [0.0] * dim if dim > 0 else []
        if query:
            client.search.query(
                collection, query, k=1,
                include_metadata=False, include_graph=False,
            )
    except SwarnDBError as exc:
        search_failed = str(exc)
    except Exception as exc:
        search_failed = f"unexpected: {exc}"

    try:
        after = client.collections.metrics(collection)
    except SwarnDBError as exc:
        report.add(6, "metrics", False, f"second call raised: {exc}")
        return

    counters_after = {n: getattr(after, n) for n in METRIC_COUNTER_NAMES}
    deltas = {
        n: counters_after[n] - counters_before[n] for n in METRIC_COUNTER_NAMES
    }
    any_increased = any(d > 0 for d in deltas.values())

    if search_failed is not None:
        # If the search failed, we cannot reasonably expect the
        # counters to have bumped, but the shape + non-negativity
        # portion of the contract is still satisfied. Surface the
        # search failure so the operator can fix it before re-running.
        report.add(
            6, "metrics", False,
            f"shape+nonneg OK but search probe failed "
            f"({search_failed}); deltas={deltas}",
        )
        return

    passed = any_increased
    detail = (
        f"counters_before={counters_before} "
        f"counters_after={counters_after} "
        f"deltas={deltas} any_increased={any_increased}"
    )
    report.add(6, "metrics", passed, detail)


def sub_test_get_missing(
    client: SwarnDBClient,
    collection: str,
    missing_id: int,
    report: HarnessReport,
) -> None:
    """Sub-test #7: vectors.get(missing_id) returns None (P10.1 Step 4)."""
    try:
        rec = client.vectors.get(collection, missing_id)
    except SwarnDBError as exc:
        report.add(
            7, "vectors.get(missing)", False,
            f"raised SwarnDBError instead of returning None: {exc}",
        )
        return
    passed = rec is None
    detail = f"missing_id={missing_id} returned={rec!r}"
    report.add(7, "vectors.get(missing)", passed, detail)


def sub_test_get_present(
    client: SwarnDBClient,
    collection: str,
    report: HarnessReport,
) -> None:
    """Sub-test #8: vectors.get(present_id) returns VectorRecord with int id.

    Discovers the ``present_id`` by running a single k=1 search
    against the live collection and using the top result's id. The
    P10.7 fix tightens ``VectorRecord.id`` to ``int``, so the
    assertion is ``isinstance(rec.id, int)``.
    """
    try:
        present_id = discover_present_id(client, collection)
    except SwarnDBError as exc:
        report.add(
            8, "vectors.get(present)", False,
            f"discover_present_id raised SwarnDBError: {exc}",
        )
        return
    except Exception as exc:
        report.add(
            8, "vectors.get(present)", False,
            f"discover_present_id raised unexpectedly: {exc}",
        )
        return

    if present_id is None:
        report.add(
            8, "vectors.get(present)", False,
            "discover_present_id returned None (no search result to "
            "harvest a present id from)",
        )
        return

    try:
        rec = client.vectors.get(collection, present_id)
    except SwarnDBError as exc:
        report.add(
            8, "vectors.get(present)", False,
            f"get(present_id={present_id}) raised SwarnDBError: {exc}",
        )
        return

    if rec is None:
        report.add(
            8, "vectors.get(present)", False,
            f"get(present_id={present_id}) returned None for an id the "
            "search just confirmed exists",
        )
        return

    shape_ok = isinstance(rec, VectorRecord)
    id_int_ok = shape_ok and isinstance(rec.id, int)
    id_matches = shape_ok and int(rec.id) == int(present_id)
    passed = shape_ok and id_int_ok and id_matches
    detail = (
        f"present_id={present_id} "
        f"shape={'VectorRecord' if shape_ok else type(rec).__name__} "
        f"rec.id={rec.id!r} rec.id.type={type(rec.id).__name__} "
        f"id_matches={id_matches}"
    )
    report.add(8, "vectors.get(present)", passed, detail)


# ---------------------------------------------------------------------------
# Optional vector-count assertion
# ---------------------------------------------------------------------------


def maybe_assert_vector_count(
    client: SwarnDBClient,
    collection: str,
    expected: Optional[int],
) -> None:
    """If --n-vectors-expected was provided, log the observed count.

    The harness reads the collection's ``vector_count`` via
    ``collections.get`` and logs whether it matches the expected
    value. This is informational only; sub-test outcomes drive the
    final exit code. The reason it is not a hard sub-test is that
    the operator may attach to a collection that has been bulk-loaded
    to an unrelated size, and the harness should still validate the
    surfaces against whatever is loaded.
    """
    if expected is None:
        return
    try:
        info = client.collections.get(collection)
    except SwarnDBError as exc:
        logger.warning(
            "could not read collection info for '%s' to check "
            "--n-vectors-expected: %s",
            collection, exc,
        )
        return
    observed = int(info.vector_count)
    if observed == expected:
        logger.info(
            "vector_count check: collection='%s' observed=%d expected=%d (MATCH)",
            collection, observed, expected,
        )
    else:
        logger.warning(
            "vector_count check: collection='%s' observed=%d expected=%d "
            "(MISMATCH; informational only)",
            collection, observed, expected,
        )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_all_sub_tests(
    client: SwarnDBClient,
    args: argparse.Namespace,
) -> HarnessReport:
    """Run every sub-test in the documented order and return the report."""
    report = HarnessReport()

    sub_test_readyz(client, report)
    sub_test_healthz(client, report)
    sub_test_recovery_status(client, report)
    sub_test_snapshot(client, args.collection_name, report)
    sub_test_persistence_status(client, args.collection_name, report)
    sub_test_metrics_live(client, args.collection_name, report)
    sub_test_get_missing(client, args.collection_name, args.missing_id, report)
    sub_test_get_present(client, args.collection_name, report)

    maybe_assert_vector_count(
        client, args.collection_name, args.n_vectors_expected,
    )
    return report


def _resolve_data_dir(args: argparse.Namespace) -> Tuple[Path, bool]:
    """Pick the data directory for the run.

    If the user passed --data-dir, the harness uses it directly and
    does NOT delete it on exit. Otherwise the harness creates a fresh
    tmpdir and removes it on exit. The data dir is only relevant if
    the harness ends up spawning its own server; in the common
    auto-detect path it is unused.
    """
    if args.data_dir:
        base = Path(args.data_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base, False
    base = Path(tempfile.mkdtemp(prefix="swarndb_p10_8_sdk_ops_"))
    return base, True


def _cleanup_data_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        logger.warning("cleanup of %s failed: %s", path, exc)


def _run_with_server(args: argparse.Namespace) -> Tuple[bool, Dict]:
    """Attach to or spawn a swarndb-server, run the sub-tests, tear down."""
    data_dir, owns_data_dir = _resolve_data_dir(args)
    log_path = data_dir.parent / "swarndb_p10_8_sdk_ops_at_scale.log"

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc = find_or_spawn_server(args, logger, _build_proc)

    client: Optional[SwarnDBClient] = None
    try:
        client = make_client(args.grpc_port)
        report = run_all_sub_tests(client, args)
        summary = report.summary_line()
        logger.info(summary)
        payload: Dict = {
            "collection_name": args.collection_name,
            "rest_port": args.rest_port,
            "grpc_port": args.grpc_port,
            "n_vectors_expected": args.n_vectors_expected,
            "missing_id": args.missing_id,
            "records": [asdict(r) for r in report.records],
            "passed": report.all_passed(),
            "summary": summary,
        }
        return report.all_passed(), payload
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
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
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SwarnDB SDK ops-surface harness at scale "
            "(Perf_Stability P10.8 Writer D)."
        ),
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help=(
            "Live collection to exercise (e.g. dbpedia_1m). The "
            "harness assumes this collection is already populated by "
            "a prior step; it does NOT create or load the collection."
        ),
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=DEFAULT_REST_PORT,
        help="REST port for /readyz auto-detect (default 18100).",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help="gRPC port for the SDK client (default 18101).",
    )
    parser.add_argument(
        "--binary-path",
        default=None,
        help=(
            "Path to the pre-built swarndb binary. The harness does "
            "NOT build. Optional: if a swarndb-server is already up "
            "on --rest-port (auto-detected via /readyz), the harness "
            "uses it and skips spawn. Required only when no external "
            "server is up on the configured port."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SWARNDB_HARNESS_DATA_DIR"),
        help=(
            "Optional data directory used only on the fallback spawn "
            "path. If omitted the harness creates a tempdir and "
            "removes it on exit."
        ),
    )
    parser.add_argument(
        "--n-vectors-expected",
        type=int,
        default=None,
        help=(
            "Optional expected vector_count for the named collection. "
            "If provided, the harness logs whether the observed count "
            "matches (informational only; does not gate exit code)."
        ),
    )
    parser.add_argument(
        "--missing-id",
        type=int,
        default=DEFAULT_MISSING_ID,
        help=(
            "An id that should not exist in the collection. The "
            "harness asserts vectors.get(collection, missing_id) "
            f"returns None. Default {DEFAULT_MISSING_ID}."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to dump per-surface results as JSON.",
    )

    parsed = parser.parse_args(argv)
    return parsed


def _validate_args(args: argparse.Namespace) -> Optional[str]:
    """Return an error string for invalid CLI input, or None if OK."""
    if not args.collection_name:
        return "--collection-name must be a non-empty string"
    if args.rest_port <= 0 or args.rest_port > 65535:
        return f"--rest-port out of range (got {args.rest_port})"
    if args.grpc_port <= 0 or args.grpc_port > 65535:
        return f"--grpc-port out of range (got {args.grpc_port})"
    if args.missing_id < 0:
        return f"--missing-id must be non-negative (got {args.missing_id})"
    if args.n_vectors_expected is not None and args.n_vectors_expected < 0:
        return (
            f"--n-vectors-expected must be non-negative "
            f"(got {args.n_vectors_expected})"
        )
    return None


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

    err = _validate_args(args)
    if err is not None:
        logger.error("[FAIL] %s", err)
        return 2

    if args.binary_path:
        binary_path = Path(args.binary_path)
        if not binary_path.exists():
            logger.error("[FAIL] binary not found at %s", binary_path)
            return 2

    logger.info(
        "P10.8 Writer D harness: SDK ops-surface validation against "
        "the live collection '%s'", args.collection_name,
    )
    logger.info(
        "Targeting SwarnDB on rest_port=%d grpc_port=%d (auto-detect or spawn)",
        args.rest_port, args.grpc_port,
    )

    try:
        ok, payload = _run_with_server(args)
    except SwarnDBError as exc:
        logger.error("[FAIL] SDK error: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("[FAIL] runtime error: %s", exc)
        return 2

    _write_output_json(args.output_json, payload)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
