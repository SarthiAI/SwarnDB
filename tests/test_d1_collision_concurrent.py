#!/usr/bin/env python3
"""
SwarnDB D-1 reconciliation harness (Perf_Stability P10 Investigation C).

P10 Investigation C identified a false-positive defect in the D-1
reconciliation pass on the bulk_insert path. When two clients issued
overlapping bulk_inserts (or a single client retried a partial bulk
after a crash), the per-row commit-credit decision used a global
store.contains(id) probe to recover from "the row was already in the
store before this call" cases. That probe could not distinguish:

    (a) a row this call committed earlier in the same stream, and
    (b) a row a CONCURRENT call (or a prior call) had already committed.

Both showed up as "store has this id" at probe time, so case (b) got
silently re-credited to the current call's inserted_count even though
the current call had never written that row. Sums of inserted_count
across the two clients could exceed the true unique-row count by the
size of the overlap range; the actual data in the collection was
correct, but the API contract on inserted_count was wrong, and any
caller using inserted_count to drive idempotency, billing, or retry
decisions would make wrong decisions downstream.

P10 Step 2C landed the fix: bulk_insert now tracks the ids it has
committed inside the current call in a local committed_ids HashSet,
and the D-1 reconciliation pass credits inserted_count only on
committed_ids.contains(id), not on store.contains(id). Rows that were
ALREADY present at the start of the call, or that landed via a
concurrent call between probe and reconcile, are correctly classified
as AlreadyExists and surfaced in the errors list.

This harness validates the fix end to end at the SDK level:

    Mode 'concurrent': two SDK clients in two threads bulk_insert
    overlapping id ranges (A: a_start..a_end, B: b_start..b_end with
    b_start < a_end < b_end so the overlap range is b_start..a_end).
    The harness asserts inserted_A + inserted_B == actual unique rows
    in the collection. Before the fix, inserted_A + inserted_B would
    EXCEED actual unique rows by up to the overlap size. After the
    fix the two numbers match exactly.

    Mode 'sequential': A inserts first, B inserts second (with the
    overlap range present). With the fix in place, B's response
    correctly reports inserted_count for the non-overlap tail of its
    range AND errors for every id in the overlap range. This mode is
    a baseline that exercises the same code path with deterministic
    ordering, so a regression here is easy to triage.

    Mode 'all': concurrent then sequential against fresh collections.

P10 Investigation C ships and reviews this harness; live execution on
the Civo binary against the 1M DBPedia (1536-dim) dataset is
P11-deferred per the initiative's hard rules. The harness exits 0 on
success, 1 on any failed assertion (false-claim detected), 2 on setup
error.

Usage:

    python3 test_d1_collision_concurrent.py \\
        --binary-path /usr/local/bin/swarndb \\
        --dimension 768 \\
        --a-start 1 --a-end 1500 \\
        --b-start 500 --b-end 2000 \\
        --batch-size 100 \\
        --mode all \\
        --rest-port 18094 --grpc-port 18095 \\
        --output-json /tmp/p10_d1_collision.json
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
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import requests

# Make the in-tree SDK importable so the harness runs from a checkout
# without a wheel install. Mirrors the pattern used by the P07, P08,
# P09, and P10-3B harnesses.
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import (  # noqa: E402
    SwarnDBError,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("d1_collision_concurrent_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and tunables
# ---------------------------------------------------------------------------

# Distinct ports so this harness can coexist on the same dev box with
# the P07 (18086/18087), P08 (18088/18089), P09 (18090/18091), and
# P10-3B (18092/18093) harnesses.
DEFAULT_REST_PORT = 18094
DEFAULT_GRPC_PORT = 18095

# /readyz contract: generous to cover cold start on a fresh data dir.
READYZ_DEADLINE_SECONDS = 60.0
READYZ_POLL_INTERVAL_SECONDS = 0.5

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

# Range defaults: A covers 1..1500, B covers 500..2000, overlap
# 500..1500 (1001 rows), union 1..2000 (2000 rows).
DEFAULT_A_START = 1
DEFAULT_A_END = 1500
DEFAULT_B_START = 500
DEFAULT_B_END = 2000

# Other defaults.
DEFAULT_DIMENSION = 768
DEFAULT_BATCH_SIZE = 100
DEFAULT_SEED = 42
DEFAULT_DELAY_MS = 0
DEFAULT_COLLECTION_NAME = "p10_d1_collision"
DEFAULT_DISTANCE_METRIC = "cosine"

# Distinct seed offsets so A and B do not draw identical vectors. The
# vectors themselves are irrelevant to the assertion; the harness
# checks counts, not recall. Distinct offsets just make the data
# easier to eyeball if a reviewer dumps the collection.
SEED_OFFSET_A = 1000
SEED_OFFSET_B = 2000


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
class ClientOutcome:
    """Aggregate outcome for one client's full bulk-insert run.

    A single client may issue multiple bulk_insert calls (one per
    chunk), so the aggregate sums inserted_count and concatenates
    errors across chunks. The id_range is the inclusive [lo, hi]
    pair the client tried to insert.
    """

    label: str
    id_lo: int
    id_hi: int
    rows_tried: int
    inserted_count: int
    errors_count: int
    chunks: int
    elapsed_seconds: float
    started_at: float
    finished_at: float
    exception: Optional[str] = None


@dataclass
class CollisionAssertions:
    """The set of assertions checked per run."""

    no_double_claims: bool
    a_total_matches: bool
    b_total_matches: bool
    actual_unique_rows: int
    expected_unique_rows: int
    inserted_a: int
    inserted_b: int
    overlap_lo: int
    overlap_hi: int


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


class SwarndbProcess:
    """Spawn-and-supervise wrapper around the swarndb binary.

    Mirrors the SwarndbProcess class used by the P07, P08, P09, and
    P10-3B harnesses. Kept local so each harness owns its lifecycle
    without a shared utility module.
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
    the P07, P08, P09, and P10-3B harnesses.
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
    """Build an SDK client. A fresh client per thread keeps the gRPC
    channel and per-call retries isolated, which matches the "two
    independent clients" intent of the test.
    """
    return SwarnDBClient(
        host="127.0.0.1",
        port=grpc_port,
        timeout=600.0,
        max_retries=5,
        retry_delay=1.0,
    )


def ensure_collection(client: SwarnDBClient, name: str, dimension: int) -> None:
    """Drop-and-create the collection so each mode starts clean."""
    if client.collections.exists(name):
        client.collections.delete(name)
    client.collections.create(
        name,
        dimension=dimension,
        distance_metric=DEFAULT_DISTANCE_METRIC,
    )
    logger.info("created collection '%s' dim=%d", name, dimension)


def drop_collection_quiet(client: SwarnDBClient, name: str) -> None:
    """Best-effort drop. Swallows errors so a mode failure does not
    leak a half-cleaned collection into the next mode.
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
    """Deterministic synthetic vectors. The harness checks counts and
    errors, not recall, so randn / float32 is sufficient. Returned as
    list-of-list so the SDK streams it directly.
    """
    rng = np.random.RandomState(seed)
    arr = rng.randn(count, dimension).astype(np.float32)
    return arr.tolist()


def build_id_range(lo: int, hi: int) -> List[int]:
    """Inclusive [lo, hi] -> list of explicit ids."""
    if hi < lo:
        return []
    return list(range(lo, hi + 1))


# ---------------------------------------------------------------------------
# Per-client worker
# ---------------------------------------------------------------------------


def run_client_bulk(
    label: str,
    grpc_port: int,
    collection: str,
    id_lo: int,
    id_hi: int,
    dimension: int,
    batch_size: int,
    seed: int,
    barrier: Optional[threading.Barrier],
    pre_delay_seconds: float,
    outcome_slot: Dict[str, ClientOutcome],
) -> None:
    """Worker run by a thread for one client.

    Each call to bulk_insert below targets a single chunk of the
    client's id range, so the per-chunk inserted_count and errors
    flow back as a sum. This matches the spec wording in P10's
    Investigation C: "Capture each client's BulkInsertResponse
    inserted_count and errors" across all chunks the client made.

    The barrier (if present) ensures A and B begin their first chunk
    at approximately the same time. The pre_delay_seconds knob lets
    the caller stagger B's start without breaking the barrier
    semantics for A.
    """
    ids = build_id_range(id_lo, id_hi)
    rows_tried = len(ids)
    vectors = generate_vectors(rows_tried, dimension, seed=seed)
    n_chunks = math.ceil(rows_tried / batch_size) if rows_tried else 0

    client = None
    inserted_total = 0
    errors_total = 0
    exception_text: Optional[str] = None
    started_at = 0.0
    finished_at = 0.0

    try:
        client = make_client(grpc_port)

        # Synchronize the two clients' first chunk so they race.
        if barrier is not None:
            barrier.wait(timeout=30.0)

        # Optional pre-delay (only used by B in delayed-concurrent
        # mode; A always passes pre_delay_seconds=0).
        if pre_delay_seconds > 0:
            time.sleep(pre_delay_seconds)

        started_at = time.perf_counter()
        for chunk_idx in range(n_chunks):
            chunk_lo = chunk_idx * batch_size
            chunk_hi = min(chunk_lo + batch_size, rows_tried)
            chunk_ids = ids[chunk_lo:chunk_hi]
            chunk_vectors = vectors[chunk_lo:chunk_hi]
            chunk_meta = [
                {"row_idx": chunk_ids[i], "client": label}
                for i in range(len(chunk_ids))
            ]
            try:
                result = client.vectors.bulk_insert(
                    collection,
                    chunk_vectors,
                    metadata_list=chunk_meta,
                    ids=chunk_ids,
                    batch_size=batch_size,
                )
            except SwarnDBError as exc:
                exception_text = f"chunk {chunk_idx + 1}/{n_chunks}: {exc}"
                logger.error(
                    "[CLIENT %s] bulk_insert failed on chunk %d/%d: %s",
                    label, chunk_idx + 1, n_chunks, exc,
                )
                break

            inserted_total += result.inserted_count
            errors_total += len(result.errors)
            logger.info(
                "[CLIENT %s] chunk %d/%d ids %d..%d -> inserted=%d errors=%d",
                label, chunk_idx + 1, n_chunks,
                chunk_ids[0], chunk_ids[-1],
                result.inserted_count, len(result.errors),
            )
        finished_at = time.perf_counter()

    except Exception as exc:
        # Any unexpected error (barrier timeout, gRPC channel build
        # failure, etc.) lands here and is reported via the outcome.
        exception_text = f"unexpected: {exc}"
        logger.exception("[CLIENT %s] unexpected error: %s", label, exc)
        if started_at == 0.0:
            started_at = time.perf_counter()
        finished_at = time.perf_counter()

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    outcome_slot[label] = ClientOutcome(
        label=label,
        id_lo=id_lo,
        id_hi=id_hi,
        rows_tried=rows_tried,
        inserted_count=inserted_total,
        errors_count=errors_total,
        chunks=n_chunks,
        elapsed_seconds=max(0.0, finished_at - started_at),
        started_at=started_at,
        finished_at=finished_at,
        exception=exception_text,
    )


# ---------------------------------------------------------------------------
# Collection state verification
# ---------------------------------------------------------------------------


def measure_actual_unique_rows(
    client: SwarnDBClient,
    collection: str,
    union_lo: int,
    union_hi: int,
) -> int:
    """Count how many ids in [union_lo, union_hi] are actually
    present in the collection by issuing per-id GET calls.

    Rationale for per-id GET: the SDK exposes vectors.get(id) which
    returns None for absent ids, so this is the most direct,
    side-effect-free probe for "is this row in the collection right
    now". For the default union size of 2000 ids this is a few seconds
    of RPCs against a local server, which is acceptable for a
    verification step that runs at the end of a single mode. Larger
    unions would benefit from a server-side count endpoint; if one
    appears, swap it in here.
    """
    present = 0
    missing = 0
    other_errors = 0
    for vid in range(union_lo, union_hi + 1):
        try:
            record = client.vectors.get(collection, vid)
            if record is None:
                missing += 1
            else:
                present += 1
        except SwarnDBError as exc:
            # Anything that is not "not found" is suspicious. Count it
            # separately so the caller can see a non-zero value in the
            # log if the server is misbehaving.
            other_errors += 1
            logger.warning(
                "get(id=%d) raised non-NotFound error: %s", vid, exc,
            )
    logger.info(
        "[VERIFY] union %d..%d -> present=%d missing=%d other_errors=%d",
        union_lo, union_hi, present, missing, other_errors,
    )
    return present


# ---------------------------------------------------------------------------
# Assertion evaluation
# ---------------------------------------------------------------------------


def evaluate_assertions(
    outcome_a: ClientOutcome,
    outcome_b: ClientOutcome,
    actual_unique_rows: int,
    union_lo: int,
    union_hi: int,
    overlap_lo: int,
    overlap_hi: int,
) -> CollisionAssertions:
    """Apply the three D-1 reconciliation assertions.

    no_double_claims is the core invariant the P10 Step 2C fix is
    expected to restore: the sum of inserted_counts across the two
    clients must equal the number of rows actually committed to the
    collection. A pre-fix run would show inserted_A + inserted_B
    exceeding actual_unique_rows by up to the overlap size.

    a_total_matches and b_total_matches are local sanity checks: each
    client's (inserted_count + errors_count) should equal the number
    of rows it tried to insert. A mismatch here means the SDK lost or
    duplicated rows in transit, which is a separate failure mode from
    the D-1 false-positive being targeted.
    """
    expected_unique = union_hi - union_lo + 1
    inserted_a = outcome_a.inserted_count
    inserted_b = outcome_b.inserted_count

    no_double_claims = (inserted_a + inserted_b) == actual_unique_rows
    a_total_matches = (
        outcome_a.inserted_count + outcome_a.errors_count
        == outcome_a.rows_tried
    )
    b_total_matches = (
        outcome_b.inserted_count + outcome_b.errors_count
        == outcome_b.rows_tried
    )

    return CollisionAssertions(
        no_double_claims=no_double_claims,
        a_total_matches=a_total_matches,
        b_total_matches=b_total_matches,
        actual_unique_rows=actual_unique_rows,
        expected_unique_rows=expected_unique,
        inserted_a=inserted_a,
        inserted_b=inserted_b,
        overlap_lo=overlap_lo,
        overlap_hi=overlap_hi,
    )


def record_assertions(
    report: HarnessReport,
    outcome_a: ClientOutcome,
    outcome_b: ClientOutcome,
    asserts: CollisionAssertions,
) -> None:
    """Push the assertion outcomes into the report with crisp detail
    strings so the operator can see the offending counts at a glance.
    """
    report.add(
        "no_double_claims",
        asserts.no_double_claims,
        (
            f"inserted_A({asserts.inserted_a}) + inserted_B({asserts.inserted_b}) "
            f"= {asserts.inserted_a + asserts.inserted_b} "
            f"vs actual_unique_rows={asserts.actual_unique_rows}"
        ),
    )
    report.add(
        "expected_unique_rows_matches_actual",
        asserts.actual_unique_rows == asserts.expected_unique_rows,
        (
            f"actual={asserts.actual_unique_rows} "
            f"expected={asserts.expected_unique_rows}"
        ),
    )
    report.add(
        "a_total_matches",
        asserts.a_total_matches,
        (
            f"inserted_A({outcome_a.inserted_count}) + errors_A({outcome_a.errors_count}) "
            f"= {outcome_a.inserted_count + outcome_a.errors_count} "
            f"vs rows_tried_A={outcome_a.rows_tried}"
        ),
    )
    report.add(
        "b_total_matches",
        asserts.b_total_matches,
        (
            f"inserted_B({outcome_b.inserted_count}) + errors_B({outcome_b.errors_count}) "
            f"= {outcome_b.inserted_count + outcome_b.errors_count} "
            f"vs rows_tried_B={outcome_b.rows_tried}"
        ),
    )
    # Surface any worker-side exceptions explicitly so a setup failure
    # cannot pass silently.
    report.add(
        "no_worker_exceptions",
        outcome_a.exception is None and outcome_b.exception is None,
        (
            f"A.exception={outcome_a.exception} "
            f"B.exception={outcome_b.exception}"
        ),
    )


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def run_concurrent(
    client_verify: SwarnDBClient,
    args: argparse.Namespace,
    collection: str,
) -> Tuple[HarnessReport, Dict]:
    """Concurrent overlap: A and B race the same overlap range."""
    report = HarnessReport("concurrent")
    ensure_collection(client_verify, collection, args.dimension)

    union_lo = min(args.a_start, args.b_start)
    union_hi = max(args.a_end, args.b_end)
    overlap_lo = max(args.a_start, args.b_start)
    overlap_hi = min(args.a_end, args.b_end)

    logger.info(
        "[MODE concurrent] A range %d..%d (%d rows), B range %d..%d (%d rows), "
        "overlap %d..%d (%d rows), expected unique=%d",
        args.a_start, args.a_end, args.a_end - args.a_start + 1,
        args.b_start, args.b_end, args.b_end - args.b_start + 1,
        overlap_lo, overlap_hi, max(0, overlap_hi - overlap_lo + 1),
        union_hi - union_lo + 1,
    )

    outcomes: Dict[str, ClientOutcome] = {}

    # Two threads, one barrier party each. Barrier ensures both start
    # the first chunk at the same time. Optional --delay-ms biases
    # B's start later so the overlap is hit mid-stream.
    barrier = threading.Barrier(parties=2, timeout=30.0)
    delay_seconds = max(0, int(args.delay_ms)) / 1000.0

    thread_a = threading.Thread(
        target=run_client_bulk,
        name="client-A",
        kwargs=dict(
            label="A",
            grpc_port=args.grpc_port,
            collection=collection,
            id_lo=args.a_start,
            id_hi=args.a_end,
            dimension=args.dimension,
            batch_size=args.batch_size,
            seed=args.seed + SEED_OFFSET_A,
            barrier=barrier,
            pre_delay_seconds=0.0,
            outcome_slot=outcomes,
        ),
        daemon=False,
    )
    thread_b = threading.Thread(
        target=run_client_bulk,
        name="client-B",
        kwargs=dict(
            label="B",
            grpc_port=args.grpc_port,
            collection=collection,
            id_lo=args.b_start,
            id_hi=args.b_end,
            dimension=args.dimension,
            batch_size=args.batch_size,
            seed=args.seed + SEED_OFFSET_B,
            barrier=barrier,
            pre_delay_seconds=delay_seconds,
            outcome_slot=outcomes,
        ),
        daemon=False,
    )

    logger.info(
        "[MODE concurrent] launching A and B with barrier; delay_ms=%d",
        args.delay_ms,
    )
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    outcome_a = outcomes.get(
        "A",
        ClientOutcome(
            label="A", id_lo=args.a_start, id_hi=args.a_end,
            rows_tried=0, inserted_count=0, errors_count=0,
            chunks=0, elapsed_seconds=0.0,
            started_at=0.0, finished_at=0.0,
            exception="thread did not record outcome",
        ),
    )
    outcome_b = outcomes.get(
        "B",
        ClientOutcome(
            label="B", id_lo=args.b_start, id_hi=args.b_end,
            rows_tried=0, inserted_count=0, errors_count=0,
            chunks=0, elapsed_seconds=0.0,
            started_at=0.0, finished_at=0.0,
            exception="thread did not record outcome",
        ),
    )

    logger.info(
        "[CLIENT A] inserted=%d errors=%d elapsed=%.2fs exception=%s",
        outcome_a.inserted_count, outcome_a.errors_count,
        outcome_a.elapsed_seconds, outcome_a.exception,
    )
    logger.info(
        "[CLIENT B] inserted=%d errors=%d elapsed=%.2fs exception=%s",
        outcome_b.inserted_count, outcome_b.errors_count,
        outcome_b.elapsed_seconds, outcome_b.exception,
    )

    actual_unique_rows = measure_actual_unique_rows(
        client_verify, collection, union_lo, union_hi,
    )
    asserts = evaluate_assertions(
        outcome_a, outcome_b, actual_unique_rows,
        union_lo, union_hi, overlap_lo, overlap_hi,
    )
    record_assertions(report, outcome_a, outcome_b, asserts)

    payload = {
        "mode": "concurrent",
        "collection": collection,
        "a": asdict(outcome_a),
        "b": asdict(outcome_b),
        "assertions": asdict(asserts),
        "union_lo": union_lo,
        "union_hi": union_hi,
        "overlap_lo": overlap_lo,
        "overlap_hi": overlap_hi,
    }

    drop_collection_quiet(client_verify, collection)
    return report, payload


def run_sequential(
    client_verify: SwarnDBClient,
    args: argparse.Namespace,
    collection: str,
) -> Tuple[HarnessReport, Dict]:
    """Sequential overlap: A inserts first, then B (with overlap).

    Baseline that exercises the same D-1 reconciliation code with
    deterministic ordering. With the P10 Step 2C fix, B's response
    should report:

        inserted_count = (b_end - max(b_start, a_end + 1) + 1)  for the tail
        errors_count   = (min(b_end, a_end) - b_start + 1)      for the overlap

    Pre-fix, the overlap rows would have been silently re-credited to
    B's inserted_count, exactly mirroring the concurrent bug.
    """
    report = HarnessReport("sequential")
    ensure_collection(client_verify, collection, args.dimension)

    union_lo = min(args.a_start, args.b_start)
    union_hi = max(args.a_end, args.b_end)
    overlap_lo = max(args.a_start, args.b_start)
    overlap_hi = min(args.a_end, args.b_end)
    overlap_size = max(0, overlap_hi - overlap_lo + 1)
    b_tail_size = max(0, args.b_end - max(args.b_start, args.a_end + 1) + 1)
    b_total = args.b_end - args.b_start + 1

    logger.info(
        "[MODE sequential] A range %d..%d, B range %d..%d, overlap %d..%d (%d rows), "
        "expected B inserted=%d errors=%d",
        args.a_start, args.a_end, args.b_start, args.b_end,
        overlap_lo, overlap_hi, overlap_size,
        b_tail_size, overlap_size,
    )

    # A goes first, no barrier (single-threaded).
    outcomes_a: Dict[str, ClientOutcome] = {}
    run_client_bulk(
        label="A",
        grpc_port=args.grpc_port,
        collection=collection,
        id_lo=args.a_start,
        id_hi=args.a_end,
        dimension=args.dimension,
        batch_size=args.batch_size,
        seed=args.seed + SEED_OFFSET_A,
        barrier=None,
        pre_delay_seconds=0.0,
        outcome_slot=outcomes_a,
    )
    outcome_a = outcomes_a["A"]
    logger.info(
        "[CLIENT A] (sequential) inserted=%d errors=%d elapsed=%.2fs",
        outcome_a.inserted_count, outcome_a.errors_count, outcome_a.elapsed_seconds,
    )

    # Then B with the overlap range.
    outcomes_b: Dict[str, ClientOutcome] = {}
    run_client_bulk(
        label="B",
        grpc_port=args.grpc_port,
        collection=collection,
        id_lo=args.b_start,
        id_hi=args.b_end,
        dimension=args.dimension,
        batch_size=args.batch_size,
        seed=args.seed + SEED_OFFSET_B,
        barrier=None,
        pre_delay_seconds=0.0,
        outcome_slot=outcomes_b,
    )
    outcome_b = outcomes_b["B"]
    logger.info(
        "[CLIENT B] (sequential) inserted=%d errors=%d elapsed=%.2fs",
        outcome_b.inserted_count, outcome_b.errors_count, outcome_b.elapsed_seconds,
    )

    actual_unique_rows = measure_actual_unique_rows(
        client_verify, collection, union_lo, union_hi,
    )
    asserts = evaluate_assertions(
        outcome_a, outcome_b, actual_unique_rows,
        union_lo, union_hi, overlap_lo, overlap_hi,
    )
    record_assertions(report, outcome_a, outcome_b, asserts)

    # Extra sequential-only assertion: B should see errors_count
    # equal to the overlap size and inserted_count equal to the
    # non-overlap tail.
    report.add(
        "sequential_b_reports_overlap_as_errors",
        outcome_b.errors_count == overlap_size,
        (
            f"errors_B={outcome_b.errors_count} expected_overlap={overlap_size}"
        ),
    )
    report.add(
        "sequential_b_inserts_only_tail",
        outcome_b.inserted_count == b_tail_size,
        (
            f"inserted_B={outcome_b.inserted_count} expected_tail={b_tail_size} "
            f"(b_total={b_total})"
        ),
    )

    payload = {
        "mode": "sequential",
        "collection": collection,
        "a": asdict(outcome_a),
        "b": asdict(outcome_b),
        "assertions": asdict(asserts),
        "union_lo": union_lo,
        "union_hi": union_hi,
        "overlap_lo": overlap_lo,
        "overlap_hi": overlap_hi,
        "expected_b_inserted_tail": b_tail_size,
        "expected_b_errors_overlap": overlap_size,
    }

    drop_collection_quiet(client_verify, collection)
    return report, payload


def run_mode_all(
    client_verify: SwarnDBClient,
    args: argparse.Namespace,
) -> Tuple[bool, Dict]:
    """Run concurrent then sequential, each against a fresh collection."""
    concurrent_name = f"{args.collection_name}_concurrent"
    sequential_name = f"{args.collection_name}_sequential"

    report_c, payload_c = run_concurrent(client_verify, args, concurrent_name)
    logger.info(report_c.summary())

    report_s, payload_s = run_sequential(client_verify, args, sequential_name)
    logger.info(report_s.summary())

    overall_ok = report_c.all_passed() and report_s.all_passed()
    payload = {
        "mode": "all",
        "concurrent": payload_c,
        "sequential": payload_s,
        "overall_pass": overall_ok,
    }
    return overall_ok, payload


# ---------------------------------------------------------------------------
# Top-level dispatch with server lifecycle
# ---------------------------------------------------------------------------


def _run_with_server(
    args: argparse.Namespace,
    work,
) -> Tuple[bool, Dict]:
    """Spin up the SwarnDB process, hand a connected verify-client to
    `work`, and tear everything down on exit.

    `work` is a callable taking (client, args) and returning
    (ok: bool, payload: dict). The payload is propagated up so the
    caller can dump it to JSON.
    """
    data_dir, owns_data_dir = _resolve_data_dir(args)
    log_path = data_dir.parent / "swarndb_d1_collision.log"

    def _build_proc() -> SwarndbProcess:
        return SwarndbProcess(
            binary=Path(args.binary_path) if args.binary_path else Path(""),
            data_dir=data_dir,
            rest_port=args.rest_port,
            grpc_port=args.grpc_port,
            log_path=log_path,
        )

    _, _, proc = find_or_spawn_server(args, logger, _build_proc)

    try:
        client_verify = make_client(args.grpc_port)
        try:
            return work(client_verify, args)
        finally:
            try:
                client_verify.close()
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


def run_concurrent_mode(args: argparse.Namespace) -> Tuple[bool, Dict]:
    """`--mode concurrent` entry point."""

    def _work(client: SwarnDBClient, a: argparse.Namespace) -> Tuple[bool, Dict]:
        report, payload = run_concurrent(client, a, a.collection_name)
        logger.info(report.summary())
        return report.all_passed(), payload

    return _run_with_server(args, _work)


def run_sequential_mode(args: argparse.Namespace) -> Tuple[bool, Dict]:
    """`--mode sequential` entry point."""

    def _work(client: SwarnDBClient, a: argparse.Namespace) -> Tuple[bool, Dict]:
        report, payload = run_sequential(client, a, a.collection_name)
        logger.info(report.summary())
        return report.all_passed(), payload

    return _run_with_server(args, _work)


def run_all_mode(args: argparse.Namespace) -> Tuple[bool, Dict]:
    """`--mode all` entry point."""

    def _work(client: SwarnDBClient, a: argparse.Namespace) -> Tuple[bool, Dict]:
        return run_mode_all(client, a)

    return _run_with_server(args, _work)


# ---------------------------------------------------------------------------
# Data directory plumbing
# ---------------------------------------------------------------------------


def _resolve_data_dir(args: argparse.Namespace) -> Tuple[Path, bool]:
    """Pick the data directory for the run.

    If the user passed --data-dir, the harness uses it directly and
    does NOT delete it on exit. Otherwise the harness creates a fresh
    tmpdir and removes it on exit.
    """
    if args.data_dir:
        base = Path(args.data_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base, False
    base = Path(tempfile.mkdtemp(prefix="swarndb_p10_d1_collision_"))
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
            "SwarnDB D-1 reconciliation harness "
            "(Perf_Stability P10 Investigation C / P11 execution)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["concurrent", "sequential", "all"],
        default="all",
        help=(
            "Which mode to run. 'concurrent' races A and B; "
            "'sequential' runs A then B; 'all' runs both against "
            "fresh collections."
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
        "--dimension",
        type=int,
        default=DEFAULT_DIMENSION,
        help="Vector dimension (default 768).",
    )
    parser.add_argument(
        "--a-start",
        type=int,
        default=DEFAULT_A_START,
        help="Client A id range start, inclusive (default 1).",
    )
    parser.add_argument(
        "--a-end",
        type=int,
        default=DEFAULT_A_END,
        help="Client A id range end, inclusive (default 1500).",
    )
    parser.add_argument(
        "--b-start",
        type=int,
        default=DEFAULT_B_START,
        help="Client B id range start, inclusive (default 500).",
    )
    parser.add_argument(
        "--b-end",
        type=int,
        default=DEFAULT_B_END,
        help="Client B id range end, inclusive (default 2000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Bulk-insert chunk size (default 100).",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Collection name (or prefix for --mode all). Default 'p10_d1_collision'.",
    )
    parser.add_argument(
        "--rest-port",
        type=int,
        default=DEFAULT_REST_PORT,
        help="REST port for /readyz polling (default 18094).",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help="gRPC port for the SDK client (default 18095).",
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
        "--delay-ms",
        type=int,
        default=DEFAULT_DELAY_MS,
        help=(
            "Optional delay before client B starts its first chunk "
            "(default 0). Both clients still synchronize on the "
            "barrier; the delay only biases B's first chunk later "
            "so the overlap range is hit mid-stream."
        ),
    )

    parsed = parser.parse_args(argv)
    return parsed


def _validate_args(args: argparse.Namespace) -> Optional[str]:
    """Return an error string for invalid CLI input, or None if ok."""
    if args.dimension <= 0:
        return f"--dimension must be positive (got {args.dimension})"
    if args.batch_size <= 0:
        return f"--batch-size must be positive (got {args.batch_size})"
    if args.a_end < args.a_start:
        return (
            f"--a-end ({args.a_end}) must be >= --a-start ({args.a_start})"
        )
    if args.b_end < args.b_start:
        return (
            f"--b-end ({args.b_end}) must be >= --b-start ({args.b_start})"
        )
    if args.delay_ms < 0:
        return f"--delay-ms must be non-negative (got {args.delay_ms})"
    # Overlap must be non-empty for the test to make sense.
    overlap_lo = max(args.a_start, args.b_start)
    overlap_hi = min(args.a_end, args.b_end)
    if overlap_hi < overlap_lo:
        return (
            f"id ranges do not overlap: A={args.a_start}..{args.a_end}, "
            f"B={args.b_start}..{args.b_end}"
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

    # --binary-path is optional in external mode (auto-detect picks up
    # a server already running on --rest-port). Validate the path only
    # when it was provided.
    if args.binary_path:
        binary_path = Path(args.binary_path)
        if not binary_path.exists():
            logger.error("[FAIL] binary not found at %s", binary_path)
            return 2

    logger.info(
        "P10 Investigation C harness: D-1 reconciliation under "
        "concurrent overlapping bulk_inserts"
    )
    logger.info(
        "Client A range: %d..%d (%d rows)",
        args.a_start, args.a_end, args.a_end - args.a_start + 1,
    )
    logger.info(
        "Client B range: %d..%d (%d rows)",
        args.b_start, args.b_end, args.b_end - args.b_start + 1,
    )
    overlap_lo = max(args.a_start, args.b_start)
    overlap_hi = min(args.a_end, args.b_end)
    union_lo = min(args.a_start, args.b_start)
    union_hi = max(args.a_end, args.b_end)
    logger.info(
        "Overlap: %d..%d (%d rows)",
        overlap_lo, overlap_hi, overlap_hi - overlap_lo + 1,
    )
    logger.info(
        "Expected unique rows after both: %d",
        union_hi - union_lo + 1,
    )
    logger.info(
        "Targeting SwarnDB on rest_port=%d grpc_port=%d (auto-detect or spawn)",
        args.rest_port, args.grpc_port,
    )

    try:
        if args.mode == "concurrent":
            ok, payload = run_concurrent_mode(args)
        elif args.mode == "sequential":
            ok, payload = run_sequential_mode(args)
        else:
            ok, payload = run_all_mode(args)
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
