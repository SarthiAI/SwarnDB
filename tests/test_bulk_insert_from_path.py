#!/usr/bin/env python3
"""
SwarnDB mmap bulk-insert-from-path harness (memory-peak-reduction P00 Step 5).

P00 of the memory-peak-reduction initiative adds a server-side bulk
ingest path that takes an absolute path to a vector file on the
server's filesystem and builds the HNSW index directly from the mmap'd
buffer. The new endpoint eliminates the gRPC inbound copy, the
per-batch Vec<Vec<f32>> deserialization, and the host-side
Arc::new(Vec::new()) allocations that currently inflate peak RSS
during the legacy bulk_insert path.

This harness validates the new endpoint end to end at the SDK level
against both supported wire formats and exercises the path security
contract documented in ADR-001 Decision 5. It does NOT measure RSS;
RSS validation lives in the existing test_memory_release.py harness
and the P04 Civo regression gate.

Wire formats covered:

    .npy   numpy native (header carries dtype, shape, fortran_order).
           The server auto-detects the .npy magic bytes and decodes the
           header with npyz. The client side writes the file with
           numpy.save.

    .f32   flat little-endian float32 buffer. Dim must be passed in
           the request. The client side writes the file with
           ndarray.tobytes() to capture the same byte order the server
           reads.

Modes (driven by --mode):

    npy                Insert 10000 x 128 vectors from a .npy file,
                       verify inserted_count, vector_count,
                       assigned_ids, errors.
    f32                Same as npy but the file is the flat .f32 buffer
                       and the request carries explicit dim.
    roundtrip          Insert via the .npy path, pick 10 random ids,
                       GET each via the SDK, compare the returned
                       vector bytes to the source array byte by byte.
    negative           Negative-path tests: invalid path, wrong dim,
                       path outside SWARNDB_BULK_INSERT_ALLOWED_ROOTS,
                       '..' traversal attempt, relative path. Each
                       must surface as an SDK exception.
    peak_rss_compare   P01 validation. Load a 100k x 128 dataset twice
                       against the same vf-server: once WITHOUT a
                       total_count_hint and once WITH it. A background
                       thread samples RSS on the server PID at ~50 ms
                       cadence; the mode asserts the with-hint peak
                       is <= the without-hint peak within a slack.
    all                Run npy, f32, roundtrip, negative, and
                       peak_rss_compare in sequence, each against its
                       own fresh collection.

Usage:

    python test_bulk_insert_from_path.py \\
        --binary-path /usr/local/bin/vf-server \\
        --mode all \\
        --rest-port 18096 --grpc-port 18097 \\
        --output-json /tmp/p00_bulk_insert_from_path.json

External mode (server already up on --rest-port and detected via
/readyz): pass --external-pid if the harness should still surface the
pid in logs (it is not used for any measurement here but matches the
P10.6 convention used by sibling harnesses).

Exit codes:

    0   All assertions passed for the selected mode(s).
    1   Any assertion failed.
    2   Setup error (binary missing, /readyz timeout, SDK init error,
        permission failure on the chosen data dir).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# Make the in-tree SDK importable so the harness runs from a checkout
# without a wheel install. Mirrors the pattern used by the P07, P08,
# P09, P10-3B, and P10 D-1 harnesses.
_HARNESS_DIR = Path(__file__).resolve().parent
_SDK_SRC = _HARNESS_DIR.parent / "sdk" / "python" / "src"
if _SDK_SRC.is_dir() and str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from swarndb import SwarnDBClient  # noqa: E402
from swarndb.exceptions import SwarnDBError  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("bulk_insert_from_path_harness")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants and tunables
# ---------------------------------------------------------------------------

# Ports kept distinct from the P07 (18086/18087), P08 (18088/18089),
# P09 (18090/18091), P10-3B (18092/18093), P10 D-1 collision
# (18094/18095), and P10.8 recall/sdk_ops (18098..18103) harnesses.
# 18096/18097 is the next clean slot above the existing matrix.
DEFAULT_REST_PORT = 18096
DEFAULT_GRPC_PORT = 18097

# /readyz contract: generous to cover cold start on a fresh data dir.
READYZ_DEADLINE_SECONDS = 60.0
READYZ_POLL_INTERVAL_SECONDS = 0.5
EXTERNAL_PROBE_TIMEOUT_SECONDS = 2.0

# Process management.
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_WAIT_SECONDS = 5.0

# Workload defaults. Small dataset because the harness is a correctness
# check, not a perf check: 10k rows at dim=128 is ~5 MB on disk and
# completes a full mode in under a minute on a dev box.
DEFAULT_DIM = 128
DEFAULT_N_VECTORS = 10_000
DEFAULT_ID_START = 1
DEFAULT_SEED = 42

# Variant used by the npy mode to cover a non-default id_start range.
ID_START_VARIANT = 1_000_000

# Number of random ids the roundtrip mode samples for the
# byte-equality check.
ROUNDTRIP_SAMPLE_COUNT = 10

# Tolerance for float round-trip comparison. The server returns
# vector bytes verbatim from its arena, but the gRPC wire encoding for
# the GetVector response still passes through repeated float32 fields,
# so an exact bitwise compare is the correctness target.
ROUNDTRIP_FLOAT_TOLERANCE = 0.0

# Collection / distance defaults.
DEFAULT_COLLECTION_PREFIX = "p00_bulk_from_path"
DEFAULT_DISTANCE_METRIC = "cosine"

# P01 peak-RSS comparison defaults.
DEFAULT_PEAK_COMPARE_N = 100_000
DEFAULT_PEAK_COMPARE_DIM = 128
DEFAULT_PEAK_COMPARE_SLACK_MB = 50
PEAK_COMPARE_SAMPLE_INTERVAL_SECONDS = 0.05


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

    def as_payload(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "all_passed": self.all_passed(),
            "records": [asdict(r) for r in self.records],
        }


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


class SwarndbProcess:
    """Spawn-and-supervise wrapper around the vf-server binary.

    Mirrors the SwarndbProcess class used by the P07, P08, P09, P10-3B,
    P10 D-1, and P10.8 harnesses. Kept local so each harness owns its
    lifecycle without a shared utility module.
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
            "spawning vf-server: binary=%s data_dir=%s rest=%d grpc=%d log=%s",
            self.binary, self.data_dir, self.rest_port, self.grpc_port,
            self.log_path,
        )
        logger.info(
            "env: SWARNDB_BULK_INSERT_ALLOWED_ROOTS=%s",
            env.get("SWARNDB_BULK_INSERT_ALLOWED_ROOTS", "<unset>"),
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
            raise RuntimeError("vf-server process not started")
        return self.proc.pid

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
            logger.warning("vf-server did not exit after SIGKILL fallback")
        self._close_log()

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None


# ---------------------------------------------------------------------------
# Auto-detect existing server (P10.6 convention)
# ---------------------------------------------------------------------------


def find_or_spawn_server(
    args: argparse.Namespace,
    log: logging.Logger,
    proc_factory,
) -> Tuple[str, str, Optional[SwarndbProcess]]:
    """Auto-detect a vf-server on args.rest_port; else spawn one.

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
        if args.external_pid is not None:
            log.info(
                "Found existing vf-server on %s; using it (external_pid=%d, "
                "skipping spawn and teardown).",
                rest_url, args.external_pid,
            )
        else:
            log.info(
                "Found existing vf-server on %s; using it "
                "(skipping spawn and teardown).",
                rest_url,
            )
        return rest_url, grpc_url, None

    if args.external_pid is not None:
        log.error(
            "--external-pid=%d was provided, but no vf-server "
            "responded on %s/readyz. Either start the external server "
            "first OR drop --external-pid to spawn a fresh server.",
            args.external_pid, rest_url,
        )
        sys.exit(2)

    if not args.binary_path:
        log.error(
            "No vf-server detected on port %d and --binary-path "
            "not provided. Either start a vf-server on the "
            "configured port OR pass --binary-path to spawn one.",
            args.rest_port,
        )
        sys.exit(2)

    log.info(
        "No vf-server on %s; spawning a fresh one from %s.",
        rest_url, args.binary_path,
    )
    proc = proc_factory()
    proc.start()
    ok, elapsed = wait_for_readyz(
        args.rest_port, READYZ_DEADLINE_SECONDS, time.time(),
    )
    if not ok:
        log.error(
            "Spawned vf-server did not return /readyz=200 within "
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


# ---------------------------------------------------------------------------
# SDK helpers
# ---------------------------------------------------------------------------


def make_client(grpc_port: int) -> SwarnDBClient:
    """Build an SDK client with generous timeouts for ingest calls."""
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


def collection_vector_count(client: SwarnDBClient, name: str) -> int:
    """Read the live vector_count from the server."""
    info = client.collections.get(name)
    return int(info.vector_count)


# ---------------------------------------------------------------------------
# Dataset generation and file writers
# ---------------------------------------------------------------------------


def generate_dataset(n_vectors: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic synthetic float32 dataset.

    Returned as a contiguous (n_vectors, dim) float32 numpy array so
    both file writers can consume it cheaply.
    """
    rng = np.random.RandomState(seed)
    arr = rng.randn(n_vectors, dim).astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def write_npy_file(arr: np.ndarray, path: Path) -> None:
    """Write the dataset as a numpy .npy file.

    numpy.save emits the standard .npy header including the
    '\\x93NUMPY' magic, version, and a literal dict for dtype, shape,
    and fortran_order. The server side detects this header and decodes
    it with npyz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr, allow_pickle=False)
    if path.suffix != ".npy":
        # numpy.save appends .npy if the user did not include it;
        # honor whatever the caller asked for.
        target = path.with_suffix(".npy")
        if target != path:
            target.rename(path)
    logger.info(
        "wrote .npy file: %s size=%d bytes shape=%s dtype=%s",
        path, path.stat().st_size, arr.shape, arr.dtype,
    )


def write_f32_file(arr: np.ndarray, path: Path) -> None:
    """Write the dataset as a flat little-endian float32 buffer.

    No header, no magic; the server must be told the dim out of band
    via the request. tobytes() preserves C-order and the native byte
    layout of the float32 dtype, which on every supported platform is
    little-endian.
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
# Mode runners
# ---------------------------------------------------------------------------


def _assert_bulk_response(
    report: HarnessReport,
    label: str,
    result: Any,
    expected_count: int,
    expected_id_start: int,
) -> None:
    """Apply the standard inserted_count / errors / assigned_ids
    assertions to a BulkInsertResult and record them in the report.
    """
    inserted = int(getattr(result, "inserted_count", 0))
    errors = list(getattr(result, "errors", []) or [])
    assigned = list(getattr(result, "assigned_ids", []) or [])
    expected_ids = list(range(expected_id_start, expected_id_start + expected_count))

    report.add(
        f"{label}.inserted_count",
        inserted == expected_count,
        f"inserted_count={inserted} expected={expected_count}",
    )
    report.add(
        f"{label}.errors_empty",
        len(errors) == 0,
        f"errors_count={len(errors)} first={errors[0] if errors else '<none>'}",
    )
    report.add(
        f"{label}.assigned_ids_match",
        assigned == expected_ids,
        (
            f"assigned[0..3]={assigned[:3]} assigned[-3..]={assigned[-3:]} "
            f"len={len(assigned)} expected_first={expected_ids[0]} "
            f"expected_last={expected_ids[-1]}"
        ),
    )


def run_npy(
    client: SwarnDBClient,
    args: argparse.Namespace,
    collection: str,
    file_dir: Path,
    id_start: int,
) -> HarnessReport:
    """Insert 10k x 128 vectors from a .npy file."""
    report = HarnessReport(f"npy(id_start={id_start})")
    ensure_collection(client, collection, args.dim)

    arr = generate_dataset(args.n_vectors, args.dim, seed=args.seed)
    npy_path = file_dir / f"{collection}.npy"
    write_npy_file(arr, npy_path)

    logger.info(
        "[MODE npy] calling bulk_insert_from_path: collection=%s path=%s "
        "dim=%d expected_count=%d id_start=%d",
        collection, npy_path, args.dim, args.n_vectors, id_start,
    )
    try:
        result = client.vectors.bulk_insert_from_path(
            collection,
            str(npy_path),
            dim=args.dim,
            expected_count=args.n_vectors,
            id_start=id_start,
        )
    except SwarnDBError as exc:
        report.add("npy.call_succeeded", False, f"SDK raised: {exc}")
        return report

    report.add("npy.call_succeeded", True, "no SDK exception")
    _assert_bulk_response(report, "npy", result, args.n_vectors, id_start)

    actual_count = collection_vector_count(client, collection)
    report.add(
        "npy.vector_count",
        actual_count == args.n_vectors,
        f"vector_count={actual_count} expected={args.n_vectors}",
    )
    return report


def run_f32(
    client: SwarnDBClient,
    args: argparse.Namespace,
    collection: str,
    file_dir: Path,
    id_start: int,
) -> HarnessReport:
    """Insert 10k x 128 vectors from a flat .f32 file."""
    report = HarnessReport("f32")
    ensure_collection(client, collection, args.dim)

    arr = generate_dataset(args.n_vectors, args.dim, seed=args.seed + 1)
    f32_path = file_dir / f"{collection}.f32"
    write_f32_file(arr, f32_path)

    logger.info(
        "[MODE f32] calling bulk_insert_from_path: collection=%s path=%s "
        "dim=%d expected_count=%d id_start=%d",
        collection, f32_path, args.dim, args.n_vectors, id_start,
    )
    try:
        result = client.vectors.bulk_insert_from_path(
            collection,
            str(f32_path),
            dim=args.dim,
            expected_count=args.n_vectors,
            id_start=id_start,
        )
    except SwarnDBError as exc:
        report.add("f32.call_succeeded", False, f"SDK raised: {exc}")
        return report

    report.add("f32.call_succeeded", True, "no SDK exception")
    _assert_bulk_response(report, "f32", result, args.n_vectors, id_start)

    actual_count = collection_vector_count(client, collection)
    report.add(
        "f32.vector_count",
        actual_count == args.n_vectors,
        f"vector_count={actual_count} expected={args.n_vectors}",
    )
    return report


def run_roundtrip(
    client: SwarnDBClient,
    args: argparse.Namespace,
    collection: str,
    file_dir: Path,
    id_start: int,
) -> HarnessReport:
    """Insert via .npy, then GET 10 random ids and byte-compare."""
    report = HarnessReport("roundtrip")
    ensure_collection(client, collection, args.dim)

    arr = generate_dataset(args.n_vectors, args.dim, seed=args.seed + 2)
    npy_path = file_dir / f"{collection}.npy"
    write_npy_file(arr, npy_path)

    try:
        result = client.vectors.bulk_insert_from_path(
            collection,
            str(npy_path),
            dim=args.dim,
            expected_count=args.n_vectors,
            id_start=id_start,
        )
    except SwarnDBError as exc:
        report.add("roundtrip.call_succeeded", False, f"SDK raised: {exc}")
        return report

    report.add("roundtrip.call_succeeded", True, "no SDK exception")
    _assert_bulk_response(report, "roundtrip", result, args.n_vectors, id_start)

    rng = random.Random(args.seed + 7)
    sample_idx = sorted(
        rng.sample(range(args.n_vectors), ROUNDTRIP_SAMPLE_COUNT)
    )

    mismatches: List[str] = []
    fetched = 0
    for row_idx in sample_idx:
        vid = id_start + row_idx
        try:
            record = client.vectors.get(collection, vid)
        except SwarnDBError as exc:
            mismatches.append(f"id={vid} GET raised {exc}")
            continue
        if record is None:
            mismatches.append(f"id={vid} returned None")
            continue
        got = np.asarray(record.vector, dtype=np.float32)
        want = arr[row_idx]
        if got.shape != want.shape:
            mismatches.append(
                f"id={vid} shape={got.shape} expected={want.shape}"
            )
            continue
        if ROUNDTRIP_FLOAT_TOLERANCE == 0.0:
            equal = np.array_equal(got, want)
        else:
            equal = bool(
                np.allclose(got, want, atol=ROUNDTRIP_FLOAT_TOLERANCE)
            )
        if not equal:
            diff = float(np.max(np.abs(got - want))) if got.size else float("nan")
            mismatches.append(f"id={vid} max_abs_diff={diff}")
        else:
            fetched += 1

    report.add(
        "roundtrip.sample_fetched",
        fetched == ROUNDTRIP_SAMPLE_COUNT,
        f"fetched={fetched}/{ROUNDTRIP_SAMPLE_COUNT}",
    )
    report.add(
        "roundtrip.byte_equality",
        len(mismatches) == 0,
        f"mismatches={mismatches[:3]} total={len(mismatches)}",
    )
    return report


def _expect_sdk_failure(
    report: HarnessReport,
    label: str,
    detail_hint: str,
    callable_thunk,
) -> None:
    """Run callable_thunk and assert it raises SwarnDBError (or a
    subclass). The detail_hint is a substring the harness logs alongside
    the result so reviewers can grep for the specific case.
    """
    try:
        callable_thunk()
    except SwarnDBError as exc:
        report.add(
            label,
            True,
            f"{detail_hint} -> SDK raised {type(exc).__name__}: {exc}",
        )
        return
    except Exception as exc:
        report.add(
            label,
            False,
            f"{detail_hint} -> unexpected non-SwarnDBError "
            f"{type(exc).__name__}: {exc}",
        )
        return
    report.add(
        label,
        False,
        f"{detail_hint} -> SDK returned success; expected an exception",
    )


def run_negative(
    client: SwarnDBClient,
    args: argparse.Namespace,
    collection: str,
    file_dir: Path,
    allowed_root: Path,
) -> HarnessReport:
    """Negative path tests against the new endpoint.

    Each case must surface as an SDK exception. The harness does not
    check the exact exception subclass because the server side maps a
    few different conditions onto INVALID_ARGUMENT and PERMISSION_DENIED
    and the SDK translates them onto a small set of types (see
    sdk/python/src/swarndb/_helpers.py:_translate_error). A reviewer
    can inspect the rendered details to confirm the right code path
    fired.
    """
    report = HarnessReport("negative")
    ensure_collection(client, collection, args.dim)

    arr = generate_dataset(args.n_vectors, args.dim, seed=args.seed + 3)
    good_path = file_dir / f"{collection}_good.npy"
    write_npy_file(arr, good_path)

    # Case 1: invalid path. File does not exist anywhere on disk.
    bogus_path = file_dir / f"does_not_exist_{int(time.time())}.npy"
    _expect_sdk_failure(
        report,
        "negative.invalid_path",
        f"missing file at {bogus_path}",
        lambda: client.vectors.bulk_insert_from_path(
            collection,
            str(bogus_path),
            dim=args.dim,
            expected_count=args.n_vectors,
            id_start=1,
        ),
    )

    # Case 2: wrong dim. The file is dim=args.dim but the request asks
    # for dim=args.dim // 2. The server must reject; it cannot infer the
    # right dim from a flat .f32 buffer alone, and even with the .npy
    # header present the server is expected to reject mismatched dims.
    wrong_dim = max(1, args.dim // 2)
    if wrong_dim == args.dim:
        wrong_dim = args.dim + 1
    _expect_sdk_failure(
        report,
        "negative.wrong_dim",
        f"declared dim={wrong_dim} against file dim={args.dim}",
        lambda: client.vectors.bulk_insert_from_path(
            collection,
            str(good_path),
            dim=wrong_dim,
            expected_count=args.n_vectors,
            id_start=1,
        ),
    )

    # Case 3: path outside SWARNDB_BULK_INSERT_ALLOWED_ROOTS.
    # The harness writes a temp file under a directory the server was
    # NOT told about and points the request at it. The server must
    # refuse to mmap a path outside its allow-list.
    outside_root = Path(tempfile.mkdtemp(prefix="swarndb_p00_outside_"))
    try:
        outside_path = outside_root / "outside.npy"
        write_npy_file(arr, outside_path)
        _expect_sdk_failure(
            report,
            "negative.outside_allowed_roots",
            (
                f"path {outside_path} is outside allowed_roots="
                f"{allowed_root}"
            ),
            lambda: client.vectors.bulk_insert_from_path(
                collection,
                str(outside_path),
                dim=args.dim,
                expected_count=args.n_vectors,
                id_start=1,
            ),
        )
    finally:
        try:
            shutil.rmtree(outside_root, ignore_errors=True)
        except Exception as exc:
            logger.warning("cleanup of outside_root %s failed: %s", outside_root, exc)

    # Case 4: '..' traversal attempt. Even if the prefix matches the
    # allowed root, the server's TOCTOU-safe walk (Decision 5.b) must
    # reject any '..' component before the open syscall fires.
    traversal_path = file_dir / ".." / file_dir.name / good_path.name
    _expect_sdk_failure(
        report,
        "negative.traversal_dotdot",
        f"path with '..' traversal {traversal_path}",
        lambda: client.vectors.bulk_insert_from_path(
            collection,
            str(traversal_path),
            dim=args.dim,
            expected_count=args.n_vectors,
            id_start=1,
        ),
    )

    # Case 5: relative path. Decision 5.b forces absolute paths only.
    # The harness deliberately strips the leading slash so the request
    # carries a relative path even though the underlying file exists.
    abs_str = str(good_path)
    rel_str = abs_str.lstrip("/") if abs_str.startswith("/") else abs_str
    if rel_str == abs_str:
        # On the unlikely OS where the good path is not absolute the
        # request would not exercise the relative-path branch; flag it
        # so the operator can investigate.
        report.add(
            "negative.relative_path",
            False,
            f"could not derive a relative form of {abs_str}",
        )
    else:
        _expect_sdk_failure(
            report,
            "negative.relative_path",
            f"relative path {rel_str}",
            lambda: client.vectors.bulk_insert_from_path(
                collection,
                rel_str,
                dim=args.dim,
                expected_count=args.n_vectors,
                id_start=1,
            ),
        )

    return report


# ---------------------------------------------------------------------------
# Allowed-roots derivation
# ---------------------------------------------------------------------------


def _resolve_data_dir(args: argparse.Namespace) -> Tuple[Path, bool]:
    """Pick the data directory for the run.

    If the user passed --data-dir, the harness uses it directly and
    only deletes it on exit when --keep-data is NOT set. Otherwise the
    harness creates a fresh tmpdir and removes it on exit unless
    --keep-data is set.
    """
    if args.data_dir:
        base = Path(args.data_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        owns = False
    else:
        base = Path(tempfile.mkdtemp(prefix="swarndb_p00_bulk_from_path_"))
        owns = True
    return base, owns


def _resolve_file_dir(args: argparse.Namespace) -> Tuple[Path, bool]:
    """File staging directory.

    All wire-format files (.npy + .f32) are written here. The server
    must include this directory in SWARNDB_BULK_INSERT_ALLOWED_ROOTS
    so the mmap path validation accepts the request.
    """
    base = Path(tempfile.mkdtemp(prefix="swarndb_p00_files_"))
    return base, True


def _cleanup_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:
        logger.warning("cleanup of %s failed: %s", path, exc)


def _allowed_roots_env(data_dir: Path, file_dir: Path) -> str:
    """SWARNDB_BULK_INSERT_ALLOWED_ROOTS value.

    Both the data dir and the file-staging dir must be allowed so the
    test fixture files land in a location the server will accept.
    The server parses this env var on ',' (see vf-server config), so
    we join with comma to keep each root a distinct entry.
    """
    parts = [str(data_dir.resolve()), str(file_dir.resolve())]
    # Some sandboxes run with the harness tempdir under /private/var on
    # macOS (resolved form of /var). Add both forms to dodge surprises.
    extra: List[str] = []
    for p in parts:
        rp = str(Path(p).resolve())
        if rp != p:
            extra.append(rp)
    return ",".join(dict.fromkeys(parts + extra))


# ---------------------------------------------------------------------------
# Top-level dispatch with server lifecycle
# ---------------------------------------------------------------------------


def _build_proc(
    args: argparse.Namespace,
    data_dir: Path,
    file_dir: Path,
    log_path: Path,
) -> SwarndbProcess:
    """SwarndbProcess factory closing over the resolved dirs."""
    allowed = _allowed_roots_env(data_dir, file_dir)
    logger.info(
        "SWARNDB_BULK_INSERT_ALLOWED_ROOTS resolved to: %s", allowed,
    )
    return SwarndbProcess(
        binary=Path(args.binary_path) if args.binary_path else Path(""),
        data_dir=data_dir,
        rest_port=args.rest_port,
        grpc_port=args.grpc_port,
        log_path=log_path,
        extra_env={"SWARNDB_BULK_INSERT_ALLOWED_ROOTS": allowed},
    )


def _run_with_server(
    args: argparse.Namespace,
    work,
) -> Tuple[bool, Dict[str, Any]]:
    """Spin up vf-server (or detect external), pass a connected client
    and the resolved file_dir into `work`, and tear everything down on
    exit.

    `work` is a callable taking (client, args, file_dir) and returning
    (ok: bool, payload: dict). The payload is propagated up so the
    caller can dump it to JSON.
    """
    data_dir, owns_data_dir = _resolve_data_dir(args)
    file_dir, owns_file_dir = _resolve_file_dir(args)
    log_path = data_dir.parent / "swarndb_p00_bulk_from_path.log"

    def _factory() -> SwarndbProcess:
        return _build_proc(args, data_dir, file_dir, log_path)

    _, _, proc = find_or_spawn_server(args, logger, _factory)
    args._spawned_proc = proc

    try:
        client = make_client(args.grpc_port)
        try:
            return work(client, args, file_dir, data_dir)
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
        args._spawned_proc = None
        if owns_file_dir and not args.keep_data:
            _cleanup_dir(file_dir)
        if owns_data_dir and not args.keep_data:
            _cleanup_dir(data_dir)
        if args.keep_data:
            logger.info(
                "--keep-data set; preserving data_dir=%s file_dir=%s",
                data_dir, file_dir,
            )


# ---------------------------------------------------------------------------
# Mode entry points
# ---------------------------------------------------------------------------


def _npy_work(
    client: SwarnDBClient,
    args: argparse.Namespace,
    file_dir: Path,
    data_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """`--mode npy` entry point. Runs the npy mode twice: once with the
    default id_start=1 and once with id_start=ID_START_VARIANT to cover
    the non-default range from the harness spec.
    """
    primary = f"{args.collection_prefix}_npy_default"
    variant = f"{args.collection_prefix}_npy_variant"

    rpt1 = run_npy(client, args, primary, file_dir, args.id_start)
    logger.info(rpt1.summary())
    drop_collection_quiet(client, primary)

    rpt2 = run_npy(client, args, variant, file_dir, ID_START_VARIANT)
    logger.info(rpt2.summary())
    drop_collection_quiet(client, variant)

    ok = rpt1.all_passed() and rpt2.all_passed()
    payload = {
        "npy_default": rpt1.as_payload(),
        "npy_variant": rpt2.as_payload(),
    }
    return ok, payload


def _f32_work(
    client: SwarnDBClient,
    args: argparse.Namespace,
    file_dir: Path,
    data_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """`--mode f32` entry point."""
    collection = f"{args.collection_prefix}_f32"
    rpt = run_f32(client, args, collection, file_dir, args.id_start)
    logger.info(rpt.summary())
    drop_collection_quiet(client, collection)
    return rpt.all_passed(), {"f32": rpt.as_payload()}


def _roundtrip_work(
    client: SwarnDBClient,
    args: argparse.Namespace,
    file_dir: Path,
    data_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """`--mode roundtrip` entry point."""
    collection = f"{args.collection_prefix}_roundtrip"
    rpt = run_roundtrip(client, args, collection, file_dir, args.id_start)
    logger.info(rpt.summary())
    drop_collection_quiet(client, collection)
    return rpt.all_passed(), {"roundtrip": rpt.as_payload()}


def _negative_work(
    client: SwarnDBClient,
    args: argparse.Namespace,
    file_dir: Path,
    data_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """`--mode negative` entry point."""
    collection = f"{args.collection_prefix}_negative"
    rpt = run_negative(client, args, collection, file_dir, data_dir)
    logger.info(rpt.summary())
    drop_collection_quiet(client, collection)
    return rpt.all_passed(), {"negative": rpt.as_payload()}


def _resolve_peak_compare_pid(
    args: argparse.Namespace,
) -> Tuple[Optional[int], bool, str]:
    """Pick the PID to sample for the P01 peak-RSS comparison.

    Returns (pid, owns_pid, reason). owns_pid=True means the harness
    spawned the vf-server itself and the RSS readings reflect the
    harness's own load. owns_pid=False means the harness is running in
    external mode (e.g. against a Docker container) and the readings
    may include unrelated workload; assertions are skipped in that
    case.
    """
    spawn_proc = getattr(args, "_spawned_proc", None)
    if spawn_proc is not None and spawn_proc.is_alive():
        return spawn_proc.pid(), True, "spawned vf-server"
    if args.external_pid is not None:
        return args.external_pid, False, "external_pid"
    return None, False, "no pid resolvable"


def _sample_rss_loop(
    pid: int,
    stop_event: threading.Event,
    samples: List[int],
    interval_seconds: float,
) -> None:
    """Background sampler thread body."""
    import psutil  # local import keeps the mode skippable

    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return
    while not stop_event.is_set():
        try:
            samples.append(proc.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if stop_event.wait(interval_seconds):
            break


def _baseline_rss_bytes(pid: int) -> int:
    """One-shot RSS sample used as the baseline before any inserts."""
    import psutil

    try:
        return int(psutil.Process(pid).memory_info().rss)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        raise RuntimeError(f"baseline RSS read failed: {exc}")


def _run_with_peak_sampler(
    pid: int,
    interval_seconds: float,
    fn,
) -> Tuple[Any, int, int]:
    """Run `fn` with an RSS sampler thread attached to `pid`.

    Returns (fn_result, peak_bytes, sample_count). If the sampler
    never sees a value (e.g. the pid vanished), peak_bytes is 0 and
    sample_count is 0; the caller must decide how to interpret that.
    """
    samples: List[int] = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=_sample_rss_loop,
        args=(pid, stop_event, samples, interval_seconds),
        name="bif-p01-rss-sampler",
        daemon=True,
    )
    sampler.start()
    try:
        result = fn()
    finally:
        stop_event.set()
        sampler.join(timeout=5.0)
    peak = max(samples) if samples else 0
    return result, peak, len(samples)


def run_peak_rss_compare(
    client: SwarnDBClient,
    args: argparse.Namespace,
    file_dir: Path,
    pid: Optional[int],
    owns_pid: bool,
    pid_reason: str,
) -> Tuple[HarnessReport, Dict[str, Any]]:
    """P01 peak-RSS comparison: same load with and without hint."""
    report = HarnessReport("peak_rss_compare")

    try:
        import psutil  # noqa: F401
    except ImportError:
        report.add(
            "peak_rss_compare.psutil_available",
            False,
            "psutil required for peak_rss_compare mode; skipping",
        )
        return report, {"skipped": True, "reason": "psutil not installed"}

    if pid is None:
        report.add(
            "peak_rss_compare.pid_resolvable",
            False,
            f"no pid resolvable ({pid_reason}); skipping",
        )
        return report, {"skipped": True, "reason": pid_reason}

    n = int(args.peak_compare_n)
    dim = int(args.peak_compare_dim)
    slack_mb = float(args.peak_compare_slack_mb)
    slack_bytes = int(slack_mb * 1024 * 1024)

    suffix = uuid.uuid4().hex[:8]
    coll_no_hint = f"bif_p01_no_hint_{suffix}"
    coll_with_hint = f"bif_p01_with_hint_{suffix}"

    arr = generate_dataset(n, dim, seed=args.seed + 11)
    npy_path = file_dir / f"bif_p01_peak_{suffix}.npy"
    write_npy_file(arr, npy_path)

    try:
        baseline_bytes = _baseline_rss_bytes(pid)
    except RuntimeError as exc:
        report.add(
            "peak_rss_compare.baseline_rss",
            False,
            f"baseline RSS read failed: {exc}",
        )
        return report, {"skipped": True, "reason": str(exc)}

    logger.info(
        "[MODE peak_rss_compare] pid=%d owns=%s reason=%s "
        "n=%d dim=%d slack_mb=%.1f baseline_mb=%.2f",
        pid, owns_pid, pid_reason, n, dim, slack_mb,
        baseline_bytes / (1024.0 * 1024.0),
    )

    ensure_collection(client, coll_no_hint, dim)

    def _call_without_hint():
        return client.vectors.bulk_insert_from_path(
            coll_no_hint,
            str(npy_path),
            dim=dim,
            expected_count=n,
            id_start=1,
            total_count_hint=0,
        )

    try:
        result_no_hint, peak_no_hint_bytes, samples_no_hint = (
            _run_with_peak_sampler(
                pid,
                PEAK_COMPARE_SAMPLE_INTERVAL_SECONDS,
                _call_without_hint,
            )
        )
    except SwarnDBError as exc:
        report.add(
            "peak_rss_compare.no_hint_call_succeeded",
            False,
            f"SDK raised on without-hint load: {exc}",
        )
        drop_collection_quiet(client, coll_no_hint)
        return report, {"skipped": True, "reason": f"no-hint load failed: {exc}"}

    inserted_no_hint = int(getattr(result_no_hint, "inserted_count", 0))
    report.add(
        "peak_rss_compare.no_hint_inserted",
        inserted_no_hint == n,
        f"inserted={inserted_no_hint} expected={n} samples={samples_no_hint}",
    )

    drop_collection_quiet(client, coll_no_hint)

    ensure_collection(client, coll_with_hint, dim)

    def _call_with_hint():
        return client.vectors.bulk_insert_from_path(
            coll_with_hint,
            str(npy_path),
            dim=dim,
            expected_count=n,
            id_start=1,
            total_count_hint=n,
        )

    try:
        result_with_hint, peak_with_hint_bytes, samples_with_hint = (
            _run_with_peak_sampler(
                pid,
                PEAK_COMPARE_SAMPLE_INTERVAL_SECONDS,
                _call_with_hint,
            )
        )
    except SwarnDBError as exc:
        report.add(
            "peak_rss_compare.with_hint_call_succeeded",
            False,
            f"SDK raised on with-hint load: {exc}",
        )
        drop_collection_quiet(client, coll_with_hint)
        return report, {"skipped": True, "reason": f"with-hint load failed: {exc}"}

    inserted_with_hint = int(getattr(result_with_hint, "inserted_count", 0))
    report.add(
        "peak_rss_compare.with_hint_inserted",
        inserted_with_hint == n,
        f"inserted={inserted_with_hint} expected={n} samples={samples_with_hint}",
    )

    drop_collection_quiet(client, coll_with_hint)

    if samples_no_hint == 0 or samples_with_hint == 0:
        report.add(
            "peak_rss_compare.samples_collected",
            False,
            (
                f"insufficient RSS samples: no_hint={samples_no_hint} "
                f"with_hint={samples_with_hint}"
            ),
        )

    mb = 1024.0 * 1024.0
    peak_no_hint_mb = peak_no_hint_bytes / mb
    peak_with_hint_mb = peak_with_hint_bytes / mb
    baseline_mb = baseline_bytes / mb
    delta_mb = peak_no_hint_mb - peak_with_hint_mb

    verdict_pass = peak_with_hint_bytes <= peak_no_hint_bytes + slack_bytes
    verdict = "PASS" if verdict_pass else "FAIL"

    banner = (
        "\n=== P01 peak-RSS comparison ===\n"
        f"  N: {n}, dim: {dim}\n"
        f"  baseline RSS: {baseline_mb:.2f} MB\n"
        f"  peak RSS (no total_count_hint): {peak_no_hint_mb:.2f} MB "
        f"(delta from baseline: {peak_no_hint_mb - baseline_mb:.2f} MB)\n"
        f"  peak RSS (with total_count_hint={n}): {peak_with_hint_mb:.2f} MB "
        f"(delta from baseline: {peak_with_hint_mb - baseline_mb:.2f} MB)\n"
        f"  delta (no_hint - with_hint): {delta_mb:.2f} MB\n"
        f"  slack: {slack_mb:.2f} MB\n"
        f"  verdict: {verdict}"
    )
    logger.info(banner)

    if owns_pid:
        report.add(
            "peak_rss_compare.with_hint_not_higher",
            verdict_pass,
            (
                f"peak_with_hint={peak_with_hint_mb:.2f} MB "
                f"peak_no_hint={peak_no_hint_mb:.2f} MB "
                f"slack={slack_mb:.2f} MB delta={delta_mb:.2f} MB"
            ),
        )
    else:
        report.add(
            "peak_rss_compare.assertion_skipped_external",
            True,
            (
                "running in external mode; RSS may include unrelated "
                "workload, assertion skipped. "
                f"peak_no_hint={peak_no_hint_mb:.2f} MB "
                f"peak_with_hint={peak_with_hint_mb:.2f} MB "
                f"delta={delta_mb:.2f} MB"
            ),
        )

    payload = {
        "skipped": False,
        "pid": pid,
        "owns_pid": owns_pid,
        "pid_reason": pid_reason,
        "n": n,
        "dim": dim,
        "slack_mb": slack_mb,
        "baseline_mb": baseline_mb,
        "peak_no_hint_mb": peak_no_hint_mb,
        "peak_with_hint_mb": peak_with_hint_mb,
        "delta_mb": delta_mb,
        "samples_no_hint": samples_no_hint,
        "samples_with_hint": samples_with_hint,
        "inserted_no_hint": inserted_no_hint,
        "inserted_with_hint": inserted_with_hint,
        "verdict": verdict,
    }

    try:
        npy_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("cleanup of %s failed: %s", npy_path, exc)

    return report, payload


def _peak_rss_compare_work(
    client: SwarnDBClient,
    args: argparse.Namespace,
    file_dir: Path,
    data_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """`--mode peak_rss_compare` entry point."""
    pid, owns_pid, pid_reason = _resolve_peak_compare_pid(args)
    report, payload = run_peak_rss_compare(
        client, args, file_dir, pid, owns_pid, pid_reason,
    )
    logger.info(report.summary())
    return report.all_passed(), {
        "peak_rss_compare": {
            **report.as_payload(),
            "metrics": payload,
        }
    }


def _all_work(
    client: SwarnDBClient,
    args: argparse.Namespace,
    file_dir: Path,
    data_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """`--mode all` entry point. Runs npy, f32, roundtrip, negative,
    and peak_rss_compare in sequence against fresh collections.
    """
    ok_npy, payload_npy = _npy_work(client, args, file_dir, data_dir)
    ok_f32, payload_f32 = _f32_work(client, args, file_dir, data_dir)
    ok_rt, payload_rt = _roundtrip_work(client, args, file_dir, data_dir)
    ok_neg, payload_neg = _negative_work(client, args, file_dir, data_dir)
    ok_peak, payload_peak = _peak_rss_compare_work(
        client, args, file_dir, data_dir,
    )

    overall = ok_npy and ok_f32 and ok_rt and ok_neg and ok_peak
    payload = {
        "mode": "all",
        "overall_pass": overall,
        **payload_npy,
        **payload_f32,
        **payload_rt,
        **payload_neg,
        **payload_peak,
    }
    return overall, payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SwarnDB mmap bulk-insert-from-path harness "
            "(memory-peak-reduction P00 Step 5)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "npy",
            "f32",
            "roundtrip",
            "negative",
            "peak_rss_compare",
            "all",
        ],
        default="all",
        help=(
            "Which mode to run. 'npy' covers the .npy auto-detect path "
            "with two id_start variants. 'f32' covers the flat float32 "
            "path. 'roundtrip' inserts via .npy and byte-compares 10 "
            "random GETs. 'negative' exercises path security and "
            "validation failures. 'peak_rss_compare' loads the same "
            "dataset twice (with and without total_count_hint) and "
            "compares peak RSS on the server PID. 'all' runs every "
            "mode in sequence against fresh collections."
        ),
    )
    parser.add_argument(
        "--binary-path",
        default=None,
        help=(
            "Path to the pre-built vf-server binary. The harness does "
            "NOT build. Optional: if a vf-server is already up on "
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
            "creates a tempdir and removes it on exit unless "
            "--keep-data is set."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help=f"Vector dimension (default {DEFAULT_DIM}).",
    )
    parser.add_argument(
        "--n-vectors",
        type=int,
        default=DEFAULT_N_VECTORS,
        help=f"Number of vectors per file (default {DEFAULT_N_VECTORS}).",
    )
    parser.add_argument(
        "--id-start",
        type=int,
        default=DEFAULT_ID_START,
        help=f"Starting id for sequential assignment (default {DEFAULT_ID_START}).",
    )
    parser.add_argument(
        "--collection-prefix",
        default=DEFAULT_COLLECTION_PREFIX,
        help=f"Collection name prefix (default '{DEFAULT_COLLECTION_PREFIX}').",
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
        "--output-json",
        default=None,
        help="Optional path to dump full results as JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for vector generation (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help=(
            "If set, do not remove the data_dir or file_dir on exit. "
            "Useful when triaging a failure to inspect what the server "
            "wrote to disk."
        ),
    )
    parser.add_argument(
        "--peak-compare-n",
        type=int,
        default=DEFAULT_PEAK_COMPARE_N,
        help=(
            "Number of vectors used by the peak_rss_compare mode "
            f"(default {DEFAULT_PEAK_COMPARE_N})."
        ),
    )
    parser.add_argument(
        "--peak-compare-dim",
        type=int,
        default=DEFAULT_PEAK_COMPARE_DIM,
        help=(
            "Vector dimension used by the peak_rss_compare mode "
            f"(default {DEFAULT_PEAK_COMPARE_DIM})."
        ),
    )
    parser.add_argument(
        "--peak-compare-slack-mb",
        type=float,
        default=DEFAULT_PEAK_COMPARE_SLACK_MB,
        help=(
            "Slack in MB applied to the peak_rss_compare assertion to "
            "absorb measurement noise on small N "
            f"(default {DEFAULT_PEAK_COMPARE_SLACK_MB})."
        ),
    )
    parser.add_argument(
        "--external-pid",
        type=int,
        default=None,
        help=(
            "Optional. PID of an externally managed vf-server (e.g. "
            "the swarndb container in a docker-compose stack). Used "
            "for log lines AND, when peak_rss_compare runs against an "
            "external server, as the PID the RSS sampler targets. In "
            "external mode peak_rss_compare records the peaks but "
            "skips the hard assertion since unrelated workload may "
            "share the process."
        ),
    )

    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> Optional[str]:
    """Return an error string for invalid CLI input, or None if ok."""
    if args.dim <= 0:
        return f"--dim must be positive (got {args.dim})"
    if args.n_vectors <= 0:
        return f"--n-vectors must be positive (got {args.n_vectors})"
    if args.id_start < 0:
        return f"--id-start must be non-negative (got {args.id_start})"
    if args.rest_port <= 0 or args.rest_port > 65535:
        return f"--rest-port out of range (got {args.rest_port})"
    if args.grpc_port <= 0 or args.grpc_port > 65535:
        return f"--grpc-port out of range (got {args.grpc_port})"
    if args.grpc_port == args.rest_port:
        return (
            f"--grpc-port ({args.grpc_port}) must differ from "
            f"--rest-port ({args.rest_port})"
        )
    if args.peak_compare_n <= 0:
        return f"--peak-compare-n must be positive (got {args.peak_compare_n})"
    if args.peak_compare_dim <= 0:
        return f"--peak-compare-dim must be positive (got {args.peak_compare_dim})"
    if args.peak_compare_slack_mb < 0:
        return (
            "--peak-compare-slack-mb must be non-negative "
            f"(got {args.peak_compare_slack_mb})"
        )
    return None


def _write_output_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("wrote results to %s", out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


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
        "memory-peak-reduction P00 harness: mmap bulk-insert-from-path "
        "(.npy + .f32 wire formats, id round-trip, path security)"
    )
    logger.info(
        "Workload: n_vectors=%d dim=%d id_start=%d seed=%d mode=%s",
        args.n_vectors, args.dim, args.id_start, args.seed, args.mode,
    )
    logger.info(
        "Targeting vf-server on rest_port=%d grpc_port=%d (auto-detect or spawn)",
        args.rest_port, args.grpc_port,
    )

    work_by_mode = {
        "npy": _npy_work,
        "f32": _f32_work,
        "roundtrip": _roundtrip_work,
        "negative": _negative_work,
        "peak_rss_compare": _peak_rss_compare_work,
        "all": _all_work,
    }
    work = work_by_mode[args.mode]

    try:
        ok, payload = _run_with_server(args, work)
    except SwarnDBError as exc:
        logger.error("[FAIL] SDK error: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("[FAIL] runtime error: %s", exc)
        return 2

    _write_output_json(args.output_json, payload)

    verdict = "PASS" if ok else "FAIL"
    logger.info(
        "\n%s\nHARNESS '%s': %s\n%s",
        "=" * 70, args.mode, verdict, "=" * 70,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
