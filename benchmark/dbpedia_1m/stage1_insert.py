#!/usr/bin/env python3
"""DBpedia 1M streaming bulk insert reference loader.

Loads ``data/train_vectors.npy`` (and ``data/train_ids.npy`` if present),
connects to a local SwarnDB server on localhost:50051, recreates the
target collection, then bulk-inserts every vector in fixed-size chunks
while logging per-chunk wall time and progress every 50k rows.

Outputs a PASS / FAIL line on the last stdout row and exits with
status 0 on success, 1 on failure.

For very large loads, prefer the file-based ``bulk_insert_from_path``
API (see the Bulk Insert From a File section in docs/python-sdk.md),
which keeps server memory bounded by the index being built.

CLI:
    python stage1_insert.py [--data-dir DIR] [--collection-name NAME]
                            [--dimension 1536] [--batch-size 5000]
                            [--limit N] [--checkpoint-every K]
                            [--defer-graph]
                            [--docker-container NAME]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SDK = SCRIPT_DIR.parent.parent / "sdk" / "python" / "src"

# Wire the local Python SDK source onto sys.path so the script runs
# from a fresh repo checkout without an SDK wheel install.
if DEFAULT_SDK.is_dir():
    sys.path.insert(0, str(DEFAULT_SDK))

from swarndb.client import SwarnDBClient  # noqa: E402


PROGRESS_INTERVAL = 50_000
DOCKER_MARKERS = (250_000, 500_000, 750_000, 1_000_000)


def _load_arrays(data_dir: Path, limit: int | None):
    """Load vectors and ids arrays from data_dir."""
    vec_path = data_dir / "train_vectors.npy"
    ids_path = data_dir / "train_ids.npy"
    if not vec_path.exists():
        print(
            f"[fatal] {vec_path} not found; run download_dataset.py first",
            file=sys.stderr,
        )
        sys.exit(2)

    vectors = np.load(vec_path, mmap_mode="r")
    if ids_path.exists():
        ids = np.load(ids_path)
    else:
        ids = np.arange(1, vectors.shape[0] + 1, dtype=np.int64)

    if limit is not None and limit < vectors.shape[0]:
        vectors = vectors[:limit]
        ids = ids[:limit]
    return vectors, ids


def _docker_stats_line(container: str | None) -> str | None:
    """Best-effort docker stats single-shot. Returns None on failure."""
    if not container:
        return None
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.run(
            [
                "docker", "stats", "--no-stream", "--format",
                "{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}",
                container,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip()
        return line or None
    except Exception:
        return None


def _maybe_emit_docker(
    container: str | None,
    inserted: int,
    fired: set,
) -> None:
    """Emit a docker stats line when crossing a marker threshold."""
    for marker in DOCKER_MARKERS:
        if marker in fired:
            continue
        if inserted >= marker:
            fired.add(marker)
            line = _docker_stats_line(container)
            if line:
                print(f"[docker @{marker}] {line}")
            else:
                print(
                    f"[docker @{marker}] stats unavailable "
                    f"(container={container or 'n/a'})"
                )


def run(args: argparse.Namespace) -> int:
    data_dir = args.data_dir
    vectors, ids = _load_arrays(data_dir, args.limit)
    n_total, dim = vectors.shape
    if dim != args.dimension:
        print(
            f"[fatal] vectors have dim {dim}, expected {args.dimension}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[info] loaded {n_total} vectors (dim={dim}) "
        f"from {data_dir / 'train_vectors.npy'}"
    )

    client = SwarnDBClient(host=args.host, port=args.port)

    print(f"[info] resetting collection '{args.collection_name}'")
    try:
        client.collections.delete(args.collection_name)
    except Exception:
        pass
    client.collections.create(
        name=args.collection_name,
        dimension=args.dimension,
        distance_metric="cosine",
    )

    batch_size = args.batch_size
    n_batches = (n_total + batch_size - 1) // batch_size
    print(
        f"[info] inserting in {n_batches} chunks of {batch_size} "
        f"(defer_graph={args.defer_graph}, "
        f"checkpoint_every={args.checkpoint_every})"
    )

    inserted_total = 0
    errors_total = 0
    fired_markers: set = set()
    next_progress = PROGRESS_INTERVAL
    t_run_start = time.perf_counter()

    for batch_idx in range(n_batches):
        lo = batch_idx * batch_size
        hi = min(lo + batch_size, n_total)
        chunk = vectors[lo:hi]
        chunk_ids = ids[lo:hi]

        chunk_list = [row.tolist() for row in np.asarray(chunk)]
        chunk_ids_list = [int(v) for v in np.asarray(chunk_ids)]

        kwargs = {"ids": chunk_ids_list}
        if args.defer_graph:
            kwargs["defer_graph"] = True
        if args.checkpoint_every and args.checkpoint_every > 0:
            kwargs["checkpoint_every"] = args.checkpoint_every

        t_chunk = time.perf_counter()
        try:
            res = client.vectors.bulk_insert(
                args.collection_name,
                chunk_list,
                **kwargs,
            )
        except Exception as exc:
            print(
                f"[fatal] bulk_insert failed at batch {batch_idx} "
                f"(rows {lo}..{hi}): {exc}",
                file=sys.stderr,
            )
            return 1
        dt_chunk = time.perf_counter() - t_chunk

        inserted_total += int(res.inserted_count)
        errors_total += len(res.errors)
        chunk_rate = (hi - lo) / dt_chunk if dt_chunk > 0 else float("inf")
        print(
            f"[chunk {batch_idx + 1}/{n_batches}] rows={hi - lo} "
            f"t={dt_chunk:.2f}s rate={chunk_rate:,.0f} vec/s "
            f"errors={len(res.errors)}"
        )

        _maybe_emit_docker(args.docker_container, inserted_total, fired_markers)

        if inserted_total >= next_progress:
            elapsed = time.perf_counter() - t_run_start
            overall = inserted_total / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] inserted={inserted_total:,} "
                f"elapsed={elapsed:.1f}s overall={overall:,.0f} vec/s"
            )
            while inserted_total >= next_progress:
                next_progress += PROGRESS_INTERVAL

    total_elapsed = time.perf_counter() - t_run_start
    overall_rate = inserted_total / total_elapsed if total_elapsed > 0 else 0.0

    try:
        info = client.collections.get(args.collection_name)
        vector_count = info.vector_count
    except Exception as exc:
        print(f"[warn] could not fetch collection info: {exc}")
        vector_count = -1

    print("")
    print("==== Stage 1 summary ====")
    print(f"collection         : {args.collection_name}")
    print(f"vectors requested  : {n_total:,}")
    print(f"vectors inserted   : {inserted_total:,}")
    print(f"errors             : {errors_total}")
    print(f"wall time          : {total_elapsed:.2f} s")
    print(f"throughput         : {overall_rate:,.0f} vec/s")
    print(f"collection.vector_count : {vector_count}")

    ok = (
        inserted_total == n_total
        and errors_total == 0
        and (vector_count == -1 or vector_count >= n_total)
    )
    if ok:
        print("[PASS] stage1_insert")
        return 0
    print("[FAIL] stage1_insert")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DBPedia 1M stage 1: bulk insert into SwarnDB.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "data",
        help="input directory holding train_vectors.npy",
    )
    parser.add_argument(
        "--collection-name",
        default="dbpedia_1m",
        help="target collection name (default: dbpedia_1m)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help="vector dimension (default: 1536)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="rows per bulk_insert call (default: 5000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional cap on rows for smoke runs",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="checkpoint cadence (0 disables; default 0)",
    )
    parser.add_argument(
        "--defer-graph",
        action="store_true",
        help="defer HNSW graph build until later",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="SwarnDB host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="SwarnDB gRPC port (default: 50051)",
    )
    parser.add_argument(
        "--docker-container",
        default=os.environ.get("SWARNDB_DOCKER_NAME"),
        help="optional docker container name for inline stats markers",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
