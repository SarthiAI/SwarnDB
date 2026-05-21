#!/usr/bin/env python3
"""Download DBPedia 1M (1536-dim OpenAI embeddings) into a numpy file.

Source: HuggingFace dataset KShivendu/dbpedia-entities-openai-1M.
The dataset ships with an "openai" column holding 1536-float embeddings
(stored as parquet). This script materialises them into
``data/train_vectors.npy`` as a contiguous float32 array of shape
(N, 1536), and writes ``data/train_ids.npy`` with 1-indexed ids.

Idempotent: if the output files already exist with the expected shape,
the download is skipped.

Prerequisites:
    pip install datasets huggingface_hub numpy

CLI:
    python download_dataset.py [--data-dir DIR] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


DATASET_NAME = "KShivendu/dbpedia-entities-openai-1M"
EXPECTED_DIM = 1536
EXPECTED_N = 1_000_000


def _candidate_embedding_columns(features) -> list:
    """Return column names that look like embedding columns."""
    candidates = []
    for name in features:
        lower = name.lower()
        if "openai" in lower or "embedding" in lower or "vector" in lower:
            candidates.append(name)
    return candidates


def _existing_is_valid(vec_path: Path, dim: int, min_n: int) -> bool:
    """Return True if vec_path already holds a valid array."""
    if not vec_path.exists():
        return False
    try:
        arr = np.load(vec_path, mmap_mode="r")
    except Exception as exc:
        print(f"[warn] could not read existing {vec_path}: {exc}")
        return False
    if arr.ndim != 2 or arr.shape[1] != dim:
        print(
            f"[warn] existing {vec_path} has shape {arr.shape}, "
            f"expected (*, {dim}); will re-download"
        )
        return False
    if arr.shape[0] < min_n:
        print(
            f"[warn] existing {vec_path} has only {arr.shape[0]} rows, "
            f"need at least {min_n}; will re-download"
        )
        return False
    print(
        f"[skip] {vec_path} already exists with shape {arr.shape}; "
        f"download skipped"
    )
    return True


def download(data_dir: Path, limit: int | None) -> int:
    """Download DBPedia 1M, return number of vectors written."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[fatal] the 'datasets' package is not installed. "
            "Run: pip install datasets huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(2)

    data_dir.mkdir(parents=True, exist_ok=True)
    vec_path = data_dir / "train_vectors.npy"
    ids_path = data_dir / "train_ids.npy"

    target_n = limit if limit is not None else EXPECTED_N
    if _existing_is_valid(vec_path, EXPECTED_DIM, target_n):
        arr = np.load(vec_path, mmap_mode="r")
        return int(arr.shape[0])

    print(f"[info] loading dataset {DATASET_NAME} (split=train)")
    ds = load_dataset(DATASET_NAME, split="train")
    n_total = len(ds)
    print(f"[info] dataset has {n_total} rows, columns: {list(ds.features)}")

    embed_cols = _candidate_embedding_columns(ds.features)
    if not embed_cols:
        print(
            f"[fatal] no embedding-like column found in {list(ds.features)}",
            file=sys.stderr,
        )
        sys.exit(3)
    embed_col = embed_cols[0]
    print(f"[info] using embedding column: {embed_col}")

    n_take = min(target_n, n_total)
    if limit is not None and limit < n_total:
        ds = ds.select(range(limit))
        print(f"[info] limited to first {limit} rows")

    # Pre-allocate the full output up front; this bounds RSS to one final
    # buffer (N * dim * 4 bytes) plus a single batch of float conversions
    # rather than ballooning to a million Python lists of Python floats.
    bytes_full = n_take * EXPECTED_DIM * 4
    print(
        f"[info] materialising {n_take} embeddings to float32 array "
        f"(target buffer ~{bytes_full / (1024 ** 3):.2f} GiB)"
    )
    arr = np.empty((n_take, EXPECTED_DIM), dtype=np.float32)

    batch_size = 10_000
    written = 0
    for start in range(0, n_take, batch_size):
        end = min(start + batch_size, n_take)
        # ds.select returns a lightweight view; only this batch's column is
        # converted to a numpy array per iteration, keeping peak RSS bounded.
        batch = ds.select(range(start, end))
        chunk = np.asarray(batch[embed_col], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != EXPECTED_DIM:
            print(
                f"[fatal] batch [{start}:{end}] has shape {chunk.shape}, "
                f"expected (*, {EXPECTED_DIM})",
                file=sys.stderr,
            )
            sys.exit(4)
        if chunk.shape[0] != (end - start):
            print(
                f"[fatal] batch [{start}:{end}] returned {chunk.shape[0]} rows, "
                f"expected {end - start}",
                file=sys.stderr,
            )
            sys.exit(4)
        arr[start:end] = chunk
        written = end
        # Rough live estimate: filled portion of the output plus one batch buffer.
        filled_gib = (written * EXPECTED_DIM * 4) / (1024 ** 3)
        batch_gib = (chunk.nbytes) / (1024 ** 3)
        print(
            f"[stream] rows {written}/{n_take} "
            f"(~{filled_gib:.2f} GiB filled, +{batch_gib:.2f} GiB batch)",
            flush=True,
        )
        # Drop references so the next iteration's batch can reuse the memory.
        del chunk
        del batch

    if arr.shape != (n_take, EXPECTED_DIM):
        print(
            f"[fatal] materialised array has shape {arr.shape}, "
            f"expected ({n_take}, {EXPECTED_DIM})",
            file=sys.stderr,
        )
        sys.exit(4)

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    print(f"[info] writing {vec_path} (shape={arr.shape}, dtype={arr.dtype})")
    np.save(vec_path, arr)

    ids_col = None
    for name in ("id", "ids", "_id"):
        if name in ds.features:
            ids_col = name
            break
    if ids_col is not None:
        try:
            ids = np.asarray(ds[ids_col], dtype=np.int64)
            print(f"[info] writing {ids_path} from column '{ids_col}'")
            np.save(ids_path, ids)
        except Exception as exc:
            print(f"[warn] failed to extract ids from '{ids_col}': {exc}")
            ids_col = None
    if ids_col is None:
        ids = np.arange(1, arr.shape[0] + 1, dtype=np.int64)
        print(f"[info] writing sequential 1-indexed ids to {ids_path}")
        np.save(ids_path, ids)

    vec_mb = vec_path.stat().st_size / (1024 * 1024)
    ids_mb = ids_path.stat().st_size / (1024 * 1024)
    print(f"[done] {vec_path} ({vec_mb:.1f} MB)")
    print(f"[done] {ids_path} ({ids_mb:.1f} MB)")
    return int(arr.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download DBPedia 1M OpenAI embeddings to numpy.",
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "data",
        help="output directory (default: <script_dir>/data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional row cap for smaller sanity runs",
    )
    args = parser.parse_args()

    n = download(args.data_dir, args.limit)
    print(f"[ok] {n} vectors available at {args.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
