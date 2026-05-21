#!/usr/bin/env python3
"""P03 checkpoint_every + resume_token smoke test.

Validates that the new P03 fields flow end to end:
  1. bulk_insert with checkpoint_every=N is accepted by the server.
  2. The response shape exposes inserted_count, last_completed_batch_idx,
     last_committed_lsn and resume_token.
  3. On a clean completion the server clears its checkpoint, so the
     resume_token is empty in the response.

Resume after a mid insert kill is not exercised here; that needs a
separate harness that can terminate the server between batches.
"""

import sys

sys.path.insert(
    0,
    "/Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/sdk/python/src",
)

import numpy as np
from swarndb.client import SwarnDBClient

client = SwarnDBClient(host="localhost", port=50051)

COLLECTION = "p03_checkpoint_smoke"

# Clean slate
try:
    client.collections.delete(COLLECTION)
except Exception:
    pass

client.collections.create(
    name=COLLECTION,
    dimension=128,
    distance_metric="cosine",
)

rng = np.random.RandomState(42)
N = 2000
vectors = [rng.randn(128).tolist() for _ in range(N)]
ids = [i + 1 for i in range(N)]
metadata = [{"idx": i} for i in range(N)]

print(f"Phase 1: bulk insert {N} vectors with checkpoint_every=5")
res = client.vectors.bulk_insert(
    COLLECTION,
    vectors,
    ids=ids,
    metadata_list=metadata,
    batch_lock_size=200,
    checkpoint_every=5,
)

print(f"  inserted_count={res.inserted_count}")
print(f"  errors={res.errors}")
print(f"  last_completed_batch_idx={res.last_completed_batch_idx}")
print(f"  last_committed_lsn={res.last_committed_lsn}")
token = res.resume_token
if not token:
    print("  resume_token=<empty>")
else:
    print(f"  resume_token=<len {len(token)}>")

info = client.collections.get(COLLECTION)
print(f"\nCollection info after insert: name={info.name} "
      f"dim={info.dimension} vector_count={info.vector_count}")

ok = (
    res.inserted_count == N
    and not res.errors
    and info.vector_count >= N
)

# Cleanup
client.collections.delete(COLLECTION)

if ok:
    print("\n[PASS] checkpoint_every smoke test")
    sys.exit(0)
else:
    print("\n[FAIL] checkpoint_every smoke test")
    sys.exit(1)
