# SwarnDB Python SDK

Official Python SDK for SwarnDB — the vector database that thinks in graphs.

Combines HNSW + IVF-PQ indexing with virtual graph traversal and 15+ vector math operations.

## Installation

```bash
pip install swarndb
```

## Quick Start

```python
from swarndb import SwarnClient

client = SwarnClient("localhost:50051")

# Create a collection
client.create_collection("my_vectors", dimension=128)

# Insert vectors
client.insert("my_vectors", vectors=[[0.1, 0.2, ...]], ids=["vec1"])

# Search
results = client.search("my_vectors", query=[0.1, 0.2, ...], top_k=10)
```

## Features

- HNSW + IVF-PQ hybrid indexing
- Virtual graph traversal
- 15+ vector math operations
- Sync and async gRPC clients
- NumPy integration

## License

ELv2
