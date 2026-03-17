# Getting Started with SwarnDB

This guide walks you through installing SwarnDB, creating your first collection, inserting vectors, running searches, and exploring the virtual graph. By the end, you will have a fully working SwarnDB instance with real data you can query.

## Prerequisites

You only need one thing: **Docker** installed on your machine.

If you do not have Docker yet, follow the official installation guide for your platform:
[https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

That is it. No compiler, no build tools, no extra dependencies.

## Installation

### Pull the image from GHCR

```bash
docker pull ghcr.io/sarthiai/swarndb:latest
```

### Run SwarnDB

```bash
docker run -d --name swarndb \
  -p 8080:8080 \
  -p 50051:50051 \
  -v swarndb_data:/data \
  ghcr.io/sarthiai/swarndb:latest
```

This starts SwarnDB in the background with:

- **Port 8080**: REST API
- **Port 50051**: gRPC API
- **swarndb_data volume**: Persistent storage for your collections and indexes

### Verify it is running

```bash
curl http://localhost:8080/health
```

You should see a response confirming the server is healthy:

```json
{"status": "healthy"}
```

## Using Docker Compose

For a more manageable setup, create a `docker-compose.yml` file:

```yaml
services:
  swarndb:
    image: ghcr.io/sarthiai/swarndb:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - swarndb_data:/data
    environment:
      SWARNDB_LOG_LEVEL: info
    restart: unless-stopped

volumes:
  swarndb_data:
```

Then start it:

```bash
docker compose up -d
```

To stop it later:

```bash
docker compose down
```

Your data is preserved in the `swarndb_data` volume and will be available when you start SwarnDB again.

## Your First Collection

A **collection** is where your vectors live. Think of it like a table in a traditional database, but optimized for similarity search.

### Step 1: Create a collection

Let's create a collection called `articles` with 384 dimensions and cosine distance:

```bash
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "articles",
    "dimension": 384,
    "distance_metric": "cosine"
  }'
```

Expected response:

```json
{
  "name": "articles",
  "dimension": 384,
  "distance_metric": "cosine"
}
```

### Step 2: Verify the collection was created

```bash
curl http://localhost:8080/api/v1/collections/articles
```

You should see the collection details returned, confirming it exists and is ready to accept vectors.

## Inserting Vectors

Now that you have a collection, let's add some vectors. In a real application, these vectors would come from an embedding model (such as `all-MiniLM-L6-v2` which produces 384-dimensional vectors). For this tutorial, we will use sample vectors.

### Insert a single vector

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "values": [0.12, -0.34, 0.56, 0.78, -0.21, 0.43, -0.65, 0.09, ...],
    "metadata": {
      "title": "Introduction to Neural Networks",
      "category": "science",
      "author": "Alice Chen",
      "year": 2024
    }
  }'
```

> **Note**: The `values` array must have exactly 384 elements to match the collection's dimension. The `...` above is a placeholder; your real request needs all 384 values. In practice, these values come from an embedding model such as `all-MiniLM-L6-v2`.

### Insert a few more vectors

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": 2,
    "values": [0.45, 0.12, -0.67, 0.33, 0.89, ...],
    "metadata": {
      "title": "Deep Learning for Image Recognition",
      "category": "science",
      "author": "Bob Martinez",
      "year": 2024
    }
  }'
```

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": 3,
    "values": [-0.23, 0.67, 0.11, -0.45, 0.34, ...],
    "metadata": {
      "title": "Quantum Computing Basics",
      "category": "science",
      "author": "Carol Zhang",
      "year": 2023
    }
  }'
```

### Bulk insert

For inserting many vectors at once, use the bulk insert endpoint. This is significantly faster than inserting one at a time:

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/vectors/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {
        "id": 4,
        "values": [0.55, -0.11, 0.22, 0.44, -0.33, ...],
        "metadata": {
          "title": "The Future of Renewable Energy",
          "category": "environment",
          "author": "David Park",
          "year": 2024
        }
      },
      {
        "id": 5,
        "values": [0.33, 0.77, -0.44, 0.11, 0.66, ...],
        "metadata": {
          "title": "Machine Learning in Healthcare",
          "category": "science",
          "author": "Eva Singh",
          "year": 2024
        }
      },
      {
        "id": 6,
        "values": [-0.15, 0.42, 0.63, -0.28, 0.51, ...],
        "metadata": {
          "title": "Climate Change and Ocean Ecosystems",
          "category": "environment",
          "author": "Frank Liu",
          "year": 2023
        }
      }
    ]
  }'
```

## Your First Search

Now that you have vectors in your collection, let's search for similar articles.

### Basic search

Find the 5 most similar articles to a query vector:

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.10, -0.30, 0.50, 0.70, -0.20, ...],
    "k": 5
  }'
```

Expected response:

```json
{
  "results": [
    {
      "id": 1,
      "score": 0.0023,
      "metadata": {
        "title": "Introduction to Neural Networks",
        "category": "science",
        "author": "Alice Chen",
        "year": 2024
      }
    },
    {
      "id": 5,
      "score": 0.1247,
      "metadata": {
        "title": "Machine Learning in Healthcare",
        "category": "science",
        "author": "Eva Singh",
        "year": 2024
      }
    }
  ]
}
```

### Search with metadata filter

You can narrow down your search results using metadata filters. For example, find similar articles only in the "science" category from 2024:

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.10, -0.30, 0.50, 0.70, -0.20, ...],
    "k": 5,
    "filter": {
      "and": [
        {"field": "category", "op": "eq", "value": "science"},
        {"field": "year", "op": "gte", "value": 2024}
      ]
    }
  }'
```

### Understanding the response

Each search result contains three fields:

- **id**: The unique identifier of the matching vector.
- **score**: The distance between the query vector and the result. With cosine distance, lower scores mean higher similarity. A score of 0.0 means the vectors are identical.
- **metadata**: The key-value pairs you attached when inserting the vector.

Results are sorted by score in ascending order (most similar first).

## Exploring the Graph

SwarnDB's **virtual graph** automatically discovers relationships between your vectors based on similarity. This lets you traverse connections between data points, similar to a graph database, but built entirely from vector proximity.

### Step 1: Set a similarity threshold

Before the graph can find relationships, you need to set a similarity threshold. This defines how similar two vectors must be to form a connection:

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/graph/threshold \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.85
  }'
```

A threshold of 0.85 means vectors with cosine similarity of 0.85 or higher will be connected as related.

### Step 2: Optimize the collection

After setting the threshold, run optimize to build the graph edges:

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/optimize
```

### Step 3: Get related vectors

Now you can find vectors that are related to a specific one:

```bash
curl http://localhost:8080/api/v1/collections/articles/graph/related/1
```

This returns all vectors connected to vector `1` in the virtual graph, along with their similarity scores.

### Step 4: Traverse the graph

For deeper exploration, use the traverse endpoint to walk multiple hops through the graph:

```bash
curl -X POST http://localhost:8080/api/v1/collections/articles/graph/traverse \
  -H "Content-Type: application/json" \
  -d '{
    "start_id": 1,
    "depth": 2,
    "max_results": 10
  }'
```

This starts at vector `1` and follows connections up to 2 hops deep, returning up to 10 related vectors. This is powerful for discovering indirect relationships. For example, if article A is related to article B, and article B is related to article C, a 2-hop traversal from A will find C even if A and C are not directly similar.

## Using the Python SDK

SwarnDB provides a Python SDK that communicates over gRPC for high performance.

### Install the SDK

```bash
pip install swarndb
```

### Synchronous client

```python
from swarndb import SwarnDBClient

# Connect to SwarnDB
with SwarnDBClient("localhost", 50051) as client:

    # Create a collection
    client.collections.create(
        name="articles",
        dimension=384,
        distance_metric="cosine"
    )

    # Insert a vector
    client.vectors.insert(
        collection="articles",
        id=1,
        vector=[0.12, -0.34, 0.56, ...],  # 384 dimensions
        metadata={
            "title": "Introduction to Neural Networks",
            "category": "science",
            "author": "Alice Chen",
            "year": 2024
        }
    )

    # Search for similar vectors
    results = client.search.query(
        collection="articles",
        vector=[0.10, -0.30, 0.50, ...],  # 384 dimensions
        k=5
    )

    for result in results:
        print(f"{result.id}: {result.score:.4f}")
        print(f"  Title: {result.metadata['title']}")

    # Set graph threshold and explore relationships
    client.graph.set_threshold(
        collection="articles",
        threshold=0.85
    )

    related = client.graph.get_related(
        collection="articles",
        vector_id=1
    )

    for neighbor in related:
        print(f"Related: {neighbor.id} (score: {neighbor.score:.4f})")
```

### Asynchronous client

For applications that use `asyncio`, SwarnDB provides an async client:

```python
import asyncio
from swarndb import AsyncSwarnDBClient

async def main():
    async with AsyncSwarnDBClient("localhost", 50051) as client:

        # Create a collection
        await client.collections.create(
            name="articles",
            dimension=384,
            distance_metric="cosine"
        )

        # Insert vectors
        await client.vectors.insert(
            collection="articles",
            id=1,
            vector=[0.12, -0.34, 0.56, ...],
            metadata={
                "title": "Introduction to Neural Networks",
                "category": "science",
                "author": "Alice Chen",
                "year": 2024
            }
        )

        # Search
        results = await client.search.query(
            collection="articles",
            vector=[0.10, -0.30, 0.50, ...],
            k=5
        )

        for result in results:
            print(f"{result.id}: {result.score:.4f}")

asyncio.run(main())
```

Both clients support the full SwarnDB API: collections, vectors, search, graph, and math operations.

## Next Steps

Now that you have SwarnDB running and know the basics, explore these topics:

- **[Core Concepts](core-concepts.md)**: Understand how HNSW indexing, the virtual graph, and metadata filtering work under the hood.
- **[API Reference](api-reference.md)**: Complete documentation for every REST and gRPC endpoint.
- **[Python SDK](python-sdk.md)**: Full reference for the Python client, including advanced features like batch operations and graph-enriched search.
- **[Configuration](configuration.md)**: Tune SwarnDB for your workload with environment variables and runtime settings.
- **[Deployment Guide](deployment.md)**: Run SwarnDB in production with Docker Compose, Kubernetes, or Helm.

For questions, issues, or contributions, visit the GitHub repository:
[https://github.com/SarthiAI/swarndb](https://github.com/SarthiAI/swarndb)
