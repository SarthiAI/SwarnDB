# API Reference

SwarnDB exposes two API surfaces: a **REST API** on port 8080 and a **gRPC API** on port 50051. This document covers every endpoint, every parameter, and every response field.

- **REST base URL:** `http://localhost:8080/api/v1`
- **gRPC endpoint:** `localhost:50051`
- **Proto definitions:** `proto/swarndb/v1/`
- **Authentication:** `X-API-Key` header or `Authorization: Bearer <token>` header (only enforced when `SWARNDB_API_KEYS` is set)
- **Error format:** `{"error": "message", "code": 400}`

All REST endpoints return JSON. All request bodies are JSON (`Content-Type: application/json`).

---

## Table of Contents

- [Health and Readiness](#health-and-readiness)
- [Collections](#collections)
- [Vectors](#vectors)
- [Search](#search)
- [Filter Expressions](#filter-expressions)
- [Graph](#graph)
- [Vector Math](#vector-math)
- [gRPC API](#grpc-api)
- [Metadata Types](#metadata-types)
- [Authentication](#authentication)

---

## Health and Readiness

### GET /health

Basic health check. Always returns 200 while the server is running.

**Response:**

| Field    | Type   | Description                      |
|----------|--------|----------------------------------|
| status   | string | Always `"ok"`                    |
| version  | string | Server version (e.g., `"0.1.0"`) |

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### GET /ready

Readiness check. Returns collection count and total vector count.

**Response:**

| Field         | Type   | Description                       |
|---------------|--------|-----------------------------------|
| ready         | bool   | Always `true` when server is up   |
| collections   | uint   | Number of loaded collections      |
| total_vectors | uint64 | Total vectors across all collections |

```bash
curl http://localhost:8080/ready
```

```json
{
  "ready": true,
  "collections": 3,
  "total_vectors": 150000
}
```

### GET /healthz

Kubernetes liveness probe. Returns 200 as long as the process is alive.

**Response:**

| Field  | Type   | Description      |
|--------|--------|------------------|
| status | string | Always `"alive"` |

```bash
curl http://localhost:8080/healthz
```

### GET /readyz

Kubernetes readiness probe. Returns 200 when all checks pass, 503 otherwise.

**Response:**

| Field  | Type   | Description                          |
|--------|--------|--------------------------------------|
| status | string | `"ready"` or `"not_ready"`           |
| checks | object | Map of check names to status strings |

Checks performed:
- `collections_accessible`: can acquire a read lock on the collections map
- `collections_loaded`: at least one collection exists, or server just started

```bash
curl http://localhost:8080/readyz
```

```json
{
  "status": "ready",
  "checks": {
    "collections_accessible": "ok",
    "collections_loaded": "ok"
  }
}
```

### GET /startupz

Kubernetes startup probe. Returns 200 once server initialization is complete, 503 while still starting.

**Response:**

| Field  | Type   | Description                        |
|--------|--------|------------------------------------|
| status | string | `"started"` or `"starting"`        |

```bash
curl http://localhost:8080/startupz
```

### GET /metrics

Returns Prometheus-format metrics for monitoring. No authentication required.

**Response:** Plain text in Prometheus exposition format.

```bash
curl http://localhost:8080/metrics
```

---

## Collections

### Create Collection

```text
POST /api/v1/collections
```

Creates a new vector collection.

**Request Body:**

| Parameter         | Type   | Required | Default    | Description                                                          |
|-------------------|--------|----------|------------|----------------------------------------------------------------------|
| name              | string | Yes      |            | Collection name. Must be alphanumeric with underscores/hyphens.      |
| dimension         | uint32 | Yes      |            | Vector dimensionality. Must be greater than 0.                       |
| distance_metric   | string | No       | `"cosine"` | One of: `"cosine"`, `"euclidean"`, `"dot_product"`, `"manhattan"`    |
| default_threshold | float  | No       | `0.0`      | Default similarity threshold for the virtual graph. 0 means no graph edges are auto-computed. |
| max_vectors       | uint64 | No       | `0`        | Maximum number of vectors. 0 means unlimited.                        |

**Response:**

| Field   | Type   | Description              |
|---------|--------|--------------------------|
| name    | string | Name of the created collection |
| success | bool   | `true` on success        |

**Status Codes:** 200 (success), 400 (invalid parameters), 409 (collection already exists), 500 (storage error)

```bash
curl -X POST http://localhost:8080/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "documents",
    "dimension": 1536,
    "distance_metric": "cosine",
    "default_threshold": 0.8
  }'
```

```json
{
  "name": "documents",
  "success": true
}
```

### List Collections

```text
GET /api/v1/collections
```

Returns all collections and their metadata.

**Response:**

| Field       | Type  | Description              |
|-------------|-------|--------------------------|
| collections | array | Array of collection info objects |

Each collection object:

| Field             | Type   | Description                                                    |
|-------------------|--------|----------------------------------------------------------------|
| name              | string | Collection name                                                |
| dimension         | uint32 | Vector dimensionality                                          |
| distance_metric   | string | Distance metric in use                                         |
| vector_count      | uint64 | Number of vectors stored                                       |
| default_threshold | float  | Default similarity threshold                                   |
| status            | string | One of: `"ready"`, `"pending_optimization"`, `"optimizing"`    |

```bash
curl http://localhost:8080/api/v1/collections
```

```json
{
  "collections": [
    {
      "name": "documents",
      "dimension": 1536,
      "distance_metric": "cosine",
      "vector_count": 50000,
      "default_threshold": 0.8,
      "status": "ready"
    }
  ]
}
```

### Get Collection

```text
GET /api/v1/collections/{name}
```

Returns metadata for a single collection.

**Path Parameters:**

| Parameter | Type   | Description     |
|-----------|--------|-----------------|
| name      | string | Collection name |

**Response:** Same fields as a single collection object in the list response.

**Status Codes:** 200 (success), 404 (not found)

```bash
curl http://localhost:8080/api/v1/collections/documents
```

```json
{
  "name": "documents",
  "dimension": 1536,
  "distance_metric": "cosine",
  "vector_count": 50000,
  "default_threshold": 0.8,
  "status": "ready"
}
```

### Delete Collection

```text
DELETE /api/v1/collections/{name}
```

Permanently deletes a collection and all its vectors.

**Path Parameters:**

| Parameter | Type   | Description     |
|-----------|--------|-----------------|
| name      | string | Collection name |

**Response:**

| Field   | Type | Description         |
|---------|------|---------------------|
| success | bool | `true` on success   |

**Status Codes:** 200 (success), 404 (not found), 500 (storage error)

```bash
curl -X DELETE http://localhost:8080/api/v1/collections/documents
```

```json
{
  "success": true
}
```

---

## Vectors

### Insert Vector

```text
POST /api/v1/collections/{collection}/vectors
```

Inserts a single vector into the collection.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter | Type        | Required | Default | Description                                      |
|-----------|-------------|----------|---------|--------------------------------------------------|
| id        | uint64      | No       | `0`     | Vector ID. 0 means auto-assign.                  |
| values    | float array | Yes      |         | Vector values. Length must match collection dimension. |
| metadata  | object      | No       | `null`  | Key-value metadata (see [Metadata Types](#metadata-types)). |

**Response:**

| Field   | Type   | Description                |
|---------|--------|----------------------------|
| id      | uint64 | Assigned or provided vector ID |
| success | bool   | `true` on success          |

**Status Codes:** 200 (success), 400 (dimension mismatch, invalid data), 404 (collection not found), 409 (duplicate ID), 500 (storage error)

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "values": [0.1, 0.2, 0.3],
    "metadata": {
      "title": "Introduction to Vector Databases",
      "category": "technology",
      "year": 2024
    }
  }'
```

```json
{
  "id": 1,
  "success": true
}
```

### Get Vector

```text
GET /api/v1/collections/{collection}/vectors/{id}
```

Retrieves a vector by ID.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |
| id         | uint64 | Vector ID       |

**Response:**

| Field    | Type        | Description                     |
|----------|-------------|---------------------------------|
| id       | uint64      | Vector ID                       |
| values   | float array | Vector values                   |
| metadata | object      | Metadata (omitted if none set)  |

**Status Codes:** 200 (success), 404 (collection or vector not found)

```bash
curl http://localhost:8080/api/v1/collections/documents/vectors/1
```

```json
{
  "id": 1,
  "values": [0.1, 0.2, 0.3],
  "metadata": {
    "title": "Introduction to Vector Databases",
    "category": "technology",
    "year": 2024
  }
}
```

### Update Vector

```text
PUT /api/v1/collections/{collection}/vectors/{id}
```

Updates a vector's values, metadata, or both. At least one of `values` or `metadata` must be provided.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |
| id         | uint64 | Vector ID       |

**Request Body:**

| Parameter | Type        | Required | Description                          |
|-----------|-------------|----------|--------------------------------------|
| values    | float array | No       | New vector values. Must match dimension. |
| metadata  | object      | No       | New metadata. Replaces existing metadata entirely. |

**Response:**

| Field   | Type | Description       |
|---------|------|-------------------|
| success | bool | `true` on success |

**Status Codes:** 200 (success), 400 (neither values nor metadata provided, dimension mismatch), 404 (not found), 500 (storage error)

```bash
curl -X PUT http://localhost:8080/api/v1/collections/documents/vectors/1 \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "title": "Updated Title",
      "category": "tech",
      "year": 2025
    }
  }'
```

```json
{
  "success": true
}
```

### Delete Vector

```text
DELETE /api/v1/collections/{collection}/vectors/{id}
```

Deletes a vector by ID.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |
| id         | uint64 | Vector ID       |

**Response:**

| Field   | Type | Description       |
|---------|------|-------------------|
| success | bool | `true` on success |

**Status Codes:** 200 (success), 404 (not found), 500 (storage error)

```bash
curl -X DELETE http://localhost:8080/api/v1/collections/documents/vectors/1
```

```json
{
  "success": true
}
```

### Bulk Insert

```text
POST /api/v1/collections/{collection}/vectors/bulk
```

Inserts multiple vectors in a single request with configurable performance options.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter           | Type    | Required | Default       | Description                                                                 |
|---------------------|---------|----------|---------------|-----------------------------------------------------------------------------|
| vectors             | array   | Yes      |               | Array of vector objects, each with `id`, `values`, and optional `metadata`. |
| batch_lock_size     | uint32  | No       | `1`           | Number of vectors to lock and insert at once. Max: 10000.                   |
| defer_graph         | bool    | No       | `false`       | Skip graph computation during insert. Rebuild later via `optimize()`.       |
| wal_flush_every     | uint32  | No       | `1`           | Flush WAL every N vectors. 0 disables WAL for this batch.                   |
| ef_construction     | uint32  | No       | `0`           | Override HNSW ef_construction for this bulk insert. 0 uses collection default. |
| index_mode          | string  | No       | `"immediate"` | `"immediate"` indexes vectors during insert. `"deferred"` indexes later via `optimize()`. |
| skip_metadata_index | bool    | No       | `false`       | Skip per-vector metadata indexing. Rebuild later via `optimize()`.          |
| parallel_build      | bool    | No       | `false`       | Use parallel HNSW construction. Only effective with `index_mode: "deferred"`. |

Each vector object in the `vectors` array:

| Field    | Type        | Required | Description                        |
|----------|-------------|----------|------------------------------------|
| id       | uint64      | No       | Vector ID. 0 for auto-assign.     |
| values   | float array | Yes      | Vector values matching collection dimension. |
| metadata | object      | No       | Key-value metadata.                |

**Response:**

| Field          | Type         | Description                     |
|----------------|--------------|---------------------------------|
| inserted_count | uint64       | Number of successfully inserted vectors |
| errors         | string array | List of error messages for failed inserts |

**Status Codes:** 200 (success, possibly partial), 400 (invalid options), 404 (collection not found), 500 (storage error)

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/vectors/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {"id": 1, "values": [0.1, 0.2, 0.3], "metadata": {"title": "Doc A"}},
      {"id": 2, "values": [0.4, 0.5, 0.6], "metadata": {"title": "Doc B"}},
      {"id": 3, "values": [0.7, 0.8, 0.9], "metadata": {"title": "Doc C"}}
    ],
    "batch_lock_size": 100,
    "defer_graph": true,
    "wal_flush_every": 500,
    "index_mode": "deferred",
    "parallel_build": true
  }'
```

```json
{
  "inserted_count": 3,
  "errors": []
}
```

### Optimize

```text
POST /api/v1/collections/{collection}/optimize
```

Triggers index rebuilding, graph recomputation, and metadata re-indexing for a collection. Call this after bulk inserts with deferred options.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:** Empty (`{}`)

**Response:**

| Field             | Type   | Description                                                   |
|-------------------|--------|---------------------------------------------------------------|
| status            | string | `"completed"`, `"already_optimized"`, or `"error"`            |
| message           | string | Human-readable description of what was done                   |
| duration_ms       | uint64 | Time taken in milliseconds                                    |
| vectors_processed | uint64 | Number of vectors processed during optimization               |

**Status Codes:** 200 (success), 404 (collection not found), 409 (already optimizing), 500 (optimization error)

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/optimize \
  -H "Content-Type: application/json" \
  -d '{}'
```

```json
{
  "status": "completed",
  "message": "rebuilt HNSW index, recomputed graph edges, re-indexed metadata",
  "duration_ms": 1250,
  "vectors_processed": 50000
}
```

---

## Search

### Search

```text
POST /api/v1/collections/{collection}/search
```

Performs vector similarity search with optional filtering and graph enrichment.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter       | Type        | Required | Default  | Description                                                                       |
|-----------------|-------------|----------|----------|-----------------------------------------------------------------------------------|
| query           | float array | Yes      |          | Query vector. Must match collection dimension.                                    |
| k               | uint32      | Yes      |          | Number of results to return.                                                      |
| filter          | object      | No       | `null`   | Filter expression (see [Filter Expressions](#filter-expressions)).                |
| strategy        | string      | No       | `"auto"` | Filter strategy: `"auto"`, `"pre_filter"`, or `"post_filter"`.                   |
| include_metadata| bool        | No       | `false`  | Include metadata in results.                                                      |
| include_graph   | bool        | No       | `false`  | Include virtual graph edges for each result.                                      |
| graph_threshold | float       | No       | `0.0`    | Minimum similarity for graph edges. 0.0 uses the collection default threshold.    |
| max_graph_edges | uint32      | No       | `10`     | Maximum graph edges to return per result.                                         |
| ef_search       | uint32      | No       | `null`   | Override HNSW ef_search for this query. Higher values improve recall at the cost of latency. |

**Response:**

| Field          | Type   | Description                                          |
|----------------|--------|------------------------------------------------------|
| results        | array  | Array of scored results                              |
| search_time_us | uint64 | Search time in microseconds                          |
| warning        | string | Warning message (e.g., stale results during optimization). Omitted if empty. |

Each result object:

| Field       | Type   | Description                                         |
|-------------|--------|-----------------------------------------------------|
| id          | uint64 | Vector ID                                           |
| score       | float  | Distance score (lower = more similar) |
| metadata    | object | Metadata (only if `include_metadata: true`)         |
| graph_edges | array  | Related edges (only if `include_graph: true`)       |

Each graph edge:

| Field     | Type   | Description                  |
|-----------|--------|------------------------------|
| target_id | uint64 | ID of the related vector     |
| similarity| float  | Similarity score of the edge |

**Status Codes:** 200 (success), 400 (dimension mismatch, invalid filter), 404 (collection not found)

**Basic search:**

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3],
    "k": 10,
    "include_metadata": true
  }'
```

```json
{
  "results": [
    {
      "id": 42,
      "score": 0.0012,
      "metadata": {"title": "Closest Document", "category": "tech"}
    },
    {
      "id": 17,
      "score": 0.0089,
      "metadata": {"title": "Second Closest", "category": "science"}
    }
  ],
  "search_time_us": 245
}
```

**Filtered search:**

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3],
    "k": 5,
    "filter": {
      "and": [
        {"field": "category", "op": "eq", "value": "technology"},
        {"field": "year", "op": "gte", "value": 2023}
      ]
    },
    "strategy": "pre_filter",
    "include_metadata": true
  }'
```

**Graph-enriched search:**

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3],
    "k": 10,
    "include_metadata": true,
    "include_graph": true,
    "graph_threshold": 0.85,
    "max_graph_edges": 5
  }'
```

```json
{
  "results": [
    {
      "id": 42,
      "score": 0.0012,
      "metadata": {"title": "Closest Document"},
      "graph_edges": [
        {"target_id": 43, "similarity": 0.95},
        {"target_id": 99, "similarity": 0.88}
      ]
    }
  ],
  "search_time_us": 312
}
```

### Batch Search

```text
POST /api/v1/search/batch
```

Executes multiple search queries in a single request. Each query can target a different collection.

**Request Body:**

| Parameter | Type  | Required | Description                                    |
|-----------|-------|----------|------------------------------------------------|
| queries   | array | Yes      | Array of search query objects (same fields as the single search request, plus a `collection` field) |

Each query object:

| Parameter       | Type        | Required | Default  | Description                                |
|-----------------|-------------|----------|----------|--------------------------------------------|
| collection      | string      | Yes      |          | Target collection name                     |
| query           | float array | Yes      |          | Query vector                               |
| k               | uint32      | Yes      |          | Number of results                          |
| filter          | object      | No       | `null`   | Filter expression                          |
| strategy        | string      | No       | `"auto"` | Filter strategy                            |
| include_metadata| bool        | No       | `false`  | Include metadata                           |
| include_graph   | bool        | No       | `false`  | Include graph edges                        |
| graph_threshold | float       | No       | `0.0`    | Minimum graph edge similarity              |
| max_graph_edges | uint32      | No       | `10`     | Max graph edges per result                 |
| ef_search       | uint32      | No       | `null`   | Override HNSW ef_search                    |

**Response:**

| Field         | Type   | Description                     |
|---------------|--------|---------------------------------|
| results       | array  | Array of search responses (one per query, same format as single search response) |
| total_time_us | uint64 | Total time for the entire batch |

```bash
curl -X POST http://localhost:8080/api/v1/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "collection": "documents",
        "query": [0.1, 0.2, 0.3],
        "k": 5,
        "include_metadata": true
      },
      {
        "collection": "images",
        "query": [0.4, 0.5, 0.6],
        "k": 3
      }
    ]
  }'
```

```json
{
  "results": [
    {
      "results": [
        {"id": 42, "score": 0.0012, "metadata": {"title": "Doc A"}}
      ],
      "search_time_us": 180
    },
    {
      "results": [
        {"id": 7, "score": 0.0034}
      ],
      "search_time_us": 95
    }
  ],
  "total_time_us": 290
}
```

---

## Filter Expressions

Filters narrow search results based on vector metadata. They are passed in the `filter` field of search requests.

### Field Filters

A field filter compares a metadata field against a value.

```json
{"field": "category", "op": "eq", "value": "technology"}
```

**Structure:**

| Field  | Type        | Description                                                     |
|--------|-------------|-----------------------------------------------------------------|
| field  | string      | Metadata field name                                             |
| op     | string      | Comparison operator (see table below)                           |
| value  | any         | Single comparison value as plain JSON (for most operators)      |
| values | array       | Array of plain JSON values (for `in` and `between` operators)   |

### Operators

| Operator  | Description                              | Value Type  | Example                                                       |
|-----------|------------------------------------------|-------------|---------------------------------------------------------------|
| `eq`      | Equal to                                 | single      | `{"field": "status", "op": "eq", "value": "active"}` |
| `ne`      | Not equal to                             | single      | `{"field": "status", "op": "ne", "value": "deleted"}` |
| `gt`      | Greater than                             | single      | `{"field": "price", "op": "gt", "value": 9.99}` |
| `gte`     | Greater than or equal to                 | single      | `{"field": "year", "op": "gte", "value": 2020}` |
| `lt`      | Less than                                | single      | `{"field": "count", "op": "lt", "value": 100}` |
| `lte`     | Less than or equal to                    | single      | `{"field": "rating", "op": "lte", "value": 5.0}` |
| `in`      | Value is in the provided list            | multi       | `{"field": "category", "op": "in", "values": ["a", "b"]}` |
| `between` | Value is between two bounds (inclusive)  | multi (2)   | `{"field": "price", "op": "between", "values": [10.0, 50.0]}` |
| `exists`  | Field exists (value is ignored)          | none        | `{"field": "thumbnail", "op": "exists", "value": true}` |
| `contains`| String contains substring                | single      | `{"field": "title", "op": "contains", "value": "vector"}` |

### Value Types

In REST API filter expressions, values are plain JSON. The type is inferred automatically:

| JSON Type    | Example       | Description              |
|--------------|---------------|--------------------------|
| string       | `"hello"`     | String comparison        |
| number (int) | `42`          | Integer comparison       |
| number (float)| `3.14`       | Float comparison         |
| boolean      | `true`        | Boolean comparison       |
| array        | `["a", "b"]`  | Used with `in` operator  |

> **Note:** The gRPC API uses typed wrappers (`string_value`, `int_value`, `float_value`, `bool_value`, `string_list_value`) as defined in the protobuf schema. The REST API does not require these wrappers.

### Logical Operators

Combine multiple filters using `and`, `or`, and `not`.

**AND (all must match):**

```json
{
  "and": [
    {"field": "category", "op": "eq", "value": "tech"},
    {"field": "year", "op": "gte", "value": 2023}
  ]
}
```

**OR (at least one must match):**

```json
{
  "or": [
    {"field": "category", "op": "eq", "value": "tech"},
    {"field": "category", "op": "eq", "value": "science"}
  ]
}
```

**NOT (negate a filter):**

```json
{
  "not": {"field": "status", "op": "eq", "value": "archived"}
}
```

**Nested example (AND with nested OR):**

```json
{
  "and": [
    {"field": "year", "op": "gte", "value": 2020},
    {
      "or": [
        {"field": "category", "op": "eq", "value": "tech"},
        {"field": "category", "op": "eq", "value": "science"}
      ]
    }
  ]
}
```

---

## Graph

SwarnDB's virtual graph connects vectors that exceed a similarity threshold. The graph is computed automatically during insertion (if a threshold is set) or rebuilt during `optimize()`.

### Get Related

```text
GET /api/v1/collections/{collection}/graph/related/{id}?threshold=0.8&max_results=10
```

Returns vectors directly connected to the given vector in the virtual graph.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |
| id         | uint64 | Vector ID       |

**Query Parameters:**

| Parameter   | Type   | Required | Default            | Description                          |
|-------------|--------|----------|--------------------|--------------------------------------|
| threshold   | float  | No       | Collection default | Minimum similarity for returned edges |
| max_results | uint32 | No       | All edges          | Maximum number of edges to return    |

**Response:**

| Field | Type  | Description          |
|-------|-------|----------------------|
| edges | array | Array of graph edges |

Each edge:

| Field     | Type   | Description              |
|-----------|--------|--------------------------|
| target_id | uint64 | ID of the related vector |
| similarity| float  | Similarity score         |

**Status Codes:** 200 (success), 404 (collection or vector not found)

```bash
curl "http://localhost:8080/api/v1/collections/documents/graph/related/42?threshold=0.85&max_results=5"
```

```json
{
  "edges": [
    {"target_id": 43, "similarity": 0.95},
    {"target_id": 99, "similarity": 0.91},
    {"target_id": 17, "similarity": 0.87}
  ]
}
```

### Traverse

```text
POST /api/v1/collections/{collection}/graph/traverse
```

Performs a multi-hop traversal through the virtual graph starting from a given vector.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter   | Type   | Required | Default            | Description                                     |
|-------------|--------|----------|--------------------|-------------------------------------------------|
| start_id    | uint64 | Yes      |                    | Starting vector ID                              |
| depth       | uint32 | No       | `2`                | Maximum traversal depth (number of hops)        |
| threshold   | float  | No       | Collection default | Minimum similarity threshold for traversal      |
| max_results | uint32 | No       | All reachable      | Maximum number of nodes to return               |

**Response:**

| Field | Type  | Description               |
|-------|-------|---------------------------|
| nodes | array | Array of traversal nodes  |

Each node:

| Field           | Type       | Description                                  |
|-----------------|------------|----------------------------------------------|
| id              | uint64     | Vector ID                                    |
| depth           | uint32     | Hop distance from start                      |
| path_similarity | float      | Cumulative similarity along the path         |
| path            | uint64 array | Ordered list of vector IDs from start to this node |

**Status Codes:** 200 (success), 404 (collection or start vector not found)

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/graph/traverse \
  -H "Content-Type: application/json" \
  -d '{
    "start_id": 42,
    "depth": 3,
    "threshold": 0.8,
    "max_results": 20
  }'
```

```json
{
  "nodes": [
    {"id": 43, "depth": 1, "path_similarity": 0.95, "path": [42, 43]},
    {"id": 99, "depth": 1, "path_similarity": 0.91, "path": [42, 99]},
    {"id": 101, "depth": 2, "path_similarity": 0.86, "path": [42, 43, 101]}
  ]
}
```

### Set Threshold

```text
POST /api/v1/collections/{collection}/graph/threshold
```

Sets the similarity threshold for graph edge computation. Can be set at the collection level (affects all vectors) or per-vector.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter | Type   | Required | Default | Description                                                     |
|-----------|--------|----------|---------|-----------------------------------------------------------------|
| vector_id | uint64 | No       | `0`     | Target vector ID. 0 sets the collection-level default threshold. |
| threshold | float  | Yes      |         | Similarity threshold (0.0 to 1.0 for cosine).                  |

**Response:**

| Field   | Type | Description       |
|---------|------|-------------------|
| success | bool | `true` on success |

**Status Codes:** 200 (success), 400 (invalid threshold), 404 (collection or vector not found)

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/graph/threshold \
  -H "Content-Type: application/json" \
  -d '{
    "vector_id": 0,
    "threshold": 0.85
  }'
```

```json
{
  "success": true
}
```

---

## Vector Math

Advanced vector operations for analytics, clustering, drift detection, and more.

### Detect Ghost Vectors

```text
POST /api/v1/collections/{collection}/math/ghosts
```

Identifies isolated vectors that are far from all cluster centroids. Ghost vectors may indicate outliers, noise, or data quality issues.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter | Type               | Required | Default       | Description                                                    |
|-----------|--------------------|----------|---------------|----------------------------------------------------------------|
| threshold | float              | No       | `0.0`         | Isolation score threshold. Vectors scoring above this are ghosts. 0.0 returns all with scores. |
| centroids | array of float arrays | No    | `null`        | Custom centroid vectors. If omitted, auto-clusters with `auto_k`. |
| auto_k    | uint32             | No       | `8`           | Number of clusters for auto-centroid computation.              |
| metric    | string             | No       | `"euclidean"` | Distance metric for isolation scoring.                         |

**Response:**

| Field          | Type   | Description                          |
|----------------|--------|--------------------------------------|
| ghosts         | array  | Array of ghost vector results        |
| compute_time_us| uint64 | Computation time in microseconds     |

Each ghost object:

| Field           | Type   | Description                                 |
|-----------------|--------|---------------------------------------------|
| id              | uint64 | Vector ID                                   |
| isolation_score | float  | Distance to nearest centroid (higher = more isolated) |

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/math/ghosts \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 2.5,
    "auto_k": 10,
    "metric": "euclidean"
  }'
```

```json
{
  "ghosts": [
    {"id": 77, "isolation_score": 4.21},
    {"id": 203, "isolation_score": 3.15}
  ],
  "compute_time_us": 8450
}
```

### Cone Search

```text
POST /api/v1/collections/{collection}/math/cone
```

Finds all vectors within a directional cone defined by a direction vector and an aperture angle. Useful for directional similarity queries.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter        | Type        | Required | Description                                      |
|------------------|-------------|----------|--------------------------------------------------|
| direction        | float array | Yes      | Direction vector defining the cone axis.         |
| aperture_radians | float       | Yes      | Half-angle of the cone in radians (0 to pi).     |

**Response:**

| Field          | Type   | Description                      |
|----------------|--------|----------------------------------|
| results        | array  | Array of vectors within the cone |
| compute_time_us| uint64 | Computation time in microseconds |

Each result:

| Field             | Type   | Description                          |
|-------------------|--------|--------------------------------------|
| id                | uint64 | Vector ID                            |
| cosine_similarity | float  | Cosine similarity to the direction   |
| angle_radians     | float  | Angle from the direction in radians  |

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/math/cone \
  -H "Content-Type: application/json" \
  -d '{
    "direction": [1.0, 0.0, 0.0],
    "aperture_radians": 0.5
  }'
```

```json
{
  "results": [
    {"id": 12, "cosine_similarity": 0.98, "angle_radians": 0.12},
    {"id": 45, "cosine_similarity": 0.92, "angle_radians": 0.38}
  ],
  "compute_time_us": 1230
}
```

### Compute Centroid

```text
POST /api/v1/collections/{collection}/math/centroid
```

Computes the centroid (mean vector) of a set of vectors. Optionally supports weighted averaging.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter  | Type         | Required | Default      | Description                                                   |
|------------|--------------|----------|--------------|---------------------------------------------------------------|
| vector_ids | uint64 array | No       | All vectors  | IDs of vectors to include. Empty means all vectors in the collection. |
| weights    | float array  | No       | Equal weight | Per-vector weights. Must match the length of `vector_ids` if provided. |

**Response:**

| Field          | Type        | Description                      |
|----------------|-------------|----------------------------------|
| centroid       | float array | The computed centroid vector      |
| compute_time_us| uint64      | Computation time in microseconds |

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/math/centroid \
  -H "Content-Type: application/json" \
  -d '{
    "vector_ids": [1, 2, 3, 4, 5],
    "weights": [1.0, 1.0, 2.0, 1.0, 1.0]
  }'
```

```json
{
  "centroid": [0.25, 0.38, 0.51],
  "compute_time_us": 42
}
```

### Interpolate (SLERP/LERP)

```text
POST /api/v1/math/interpolate
```

Interpolates between two vectors using linear (LERP) or spherical linear (SLERP) interpolation. Can produce a single result or a sequence of evenly spaced points.

> Note: This endpoint does not require a collection. It operates on raw vectors.

**Request Body:**

| Parameter      | Type        | Required | Default  | Description                                                        |
|----------------|-------------|----------|----------|--------------------------------------------------------------------|
| a              | float array | Yes      |          | Start vector.                                                      |
| b              | float array | Yes      |          | End vector.                                                        |
| t              | float       | No       | `0.0`    | Interpolation parameter (0.0 = a, 1.0 = b). Ignored if `sequence_count > 0`. |
| method         | string      | No       | `"lerp"` | Interpolation method: `"lerp"` or `"slerp"`.                      |
| sequence_count | uint32      | No       | `0`      | If greater than 0, generates a sequence of evenly spaced interpolated points. |

**Response:**

| Field          | Type                   | Description                                                    |
|----------------|------------------------|----------------------------------------------------------------|
| results        | array of float arrays  | One interpolated vector (if `sequence_count` is 0) or a sequence of vectors. |
| compute_time_us| uint64                 | Computation time in microseconds                               |

```bash
curl -X POST http://localhost:8080/api/v1/math/interpolate \
  -H "Content-Type: application/json" \
  -d '{
    "a": [1.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0],
    "t": 0.5,
    "method": "slerp"
  }'
```

```json
{
  "results": [[0.707, 0.707, 0.0]],
  "compute_time_us": 5
}
```

**Sequence example:**

```bash
curl -X POST http://localhost:8080/api/v1/math/interpolate \
  -H "Content-Type: application/json" \
  -d '{
    "a": [1.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0],
    "method": "slerp",
    "sequence_count": 5
  }'
```

```json
{
  "results": [
    [1.0, 0.0, 0.0],
    [0.924, 0.383, 0.0],
    [0.707, 0.707, 0.0],
    [0.383, 0.924, 0.0],
    [0.0, 1.0, 0.0]
  ],
  "compute_time_us": 12
}
```

### Detect Drift

```text
POST /api/v1/collections/{collection}/math/drift
```

Compares two windows (groups) of vectors to detect distributional drift. Useful for monitoring embedding quality over time.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter   | Type         | Required | Default       | Description                                                        |
|-------------|--------------|----------|---------------|--------------------------------------------------------------------|
| window1_ids | uint64 array | Yes      |               | Vector IDs for the first (baseline) window.                       |
| window2_ids | uint64 array | Yes      |               | Vector IDs for the second (comparison) window.                    |
| metric      | string       | No       | `"euclidean"` | Distance metric for drift computation.                            |
| threshold   | float        | No       | `null`        | If set, the `has_drifted` field indicates whether drift exceeds this threshold. |

**Response:**

| Field                 | Type   | Description                                                   |
|-----------------------|--------|---------------------------------------------------------------|
| centroid_shift        | float  | Distance between the centroids of the two windows             |
| mean_distance_window1 | float  | Average intra-window distance for window 1                   |
| mean_distance_window2 | float  | Average intra-window distance for window 2                   |
| spread_change         | float  | Difference in spread (window2 - window1)                     |
| has_drifted           | bool   | `true` if centroid_shift exceeds the provided threshold       |
| compute_time_us       | uint64 | Computation time in microseconds                             |

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/math/drift \
  -H "Content-Type: application/json" \
  -d '{
    "window1_ids": [1, 2, 3, 4, 5],
    "window2_ids": [101, 102, 103, 104, 105],
    "metric": "cosine",
    "threshold": 0.1
  }'
```

```json
{
  "centroid_shift": 0.15,
  "mean_distance_window1": 0.32,
  "mean_distance_window2": 0.41,
  "spread_change": 0.09,
  "has_drifted": true,
  "compute_time_us": 520
}
```

### K-Means Clustering

```text
POST /api/v1/collections/{collection}/math/cluster
```

Performs k-means clustering on vectors in the collection.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter      | Type   | Required | Default       | Description                                       |
|----------------|--------|----------|---------------|---------------------------------------------------|
| k              | uint32 | Yes      |               | Number of clusters.                               |
| max_iterations | uint32 | No       | `100`         | Maximum number of k-means iterations.             |
| tolerance      | float  | No       | `0.0001`      | Convergence tolerance. Stops when centroid movement is below this. |
| metric         | string | No       | `"euclidean"` | Distance metric for clustering.                   |

**Response:**

| Field          | Type                  | Description                              |
|----------------|-----------------------|------------------------------------------|
| centroids      | array of float arrays | Computed cluster centroid vectors         |
| assignments    | array                 | Per-vector cluster assignments           |
| iterations     | uint32                | Number of iterations performed           |
| converged      | bool                  | Whether the algorithm converged          |
| compute_time_us| uint64                | Computation time in microseconds         |

Each assignment:

| Field               | Type   | Description                           |
|---------------------|--------|---------------------------------------|
| id                  | uint64 | Vector ID                             |
| cluster             | uint32 | Assigned cluster index (0-based)      |
| distance_to_centroid| float  | Distance from the vector to its centroid |

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/math/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "k": 5,
    "max_iterations": 200,
    "tolerance": 0.001,
    "metric": "euclidean"
  }'
```

```json
{
  "centroids": [
    [0.12, 0.34, 0.56],
    [0.78, 0.90, 0.11]
  ],
  "assignments": [
    {"id": 1, "cluster": 0, "distance_to_centroid": 0.23},
    {"id": 2, "cluster": 1, "distance_to_centroid": 0.15}
  ],
  "iterations": 47,
  "converged": true,
  "compute_time_us": 15200
}
```

### PCA (Dimensionality Reduction)

```text
POST /api/v1/collections/{collection}/math/pca
```

Performs Principal Component Analysis on vectors in the collection, projecting them into a lower-dimensional space.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter    | Type         | Required | Default     | Description                                                   |
|--------------|--------------|----------|-------------|---------------------------------------------------------------|
| n_components | uint32       | No       | `2`         | Number of principal components to compute.                    |
| vector_ids   | uint64 array | No       | All vectors | IDs of vectors to include. Empty means all vectors.           |

**Response:**

| Field              | Type                  | Description                                          |
|--------------------|-----------------------|------------------------------------------------------|
| components         | array of float arrays | Principal component vectors (eigenvectors)           |
| explained_variance | float array           | Variance explained by each component                 |
| mean               | float array           | Mean vector (centroid) of the input data             |
| projected          | array of float arrays | Input vectors projected into the reduced space       |
| compute_time_us    | uint64                | Computation time in microseconds                     |

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/math/pca \
  -H "Content-Type: application/json" \
  -d '{
    "n_components": 3,
    "vector_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  }'
```

```json
{
  "components": [
    [0.57, 0.42, 0.71],
    [0.82, -0.38, -0.43],
    [-0.01, 0.82, -0.57]
  ],
  "explained_variance": [12.5, 8.3, 3.1],
  "mean": [0.33, 0.44, 0.55],
  "projected": [
    [1.2, -0.5, 0.3],
    [0.8, 0.9, -0.1]
  ],
  "compute_time_us": 2100
}
```

### Compute Analogy

```text
POST /api/v1/math/analogy
```

Computes vector analogies using the classic "A is to B as C is to ?" formula (`result = B - A + C`), or performs general weighted vector arithmetic.

> Note: This endpoint does not require a collection. It operates on raw vectors.

**Request Body:**

| Parameter | Type        | Required | Default | Description                                                            |
|-----------|-------------|----------|---------|------------------------------------------------------------------------|
| a         | float array | No       | `null`  | Vector A (the "from" in the analogy). Required for analogy mode.       |
| b         | float array | No       | `null`  | Vector B (the "to" in the analogy). Required for analogy mode.         |
| c         | float array | No       | `null`  | Vector C (the "as" in the analogy). Required for analogy mode.         |
| normalize | bool        | No       | `false` | Normalize the result vector to unit length.                            |
| terms     | array       | No       | `[]`    | For general arithmetic mode. Array of `{vector, weight}` terms. Overrides a/b/c if non-empty. |

Each term object (for general arithmetic):

| Field  | Type        | Description                    |
|--------|-------------|--------------------------------|
| vector | float array | A vector                       |
| weight | float       | Weight multiplier for this vector |

**Response:**

| Field          | Type        | Description                      |
|----------------|-------------|----------------------------------|
| result         | float array | The computed result vector       |
| compute_time_us| uint64      | Computation time in microseconds |

**Analogy mode (A:B :: C:?):**

```bash
curl -X POST http://localhost:8080/api/v1/math/analogy \
  -H "Content-Type: application/json" \
  -d '{
    "a": [1.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0],
    "c": [0.5, 0.0, 0.5],
    "normalize": true
  }'
```

```json
{
  "result": [-0.408, 0.816, 0.408],
  "compute_time_us": 3
}
```

**General arithmetic mode:**

```bash
curl -X POST http://localhost:8080/api/v1/math/analogy \
  -H "Content-Type: application/json" \
  -d '{
    "terms": [
      {"vector": [1.0, 0.0, 0.0], "weight": 0.5},
      {"vector": [0.0, 1.0, 0.0], "weight": 0.3},
      {"vector": [0.0, 0.0, 1.0], "weight": 0.2}
    ],
    "normalize": false
  }'
```

```json
{
  "result": [0.5, 0.3, 0.2],
  "compute_time_us": 2
}
```

### Diversity Sampling (MMR)

```text
POST /api/v1/collections/{collection}/math/diversity
```

Selects a diverse subset of vectors using Maximal Marginal Relevance (MMR). Balances relevance to the query with diversity among selected results.

**Path Parameters:**

| Parameter  | Type   | Description     |
|------------|--------|-----------------|
| collection | string | Collection name |

**Request Body:**

| Parameter     | Type         | Required | Default     | Description                                                            |
|---------------|--------------|----------|-------------|------------------------------------------------------------------------|
| query         | float array  | Yes      |             | Query vector.                                                          |
| k             | uint32       | Yes      |             | Number of diverse results to select.                                   |
| lambda        | float        | Yes      |             | Trade-off parameter. `1.0` = pure relevance, `0.0` = pure diversity.  |
| candidate_ids | uint64 array | No       | All vectors | IDs of candidate vectors. Empty means all vectors in the collection.   |

**Response:**

| Field          | Type   | Description                      |
|----------------|--------|----------------------------------|
| results        | array  | Array of diverse results         |
| compute_time_us| uint64 | Computation time in microseconds |

Each result:

| Field           | Type   | Description                                     |
|-----------------|--------|-------------------------------------------------|
| id              | uint64 | Vector ID                                       |
| relevance_score | float  | Similarity to the query vector                  |
| mmr_score       | float  | Final MMR score (combining relevance and diversity) |

```bash
curl -X POST http://localhost:8080/api/v1/collections/documents/math/diversity \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3],
    "k": 5,
    "lambda": 0.7,
    "candidate_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  }'
```

```json
{
  "results": [
    {"id": 3, "relevance_score": 0.95, "mmr_score": 0.92},
    {"id": 7, "relevance_score": 0.88, "mmr_score": 0.79},
    {"id": 1, "relevance_score": 0.91, "mmr_score": 0.74},
    {"id": 9, "relevance_score": 0.82, "mmr_score": 0.68},
    {"id": 5, "relevance_score": 0.85, "mmr_score": 0.61}
  ],
  "compute_time_us": 340
}
```

---

## gRPC API

SwarnDB's gRPC API runs on port 50051 and provides the same functionality as the REST API with the performance benefits of Protocol Buffers and HTTP/2.

Proto definitions are located at `proto/swarndb/v1/`.

### Services and RPCs

#### CollectionService

Defined in `proto/swarndb/v1/collection.proto`.

| RPC              | Request                   | Response                   | Description          |
|------------------|---------------------------|----------------------------|----------------------|
| CreateCollection | CreateCollectionRequest    | CreateCollectionResponse   | Create a collection  |
| DeleteCollection | DeleteCollectionRequest    | DeleteCollectionResponse   | Delete a collection  |
| GetCollection    | GetCollectionRequest       | GetCollectionResponse      | Get collection info  |
| ListCollections  | ListCollectionsRequest     | ListCollectionsResponse    | List all collections |

#### VectorService

Defined in `proto/swarndb/v1/vector.proto`.

| RPC                  | Request                      | Response            | Description                          |
|----------------------|------------------------------|---------------------|--------------------------------------|
| Insert               | InsertRequest                | InsertResponse      | Insert a single vector               |
| Get                  | GetVectorRequest             | GetVectorResponse   | Get a vector by ID                   |
| Update               | UpdateRequest                | UpdateResponse      | Update a vector                      |
| Delete               | DeleteVectorRequest          | DeleteVectorResponse| Delete a vector                      |
| BulkInsert           | stream InsertRequest         | BulkInsertResponse  | Stream-based bulk insert             |
| BulkInsertWithOptions| stream BulkInsertStreamMessage| BulkInsertResponse | Stream-based bulk insert with tuning options (first message is options, rest are vectors) |
| Optimize             | OptimizeRequest              | OptimizeResponse    | Trigger index/graph rebuild          |

#### SearchService

Defined in `proto/swarndb/v1/search.proto`.

| RPC         | Request            | Response            | Description          |
|-------------|--------------------|---------------------|----------------------|
| Search      | SearchRequest      | SearchResponse      | Single search query  |
| BatchSearch | BatchSearchRequest | BatchSearchResponse | Multiple queries     |

#### GraphService

Defined in `proto/swarndb/v1/graph.proto`.

| RPC          | Request             | Response             | Description                 |
|--------------|---------------------|----------------------|-----------------------------|
| GetRelated   | GetRelatedRequest   | GetRelatedResponse   | Get connected vectors       |
| Traverse     | TraverseRequest     | TraverseResponse     | Multi-hop graph traversal   |
| SetThreshold | SetThresholdRequest | SetThresholdResponse | Set similarity threshold    |

#### VectorMathService

Defined in `proto/swarndb/v1/vector_math.proto`.

| RPC              | Request                  | Response                  | Description                    |
|------------------|--------------------------|---------------------------|--------------------------------|
| DetectGhosts     | DetectGhostsRequest      | DetectGhostsResponse      | Find isolated vectors          |
| ConeSearch       | ConeSearchRequest        | ConeSearchResponse        | Directional cone search        |
| ComputeCentroid  | ComputeCentroidRequest   | ComputeCentroidResponse   | Compute mean vector            |
| Interpolate      | InterpolateRequest       | InterpolateResponse       | LERP/SLERP interpolation       |
| DetectDrift      | DetectDriftRequest       | DetectDriftResponse       | Compare vector distributions   |
| Cluster          | ClusterRequest           | ClusterResponse           | K-means clustering             |
| ReduceDimensions | ReduceDimensionsRequest  | ReduceDimensionsResponse  | PCA dimensionality reduction   |
| ComputeAnalogy   | ComputeAnalogyRequest    | ComputeAnalogyResponse    | Vector analogy/arithmetic      |
| DiversitySample  | DiversitySampleRequest   | DiversitySampleResponse   | MMR diversity sampling         |

### Python gRPC Example

```python
from swarndb import SwarnDBClient

client = SwarnDBClient("localhost:50051")

# Create a collection
client.collections.create(
    name="documents",
    dimension=1536,
    distance_metric="cosine",
    default_threshold=0.8
)

# Insert vectors
client.vectors.insert(
    collection="documents",
    values=[0.1, 0.2, 0.3, ...],  # 1536-dim vector
    metadata={"title": "My Document", "category": "tech"}
)

# Search
results = client.search.search(
    collection="documents",
    query=[0.1, 0.2, 0.3, ...],
    k=10,
    include_metadata=True,
    include_graph=True,
    graph_threshold=0.85
)

for r in results.results:
    print(f"ID: {r.id}, Score: {r.score}")
    for edge in r.graph_edges:
        print(f"  Related: {edge.target_id} (sim: {edge.similarity})")

# Graph traversal
nodes = client.graph.traverse(
    collection="documents",
    start_id=42,
    depth=3,
    threshold=0.8
)

# Clustering
clusters = client.math.cluster(
    collection="documents",
    k=5,
    max_iterations=100
)

# Async client
from swarndb import AsyncSwarnDBClient

async_client = AsyncSwarnDBClient("localhost:50051")
results = await async_client.search.search(
    collection="documents",
    query=[0.1, 0.2, 0.3, ...],
    k=10
)
```

---

## Metadata Types

SwarnDB supports five metadata value types. When setting metadata via the REST API, values are automatically inferred from JSON types. When using gRPC or filter expressions, values must be explicitly typed.

### Supported Types

| Type        | Rust Type | JSON Representation           | gRPC Wrapper                                     |
|-------------|-----------|-------------------------------|--------------------------------------------------|
| String      | String    | `"hello"`                     | `{"string_value": "hello"}`                      |
| Integer     | i64       | `42`                          | `{"int_value": 42}`                              |
| Float       | f64       | `3.14`                        | `{"float_value": 3.14}`                          |
| Boolean     | bool      | `true`                        | `{"bool_value": true}`                           |
| String List | Vec\<String\> | `["a", "b", "c"]`        | `{"string_list_value": {"values": ["a", "b"]}}` |

### REST API Metadata Examples

When inserting or updating via REST, metadata is a plain JSON object. Types are inferred:

```json
{
  "metadata": {
    "title": "Vector Databases 101",
    "year": 2024,
    "rating": 4.8,
    "published": true,
    "tags": ["database", "vectors", "search"]
  }
}
```

### Filter Metadata Values

In REST API filter expressions, values are plain JSON (no typed wrappers needed):

```json
{
  "field": "year",
  "op": "gte",
  "value": 2023
}
```

```json
{
  "field": "rating",
  "op": "between",
  "values": [4.0, 5.0]
}
```

```json
{
  "field": "published",
  "op": "eq",
  "value": true
}
```

> **Note:** The gRPC API uses typed wrappers for filter values (e.g., `{"int_value": 2023}`, `{"string_value": "tech"}`). See the [gRPC API](#grpc-api) section and the protobuf definitions for the gRPC filter format.
```

### gRPC Metadata

In gRPC, metadata is a `map<string, MetadataValue>` where each value uses the `oneof` typed wrapper:

```protobuf
message MetadataValue {
  oneof value {
    string string_value = 1;
    int64 int_value = 2;
    double float_value = 3;
    bool bool_value = 4;
    StringList string_list_value = 5;
  }
}

message StringList {
  repeated string values = 1;
}
```

---

## Authentication

Authentication is optional and only enforced when the `SWARNDB_API_KEYS` environment variable is set.

**Configuration:**

```bash
# Comma-separated list of valid API keys
export SWARNDB_API_KEYS="key1,key2,key3"
```

**Usage:**

Pass the API key in either header format:

```bash
# X-API-Key header
curl -H "X-API-Key: key1" http://localhost:8080/api/v1/collections

# Authorization Bearer header
curl -H "Authorization: Bearer key1" http://localhost:8080/api/v1/collections
```

When `SWARNDB_API_KEYS` is not set, all requests are allowed without authentication.

When authentication fails, the server returns:

```json
{
  "error": "unauthorized",
  "code": 401
}
```
