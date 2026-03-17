# Configuration

This guide covers every configuration option available in SwarnDB, how to set them, and recommended values for common deployment scenarios.

---

## 1. Overview

SwarnDB supports three layers of configuration, applied in this order of precedence:

```text
  Environment variables   (highest priority, always wins)
         |
         v
  JSON config file        (swarndb.json or SWARNDB_CONFIG path)
         |
         v
  Built-in defaults       (lowest priority, used when nothing else is set)
```

For most deployments, environment variables are all you need. No config file is required.

---

## 2. Server Configuration

These variables control the basic server setup: where it listens, where it stores data, and how verbose the logs are.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SWARNDB_HOST` | `0.0.0.0` | string | Bind address for both gRPC and REST servers. Set to `127.0.0.1` to restrict to localhost only. |
| `SWARNDB_GRPC_PORT` | `50051` | u16 | Port for the gRPC API. Used by the Python SDK and other gRPC clients. |
| `SWARNDB_REST_PORT` | `8080` | u16 | Port for the REST/HTTP API. Used by curl, browsers, and HTTP clients. |
| `SWARNDB_DATA_DIR` | `./data` | string | Directory for persistent storage (WAL files, segments, indexes). Must be writable. Use an absolute path in production. |
| `SWARNDB_LOG_LEVEL` | `info` | string | Logging verbosity. One of: `trace`, `debug`, `info`, `warn`, `error`. |

---

## 3. Security Configuration

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SWARNDB_API_KEYS` | *(empty)* | string | Comma-separated list of API keys. When empty, authentication is disabled entirely. |

See the [Authentication](#10-authentication) section below for details on how API key auth works.

---

## 4. Connection Configuration

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SWARNDB_MAX_CONNECTIONS` | `1000` | usize | Maximum number of concurrent client connections across both gRPC and REST. |

---

## 5. Timeout Configuration

Timeouts protect the server from long-running operations that could starve other requests.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SWARNDB_REQUEST_TIMEOUT_MS` | `10000` | u64 | General request timeout in milliseconds. Applies to collection management, vector CRUD, and other non-search operations. |
| `SWARNDB_SEARCH_TIMEOUT_MS` | `5000` | u64 | Timeout for search operations in milliseconds. Search is typically fast, so a lower timeout prevents runaway queries. |
| `SWARNDB_BULK_TIMEOUT_MS` | `30000` | u64 | Timeout for bulk insert operations in milliseconds. Bulk inserts may take longer due to WAL writes and index updates. |

---

## 6. Index Configuration

These variables set upper bounds on index parameters. They act as safety guardrails, preventing clients from setting values that could cause excessive memory use or slow builds.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SWARNDB_MAX_EF_SEARCH` | `10000` | usize | Maximum allowed `ef_search` value in any query. Prevents clients from requesting extremely wide searches. |
| `SWARNDB_MAX_EF_CONSTRUCTION` | `2000` | u32 | Maximum allowed `ef_construction` override for bulk insert operations. |
| `SWARNDB_MAX_BATCH_LOCK_SIZE` | `10000` | u32 | Maximum batch lock size for bulk insert operations. Controls how many vectors are locked at once during batch writes. |
| `SWARNDB_MAX_WAL_FLUSH_INTERVAL` | `100000` | u32 | Maximum WAL flush interval override. Controls the upper bound on how many operations can be buffered before a WAL flush. |

### HNSW default parameters (not configurable via env vars)

These are the built-in defaults for the HNSW index. They are set at index creation time and can be overridden per collection via the API.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Connections per node (32 at layer 0) |
| `ef_construction` | 200 | Search width during index building |
| `ef_search` | 50 | Search width during queries |
| `max_level_cap` | 16 | Maximum number of HNSW layers |

---

## 7. Adaptive Concurrency Configuration

SwarnDB includes an adaptive concurrency controller that automatically adjusts the number of concurrent operations based on observed latency.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `SWARNDB_MIN_CONCURRENCY` | `10` | usize | Minimum concurrent operations allowed. The controller will never go below this. |
| `SWARNDB_MAX_CONCURRENCY` | `200` | usize | Maximum concurrent operations allowed. The controller will never exceed this. |
| `SWARNDB_TARGET_P99_LATENCY_MS` | `500` | u64 | Target p99 latency in milliseconds. The controller increases concurrency when latency is below this target and decreases it when latency exceeds the target. |

---

## 8. Setting Configuration

### Via Docker run

```bash
docker run -d \
  -e SWARNDB_HOST=0.0.0.0 \
  -e SWARNDB_GRPC_PORT=50051 \
  -e SWARNDB_REST_PORT=8080 \
  -e SWARNDB_DATA_DIR=/data \
  -e SWARNDB_LOG_LEVEL=info \
  -e SWARNDB_API_KEYS=key1,key2 \
  -v swarndb_data:/data \
  -p 50051:50051 \
  -p 8080:8080 \
  ghcr.io/sarthiai/swarndb:latest
```

### Via Docker Compose

```yaml
services:
  swarndb:
    image: ghcr.io/sarthiai/swarndb:latest
    environment:
      SWARNDB_HOST: "0.0.0.0"
      SWARNDB_GRPC_PORT: "50051"
      SWARNDB_REST_PORT: "8080"
      SWARNDB_DATA_DIR: "/data"
      SWARNDB_LOG_LEVEL: "info"
      SWARNDB_API_KEYS: "my-secret-key-1,my-secret-key-2"
    volumes:
      - swarndb_data:/data
    ports:
      - "50051:50051"
      - "8080:8080"

volumes:
  swarndb_data:
```

### Via Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: swarndb-config
data:
  SWARNDB_HOST: "0.0.0.0"
  SWARNDB_GRPC_PORT: "50051"
  SWARNDB_REST_PORT: "8080"
  SWARNDB_DATA_DIR: "/data"
  SWARNDB_LOG_LEVEL: "info"
  SWARNDB_MAX_CONNECTIONS: "1000"
  SWARNDB_REQUEST_TIMEOUT_MS: "10000"
  SWARNDB_SEARCH_TIMEOUT_MS: "5000"
  SWARNDB_BULK_TIMEOUT_MS: "30000"
```

API keys should be stored in a Kubernetes Secret, not a ConfigMap.

### Via .env file (local development)

Create a `.env` file in the working directory:

```bash
SWARNDB_HOST=0.0.0.0
SWARNDB_GRPC_PORT=50051
SWARNDB_REST_PORT=8080
SWARNDB_DATA_DIR=./data
SWARNDB_LOG_LEVEL=debug
SWARNDB_API_KEYS=
```

### Via JSON config file

SwarnDB also supports a JSON configuration file. Set `SWARNDB_CONFIG` to the file path, or place a `swarndb.json` in the working directory:

```json
{
  "host": "0.0.0.0",
  "grpc_port": 50051,
  "rest_port": 8080,
  "data_dir": "/data",
  "log_level": "info",
  "api_keys": ["key1", "key2"],
  "max_connections": 1000
}
```

Environment variables always override values from the JSON file.

---

## 9. Example Configurations

### a. Development

Verbose logging, no authentication, default ports. Good for local testing.

```bash
SWARNDB_LOG_LEVEL=debug
SWARNDB_DATA_DIR=./dev-data
SWARNDB_API_KEYS=
```

### b. Production

Standard logging, authentication enabled, explicit data directory, tuned timeouts.

```bash
SWARNDB_HOST=0.0.0.0
SWARNDB_GRPC_PORT=50051
SWARNDB_REST_PORT=8080
SWARNDB_DATA_DIR=/var/lib/swarndb/data
SWARNDB_LOG_LEVEL=info
SWARNDB_API_KEYS=prod-key-abc123,prod-key-def456
SWARNDB_MAX_CONNECTIONS=1000
SWARNDB_REQUEST_TIMEOUT_MS=10000
SWARNDB_SEARCH_TIMEOUT_MS=5000
SWARNDB_BULK_TIMEOUT_MS=30000
```

### c. High-throughput

Increased connection limits, relaxed timeouts, higher concurrency ceiling. For workloads with many concurrent clients.

```bash
SWARNDB_MAX_CONNECTIONS=5000
SWARNDB_MAX_CONCURRENCY=500
SWARNDB_MIN_CONCURRENCY=50
SWARNDB_TARGET_P99_LATENCY_MS=200
SWARNDB_REQUEST_TIMEOUT_MS=15000
SWARNDB_SEARCH_TIMEOUT_MS=8000
SWARNDB_BULK_TIMEOUT_MS=60000
SWARNDB_MAX_EF_SEARCH=10000
SWARNDB_MAX_BATCH_LOCK_SIZE=50000
```

---

## 10. Authentication

SwarnDB supports API key authentication. When enabled, every request must include a valid API key.

### How it works

1. Set one or more API keys via `SWARNDB_API_KEYS` (comma-separated).
2. Clients include the key in every request.
3. SwarnDB validates the key using a timing-safe comparison to prevent side-channel attacks.
4. If the key is invalid or missing, the request is rejected with a 401 (REST) or UNAUTHENTICATED (gRPC) error.

### Setting API keys

```bash
SWARNDB_API_KEYS=my-secret-key-1,my-secret-key-2,another-key
```

Each key is trimmed of whitespace. Empty strings are ignored.

### Passing the API key in requests

**REST API** (either header works):

```bash
# Using X-API-Key header
curl -H "X-API-Key: my-secret-key-1" http://localhost:8080/api/v1/collections

# Using Authorization Bearer header
curl -H "Authorization: Bearer my-secret-key-1" http://localhost:8080/api/v1/collections
```

**gRPC** (via metadata):

```python
from swarndb import SwarnDBClient

client = SwarnDBClient(
    host="localhost",
    port=50051,
    api_key="my-secret-key-1"
)
```

### Disabling authentication

Leave `SWARNDB_API_KEYS` empty or unset:

```bash
SWARNDB_API_KEYS=
```

When no API keys are configured, all requests are accepted without authentication. This is the default behavior.

---

## 11. Logging

SwarnDB uses structured logging with the `tracing` framework (Rust) and outputs JSON-formatted log entries.

### Log levels

| Level | What it includes |
|-------|-----------------|
| `error` | Unrecoverable failures, data corruption, startup failures |
| `warn` | Recoverable issues, deprecated usage, configuration problems |
| `info` | Server lifecycle (start, stop), collection operations, optimization events |
| `debug` | Individual request details, index operations, WAL events |
| `trace` | Per-vector operations, distance computations, internal state changes |

### Choosing a log level

- **Production**: Use `info`. It provides enough context to monitor health without excessive volume.
- **Debugging issues**: Use `debug` to see request-level details.
- **Performance profiling**: Use `trace` only briefly, as it generates very high log volume.

### Setting the log level

```bash
SWARNDB_LOG_LEVEL=info
```

### Log output in Docker

Logs are written to stdout by default, which Docker captures automatically. Use Docker's logging drivers to control rotation:

```yaml
services:
  swarndb:
    image: ghcr.io/sarthiai/swarndb:latest
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
```

This keeps up to 5 log files of 50 MB each, rotating automatically.
