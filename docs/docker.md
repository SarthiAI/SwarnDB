# Docker Guide

## Quick Start

```bash
docker run -d -p 8080:8080 -p 50051:50051 sarthiai/swarndb
```

Verify:

```bash
curl http://localhost:8080/health
```

## Ports

| Port | Protocol | Purpose |
|:-----|:---------|:--------|
| 8080 | REST | HTTP API |
| 50051 | gRPC | gRPC API |

## Data Persistence

Without a volume, **all data is lost when the container is removed**.

Mount a named volume to persist data across container restarts:

```bash
docker run -d -p 8080:8080 -p 50051:50051 -v swarndb_data:/data sarthiai/swarndb
```

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|:--|:--|:--|
| `SWARNDB_HOST` | `0.0.0.0` | Bind address |
| `SWARNDB_GRPC_PORT` | `50051` | gRPC listener port |
| `SWARNDB_REST_PORT` | `8080` | REST listener port |
| `SWARNDB_DATA_DIR` | `./data` | Data storage directory |
| `SWARNDB_LOG_LEVEL` | `info` | Log verbosity (`trace`, `debug`, `info`, `warn`, `error`) |
| `SWARNDB_API_KEYS` | *(empty)* | Comma-separated API keys; empty disables auth |
| `SWARNDB_MAX_CONNECTIONS` | `1000` | Maximum concurrent connections |
| `SWARNDB_REQUEST_TIMEOUT_MS` | `10000` | Request timeout in milliseconds |

Example with custom configuration:

```bash
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  -v swarndb_data:/data \
  -e SWARNDB_LOG_LEVEL=debug \
  -e SWARNDB_API_KEYS=my-secret-key \
  -e SWARNDB_MAX_CONNECTIONS=2000 \
  sarthiai/swarndb
```

## Docker Compose

```yaml
services:
  swarndb:
    image: sarthiai/swarndb:latest
    container_name: swarndb
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - swarndb_data:/data
    environment:
      - SWARNDB_DATA_DIR=/data
      - SWARNDB_LOG_LEVEL=info
      - SWARNDB_MAX_CONNECTIONS=1000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "1"
          memory: 2G
    restart: unless-stopped

volumes:
  swarndb_data:
```

```bash
docker compose up -d
```

## Building from Source

```bash
git clone https://github.com/SarthiAI/SwarnDB.git
cd SwarnDB
docker build -t swarndb .
docker run -d -p 8080:8080 -p 50051:50051 -v swarndb_data:/data swarndb
```

## Health Check

```bash
curl http://localhost:8080/health
```
