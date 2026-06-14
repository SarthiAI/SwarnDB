# Deployment Guide

This guide covers all the ways to deploy SwarnDB in production, from a single Docker container to a full Kubernetes cluster with Helm.

## Docker

Docker is the primary and simplest way to run SwarnDB.

### Pulling from Docker Hub

```bash
docker pull sarthiai/swarndb:latest
```

To pin a specific version:

```bash
docker pull sarthiai/swarndb:v1.0.0
```

### Running with environment variables

SwarnDB is configured entirely through environment variables:

```bash
docker run -d --name swarndb \
  -p 8080:8080 \
  -p 50051:50051 \
  -v swarndb_data:/data \
  -e SWARNDB_LOG_LEVEL=info \
  -e SWARNDB_API_KEYS=your-secret-key-here \
  -e SWARNDB_MAX_CONNECTIONS=1000 \
  -e SWARNDB_REQUEST_TIMEOUT_MS=30000 \  # default is 10000ms; 30000ms is a production override
  sarthiai/swarndb:latest
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SWARNDB_HOST` | `0.0.0.0` | Bind address |
| `SWARNDB_REST_PORT` | `8080` | REST API port |
| `SWARNDB_GRPC_PORT` | `50051` | gRPC API port |
| `SWARNDB_DATA_DIR` | `./data` | Data directory path |
| `SWARNDB_LOG_LEVEL` | `info` | Log level: trace, debug, info, warn, error |
| `SWARNDB_API_KEYS` | (empty) | Comma-separated API keys. Empty means no auth |
| `SWARNDB_MAX_CONNECTIONS` | `1000` | Maximum concurrent client connections |
| `SWARNDB_REQUEST_TIMEOUT_MS` | `10000` | Request timeout in milliseconds |

### Persistent storage with volumes

Always use a Docker volume or bind mount for the `/data` directory. Without persistent storage, all collections and indexes are lost when the container stops.

**Named volume** (recommended):

```bash
docker run -d --name swarndb \
  -v swarndb_data:/data \
  sarthiai/swarndb:latest
```

**Bind mount** (for direct filesystem access):

```bash
docker run -d --name swarndb \
  -v /path/on/host/swarndb-data:/data \
  sarthiai/swarndb:latest
```

### Resource limits

Set memory and CPU limits to prevent SwarnDB from consuming all available resources:

```bash
docker run -d --name swarndb \
  --memory=4g \
  --cpus=2 \
  -p 8080:8080 \
  -p 50051:50051 \
  -v swarndb_data:/data \
  sarthiai/swarndb:latest
```

**Sizing guidelines:**

- **Small workloads** (up to 100K vectors): 2 GB memory, 2 CPUs
- **Medium workloads** (100K to 1M vectors): 4 to 8 GB memory, 4 CPUs
- **Large workloads** (1M+ vectors): 16+ GB memory, 8+ CPUs

### Health checks

SwarnDB exposes health endpoints that Docker can use to monitor the container:

```bash
docker run -d --name swarndb \
  --health-cmd="curl -f http://localhost:8080/health || exit 1" \
  --health-interval=30s \
  --health-timeout=5s \
  --health-retries=3 \
  --health-start-period=10s \
  -p 8080:8080 \
  -p 50051:50051 \
  -v swarndb_data:/data \
  sarthiai/swarndb:latest
```

Available health endpoints:

- `GET /health`: Liveness probe. Returns 200 as soon as the server process is up, even while collections are still recovering. The body carries `{"status":"ok"}` when fully initialized and `{"status":"recovering","collections_loaded":N,"collections_total":M,"in_progress":[...]}` while recovery is in flight.
- `GET /readyz`: Readiness probe. Returns 503 while any collection is still recovering; returns 200 only after every persisted collection is loaded and the server is ready to accept production traffic.

Use `/health` for Docker container healthchecks and Kubernetes liveness probes. Use `/readyz` for Kubernetes readiness probes and external load balancer health gates: it is the signal that traffic can be routed without retrying through a recovering collection.

### Security: non-root user, read-only filesystem

For production, run SwarnDB with a non-root user and a read-only filesystem:

```bash
docker run -d --name swarndb \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp:noexec,nosuid,size=256m \
  --security-opt no-new-privileges \
  -p 8080:8080 \
  -p 50051:50051 \
  -v swarndb_data:/data \
  -e SWARNDB_DATA_DIR=/data \
  sarthiai/swarndb:latest
```

This configuration:

- Runs the process as a non-root user (UID 1000)
- Makes the container filesystem read-only
- Provides a writable `/tmp` directory for temporary files
- Prevents privilege escalation

## Docker Compose (Production Setup)

For production deployments, Docker Compose provides a declarative way to configure SwarnDB with all recommended settings.

Create a `docker-compose.yml`:

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
      SWARNDB_HOST: "0.0.0.0"
      SWARNDB_REST_PORT: "8080"
      SWARNDB_GRPC_PORT: "50051"
      SWARNDB_DATA_DIR: "/data"
      SWARNDB_LOG_LEVEL: "info"
      SWARNDB_API_KEYS: "${SWARNDB_API_KEYS}"
      SWARNDB_MAX_CONNECTIONS: "1000"
      SWARNDB_REQUEST_TIMEOUT_MS: "30000"  # default is 10000ms; 30000ms is a production override
    deploy:
      resources:
        limits:
          memory: 4g
          cpus: "2"
        reservations:
          memory: 2g
          cpus: "1"
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=256m
    security_opt:
      - no-new-privileges:true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
    restart: unless-stopped

volumes:
  swarndb_data:
    driver: local
```

Create a `.env` file alongside it to keep secrets out of the compose file:

```bash
SWARNDB_API_KEYS=your-production-api-key-here
```

Start the stack:

```bash
docker compose up -d
```

Check status:

```bash
docker compose ps
docker compose logs -f swarndb
```

## Kubernetes

Kubernetes is recommended for production deployments that need scaling, self-healing, and rolling updates.

### Prerequisites

- `kubectl` installed and configured with cluster access
- A Kubernetes cluster (v1.24 or later recommended)

### Quick deploy

The repository includes ready-to-use Kubernetes manifests:

```bash
kubectl apply -k k8s/
```

This deploys the following resources:

| Manifest | Purpose |
|----------|---------|
| `namespace.yaml` | Creates the `swarndb` namespace |
| `serviceaccount.yaml` | Service account for SwarnDB pods |
| `role.yaml` and `rolebinding.yaml` | RBAC permissions scoped to the namespace |
| `configmap.yaml` | All SwarnDB configuration as environment variables |
| `secret.yaml` | Sensitive values like API keys (base64 encoded) |
| `pvc.yaml` | Persistent volume claim for data storage |
| `deployment.yaml` | SwarnDB deployment with health probes and resource limits |
| `service.yaml` | ClusterIP service exposing ports 8080 and 50051 |
| `hpa.yaml` | Horizontal Pod Autoscaler based on CPU utilization |
| `networkpolicy.yaml` | Network policy restricting traffic to required ports |
| `ingress.yaml` | Ingress resource for external HTTP access |
| `kustomization.yaml` | Kustomize overlay for easy customization |

### Configuration via ConfigMap and Secrets

Configuration values are stored in the ConfigMap (`k8s/configmap.yaml`):

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: swarndb-config
  namespace: swarndb
data:
  SWARNDB_HOST: "0.0.0.0"
  SWARNDB_REST_PORT: "8080"
  SWARNDB_GRPC_PORT: "50051"
  SWARNDB_DATA_DIR: "/data"
  SWARNDB_LOG_LEVEL: "info"
  SWARNDB_MAX_CONNECTIONS: "1000"
  SWARNDB_REQUEST_TIMEOUT_MS: "30000"  # default is 10000ms; 30000ms is a production override
```

Sensitive values go in the Secret (`k8s/secret.yaml`):

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: swarndb-secrets
  namespace: swarndb
type: Opaque
data:
  SWARNDB_API_KEYS: <base64-encoded-api-keys>
```

To encode your API key:

```bash
echo -n "your-api-key-here" | base64
```

### Scaling with HPA

The Horizontal Pod Autoscaler automatically scales SwarnDB pods based on CPU usage:

```bash
# View current HPA status
kubectl get hpa -n swarndb

# Manually scale if needed
kubectl scale deployment swarndb -n swarndb --replicas=3
```

The default HPA configuration targets 70% CPU utilization and scales between 1 and 10 replicas.

### Monitoring with health probes

The deployment includes three types of probes:

- **Startup probe**: Waits for SwarnDB to initialize (checks `/health`, allows up to 60 seconds)
- **Liveness probe**: Restarts the pod if SwarnDB becomes unresponsive (checks `/health` every 15 seconds)
- **Readiness probe**: Removes the pod from the service if it is not ready to handle requests (checks `/readyz` every 10 seconds; returns 503 while collections are recovering after an unclean shutdown)

**Known limitation:** During a large file-based `bulk_insert_from_path` load, the health probes can show elevated latency or transient non-200 responses, which can cause an orchestrator with tight timeouts to restart the pod mid-load. See [Known Issues and Limitations](known-issues.md) for the mitigations.

## Helm

Helm provides a templated, configurable deployment with sensible defaults.

### Install

```bash
helm install swarndb helm/swarndb
```

To install in a specific namespace:

```bash
helm install swarndb helm/swarndb --namespace swarndb --create-namespace
```

### Configuration via values.yaml

Override default values by creating a custom `values.yaml`:

```yaml
replicaCount: 3

image:
  repository: sarthiai/swarndb
  tag: latest
  pullPolicy: IfNotPresent

resources:
  limits:
    memory: 8Gi
    cpu: "4"
  requests:
    memory: 4Gi
    cpu: "2"

persistence:
  enabled: true
  size: 50Gi
  storageClass: standard

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: swarndb.example.com
      paths:
        - path: /
          pathType: Prefix

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

env:
  SWARNDB_LOG_LEVEL: info
  SWARNDB_MAX_CONNECTIONS: "2000"
  SWARNDB_REQUEST_TIMEOUT_MS: "30000"  # default is 10000ms; 30000ms is a production override

secrets:
  SWARNDB_API_KEYS: "your-production-key"
```

Install with your custom values:

```bash
helm install swarndb helm/swarndb -f values.yaml
```

### Key configuration values

| Value | Default | Description |
|-------|---------|-------------|
| `replicaCount` | `1` | Number of SwarnDB replicas |
| `image.repository` | `sarthiai/swarndb` | Container image |
| `image.tag` | `latest` | Image tag |
| `resources.limits.memory` | `4Gi` | Memory limit |
| `resources.limits.cpu` | `2` | CPU limit |
| `persistence.enabled` | `true` | Enable persistent storage |
| `persistence.size` | `10Gi` | Storage size |
| `ingress.enabled` | `false` | Enable ingress |
| `autoscaling.enabled` | `false` | Enable HPA |
| `autoscaling.minReplicas` | `1` | Minimum replicas |
| `autoscaling.maxReplicas` | `10` | Maximum replicas |

### Upgrading

To upgrade SwarnDB to a new version or apply configuration changes:

```bash
helm upgrade swarndb helm/swarndb -f values.yaml
```

To roll back if something goes wrong:

```bash
helm rollback swarndb
```

## TLS and mTLS

SwarnDB does not terminate TLS itself. Both listeners serve plaintext: the REST API is plain HTTP and the gRPC API is plaintext HTTP/2, on the ports set by `SWARNDB_REST_PORT` and `SWARNDB_GRPC_PORT`. There are no server-side certificate, private-key, or client-CA settings, and none of the `SWARNDB_*` environment variables configure TLS. Do not look for a server TLS knob: there is not one.

The supported pattern is to terminate TLS in front of SwarnDB at a load balancer, ingress controller, or reverse proxy (for example an L7 load balancer, an NGINX or Envoy ingress, or a cloud load balancer), and keep SwarnDB's plaintext ports reachable only from that proxy and your internal network.

### Recommended setup

1. Put a TLS-terminating proxy in front of SwarnDB. The proxy holds the server certificate and private key and forwards decrypted traffic to SwarnDB over the internal network. For REST, this is ordinary HTTPS termination. For gRPC, the proxy must terminate TLS and forward HTTP/2 (gRPC) to port 50051.
2. Bind SwarnDB so its ports are not publicly reachable. Keep `SWARNDB_HOST` on the internal interface, and use network policies, security groups, or firewall rules so only the proxy can reach ports 8080 and 50051.
3. Point clients at the proxy's TLS endpoint. The Python SDK encrypts client-to-proxy traffic when you pass `secure=True`:

   ```python
   from swarndb import SwarnDBClient

   client = SwarnDBClient(
       host="swarndb.example.com",   # the TLS endpoint of your proxy
       port=443,                       # the proxy's TLS gRPC port
       api_key="my-secret-key-1",
       secure=True,
   )
   ```

   With `secure=True`, the SDK opens a TLS gRPC channel that verifies the server certificate against the system trust store, and its REST calls use `https`. With the default `secure=False`, both gRPC and REST are plaintext, which is appropriate only when the SDK and SwarnDB share a trusted private network.

### Mutual TLS (client certificates)

Mutual TLS is not provided by SwarnDB and is not provided by the Python SDK. The server has no client-CA setting to validate client certificates, and the SDK's `secure=True` flag verifies the server's certificate but does not present a client certificate. If you require mTLS, enforce it at the terminating proxy (the proxy validates client certificates, then forwards plaintext to SwarnDB). For client-to-server identity within SwarnDB itself, use API key authentication (`SWARNDB_API_KEYS`); see [Configuration](configuration.md#11-authentication).

## Configuration Reference

For a complete list of all environment variables and their defaults, see [Configuration](configuration.md).

## Backup and Restore

SwarnDB stores everything for a collection as plain files under the data directory. A backup is therefore a copy of that directory taken at a point when the on-disk state is consistent, and a restore is putting those files back and letting the server replay its write-ahead log on the next start. There is no separate backup server or backup API to call: the unit of backup is the data volume itself.

### What is on disk

Every collection lives in its own subdirectory under `SWARNDB_DATA_DIR` (the container's `/data` volume in the Docker examples above), named after the collection. Inside each collection directory you will find:

- `config.json`: the collection's settings (dimension, metric, index parameters).
- `segment_NNNNNNNN.vfs`: the on-disk vector segments.
- Snapshot files (`hnsw.base`, `hnsw.delta`, and, for hybrid collections, `graph.base` and `graph.delta`): the persisted index state as of the last snapshot.
- WAL files (`.log`) plus `wal_meta.json`: the write-ahead log of mutations since the last snapshot.
- `shutdown_clean`: a marker written on a clean shutdown so the next start can take the fast recovery path.

A correct backup captures the whole collection directory (all of the above), not just the segments. Backing up the entire `SWARNDB_DATA_DIR` captures every collection at once.

### Taking a consistent backup

The cleanest backup is taken from a stopped server. The recommended sequence:

1. Force a snapshot of each collection so the latest writes are folded into the snapshot files and the WAL is short. For every collection, call the force-snapshot endpoint and confirm the LSN advanced:

   ```bash
   curl -X POST http://localhost:8080/api/v1/collections/<collection>/snapshot
   # -> { "last_snapshot_lsn": 1024 }
   ```

   You can confirm a write is durable with the persistence-status endpoint, which reports `last_snapshot_lsn`, `current_lsn`, and `next_lsn`:

   ```bash
   curl http://localhost:8080/api/v1/collections/<collection>/persistence_status
   ```

2. Stop the server so no further writes land while you copy. With Docker:

   ```bash
   docker stop swarndb
   ```

3. Copy the data directory or snapshot the volume. With a bind mount this is a plain directory copy:

   ```bash
   cp -a /path/on/host/swarndb-data /path/on/host/swarndb-data.backup-$(date +%Y%m%d)
   ```

   With a named Docker volume, archive it through a throwaway container:

   ```bash
   docker run --rm \
     -v swarndb_data:/data:ro \
     -v "$(pwd)":/backup \
     alpine tar czf /backup/swarndb-backup-$(date +%Y%m%d).tar.gz -C /data .
   ```

4. Start the server again:

   ```bash
   docker start swarndb
   ```

On Kubernetes, the equivalent is to take a volume snapshot of the PersistentVolumeClaim, ideally after forcing a snapshot per collection. A storage-layer VolumeSnapshot that captures the PVC atomically is preferable to copying files out of a running pod.

### A note on online (hot) backups

Copying the data directory while the server is still running and accepting writes is possible, but it is not guaranteed to be point-in-time consistent: a snapshot can be rewritten or the WAL can roll over partway through your copy, and you may capture a mix of old and new files. The force-snapshot step before copying narrows but does not close this window. For a guaranteed-consistent backup, stop the server (or use a storage-layer snapshot that captures the whole volume atomically) before copying. If you must back up hot, force a snapshot first, copy the whole collection directory including the WAL, and treat the result as crash-consistent rather than clean: recovery will still replay the WAL on restore, but verify the restored copy before relying on it.

### Restoring from a backup

Restore replaces the data directory with your backup and lets the server recover on start:

1. Stop the server:

   ```bash
   docker stop swarndb
   ```

2. Replace the data directory with the backup contents. Make sure you restore the full collection directories, including the WAL files and the snapshot files, not just the segments. For a bind mount:

   ```bash
   rm -rf /path/on/host/swarndb-data/*
   cp -a /path/on/host/swarndb-data.backup-YYYYMMDD/. /path/on/host/swarndb-data/
   ```

   For a named volume, extract the archive back into it:

   ```bash
   docker run --rm \
     -v swarndb_data:/data \
     -v "$(pwd)":/backup \
     alpine sh -c "rm -rf /data/* && tar xzf /backup/swarndb-backup-YYYYMMDD.tar.gz -C /data"
   ```

3. Start the server. On startup SwarnDB inspects each collection directory and recovers it: if the clean-shutdown marker and a snapshot base are present it loads directly from the snapshot; otherwise it loads the snapshot and replays the WAL on top, or rebuilds from the WAL if no snapshot base exists. A restored copy that was taken from a stopped server recovers cleanly; a crash-consistent copy still recovers by replaying the WAL.

   ```bash
   docker start swarndb
   ```

4. Confirm recovery completed before sending traffic. The readiness endpoint returns 503 while any collection is still recovering and 200 once every collection is loaded:

   ```bash
   curl -f http://localhost:8080/readyz
   ```

   You can also inspect the boot recovery path per collection via `GET /recovery_status`.

After a successful restore you may optionally reclaim space by pruning superseded WAL segments and compacting segments per collection:

```bash
curl -X POST http://localhost:8080/api/v1/collections/<collection>/prune-wal
curl -X POST http://localhost:8080/api/v1/collections/<collection>/compact
```

### Restore testing

Treat a backup as unverified until you have restored it. Periodically restore into a throwaway environment, start the server, wait for `/readyz` to return 200, and run a search against each collection to confirm the data is intact.

## Production Checklist

Before going to production, review each item in this checklist:

**Authentication and Security**

- [ ] Enable API key authentication by setting `SWARNDB_API_KEYS`
- [ ] Run the container as a non-root user
- [ ] Enable read-only filesystem with writable tmpfs for `/tmp`
- [ ] Set `no-new-privileges` security option
- [ ] Apply network policies to restrict traffic to required ports only

**Resource Management**

- [ ] Set memory and CPU limits appropriate for your workload
- [ ] Configure persistent storage with a reliable volume driver
- [ ] Set request timeouts to prevent runaway queries

**Reliability**

- [ ] Enable health checks (Docker) or health probes (Kubernetes)
- [ ] Configure restart policy (`unless-stopped` for Docker, or rely on Kubernetes)
- [ ] Set up Horizontal Pod Autoscaler if running on Kubernetes
- [ ] Test recovery by restarting the container and verifying data persistence

**Observability**

- [ ] Set the log level to `info` for production (avoid `debug` or `trace`)
- [ ] Configure log rotation to prevent disk exhaustion (50 MB max, 5 files)
- [ ] Monitor the `/health` and `/readyz` endpoints from your monitoring system

**Backups**

- [ ] Schedule regular backups of the data volume (see [Backup and Restore](#backup-and-restore))
- [ ] Force a snapshot per collection before each backup, and copy the full data directory (segments, snapshot files, and WAL)
- [ ] Test backup restoration to verify data integrity, then confirm `/readyz` returns 200 and searches return results

**Networking and TLS**

- [ ] Terminate TLS at the load balancer, ingress, or a reverse proxy. SwarnDB does not serve TLS itself; both the REST and gRPC listeners are plaintext (see [TLS and mTLS](#tls-and-mtls))
- [ ] Point the Python SDK at the TLS endpoint with `secure=True` so client-to-proxy traffic is encrypted
- [ ] Keep the plaintext gRPC port (50051) and REST port (8080) reachable only from the proxy or internal network, never exposed directly to the public internet
- [ ] If you need mutual TLS (client certificates), enforce it at the proxy; the server and the SDK do not perform mTLS
- [ ] Restrict the gRPC port (50051) to internal services only if the REST API is the public interface
- [ ] Configure appropriate connection limits for your expected traffic
