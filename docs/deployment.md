# Deployment Guide

This guide covers all the ways to deploy SwarnDB in production, from a single Docker container to a full Kubernetes cluster with Helm.

## Docker

Docker is the primary and simplest way to run SwarnDB.

### Pulling from GHCR

```bash
docker pull ghcr.io/sarthiai/swarndb:latest
```

To pin a specific version:

```bash
docker pull ghcr.io/sarthiai/swarndb:v1.0.0
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
  ghcr.io/sarthiai/swarndb:latest
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
  ghcr.io/sarthiai/swarndb:latest
```

**Bind mount** (for direct filesystem access):

```bash
docker run -d --name swarndb \
  -v /path/on/host/swarndb-data:/data \
  ghcr.io/sarthiai/swarndb:latest
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
  ghcr.io/sarthiai/swarndb:latest
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
  ghcr.io/sarthiai/swarndb:latest
```

Available health endpoints:

- `GET /health`: Basic health check. Returns 200 if the server is running.
- `GET /ready`: Readiness check. Returns 200 when the server is ready to accept requests.

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
  ghcr.io/sarthiai/swarndb:latest
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
    image: ghcr.io/sarthiai/swarndb:latest
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
- **Readiness probe**: Removes the pod from the service if it is not ready to handle requests (checks `/ready` every 10 seconds)

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
  repository: ghcr.io/sarthiai/swarndb
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
| `image.repository` | `ghcr.io/sarthiai/swarndb` | Container image |
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

## Configuration Reference

For a complete list of all environment variables and their defaults, see [Configuration](configuration.md).

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
- [ ] Monitor the `/health` and `/ready` endpoints from your monitoring system

**Backups**

- [ ] Schedule regular backups of the data volume
- [ ] Test backup restoration to verify data integrity
- [ ] Document the backup and restore procedure for your team

**Networking**

- [ ] Use TLS termination at the load balancer or ingress level
- [ ] Restrict gRPC port (50051) to internal services only if the REST API is the public interface
- [ ] Configure appropriate connection limits for your expected traffic
