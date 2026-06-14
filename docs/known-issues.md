# Known Issues and Limitations

This page lists known limitations and their recommended mitigations, so you can plan around them in production.

## Health probe latency during bulk load from disk (from_path)

- **What:** During a large `bulk_insert_from_path` (the server-side bulk load from a file on disk), the HTTP health endpoints (`/readyz`, `/health`, `/healthz`) can become slow to respond or return a non-200 status for the duration of the load.
- **Scope:** Observed at 1M-vector scale. It affects the file-based bulk-load path specifically. The normal insert path and the deferred-insert-plus-optimize path keep the probes responsive.
- **Why:** The bulk-load-from-disk operation can keep the server busy enough that health checks are not answered promptly during the load. A fix is planned.
- **Impact:** An orchestrator (for example Kubernetes) with tight liveness or readiness timeouts could treat the pod as unhealthy and restart it in the middle of the load.
- **Mitigation:** Run large file-based bulk loads during a maintenance window; or temporarily relax the liveness and readiness `timeoutSeconds` and `failureThreshold` for the duration of the load; or use the deferred-insert-plus-optimize path, which keeps the probes responsive.
