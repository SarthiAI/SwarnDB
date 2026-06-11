// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::net::SocketAddr;
use std::path::Path;

// ADR-025: jemalloc as the global allocator so freed pages return to the OS
// after a bulk load. Gated to non-MSVC x86_64 / aarch64; the system allocator
// is kept everywhere else.
#[cfg(all(not(target_env = "msvc"), any(target_arch = "x86_64", target_arch = "aarch64")))]
use tikv_jemallocator::Jemalloc;

#[cfg(all(not(target_env = "msvc"), any(target_arch = "x86_64", target_arch = "aarch64")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// Bake the decay config so a correct setting ships even when no env var is set.
// The symbol is prefixed `_rjem_` because unprefixing is not enabled. Operators
// may still override at runtime via MALLOC_CONF and _RJEM_MALLOC_CONF.
#[cfg(all(not(target_env = "msvc"), any(target_arch = "x86_64", target_arch = "aarch64")))]
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
pub static _rjem_malloc_conf: &[u8] =
    b"background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000\0";

use axum::middleware;
use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;
use tonic::transport::Server as TonicServer;

use vf_server::auth::{api_key_auth, AuthState};
use vf_server::config::ServerConfig;
use vf_server::grpc_collection::CollectionServiceImpl;
use vf_server::grpc_extraction::ExtractionServiceImpl;
use vf_server::grpc_graph::GraphServiceImpl;
use vf_server::grpc_search::SearchServiceImpl;
use vf_server::grpc_vector::VectorServiceImpl;
use vf_server::grpc_vector_math::VectorMathServiceImpl;
use vf_server::health::health_router;
use vf_server::logging::init_logging;
use vf_server::metrics::{metrics_handler, setup_metrics};
use vf_server::proto::swarndb::v1::collection_service_server::CollectionServiceServer;
use vf_server::proto::swarndb::v1::extraction_service_server::ExtractionServiceServer;
use vf_server::proto::swarndb::v1::graph_service_server::GraphServiceServer;
use vf_server::proto::swarndb::v1::search_service_server::SearchServiceServer;
use vf_server::proto::swarndb::v1::vector_math_service_server::VectorMathServiceServer;
use vf_server::proto::swarndb::v1::vector_service_server::VectorServiceServer;
use vf_server::rest::rest_router;
use vf_server::shutdown::{graceful_shutdown, wait_for_shutdown, ShutdownSignal};
use vf_server::snapshot::{start_snapshot_scheduler, SnapshotConfig};
use vf_server::state::AppState;

#[tokio::main]
async fn main() {
    // 1. Load configuration
    let config = ServerConfig::load();

    // 2. Initialize structured logging
    init_logging(&config.log_level);

    tracing::info!(
        grpc_port = config.grpc_port,
        rest_port = config.rest_port,
        data_dir = %config.data_dir,
        "SwarnDB server starting"
    );

    // F3: cap the global rayon pool below the core count so a parallel index
    // build cannot starve the async runtime that serves /healthz and /readyz.
    // Every bulk-build par_iter runs on this pool, so capping it here bounds the
    // build's parallelism. Reserve is configurable; default holds back 1 core.
    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let build_threads = total_cores.saturating_sub(config.index_build_core_reserve).max(1);
    match rayon::ThreadPoolBuilder::new()
        .num_threads(build_threads)
        .thread_name(|i| format!("index-build-{}", i))
        .build_global()
    {
        Ok(()) => tracing::info!(
            total_cores,
            reserved = config.index_build_core_reserve,
            build_threads,
            "global build pool capped to keep probes responsive"
        ),
        Err(e) => tracing::warn!(
            "could not cap the global build pool ({e}); using rayon defaults"
        ),
    }

    // 3. Acquire data directory lock (prevents dual instances)
    let _data_lock = vf_storage::file_lock::ProcessLock::acquire(
        Path::new(&config.data_dir),
    ).unwrap_or_else(|e| {
        tracing::error!("failed to acquire data directory lock: {}", e);
        std::process::exit(1);
    });

    // 4. Initialize Prometheus metrics
    let metrics_handle = setup_metrics();

    // 5. Build the application state SKELETON. This step opens the data
    //    directory, scans for persisted collections, and records the set of
    //    names that will be recovered, but does NOT load any collection. The
    //    actual recovery runs on a background task after the listeners are
    //    bound so that docker start always brings the ports up within seconds
    //    even on a 1M-vector cold boot.
    let state = AppState::new_empty(
        Path::new(&config.data_dir),
        config.max_ef_search,
        config.max_batch_lock_size,
        config.max_wal_flush_interval,
        config.max_ef_construction,
        config.clone(),
    ).unwrap_or_else(|e| {
        tracing::error!("failed to initialize AppState skeleton: {}", e);
        std::process::exit(1);
    });

    // The shared status flag and loading-state view live on `state` itself
    // from now on. Health and probe handlers read it via the AppState clone.
    let server_status = state.server_status.clone();

    // 6. Build gRPC server (spawned BEFORE recovery so the port is reachable
    //    immediately).
    let grpc_addr: SocketAddr = config
        .grpc_addr()
        .parse()
        .expect("invalid gRPC bind address");

    let grpc_state = state.clone();
    let max_ef_search = config.max_ef_search;
    let max_batch_lock_size = config.max_batch_lock_size;
    let max_wal_flush_interval = config.max_wal_flush_interval;
    let max_ef_construction = config.max_ef_construction;
    // ADR-013: raise the per-service gRPC decode/encode caps off tonic's 4 MB
    // default so large extraction submissions are accepted up to this bound.
    let max_msg = config.max_grpc_message_bytes;
    let grpc_handle = tokio::spawn(async move {
        tracing::info!(%grpc_addr, "gRPC server listening");

        let collection_svc =
            CollectionServiceServer::new(CollectionServiceImpl::new(grpc_state.clone()))
                .max_decoding_message_size(max_msg)
                .max_encoding_message_size(max_msg);
        let vector_svc = VectorServiceServer::new(VectorServiceImpl::new(
            grpc_state.clone(),
            max_batch_lock_size,
            max_wal_flush_interval,
            max_ef_construction,
        ))
        .max_decoding_message_size(max_msg)
        .max_encoding_message_size(max_msg);
        let search_svc =
            SearchServiceServer::new(SearchServiceImpl::new(grpc_state.clone(), max_ef_search))
                .max_decoding_message_size(max_msg)
                .max_encoding_message_size(max_msg);
        let graph_svc = GraphServiceServer::new(GraphServiceImpl::new(grpc_state.clone()))
            .max_decoding_message_size(max_msg)
            .max_encoding_message_size(max_msg);
        let vector_math_svc =
            VectorMathServiceServer::new(VectorMathServiceImpl::new(grpc_state.clone()))
                .max_decoding_message_size(max_msg)
                .max_encoding_message_size(max_msg);
        let extraction_svc =
            ExtractionServiceServer::new(ExtractionServiceImpl::new(grpc_state.clone()))
                .max_decoding_message_size(max_msg)
                .max_encoding_message_size(max_msg);

        if let Err(e) = TonicServer::builder()
            .add_service(collection_svc)
            .add_service(vector_svc)
            .add_service(search_svc)
            .add_service(graph_svc)
            .add_service(vector_math_svc)
            .add_service(extraction_svc)
            .serve(grpc_addr)
            .await
        {
            tracing::error!("gRPC server error: {}", e);
        }
    });

    // 7. Build REST server (also spawned BEFORE recovery).
    let rest_addr: SocketAddr = config
        .rest_addr()
        .parse()
        .expect("invalid REST bind address");

    // Build REST router with optional auth (applied only to API routes)
    let api_router = if !config.api_keys.is_empty() {
        tracing::info!(
            key_count = config.api_keys.len(),
            "API key authentication enabled"
        );
        let auth_state = AuthState::new(config.api_keys.clone());
        rest_router(state.clone())
            .layer(middleware::from_fn_with_state(auth_state, api_key_auth))
    } else {
        rest_router(state.clone())
    };

    // Health and metrics routes are NOT behind auth.
    let health_routes = health_router(state.clone(), server_status.clone());
    let metrics_route = Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(metrics_handle);

    let app = api_router.merge(health_routes).merge(metrics_route);

    let rest_handle = tokio::spawn(async move {
        tracing::info!(%rest_addr, "REST server listening");

        let listener = TcpListener::bind(rest_addr)
            .await
            .expect("failed to bind REST listener");

        if let Err(e) = axum::serve(listener, app.into_make_service()).await {
            tracing::error!("REST server error: {}", e);
        }
    });

    // 8. Kick off background recovery. The listeners are already bound at
    //    this point so /health, /readyz, and every guarded handler can serve
    //    503s with a meaningful body until each collection finishes loading.
    //    The recovery runs on a dedicated rayon pool inside `recover_collections`;
    //    we move the call into a tokio blocking task so the async runtime is
    //    free to handle the freshly bound traffic.
    let recovery_state = state.clone();
    let recovery_status = server_status.clone();
    let _recovery_handle = tokio::task::spawn_blocking(move || {
        // Outer catch_unwind guards setup paths (rayon pool builder, pool.install)
        // that sit outside the per-collection AssertUnwindSafe in recover_collections.
        // On panic, /health surfaces failure via is_initialized() staying false.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recovery_state.recover_collections();
            recovery_status.mark_initialized();
            tracing::info!("SwarnDB server fully initialized and ready");
        }));
        if let Err(payload) = result {
            tracing::error!("server recovery task panicked: {:?}", payload);
        }
    });

    // 9. Start background snapshot scheduler
    let shutdown_signal = ShutdownSignal::new();
    let snapshot_shutdown_rx = shutdown_signal.subscribe();
    let snapshot_config = SnapshotConfig::from_env(
        config.snapshot_check_interval_secs,
        config.snapshot_mutation_threshold,
        config.snapshot_interval_secs,
    );
    let snapshot_state = std::sync::Arc::new(state.clone());
    let snapshot_handle = tokio::spawn(async move {
        start_snapshot_scheduler(snapshot_state, snapshot_shutdown_rx, snapshot_config).await;
    });

    // 10. Start background WAL pruner
    let wal_prune_interval = config.wal_prune_interval_secs;
    let wal_prune_handle = if wal_prune_interval > 0 {
        let wal_prune_state = std::sync::Arc::new(state.clone());
        let mut wal_prune_shutdown_rx = shutdown_signal.subscribe();
        Some(tokio::spawn(async move {
            let interval = std::time::Duration::from_secs(wal_prune_interval);
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {},
                    _ = wal_prune_shutdown_rx.changed() => {
                        tracing::info!("WAL pruner shutting down");
                        return;
                    }
                }
                // Prune WAL for all collections. Brief map read lock just to
                // snapshot the name list; the actual prune work runs without
                // holding the map lock.
                let collection_names: Vec<String> = {
                    let collections = wal_prune_state.collections.read();
                    collections.keys().cloned().collect()
                };
                for name in collection_names {
                    match wal_prune_state.prune_wal_for_collection(&name) {
                        Ok((count, bytes)) => {
                            if count > 0 {
                                tracing::info!("WAL pruner: pruned {} files ({} bytes) for '{}'", count, bytes, name);
                            }
                        }
                        Err(e) => tracing::warn!("WAL pruner failed for '{}': {}", name, e),
                    }
                }
            }
        }))
    } else {
        None
    };

    // 10b. Start the extraction worker pool. It consumes the shared job queue
    //      and drains on the same shutdown watch as the other background tasks.
    let extraction_shutdown_rx = shutdown_signal.subscribe();
    let extraction_handle = state.extraction.spawn_workers(extraction_shutdown_rx);

    // 11. Wait for shutdown signal
    wait_for_shutdown().await;

    // 12. Graceful shutdown
    tracing::info!("shutdown signal received, stopping servers...");

    // Signal background tasks to stop
    shutdown_signal.trigger();
    let _ = snapshot_handle.await;
    if let Some(handle) = wal_prune_handle {
        let _ = handle.await;
    }
    // Drain the extraction worker pool so any in-flight job's graph writes and
    // delta syncs complete before the collections are flushed below.
    let _ = extraction_handle.await;

    // Abort server tasks
    // Note: recovery handle is dropped on shutdown; partial loads are discarded.
    grpc_handle.abort();
    rest_handle.abort();

    // Flush collections and release resources
    graceful_shutdown(state).await;

    tracing::info!("SwarnDB server stopped");
}
