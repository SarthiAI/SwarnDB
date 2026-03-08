// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use std::net::SocketAddr;
use std::path::Path;

use axum::middleware;
use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;
use tonic::transport::Server as TonicServer;

use vf_server::auth::{api_key_auth, AuthState};
use vf_server::config::ServerConfig;
use vf_server::grpc_collection::CollectionServiceImpl;
use vf_server::grpc_graph::GraphServiceImpl;
use vf_server::grpc_search::SearchServiceImpl;
use vf_server::grpc_vector::VectorServiceImpl;
use vf_server::grpc_vector_math::VectorMathServiceImpl;
use vf_server::health::{health_router, ServerStatus};
use vf_server::logging::init_logging;
use vf_server::metrics::{metrics_handler, setup_metrics};
use vf_server::proto::swarndb::v1::collection_service_server::CollectionServiceServer;
use vf_server::proto::swarndb::v1::graph_service_server::GraphServiceServer;
use vf_server::proto::swarndb::v1::search_service_server::SearchServiceServer;
use vf_server::proto::swarndb::v1::vector_math_service_server::VectorMathServiceServer;
use vf_server::proto::swarndb::v1::vector_service_server::VectorServiceServer;
use vf_server::rest::rest_router;
use vf_server::shutdown::{graceful_shutdown, wait_for_shutdown};
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

    // 3. Initialize Prometheus metrics
    let metrics_handle = setup_metrics();

    // 4. Create application state
    let state = AppState::new(
        Path::new(&config.data_dir),
        config.max_ef_search,
        config.max_batch_lock_size,
        config.max_wal_flush_interval,
        config.max_ef_construction,
    ).unwrap_or_else(|e| {
        tracing::error!("failed to initialize AppState: {}", e);
        std::process::exit(1);
    });

    // 5. Build gRPC server
    let grpc_addr: SocketAddr = config
        .grpc_addr()
        .parse()
        .expect("invalid gRPC bind address");

    let grpc_state = state.clone();
    let max_ef_search = config.max_ef_search;
    let max_batch_lock_size = config.max_batch_lock_size;
    let max_wal_flush_interval = config.max_wal_flush_interval;
    let max_ef_construction = config.max_ef_construction;
    let grpc_handle = tokio::spawn(async move {
        tracing::info!(%grpc_addr, "gRPC server listening");

        let collection_svc =
            CollectionServiceServer::new(CollectionServiceImpl::new(grpc_state.clone()));
        let vector_svc = VectorServiceServer::new(VectorServiceImpl::new(
            grpc_state.clone(),
            max_batch_lock_size,
            max_wal_flush_interval,
            max_ef_construction,
        ));
        let search_svc = SearchServiceServer::new(SearchServiceImpl::new(grpc_state.clone(), max_ef_search));
        let graph_svc = GraphServiceServer::new(GraphServiceImpl::new(grpc_state.clone()));
        let vector_math_svc =
            VectorMathServiceServer::new(VectorMathServiceImpl::new(grpc_state.clone()));

        if let Err(e) = TonicServer::builder()
            .add_service(collection_svc)
            .add_service(vector_svc)
            .add_service(search_svc)
            .add_service(graph_svc)
            .add_service(vector_math_svc)
            .serve(grpc_addr)
            .await
        {
            tracing::error!("gRPC server error: {}", e);
        }
    });

    // 6. Build REST server
    let rest_addr: SocketAddr = config
        .rest_addr()
        .parse()
        .expect("invalid REST bind address");

    let server_status = ServerStatus::new();
    let server_status_clone = server_status.clone();

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

    // Health and metrics routes are NOT behind auth
    let health_routes = health_router(state.clone(), server_status_clone);
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

    // Mark server as fully initialized
    server_status.mark_initialized();
    tracing::info!("SwarnDB server fully initialized and ready");

    // 7. Wait for shutdown signal
    wait_for_shutdown().await;

    // 8. Graceful shutdown
    tracing::info!("shutdown signal received, stopping servers...");

    // Abort server tasks
    grpc_handle.abort();
    rest_handle.abort();

    // Flush collections and release resources
    graceful_shutdown(state).await;

    tracing::info!("SwarnDB server stopped");
}
