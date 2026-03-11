// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::middleware;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;
use tonic::transport::Server as TonicServer;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::set_header::SetResponseHeaderLayer;

use vf_server::auth::{api_key_auth, AuthState, GrpcAuthInterceptor};
use vf_server::concurrency::{build_concurrency_layer, ConcurrencyConfig, ConcurrencyError};
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
use vf_server::shutdown::{graceful_shutdown, wait_for_shutdown, ShutdownSignal};
use vf_server::state::AppState;
use vf_server::validation::request_size_limit_layer;

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
        config.compute_timeout_secs,
        config.max_k,
        config.max_bulk_insert_messages,
        config.max_bulk_insert_payload_bytes,
        config.max_ef,
    ).unwrap_or_else(|e| {
        tracing::error!("failed to initialize AppState: {}", e);
        std::process::exit(1);
    });

    // Create shutdown signal for coordinated graceful shutdown
    let shutdown_signal = ShutdownSignal::new();
    let grpc_shutdown_rx = shutdown_signal.subscribe();
    let rest_shutdown_rx = shutdown_signal.subscribe();

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
    let grpc_api_keys = config.api_keys.clone();
    let grpc_max_message_size = config.max_message_size;
    let grpc_tls_cert = config.tls_cert_path.clone();
    let grpc_tls_key = config.tls_key_path.clone();
    let grpc_handle = tokio::spawn(async move {
        tracing::info!(%grpc_addr, "gRPC server listening");

        let grpc_auth = AuthState::new(grpc_api_keys);
        let interceptor = GrpcAuthInterceptor::new(grpc_auth);

        let collection_svc = CollectionServiceServer::new(
            CollectionServiceImpl::new(grpc_state.clone()),
        )
        .max_decoding_message_size(grpc_max_message_size)
        .max_encoding_message_size(grpc_max_message_size);
        let vector_svc = VectorServiceServer::new(
            VectorServiceImpl::new(
                grpc_state.clone(),
                max_batch_lock_size,
                max_wal_flush_interval,
                max_ef_construction,
            ),
        )
        .max_decoding_message_size(grpc_max_message_size)
        .max_encoding_message_size(grpc_max_message_size);
        let search_svc = SearchServiceServer::new(
            SearchServiceImpl::new(grpc_state.clone(), max_ef_search),
        )
        .max_decoding_message_size(grpc_max_message_size)
        .max_encoding_message_size(grpc_max_message_size);
        let graph_svc = GraphServiceServer::new(
            GraphServiceImpl::new(grpc_state.clone()),
        )
        .max_decoding_message_size(grpc_max_message_size)
        .max_encoding_message_size(grpc_max_message_size);
        let vector_math_svc = VectorMathServiceServer::new(
            VectorMathServiceImpl::new(grpc_state.clone()),
        )
        .max_decoding_message_size(grpc_max_message_size)
        .max_encoding_message_size(grpc_max_message_size);

        let mut builder = TonicServer::builder()
            .layer(tonic::service::interceptor(interceptor));

        // Configure TLS if cert and key paths are provided
        if let (Some(cert_path), Some(key_path)) = (grpc_tls_cert, grpc_tls_key) {
            let cert = tokio::fs::read(&cert_path).await.unwrap_or_else(|e| {
                tracing::error!("failed to read TLS cert {}: {}", cert_path, e);
                std::process::exit(1);
            });
            let key = tokio::fs::read(&key_path).await.unwrap_or_else(|e| {
                tracing::error!("failed to read TLS key {}: {}", key_path, e);
                std::process::exit(1);
            });
            let identity = tonic::transport::Identity::from_pem(cert, key);
            let tls_config = tonic::transport::ServerTlsConfig::new().identity(identity);
            builder = builder.tls_config(tls_config).unwrap_or_else(|e| {
                tracing::error!("failed to configure gRPC TLS: {}", e);
                std::process::exit(1);
            });
            tracing::info!("gRPC TLS enabled");
        }

        let mut shutdown_rx = grpc_shutdown_rx;
        let shutdown_fut = async move {
            let _ = shutdown_rx.changed().await;
        };

        if let Err(e) = builder
            .add_service(collection_svc)
            .add_service(vector_svc)
            .add_service(search_svc)
            .add_service(graph_svc)
            .add_service(vector_math_svc)
            .serve_with_shutdown(grpc_addr, shutdown_fut)
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

    // Build concurrency/rate-limiting layer from config
    let concurrency_config = ConcurrencyConfig {
        max_concurrent_searches: config.max_concurrent_searches,
        max_concurrent_connections: config.max_connections,
        search_queue_size: config.search_queue_size,
        request_timeout: Duration::from_millis(config.request_timeout_ms),
        search_timeout: Duration::from_millis(config.search_timeout_ms),
        bulk_timeout: Duration::from_millis(config.bulk_timeout_ms),
        min_concurrency: config.min_concurrency,
        max_concurrency: config.max_concurrency,
        target_p99_latency_ms: config.target_p99_latency_ms,
        ema_alpha: config.concurrency_ema_alpha,
        high_latency_threshold: config.concurrency_high_threshold,
        low_latency_threshold: config.concurrency_low_threshold,
        decrease_rate: config.concurrency_decrease_rate,
        increase_rate: config.concurrency_increase_rate,
    };
    let (concurrency_layer, _concurrency_metrics) = build_concurrency_layer(&concurrency_config);

    // Build body size limit layer (256 MB default, configurable via SWARNDB_MAX_REQUEST_BODY_BYTES)
    let body_limit = request_size_limit_layer(config.max_request_body_bytes);

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

    // Health routes are NOT behind auth; metrics IS behind auth
    let health_routes = health_router(state.clone(), server_status_clone);
    let metrics_route = Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(metrics_handle);

    // Put metrics behind the same auth as API routes
    let authed_metrics = if !config.api_keys.is_empty() {
        let auth_state = AuthState::new(config.api_keys.clone());
        metrics_route.layer(middleware::from_fn_with_state(auth_state, api_key_auth))
    } else {
        metrics_route
    };

    // Build CORS layer.
    // Default: allow any origin (open-source friendly). Restrict via SWARNDB_CORS_ORIGINS.
    let cors_layer = if config.cors_origins.is_empty() {
        CorsLayer::new()
            .allow_origin(AllowOrigin::any())
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::PUT,
                axum::http::Method::DELETE,
                axum::http::Method::OPTIONS,
            ])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
                axum::http::header::ACCEPT,
            ])
    } else {
        let origins: Vec<HeaderValue> = config
            .cors_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::PUT,
                axum::http::Method::DELETE,
                axum::http::Method::OPTIONS,
            ])
            .allow_headers([
                axum::http::header::CONTENT_TYPE,
                axum::http::header::AUTHORIZATION,
                axum::http::header::ACCEPT,
            ])
    };

    // Build a combined concurrency + error-handling layer via ServiceBuilder.
    // HandleErrorLayer sits above the ConcurrencyLimitLayer so that
    // ConcurrencyError is mapped into a proper HTTP response, keeping the
    // final error type as Infallible (required by axum Router).
    let concurrency_stack = tower::ServiceBuilder::new()
        .layer(axum::error_handling::HandleErrorLayer::new(
            |err: ConcurrencyError<std::convert::Infallible>| async move {
                match err {
                    ConcurrencyError::Overloaded => {
                        (StatusCode::SERVICE_UNAVAILABLE, "server overloaded").into_response()
                    }
                    ConcurrencyError::TimedOut => {
                        (StatusCode::GATEWAY_TIMEOUT, "request timed out").into_response()
                    }
                    ConcurrencyError::Inner(infallible) => match infallible {},
                }
            },
        ))
        .layer(concurrency_layer);

    // Apply rate limiting, body size limit, CORS, and security headers.
    // Health routes are merged AFTER the concurrency layer so they are
    // exempt from rate limiting (probes must never be rate-limited).
    let app = api_router
        .merge(authed_metrics)
        .layer(body_limit)
        .layer(concurrency_stack)
        .merge(health_routes)
        .layer(cors_layer)
        .layer(SetResponseHeaderLayer::overriding(
            HeaderName::from_static("x-content-type-options"),
            HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            HeaderName::from_static("x-frame-options"),
            HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            HeaderName::from_static("x-xss-protection"),
            HeaderValue::from_static("0"),
        ))
        .layer(SetResponseHeaderLayer::overriding(
            HeaderName::from_static("content-security-policy"),
            HeaderValue::from_static("default-src 'none'"),
        ));

    let rest_tls_cert = config.tls_cert_path.clone();
    let rest_tls_key = config.tls_key_path.clone();
    let rest_handle = tokio::spawn(async move {
        tracing::info!(%rest_addr, "REST server listening");

        let listener = TcpListener::bind(rest_addr)
            .await
            .expect("failed to bind REST listener");

        let mut shutdown_rx = rest_shutdown_rx;

        if let (Some(cert_path), Some(key_path)) = (rest_tls_cert, rest_tls_key) {
            // TLS-enabled REST server using axum-server with rustls
            let rustls_config = {
                let cert_pem = tokio::fs::read(&cert_path).await.unwrap_or_else(|e| {
                    tracing::error!("failed to read TLS cert {}: {}", cert_path, e);
                    std::process::exit(1);
                });
                let key_pem = tokio::fs::read(&key_path).await.unwrap_or_else(|e| {
                    tracing::error!("failed to read TLS key {}: {}", key_path, e);
                    std::process::exit(1);
                });

                let certs = rustls_pemfile::certs(&mut cert_pem.as_slice())
                    .filter_map(|r| r.ok())
                    .collect::<Vec<_>>();
                let key = rustls_pemfile::private_key(&mut key_pem.as_slice())
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| {
                        tracing::error!("no valid private key found in {}", key_path);
                        std::process::exit(1);
                    });

                let mut config = tokio_rustls::rustls::ServerConfig::builder()
                    .with_no_client_auth()
                    .with_single_cert(certs, key)
                    .unwrap_or_else(|e| {
                        tracing::error!("failed to build rustls config: {}", e);
                        std::process::exit(1);
                    });
                config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
                std::sync::Arc::new(config)
            };

            let tls_acceptor = tokio_rustls::TlsAcceptor::from(rustls_config);
            tracing::info!("REST TLS enabled");

            // Accept connections until shutdown signal is received
            loop {
                tokio::select! {
                    result = listener.accept() => {
                        let (tcp_stream, _remote_addr) = match result {
                            Ok(conn) => conn,
                            Err(e) => {
                                tracing::warn!("failed to accept TCP connection: {}", e);
                                continue;
                            }
                        };

                        let tls_acceptor = tls_acceptor.clone();
                        let app = app.clone();

                        tokio::spawn(async move {
                            let tls_stream = match tls_acceptor.accept(tcp_stream).await {
                                Ok(s) => s,
                                Err(e) => {
                                    tracing::debug!("TLS handshake failed: {}", e);
                                    return;
                                }
                            };
                            let io = hyper_util::rt::TokioIo::new(tls_stream);
                            let service = hyper_util::service::TowerToHyperService::new(app);
                            if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                                hyper_util::rt::TokioExecutor::new(),
                            )
                            .serve_connection(io, service)
                            .await
                            {
                                tracing::debug!("connection error: {}", e);
                            }
                        });
                    }
                    _ = shutdown_rx.changed() => {
                        tracing::info!("REST TLS server stopping accept loop");
                        break;
                    }
                }
            }
        } else {
            let shutdown_fut = async move {
                let _ = shutdown_rx.changed().await;
            };
            if let Err(e) = axum::serve(listener, app.into_make_service())
                .with_graceful_shutdown(shutdown_fut)
                .await
            {
                tracing::error!("REST server error: {}", e);
            }
        }
    });

    // Mark server as fully initialized
    server_status.mark_initialized();
    tracing::info!("SwarnDB server fully initialized and ready");

    // 7. Wait for shutdown signal
    wait_for_shutdown().await;

    // 8. Graceful shutdown
    tracing::info!("shutdown signal received, stopping servers...");

    // Signal servers to stop accepting new connections
    shutdown_signal.trigger();

    // Obtain abort handles before moving JoinHandles into the timeout future
    let grpc_abort = grpc_handle.abort_handle();
    let rest_abort = rest_handle.abort_handle();

    // Wait for in-flight requests to complete with a timeout
    let drain_timeout = Duration::from_secs(5);
    tracing::info!("waiting up to {:?} for in-flight requests to complete...", drain_timeout);

    if tokio::time::timeout(drain_timeout, async {
        let _ = grpc_handle.await;
        let _ = rest_handle.await;
    })
    .await
    .is_err()
    {
        tracing::warn!("drain timeout expired, aborting remaining server tasks");
        grpc_abort.abort();
        rest_abort.abort();
    }

    // Flush collections and release resources
    graceful_shutdown(state).await;

    tracing::info!("SwarnDB server stopped");
}
