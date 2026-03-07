// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Connection pooling and request queuing middleware.
//!
//! Provides concurrency limiting, request timeout, and load shedding layers
//! built on Tower's `Layer`/`Service` traits and `tokio::sync::Semaphore`.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tower::{Layer, Service};

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for concurrency limiting and request queuing.
#[derive(Clone, Debug)]
pub struct ConcurrencyConfig {
    /// Maximum number of concurrent search requests processed simultaneously.
    pub max_concurrent_searches: usize,
    /// Maximum number of concurrent connections the server will accept.
    pub max_concurrent_connections: usize,
    /// Maximum number of requests that can be queued when at capacity.
    pub search_queue_size: usize,
    /// Per-request timeout duration.
    pub request_timeout: Duration,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent_searches: 100,
            max_concurrent_connections: 1000,
            search_queue_size: 500,
            request_timeout: Duration::from_secs(30),
        }
    }
}

// ── Metrics ─────────────────────────────────────────────────────────────

/// Tracks active connections and queued requests for observability.
#[derive(Clone, Debug)]
pub struct ConcurrencyMetrics {
    inner: Arc<ConcurrencyMetricsInner>,
}

#[derive(Debug)]
struct ConcurrencyMetricsInner {
    active_connections: AtomicUsize,
    queued_requests: AtomicUsize,
    rejected_requests: AtomicUsize,
    timed_out_requests: AtomicUsize,
}

impl ConcurrencyMetrics {
    /// Creates a new metrics tracker.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ConcurrencyMetricsInner {
                active_connections: AtomicUsize::new(0),
                queued_requests: AtomicUsize::new(0),
                rejected_requests: AtomicUsize::new(0),
                timed_out_requests: AtomicUsize::new(0),
            }),
        }
    }

    /// Returns the current number of active connections being processed.
    pub fn active_connections(&self) -> usize {
        self.inner.active_connections.load(Ordering::Relaxed)
    }

    /// Returns the current number of requests waiting in queue.
    pub fn queued_requests(&self) -> usize {
        self.inner.queued_requests.load(Ordering::Relaxed)
    }

    /// Returns the total number of requests rejected due to overload.
    pub fn rejected_requests(&self) -> usize {
        self.inner.rejected_requests.load(Ordering::Relaxed)
    }

    /// Returns the total number of requests that timed out.
    pub fn timed_out_requests(&self) -> usize {
        self.inner.timed_out_requests.load(Ordering::Relaxed)
    }

    fn inc_active(&self) {
        self.inner.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    fn dec_active(&self) {
        self.inner.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    fn inc_queued(&self) {
        self.inner.queued_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn dec_queued(&self) {
        self.inner.queued_requests.fetch_sub(1, Ordering::Relaxed);
    }

    fn inc_rejected(&self) {
        self.inner.rejected_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_timed_out(&self) {
        self.inner.timed_out_requests.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for ConcurrencyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Concurrency error ───────────────────────────────────────────────────

/// Errors returned by the concurrency middleware.
#[derive(Debug, Clone)]
pub enum ConcurrencyError<E> {
    /// The server is overloaded and cannot accept more requests.
    Overloaded,
    /// The request timed out waiting for processing.
    TimedOut,
    /// An error from the inner service.
    Inner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for ConcurrencyError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overloaded => write!(f, "server overloaded, request rejected"),
            Self::TimedOut => write!(f, "request timed out"),
            Self::Inner(e) => write!(f, "{}", e),
        }
    }
}

impl<E: std::fmt::Display + std::fmt::Debug> std::error::Error for ConcurrencyError<E> {}

// ── ConcurrencyLimit Layer ──────────────────────────────────────────────

/// Tower layer that limits concurrency with a semaphore-based queue.
///
/// When the semaphore is full, incoming requests wait in queue up to
/// `queue_size`. If the queue is also full, requests are immediately
/// rejected (load-shed).
#[derive(Clone, Debug)]
pub struct ConcurrencyLimitLayer {
    semaphore: Arc<Semaphore>,
    queue_semaphore: Arc<Semaphore>,
    metrics: ConcurrencyMetrics,
    timeout: Duration,
}

impl ConcurrencyLimitLayer {
    /// Create a new concurrency limit layer.
    ///
    /// - `max_concurrent`: max requests processed simultaneously
    /// - `queue_size`: max requests waiting in queue
    /// - `timeout`: per-request timeout
    /// - `metrics`: shared metrics tracker
    pub fn new(
        max_concurrent: usize,
        queue_size: usize,
        timeout: Duration,
        metrics: ConcurrencyMetrics,
    ) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            queue_semaphore: Arc::new(Semaphore::new(max_concurrent + queue_size)),
            metrics,
            timeout,
        }
    }
}

impl<S> Layer<S> for ConcurrencyLimitLayer {
    type Service = ConcurrencyLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ConcurrencyLimitService {
            inner,
            semaphore: self.semaphore.clone(),
            queue_semaphore: self.queue_semaphore.clone(),
            metrics: self.metrics.clone(),
            timeout: self.timeout,
        }
    }
}

// ── ConcurrencyLimit Service ────────────────────────────────────────────

/// Tower service that enforces concurrency limits, queuing, and timeouts.
#[derive(Clone, Debug)]
pub struct ConcurrencyLimitService<S> {
    inner: S,
    semaphore: Arc<Semaphore>,
    queue_semaphore: Arc<Semaphore>,
    metrics: ConcurrencyMetrics,
    timeout: Duration,
}

impl<S, Req> Service<Req> for ConcurrencyLimitService<S>
where
    S: Service<Req> + Clone + Send + 'static,
    S::Future: Send,
    S::Response: Send,
    S::Error: Send + std::fmt::Display + std::fmt::Debug,
    Req: Send + 'static,
{
    type Response = S::Response;
    type Error = ConcurrencyError<S::Error>;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(ConcurrencyError::Inner)
    }

    fn call(&mut self, req: Req) -> Self::Future {
        let semaphore = self.semaphore.clone();
        let queue_semaphore = self.queue_semaphore.clone();
        let metrics = self.metrics.clone();
        let timeout = self.timeout;
        let mut inner = self.inner.clone();

        Box::pin(async move {
            // Try to acquire the queue slot (load-shed if queue is full).
            let _queue_permit: OwnedSemaphorePermit = match queue_semaphore.try_acquire_owned() {
                Ok(permit) => permit,
                Err(TryAcquireError::NoPermits) => {
                    metrics.inc_rejected();
                    tracing::warn!("request rejected: server overloaded");
                    return Err(ConcurrencyError::Overloaded);
                }
                Err(TryAcquireError::Closed) => {
                    metrics.inc_rejected();
                    return Err(ConcurrencyError::Overloaded);
                }
            };

            // Wait for a processing slot (with timeout).
            metrics.inc_queued();
            let permit = tokio::time::timeout(timeout, semaphore.acquire_owned()).await;
            metrics.dec_queued();

            let _permit: OwnedSemaphorePermit = match permit {
                Ok(Ok(p)) => p,
                Ok(Err(_closed)) => {
                    metrics.inc_rejected();
                    return Err(ConcurrencyError::Overloaded);
                }
                Err(_elapsed) => {
                    metrics.inc_timed_out();
                    tracing::warn!("request timed out waiting for processing slot");
                    return Err(ConcurrencyError::TimedOut);
                }
            };

            // Process the request (with timeout).
            metrics.inc_active();
            let result = tokio::time::timeout(timeout, inner.call(req)).await;
            metrics.dec_active();

            match result {
                Ok(Ok(resp)) => Ok(resp),
                Ok(Err(e)) => Err(ConcurrencyError::Inner(e)),
                Err(_elapsed) => {
                    metrics.inc_timed_out();
                    tracing::warn!("request timed out during processing");
                    Err(ConcurrencyError::TimedOut)
                }
            }
        })
    }
}

// ── Builder ─────────────────────────────────────────────────────────────

/// Build a concurrency layer stack from configuration.
///
/// Returns a `ConcurrencyLimitLayer` that combines:
/// - Concurrency limiting (semaphore-based, bounded to `max_concurrent_connections`)
/// - Request queuing (overflow up to `search_queue_size`)
/// - Load shedding (immediate rejection when queue is full)
/// - Per-request timeout (`request_timeout`)
///
/// Also returns the associated `ConcurrencyMetrics` for observability.
pub fn build_concurrency_layer(
    config: &ConcurrencyConfig,
) -> (ConcurrencyLimitLayer, ConcurrencyMetrics) {
    let metrics = ConcurrencyMetrics::new();

    let layer = ConcurrencyLimitLayer::new(
        config.max_concurrent_connections,
        config.search_queue_size,
        config.request_timeout,
        metrics.clone(),
    );

    tracing::info!(
        max_concurrent = config.max_concurrent_connections,
        queue_size = config.search_queue_size,
        timeout_ms = config.request_timeout.as_millis() as u64,
        "concurrency layer configured"
    );

    (layer, metrics)
}

/// Build a search-specific concurrency layer.
///
/// Uses `max_concurrent_searches` for tighter control over search endpoints.
pub fn build_search_concurrency_layer(
    config: &ConcurrencyConfig,
) -> (ConcurrencyLimitLayer, ConcurrencyMetrics) {
    let metrics = ConcurrencyMetrics::new();

    let layer = ConcurrencyLimitLayer::new(
        config.max_concurrent_searches,
        config.search_queue_size,
        config.request_timeout,
        metrics.clone(),
    );

    tracing::info!(
        max_concurrent_searches = config.max_concurrent_searches,
        queue_size = config.search_queue_size,
        timeout_ms = config.request_timeout.as_millis() as u64,
        "search concurrency layer configured"
    );

    (layer, metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = ConcurrencyConfig::default();
        assert_eq!(config.max_concurrent_searches, 100);
        assert_eq!(config.max_concurrent_connections, 1000);
        assert_eq!(config.search_queue_size, 500);
        assert_eq!(config.request_timeout, Duration::from_secs(30));
    }

    #[test]
    fn metrics_initial_values() {
        let metrics = ConcurrencyMetrics::new();
        assert_eq!(metrics.active_connections(), 0);
        assert_eq!(metrics.queued_requests(), 0);
        assert_eq!(metrics.rejected_requests(), 0);
        assert_eq!(metrics.timed_out_requests(), 0);
    }

    #[test]
    fn metrics_increment_decrement() {
        let metrics = ConcurrencyMetrics::new();
        metrics.inc_active();
        metrics.inc_active();
        assert_eq!(metrics.active_connections(), 2);
        metrics.dec_active();
        assert_eq!(metrics.active_connections(), 1);

        metrics.inc_queued();
        assert_eq!(metrics.queued_requests(), 1);
        metrics.dec_queued();
        assert_eq!(metrics.queued_requests(), 0);

        metrics.inc_rejected();
        assert_eq!(metrics.rejected_requests(), 1);

        metrics.inc_timed_out();
        assert_eq!(metrics.timed_out_requests(), 1);
    }

    #[test]
    fn build_concurrency_layer_returns_layer_and_metrics() {
        let config = ConcurrencyConfig::default();
        let (_layer, metrics) = build_concurrency_layer(&config);
        assert_eq!(metrics.active_connections(), 0);
    }

    #[test]
    fn build_search_concurrency_layer_returns_layer_and_metrics() {
        let config = ConcurrencyConfig::default();
        let (_layer, metrics) = build_search_concurrency_layer(&config);
        assert_eq!(metrics.active_connections(), 0);
    }

    #[test]
    fn concurrency_error_display() {
        let err: ConcurrencyError<String> = ConcurrencyError::Overloaded;
        assert_eq!(format!("{}", err), "server overloaded, request rejected");

        let err: ConcurrencyError<String> = ConcurrencyError::TimedOut;
        assert_eq!(format!("{}", err), "request timed out");

        let err: ConcurrencyError<String> = ConcurrencyError::Inner("oops".to_string());
        assert_eq!(format!("{}", err), "oops");
    }
}
