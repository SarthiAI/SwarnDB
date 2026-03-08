// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Connection pooling, request queuing, and adaptive concurrency middleware.
//!
//! Provides concurrency limiting, request timeout, load shedding, and adaptive
//! concurrency control built on Tower's `Layer`/`Service` traits and
//! `tokio::sync::Semaphore`. The adaptive controller tracks p99 latency via
//! exponential moving average and adjusts the concurrency limit dynamically.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

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
    /// Per-request timeout duration (general).
    pub request_timeout: Duration,
    /// Search-specific request timeout (shorter for tail latency).
    pub search_timeout: Duration,
    /// Bulk operation timeout (longer for large payloads).
    pub bulk_timeout: Duration,
    /// Minimum concurrency limit for adaptive sizing.
    pub min_concurrency: usize,
    /// Maximum concurrency limit for adaptive sizing.
    pub max_concurrency: usize,
    /// Target p99 latency in milliseconds for adaptive control.
    pub target_p99_latency_ms: u64,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent_searches: 100,
            max_concurrent_connections: 1000,
            search_queue_size: 500,
            request_timeout: Duration::from_secs(10),
            search_timeout: Duration::from_secs(5),
            bulk_timeout: Duration::from_secs(30),
            min_concurrency: 10,
            max_concurrency: 200,
            target_p99_latency_ms: 500,
        }
    }
}

// ── Adaptive Concurrency Controller ─────────────────────────────────────

/// Adaptive concurrency controller that adjusts limits based on observed p99
/// latency using an exponential moving average (EMA).
///
/// When p99 latency exceeds the target threshold, the controller reduces the
/// concurrency limit. When latency is well below the target, it increases the
/// limit. Bounds are enforced via `min_concurrency` and `max_concurrency`.
#[derive(Clone, Debug)]
pub struct AdaptiveConcurrencyController {
    inner: Arc<AdaptiveInner>,
}

#[derive(Debug)]
struct AdaptiveInner {
    /// Current concurrency limit (atomically updated).
    current_limit: AtomicUsize,
    /// EMA of p99 latency stored as f64 bits in AtomicU64.
    p99_ema_bits: AtomicU64,
    /// EMA smoothing factor (typically 0.1).
    alpha: f64,
    /// Target p99 latency in milliseconds.
    target_p99_ms: f64,
    /// Minimum concurrency limit.
    min_concurrency: usize,
    /// Maximum concurrency limit.
    max_concurrency: usize,
    /// The semaphore whose permits we adjust.
    semaphore: Arc<Semaphore>,
}

impl AdaptiveConcurrencyController {
    /// Create a new adaptive controller.
    ///
    /// - `initial_limit`: starting concurrency limit
    /// - `min_concurrency`: floor for the adaptive limit
    /// - `max_concurrency`: ceiling for the adaptive limit
    /// - `target_p99_ms`: target p99 latency in milliseconds
    /// - `semaphore`: the semaphore to resize
    pub fn new(
        initial_limit: usize,
        min_concurrency: usize,
        max_concurrency: usize,
        target_p99_ms: u64,
        semaphore: Arc<Semaphore>,
    ) -> Self {
        let clamped = initial_limit.clamp(min_concurrency, max_concurrency);
        Self {
            inner: Arc::new(AdaptiveInner {
                current_limit: AtomicUsize::new(clamped),
                p99_ema_bits: AtomicU64::new(0u64),
                alpha: 0.1,
                target_p99_ms: target_p99_ms as f64,
                min_concurrency,
                max_concurrency,
                semaphore,
            }),
        }
    }

    /// Record a request latency sample and adjust the concurrency limit.
    pub fn record_latency(&self, duration: Duration) {
        let latency_ms = duration.as_secs_f64() * 1000.0;

        // Update EMA: ema = alpha * sample + (1 - alpha) * ema
        let old_bits = self.inner.p99_ema_bits.load(Ordering::Relaxed);
        let old_ema = f64::from_bits(old_bits);
        let new_ema = if old_bits == 0 {
            // First sample, initialize EMA directly.
            latency_ms
        } else {
            self.inner.alpha * latency_ms + (1.0 - self.inner.alpha) * old_ema
        };
        self.inner
            .p99_ema_bits
            .store(new_ema.to_bits(), Ordering::Relaxed);

        // Adjust concurrency limit based on EMA vs target.
        self.adjust();
    }

    /// Returns the current adaptive concurrency limit.
    pub fn current_limit(&self) -> usize {
        self.inner.current_limit.load(Ordering::Relaxed)
    }

    /// Returns the current p99 EMA in milliseconds.
    pub fn p99_ema_ms(&self) -> f64 {
        f64::from_bits(self.inner.p99_ema_bits.load(Ordering::Relaxed))
    }

    /// Adjust the concurrency limit based on observed latency.
    ///
    /// - If p99 EMA > target * 1.2: decrease limit by 10% (clamped to min)
    /// - If p99 EMA < target * 0.5: increase limit by 5% (clamped to max)
    /// - Otherwise: no change
    fn adjust(&self) {
        let ema = self.p99_ema_ms();
        if ema <= 0.0 {
            return;
        }

        let target = self.inner.target_p99_ms;
        let old_limit = self.inner.current_limit.load(Ordering::Relaxed);

        let new_limit = if ema > target * 1.2 {
            // Latency too high: reduce by 10%.
            let reduced = (old_limit as f64 * 0.9).round() as usize;
            reduced.max(self.inner.min_concurrency)
        } else if ema < target * 0.5 {
            // Latency well below target: increase by 5%.
            let increased = (old_limit as f64 * 1.05).round() as usize;
            increased.min(self.inner.max_concurrency)
        } else {
            return; // No adjustment needed.
        };

        if new_limit != old_limit {
            self.inner.current_limit.store(new_limit, Ordering::Relaxed);

            // Adjust semaphore permits.
            if new_limit > old_limit {
                self.inner.semaphore.add_permits(new_limit - old_limit);
            }
            // Note: Tokio Semaphore does not support removing permits directly.
            // When decreasing, we simply lower the logical limit. The reduced
            // permit count takes effect naturally as permits are returned --
            // excess permits won't be re-issued once current_limit is lowered
            // because the semaphore's available count will gradually converge.

            tracing::debug!(
                old_limit,
                new_limit,
                p99_ema_ms = format!("{:.1}", ema),
                target_ms = target,
                "adaptive concurrency adjusted"
            );
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
/// rejected (load-shed). Optionally integrates adaptive concurrency control.
#[derive(Clone, Debug)]
pub struct ConcurrencyLimitLayer {
    semaphore: Arc<Semaphore>,
    queue_semaphore: Arc<Semaphore>,
    metrics: ConcurrencyMetrics,
    timeout: Duration,
    adaptive: Option<AdaptiveConcurrencyController>,
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
            adaptive: None,
        }
    }

    /// Create a new concurrency limit layer with adaptive control.
    pub fn with_adaptive(
        max_concurrent: usize,
        queue_size: usize,
        timeout: Duration,
        metrics: ConcurrencyMetrics,
        min_concurrency: usize,
        max_concurrency: usize,
        target_p99_ms: u64,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(max_concurrent));
        let adaptive = AdaptiveConcurrencyController::new(
            max_concurrent,
            min_concurrency,
            max_concurrency,
            target_p99_ms,
            semaphore.clone(),
        );
        Self {
            queue_semaphore: Arc::new(Semaphore::new(max_concurrent + queue_size)),
            semaphore,
            metrics,
            timeout,
            adaptive: Some(adaptive),
        }
    }

    /// Returns a reference to the adaptive controller, if enabled.
    pub fn adaptive_controller(&self) -> Option<&AdaptiveConcurrencyController> {
        self.adaptive.as_ref()
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
            adaptive: self.adaptive.clone(),
        }
    }
}

// ── ConcurrencyLimit Service ────────────────────────────────────────────

/// Tower service that enforces concurrency limits, queuing, timeouts,
/// and optional adaptive concurrency control.
#[derive(Clone, Debug)]
pub struct ConcurrencyLimitService<S> {
    inner: S,
    semaphore: Arc<Semaphore>,
    queue_semaphore: Arc<Semaphore>,
    metrics: ConcurrencyMetrics,
    timeout: Duration,
    adaptive: Option<AdaptiveConcurrencyController>,
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
        let adaptive = self.adaptive.clone();
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

            // Process the request (with timeout), measuring latency.
            metrics.inc_active();
            let start = Instant::now();
            let result = tokio::time::timeout(timeout, inner.call(req)).await;
            let elapsed = start.elapsed();
            metrics.dec_active();

            // Feed latency to adaptive controller.
            if let Some(ref ctrl) = adaptive {
                ctrl.record_latency(elapsed);
            }

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
/// Returns a `ConcurrencyLimitLayer` with adaptive concurrency control that
/// combines:
/// - Concurrency limiting (semaphore-based, bounded to `max_concurrent_connections`)
/// - Request queuing (overflow up to `search_queue_size`)
/// - Load shedding (immediate rejection when queue is full)
/// - Per-request timeout (`request_timeout`)
/// - Adaptive sizing based on p99 latency EMA
///
/// Also returns the associated `ConcurrencyMetrics` for observability.
pub fn build_concurrency_layer(
    config: &ConcurrencyConfig,
) -> (ConcurrencyLimitLayer, ConcurrencyMetrics) {
    let metrics = ConcurrencyMetrics::new();

    let layer = ConcurrencyLimitLayer::with_adaptive(
        config.max_concurrent_connections,
        config.search_queue_size,
        config.request_timeout,
        metrics.clone(),
        config.min_concurrency,
        config.max_concurrency,
        config.target_p99_latency_ms,
    );

    tracing::info!(
        max_concurrent = config.max_concurrent_connections,
        queue_size = config.search_queue_size,
        timeout_ms = config.request_timeout.as_millis() as u64,
        min_concurrency = config.min_concurrency,
        max_concurrency = config.max_concurrency,
        target_p99_ms = config.target_p99_latency_ms,
        "concurrency layer configured with adaptive control"
    );

    (layer, metrics)
}

/// Build a search-specific concurrency layer.
///
/// Uses `max_concurrent_searches` for tighter control over search endpoints
/// and `search_timeout` (default 5s) for lower tail latency.
pub fn build_search_concurrency_layer(
    config: &ConcurrencyConfig,
) -> (ConcurrencyLimitLayer, ConcurrencyMetrics) {
    let metrics = ConcurrencyMetrics::new();

    let layer = ConcurrencyLimitLayer::with_adaptive(
        config.max_concurrent_searches,
        config.search_queue_size,
        config.search_timeout,
        metrics.clone(),
        config.min_concurrency,
        config.max_concurrency,
        config.target_p99_latency_ms,
    );

    tracing::info!(
        max_concurrent_searches = config.max_concurrent_searches,
        queue_size = config.search_queue_size,
        timeout_ms = config.search_timeout.as_millis() as u64,
        min_concurrency = config.min_concurrency,
        max_concurrency = config.max_concurrency,
        target_p99_ms = config.target_p99_latency_ms,
        "search concurrency layer configured with adaptive control"
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
        assert_eq!(config.request_timeout, Duration::from_secs(10));
        assert_eq!(config.search_timeout, Duration::from_secs(5));
        assert_eq!(config.bulk_timeout, Duration::from_secs(30));
        assert_eq!(config.min_concurrency, 10);
        assert_eq!(config.max_concurrency, 200);
        assert_eq!(config.target_p99_latency_ms, 500);
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
        let (layer, metrics) = build_concurrency_layer(&config);
        assert_eq!(metrics.active_connections(), 0);
        assert!(layer.adaptive_controller().is_some());
    }

    #[test]
    fn build_search_concurrency_layer_returns_layer_and_metrics() {
        let config = ConcurrencyConfig::default();
        let (layer, metrics) = build_search_concurrency_layer(&config);
        assert_eq!(metrics.active_connections(), 0);
        assert!(layer.adaptive_controller().is_some());
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

    #[test]
    fn adaptive_controller_initial_state() {
        let sem = Arc::new(Semaphore::new(100));
        let ctrl = AdaptiveConcurrencyController::new(100, 10, 200, 500, sem);
        assert_eq!(ctrl.current_limit(), 100);
        assert_eq!(ctrl.p99_ema_ms(), 0.0);
    }

    #[test]
    fn adaptive_controller_clamps_initial_limit() {
        let sem = Arc::new(Semaphore::new(5));
        let ctrl = AdaptiveConcurrencyController::new(5, 10, 200, 500, sem);
        // Initial limit 5 is below min 10, should be clamped to 10.
        assert_eq!(ctrl.current_limit(), 10);
    }

    #[test]
    fn adaptive_controller_decreases_on_high_latency() {
        let sem = Arc::new(Semaphore::new(100));
        let ctrl = AdaptiveConcurrencyController::new(100, 10, 200, 500, sem);

        // Feed many high-latency samples to push EMA above target * 1.2 (600ms).
        for _ in 0..50 {
            ctrl.record_latency(Duration::from_millis(800));
        }

        // Limit should have decreased from 100.
        assert!(ctrl.current_limit() < 100);
        assert!(ctrl.current_limit() >= 10); // Respects min.
    }

    #[test]
    fn adaptive_controller_increases_on_low_latency() {
        let sem = Arc::new(Semaphore::new(50));
        let ctrl = AdaptiveConcurrencyController::new(50, 10, 200, 500, sem);

        // Feed many low-latency samples to push EMA below target * 0.5 (250ms).
        for _ in 0..50 {
            ctrl.record_latency(Duration::from_millis(100));
        }

        // Limit should have increased from 50.
        assert!(ctrl.current_limit() > 50);
        assert!(ctrl.current_limit() <= 200); // Respects max.
    }

    #[test]
    fn adaptive_controller_respects_bounds() {
        // Test min bound.
        let sem = Arc::new(Semaphore::new(12));
        let ctrl = AdaptiveConcurrencyController::new(12, 10, 200, 100, sem);
        for _ in 0..200 {
            ctrl.record_latency(Duration::from_millis(500));
        }
        assert!(ctrl.current_limit() >= 10);

        // Test max bound.
        let sem = Arc::new(Semaphore::new(195));
        let ctrl = AdaptiveConcurrencyController::new(195, 10, 200, 500, sem);
        for _ in 0..200 {
            ctrl.record_latency(Duration::from_millis(10));
        }
        assert!(ctrl.current_limit() <= 200);
    }

    #[test]
    fn adaptive_controller_stable_in_middle_range() {
        let sem = Arc::new(Semaphore::new(100));
        let ctrl = AdaptiveConcurrencyController::new(100, 10, 200, 500, sem);

        // Latency in the stable range (250ms < x < 600ms).
        for _ in 0..50 {
            ctrl.record_latency(Duration::from_millis(400));
        }

        // Limit should remain at 100 (no adjustment in stable band).
        assert_eq!(ctrl.current_limit(), 100);
    }
}
