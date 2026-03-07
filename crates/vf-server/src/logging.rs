// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Structured logging with tracing for the SwarnDB server.
//!
//! Provides JSON-formatted structured log output, configurable log levels,
//! request ID generation, and latency span helpers for operation tracing.

use tracing::Span;
use tracing_subscriber::{
    fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};
use uuid::Uuid;

/// Initialize the global tracing subscriber with structured JSON output.
///
/// Sets up a layered subscriber with:
/// - An [`EnvFilter`] parsed from the given `log_level` string (e.g., "info", "debug",
///   "vf_server=debug,tower_http=trace"). Falls back to "info" if parsing fails.
/// - JSON formatting for machine-readable structured logs, including timestamps,
///   target module, span context, and thread information.
///
/// The `RUST_LOG` environment variable takes precedence over `log_level` when the
/// `EnvFilter` is constructed, following standard `tracing_subscriber` behavior.
///
/// # Panics
///
/// Panics if a global subscriber has already been set (e.g., called twice).
///
/// # Examples
///
/// ```no_run
/// vf_server::logging::init_logging("info");
/// ```
pub fn init_logging(log_level: &str) {
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(log_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let json_layer = fmt::layer()
        .json()
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_span_list(true)
        .with_current_span(true)
        .flatten_event(false);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(json_layer)
        .init();

    tracing::info!(
        log_level = log_level,
        "Logging initialized with structured JSON output"
    );
}

/// Generate a new UUID v4 request ID.
///
/// Returns a hyphenated UUID string suitable for use as a unique request identifier
/// in structured log spans.
///
/// # Examples
///
/// ```
/// let id = vf_server::logging::generate_request_id();
/// assert_eq!(id.len(), 36); // UUID v4 hyphenated format
/// ```
pub fn generate_request_id() -> String {
    Uuid::new_v4().to_string()
}

/// Create a tracing span for an incoming request with a unique request ID.
///
/// The returned span includes:
/// - `request_id`: a UUID v4 identifying this request
/// - `method`: the RPC or HTTP method name
///
/// All log events and child spans within this span will automatically include
/// the request ID in their structured output.
///
/// # Examples
///
/// ```no_run
/// use tracing::Instrument;
///
/// let span = vf_server::logging::request_span("SearchVectors");
/// async {
///     tracing::info!("processing search request");
/// }.instrument(span);
/// ```
pub fn request_span(method: &str) -> Span {
    let request_id = generate_request_id();
    tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %method,
    )
}

/// Create a tracing span for timing a named operation.
///
/// Use this to wrap database operations, index lookups, or any unit of work
/// where latency tracking is desired. The span records the operation name
/// and is emitted at the INFO level so that elapsed time appears in JSON logs
/// when the span closes.
///
/// # Examples
///
/// ```no_run
/// use tracing::Instrument;
///
/// async fn do_search() {
///     let span = vf_server::logging::operation_span("hnsw_search");
///     async {
///         // ... perform search ...
///         tracing::info!(results = 42, "search complete");
///     }.instrument(span);
/// }
/// ```
pub fn operation_span(operation: &str) -> Span {
    tracing::info_span!(
        "operation",
        op = %operation,
    )
}

/// Create a tracing span for an operation that may fail, with structured error context.
///
/// The returned span includes an `otel.status_code` field (initially empty) that can be
/// set to "ERROR" on failure, and an `error.message` field for the error description.
/// This pattern is compatible with OpenTelemetry conventions.
///
/// # Examples
///
/// ```no_run
/// use tracing::Instrument;
///
/// let span = vf_server::logging::error_span("insert_vector");
/// let _guard = span.enter();
/// // On error:
/// span.record("otel.status_code", "ERROR");
/// span.record("error.message", "dimension mismatch: expected 128, got 64");
/// tracing::error!("operation failed");
/// ```
pub fn error_span(operation: &str) -> Span {
    tracing::info_span!(
        "failable_op",
        op = %operation,
        otel.status_code = tracing::field::Empty,
        error.message = tracing::field::Empty,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_request_id_returns_valid_uuid() {
        let id = generate_request_id();
        assert_eq!(id.len(), 36);
        // Verify it parses as a valid UUID.
        assert!(Uuid::parse_str(&id).is_ok());
    }

    #[test]
    fn generate_request_id_is_unique() {
        let id1 = generate_request_id();
        let id2 = generate_request_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn request_span_has_expected_fields() {
        let span = request_span("TestMethod");
        assert!(span.metadata().is_some());
        assert_eq!(span.metadata().unwrap().name(), "request");
    }

    #[test]
    fn operation_span_has_expected_name() {
        let span = operation_span("test_op");
        assert_eq!(span.metadata().unwrap().name(), "operation");
    }

    #[test]
    fn error_span_has_expected_name() {
        let span = error_span("test_failable");
        assert_eq!(span.metadata().unwrap().name(), "failable_op");
    }
}
