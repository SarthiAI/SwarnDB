// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Server configuration with JSON file support and environment variable overrides.

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

/// Configuration for the SwarnDB server.
#[derive(Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind address for the server.
    #[serde(default = "default_host")]
    pub host: String,

    /// Port for the gRPC service.
    #[serde(default = "default_grpc_port")]
    pub grpc_port: u16,

    /// Port for the REST API.
    #[serde(default = "default_rest_port")]
    pub rest_port: u16,

    /// Directory for persistent data storage.
    #[serde(default = "default_data_dir")]
    pub data_dir: String,

    /// Logging level (trace, debug, info, warn, error).
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// API keys for authentication. Empty means no auth required.
    #[serde(default, skip_serializing)]
    pub api_keys: Vec<String>,

    /// Maximum number of concurrent connections.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Request timeout in milliseconds.
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,

    /// Maximum number of concurrent search requests processed simultaneously.
    #[serde(default = "default_max_concurrent_searches")]
    pub max_concurrent_searches: usize,

    /// Minimum concurrency limit for adaptive sizing.
    #[serde(default = "default_min_concurrency")]
    pub min_concurrency: usize,

    /// Maximum concurrency limit for adaptive sizing.
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,

    /// Target p99 latency in milliseconds for adaptive concurrency control.
    #[serde(default = "default_target_p99_latency_ms")]
    pub target_p99_latency_ms: u64,

    /// EMA smoothing factor for adaptive concurrency (0.0..=1.0).
    #[serde(default = "default_concurrency_ema_alpha")]
    pub concurrency_ema_alpha: f64,

    /// High latency threshold multiplier for adaptive concurrency (triggers decrease).
    #[serde(default = "default_concurrency_high_threshold")]
    pub concurrency_high_threshold: f64,

    /// Low latency threshold multiplier for adaptive concurrency (triggers increase).
    #[serde(default = "default_concurrency_low_threshold")]
    pub concurrency_low_threshold: f64,

    /// Decrease rate for adaptive concurrency when latency is high (0.0..=1.0).
    #[serde(default = "default_concurrency_decrease_rate")]
    pub concurrency_decrease_rate: f64,

    /// Increase rate for adaptive concurrency when latency is low (0.0..=1.0).
    #[serde(default = "default_concurrency_increase_rate")]
    pub concurrency_increase_rate: f64,

    /// Maximum number of requests that can be queued when at capacity.
    #[serde(default = "default_search_queue_size")]
    pub search_queue_size: usize,

    /// Search-specific request timeout in milliseconds.
    #[serde(default = "default_search_timeout_ms")]
    pub search_timeout_ms: u64,

    /// Bulk operation timeout in milliseconds.
    #[serde(default = "default_bulk_timeout_ms")]
    pub bulk_timeout_ms: u64,

    /// Maximum allowed ef_search parameter for HNSW queries.
    #[serde(default = "default_max_ef_search")]
    pub max_ef_search: usize,

    /// Maximum allowed batch_lock_size for bulk insert operations.
    #[serde(default = "default_max_batch_lock_size")]
    pub max_batch_lock_size: u32,

    /// Maximum allowed wal_flush_every interval for bulk insert operations.
    #[serde(default = "default_max_wal_flush_interval")]
    pub max_wal_flush_interval: u32,

    /// Maximum allowed ef_construction override for bulk insert operations.
    #[serde(default = "default_max_ef_construction")]
    pub max_ef_construction: u32,

    /// Maximum effective search width cap for HNSW. Prevents excessive computation from large k values.
    #[serde(default = "default_max_ef")]
    pub max_ef: usize,

    /// Allowed CORS origins. Empty means allow any origin (open-source default).
    #[serde(default)]
    pub cors_origins: Vec<String>,

    /// Compute-intensive operation timeout in seconds.
    #[serde(default = "default_compute_timeout_secs")]
    pub compute_timeout_secs: u64,

    /// Maximum number of messages in a single gRPC bulk_insert stream.
    #[serde(default = "default_max_bulk_insert_messages")]
    pub max_bulk_insert_messages: u64,

    /// Maximum total payload bytes in a single gRPC bulk_insert stream.
    #[serde(default = "default_max_bulk_insert_payload_bytes")]
    pub max_bulk_insert_payload_bytes: u64,

    /// Maximum allowed k value for search queries.
    #[serde(default = "default_max_k")]
    pub max_k: u32,

    /// Maximum gRPC message size in bytes (encoding and decoding).
    #[serde(default = "default_max_message_size")]
    pub max_message_size: usize,

    /// Maximum REST request body size in bytes (default: 256 MB).
    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,

    /// Path to TLS certificate file (PEM). When set along with tls_key_path, enables TLS.
    #[serde(default)]
    pub tls_cert_path: Option<String>,

    /// Path to TLS private key file (PEM). When set along with tls_cert_path, enables TLS.
    #[serde(default)]
    pub tls_key_path: Option<String>,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_compute_timeout_secs() -> u64 {
    30
}

fn default_grpc_port() -> u16 {
    50051
}

fn default_rest_port() -> u16 {
    8080
}

fn default_data_dir() -> String {
    "./data".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_max_connections() -> usize {
    1000
}

fn default_request_timeout_ms() -> u64 {
    30000
}

fn default_max_concurrent_searches() -> usize {
    500
}

fn default_min_concurrency() -> usize {
    10
}

fn default_max_concurrency() -> usize {
    200
}

fn default_target_p99_latency_ms() -> u64 {
    500
}

fn default_concurrency_ema_alpha() -> f64 {
    0.1
}

fn default_concurrency_high_threshold() -> f64 {
    1.2
}

fn default_concurrency_low_threshold() -> f64 {
    0.5
}

fn default_concurrency_decrease_rate() -> f64 {
    0.10
}

fn default_concurrency_increase_rate() -> f64 {
    0.05
}

fn default_search_queue_size() -> usize {
    1000
}

fn default_search_timeout_ms() -> u64 {
    5000
}

fn default_bulk_timeout_ms() -> u64 {
    300000
}

fn default_max_ef_search() -> usize {
    10_000
}

fn default_max_batch_lock_size() -> u32 {
    10_000
}

fn default_max_wal_flush_interval() -> u32 {
    100_000
}

fn default_max_ef_construction() -> u32 {
    2000
}

fn default_max_ef() -> usize {
    100_000
}

fn default_max_bulk_insert_messages() -> u64 {
    100_000
}

fn default_max_bulk_insert_payload_bytes() -> u64 {
    512 * 1024 * 1024 // 512 MB
}

fn default_max_k() -> u32 {
    10_000
}

fn default_max_message_size() -> usize {
    64 * 1024 * 1024 // 64 MB
}

fn default_max_request_body_bytes() -> usize {
    256 * 1024 * 1024 // 256 MB
}

impl std::fmt::Debug for ServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerConfig")
            .field("host", &self.host)
            .field("grpc_port", &self.grpc_port)
            .field("rest_port", &self.rest_port)
            .field("data_dir", &self.data_dir)
            .field("log_level", &self.log_level)
            .field("api_keys", &format!("[REDACTED; {} keys]", self.api_keys.len()))
            .field("max_connections", &self.max_connections)
            .field("request_timeout_ms", &self.request_timeout_ms)
            .field("max_concurrent_searches", &self.max_concurrent_searches)
            .field("min_concurrency", &self.min_concurrency)
            .field("max_concurrency", &self.max_concurrency)
            .field("target_p99_latency_ms", &self.target_p99_latency_ms)
            .field("concurrency_ema_alpha", &self.concurrency_ema_alpha)
            .field("concurrency_high_threshold", &self.concurrency_high_threshold)
            .field("concurrency_low_threshold", &self.concurrency_low_threshold)
            .field("concurrency_decrease_rate", &self.concurrency_decrease_rate)
            .field("concurrency_increase_rate", &self.concurrency_increase_rate)
            .field("search_queue_size", &self.search_queue_size)
            .field("search_timeout_ms", &self.search_timeout_ms)
            .field("bulk_timeout_ms", &self.bulk_timeout_ms)
            .field("max_ef_search", &self.max_ef_search)
            .field("max_batch_lock_size", &self.max_batch_lock_size)
            .field("max_wal_flush_interval", &self.max_wal_flush_interval)
            .field("max_ef_construction", &self.max_ef_construction)
            .field("max_ef", &self.max_ef)
            .field("cors_origins", &self.cors_origins)
            .field("compute_timeout_secs", &self.compute_timeout_secs)
            .field("max_bulk_insert_messages", &self.max_bulk_insert_messages)
            .field("max_bulk_insert_payload_bytes", &self.max_bulk_insert_payload_bytes)
            .field("max_k", &self.max_k)
            .field("max_message_size", &self.max_message_size)
            .field("max_request_body_bytes", &self.max_request_body_bytes)
            .field("tls_cert_path", &self.tls_cert_path)
            .field("tls_key_path", &self.tls_key_path)
            .finish()
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            grpc_port: default_grpc_port(),
            rest_port: default_rest_port(),
            data_dir: default_data_dir(),
            log_level: default_log_level(),
            api_keys: Vec::new(),
            max_connections: default_max_connections(),
            request_timeout_ms: default_request_timeout_ms(),
            max_concurrent_searches: default_max_concurrent_searches(),
            min_concurrency: default_min_concurrency(),
            max_concurrency: default_max_concurrency(),
            target_p99_latency_ms: default_target_p99_latency_ms(),
            concurrency_ema_alpha: default_concurrency_ema_alpha(),
            concurrency_high_threshold: default_concurrency_high_threshold(),
            concurrency_low_threshold: default_concurrency_low_threshold(),
            concurrency_decrease_rate: default_concurrency_decrease_rate(),
            concurrency_increase_rate: default_concurrency_increase_rate(),
            search_queue_size: default_search_queue_size(),
            search_timeout_ms: default_search_timeout_ms(),
            bulk_timeout_ms: default_bulk_timeout_ms(),
            max_ef_search: default_max_ef_search(),
            max_batch_lock_size: default_max_batch_lock_size(),
            max_wal_flush_interval: default_max_wal_flush_interval(),
            max_ef_construction: default_max_ef_construction(),
            max_ef: default_max_ef(),
            cors_origins: Vec::new(),
            compute_timeout_secs: default_compute_timeout_secs(),
            max_bulk_insert_messages: default_max_bulk_insert_messages(),
            max_bulk_insert_payload_bytes: default_max_bulk_insert_payload_bytes(),
            max_k: default_max_k(),
            max_message_size: default_max_message_size(),
            max_request_body_bytes: default_max_request_body_bytes(),
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

impl ServerConfig {
    /// Load configuration with precedence: env vars > config file > defaults.
    ///
    /// Checks `SWARNDB_CONFIG` env var for config file path, falling back to
    /// `swarndb.json` in the current directory. If no config file is found,
    /// starts from defaults.
    pub fn load() -> Self {
        let config_path = env::var("SWARNDB_CONFIG").unwrap_or_else(|_| "swarndb.json".to_string());

        let mut config = if Path::new(&config_path).exists() {
            match Self::from_file(&config_path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("Failed to load config from {}: {}, using defaults", config_path, e);
                    Self::default()
                }
            }
        } else {
            Self::default()
        };

        config.apply_env_overrides();
        config
    }

    /// Load configuration from a JSON file at the given path.
    pub fn from_file(path: &str) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path).map_err(|e| {
            ConfigError::FileError(format!("{}: {}", path, e))
        })?;

        serde_json::from_str(&contents).map_err(|e| {
            ConfigError::ParseError(format!("{}: {}", path, e))
        })
    }

    /// Apply environment variable overrides to this configuration.
    ///
    /// Environment variables:
    /// - `SWARNDB_HOST` -> host
    /// - `SWARNDB_GRPC_PORT` -> grpc_port
    /// - `SWARNDB_REST_PORT` -> rest_port
    /// - `SWARNDB_DATA_DIR` -> data_dir
    /// - `SWARNDB_WAL_MAX_SIZE` -> WAL rotation size in bytes (read by vf-storage)
    /// - `SWARNDB_LOG_LEVEL` -> log_level
    /// - `SWARNDB_API_KEYS` -> api_keys (comma-separated)
    /// - `SWARNDB_MAX_CONNECTIONS` -> max_connections
    /// - `SWARNDB_REQUEST_TIMEOUT_MS` -> request_timeout_ms
    /// - `SWARNDB_MAX_CONCURRENT_SEARCHES` -> max_concurrent_searches
    /// - `SWARNDB_MIN_CONCURRENCY` -> min_concurrency
    /// - `SWARNDB_MAX_CONCURRENCY` -> max_concurrency
    /// - `SWARNDB_TARGET_P99_LATENCY_MS` -> target_p99_latency_ms
    /// - `SWARNDB_CONCURRENCY_EMA_ALPHA` -> concurrency_ema_alpha
    /// - `SWARNDB_CONCURRENCY_HIGH_THRESHOLD` -> concurrency_high_threshold
    /// - `SWARNDB_CONCURRENCY_LOW_THRESHOLD` -> concurrency_low_threshold
    /// - `SWARNDB_CONCURRENCY_DECREASE_RATE` -> concurrency_decrease_rate
    /// - `SWARNDB_CONCURRENCY_INCREASE_RATE` -> concurrency_increase_rate
    /// - `SWARNDB_SEARCH_QUEUE_SIZE` -> search_queue_size
    /// - `SWARNDB_SEARCH_TIMEOUT_MS` -> search_timeout_ms
    /// - `SWARNDB_BULK_TIMEOUT_MS` -> bulk_timeout_ms
    /// - `SWARNDB_MAX_EF_SEARCH` -> max_ef_search
    /// - `SWARNDB_MAX_BATCH_LOCK_SIZE` -> max_batch_lock_size
    /// - `SWARNDB_MAX_WAL_FLUSH_INTERVAL` -> max_wal_flush_interval
    /// - `SWARNDB_MAX_EF_CONSTRUCTION` -> max_ef_construction
    /// - `SWARNDB_MAX_EF` -> max_ef
    /// - `SWARNDB_MAX_MESSAGE_SIZE` -> max_message_size
    /// - `SWARNDB_MAX_REQUEST_BODY_BYTES` -> max_request_body_bytes
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = env::var("SWARNDB_HOST") {
            self.host = val;
        }

        if let Ok(val) = env::var("SWARNDB_GRPC_PORT") {
            if let Ok(port) = val.parse::<u16>() {
                self.grpc_port = port;
            } else {
                tracing::warn!("Invalid SWARNDB_GRPC_PORT value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_REST_PORT") {
            if let Ok(port) = val.parse::<u16>() {
                self.rest_port = port;
            } else {
                tracing::warn!("Invalid SWARNDB_REST_PORT value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_DATA_DIR") {
            self.data_dir = val;
        }

        if let Ok(val) = env::var("SWARNDB_LOG_LEVEL") {
            self.log_level = val;
        }

        if let Ok(val) = env::var("SWARNDB_API_KEYS") {
            self.api_keys = val.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        }

        if let Ok(val) = env::var("SWARNDB_MAX_CONNECTIONS") {
            if let Ok(n) = val.parse::<usize>() {
                self.max_connections = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_CONNECTIONS value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_REQUEST_TIMEOUT_MS") {
            if let Ok(n) = val.parse::<u64>() {
                self.request_timeout_ms = n;
            } else {
                tracing::warn!("Invalid SWARNDB_REQUEST_TIMEOUT_MS value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_CONCURRENT_SEARCHES") {
            if let Ok(n) = val.parse::<usize>() {
                self.max_concurrent_searches = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_CONCURRENT_SEARCHES value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MIN_CONCURRENCY") {
            if let Ok(n) = val.parse::<usize>() {
                self.min_concurrency = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MIN_CONCURRENCY value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_CONCURRENCY") {
            if let Ok(n) = val.parse::<usize>() {
                self.max_concurrency = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_CONCURRENCY value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_TARGET_P99_LATENCY_MS") {
            if let Ok(n) = val.parse::<u64>() {
                self.target_p99_latency_ms = n;
            } else {
                tracing::warn!("Invalid SWARNDB_TARGET_P99_LATENCY_MS value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_CONCURRENCY_EMA_ALPHA") {
            if let Ok(n) = val.parse::<f64>() {
                self.concurrency_ema_alpha = n;
            } else {
                tracing::warn!("Invalid SWARNDB_CONCURRENCY_EMA_ALPHA value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_CONCURRENCY_HIGH_THRESHOLD") {
            if let Ok(n) = val.parse::<f64>() {
                self.concurrency_high_threshold = n;
            } else {
                tracing::warn!("Invalid SWARNDB_CONCURRENCY_HIGH_THRESHOLD value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_CONCURRENCY_LOW_THRESHOLD") {
            if let Ok(n) = val.parse::<f64>() {
                self.concurrency_low_threshold = n;
            } else {
                tracing::warn!("Invalid SWARNDB_CONCURRENCY_LOW_THRESHOLD value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_CONCURRENCY_DECREASE_RATE") {
            if let Ok(n) = val.parse::<f64>() {
                self.concurrency_decrease_rate = n;
            } else {
                tracing::warn!("Invalid SWARNDB_CONCURRENCY_DECREASE_RATE value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_CONCURRENCY_INCREASE_RATE") {
            if let Ok(n) = val.parse::<f64>() {
                self.concurrency_increase_rate = n;
            } else {
                tracing::warn!("Invalid SWARNDB_CONCURRENCY_INCREASE_RATE value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_SEARCH_QUEUE_SIZE") {
            if let Ok(n) = val.parse::<usize>() {
                self.search_queue_size = n;
            } else {
                tracing::warn!("Invalid SWARNDB_SEARCH_QUEUE_SIZE value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_SEARCH_TIMEOUT_MS") {
            if let Ok(n) = val.parse::<u64>() {
                self.search_timeout_ms = n;
            } else {
                tracing::warn!("Invalid SWARNDB_SEARCH_TIMEOUT_MS value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_BULK_TIMEOUT_MS") {
            if let Ok(n) = val.parse::<u64>() {
                self.bulk_timeout_ms = n;
            } else {
                tracing::warn!("Invalid SWARNDB_BULK_TIMEOUT_MS value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_EF_SEARCH") {
            if let Ok(n) = val.parse::<usize>() {
                self.max_ef_search = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_EF_SEARCH value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_BATCH_LOCK_SIZE") {
            if let Ok(n) = val.parse::<u32>() {
                self.max_batch_lock_size = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_BATCH_LOCK_SIZE value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_WAL_FLUSH_INTERVAL") {
            if let Ok(n) = val.parse::<u32>() {
                self.max_wal_flush_interval = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_WAL_FLUSH_INTERVAL value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_EF_CONSTRUCTION") {
            if let Ok(n) = val.parse::<u32>() {
                self.max_ef_construction = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_EF_CONSTRUCTION value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_EF") {
            if let Ok(n) = val.parse::<usize>() {
                if n == 0 {
                    tracing::warn!("SWARNDB_MAX_EF=0 is invalid, keeping default ({})", self.max_ef);
                } else {
                    self.max_ef = n;
                }
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_EF value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_CORS_ORIGINS") {
            self.cors_origins = val.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        }

        if let Ok(val) = env::var("SWARNDB_COMPUTE_TIMEOUT_SECS") {
            if let Ok(n) = val.parse::<u64>() {
                self.compute_timeout_secs = n;
            } else {
                tracing::warn!("Invalid SWARNDB_COMPUTE_TIMEOUT_SECS value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_BULK_INSERT_MESSAGES") {
            if let Ok(n) = val.parse::<u64>() {
                self.max_bulk_insert_messages = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_BULK_INSERT_MESSAGES value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_BULK_INSERT_PAYLOAD_BYTES") {
            if let Ok(n) = val.parse::<u64>() {
                self.max_bulk_insert_payload_bytes = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_BULK_INSERT_PAYLOAD_BYTES value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_K") {
            if let Ok(n) = val.parse::<u32>() {
                self.max_k = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_K value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_MESSAGE_SIZE") {
            if let Ok(n) = val.parse::<usize>() {
                self.max_message_size = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_MESSAGE_SIZE value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_MAX_REQUEST_BODY_BYTES") {
            if let Ok(n) = val.parse::<usize>() {
                self.max_request_body_bytes = n;
            } else {
                tracing::warn!("Invalid SWARNDB_MAX_REQUEST_BODY_BYTES value: {}", val);
            }
        }

        if let Ok(val) = env::var("SWARNDB_TLS_CERT_PATH") {
            self.tls_cert_path = Some(val);
        }

        if let Ok(val) = env::var("SWARNDB_TLS_KEY_PATH") {
            self.tls_key_path = Some(val);
        }
    }

    /// Returns the gRPC socket address string (e.g., "0.0.0.0:50051").
    pub fn grpc_addr(&self) -> String {
        format!("{}:{}", self.host, self.grpc_port)
    }

    /// Returns the REST socket address string (e.g., "0.0.0.0:8080").
    pub fn rest_addr(&self) -> String {
        format!("{}:{}", self.host, self.rest_port)
    }

    /// Returns true if TLS is configured (both cert and key paths are set).
    pub fn is_tls_enabled(&self) -> bool {
        self.tls_cert_path.is_some() && self.tls_key_path.is_some()
    }
}

/// Errors that can occur during configuration loading.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Failed to read the configuration file.
    #[error("config file error: {0}")]
    FileError(String),

    /// Failed to parse the configuration file contents.
    #[error("parse error: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_expected_values() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.grpc_port, 50051);
        assert_eq!(config.rest_port, 8080);
        assert_eq!(config.data_dir, "./data");
        assert_eq!(config.log_level, "info");
        assert!(config.api_keys.is_empty());
        assert_eq!(config.max_connections, 1000);
        assert_eq!(config.request_timeout_ms, 30000);
        assert_eq!(config.max_concurrent_searches, 500);
        assert_eq!(config.min_concurrency, 10);
        assert_eq!(config.max_concurrency, 200);
        assert_eq!(config.target_p99_latency_ms, 500);
        assert_eq!(config.search_timeout_ms, 5000);
        assert_eq!(config.bulk_timeout_ms, 300000);
        assert_eq!(config.max_ef_search, 10_000);
        assert_eq!(config.max_batch_lock_size, 10_000);
        assert_eq!(config.max_wal_flush_interval, 100_000);
        assert_eq!(config.max_ef_construction, 2000);
        assert_eq!(config.max_ef, 100_000);
    }

    #[test]
    fn grpc_addr_formats_correctly() {
        let config = ServerConfig::default();
        assert_eq!(config.grpc_addr(), "0.0.0.0:50051");
    }

    #[test]
    fn rest_addr_formats_correctly() {
        let config = ServerConfig::default();
        assert_eq!(config.rest_addr(), "0.0.0.0:8080");
    }

    #[test]
    fn from_file_returns_error_for_missing_file() {
        let result = ServerConfig::from_file("/nonexistent/config.json");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ConfigError::FileError(_)));
    }

    #[test]
    fn serialization_roundtrip() {
        let config = ServerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.host, config.host);
        assert_eq!(parsed.grpc_port, config.grpc_port);
        assert_eq!(parsed.rest_port, config.rest_port);
    }

    #[test]
    fn partial_json_uses_defaults() {
        let json = r#"{"host": "127.0.0.1", "grpc_port": 9090}"#;
        let config: ServerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.grpc_port, 9090);
        assert_eq!(config.rest_port, 8080); // default
        assert_eq!(config.data_dir, "./data"); // default
    }
}
