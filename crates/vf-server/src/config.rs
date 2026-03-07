// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Server configuration with JSON file support and environment variable overrides.

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

/// Configuration for the SwarnDB server.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    #[serde(default)]
    pub api_keys: Vec<String>,

    /// Maximum number of concurrent connections.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Request timeout in milliseconds.
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
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
    /// - `SWARNDB_LOG_LEVEL` -> log_level
    /// - `SWARNDB_API_KEYS` -> api_keys (comma-separated)
    /// - `SWARNDB_MAX_CONNECTIONS` -> max_connections
    /// - `SWARNDB_REQUEST_TIMEOUT_MS` -> request_timeout_ms
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
    }

    /// Returns the gRPC socket address string (e.g., "0.0.0.0:50051").
    pub fn grpc_addr(&self) -> String {
        format!("{}:{}", self.host, self.grpc_port)
    }

    /// Returns the REST socket address string (e.g., "0.0.0.0:8080").
    pub fn rest_addr(&self) -> String {
        format!("{}:{}", self.host, self.rest_port)
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
