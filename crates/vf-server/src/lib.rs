// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

pub mod auth;
pub mod concurrency;
pub mod config;
pub mod convert;
pub mod grpc_collection;
pub mod grpc_graph;
pub mod grpc_search;
pub mod grpc_vector;
pub mod grpc_vector_math;
pub mod health;
pub mod logging;
pub mod metrics;
pub mod rest;
pub mod shutdown;
pub mod state;
pub mod validation;

pub use config::{ConfigError, ServerConfig};
pub use validation::{ValidationConfig, ValidationError};

pub mod proto {
    pub mod swarndb {
        pub mod v1 {
            tonic::include_proto!("swarndb.v1");
        }
    }
}
