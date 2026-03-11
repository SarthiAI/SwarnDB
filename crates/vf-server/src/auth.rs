// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use tonic::service::Interceptor;

#[derive(Clone)]
pub struct AuthState {
    pub api_keys: Vec<String>,
}

impl AuthState {
    pub fn new(keys: Vec<String>) -> Self {
        Self { api_keys: keys }
    }

    pub fn is_auth_required(&self) -> bool {
        !self.api_keys.is_empty()
    }
}

/// Constant-time comparison that hashes both inputs first to prevent
/// timing side-channels from length differences or byte-by-byte comparison.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    use sha2::{Sha256, Digest};
    let hash_a = Sha256::digest(a);
    let hash_b = Sha256::digest(b);
    let mut result: u8 = 0;
    for (x, y) in hash_a.iter().zip(hash_b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

pub async fn api_key_auth(
    State(auth): State<AuthState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if !auth.is_auth_required() {
        return Ok(next.run(request).await);
    }

    let api_key = request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            // Fall back to Authorization: Bearer <key>
            request
                .headers()
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
        });

    match api_key {
        Some(key) if auth.api_keys.iter().any(|k| constant_time_eq(k.as_bytes(), key.as_bytes())) => {
            Ok(next.run(request).await)
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// gRPC interceptor that checks the API key from request metadata.
///
/// Checks "x-api-key" and "authorization" metadata headers.
/// If no API keys are configured, all requests are allowed.
#[derive(Clone)]
pub struct GrpcAuthInterceptor {
    auth: AuthState,
}

impl GrpcAuthInterceptor {
    pub fn new(auth: AuthState) -> Self {
        Self { auth }
    }
}

impl Interceptor for GrpcAuthInterceptor {
    fn call(
        &mut self,
        request: tonic::Request<()>,
    ) -> Result<tonic::Request<()>, tonic::Status> {
        if !self.auth.is_auth_required() {
            return Ok(request);
        }

        let metadata = request.metadata();

        // Check x-api-key header
        let api_key = metadata
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .or_else(|| {
                // Fall back to authorization header (Bearer <key>)
                metadata
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.strip_prefix("Bearer "))
            });

        match api_key {
            Some(key)
                if self
                    .auth
                    .api_keys
                    .iter()
                    .any(|k| constant_time_eq(k.as_bytes(), key.as_bytes())) =>
            {
                Ok(request)
            }
            _ => Err(tonic::Status::unauthenticated("invalid or missing API key")),
        }
    }
}
