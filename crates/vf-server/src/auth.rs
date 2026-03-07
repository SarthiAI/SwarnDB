// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};

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

/// Constant-time byte comparison to prevent timing attacks.
/// Note: length difference is still detectable, but API keys should
/// typically be uniform length, and timing attacks over the network
/// require microsecond precision which is impractical.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
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
        .and_then(|v| v.to_str().ok());

    match api_key {
        Some(key) if auth.api_keys.iter().any(|k| constant_time_eq(k.as_bytes(), key.as_bytes())) => {
            Ok(next.run(request).await)
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
