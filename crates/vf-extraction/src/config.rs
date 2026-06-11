// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! The per-collection LLM configuration in three forms: the in-memory plaintext
//! config (api key zeroized and never serialized), the on-disk sealed config,
//! and a redacted view safe to return over the API.

use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::crypto::MasterKey;
use crate::error::ExtractionError;

/// In-memory LLM config. The plaintext api key is zeroized on drop and is never
/// serialized as plaintext (it is skipped by serde).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmConfig {
    pub base_url: String,
    #[serde(skip)]
    pub api_key: Zeroizing<String>,
    pub model_name: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub timeout_seconds: u64,
}

impl LlmConfig {
    /// Build an in-memory config from plain parts. The api key is wrapped in a
    /// zeroizing buffer so callers outside this crate never touch `zeroize`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_url: String,
        api_key: String,
        model_name: String,
        temperature: f32,
        max_tokens: u32,
        timeout_seconds: u64,
    ) -> Self {
        Self {
            base_url,
            api_key: Zeroizing::new(api_key),
            model_name,
            temperature,
            max_tokens,
            timeout_seconds,
        }
    }

    /// Seal this config to its on-disk form. The api key is encrypted under `mk`.
    pub fn seal(&self, mk: &MasterKey) -> Result<SealedLlmConfig, ExtractionError> {
        let api_key_sealed = mk.seal(self.api_key.as_str())?;
        Ok(SealedLlmConfig {
            base_url: self.base_url.clone(),
            api_key_sealed,
            model_name: self.model_name.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            timeout_seconds: self.timeout_seconds,
        })
    }

    /// A view safe to return over the API: the api key becomes a set/unset flag.
    pub fn redacted(&self) -> RedactedLlmConfig {
        RedactedLlmConfig {
            base_url: self.base_url.clone(),
            api_key: "***".to_string(),
            api_key_set: !self.api_key.is_empty(),
            model_name: self.model_name.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            timeout_seconds: self.timeout_seconds,
        }
    }
}

/// On-disk LLM config. The api key is stored sealed as base64(nonce || ct).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SealedLlmConfig {
    pub base_url: String,
    pub api_key_sealed: String,
    pub model_name: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub timeout_seconds: u64,
}

impl SealedLlmConfig {
    /// Unseal back into an in-memory config by decrypting the api key under `mk`.
    pub fn unseal(&self, mk: &MasterKey) -> Result<LlmConfig, ExtractionError> {
        let api_key = mk.open(&self.api_key_sealed)?;
        Ok(LlmConfig {
            base_url: self.base_url.clone(),
            api_key: Zeroizing::new(api_key.as_str().to_string()),
            model_name: self.model_name.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            timeout_seconds: self.timeout_seconds,
        })
    }
}

/// A redacted LLM config: the same fields minus the secret, plus a set flag.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedactedLlmConfig {
    pub base_url: String,
    /// Always the redaction marker; the real key is never returned.
    pub api_key: String,
    pub api_key_set: bool,
    pub model_name: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub timeout_seconds: u64,
}
