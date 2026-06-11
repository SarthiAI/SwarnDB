// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! At-rest sealing of api keys with AES-256-GCM. The master key comes from an
//! environment variable holding base64 of 32 bytes. Sealed values are
//! base64(nonce_12 || ciphertext). Key bytes and opened plaintext are zeroized.

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, Key, Nonce};
use base64::Engine;
use zeroize::Zeroizing;

use crate::error::ExtractionError;

/// Length in bytes of the AES-256 master key.
const KEY_LEN: usize = 32;
/// Length in bytes of the AES-GCM nonce.
const NONCE_LEN: usize = 12;

/// The server master key, held zeroized in memory.
pub struct MasterKey {
    key: Zeroizing<[u8; KEY_LEN]>,
}

impl MasterKey {
    /// Build a master key from `var`. Returns `Ok(None)` when the variable is
    /// unset, and an error when it is set but not valid base64 of 32 bytes.
    pub fn from_base64_env(var: &str) -> Result<Option<Self>, ExtractionError> {
        let encoded = match std::env::var(var) {
            Ok(v) if !v.trim().is_empty() => v,
            Ok(_) => return Ok(None),
            Err(std::env::VarError::NotPresent) => return Ok(None),
            Err(e) => return Err(ExtractionError::Config(e.to_string())),
        };
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded.trim().as_bytes())
            .map_err(|e| ExtractionError::Config(format!("master key not valid base64: {}", e)))?;
        if decoded.len() != KEY_LEN {
            return Err(ExtractionError::Config(format!(
                "master key must be {} bytes, got {}",
                KEY_LEN,
                decoded.len()
            )));
        }
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        key.copy_from_slice(&decoded);
        Ok(Some(Self { key }))
    }

    /// Build a master key directly from raw bytes. Errors when not 32 bytes long.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ExtractionError> {
        if bytes.len() != KEY_LEN {
            return Err(ExtractionError::Config(format!(
                "master key must be {} bytes, got {}",
                KEY_LEN,
                bytes.len()
            )));
        }
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        key.copy_from_slice(bytes);
        Ok(Self { key })
    }

    /// Seal a plaintext string, returning base64(nonce_12 || ciphertext).
    pub fn seal(&self, plaintext: &str) -> Result<String, ExtractionError> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(self.key.as_slice()));
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, plaintext.as_bytes())
            .map_err(|e| ExtractionError::Crypto(format!("seal failed: {}", e)))?;
        let mut combined = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        combined.extend_from_slice(nonce.as_slice());
        combined.extend_from_slice(&ciphertext);
        Ok(base64::engine::general_purpose::STANDARD.encode(&combined))
    }

    /// Open a sealed value produced by `seal`, returning zeroized plaintext.
    pub fn open(&self, sealed: &str) -> Result<Zeroizing<String>, ExtractionError> {
        let combined = base64::engine::general_purpose::STANDARD
            .decode(sealed.as_bytes())
            .map_err(|e| ExtractionError::Crypto(format!("sealed value not valid base64: {}", e)))?;
        if combined.len() < NONCE_LEN {
            return Err(ExtractionError::Crypto(
                "sealed value too short to contain a nonce".to_string(),
            ));
        }
        let (nonce_bytes, ciphertext) = combined.split_at(NONCE_LEN);
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(self.key.as_slice()));
        let nonce = Nonce::from_slice(nonce_bytes);
        let plaintext: Zeroizing<Vec<u8>> = Zeroizing::new(
            cipher
                .decrypt(nonce, ciphertext)
                .map_err(|e| ExtractionError::Crypto(format!("open failed: {}", e)))?,
        );
        let text = std::str::from_utf8(&plaintext)
            .map_err(|e| ExtractionError::Crypto(format!("opened value not valid utf8: {}", e)))?
            .to_string();
        Ok(Zeroizing::new(text))
    }
}
