// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! Shared helpers for the bulk insert checkpoint and resume-token flow.
//!
//! These helpers are used by both the REST and gRPC bulk insert handlers to
//! produce identical resume tokens and to resolve the on-disk checkpoint
//! path for a collection. Keeping them in one place avoids drift between
//! the two transports.

use base64::Engine;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use crate::state::AppState;

// Hash a collection name to a stable u64. Used only to detect on-disk
// checkpoint drift across restarts. Collisions across collection names are
// tolerated because the resume token also embeds the collection name in
// plain text, so the server has a second confirmation channel.
pub fn hash_collection_name(collection: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    collection.hash(&mut hasher);
    hasher.finish()
}

// Build the resume token by base64-encoding "<collection>:<lsn>". The format
// is opaque to the client; only the server needs to round-trip it.
pub fn encode_resume_token(collection: &str, last_committed_lsn: u64) -> String {
    let raw = format!("{}:{}", collection, last_committed_lsn);
    base64::engine::general_purpose::STANDARD.encode(raw.as_bytes())
}

// Resolve <data_dir>/<collection>/bulk_insert.checkpoint, or None when the
// storage layer cannot locate the collection (in-memory only collections).
pub fn resolve_bulk_checkpoint_path(state: &AppState, collection: &str) -> Option<PathBuf> {
    let cm = state.collection_manager.read();
    let storage_coll = cm.get_collection(collection).ok()?;
    let dir = storage_coll.collection_dir().to_path_buf();
    Some(dir.join("bulk_insert.checkpoint"))
}
