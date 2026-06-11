// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Document-update diff types for re-extraction. A diff classifies each chunk
//! of an updated document against the per-collection stored content hashes so
//! re-extraction only re-pays the LLM for changed and new chunks and prunes the
//! edges and nodes of removed chunks.

use serde::{Deserialize, Serialize};

/// How a chunk changed relative to the stored content hashes for its document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkDiffAction {
    Unchanged,
    Changed,
    New,
    Deleted,
}

/// The diff verdict for a single chunk.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkDiff {
    pub chunk_id: u64,
    pub action: ChunkDiffAction,
}

/// The outcome of a document re-extraction: per-action counts plus the id of
/// the enqueued extraction job (empty when nothing needed re-extraction).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReextractSummary {
    pub job_id: String,
    pub unchanged: u64,
    pub changed: u64,
    pub added: u64,
    pub deleted: u64,
    pub edges_deleted: u64,
    pub nodes_deleted: u64,
}
