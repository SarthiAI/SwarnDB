// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Corpus-level re-extraction types (Area 3). A corpus job re-extracts a set of
//! documents or a whole collection by driving the existing per-document
//! diff + re-extract path once per document, aggregating per-document progress
//! into a master status. The master status is resumable: its committed progress
//! (the set of completed doc ids) is persisted so a re-issued call skips already
//! finished documents and continues.

use serde::{Deserialize, Serialize};

use crate::pipeline::JobState;

/// One document's outcome within a corpus re-extraction run.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorpusDocState {
    /// Not yet started.
    Pending,
    /// Its sub-job finished cleanly (or had nothing to do).
    Completed,
    /// Its sub-job ended in completed_with_errors: a partial success per
    /// ADR-008. Most chunks landed; some failed. Reported honestly, not as a
    /// hard failure.
    CompletedWithErrors,
    /// Its sub-job ended in failed (or cancelled).
    Failed,
    /// Skipped because a prior run already completed it (resume).
    Skipped,
}

/// Per-document progress recorded on the master corpus status.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusDocProgress {
    pub doc_id: String,
    pub state: CorpusDocState,
    /// The per-document sub-job id (empty when nothing needed re-extraction).
    pub job_id: String,
    pub changed: u64,
    pub added: u64,
    pub deleted: u64,
}

/// The master status of a corpus re-extraction job. Reuses `JobState` for the
/// overall lifecycle and mirrors the ADR-008 partial-success model: a mix of
/// completed and failed documents finalizes to `CompletedWithErrors`.
///
/// The committed progress is the `completed_docs` set together with the
/// per-document `documents` list; both are persisted to disk after each
/// document so a re-issued call with the same resume token (the corpus job id)
/// skips already finished documents.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusJobStatus {
    pub corpus_job_id: String,
    pub collection: String,
    pub state: JobState,
    /// Total documents targeted by this run.
    pub total_documents: usize,
    /// Documents finished (completed, failed, or skipped).
    pub processed_documents: usize,
    /// Documents whose sub-job ended in a hard failed state. A document that
    /// ended in completed_with_errors is a partial success (ADR-008) and is NOT
    /// counted here; it is surfaced via its per-document CompletedWithErrors
    /// state in `documents`.
    pub failed_documents: usize,
    /// Documents skipped because a prior run already completed them.
    pub skipped_documents: usize,
    /// Aggregate chunk-level counters rolled up from every sub-job.
    pub changed_chunks: u64,
    pub added_chunks: u64,
    pub deleted_chunks: u64,
    pub edges_deleted: u64,
    pub nodes_deleted: u64,
    pub entities_written: u64,
    pub edges_written: u64,
    /// Per-document progress. Bounded by the number of target documents.
    pub documents: Vec<CorpusDocProgress>,
    /// A terminal error message when the whole run failed before any document.
    pub error: Option<String>,
}

impl CorpusJobStatus {
    /// A fresh queued master status over `total_documents` documents.
    pub fn queued(corpus_job_id: String, collection: String, total_documents: usize) -> Self {
        Self {
            corpus_job_id,
            collection,
            state: JobState::Queued,
            total_documents,
            processed_documents: 0,
            failed_documents: 0,
            skipped_documents: 0,
            changed_chunks: 0,
            added_chunks: 0,
            deleted_chunks: 0,
            edges_deleted: 0,
            nodes_deleted: 0,
            entities_written: 0,
            edges_written: 0,
            documents: Vec::new(),
            error: None,
        }
    }
}
