// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! The extraction manager: per-collection runtime state, the sealed-config and
//! ontology persistence, a bounded tokio worker pool over a fair per-chunk
//! scheduler, and the public API the server's RPC and REST handlers call into.
//!
//! Scheduling (ADR-016): work is dispatched ONE chunk at a time, round-robin
//! across all jobs that have pending chunks, so a large job cannot monopolize the
//! pool and starve a freshly submitted one. Cancellation is honored cooperatively
//! before each chunk is dispatched and before a worker processes it, so a
//! cancelled job (or a job whose collection was dropped) stops promptly instead of
//! after every queued chunk drains.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use metrics::{counter, gauge};

use tokio::sync::{watch, Notify};
use vf_core::types::{CollectionConfig, Mode};
use vf_graph::NodeId;

use crate::adapter::{ChunkContent, CostEstimate, ExtractionAdapter, ExtractionResult};
use crate::cache::{
    cache_key, chunk_content_hash, custom_prompt_hash, global_cache_dir, normalize_text,
    ExtractionCache,
};
use crate::config::{LlmConfig, RedactedLlmConfig, SealedLlmConfig};
use crate::corpus::{CorpusDocProgress, CorpusDocState, CorpusJobStatus};
use crate::cost::{HeuristicEstimator, PricingTable, TokenEstimator};
use crate::crypto::MasterKey;
use crate::diff::{ChunkDiff, ChunkDiffAction, ReextractSummary};
use crate::error::ExtractionError;
use crate::ontology::{template_by_name, Ontology, OntologyProposal, ProposalStatus};
use crate::openai::OpenAICompatibleAdapter;
use crate::pipeline::{
    process_chunk, ChunkError, JobState, JobStatus, RejectRule, MAX_CHUNK_ERRORS,
};
use crate::prompt::PROMPT_VERSION;
use crate::writer::GraphWriter;

/// Sidecar file names under a collection's extraction dir.
const LLM_CONFIG_FILE: &str = "llm_config.json";
const ONTOLOGY_FILE: &str = "ontology.json";
const PROPOSALS_FILE: &str = "proposals.json";
const REJECT_FILE: &str = "reject.json";
const CHUNK_HASHES_FILE: &str = "chunk_hashes.json";
/// Subdir under a collection's extraction dir holding one progress file per
/// corpus re-extraction job, named `<corpus_job_id>.json`. The committed
/// progress in these files is what makes a corpus job resumable across restarts.
const CORPUS_JOBS_DIR: &str = "corpus_jobs";
/// How long the corpus orchestrator sleeps between polls of a sub-job's status
/// while waiting for it to finish. Cancellation is honored between polls.
const CORPUS_POLL_INTERVAL_MS: u64 = 200;

/// On-disk shape of ontology.json: the chosen template plus the extension and
/// whether the extension replaces the template entirely.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
struct OntologyFile {
    template_name: Option<String>,
    extension: Ontology,
    replace: bool,
}

/// One chunk of work, tagged with the job it belongs to. The scheduling unit is
/// a single chunk (ADR-016 fairness), not a whole batch.
struct ChunkWork {
    job_id: String,
    chunk: ChunkContent,
}

/// RAII guard that tracks one active worker for the duration of a chunk's
/// processing. Increments the active-workers gauge on creation and decrements
/// on drop, so any early return (cancel, error) still releases the count.
struct ActiveWorkerGuard {
    counter: Arc<AtomicUsize>,
}

impl ActiveWorkerGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
        gauge!("swarndb_extraction_active_workers").set(n as f64);
        Self { counter }
    }
}

impl Drop for ActiveWorkerGuard {
    fn drop(&mut self) {
        let n = self.counter.fetch_sub(1, Ordering::Relaxed).saturating_sub(1);
        gauge!("swarndb_extraction_active_workers").set(n as f64);
    }
}

/// A registered job: its mutable status, a cooperative cancel flag, and its own
/// queue of not-yet-dispatched chunks.
///
/// A job can span several batches (ADR-013). `sealed` flips true once the client
/// sends the final batch (a single-shot StartExtraction job starts sealed). With
/// per-chunk fair dispatch (ADR-016) the finalization unit is no longer the
/// batch but the single `outstanding` counter: it is the total number of this
/// job's chunks that are queued-or-in-flight, incremented when a batch is
/// scheduled and decremented when a chunk finishes, is drained on cancel, or is
/// an empty batch with nothing to do. The job finalizes exactly when
/// `outstanding` reaches zero AND it is sealed, which is robust regardless of how
/// chunks interleave across batches and workers.
///
/// `queue` (ADR-016) holds the job's own pending chunks. The fair scheduler pulls
/// one chunk from it per turn, so a large job's chunks interleave with every other
/// active job's chunks rather than running to completion first.
struct JobEntry {
    status: std::sync::Mutex<JobStatus>,
    cancel: AtomicBool,
    /// Set once the final batch has been accepted (single-shot jobs start sealed).
    sealed: AtomicBool,
    /// This job's own queue of chunks awaiting dispatch (ADR-016 fairness).
    queue: std::sync::Mutex<VecDeque<ChunkContent>>,
    /// Total chunks queued-or-in-flight; the single finalization counter. Includes
    /// a +1 "open" token while a StartExtraction job is unsealed-with-no-chunks so
    /// it cannot finalize before its first append; see `JobEntry::new`.
    outstanding: AtomicUsize,
}

impl JobEntry {
    /// A single-batch job starts sealed. `outstanding` starts at zero; the chunks
    /// of the first batch bump it the moment they are scheduled, before the job id
    /// is returned, so a worker can never observe outstanding==0-and-sealed before
    /// the batch is counted. A later AppendExtractionChunks (ADR-013) re-opens it
    /// by clearing the seal and adding more outstanding chunks.
    fn new(status: JobStatus) -> Self {
        Self {
            status: std::sync::Mutex::new(status),
            cancel: AtomicBool::new(false),
            sealed: AtomicBool::new(true),
            queue: std::sync::Mutex::new(VecDeque::new()),
            outstanding: AtomicUsize::new(0),
        }
    }
}

/// The round-robin scheduler state shared across workers. A plain std mutex is
/// correct because it is held only for the O(1) ready-queue rotation, never
/// across an await.
///
/// `order` is the rotation of job ids that currently have pending chunks. A
/// worker pops the front, takes one chunk from that job, and pushes the job id
/// back if it still has chunks. `present` mirrors the set of ids in `order` so an
/// append never double-enqueues a job that is already in the rotation.
struct Scheduler {
    order: VecDeque<String>,
    present: std::collections::HashSet<String>,
}

impl Scheduler {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
            present: std::collections::HashSet::new(),
        }
    }

    /// Add a job id to the rotation if it is not already present.
    fn enqueue(&mut self, job_id: &str) {
        if self.present.insert(job_id.to_string()) {
            self.order.push_back(job_id.to_string());
        }
    }

    /// Pop the next job id to service, removing it from the present set. The
    /// caller re-enqueues it if the job still has pending chunks after taking one.
    fn next(&mut self) -> Option<String> {
        let id = self.order.pop_front()?;
        self.present.remove(&id);
        Some(id)
    }
}

/// A registered corpus re-extraction job (Area 3): its master status and a
/// cooperative cancel flag honored between documents. The orchestration runs in
/// a background task and updates `status` after each document; the status is
/// also persisted to disk so the job is resumable across a restart.
struct CorpusJobEntry {
    status: std::sync::Mutex<CorpusJobStatus>,
    cancel: AtomicBool,
}

/// All runtime state for one Hybrid collection.
struct CollectionRuntime {
    /// The unsealed LLM config, if one has been set.
    llm_config: std::sync::Mutex<Option<LlmConfig>>,
    /// The merged ontology view (template + extension + approved proposals).
    ontology_merged: std::sync::Mutex<Ontology>,
    /// The chosen base template name, if any.
    template_name: std::sync::Mutex<Option<String>>,
    /// The user extension and whether it replaces the template.
    extension: std::sync::Mutex<Ontology>,
    replace: std::sync::Mutex<bool>,
    /// Stored proposals awaiting review.
    proposals: std::sync::Mutex<Vec<OntologyProposal>>,
    /// The reject list of edge patterns to never re-create.
    reject_list: std::sync::Mutex<Vec<RejectRule>>,
    /// Per-document chunk content hashes: doc_id -> chunk_id -> content hash.
    /// Drives the document-update diff and incremental re-extraction.
    chunk_hashes: std::sync::Mutex<HashMap<String, HashMap<u64, String>>>,
    /// The write seam into this collection's typed graph store.
    graph_writer: Arc<dyn GraphWriter>,
    /// The collection's extraction directory on disk.
    dir: PathBuf,
}

impl CollectionRuntime {
    /// Recompute and store the merged ontology from the current template,
    /// extension, replace flag, and approved proposals.
    fn recompute_merged(&self) {
        let template_name = self.template_name.lock().ok().and_then(|g| g.clone());
        let extension = self
            .extension
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        let replace = self.replace.lock().map(|g| *g).unwrap_or(false);
        let approved: Vec<OntologyProposal> = self
            .proposals
            .lock()
            .map(|g| {
                g.iter()
                    .filter(|p| p.status == ProposalStatus::Approved)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        let base = if replace {
            Ontology::default()
        } else {
            template_name
                .as_deref()
                .and_then(template_by_name)
                .unwrap_or_default()
        };
        let merged = Ontology::merge(&base, &extension, &approved);
        if let Ok(mut slot) = self.ontology_merged.lock() {
            *slot = merged;
        }
    }
}

/// The extraction manager, shared as an `Arc` across the server.
pub struct ExtractionManager {
    master_key: Option<MasterKey>,
    pricing: Arc<PricingTable>,
    estimator: Arc<dyn TokenEstimator>,
    concurrency: usize,
    cache_max: usize,
    /// Fair round-robin scheduler over jobs that have pending chunks (ADR-016).
    scheduler: std::sync::Mutex<Scheduler>,
    /// Wakes idle workers when new chunks are scheduled or shutdown is requested.
    notify: Arc<Notify>,
    jobs: std::sync::Mutex<HashMap<String, Arc<JobEntry>>>,
    /// Master entries for in-flight corpus re-extraction jobs (Area 3), keyed by
    /// corpus job id. The per-document sub-jobs live in `jobs` like any other.
    corpus_jobs: std::sync::Mutex<HashMap<String, Arc<CorpusJobEntry>>>,
    collections: std::sync::Mutex<HashMap<String, Arc<CollectionRuntime>>>,
    /// The single global extraction cache, shared by every collection and keyed
    /// only by (chunk hash, model, prompt version) plus an optional custom-prompt
    /// digest, so a dropped-and-recreated collection re-uses prior results. Opened
    /// once from the first collection's derived global dir. Locked only for the
    /// synchronous get/put around the LLM await (never across it).
    global_cache: std::sync::OnceLock<Arc<std::sync::Mutex<ExtractionCache>>>,
    /// Pending chunks across all job queues (scheduled but not yet pulled).
    queue_depth: AtomicUsize,
    /// Workers currently processing a chunk.
    active_workers: Arc<AtomicUsize>,
}

impl ExtractionManager {
    /// Build a shared manager. `concurrency` and `cache_max` come from server
    /// config; the estimator defaults to the heuristic estimator.
    pub fn new(
        master_key: Option<MasterKey>,
        concurrency: usize,
        cache_max: usize,
        pricing: Arc<PricingTable>,
    ) -> Arc<Self> {
        let estimator: Arc<dyn TokenEstimator> = Arc::new(HeuristicEstimator::default());
        Arc::new(Self {
            master_key,
            pricing,
            estimator,
            concurrency: concurrency.max(1),
            cache_max: cache_max.max(1),
            scheduler: std::sync::Mutex::new(Scheduler::new()),
            notify: Arc::new(Notify::new()),
            jobs: std::sync::Mutex::new(HashMap::new()),
            corpus_jobs: std::sync::Mutex::new(HashMap::new()),
            collections: std::sync::Mutex::new(HashMap::new()),
            global_cache: std::sync::OnceLock::new(),
            queue_depth: AtomicUsize::new(0),
            active_workers: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Open (once) and return the shared global cache, deriving its directory
    /// from a collection's extraction dir. Every collection shares the same data
    /// root, so the derived dir is identical no matter which collection opens it
    /// first; `OnceLock` makes the open happen exactly once under races. If the
    /// open fails the error is logged and `None` is returned, so a cache outage
    /// degrades to all-misses rather than failing collection registration.
    fn open_global_cache(
        &self,
        extraction_dir: &Path,
    ) -> Option<Arc<std::sync::Mutex<ExtractionCache>>> {
        if let Some(cache) = self.global_cache.get() {
            return Some(Arc::clone(cache));
        }
        let dir = global_cache_dir(extraction_dir);
        match ExtractionCache::open(&dir, self.cache_max) {
            Ok(cache) => {
                let cache = Arc::new(std::sync::Mutex::new(cache));
                // If another thread won the race, keep that instance.
                let _ = self.global_cache.set(Arc::clone(&cache));
                Some(Arc::clone(self.global_cache.get().unwrap_or(&cache)))
            }
            Err(e) => {
                tracing::warn!(error = %e, "global extraction cache open failed; proceeding without cache");
                None
            }
        }
    }

    /// The shared global cache handle if it has been opened.
    fn cache_handle(&self) -> Option<Arc<std::sync::Mutex<ExtractionCache>>> {
        self.global_cache.get().map(Arc::clone)
    }

    // ── Worker pool ──────────────────────────────────────────────────

    /// Spawn the worker pool. Each of `concurrency` workers loops pulling one
    /// chunk at a time from the fair scheduler, parking on the notify when the
    /// scheduler is empty. The watch channel signals graceful shutdown. Returns a
    /// handle the server awaits in its shutdown path.
    pub fn spawn_workers(
        self: &Arc<Self>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        let mut workers = Vec::with_capacity(self.concurrency);
        for _ in 0..self.concurrency {
            let manager = Arc::clone(self);
            let notify = Arc::clone(&self.notify);
            let mut shutdown = shutdown_rx.clone();
            workers.push(tokio::spawn(async move {
                loop {
                    if *shutdown.borrow() {
                        break;
                    }
                    // Drain ready chunks. `take_next_chunk` pops one chunk from
                    // the fairest job; when nothing is ready we park until a new
                    // chunk is scheduled or shutdown fires.
                    match manager.take_next_chunk() {
                        Some(work) => manager.process_work(work).await,
                        None => {
                            tokio::select! {
                                biased;
                                _ = shutdown.changed() => continue,
                                _ = notify.notified() => continue,
                            }
                        }
                    }
                }
            }));
        }

        tokio::spawn(async move {
            for w in workers {
                let _ = w.await;
            }
        })
    }

    /// Pop the next chunk to process, round-robin across jobs with work (ADR-016).
    /// Honors cancellation BEFORE dispatch: a cancelled job's queued chunks are
    /// drained and skipped here so they never reach a worker, decrementing its
    /// `outstanding` so the job finalizes to Cancelled. Returns `None` when no job
    /// currently has a dispatchable chunk.
    fn take_next_chunk(&self) -> Option<ChunkWork> {
        loop {
            // Pop a candidate job id under the scheduler lock only.
            let job_id = {
                let mut sched = self.scheduler.lock().ok()?;
                sched.next()?
            };

            let entry = match self.job_entry(&job_id) {
                Some(e) => e,
                // Job vanished (collection dropped + job removed): just skip it.
                None => continue,
            };

            // Honored cancellation BEFORE dispatch: drain and discard this job's
            // queued chunks so a cancelled or dropped job stops promptly instead
            // of running every chunk it had queued. Each drained chunk decrements
            // outstanding so the job can finalize to Cancelled.
            if entry.cancel.load(Ordering::Relaxed) {
                let drained = {
                    let mut q = match entry.queue.lock() {
                        Ok(q) => q,
                        Err(_) => continue,
                    };
                    let n = q.len();
                    q.clear();
                    n
                };
                if drained > 0 {
                    let depth = self
                        .queue_depth
                        .fetch_sub(drained, Ordering::Relaxed)
                        .saturating_sub(drained);
                    gauge!("swarndb_extraction_queue_depth").set(depth as f64);
                    self.release_outstanding(&entry, drained);
                }
                // Do NOT re-enqueue: the job has no dispatchable chunks now.
                continue;
            }

            // Take exactly one chunk from this job's queue.
            let next = {
                let mut q = match entry.queue.lock() {
                    Ok(q) => q,
                    Err(_) => continue,
                };
                q.pop_front()
            };

            match next {
                Some(chunk) => {
                    // Re-enqueue the job if it still has queued chunks, so the
                    // rotation keeps every active job moving. The chunk stays
                    // counted in `outstanding` until `process_work` finishes it.
                    let has_more = entry
                        .queue
                        .lock()
                        .map(|q| !q.is_empty())
                        .unwrap_or(false);
                    if has_more {
                        if let Ok(mut sched) = self.scheduler.lock() {
                            sched.enqueue(&job_id);
                        }
                    }
                    let depth = self
                        .queue_depth
                        .fetch_sub(1, Ordering::Relaxed)
                        .saturating_sub(1);
                    gauge!("swarndb_extraction_queue_depth").set(depth as f64);
                    return Some(ChunkWork { job_id, chunk });
                }
                // Empty queue (already drained by a sibling worker): try the next
                // job id without re-enqueuing this one.
                None => continue,
            }
        }
    }

    /// Process one scheduled chunk: honor cancellation, run the LLM + write path,
    /// record progress or a per-chunk error, then release its outstanding count
    /// and finalize the job if this was its last outstanding work.
    async fn process_work(self: &Arc<Self>, work: ChunkWork) {
        let entry = match self.job_entry(&work.job_id) {
            Some(e) => e,
            None => return,
        };

        // Move the job out of Queued the moment a worker picks up its first chunk.
        // Idempotent: re-setting Running over Running is a no-op, and a job already
        // driven to a terminal state by a sibling is left alone below.
        self.mark_running(&entry);

        // Honored cancellation BEFORE processing: a cancel that landed after this
        // chunk was dispatched still skips the work.
        if entry.cancel.load(Ordering::Relaxed) {
            self.finish_chunk(&entry);
            return;
        }

        let collection = {
            entry
                .status
                .lock()
                .map(|s| s.collection.clone())
                .unwrap_or_default()
        };
        let runtime = match self.runtime(&collection) {
            Ok(r) => r,
            Err(e) => {
                // The collection vanished under us (dropped mid-flight): record a
                // per-chunk error and finish this unit of work.
                self.record_chunk_error(
                    &entry,
                    &work.chunk.doc_id,
                    work.chunk.chunk_id,
                    e.to_string(),
                );
                self.finish_chunk(&entry);
                return;
            }
        };

        // Build the adapter from the current LLM config. A build failure here is
        // isolated to this chunk (it was the per-batch fatal in the old model, but
        // with per-chunk dispatch we degrade to a recorded per-chunk error so one
        // transient config read cannot silently strand the rest of the job).
        let adapter = match self.build_adapter(&runtime) {
            Ok(a) => a,
            Err(e) => {
                self.record_chunk_error(
                    &entry,
                    &work.chunk.doc_id,
                    work.chunk.chunk_id,
                    e.to_string(),
                );
                self.finish_chunk(&entry);
                return;
            }
        };
        let model = adapter.model_id().to_string();
        let chunk = &work.chunk;

        // Mark this worker active for the chunk; the guard releases the count on
        // any exit path (success, error, early return).
        let _active = ActiveWorkerGuard::new(Arc::clone(&self.active_workers));

        // Snapshot the immutable-ish runtime bits under short std locks so no std
        // guard is held across the LLM await.
        let ontology = runtime
            .ontology_merged
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        let reject_list = runtime
            .reject_list
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        let writer = Arc::clone(&runtime.graph_writer);

        // Resolve from the shared global cache or a fresh LLM call. Fold the
        // per-collection custom prompt into the key (None for the default prompt
        // keeps the historical key, see cache::cache_key). The key carries no
        // collection id, so a dropped-and-recreated collection re-uses results.
        let custom_prompt = custom_prompt_hash(
            ontology.system_prompt.as_deref(),
            ontology.extra_guidance.as_deref(),
            ontology.link_passages,
        );
        let key = cache_key(
            &normalize_text(&chunk.text),
            &model,
            PROMPT_VERSION,
            custom_prompt.as_deref(),
        );
        let cache_arc = self.cache_handle();
        let hit: Option<ExtractionResult> = match &cache_arc {
            Some(c) => match c.lock() {
                Ok(mut cache) => cache.get(&key),
                Err(_) => None,
            },
            None => None,
        };
        let (result, cache_hit) = match hit {
            Some(result) => (result, true),
            None => {
                // A single chunk's LLM failure must not kill the whole job: record
                // it as a per-chunk error and finish this unit of work.
                let fresh = match adapter.extract(chunk, &ontology, PROMPT_VERSION).await {
                    Ok(r) => r,
                    Err(e) => {
                        self.record_chunk_error(
                            &entry,
                            &chunk.doc_id,
                            chunk.chunk_id,
                            e.to_string(),
                        );
                        self.finish_chunk(&entry);
                        return;
                    }
                };
                // A cache-write failure must not discard an extraction that already
                // succeeded: a stale cache is far cheaper than losing good work, so
                // warn and proceed with the fresh result.
                if let Some(c) = &cache_arc {
                    if let Ok(mut cache) = c.lock() {
                        if let Err(e) = cache.put(&key, &fresh) {
                            tracing::warn!(
                                doc = %chunk.doc_id,
                                chunk = chunk.chunk_id,
                                error = %e,
                                "extraction cache put failed; proceeding with fresh result"
                            );
                        }
                    }
                }
                (fresh, false)
            }
        };

        // Area 4: accumulate the ACTUAL token usage and cost for this chunk.
        // A cache hit cost zero tokens, so only a fresh LLM call contributes.
        if !cache_hit {
            self.record_actual_usage(
                &entry,
                &model,
                result.input_tokens as u64,
                result.output_tokens as u64,
                result.usage_reported,
            );
        }

        // Synchronous: ontology validation, re-extraction, reject filtering,
        // dedup, and graph writes. No lock and no await inside.
        let outcome = process_chunk(
            &result,
            cache_hit,
            &ontology,
            writer.as_ref(),
            &reject_list,
            &model,
            PROMPT_VERSION,
            chunk,
        );

        match outcome {
            Ok(out) => {
                counter!("swarndb_extraction_chunks_processed_total").increment(1);
                counter!("swarndb_extraction_edges_written_total")
                    .increment(out.edges_written as u64);
                if !out.new_proposals.is_empty() {
                    self.record_proposals(&runtime, out.new_proposals);
                }
                self.record_progress(
                    &entry,
                    out.entities_written,
                    out.edges_written,
                    out.cache_hit,
                );
                // Record the chunk content hash only on successful extraction, so a
                // failed or never-run chunk stays New/Changed next time. The hash
                // map is updated in memory per chunk but flushed to disk only once,
                // when the job finalizes (see `release_outstanding`), so a large
                // job does not pay a disk write per chunk.
                let hash = chunk_content_hash(&chunk.text);
                if let Ok(mut map) = runtime.chunk_hashes.lock() {
                    map.entry(chunk.doc_id.clone())
                        .or_default()
                        .insert(chunk.chunk_id, hash);
                }
            }
            Err(e) => {
                // Isolate the failure to this chunk.
                self.record_chunk_error(&entry, &chunk.doc_id, chunk.chunk_id, e.to_string());
            }
        }

        self.finish_chunk(&entry);
    }

    /// One unit of work finished (success, skip, or error): release its one
    /// outstanding count and finalize the job if it was the last.
    fn finish_chunk(&self, entry: &Arc<JobEntry>) {
        self.release_outstanding(entry, 1);
    }

    /// Release `n` outstanding chunks (finished, drained, or empty-batch) and, if
    /// the job now has zero outstanding chunks AND is sealed, drive it to its
    /// terminal state. Safe to call with `n == 0` to evaluate an already-zero job
    /// (used for empty batches).
    ///
    /// Terminal-state selection mirrors ADR-008 partial-success:
    ///   - cancel in effect             -> Cancelled;
    ///   - no failures                  -> Completed;
    ///   - failures but no success      -> Failed;
    ///   - a mix of both                -> CompletedWithErrors.
    fn release_outstanding(&self, entry: &Arc<JobEntry>, n: usize) {
        let remaining = if n == 0 {
            entry.outstanding.load(Ordering::SeqCst)
        } else {
            entry
                .outstanding
                .fetch_sub(n, Ordering::SeqCst)
                .saturating_sub(n)
        };
        // Not done while any chunk is still outstanding, or while more batches may
        // still arrive (unsealed).
        if remaining != 0 || !entry.sealed.load(Ordering::SeqCst) {
            return;
        }

        let cancelled = entry.cancel.load(Ordering::Relaxed);

        // Pick the terminal state under the status lock, and set it in the same
        // critical section so two threads racing to finalize cannot both write a
        // terminal state: the first sets it, the second sees the terminal state
        // and bails. Capture the collection only when THIS call did the
        // finalizing, so the post-lock hash flush runs exactly once per job.
        let collection = match entry.status.lock() {
            Ok(mut s) => {
                if matches!(
                    s.state,
                    JobState::Completed
                        | JobState::CompletedWithErrors
                        | JobState::Failed
                        | JobState::Cancelled
                ) {
                    return;
                }
                if cancelled {
                    s.state = JobState::Cancelled;
                } else if s.failed_chunks == 0 {
                    s.state = JobState::Completed;
                } else if s.processed_chunks == 0 {
                    let detail = s
                        .chunk_errors
                        .first()
                        .map(|c| c.error.clone())
                        .unwrap_or_else(|| "unknown error".to_string());
                    s.state = JobState::Failed;
                    s.error = Some(format!(
                        "all {} chunks failed; first error: {}",
                        s.failed_chunks, detail
                    ));
                } else {
                    s.state = JobState::CompletedWithErrors;
                }
                s.collection.clone()
            }
            Err(_) => return,
        };

        // Flush the chunk-content hashes once, now that the job is terminal, so a
        // large job pays one disk write at the end rather than one per chunk. Best-
        // effort: the collection may have been dropped, in which case there is
        // nothing to persist and the lookup simply fails.
        if let Ok(runtime) = self.runtime(&collection) {
            let _ = self.persist_chunk_hashes(&runtime);
        }
    }

    /// Build the per-collection adapter from its unsealed config.
    fn build_adapter(
        &self,
        runtime: &CollectionRuntime,
    ) -> Result<Box<dyn ExtractionAdapter>, ExtractionError> {
        let config = runtime
            .llm_config
            .lock()
            .ok()
            .and_then(|g| g.clone())
            .ok_or_else(|| ExtractionError::Config("no llm config set".to_string()))?;
        let adapter = OpenAICompatibleAdapter::new(
            &config,
            Arc::clone(&self.estimator),
            Arc::clone(&self.pricing),
        )?;
        Ok(Box::new(adapter))
    }

    // ── Job-status helpers (short std-lock scopes only) ──────────────

    fn job_entry(&self, job_id: &str) -> Option<Arc<JobEntry>> {
        self.jobs.lock().ok().and_then(|g| g.get(job_id).cloned())
    }

    /// Move a job from Queued to Running on first chunk pickup, leaving any
    /// terminal or already-Running state untouched.
    fn mark_running(&self, entry: &JobEntry) {
        if let Ok(mut s) = entry.status.lock() {
            if s.state == JobState::Queued {
                s.state = JobState::Running;
            }
        }
    }

    fn record_progress(
        &self,
        entry: &JobEntry,
        entities: usize,
        edges: usize,
        cache_hit: bool,
    ) {
        if let Ok(mut s) = entry.status.lock() {
            s.processed_chunks += 1;
            s.entities_written += entities;
            s.edges_written += edges;
            if cache_hit {
                s.cache_hits += 1;
            } else {
                s.cache_misses += 1;
            }
        }
    }

    /// Accumulate one fresh LLM call's ACTUAL token usage and cost onto the job
    /// status (Area 4). Prices the incremental tokens through the pricing table;
    /// an unknown model adds zero cost (never a fabricated number). The
    /// `usage_provider_reported` flag stays true only while every call so far
    /// reported its own usage; one estimated call flips it false for good.
    fn record_actual_usage(
        &self,
        entry: &JobEntry,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        usage_reported: bool,
    ) {
        let cost = match self.pricing.price(model) {
            Some(p) => {
                (input_tokens as f64 / 1000.0) * p.input_usd_per_1k
                    + (output_tokens as f64 / 1000.0) * p.output_usd_per_1k
            }
            None => 0.0,
        };
        if let Ok(mut s) = entry.status.lock() {
            s.actual_input_tokens = s.actual_input_tokens.saturating_add(input_tokens);
            s.actual_output_tokens = s.actual_output_tokens.saturating_add(output_tokens);
            s.actual_cost_usd += cost;
            if !usage_reported {
                s.usage_provider_reported = false;
            }
        }
    }

    /// Record a single chunk's failure: always bump the `failed_chunks` count,
    /// and append a sample to `chunk_errors` only while it is below the cap so a
    /// flood of failures cannot grow the job status without bound.
    fn record_chunk_error(&self, entry: &JobEntry, doc_id: &str, chunk_id: u64, error: String) {
        if let Ok(mut s) = entry.status.lock() {
            s.failed_chunks += 1;
            if s.chunk_errors.len() < MAX_CHUNK_ERRORS {
                s.chunk_errors.push(ChunkError {
                    doc_id: doc_id.to_string(),
                    chunk_id,
                    error,
                });
            }
        }
    }

    fn record_proposals(&self, runtime: &CollectionRuntime, fresh: Vec<OntologyProposal>) {
        if let Ok(mut store) = runtime.proposals.lock() {
            for p in fresh {
                if !store.iter().any(|existing| existing.id == p.id) {
                    store.push(p);
                }
            }
        }
        // Persist best-effort; a failed write does not abort the job.
        let _ = self.persist_proposals(runtime);
    }

    // ── Scheduling helpers ────────────────────────────────────────────

    /// Push a batch of chunks onto a job's own queue and register the job in the
    /// fair rotation, then wake up to `chunks.len()` parked workers (ADR-016).
    /// `outstanding` is bumped by the batch size BEFORE the job enters the
    /// scheduler, so a worker can never observe outstanding==0-and-sealed and
    /// finalize the job before this batch is counted.
    fn schedule_chunks(&self, job_id: &str, entry: &Arc<JobEntry>, chunks: Vec<ChunkContent>) {
        if chunks.is_empty() {
            return;
        }
        let n = chunks.len();
        entry.outstanding.fetch_add(n, Ordering::SeqCst);
        if let Ok(mut q) = entry.queue.lock() {
            q.extend(chunks);
        }
        if let Ok(mut sched) = self.scheduler.lock() {
            sched.enqueue(job_id);
        }
        let depth = self.queue_depth.fetch_add(n, Ordering::Relaxed) + n;
        gauge!("swarndb_extraction_queue_depth").set(depth as f64);
        // Wake one parked worker per scheduled chunk, bounded by the pool size, so
        // a fresh job's chunks get a worker promptly without a thundering herd.
        for _ in 0..n.min(self.concurrency) {
            self.notify.notify_one();
        }
    }

    // ── Collection registration and loading ──────────────────────────

    /// Register a collection's runtime from its config. Creates the extraction
    /// dir, opens the shared global cache, and seeds the ontology from the
    /// config's template.
    pub fn register_collection(
        self: &Arc<Self>,
        coll: &str,
        graph_writer: Arc<dyn GraphWriter>,
        dir: PathBuf,
        config: &CollectionConfig,
    ) -> Result<(), ExtractionError> {
        if config.effective_mode() != Mode::Hybrid {
            return Err(ExtractionError::Config(
                "extraction is only available for hybrid collections".to_string(),
            ));
        }
        std::fs::create_dir_all(&dir).map_err(|e| ExtractionError::Io(e.to_string()))?;
        // Open the single global cache (once) from this collection's dir.
        self.open_global_cache(&dir);

        let runtime = Arc::new(CollectionRuntime {
            llm_config: std::sync::Mutex::new(None),
            ontology_merged: std::sync::Mutex::new(Ontology::default()),
            template_name: std::sync::Mutex::new(None),
            extension: std::sync::Mutex::new(Ontology::default()),
            replace: std::sync::Mutex::new(false),
            proposals: std::sync::Mutex::new(Vec::new()),
            reject_list: std::sync::Mutex::new(Vec::new()),
            chunk_hashes: std::sync::Mutex::new(HashMap::new()),
            graph_writer,
            dir,
        });
        runtime.recompute_merged();

        self.insert_runtime(coll, runtime);
        Ok(())
    }

    /// Load a collection's runtime from its on-disk sidecars. Reads any of
    /// llm_config.json, ontology.json, proposals.json, reject.json,
    /// chunk_hashes.json that exist and opens the shared global cache.
    pub fn load_collection(
        self: &Arc<Self>,
        coll: &str,
        graph_writer: Arc<dyn GraphWriter>,
        dir: PathBuf,
    ) -> Result<(), ExtractionError> {
        std::fs::create_dir_all(&dir).map_err(|e| ExtractionError::Io(e.to_string()))?;
        // Open the single global cache (once) from this collection's dir.
        self.open_global_cache(&dir);

        // LLM config (unseal if a master key is configured).
        let llm_config = match read_json::<SealedLlmConfig>(&dir.join(LLM_CONFIG_FILE))? {
            Some(sealed) => match &self.master_key {
                Some(mk) => Some(sealed.unseal(mk)?),
                None => {
                    return Err(ExtractionError::Config(
                        "llm config on disk but no master key configured to unseal it".to_string(),
                    ))
                }
            },
            None => None,
        };

        let ontology_file =
            read_json::<OntologyFile>(&dir.join(ONTOLOGY_FILE))?.unwrap_or_default();
        let proposals =
            read_json::<Vec<OntologyProposal>>(&dir.join(PROPOSALS_FILE))?.unwrap_or_default();
        let reject_list =
            read_json::<Vec<RejectRule>>(&dir.join(REJECT_FILE))?.unwrap_or_default();
        let chunk_hashes =
            read_json::<HashMap<String, HashMap<u64, String>>>(&dir.join(CHUNK_HASHES_FILE))?
                .unwrap_or_default();

        let runtime = Arc::new(CollectionRuntime {
            llm_config: std::sync::Mutex::new(llm_config),
            ontology_merged: std::sync::Mutex::new(Ontology::default()),
            template_name: std::sync::Mutex::new(ontology_file.template_name),
            extension: std::sync::Mutex::new(ontology_file.extension),
            replace: std::sync::Mutex::new(ontology_file.replace),
            proposals: std::sync::Mutex::new(proposals),
            reject_list: std::sync::Mutex::new(reject_list),
            chunk_hashes: std::sync::Mutex::new(chunk_hashes),
            graph_writer,
            dir,
        });
        runtime.recompute_merged();

        self.insert_runtime(coll, runtime);
        Ok(())
    }

    fn insert_runtime(&self, coll: &str, runtime: Arc<CollectionRuntime>) {
        if let Ok(mut map) = self.collections.lock() {
            map.insert(coll.to_string(), runtime);
        }
    }

    fn runtime(&self, coll: &str) -> Result<Arc<CollectionRuntime>, ExtractionError> {
        self.collections
            .lock()
            .ok()
            .and_then(|g| g.get(coll).cloned())
            .ok_or_else(|| {
                ExtractionError::Config(format!("collection {} is not extraction-enabled", coll))
            })
    }

    /// Cancel all in-flight extraction jobs for a collection and forget its
    /// runtime (ADR-016 collection-drop hook). Called from the collection-delete
    /// path so a dropped collection never leaves a runaway extraction job.
    ///
    /// Each matching job's cancel flag is set, so `take_next_chunk` drains its
    /// queued chunks and `process_work` skips any already-dispatched chunk; the
    /// job then finalizes to Cancelled. The runtime is removed last so the cancel
    /// is visible to workers before the collection lookup can start failing.
    pub fn cancel_collection_jobs(&self, coll: &str) {
        // Snapshot the matching job ids under a short lock, then set their cancel
        // flags. Cancellation is cooperative, so it is safe even for jobs already
        // mid-flight.
        let matching: Vec<Arc<JobEntry>> = match self.jobs.lock() {
            Ok(jobs) => jobs
                .values()
                .filter(|e| {
                    e.status
                        .lock()
                        .map(|s| s.collection == coll)
                        .unwrap_or(false)
                })
                .cloned()
                .collect(),
            Err(_) => Vec::new(),
        };
        for entry in &matching {
            entry.cancel.store(true, Ordering::Relaxed);
        }
        // Also cancel any corpus re-extraction jobs over this collection so the
        // orchestration stops promptly rather than failing document by document.
        let corpus_matching: Vec<Arc<CorpusJobEntry>> = match self.corpus_jobs.lock() {
            Ok(jobs) => jobs
                .values()
                .filter(|e| {
                    e.status
                        .lock()
                        .map(|s| s.collection == coll)
                        .unwrap_or(false)
                })
                .cloned()
                .collect(),
            Err(_) => Vec::new(),
        };
        for entry in &corpus_matching {
            entry.cancel.store(true, Ordering::Relaxed);
        }
        // Wake parked workers so the cancel is acted on without waiting for new
        // work to arrive.
        for _ in 0..self.concurrency {
            self.notify.notify_one();
        }
        // Drop the collection runtime so no further appends register against it.
        if let Ok(mut map) = self.collections.lock() {
            map.remove(coll);
        }
    }

    // ── LLM config API ───────────────────────────────────────────────

    /// Set the LLM config, sealing the api key to disk under the master key. An
    /// api key is required; rotate_llm_config changes just the key later.
    pub fn set_llm_config(&self, coll: &str, config: LlmConfig) -> Result<(), ExtractionError> {
        let runtime = self.runtime(coll)?;
        if config.api_key.is_empty() {
            return Err(ExtractionError::Config(
                "an api key is required to set the llm config".to_string(),
            ));
        }
        let mk = self.master_key.as_ref().ok_or_else(|| {
            ExtractionError::Config(
                "an api key was supplied but no master key is configured to seal it".to_string(),
            )
        })?;
        let sealed = config.seal(mk)?;
        write_json(&runtime.dir.join(LLM_CONFIG_FILE), &sealed)?;
        if let Ok(mut slot) = runtime.llm_config.lock() {
            *slot = Some(config);
        }
        Ok(())
    }

    /// Return a redacted view of the LLM config.
    pub fn get_llm_config(&self, coll: &str) -> Result<RedactedLlmConfig, ExtractionError> {
        let runtime = self.runtime(coll)?;
        let guard = runtime
            .llm_config
            .lock()
            .map_err(|_| ExtractionError::Config("llm config lock poisoned".to_string()))?;
        guard
            .as_ref()
            .map(|c| c.redacted())
            .ok_or_else(|| ExtractionError::Config("no llm config set".to_string()))
    }

    /// Rotate just the api key, re-sealing the existing config to disk.
    pub fn rotate_llm_config(
        &self,
        coll: &str,
        new_api_key: &str,
    ) -> Result<(), ExtractionError> {
        let runtime = self.runtime(coll)?;
        let mk = self.master_key.as_ref().ok_or_else(|| {
            ExtractionError::Config("no master key configured to seal the rotated key".to_string())
        })?;
        let mut updated = {
            let guard = runtime
                .llm_config
                .lock()
                .map_err(|_| ExtractionError::Config("llm config lock poisoned".to_string()))?;
            guard
                .clone()
                .ok_or_else(|| ExtractionError::Config("no llm config to rotate".to_string()))?
        };
        updated.api_key = zeroize::Zeroizing::new(new_api_key.to_string());
        let sealed = updated.seal(mk)?;
        write_json(&runtime.dir.join(LLM_CONFIG_FILE), &sealed)?;
        if let Ok(mut slot) = runtime.llm_config.lock() {
            *slot = Some(updated);
        }
        Ok(())
    }

    // ── Ontology API ─────────────────────────────────────────────────

    /// Set the ontology: choose a base template, supply an extension, and choose
    /// whether the extension replaces the template entirely. Persists and
    /// recomputes the merged view.
    pub fn set_ontology(
        &self,
        coll: &str,
        base_template: Option<String>,
        extension: Ontology,
        replace: bool,
    ) -> Result<(), ExtractionError> {
        let runtime = self.runtime(coll)?;
        if let Some(name) = &base_template {
            if template_by_name(name).is_none() {
                return Err(ExtractionError::Ontology(format!(
                    "unknown ontology template: {}",
                    name
                )));
            }
        }
        if let Ok(mut slot) = runtime.template_name.lock() {
            *slot = base_template.clone();
        }
        if let Ok(mut slot) = runtime.extension.lock() {
            *slot = extension.clone();
        }
        if let Ok(mut slot) = runtime.replace.lock() {
            *slot = replace;
        }
        runtime.recompute_merged();

        let file = OntologyFile {
            template_name: base_template,
            extension,
            replace,
        };
        write_json(&runtime.dir.join(ONTOLOGY_FILE), &file)?;
        Ok(())
    }

    /// Return the merged ontology for a collection.
    pub fn get_ontology(&self, coll: &str) -> Result<Ontology, ExtractionError> {
        let runtime = self.runtime(coll)?;
        runtime
            .ontology_merged
            .lock()
            .map(|g| g.clone())
            .map_err(|_| ExtractionError::Ontology("ontology lock poisoned".to_string()))
    }

    // ── Cost preview ─────────────────────────────────────────────────

    /// Estimate the cost of extracting `chunks`, subtracting chunks already in
    /// the cache so the preview reflects only the LLM calls that would run.
    pub async fn cost_preview(
        &self,
        coll: &str,
        chunks: &[ChunkContent],
    ) -> Result<CostEstimate, ExtractionError> {
        let runtime = self.runtime(coll)?;
        let adapter = self.build_adapter(&runtime)?;
        let model = adapter.model_id().to_string();

        // The custom-prompt digest must match what the worker keys on, so derive
        // it from the same merged ontology the worker reads. None for a default
        // prompt keeps the historical key (see cache::cache_key).
        let custom_prompt = {
            let ontology = runtime
                .ontology_merged
                .lock()
                .map(|g| g.clone())
                .unwrap_or_default();
            custom_prompt_hash(
                ontology.system_prompt.as_deref(),
                ontology.extra_guidance.as_deref(),
                ontology.link_passages,
            )
        };

        // Filter out cache hits so the preview prices only the misses, reading the
        // shared global cache. The std cache lock is held only across this
        // synchronous loop, never an await. With no cache open (outage), every
        // chunk is priced so the preview never under-quotes.
        let mut to_price: Vec<ChunkContent> = Vec::with_capacity(chunks.len());
        {
            let cache_arc = self.cache_handle();
            let mut cache_guard = match &cache_arc {
                Some(c) => Some(
                    c.lock()
                        .map_err(|_| ExtractionError::Config("cache lock poisoned".to_string()))?,
                ),
                None => None,
            };
            for chunk in chunks {
                let key = cache_key(
                    &normalize_text(&chunk.text),
                    &model,
                    PROMPT_VERSION,
                    custom_prompt.as_deref(),
                );
                let hit = match &mut cache_guard {
                    Some(cache) => cache.get(&key).is_some(),
                    None => false,
                };
                if !hit {
                    to_price.push(chunk.clone());
                }
            }
        }

        let mut estimate = adapter.estimate_cost(&to_price);
        // Report the full chunk count for clarity even though only misses cost.
        estimate.chunks = chunks.len();
        Ok(estimate)
    }

    // ── Extraction lifecycle ─────────────────────────────────────────

    /// Enqueue an extraction job and return its id immediately. The chunks are
    /// scheduled onto the job's own queue and serviced fairly (ADR-016).
    pub fn start_extraction(
        &self,
        coll: &str,
        chunks: Vec<ChunkContent>,
    ) -> Result<String, ExtractionError> {
        let runtime = self.runtime(coll)?;
        // Require a config up front so a misconfigured collection fails fast.
        if runtime
            .llm_config
            .lock()
            .map(|g| g.is_none())
            .unwrap_or(true)
        {
            return Err(ExtractionError::Config("no llm config set".to_string()));
        }

        let job_id = uuid::Uuid::new_v4().to_string();
        let status = JobStatus::queued(job_id.clone(), coll.to_string(), chunks.len());
        // StartExtraction creates a sealed single-batch job so a legacy single-
        // shot client finalizes exactly as before. A later AppendExtractionChunks
        // re-opens the job (ADR-013); the residual race (batch one finalizes
        // before the first append arrives) is the documented trade-off, made
        // negligible by the LLM batch taking far longer than enqueue latency.
        let entry = Arc::new(JobEntry::new(status));
        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.insert(job_id.clone(), Arc::clone(&entry));
        }

        // An empty StartExtraction is sealed with zero outstanding chunks; no
        // worker will ever touch it, so finalize it inline (Completed) rather than
        // leaving it Queued forever.
        if chunks.is_empty() {
            self.release_outstanding(&entry, 0);
        } else {
            self.schedule_chunks(&job_id, &entry, chunks);
        }

        Ok(job_id)
    }

    /// Append a batch of chunks to an existing job (ADR-013). Validates the job
    /// exists and belongs to `coll`, accumulates `total_chunks`, re-opens the job
    /// so it does not finalize until the final batch is processed, and schedules
    /// the batch onto the job's own queue. `last_batch=true` seals the job so its
    /// terminal accounting becomes final once drained. Returns the running
    /// `total_chunks`.
    pub fn append_extraction(
        &self,
        coll: &str,
        job_id: &str,
        chunks: Vec<ChunkContent>,
        last_batch: bool,
    ) -> Result<u64, ExtractionError> {
        let entry = self
            .job_entry(job_id)
            .ok_or_else(|| ExtractionError::JobNotFound(job_id.to_string()))?;

        // Validate ownership and that the job is still accepting work. A job that
        // already reached a terminal state (or was cancelled) cannot be appended.
        let total_chunks = {
            let mut s = entry
                .status
                .lock()
                .map_err(|_| ExtractionError::Config("job status lock poisoned".to_string()))?;
            if s.collection != coll {
                return Err(ExtractionError::JobNotFound(job_id.to_string()));
            }
            if matches!(
                s.state,
                JobState::Completed
                    | JobState::CompletedWithErrors
                    | JobState::Failed
                    | JobState::Cancelled
            ) {
                return Err(ExtractionError::Config(format!(
                    "job {job_id} is no longer accepting chunks"
                )));
            }
            s.total_chunks += chunks.len();
            s.total_chunks as u64
        };

        // Re-open the job before touching the seal. Add one guard token to
        // `outstanding` first so that, while we adjust the seal and schedule, a
        // fast-draining worker that finishes the prior batch's last chunk cannot
        // observe outstanding==0-and-sealed and finalize past this append. The
        // guard token is released at the end of this call.
        entry.outstanding.fetch_add(1, Ordering::SeqCst);
        if last_batch {
            entry.sealed.store(true, Ordering::SeqCst);
        } else {
            entry.sealed.store(false, Ordering::SeqCst);
        }

        // Schedule the batch (bumps outstanding by chunks.len() and wakes workers).
        // An empty batch schedules nothing; a `last_batch=true` empty append simply
        // seals the job, and the guard-token release below may then finalize it.
        if !chunks.is_empty() {
            self.schedule_chunks(job_id, &entry, chunks);
        }

        // Release the guard token. If this was an empty final batch and the job
        // had already drained, this is what drives it to its terminal state.
        self.release_outstanding(&entry, 1);
        Ok(total_chunks)
    }

    /// The current status of a job.
    pub fn job_status(&self, coll: &str, job_id: &str) -> Result<JobStatus, ExtractionError> {
        let entry = self
            .job_entry(job_id)
            .ok_or_else(|| ExtractionError::JobNotFound(job_id.to_string()))?;
        let status = entry
            .status
            .lock()
            .map_err(|_| ExtractionError::Config("job status lock poisoned".to_string()))?
            .clone();
        if status.collection != coll {
            return Err(ExtractionError::JobNotFound(job_id.to_string()));
        }
        Ok(status)
    }

    /// Request cancellation of a job. The cancel is honored before each chunk is
    /// dispatched and before a worker processes it, so the job stops promptly: its
    /// queued chunks are drained and skipped and it transitions to Cancelled
    /// without running every remaining chunk (ADR-016).
    pub fn cancel_extraction(&self, coll: &str, job_id: &str) -> Result<(), ExtractionError> {
        let entry = self
            .job_entry(job_id)
            .ok_or_else(|| ExtractionError::JobNotFound(job_id.to_string()))?;
        let belongs = entry
            .status
            .lock()
            .map(|s| s.collection == coll)
            .unwrap_or(false);
        if !belongs {
            return Err(ExtractionError::JobNotFound(job_id.to_string()));
        }
        entry.cancel.store(true, Ordering::Relaxed);
        // Wake parked workers so the drain-and-skip happens promptly even if the
        // pool is otherwise idle.
        for _ in 0..self.concurrency {
            self.notify.notify_one();
        }
        Ok(())
    }

    // ── Proposals API ────────────────────────────────────────────────

    /// List the stored proposals for a collection.
    pub fn list_proposals(&self, coll: &str) -> Result<Vec<OntologyProposal>, ExtractionError> {
        let runtime = self.runtime(coll)?;
        runtime
            .proposals
            .lock()
            .map(|g| g.clone())
            .map_err(|_| ExtractionError::Config("proposals lock poisoned".to_string()))
    }

    /// Approve a proposal by id, persist, and recompute the merged ontology.
    pub fn approve_proposal(&self, coll: &str, id: &str) -> Result<(), ExtractionError> {
        self.set_proposal_status(coll, id, ProposalStatus::Approved)
    }

    /// Reject a proposal by id and persist.
    pub fn reject_proposal(&self, coll: &str, id: &str) -> Result<(), ExtractionError> {
        self.set_proposal_status(coll, id, ProposalStatus::Rejected)
    }

    fn set_proposal_status(
        &self,
        coll: &str,
        id: &str,
        status: ProposalStatus,
    ) -> Result<(), ExtractionError> {
        let runtime = self.runtime(coll)?;
        {
            let mut guard = runtime
                .proposals
                .lock()
                .map_err(|_| ExtractionError::Config("proposals lock poisoned".to_string()))?;
            let found = guard.iter_mut().find(|p| p.id == id);
            match found {
                Some(p) => p.status = status,
                None => return Err(ExtractionError::JobNotFound(id.to_string())),
            }
        }
        self.persist_proposals(&runtime)?;
        // Approval can widen the ontology, so recompute the merged view.
        runtime.recompute_merged();
        Ok(())
    }

    fn persist_proposals(&self, runtime: &CollectionRuntime) -> Result<(), ExtractionError> {
        let snapshot = runtime
            .proposals
            .lock()
            .map(|g| g.clone())
            .map_err(|_| ExtractionError::Config("proposals lock poisoned".to_string()))?;
        write_json(&runtime.dir.join(PROPOSALS_FILE), &snapshot)
    }

    /// Replace the reject list for a collection and persist it.
    pub fn set_reject_list(
        &self,
        coll: &str,
        rules: Vec<RejectRule>,
    ) -> Result<(), ExtractionError> {
        let runtime = self.runtime(coll)?;
        if let Ok(mut slot) = runtime.reject_list.lock() {
            *slot = rules.clone();
        }
        write_json(&runtime.dir.join(REJECT_FILE), &rules)
    }

    /// Append a single reject rule (no-op if an identical rule already exists)
    /// and persist the full reject list.
    pub fn add_reject_rule(&self, coll: &str, rule: RejectRule) -> Result<(), ExtractionError> {
        let runtime = self.runtime(coll)?;
        let snapshot = {
            let mut guard = runtime
                .reject_list
                .lock()
                .map_err(|_| ExtractionError::Config("reject list lock poisoned".to_string()))?;
            if !guard.iter().any(|existing| existing == &rule) {
                guard.push(rule);
            }
            guard.clone()
        };
        write_json(&runtime.dir.join(REJECT_FILE), &snapshot)
    }

    // ── Document-update diff and re-extraction ───────────────────────

    /// Classify each chunk of an updated document against the stored content
    /// hashes. Read-only: no graph mutation and no hash update happens here.
    pub fn diff_document(
        &self,
        coll: &str,
        doc_id: &str,
        chunks: &[ChunkContent],
    ) -> Result<Vec<ChunkDiff>, ExtractionError> {
        let runtime = self.runtime(coll)?;
        let stored = {
            let guard = runtime
                .chunk_hashes
                .lock()
                .map_err(|_| ExtractionError::Config("chunk hashes lock poisoned".to_string()))?;
            guard.get(doc_id).cloned().unwrap_or_default()
        };

        let mut diffs = Vec::with_capacity(chunks.len());
        let mut incoming: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(chunks.len());
        for chunk in chunks {
            incoming.insert(chunk.chunk_id);
            let hash = chunk_content_hash(&chunk.text);
            let action = match stored.get(&chunk.chunk_id) {
                None => ChunkDiffAction::New,
                Some(prev) if prev == &hash => ChunkDiffAction::Unchanged,
                Some(_) => ChunkDiffAction::Changed,
            };
            diffs.push(ChunkDiff {
                chunk_id: chunk.chunk_id,
                action,
            });
        }
        // Stored chunks absent from the incoming set were removed.
        for chunk_id in stored.keys() {
            if !incoming.contains(chunk_id) {
                diffs.push(ChunkDiff {
                    chunk_id: *chunk_id,
                    action: ChunkDiffAction::Deleted,
                });
            }
        }
        Ok(diffs)
    }

    /// Re-extract an updated document. Deleted chunks have their auto-edges (and
    /// now-orphaned content nodes) pruned and their stored hashes removed here;
    /// changed and new chunks are enqueued for extraction and the worker records
    /// their hashes only on success. Returns per-action counts and the job id.
    pub fn reextract_document(
        &self,
        coll: &str,
        doc_id: &str,
        chunks: Vec<ChunkContent>,
    ) -> Result<ReextractSummary, ExtractionError> {
        let runtime = self.runtime(coll)?;
        self.reextract_one(&runtime, coll, doc_id, chunks)
    }

    /// The shared per-document diff + prune + enqueue core, reused by both the
    /// single-document `reextract_document` and the corpus orchestrator so the
    /// P04 logic is never duplicated. Prunes deleted chunks and enqueues the
    /// changed + new chunks as one extraction sub-job, returning the summary.
    fn reextract_one(
        &self,
        runtime: &Arc<CollectionRuntime>,
        coll: &str,
        doc_id: &str,
        chunks: Vec<ChunkContent>,
    ) -> Result<ReextractSummary, ExtractionError> {
        let diffs = self.diff_document(coll, doc_id, &chunks)?;

        let mut summary = ReextractSummary::default();
        for d in &diffs {
            match d.action {
                ChunkDiffAction::Unchanged => summary.unchanged += 1,
                ChunkDiffAction::Changed => summary.changed += 1,
                ChunkDiffAction::New => summary.added += 1,
                ChunkDiffAction::Deleted => summary.deleted += 1,
            }
        }

        let writer = Arc::clone(&runtime.graph_writer);

        // Prune deleted chunks: drop their auto-edges, then any now-orphaned
        // content node, and forget their stored hashes.
        let deleted_ids: Vec<u64> = diffs
            .iter()
            .filter(|d| d.action == ChunkDiffAction::Deleted)
            .map(|d| d.chunk_id)
            .collect();

        for chunk_id in &deleted_ids {
            let prior = writer.edges_from_chunk(doc_id, *chunk_id);
            for edge in prior {
                if edge.is_manual || edge.verified {
                    continue;
                }
                let lsn = writer.next_lsn();
                if writer.delete_edge(edge.id, lsn)? {
                    summary.edges_deleted += 1;
                }
            }
            // Drop the content node only if it now has no incident edges, so a
            // surviving manual/verified edge keeps it alive.
            let node_id = NodeId(*chunk_id);
            if writer.get_node(node_id).is_some() && !writer.node_has_incident_edges(node_id) {
                let lsn = writer.next_lsn();
                if writer.delete_node(node_id, lsn)? {
                    summary.nodes_deleted += 1;
                }
            }
        }

        if !deleted_ids.is_empty() {
            if let Ok(mut map) = runtime.chunk_hashes.lock() {
                if let Some(doc_map) = map.get_mut(doc_id) {
                    for chunk_id in &deleted_ids {
                        doc_map.remove(chunk_id);
                    }
                    if doc_map.is_empty() {
                        map.remove(doc_id);
                    }
                }
            }
            // Persist after the deletions mutate the stored map.
            let _ = self.persist_chunk_hashes(runtime);
        }

        // Enqueue the changed and new chunks; the worker records their hashes on
        // success, not here.
        let to_extract: Vec<ChunkContent> = {
            let changed_new: std::collections::HashSet<u64> = diffs
                .iter()
                .filter(|d| {
                    d.action == ChunkDiffAction::Changed || d.action == ChunkDiffAction::New
                })
                .map(|d| d.chunk_id)
                .collect();
            chunks
                .into_iter()
                .filter(|c| changed_new.contains(&c.chunk_id))
                .collect()
        };

        if !to_extract.is_empty() {
            summary.job_id = self.start_extraction(coll, to_extract)?;
        }

        Ok(summary)
    }

    // ── Corpus-level re-extraction orchestration (Area 3) ─────────────

    /// Start a corpus re-extraction over a set of documents (or, when
    /// `doc_id_filter` is empty, over every supplied document). Each document is
    /// driven through the same per-document diff + re-extract path; per-document
    /// progress aggregates into a resumable master status.
    ///
    /// `documents` carries `(doc_id, chunks)` pairs: the current content the
    /// caller wants re-extracted. A document supplied with no chunks is treated
    /// as a full deletion (every prior chunk pruned), matching the single-doc
    /// path. `doc_id_filter`, when non-empty, restricts the run to those doc ids
    /// among the supplied documents.
    ///
    /// Resumability: pass the prior run's `corpus_job_id` as `resume_token` to
    /// continue it; documents the prior run already completed are skipped via the
    /// committed progress file. With no token a fresh corpus job id is minted.
    ///
    /// Returns the corpus job id immediately; the orchestration runs in the
    /// background. Cancellation is cooperative and honored between documents.
    pub fn start_corpus_reextraction(
        self: &Arc<Self>,
        coll: &str,
        documents: Vec<(String, Vec<ChunkContent>)>,
        doc_id_filter: Vec<String>,
        resume_token: Option<String>,
    ) -> Result<String, ExtractionError> {
        let runtime = self.runtime(coll)?;
        // Require a config up front so a misconfigured collection fails fast,
        // mirroring start_extraction.
        if runtime
            .llm_config
            .lock()
            .map(|g| g.is_none())
            .unwrap_or(true)
        {
            return Err(ExtractionError::Config("no llm config set".to_string()));
        }

        // Apply the optional doc-id filter. An empty filter means every supplied
        // document. Filtering here keeps the background task's work bounded to
        // exactly the targeted documents.
        let targets: Vec<(String, Vec<ChunkContent>)> = if doc_id_filter.is_empty() {
            documents
        } else {
            let wanted: std::collections::HashSet<&str> =
                doc_id_filter.iter().map(|s| s.as_str()).collect();
            documents
                .into_iter()
                .filter(|(doc_id, _)| wanted.contains(doc_id.as_str()))
                .collect()
        };

        // Resume: reuse the token as the corpus job id and load any prior
        // committed progress so already-completed documents are skipped. A fresh
        // run mints a new id and starts with empty progress.
        let corpus_job_id = resume_token
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let prior = self.load_corpus_progress(&runtime, &corpus_job_id);
        let already_done: std::collections::HashSet<String> = prior
            .as_ref()
            .map(|p| {
                p.documents
                    .iter()
                    .filter(|d| {
                        d.state == CorpusDocState::Completed || d.state == CorpusDocState::Skipped
                    })
                    .map(|d| d.doc_id.clone())
                    .collect()
            })
            .unwrap_or_default();

        let mut status =
            CorpusJobStatus::queued(corpus_job_id.clone(), coll.to_string(), targets.len());
        // Carry forward prior aggregate counters so a resumed run reports the
        // running totals, not just this continuation's slice.
        if let Some(p) = &prior {
            status.changed_chunks = p.changed_chunks;
            status.added_chunks = p.added_chunks;
            status.deleted_chunks = p.deleted_chunks;
            status.edges_deleted = p.edges_deleted;
            status.nodes_deleted = p.nodes_deleted;
            status.entities_written = p.entities_written;
            status.edges_written = p.edges_written;

            // Seed the reported per-document list and the doc-level counters
            // from the prior run so a resumed job's status is cumulative, not a
            // fresh slice. A prior doc that is re-supplied this run will be
            // re-recorded as it is processed, so it is de-duplicated here (skip
            // it) to avoid double-counting. total_documents stays the count of
            // this run's targets; carried-forward prior docs are extra context.
            let this_run: std::collections::HashSet<&str> =
                targets.iter().map(|(id, _)| id.as_str()).collect();
            for d in &p.documents {
                if this_run.contains(d.doc_id.as_str()) {
                    continue;
                }
                match d.state {
                    CorpusDocState::Failed => status.failed_documents += 1,
                    CorpusDocState::Skipped => status.skipped_documents += 1,
                    _ => {}
                }
                status.processed_documents += 1;
                status.documents.push(d.clone());
            }
        }

        let entry = Arc::new(CorpusJobEntry {
            status: std::sync::Mutex::new(status),
            cancel: AtomicBool::new(false),
        });
        if let Ok(mut map) = self.corpus_jobs.lock() {
            map.insert(corpus_job_id.clone(), Arc::clone(&entry));
        }

        // An empty target set finalizes immediately rather than spawning a task.
        if targets.is_empty() {
            self.finalize_corpus(&runtime, &entry);
            return Ok(corpus_job_id);
        }

        // Drive the documents in a background task so the call returns at once.
        let manager = Arc::clone(self);
        let coll_owned = coll.to_string();
        tokio::spawn(async move {
            manager
                .run_corpus(coll_owned, entry, targets, already_done)
                .await;
        });

        Ok(corpus_job_id)
    }

    /// The corpus orchestration body. Processes documents one at a time so peak
    /// memory stays bounded to a single document's chunks, awaiting each
    /// document's sub-job before moving on. Cancellation is honored between
    /// documents; progress is persisted after every document so the run resumes.
    async fn run_corpus(
        self: Arc<Self>,
        coll: String,
        entry: Arc<CorpusJobEntry>,
        targets: Vec<(String, Vec<ChunkContent>)>,
        already_done: std::collections::HashSet<String>,
    ) {
        // Move out of Queued the moment the orchestration starts.
        if let Ok(mut s) = entry.status.lock() {
            if s.state == JobState::Queued {
                s.state = JobState::Running;
            }
        }

        for (doc_id, chunks) in targets {
            // Cooperative cancel between documents: stop promptly without
            // touching the next document.
            if entry.cancel.load(Ordering::Relaxed) {
                break;
            }

            // Skip a document a prior run already completed (resume).
            if already_done.contains(&doc_id) {
                self.record_corpus_doc(
                    &entry,
                    CorpusDocProgress {
                        doc_id: doc_id.clone(),
                        state: CorpusDocState::Skipped,
                        job_id: String::new(),
                        changed: 0,
                        added: 0,
                        deleted: 0,
                    },
                );
                self.persist_corpus_progress(&coll, &entry);
                continue;
            }

            // The collection may have been dropped mid-run; record the document
            // as failed and continue so one drop does not strand the whole job.
            let runtime = match self.runtime(&coll) {
                Ok(r) => r,
                Err(e) => {
                    self.record_corpus_doc(
                        &entry,
                        CorpusDocProgress {
                            doc_id: doc_id.clone(),
                            state: CorpusDocState::Failed,
                            job_id: String::new(),
                            changed: 0,
                            added: 0,
                            deleted: 0,
                        },
                    );
                    if let Ok(mut s) = entry.status.lock() {
                        s.error = Some(e.to_string());
                    }
                    self.persist_corpus_progress(&coll, &entry);
                    continue;
                }
            };

            // Run the shared per-document path: prune deletions and enqueue the
            // changed + new chunks as one sub-job. The chunks are consumed here,
            // so this document's memory is released before the next one.
            let summary = match self.reextract_one(&runtime, &coll, &doc_id, chunks) {
                Ok(s) => s,
                Err(e) => {
                    self.record_corpus_doc(
                        &entry,
                        CorpusDocProgress {
                            doc_id: doc_id.clone(),
                            state: CorpusDocState::Failed,
                            job_id: String::new(),
                            changed: 0,
                            added: 0,
                            deleted: 0,
                        },
                    );
                    if let Ok(mut s) = entry.status.lock() {
                        s.error = Some(e.to_string());
                    }
                    self.persist_corpus_progress(&coll, &entry);
                    continue;
                }
            };

            // Await this document's sub-job (if any) so per-document state is
            // truthful and memory is not over-committed by racing every doc.
            let doc_state = self.await_subjob(&coll, &entry, &summary.job_id).await;
            self.record_corpus_doc(
                &entry,
                CorpusDocProgress {
                    doc_id: doc_id.clone(),
                    state: doc_state,
                    job_id: summary.job_id.clone(),
                    changed: summary.changed,
                    added: summary.added,
                    deleted: summary.deleted,
                },
            );
            // Roll the document's chunk-level totals into the master aggregate.
            self.accumulate_corpus(&entry, &summary);
            self.persist_corpus_progress(&coll, &entry);
        }

        // Finalize the master state from the aggregated per-document outcomes.
        if let Ok(runtime) = self.runtime(&coll) {
            self.finalize_corpus(&runtime, &entry);
        } else {
            // Collection gone: still set a terminal state in memory.
            self.finalize_corpus_in_memory(&entry);
        }
        self.persist_corpus_progress(&coll, &entry);
    }

    /// Await a sub-job to a terminal state, polling its status with a short
    /// sleep. Returns the corpus-document state implied by the sub-job's
    /// terminal state. An empty `job_id` means nothing needed re-extraction, so
    /// the document is Completed immediately. Corpus cancel cancels the sub-job.
    async fn await_subjob(
        &self,
        coll: &str,
        entry: &Arc<CorpusJobEntry>,
        job_id: &str,
    ) -> CorpusDocState {
        if job_id.is_empty() {
            return CorpusDocState::Completed;
        }
        loop {
            // A corpus-level cancel cancels the in-flight sub-job and stops the
            // wait; the document counts as Failed (it did not finish cleanly).
            if entry.cancel.load(Ordering::Relaxed) {
                let _ = self.cancel_extraction(coll, job_id);
                return CorpusDocState::Failed;
            }
            match self.job_status(coll, job_id) {
                Ok(s) => match s.state {
                    JobState::Completed => return CorpusDocState::Completed,
                    // Partial success (ADR-008): report honestly, not as a hard
                    // failure.
                    JobState::CompletedWithErrors => {
                        return CorpusDocState::CompletedWithErrors
                    }
                    JobState::Failed | JobState::Cancelled => return CorpusDocState::Failed,
                    JobState::Queued | JobState::Running => {}
                },
                // The sub-job vanished: treat as failed rather than spinning.
                Err(_) => return CorpusDocState::Failed,
            }
            tokio::time::sleep(std::time::Duration::from_millis(CORPUS_POLL_INTERVAL_MS)).await;
        }
    }

    /// Record one document's progress onto the master status under a short lock.
    fn record_corpus_doc(&self, entry: &Arc<CorpusJobEntry>, progress: CorpusDocProgress) {
        if let Ok(mut s) = entry.status.lock() {
            s.processed_documents += 1;
            match progress.state {
                CorpusDocState::Failed => s.failed_documents += 1,
                CorpusDocState::Skipped => s.skipped_documents += 1,
                // CompletedWithErrors is a partial success (ADR-008), not a hard
                // failure: it is not counted in failed_documents.
                _ => {}
            }
            s.documents.push(progress);
        }
    }

    /// Roll a sub-job's final chunk-level counters into the master aggregate.
    /// Reads the sub-job's own status (entities / edges written) when present.
    fn accumulate_corpus(&self, entry: &Arc<CorpusJobEntry>, summary: &ReextractSummary) {
        let (entities, edges) = if summary.job_id.is_empty() {
            (0, 0)
        } else {
            self.job_entry(&summary.job_id)
                .and_then(|e| {
                    e.status
                        .lock()
                        .ok()
                        .map(|s| (s.entities_written as u64, s.edges_written as u64))
                })
                .unwrap_or((0, 0))
        };
        if let Ok(mut s) = entry.status.lock() {
            s.changed_chunks = s.changed_chunks.saturating_add(summary.changed);
            s.added_chunks = s.added_chunks.saturating_add(summary.added);
            s.deleted_chunks = s.deleted_chunks.saturating_add(summary.deleted);
            s.edges_deleted = s.edges_deleted.saturating_add(summary.edges_deleted);
            s.nodes_deleted = s.nodes_deleted.saturating_add(summary.nodes_deleted);
            s.entities_written = s.entities_written.saturating_add(entities);
            s.edges_written = s.edges_written.saturating_add(edges);
        }
    }

    /// Drive the master status to its terminal state (ADR-008 partial-success):
    /// cancelled -> Cancelled; all clean -> Completed; all failed -> Failed; a
    /// mix -> CompletedWithErrors. Then persist the terminal status.
    fn finalize_corpus(&self, runtime: &CollectionRuntime, entry: &Arc<CorpusJobEntry>) {
        self.finalize_corpus_in_memory(entry);
        let _ = self.write_corpus_progress(runtime, entry);
    }

    /// Pick and set the terminal master state without persisting (used when the
    /// collection has already been dropped).
    fn finalize_corpus_in_memory(&self, entry: &Arc<CorpusJobEntry>) {
        let cancelled = entry.cancel.load(Ordering::Relaxed);
        if let Ok(mut s) = entry.status.lock() {
            if matches!(
                s.state,
                JobState::Completed
                    | JobState::CompletedWithErrors
                    | JobState::Failed
                    | JobState::Cancelled
            ) {
                return;
            }
            // Decide from the recorded document universe (cumulative across a
            // resumed run), not from total_documents which only counts this
            // run's targets. Skipped docs are prior successes (they completed in
            // an earlier run) and so prevent a Failed verdict. Partial-success
            // docs (ADR-008) are not hard failures but taint the run, pushing
            // the master off a clean Completed. Failed means zero successes
            // across the cumulative universe.
            let with_errors = s
                .documents
                .iter()
                .filter(|d| d.state == CorpusDocState::CompletedWithErrors)
                .count();
            // A success is any doc that completed cleanly, including prior-run
            // skips counted as completed.
            let succeeded = s
                .documents
                .iter()
                .filter(|d| {
                    d.state == CorpusDocState::Completed || d.state == CorpusDocState::Skipped
                })
                .count();
            if cancelled {
                s.state = JobState::Cancelled;
            } else if s.failed_documents == 0 && with_errors == 0 {
                s.state = JobState::Completed;
            } else if succeeded == 0 && with_errors == 0 {
                s.state = JobState::Failed;
            } else {
                s.state = JobState::CompletedWithErrors;
            }
        }
    }

    /// The current status of a corpus re-extraction job. Validates the job
    /// belongs to `coll`, mirroring `job_status`. Falls back to the persisted
    /// progress file when the in-memory entry is gone (e.g. after a restart) so
    /// a resumable job can still be inspected.
    pub fn corpus_job_status(
        &self,
        coll: &str,
        corpus_job_id: &str,
    ) -> Result<CorpusJobStatus, ExtractionError> {
        if let Some(entry) = self
            .corpus_jobs
            .lock()
            .ok()
            .and_then(|g| g.get(corpus_job_id).cloned())
        {
            let status = entry
                .status
                .lock()
                .map_err(|_| ExtractionError::Config("corpus status lock poisoned".to_string()))?
                .clone();
            if status.collection != coll {
                return Err(ExtractionError::JobNotFound(corpus_job_id.to_string()));
            }
            return Ok(status);
        }
        // Not in memory: load the committed progress from disk.
        let runtime = self.runtime(coll)?;
        match self.load_corpus_progress(&runtime, corpus_job_id) {
            Some(status) if status.collection == coll => Ok(status),
            _ => Err(ExtractionError::JobNotFound(corpus_job_id.to_string())),
        }
    }

    /// Request cancellation of a corpus re-extraction job. Honored between
    /// documents and against the in-flight sub-job, so the run stops promptly
    /// and finalizes to Cancelled.
    pub fn cancel_corpus_reextraction(
        &self,
        coll: &str,
        corpus_job_id: &str,
    ) -> Result<(), ExtractionError> {
        let entry = self
            .corpus_jobs
            .lock()
            .ok()
            .and_then(|g| g.get(corpus_job_id).cloned())
            .ok_or_else(|| ExtractionError::JobNotFound(corpus_job_id.to_string()))?;
        let belongs = entry
            .status
            .lock()
            .map(|s| s.collection == coll)
            .unwrap_or(false);
        if !belongs {
            return Err(ExtractionError::JobNotFound(corpus_job_id.to_string()));
        }
        entry.cancel.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Path to a corpus job's progress file under the collection's dir.
    fn corpus_progress_path(runtime: &CollectionRuntime, corpus_job_id: &str) -> PathBuf {
        runtime
            .dir
            .join(CORPUS_JOBS_DIR)
            .join(format!("{corpus_job_id}.json"))
    }

    /// Load a corpus job's committed progress from disk, if present.
    fn load_corpus_progress(
        &self,
        runtime: &CollectionRuntime,
        corpus_job_id: &str,
    ) -> Option<CorpusJobStatus> {
        let path = Self::corpus_progress_path(runtime, corpus_job_id);
        read_json::<CorpusJobStatus>(&path).ok().flatten()
    }

    /// Persist a corpus job's current status to disk (best-effort), looking up
    /// the collection runtime by name. A persistence failure is logged, never
    /// fatal: the in-memory status remains the source of truth for this process.
    fn persist_corpus_progress(&self, coll: &str, entry: &Arc<CorpusJobEntry>) {
        if let Ok(runtime) = self.runtime(coll) {
            if let Err(e) = self.write_corpus_progress(&runtime, entry) {
                tracing::warn!(error = %e, "corpus progress persist failed; in-memory status retained");
            }
        }
    }

    /// Write a corpus job's status snapshot to its progress file.
    fn write_corpus_progress(
        &self,
        runtime: &CollectionRuntime,
        entry: &Arc<CorpusJobEntry>,
    ) -> Result<(), ExtractionError> {
        let snapshot = entry
            .status
            .lock()
            .map(|s| s.clone())
            .map_err(|_| ExtractionError::Config("corpus status lock poisoned".to_string()))?;
        let path = Self::corpus_progress_path(runtime, &snapshot.corpus_job_id);
        write_json(&path, &snapshot)
    }

    fn persist_chunk_hashes(&self, runtime: &CollectionRuntime) -> Result<(), ExtractionError> {
        let snapshot = runtime
            .chunk_hashes
            .lock()
            .map(|g| g.clone())
            .map_err(|_| ExtractionError::Config("chunk hashes lock poisoned".to_string()))?;
        write_json(&runtime.dir.join(CHUNK_HASHES_FILE), &snapshot)
    }
}

// ── Sidecar persistence helpers (atomic write via temp + rename) ─────

/// Read and deserialize a JSON sidecar, returning `None` when the file is absent.
fn read_json<T: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<Option<T>, ExtractionError> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path).map_err(|e| ExtractionError::Io(e.to_string()))?;
    let value =
        serde_json::from_slice(&bytes).map_err(|e| ExtractionError::Config(e.to_string()))?;
    Ok(Some(value))
}

/// Serialize and atomically write a JSON sidecar: write a temp file then rename
/// over the target so a concurrent reader never sees a partial file.
fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), ExtractionError> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| ExtractionError::Io(e.to_string()))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ExtractionError::Io(e.to_string()))?;
    }
    // Per-write unique temp name so two concurrent writers to the same sidecar
    // never clobber each other's temp file before the atomic rename.
    let tmp = path.with_extension(format!("tmp.{}", uuid::Uuid::new_v4()));
    std::fs::write(&tmp, &bytes).map_err(|e| ExtractionError::Io(e.to_string()))?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        // Best-effort cleanup of the temp file so a failed rename leaves no litter.
        let _ = std::fs::remove_file(&tmp);
        return Err(ExtractionError::Io(e.to_string()));
    }
    Ok(())
}
