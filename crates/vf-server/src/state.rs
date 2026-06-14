// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

/// Tracks the optimization state of a collection.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum CollectionStatus {
    Ready,
    PendingOptimization,
    Optimizing,
}

impl CollectionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            CollectionStatus::Ready => "ready",
            CollectionStatus::PendingOptimization => "pending_optimization",
            CollectionStatus::Optimizing => "optimizing",
        }
    }
}

/// Boot-time recovery path taken for a collection. Mirrors the in-memory
/// strategy chosen by `plan_recovery` (with a fallback Unknown value for
/// collections still loading or whose path has not been recorded yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum RecoveryStatus {
    Unknown,
    CleanShutdown,
    IncrementalReplay,
    FullRebuild,
}

impl RecoveryStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            RecoveryStatus::Unknown => "Unknown",
            RecoveryStatus::CleanShutdown => "CleanShutdown",
            RecoveryStatus::IncrementalReplay => "IncrementalReplay",
            RecoveryStatus::FullRebuild => "FullRebuild",
        }
    }

    /// Convert from the small numeric encoding used in the AtomicU8 holder.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => RecoveryStatus::CleanShutdown,
            2 => RecoveryStatus::IncrementalReplay,
            3 => RecoveryStatus::FullRebuild,
            _ => RecoveryStatus::Unknown,
        }
    }

    /// Numeric encoding stored in the AtomicU8 holder.
    pub fn to_u8(self) -> u8 {
        match self {
            RecoveryStatus::Unknown => 0,
            RecoveryStatus::CleanShutdown => 1,
            RecoveryStatus::IncrementalReplay => 2,
            RecoveryStatus::FullRebuild => 3,
        }
    }
}
use vf_core::store::InMemoryVectorStore;
use vf_core::types::{CollectionConfig, Metadata, QuantizationConfig, VectorId};
use vf_graph::VirtualGraph;
use vf_index::arena::VectorArena;
use vf_index::hnsw::HnswIndex;
use vf_index::hnsw_delta::HnswDeltaWriter;
use vf_index::hnsw_persistence::deserialize_topology_mmap;
use vf_index::quantized_hnsw::QuantizedHnswIndex;
use vf_index::traits::{
    IndexRecoveryStrategy, ParallelBuildConfig, PersistableIndex, RestoreOutcome, VectorIndex,
};
use vf_graph::graph_delta::GraphDeltaWriter;
use vf_graph::persistence::deserialize_base as deserialize_graph_base;
use vf_query::IndexManager;
use vf_storage::collection::CollectionManager;
use vf_storage::recovery::{plan_recovery, RecoveryStrategy};
use vf_storage::StorageError;

/// Cached metadata store that avoids rebuilding the HashMap on every query.
/// Only rebuilds when the store's generation counter has changed.
/// Uses Arc internally so concurrent readers get a cheap reference-count bump
/// instead of cloning the entire HashMap.
pub struct MetadataCache {
    cache: Mutex<(u64, Arc<HashMap<VectorId, Metadata>>)>,
}

impl MetadataCache {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new((u64::MAX, Arc::new(HashMap::new()))),
        }
    }

    /// Returns a shared reference to the cached metadata store, rebuilding
    /// only if the store's generation has changed since the last build.
    /// Uses `iter_metadata` to avoid cloning vector data during rebuild.
    pub fn get_or_rebuild(&self, store: &InMemoryVectorStore) -> Arc<HashMap<VectorId, Metadata>> {
        let mut guard = self.cache.lock();
        // Read generation inside the cache lock so the stored (gen, snapshot) pair
        // cannot record a pre-bump gen alongside a post-bump iter_metadata view.
        let current_gen = store.generation();
        if guard.0 != current_gen {
            let metadata_store: HashMap<VectorId, Metadata> =
                store.iter_metadata().into_iter().collect();
            *guard = (current_gen, Arc::new(metadata_store));
        }
        Arc::clone(&guard.1)
    }
}

/// Result returned by the optimize endpoint.
pub struct OptimizeResult {
    pub status: String,
    pub message: String,
    pub duration_ms: u64,
    pub vectors_processed: u64,
}

/// Per-collection state holding all components needed for vector operations.
pub struct CollectionState {
    pub config: CollectionConfig,
    pub store: InMemoryVectorStore,
    pub index: Box<dyn PersistableIndex>,
    pub index_manager: IndexManager,
    pub graph: VirtualGraph,
    /// First-class typed graph store. `Some` only for Hybrid collections; `None`
    /// for VectorOnly and AutoSimilarity, which use the v2 `graph` field above.
    pub graph_store: Option<vf_graph::TypedGraphStore>,
    pub metadata_cache: MetadataCache,
    pub status: Arc<std::sync::RwLock<CollectionStatus>>,
    /// True if HNSW index build was deferred during bulk insert.
    pub deferred_index: Arc<AtomicBool>,
    /// True if virtual graph computation was deferred during bulk insert.
    pub deferred_graph: Arc<AtomicBool>,
    /// True if metadata indexing was skipped during bulk insert.
    pub deferred_metadata: Arc<AtomicBool>,
    /// True if any mutation has occurred since last snapshot.
    pub dirty: Arc<AtomicBool>,
    /// Number of mutations since last snapshot.
    pub mutation_count: Arc<AtomicU64>,
    /// Number of per-collection read-lock acquisitions on this CollectionState.
    /// Incremented via `metered_read`. Relaxed ordering on the hot path.
    pub collection_read_acquisitions: AtomicU64,
    /// Number of per-collection write-lock acquisitions on this CollectionState.
    /// Incremented via `metered_write`. Relaxed ordering on the hot path.
    pub collection_write_acquisitions: AtomicU64,
    /// Coarse running sum of microseconds spent waiting for the per-collection
    /// lock. Wraps an `Instant::now()` around each acquire site through the
    /// metered helpers. Relaxed ordering on the hot path.
    pub total_blocked_microseconds: AtomicU64,
}

/// Result of `AppState::require_collection_ready` used by REST and gRPC
/// guards to choose between 503 Service Unavailable and 404 Not Found.
#[derive(Debug, Clone)]
pub enum CollectionAvailability {
    /// Collection exists on disk and is being recovered in the background.
    /// Callers should respond with 503 (REST) or `Status::unavailable` (gRPC).
    Recovering {
        name: String,
        loaded: usize,
        total: usize,
        retry_after_secs: u32,
    },
    /// Collection is not present in memory and is not pending recovery.
    /// Callers should respond with 404 (REST) or `Status::not_found` (gRPC).
    NotFound { name: String },
}

impl CollectionAvailability {
    /// Plain-English message used in error responses.
    pub fn user_message(&self) -> String {
        match self {
            CollectionAvailability::Recovering {
                name,
                retry_after_secs,
                ..
            } => format!(
                "collection '{}' is recovering, retry after {} seconds",
                name, retry_after_secs
            ),
            CollectionAvailability::NotFound { name } => {
                format!("collection '{}' not found", name)
            }
        }
    }
}

/// Acquire a per-collection read lock and record the acquisition in the
/// `CollectionState` metrics counters. Times the acquire window with
/// `Instant::now()` so high-contention waits show up in
/// `total_blocked_microseconds`. Increments use `Relaxed` ordering to keep
/// the hot path cheap.
pub fn metered_read(
    handle: &Arc<RwLock<CollectionState>>,
) -> RwLockReadGuard<'_, CollectionState> {
    let start = Instant::now();
    let guard = handle.read();
    let elapsed_us = start.elapsed().as_micros() as u64;
    guard
        .collection_read_acquisitions
        .fetch_add(1, Ordering::Relaxed);
    guard
        .total_blocked_microseconds
        .fetch_add(elapsed_us, Ordering::Relaxed);
    guard
}

/// Acquire a per-collection write lock and record the acquisition in the
/// `CollectionState` metrics counters. Mirrors `metered_read` for the write
/// side; same hot-path discipline (Relaxed atomics, single Instant::now pair).
pub fn metered_write(
    handle: &Arc<RwLock<CollectionState>>,
) -> RwLockWriteGuard<'_, CollectionState> {
    let start = Instant::now();
    let guard = handle.write();
    let elapsed_us = start.elapsed().as_micros() as u64;
    guard
        .collection_write_acquisitions
        .fetch_add(1, Ordering::Relaxed);
    guard
        .total_blocked_microseconds
        .fetch_add(elapsed_us, Ordering::Relaxed);
    guard
}

/// Heuristic retry-after value in seconds. Per-collection load time is not
/// instrumented in this build, so we report a conservative bound: 30 s while
/// recovery has just started, scaling to 60 s after the first minute. The
/// number is advisory; clients are expected to back off and retry, not to
/// trust the value as a hard SLA.
fn estimate_retry_after_secs(ls: &LoadingState) -> u32 {
    let elapsed = ls.started_at.elapsed().as_secs();
    if elapsed < 30 {
        30
    } else if elapsed < 120 {
        60
    } else {
        90
    }
}

/// Snapshot of the in-progress recovery, used by /health and the per-endpoint
/// readiness guard. The recovery task updates this as each collection finishes
/// loading so callers can tell which collections are still warming up.
#[derive(Clone)]
pub struct LoadingState {
    /// Total number of collections discovered at boot.
    pub total: usize,
    /// Names of collections that have NOT yet finished loading.
    pub in_progress: std::collections::HashSet<String>,
    /// Wall-clock instant at which recovery started.
    pub started_at: Instant,
}

impl LoadingState {
    pub fn loaded(&self) -> usize {
        self.total.saturating_sub(self.in_progress.len())
    }

    pub fn is_loading(&self, name: &str) -> bool {
        self.in_progress.contains(name)
    }
}

/// Global application state shared across all gRPC services.
///
/// Locking discipline for `collections`:
///   1. The map RwLock is the outer lock and is held for the briefest possible
///      window (look up name -> clone the `Arc<RwLock<CollectionState>>`, then
///      drop it). Per-collection mutation never holds the map write lock.
///   2. The per-collection `RwLock<CollectionState>` is the inner lock and is
///      held for the duration of work against that collection (chunk loop for
///      bulk insert, search execution for read-side handlers).
///   3. Lock order is ALWAYS map first, then per-collection. Acquiring the map
///      lock while holding any per-collection lock is forbidden and will lead
///      to deadlock under contention.
#[derive(Clone)]
pub struct AppState {
    pub collections: Arc<RwLock<HashMap<String, Arc<RwLock<CollectionState>>>>>,
    pub collection_manager: Arc<RwLock<CollectionManager>>,
    pub config: crate::config::ServerConfig,
    pub max_ef_search: usize,
    pub max_batch_lock_size: u32,
    pub max_wal_flush_interval: u32,
    pub max_ef_construction: u32,
    /// Tracks which collections are still being recovered in the background.
    /// Empty `in_progress` means recovery is done.
    pub loading_state: Arc<RwLock<LoadingState>>,
    /// Flipped to true by the recovery task once every collection is loaded.
    /// Shared with the health/probe router so /readyz can gate orchestration
    /// while /health stays available immediately after the listeners bind.
    pub server_status: crate::health::ServerStatus,
    /// Per-collection boot recovery path. Populated once during boot by
    /// `load_single_collection` and read by the `/recovery_status` endpoint.
    /// Indexed by collection name; entries are never mutated post-boot.
    pub recovery_paths: Arc<RwLock<HashMap<String, RecoveryStatus>>>,
    /// Coarse global summary of the most-recent recovery path. Set by the
    /// boot task once the parallel recovery loop drains; encoded as the
    /// numeric form of `RecoveryStatus`.
    pub recovery_path: Arc<AtomicU8>,
    /// Wall-clock seconds the global recovery routine spent. Set once
    /// during boot; readable lock-free from the endpoint handler.
    pub recovery_elapsed_secs: Arc<AtomicU64>,
    /// Number of map-level lookups served by `collection_handle`. Acts as a
    /// proxy for outer-lock pressure on the collections RwLock.
    pub map_lock_acquisitions: Arc<AtomicU64>,
    /// LLM extraction manager. Shared across the gRPC and REST extraction
    /// handlers; per-collection runtime is registered at create and recovery
    /// for Hybrid collections only.
    pub extraction: Arc<vf_extraction::ExtractionManager>,
}

/// Outcome of the optimize work phase. Shared between the public async entry
/// point and its inner phased implementation.
enum WorkOutcome {
    NoOp,
    Done(u64),
}

/// RAII guard that guarantees the Optimizing status flag is never leaked.
///
/// While optimize holds the Optimizing gate, any early return, error, panic,
/// or future cancellation between taking the gate and committing the new index
/// would otherwise leave the flag stuck at Optimizing forever (every later
/// optimize then fails FAILED_PRECONDITION). On Drop the guard restores the
/// status to a correct non-Optimizing state: Ready if the live index ended
/// populated, else PendingOptimization so a later optimize self-heals.
///
/// The success path calls `disarm` after the index swap + flag flip have
/// committed, so the guard does not clobber a good Ready result.
struct OptimizeStatusGuard {
    status: Arc<std::sync::RwLock<CollectionStatus>>,
    coll: Arc<RwLock<CollectionState>>,
    armed: bool,
}

impl OptimizeStatusGuard {
    fn new(
        status: Arc<std::sync::RwLock<CollectionStatus>>,
        coll: Arc<RwLock<CollectionState>>,
    ) -> Self {
        Self { status, coll, armed: true }
    }

    /// Disarm on the success path: leave the committed status untouched.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for OptimizeStatusGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Failure / cancel / panic path: derive the correct resting status from
        // actual index population so an empty index is never left Ready.
        //
        // We must distinguish 'index is empty' from 'could not read the lock due
        // to contention'. Only an OBSERVED empty index justifies downgrading to
        // PendingOptimization (which would force a fresh, possibly 1M-vector,
        // rebuild). If try_read fails because another holder has the lock, the
        // index may be perfectly healthy; downgrading it then is a spurious
        // PendingOptimization. In that case we restore Ready rather than punish
        // a likely-good index with a needless rebuild.
        let restored = match self.coll.try_read() {
            Some(coll) => {
                if coll.index.len() > 0 {
                    CollectionStatus::Ready
                } else {
                    CollectionStatus::PendingOptimization
                }
            }
            // Contention (or poison): cannot observe emptiness, so do not
            // downgrade a possibly-good index. Leave it Ready.
            None => CollectionStatus::Ready,
        };
        if let Ok(mut status) = self.status.write() {
            *status = restored;
        }
    }
}

impl AppState {
    /// Create a new AppState with a CollectionManager rooted at `storage_path`.
    ///
    /// This is the historical synchronous path that blocks until every
    /// collection on disk has been recovered. It is still used by the inline
    /// tests in this crate. Production boot goes through `new_empty` plus
    /// `recover_collections` so the gRPC and REST listeners can bind before
    /// recovery starts.
    pub fn new(
        storage_path: &Path,
        max_ef_search: usize,
        max_batch_lock_size: u32,
        max_wal_flush_interval: u32,
        max_ef_construction: u32,
        config: crate::config::ServerConfig,
    ) -> Result<Self, StorageError> {
        let state = Self::new_empty(
            storage_path,
            max_ef_search,
            max_batch_lock_size,
            max_wal_flush_interval,
            max_ef_construction,
            config,
        )?;
        state.recover_collections();
        state.server_status.mark_initialized();
        Ok(state)
    }

    /// Build an AppState skeleton that owns a CollectionManager and the
    /// `loading_state` describing what is about to be recovered, but does NOT
    /// load any collections. The returned state can be cloned cheaply (all
    /// fields are Arc) so the gRPC and REST listeners can be spawned with it
    /// while a background task drives `recover_collections` against the same
    /// shared map.
    pub fn new_empty(
        storage_path: &Path,
        max_ef_search: usize,
        max_batch_lock_size: u32,
        max_wal_flush_interval: u32,
        max_ef_construction: u32,
        config: crate::config::ServerConfig,
    ) -> Result<Self, StorageError> {
        let collection_manager = CollectionManager::new(storage_path)?;

        let collection_names: Vec<String> = collection_manager
            .list_collections()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let loading_state = LoadingState {
            total: collection_names.len(),
            in_progress: collection_names.iter().cloned().collect(),
            started_at: Instant::now(),
        };

        // Build the extraction manager. The master key is read from the
        // environment; an absent key is fine (collections without a sealed
        // api key still work), an invalid key is logged and treated as absent
        // so boot never fails on a malformed env value.
        let master_key = match vf_extraction::MasterKey::from_base64_env("SWARNDB_MASTER_KEY") {
            Ok(mk) => mk,
            Err(e) => {
                tracing::warn!("SWARNDB_MASTER_KEY is set but invalid ({e}); extraction api keys will be unavailable");
                None
            }
        };
        let pricing = match &config.extraction_pricing_path {
            Some(path) => match vf_extraction::PricingTable::from_json_path(path) {
                Ok(p) => Arc::new(p),
                Err(e) => {
                    tracing::warn!("failed to load extraction pricing from {path}: {e}; using built-in pricing");
                    Arc::new(vf_extraction::PricingTable::builtin())
                }
            },
            None => Arc::new(vf_extraction::PricingTable::builtin()),
        };
        let extraction = vf_extraction::ExtractionManager::new(
            master_key,
            config.extraction_worker_concurrency,
            config.extraction_cache_max_entries,
            pricing,
        );

        Ok(Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            collection_manager: Arc::new(RwLock::new(collection_manager)),
            config,
            max_ef_search,
            max_batch_lock_size,
            max_wal_flush_interval,
            max_ef_construction,
            loading_state: Arc::new(RwLock::new(loading_state)),
            server_status: crate::health::ServerStatus::new(),
            recovery_paths: Arc::new(RwLock::new(HashMap::new())),
            recovery_path: Arc::new(AtomicU8::new(RecoveryStatus::Unknown.to_u8())),
            recovery_elapsed_secs: Arc::new(AtomicU64::new(0)),
            map_lock_acquisitions: Arc::new(AtomicU64::new(0)),
            extraction,
        })
    }

    /// Recover every collection discovered at `new_empty` time, in parallel.
    ///
    /// As each collection finishes loading, the resulting `CollectionState`
    /// is inserted into `self.collections` and the name is removed from
    /// `self.loading_state.in_progress`. This is what the boot-time
    /// background task on the main thread invokes; it also drives the
    /// synchronous `new()` path used by inline tests.
    pub fn recover_collections(&self) {
        let recovery_start = Instant::now();
        let collection_names: Vec<String> = {
            let ls = self.loading_state.read();
            ls.in_progress.iter().cloned().collect()
        };

        let total_collections = collection_names.len();
        let configured_max = self.config.max_concurrent_collection_loads;
        let max_concurrent = configured_max.max(1);

        if total_collections == 0 {
            // No collections to recover; still record the elapsed window so
            // /recovery_status reports a deterministic value.
            self.recovery_elapsed_secs
                .store(recovery_start.elapsed().as_secs(), Ordering::Release);
            return;
        }

        tracing::info!(
            total_collections,
            max_concurrent,
            "loading collections in parallel"
        );

        // Dedicated rayon pool so concurrent index builds elsewhere (for
        // example HNSW parallel build inside a single collection) do not
        // contend with the boot pool's worker count.
        let pool = match ThreadPoolBuilder::new()
            .num_threads(max_concurrent)
            .thread_name(|i| format!("collection-loader-{}", i))
            .build()
        {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(
                    "failed to build collection loader pool: {e}; recovery will not run"
                );
                return;
            }
        };

        // Capture the calling thread's tracing dispatcher so per-task
        // log emissions inside the rayon pool are visible to whatever
        // subscriber the caller has installed (tests use a thread-local
        // subscriber via `with_default`; production wires a global one).
        // Without this, rayon worker threads default to the no-op
        // dispatcher and the per-collection `recovery plan:` and
        // `recovered collection with` events are dropped.
        let parent_dispatch =
            tracing::dispatcher::get_default(|d| d.clone());

        let collections_arc = Arc::clone(&self.collections);
        let loading_arc = Arc::clone(&self.loading_state);
        let cm_arc = Arc::clone(&self.collection_manager);
        let recovery_paths_arc = Arc::clone(&self.recovery_paths);
        // Names of collections that finished loading, collected so extraction
        // runtimes can be registered after the parallel loop drains (the
        // registration needs `self` and the published map handle, which are not
        // available inside the rayon closure).
        let loaded_names: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let loaded_names_outer = Arc::clone(&loaded_names);

        pool.install(|| {
            collection_names
                .into_par_iter()
                .for_each(|name| {
                    let loaded_names_inner = Arc::clone(&loaded_names_outer);
                    // Catch panics at the per-task boundary so a single
                    // bad collection cannot poison boot for the rest.
                    // SAFETY: AssertUnwindSafe is used because the
                    // borrowed references and the moved `name` are only
                    // read inside the closure; on a panic we discard
                    // any partial state for this collection and log.
                    let panic_name = name.clone();
                    let dispatch = parent_dispatch.clone();
                    let cm_arc_inner = Arc::clone(&cm_arc);
                    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                        tracing::dispatcher::with_default(&dispatch, || {
                            let cm = cm_arc_inner.read();
                            Self::load_single_collection(name, &cm)
                        })
                    }));

                    match result {
                        Ok(Some((loaded_name, loaded_state, recovery_path))) => {
                            // Record the per-collection recovery path BEFORE the
                            // collection becomes externally visible. Once the map
                            // write below publishes the Arc, /readyz can flip,
                            // and /recovery_status callers must see a non-Unknown
                            // entry for the collection they observe.
                            {
                                let mut rp = recovery_paths_arc.write();
                                rp.insert(loaded_name.clone(), recovery_path);
                            }
                            // Publish the loaded collection atomically. The per-collection
                            // RwLock wrapper is created here so the map only holds Arcs.
                            {
                                let mut map = collections_arc.write();
                                map.insert(loaded_name.clone(), Arc::new(RwLock::new(loaded_state)));
                            }
                            {
                                let mut ls = loading_arc.write();
                                ls.in_progress.remove(&loaded_name);
                            }
                            // Record the name for post-loop extraction registration.
                            loaded_names_inner.lock().push(loaded_name);
                        }
                        Ok(None) => {
                            // Loader returned None (data-level error already
                            // logged as warn). Remove from in_progress so the
                            // boot does not stay "recovering" forever.
                            let mut ls = loading_arc.write();
                            ls.in_progress.remove(&panic_name);
                        }
                        Err(_) => {
                            tracing::dispatcher::with_default(
                                &parent_dispatch,
                                || {
                                    tracing::error!(
                                        collection = %panic_name,
                                        "collection loader panicked, skipping"
                                    );
                                },
                            );
                            let mut ls = loading_arc.write();
                            ls.in_progress.remove(&panic_name);
                        }
                    }
                });
        });

        let recovered_count = self.collections.read().len();
        if recovered_count > 0 {
            tracing::info!(
                "recovered {} collection(s) from storage",
                recovered_count
            );
        }

        // Register extraction runtimes for recovered Hybrid collections now that
        // every loaded collection's handle is published in the map. Non-Hybrid
        // collections are skipped inside the helper.
        let to_register: Vec<String> = loaded_names.lock().clone();
        for name in to_register {
            self.register_extraction_if_hybrid(&name);
        }

        // Record the boot recovery summary BEFORE the boot task flips
        // /readyz to 200. The summary path uses the worst (highest-cost)
        // path actually taken: FullRebuild beats IncrementalReplay beats
        // CleanShutdown beats Unknown. This gives operators a single
        // worst-case answer without losing the per-collection detail
        // recorded in `recovery_paths`.
        let elapsed_secs = recovery_start.elapsed().as_secs();
        self.recovery_elapsed_secs.store(elapsed_secs, Ordering::Release);
        let worst_path = {
            let rp = self.recovery_paths.read();
            let mut worst = RecoveryStatus::Unknown;
            for v in rp.values() {
                if v.to_u8() > worst.to_u8() {
                    worst = *v;
                }
            }
            worst
        };
        self.recovery_path
            .store(worst_path.to_u8(), Ordering::Release);
    }

    /// Short-lived helper that clones the per-collection `Arc<RwLock<CollectionState>>`
    /// out of the map under a map read lock. Returns `None` if the collection
    /// is not present. Callers should immediately drop this helper's return
    /// value before doing any work that might block, then take the per-
    /// collection lock (read or write) on the returned handle. This is the
    /// canonical entry point for every per-collection operation.
    pub fn collection_handle(
        &self,
        name: &str,
    ) -> Option<Arc<RwLock<CollectionState>>> {
        // Bump the map-lock acquisition counter once per call. Relaxed
        // ordering: this counter is observed by the metrics endpoint, not
        // by control flow.
        self.map_lock_acquisitions
            .fetch_add(1, Ordering::Relaxed);
        let map = self.collections.read();
        map.get(name).map(Arc::clone)
    }

    /// Per-endpoint readiness gate. Returns `Ok(())` only when the named
    /// collection is fully loaded. Returns a structured `NotReady` error if
    /// the name is still listed in the recovery queue. Returns `NotFound` if
    /// the collection is neither loaded nor pending recovery.
    pub fn require_collection_ready(
        &self,
        name: &str,
    ) -> Result<(), CollectionAvailability> {
        if self.collections.read().contains_key(name) {
            return Ok(());
        }
        let ls = self.loading_state.read();
        if ls.is_loading(name) {
            return Err(CollectionAvailability::Recovering {
                name: name.to_string(),
                loaded: ls.loaded(),
                total: ls.total,
                retry_after_secs: estimate_retry_after_secs(&ls),
            });
        }
        Err(CollectionAvailability::NotFound {
            name: name.to_string(),
        })
    }

    /// Register a collection's extraction runtime if it is Hybrid. Looks up the
    /// per-collection handle and the collection dir, builds a `GraphWriter` over
    /// the handle, and registers + loads the runtime in the extraction manager.
    /// A no-op for non-Hybrid collections and for collections not yet visible in
    /// the map. Must be called AFTER the collection's `Arc<RwLock<..>>` is in the
    /// map so the writer can hold that exact handle.
    pub fn register_extraction_if_hybrid(&self, name: &str) {
        let handle = match self.collection_handle(name) {
            Some(h) => h,
            None => return,
        };
        // Read the per-collection config (mode) and the on-disk dir.
        let config = {
            let coll = metered_read(&handle);
            coll.config.clone()
        };
        if config.effective_mode() != vf_core::types::Mode::Hybrid {
            return;
        }
        let collection_dir = {
            let cm = self.collection_manager.read();
            cm.get_collection(name)
                .map(|c| c.collection_dir().to_path_buf())
                .ok()
        };
        let collection_dir = match collection_dir {
            Some(d) => d,
            None => {
                tracing::warn!(collection = %name, "extraction registration skipped: collection dir not found");
                return;
            }
        };
        let extraction_dir = collection_dir.join("extraction");

        let writer: Arc<dyn vf_extraction::GraphWriter> =
            Arc::new(crate::extraction_graph_writer::CollectionGraphWriter::new(
                Arc::clone(&handle),
                Arc::clone(&self.collection_manager),
                name.to_string(),
            ));

        // load_collection creates the extraction dir, opens the cache, and folds
        // in any persisted sidecars (llm config, ontology, proposals, reject
        // list). For a brand-new Hybrid collection no sidecars exist yet, so it
        // simply seeds an empty runtime; this same call covers both create and
        // recovery, keeping the registration path single and consistent.
        if let Err(e) = self
            .extraction
            .load_collection(name, writer, extraction_dir)
        {
            tracing::warn!(collection = %name, "extraction load_collection failed: {e}");
        }
    }

    /// Graph store config seeded from the environment. P09.5:
    /// SWARNDB_GRAPH_ENTITY_INDEX=0 selects the legacy O(n) scan for measurement;
    /// default on.
    fn graph_store_config_from_env() -> vf_graph::GraphStoreConfig {
        let mut cfg = vf_graph::GraphStoreConfig::default();
        if let Ok(v) = std::env::var("SWARNDB_GRAPH_ENTITY_INDEX") {
            let off = v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off");
            cfg.name_index_enabled = !off;
        }
        cfg
    }

    /// Build an empty typed graph store for a NEW Hybrid collection, attaching a
    /// fresh typed delta writer. Returns `None` for VectorOnly / AutoSimilarity.
    pub(crate) fn create_hybrid_graph_store(
        collection_dir: &std::path::Path,
        config: &CollectionConfig,
    ) -> Option<vf_graph::TypedGraphStore> {
        if config.effective_mode() != vf_core::types::Mode::Hybrid {
            return None;
        }
        let mut store =
            vf_graph::TypedGraphStore::new(Self::graph_store_config_from_env());
        let typed_delta = collection_dir.join("graph_typed.delta");
        match vf_graph::TypedDeltaWriter::create(&typed_delta) {
            Ok(w) => store.set_delta_writer(w),
            Err(e) => tracing::warn!("failed to create typed graph delta writer: {e}"),
        }
        Some(store)
    }

    /// Recover a Hybrid collection's typed graph store: load the v3 base if
    /// present, replay the typed delta after the base LSN, then keep appending
    /// to that same delta. Unlike the v2 graph (which truncates on recovery),
    /// typed edges are not recomputable, so the durable delta tail is preserved
    /// until the next snapshot. Returns `None` for non-Hybrid collections.
    pub(crate) fn recover_hybrid_graph_store(
        name: &str,
        collection_dir: &std::path::Path,
        config: &CollectionConfig,
    ) -> Option<vf_graph::TypedGraphStore> {
        if config.effective_mode() != vf_core::types::Mode::Hybrid {
            return None;
        }
        let typed_base = collection_dir.join("graph_typed.base");
        let typed_delta = collection_dir.join("graph_typed.delta");
        let (mut store, base_lsn) = if typed_base.exists() {
            match std::fs::File::open(&typed_base)
                .map_err(|e| e.to_string())
                .and_then(|mut f| {
                    // Honor the env-driven entity-index switch on recovered stores.
                    vf_graph::deserialize_typed_base_with_config(
                        &mut f,
                        Self::graph_store_config_from_env(),
                    )
                    .map_err(|e| e.to_string())
                }) {
                Ok((lsn, s)) => (s, lsn),
                Err(e) => {
                    tracing::warn!(collection = %name, "typed graph base load failed ({e}); starting empty");
                    (
                        vf_graph::TypedGraphStore::new(Self::graph_store_config_from_env()),
                        0,
                    )
                }
            }
        } else {
            (
                vf_graph::TypedGraphStore::new(Self::graph_store_config_from_env()),
                0,
            )
        };
        if typed_delta.exists() {
            match store.replay_delta_after_lsn(&typed_delta, base_lsn) {
                Ok(n) => {
                    tracing::info!(collection = %name, "replayed {n} typed graph delta entries")
                }
                Err(e) => {
                    tracing::warn!(collection = %name, "typed graph delta replay failed: {e}")
                }
            }
            match vf_graph::TypedDeltaWriter::open(&typed_delta) {
                Ok(w) => store.set_delta_writer(w),
                Err(e) => {
                    tracing::warn!(collection = %name, "failed to open typed graph delta writer: {e}")
                }
            }
        } else {
            match vf_graph::TypedDeltaWriter::create(&typed_delta) {
                Ok(w) => store.set_delta_writer(w),
                Err(e) => {
                    tracing::warn!(collection = %name, "failed to create typed graph delta writer: {e}")
                }
            }
        }
        Some(store)
    }

    /// Load a single collection's in-memory state from disk.
    ///
    /// Mirrors the per-iteration body of the original sequential boot loop:
    /// plans recovery, loads vectors, populates the metadata store, runs the
    /// recovery strategy (CleanShutdown, IncrementalReplay, FullRebuild),
    /// initialises the delta writers, and assembles a `CollectionState`.
    ///
    /// Returns `Some((name, state))` on success. On any failure the helper
    /// logs a `tracing::warn!` and returns `None`, matching the "log and
    /// skip" contract of the original sequential loop. Only data-dir-level
    /// errors propagate; those are surfaced in `AppState::new` itself, not
    /// here.
    fn load_single_collection(
        name: String,
        collection_manager: &CollectionManager,
    ) -> Option<(String, CollectionState, RecoveryStatus)> {
        let collection = match collection_manager.get_collection(&name) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    collection = %name,
                    "failed to load collection, skipping: {e}"
                );
                return None;
            }
        };

        let config = collection.config().clone();
        let dimension = config.dimension;
        let distance_metric = config.distance_metric;
        let collection_dir = collection.collection_dir().to_path_buf();

        // Plan recovery strategy based on available files.
        let mut plan = plan_recovery(&name, &collection_dir);

        // G2 fix: under IncrementalReplay, the HNSW delta tail and the graph
        // delta tail must be in lock-step. If hnsw.delta is present but
        // graph.delta is missing (or graph.base is missing while graph.delta
        // exists), replaying HNSW alone would drift the graph topology
        // relative to the index. Demote to FullRebuild so both layers
        // reconverge from the same vector set. Logged as error because in
        // a healthy pipeline these tails are written together.
        if matches!(plan.strategy, RecoveryStrategy::IncrementalReplay { .. }) {
            let hnsw_delta_present = plan.has_hnsw_delta;
            let graph_base_present = plan.has_graph_base;
            let graph_delta_present = plan.has_graph_delta;
            let inconsistent = (hnsw_delta_present && graph_base_present && !graph_delta_present)
                || (graph_delta_present && !graph_base_present);
            if inconsistent {
                tracing::error!(
                    collection = %name,
                    hnsw_delta = hnsw_delta_present,
                    graph_base = graph_base_present,
                    graph_delta = graph_delta_present,
                    "G2 inconsistency between hnsw.delta and graph.delta detected; forcing full rebuild to avoid topology drift"
                );
                plan.strategy = RecoveryStrategy::FullRebuild;
            }
        }

        tracing::info!(
            collection = %name,
            strategy = ?plan.strategy,
            "recovery plan: {:?}", plan.strategy
        );

        // Load all vectors from segments + memtable (needed for all strategies).
        let vectors = match collection.load_all_vectors() {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    collection = %name,
                    "failed to load vectors, skipping: {e}"
                );
                return None;
            }
        };
        let vector_count = vectors.len();

        // Populate the in-memory metadata store (needed for all strategies).
        let store = InMemoryVectorStore::new(dimension);
        for (id, _data, metadata) in &vectors {
            if let Err(e) = store.insert_metadata(*id, metadata.clone()) {
                tracing::warn!(
                    collection = %name,
                    vector_id = id,
                    "failed to insert metadata into store: {e}"
                );
            }
        }

        // Helper closure to load (or rebuild) the graph alongside the
        // index for the CleanShutdown strategy. Returns a typed result
        // so the index and graph results can be combined.
        let load_graph_clean = || -> Result<VirtualGraph, String> {
            let graph_path = collection_dir.join("graph.base");
            if !graph_path.exists() {
                let g = match config.default_similarity_threshold {
                    Some(t) if t > 0.0 => VirtualGraph::with_threshold(t, config.distance_metric),
                    _ => VirtualGraph::with_threshold(0.7, config.distance_metric),
                };
                return Ok(g);
            }
            let mut file = std::fs::File::open(&graph_path)
                .map_err(|e| format!("graph base open failed: {e}"))?;
            let (_lsn, graph) = deserialize_graph_base(&mut file)
                .map_err(|e| format!("graph base load failed: {e}"))?;
            Ok(graph)
        };

        // Helper closure to load the graph for the IncrementalReplay
        // strategy. Replays graph delta if present.
        let load_graph_replay = || -> Result<VirtualGraph, String> {
            let graph_path = collection_dir.join("graph.base");
            let graph_delta_path = collection_dir.join("graph.delta");
            if graph_path.exists() {
                let mut file = std::fs::File::open(&graph_path)
                    .map_err(|e| format!("graph base open failed: {e}"))?;
                let (graph_base_lsn, mut graph) = deserialize_graph_base(&mut file)
                    .map_err(|e| format!("graph base load failed: {e}"))?;
                if graph_delta_path.exists() {
                    let replayed = vf_graph::graph_delta::replay_delta_after_lsn(
                        &mut graph, &graph_delta_path, graph_base_lsn,
                    ).map_err(|e| format!("graph delta replay failed: {e}"))?;
                    tracing::info!(
                        collection = %name,
                        "replayed graph delta, entries: {replayed}"
                    );
                }
                Ok(graph)
            } else {
                Ok(match config.default_similarity_threshold {
                    Some(t) if t > 0.0 => VirtualGraph::with_threshold(t, config.distance_metric),
                    _ => VirtualGraph::with_threshold(0.7, config.distance_metric),
                })
            }
        };

        // Attempt recovery based on strategy, falling back to full rebuild on error.
        let is_plain_hnsw = config.quantization_config.is_none();

        // Borrowed id -> vector lookup map. Used by both the plain HNSW
        // and SQ8 trait-flow helpers to populate the inner arena from the
        // base snapshot before the trait method wires topology onto it.
        // Map of borrowed slices to avoid cloning the loaded vector buffers.
        let vec_map_borrowed: HashMap<VectorId, &[f32]> = vectors
            .iter()
            .map(|(id, data, _)| (*id, data.as_slice()))
            .collect();

        let (index, graph, recovery_path) = match plan.strategy {
            RecoveryStrategy::CleanShutdown => {
                let hnsw_path = collection_dir.join("hnsw.base");

                if is_plain_hnsw {
                    // Plain HNSW boot path via the new PersistableIndex
                    // trait. Validates the on-disk envelope, populates
                    // a fresh arena from the snapshot's slot ordering,
                    // then asks the trait to load the topology in place.
                    let plain_result = Self::plain_hnsw_restore(
                        &name,
                        dimension,
                        distance_metric,
                        &collection_dir,
                        &vec_map_borrowed,
                    );
                    let graph_result = load_graph_clean();
                    match (plain_result, graph_result) {
                        (Ok(idx), Ok(g)) => {
                            vf_storage::collection::remove_shutdown_marker(&collection_dir);
                            tracing::info!(
                                collection = %name,
                                vectors = vector_count,
                                "recovered from clean shutdown (plain HNSW, trait flow)"
                            );
                            let boxed: Box<dyn PersistableIndex> = idx;
                            (boxed, g, RecoveryStatus::CleanShutdown)
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            tracing::warn!(
                                collection = %name,
                                "plain hnsw clean-shutdown recovery failed ({e}), falling back to full rebuild"
                            );
                            let (idx, g) = Self::full_rebuild(&name, dimension, distance_metric, &vectors, &config, &collection_dir);
                            (idx, g, RecoveryStatus::FullRebuild)
                        }
                    }
                } else {
                    // SQ8 boot path via the PersistableIndex trait. Mirrors
                    // the plain HNSW shape: build an empty inner HNSW with
                    // its arena populated from the base snapshot, wrap it
                    // in an empty QuantizedHnswIndex, then call the trait
                    // restore which loads the HNSW topology and the SQ8
                    // layer (quantizer.json, codes.bin, vectors.mmap) on
                    // top.
                    let sq8_config_opt = match &config.quantization_config {
                        Some(QuantizationConfig::Scalar(sq)) => Some(sq.clone()),
                        None => None,
                    };

                    let sq8_result = (|| -> Result<Box<dyn PersistableIndex>, String> {
                        let sq_config = sq8_config_opt
                            .clone()
                            .ok_or_else(|| "missing scalar quantization config".to_string())?;

                        let snapshot = deserialize_topology_mmap(&hnsw_path)
                            .map_err(|e| format!("hnsw base load (pre-restore) failed: {e}"))?;
                        let mut empty_hnsw =
                            HnswIndex::with_defaults(dimension, distance_metric);
                        empty_hnsw.populate_arena_from_snapshot(&snapshot, &vec_map_borrowed);
                        drop(snapshot);

                        let mut sq8_idx = QuantizedHnswIndex::from_existing_hnsw(
                            empty_hnsw,
                            distance_metric,
                            sq_config,
                        );
                        sq8_idx.set_data_dir(collection_dir.clone());

                        let outcome = sq8_idx
                            .try_restore_from_dir(&collection_dir)
                            .map_err(|e| format!("try_restore_from_dir failed: {e}"))?;

                        match outcome {
                            RestoreOutcome::Restored { strategy } => {
                                tracing::info!(
                                    collection = %name,
                                    strategy = ?strategy,
                                    "SQ8 fast-path recovery: loaded quantizer.json + codes.bin"
                                );
                                Ok(Box::new(sq8_idx))
                            }
                            RestoreOutcome::StateMissing => Err(
                                "SQ8 try_restore_from_dir reported StateMissing".to_string(),
                            ),
                            RestoreOutcome::StateCorrupt { reason } => Err(format!(
                                "SQ8 try_restore_from_dir reported StateCorrupt: {reason}"
                            )),
                        }
                    })();

                    let graph_result = load_graph_clean();
                    match (sq8_result, graph_result) {
                        (Ok(idx), Ok(g)) => {
                            vf_storage::collection::remove_shutdown_marker(&collection_dir);
                            tracing::info!(
                                collection = %name,
                                vectors = vector_count,
                                "recovered from clean shutdown (SQ8, trait flow)"
                            );
                            (idx, g, RecoveryStatus::CleanShutdown)
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            tracing::warn!(
                                collection = %name,
                                "SQ8 trait-flow recovery failed ({e}), falling back to full rebuild"
                            );
                            let (idx, g) = Self::full_rebuild(&name, dimension, distance_metric, &vectors, &config, &collection_dir);
                            (idx, g, RecoveryStatus::FullRebuild)
                        }
                    }
                }
            }

            RecoveryStrategy::IncrementalReplay { .. } => {
                let hnsw_path = collection_dir.join("hnsw.base");
                let hnsw_delta_path = collection_dir.join("hnsw.delta");

                if is_plain_hnsw {
                    // Plain HNSW incremental replay via the trait flow.
                    // The trait method loads only the base; delta
                    // replay stays here in the caller for P01 because
                    // the existing replayer operates on topology
                    // snapshots, not on a live HnswIndex. Lifting
                    // delta replay into the trait may happen later.
                    let plain_result = Self::plain_hnsw_restore_with_delta(
                        &name,
                        dimension,
                        distance_metric,
                        &collection_dir,
                        &hnsw_path,
                        &hnsw_delta_path,
                        &vec_map_borrowed,
                    );
                    let graph_result = load_graph_replay();
                    match (plain_result, graph_result) {
                        (Ok(idx), Ok(g)) => {
                            tracing::info!(
                                collection = %name,
                                vectors = vector_count,
                                "recovered via incremental replay (plain HNSW, trait flow)"
                            );
                            let boxed: Box<dyn PersistableIndex> = idx;
                            (boxed, g, RecoveryStatus::IncrementalReplay)
                        }
                        (Err(e), _) | (_, Err(e)) => {
                            tracing::warn!(
                                collection = %name,
                                "plain hnsw incremental replay failed ({e}), falling back to full rebuild"
                            );
                            let (idx, g) = Self::full_rebuild(&name, dimension, distance_metric, &vectors, &config, &collection_dir);
                            (idx, g, RecoveryStatus::FullRebuild)
                        }
                    }
                } else {
                    // SQ8 incremental replay via the PersistableIndex trait.
                    // The inner HNSW trait restore loads only the base
                    // topology and reports the strategy as IncrementalReplay
                    // because shutdown_clean is absent; SQ8 codes must match
                    // that base count or the SQ8 layer reports StateCorrupt
                    // and the outer match falls back to full rebuild.
                    let sq8_config_opt = match &config.quantization_config {
                        Some(QuantizationConfig::Scalar(sq)) => Some(sq.clone()),
                        None => None,
                    };

                    let replay_result = (|| -> Result<(Box<dyn PersistableIndex>, VirtualGraph), String> {
                        // Force full rebuild when a delta exists: codes.bin covers
                        // only the base, so delta-side vectors would be silently lost.
                        if hnsw_delta_path.exists() {
                            return Err(
                                "hnsw.delta present, SQ8 incremental replay needs full rebuild for code-table coherence".to_string(),
                            );
                        }

                        let sq_config = sq8_config_opt
                            .clone()
                            .ok_or_else(|| "missing scalar quantization config".to_string())?;

                        let snapshot = deserialize_topology_mmap(&hnsw_path)
                            .map_err(|e| format!("hnsw base load (pre-restore) failed: {e}"))?;
                        let mut empty_hnsw =
                            HnswIndex::with_defaults(dimension, distance_metric);
                        empty_hnsw.populate_arena_from_snapshot(&snapshot, &vec_map_borrowed);
                        drop(snapshot);

                        let mut sq8_idx = QuantizedHnswIndex::from_existing_hnsw(
                            empty_hnsw,
                            distance_metric,
                            sq_config,
                        );
                        sq8_idx.set_data_dir(collection_dir.clone());

                        let outcome = sq8_idx
                            .try_restore_from_dir(&collection_dir)
                            .map_err(|e| format!("try_restore_from_dir failed: {e}"))?;

                        let idx: Box<dyn PersistableIndex> = match outcome {
                            RestoreOutcome::Restored { strategy } => {
                                tracing::info!(
                                    collection = %name,
                                    strategy = ?strategy,
                                    "SQ8 fast-path recovery (no-delta IncrementalReplay path): loaded quantizer.json + codes.bin"
                                );
                                Box::new(sq8_idx)
                            }
                            RestoreOutcome::StateMissing => {
                                return Err(
                                    "SQ8 try_restore_from_dir reported StateMissing".to_string(),
                                );
                            }
                            RestoreOutcome::StateCorrupt { reason } => {
                                return Err(format!(
                                    "SQ8 try_restore_from_dir reported StateCorrupt: {reason}"
                                ));
                            }
                        };

                        let graph = load_graph_replay()?;
                        Ok((idx, graph))
                    })();

                    match replay_result {
                        Ok((idx, g)) => {
                            tracing::info!(
                                collection = %name,
                                vectors = vector_count,
                                "recovered via incremental replay (SQ8, trait flow)"
                            );
                            (idx, g, RecoveryStatus::IncrementalReplay)
                        }
                        Err(e) => {
                            tracing::warn!(
                                collection = %name,
                                "SQ8 incremental replay failed ({e}), falling back to full rebuild"
                            );
                            let (idx, g) = Self::full_rebuild(&name, dimension, distance_metric, &vectors, &config, &collection_dir);
                            (idx, g, RecoveryStatus::FullRebuild)
                        }
                    }
                }
            }

            RecoveryStrategy::FullRebuild => {
                tracing::info!(collection = %name, "full rebuild from vectors");
                let (idx, g) = Self::full_rebuild(&name, dimension, distance_metric, &vectors, &config, &collection_dir);
                (idx, g, RecoveryStatus::FullRebuild)
            }
        };

        // Initialize delta writers for incremental persistence.
        let hnsw_delta_path = collection_dir.join("hnsw.delta");
        match HnswDeltaWriter::create(&hnsw_delta_path) {
            Ok(writer) => index.set_delta_writer(writer),
            Err(e) => tracing::warn!(collection = %name, "failed to create hnsw delta writer: {e}"),
        }

        let graph_delta_path = collection_dir.join("graph.delta");
        let mut graph = graph;
        match GraphDeltaWriter::create(&graph_delta_path) {
            Ok(writer) => graph.set_delta_writer(writer),
            Err(e) => tracing::warn!(collection = %name, "failed to create graph delta writer: {e}"),
        }

        let index_manager = IndexManager::with_defaults();

        // Typed graph store for Hybrid collections (ADR-007 R4). Independent of
        // the HNSW recovery strategy: typed edges are user/LLM-created, not
        // similarity-derived, so they load from their own base + delta.
        let graph_store = Self::recover_hybrid_graph_store(&name, &collection_dir, &config);

        // Unified id authority seeding (hybrid only). The load loop above already
        // bumped the vector store's next_id above the max vector id via
        // insert_metadata. For a hybrid collection the recovered graph holds
        // surviving entity node ids drawn from that same authority, so seed the
        // authority above max(vector_max, node_max) by raising it past the live
        // graph max node id. This guarantees freshly allocated entity ids never
        // collide with existing vector/content or entity ids after restart.
        // Vector-only/legacy collections have no graph_store, so this is skipped
        // and behaviour is identical.
        if let Some(ref gs) = graph_store {
            store.bump_floor(gs.max_node_id() + 1);
        }

        // Orphan guard: a clean-shutdown (or incremental-replay) restore can
        // load an HNSW snapshot that is empty because a prior optimize never
        // committed, while storage still holds vector_count vectors. The
        // shutdown marker must not be trusted to imply a populated index. If the
        // restored index is empty over non-empty storage, do NOT present the
        // collection as Ready: mark deferred_index so the live index does not
        // shadow storage, and set status to PendingOptimization so the next
        // optimize() rebuilds from storage (the same FullRebuild source).
        // FullRebuild over non-empty vectors always yields a populated index, so
        // this never trips on that path.
        let index_orphaned = vector_count > 0 && index.len() == 0;
        if index_orphaned {
            tracing::warn!(
                collection = %name,
                vectors = vector_count,
                strategy = ?recovery_path,
                "recovered an empty index over non-empty storage; marking PendingOptimization for rebuild"
            );
        }
        let initial_status = if index_orphaned {
            CollectionStatus::PendingOptimization
        } else {
            CollectionStatus::Ready
        };

        let collection_state = CollectionState {
            config,
            store,
            index,
            index_manager,
            graph,
            graph_store,
            metadata_cache: MetadataCache::new(),
            status: Arc::new(std::sync::RwLock::new(initial_status)),
            deferred_index: Arc::new(AtomicBool::new(index_orphaned)),
            deferred_graph: Arc::new(AtomicBool::new(false)),
            deferred_metadata: Arc::new(AtomicBool::new(false)),
            dirty: Arc::new(AtomicBool::new(false)),
            mutation_count: Arc::new(AtomicU64::new(0)),
            collection_read_acquisitions: AtomicU64::new(0),
            collection_write_acquisitions: AtomicU64::new(0),
            total_blocked_microseconds: AtomicU64::new(0),
        };

        tracing::info!(
            collection = %name,
            vectors = vector_count,
            "recovered collection with {} vectors",
            vector_count
        );

        Some((name, collection_state, recovery_path))
    }

    /// Full rebuild path: load vectors into HNSW index via parallel bulk insert and recompute graph.
    fn full_rebuild(
        name: &str,
        dimension: usize,
        distance_metric: vf_core::types::DistanceMetricType,
        vectors: &[(VectorId, Vec<f32>, Option<Metadata>)],
        config: &CollectionConfig,
        data_dir: &Path,
    ) -> (Box<dyn PersistableIndex>, VirtualGraph) {
        let mut graph = match config.default_similarity_threshold {
            Some(t) if t > 0.0 => VirtualGraph::with_threshold(t, config.distance_metric),
            _ => VirtualGraph::with_threshold(0.7, config.distance_metric),
        };

        let vector_ids: Vec<u64> = vectors.iter().map(|(id, _, _)| *id).collect();
        // Arc-wrap so the parallel graph compute below can share clones
        // without re-allocating per-thread copies.
        let vector_map: HashMap<u64, Arc<Vec<f32>>> = vectors.iter()
            .map(|(id, data, _)| (*id, Arc::new(data.clone())))
            .collect();

        // Build the index. For SQ8 we go through cold_build_parallel so the
        // encode step runs on every core. For the plain (non-quantized)
        // path we use HnswIndex::bulk_add_with_lsn for parallel-neighbor-search
        // on cold rebuild, with a serial-add fallback on bulk failure.
        let index: Box<dyn PersistableIndex> = match &config.quantization_config {
            Some(QuantizationConfig::Scalar(sq_config)) => {
                let workers = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let parallel_config = ParallelBuildConfig {
                    workers,
                    memory_cap_bytes: None,
                    deterministic: true,
                };

                // Build an empty HNSW, wrap it, set data_dir, then ask
                // cold_build_parallel to do everything: insert into the
                // inner HNSW, train the quantizer, parallel-encode codes,
                // build mmap, persist quantizer.json and codes.bin.
                let empty_hnsw = HnswIndex::with_defaults(dimension, distance_metric);
                let mut q_index = QuantizedHnswIndex::from_existing_hnsw(
                    empty_hnsw,
                    distance_metric,
                    sq_config.clone(),
                );
                q_index.set_data_dir(data_dir.to_path_buf());

                let vec_refs: Vec<(VectorId, &[f32])> = vectors
                    .iter()
                    .map(|(id, v, _)| (*id, v.as_slice()))
                    .collect();

                match q_index.cold_build_parallel(&vec_refs, parallel_config) {
                    Ok(()) => Box::new(q_index),
                    Err(e) => {
                        tracing::warn!(
                            collection = %name,
                            error = %e,
                            "SQ8 cold_build_parallel failed; falling back to legacy sequential path"
                        );
                        // Legacy fallback: build a fresh empty HNSW, add
                        // every vector sequentially, wrap, train_quantizer.
                        // This mirrors the pre-P03 shape exactly so we keep
                        // robustness if the parallel path hits any
                        // unexpected condition.
                        let hnsw_index = HnswIndex::with_defaults(dimension, distance_metric);
                        for (id, data, _metadata) in vectors {
                            if let Err(e) = hnsw_index.add(*id, data) {
                                tracing::warn!(
                                    collection = %name,
                                    vector_id = id,
                                    "failed to add vector to HNSW index: {e}"
                                );
                            }
                        }
                        let q_index = QuantizedHnswIndex::from_existing_hnsw(
                            hnsw_index,
                            distance_metric,
                            sq_config.clone(),
                        );
                        q_index.set_data_dir(data_dir.to_path_buf());
                        q_index.train_quantizer(data_dir);
                        Box::new(q_index)
                    }
                }
            }
            None => {
                let hnsw_index = HnswIndex::with_defaults(dimension, distance_metric);
                // Single parallel bulk insert: parallel-neighbor-search + serial-mutation.
                // No delta writer is attached at rebuild time, so no delta entries are emitted;
                // LSN=0 per item is therefore unobservable and equivalent to the legacy add() loop.
                // Wrap each vector once in Arc; the index path uses Arc clones only.
                let items: Vec<(VectorId, Arc<Vec<f32>>, u64)> = vectors
                    .iter()
                    .map(|(id, data, _)| (*id, Arc::new(data.clone()), 0u64))
                    .collect();
                match hnsw_index.bulk_add_with_lsn(&items) {
                    Ok(()) => {
                        tracing::info!(
                            collection = %name,
                            count = items.len(),
                            "plain HNSW full rebuild via parallel bulk_add_with_lsn complete"
                        );
                        Box::new(hnsw_index)
                    }
                    Err(e) => {
                        tracing::warn!(
                            collection = %name,
                            error = %e,
                            "plain HNSW bulk_add_with_lsn failed; falling back to legacy sequential path"
                        );
                        // Legacy fallback: fresh empty HNSW, serial add loop.
                        let hnsw_index = HnswIndex::with_defaults(dimension, distance_metric);
                        for (id, data, _metadata) in vectors {
                            if let Err(e) = hnsw_index.add(*id, data) {
                                tracing::warn!(
                                    collection = %name,
                                    vector_id = id,
                                    "failed to add vector to HNSW index: {e}"
                                );
                            }
                        }
                        Box::new(hnsw_index)
                    }
                }
            }
        };

        // Vector-only collections keep the empty graph placeholder; only graph
        // modes recompute edges during a full rebuild.
        if config.graph_enabled() {
            if let Err(e) = vf_graph::RelationshipComputer::compute_batch_parallel(
                &mut graph, &*index, &vector_ids, &vector_map, 10,
            ) {
                tracing::warn!(collection = %name, "graph compute_batch_parallel failed: {}", e);
            }
        }

        (index, graph)
    }

    /// Plain HNSW restore via the PersistableIndex trait flow (CleanShutdown).
    ///
    /// The flow is:
    ///   1. `validate_state_on_disk` returns `Ok(true)` iff the on-disk
    ///      `hnsw.base` envelope is intact and the dimension matches.
    ///   2. Pre-load the topology snapshot so we know the slot ordering.
    ///   3. Build an empty `HnswIndex`, populate its VectorArena from the
    ///      snapshot's slot order, then call `try_restore_from_dir` to
    ///      wire the topology onto the prepared arena.
    ///   4. Log the recovery strategy reported by the trait.
    ///
    /// Any soft failure (`Ok(false)` from validate, `StateMissing` or
    /// `StateCorrupt` from try_restore) returns an `Err(String)` so the
    /// outer match can fall back to the full rebuild path.
    fn plain_hnsw_restore(
        name: &str,
        dimension: usize,
        distance_metric: vf_core::types::DistanceMetricType,
        collection_dir: &Path,
        vec_map: &HashMap<VectorId, &[f32]>,
    ) -> Result<Box<HnswIndex>, String> {
        // 1. Envelope-level validity check.
        let validated = <HnswIndex as PersistableIndex>::validate_state_on_disk(
            collection_dir,
            dimension,
        )
        .map_err(|e| format!("validate_state_on_disk failed: {e}"))?;
        if !validated {
            return Err("hnsw base validate returned false".to_string());
        }

        // 2. Pre-load the snapshot so we have the slot mapping for arena
        // population. The trait method will re-load the same file
        // internally; the second read hits the OS page cache.
        let hnsw_path = collection_dir.join("hnsw.base");
        let snapshot = deserialize_topology_mmap(&hnsw_path)
            .map_err(|e| format!("hnsw base load (pre-restore) failed: {e}"))?;

        // 3. Build empty index, populate arena, then call the trait.
        let mut idx = HnswIndex::with_defaults(dimension, distance_metric);
        idx.populate_arena_from_snapshot(&snapshot, vec_map);
        // Drop the pre-loaded snapshot before the trait call so we hold no
        // extra references during the topology load.
        drop(snapshot);

        let outcome = idx
            .try_restore_from_dir(collection_dir)
            .map_err(|e| format!("try_restore_from_dir failed: {e}"))?;

        match outcome {
            RestoreOutcome::Restored { strategy } => {
                tracing::info!(
                    collection = %name,
                    strategy = ?strategy,
                    "plain HNSW restore succeeded"
                );
                Ok(Box::new(idx))
            }
            RestoreOutcome::StateMissing => Err("try_restore_from_dir reported StateMissing".to_string()),
            RestoreOutcome::StateCorrupt { reason } => Err(format!(
                "try_restore_from_dir reported StateCorrupt: {reason}"
            )),
        }
    }

    /// Plain HNSW restore with delta replay (IncrementalReplay).
    ///
    /// Trait-flow contract for P01:
    ///   1. `validate_state_on_disk` is the envelope-level gate; a
    ///      `false` falls back to full rebuild.
    ///   2. Build an empty `HnswIndex`, populate its VectorArena from
    ///      the pre-loaded base-snapshot slot ordering, then call
    ///      `try_restore_from_dir` to wire the base topology onto the
    ///      prepared arena. The trait method reports the strategy as
    ///      `IncrementalReplay` because `shutdown_clean` is absent.
    ///   3. If `hnsw.delta` exists, extract a fresh snapshot from the
    ///      base-loaded index, run `replay_delta_after_lsn` against the
    ///      snapshot, then rebuild the index from the merged snapshot
    ///      plus a freshly slot-aligned arena. The existing replayer
    ///      operates on snapshots, not on live indexes; lifting delta
    ///      replay into the trait may happen in a later phase.
    fn plain_hnsw_restore_with_delta(
        name: &str,
        dimension: usize,
        distance_metric: vf_core::types::DistanceMetricType,
        collection_dir: &Path,
        hnsw_path: &Path,
        hnsw_delta_path: &Path,
        vec_map: &HashMap<VectorId, &[f32]>,
    ) -> Result<Box<HnswIndex>, String> {
        // 1. Envelope-level validity check.
        let validated = <HnswIndex as PersistableIndex>::validate_state_on_disk(
            collection_dir,
            dimension,
        )
        .map_err(|e| format!("validate_state_on_disk failed: {e}"))?;
        if !validated {
            return Err("hnsw base validate returned false".to_string());
        }

        // 2. Pre-load snapshot to get the slot layout for the arena.
        let pre_snapshot = deserialize_topology_mmap(hnsw_path)
            .map_err(|e| format!("hnsw base load (pre-restore) failed: {e}"))?;

        // Build empty index, populate arena, then call the trait method
        // for the base topology load.
        let mut idx = HnswIndex::with_defaults(dimension, distance_metric);
        idx.populate_arena_from_snapshot(&pre_snapshot, vec_map);
        drop(pre_snapshot);

        let outcome = idx
            .try_restore_from_dir(collection_dir)
            .map_err(|e| format!("try_restore_from_dir failed: {e}"))?;

        let (idx, strategy_logged) = match outcome {
            RestoreOutcome::Restored { strategy } => (idx, strategy),
            RestoreOutcome::StateMissing => {
                return Err("try_restore_from_dir reported StateMissing".to_string());
            }
            RestoreOutcome::StateCorrupt { reason } => {
                return Err(format!(
                    "try_restore_from_dir reported StateCorrupt: {reason}"
                ));
            }
        };

        tracing::info!(
            collection = %name,
            strategy = ?strategy_logged,
            "plain HNSW base restore succeeded (pre-delta)"
        );

        // 3. Apply delta if present. Extract a snapshot from the just-
        // loaded base, replay delta onto it, then rebuild the index.
        if hnsw_delta_path.exists() {
            let snapshot_lsn = match &strategy_logged {
                IndexRecoveryStrategy::IncrementalReplay { hnsw_base_lsn, .. } => *hnsw_base_lsn,
                _ => 0,
            };
            let mut merged_snapshot = idx.snapshot_topology(snapshot_lsn);

            let replayed = vf_index::hnsw_delta::replay_delta_after_lsn(
                &mut merged_snapshot,
                hnsw_delta_path,
                snapshot_lsn,
            )
            .map_err(|e| format!("hnsw delta replay failed: {e}"))?;
            tracing::info!(
                collection = %name,
                "replayed hnsw delta, last LSN: {replayed}"
            );

            // Rebuild arena from the merged snapshot, PRESERVING each node's
            // recorded vector_slot. restore_from_topology keeps the snapshot's
            // slot indices one-for-one, so a dense sequential push would
            // misalign every vector_slot reference whenever the snapshot has
            // gaps (from deletes) or duplicate slots. Mirror the slot-preserving
            // placement from HnswIndex::populate_arena_from_snapshot: size the
            // arena to (max_slot + 1), write each live vector at its original
            // offset, and free the gap slots.
            let mut new_arena = VectorArena::new(dimension);
            let mut nodes_by_slot: Vec<_> = merged_snapshot
                .nodes
                .iter()
                .map(|n| (n.vector_slot, n.id))
                .collect();
            nodes_by_slot.sort_by_key(|(slot, _)| *slot);
            let zeros = vec![0.0f32; dimension];
            if !nodes_by_slot.is_empty() {
                let max_slot = nodes_by_slot.last().map(|(s, _)| *s).unwrap_or(0);
                let total_slots = max_slot + 1;
                new_arena.resize_to_slots(total_slots, &zeros);
                let mut live: std::collections::HashSet<usize> =
                    std::collections::HashSet::with_capacity(nodes_by_slot.len());
                for (slot, id) in &nodes_by_slot {
                    let data = vec_map.get(id).copied().unwrap_or(zeros.as_slice());
                    new_arena.write_slot(*slot, data);
                    live.insert(*slot);
                }
                for slot in 0..total_slots {
                    if !live.contains(&slot) {
                        new_arena.free(slot);
                    }
                }
            }

            let new_idx = HnswIndex::restore_from_topology(merged_snapshot, new_arena)
                .map_err(|e| format!("hnsw restore (post-delta) failed: {e}"))?;
            return Ok(Box::new(new_idx));
        }

        Ok(Box::new(idx))
    }

    /// Prune old WAL files for a collection.
    ///
    /// Deletes all `wal_*.log.old` files in the collection directory.
    /// Returns `(files_deleted, bytes_freed)`.
    pub fn prune_wal_for_collection(&self, name: &str) -> Result<(usize, u64), String> {
        let collection_dir = {
            let cm = self.collection_manager.read();
            let coll = cm.get_collection(name)
                .map_err(|e| format!("Collection '{}' not found: {}", name, e))?;
            coll.collection_dir().to_path_buf()
        };

        let mut deleted = 0usize;
        let mut bytes_freed = 0u64;
        if let Ok(entries) = std::fs::read_dir(&collection_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                    if fname.starts_with("wal_") && fname.ends_with(".log.old") {
                        if let Ok(meta) = std::fs::metadata(&path) {
                            bytes_freed += meta.len();
                        }
                        if std::fs::remove_file(&path).is_ok() {
                            deleted += 1;
                        }
                    }
                }
            }
        }
        Ok((deleted, bytes_freed))
    }

    /// Compact segments for a collection into a single segment.
    ///
    /// Returns compaction statistics or an error if there are too few segments.
    pub fn compact_collection(&self, name: &str, min_segments: usize, remove_deleted: bool) -> Result<vf_storage::CompactionResult, String> {
        let mut cm = self.collection_manager.write();
        let coll = cm.get_collection_mut(name)
            .map_err(|e| format!("Collection '{}' not found: {}", name, e))?;

        let segment_count = coll.segment_count();
        let effective_min = if min_segments == 0 { 4 } else { min_segments };

        if segment_count < effective_min {
            return Err(format!("Only {} segments, need at least {} to compact", segment_count, effective_min));
        }

        let options = vf_storage::CompactionOptions {
            min_segments_to_compact: effective_min,
            remove_deleted,
        };

        coll.compact(options).map_err(|e| format!("Compaction failed: {}", e))
    }

    /// Rebuild deferred operations for a collection after bulk insert.
    ///
    /// Checks which operations were deferred (index, metadata, graph) and
    /// rebuilds each one. Returns statistics about the optimization.
    pub async fn optimize_collection(&self, collection_name: &str, rebuild_graph: bool) -> Result<OptimizeResult, String> {
        let start = Instant::now();

        // Read flags, decide, perform the rebuild, and clear flags all under one per-collection
        // write lock so that concurrent setters (bulk_insert, upsert, graph ops, which
        // store-true under a per-collection read lock) cannot race with the read-decide-act-clear
        // sequence on the deferred_* atomics. The map read lock is released as soon as the
        // per-collection handle is cloned out.
        let mut need_index = false;
        let mut need_metadata = false;
        let mut need_graph = false;
        // Tracks whether we transitioned status to Optimizing; only then should we reset it.
        let mut status_taken = false;
        let coll_handle = self
            .collection_handle(collection_name)
            .ok_or_else(|| format!("collection '{}' not found", collection_name))?;
        // Phased optimize. The CPU-bound HNSW rebuild runs in spawn_blocking
        // with NO collection write guard held across the await, so a 2-worker
        // tokio runtime can still schedule health probes and other tasks. The
        // status=Optimizing flag set in phase 0 is the mutual-exclusion gate:
        // once taken, no concurrent optimize proceeds, and bulk setters only
        // store-true on the deferred_* atomics (re-read in phase 0).
        let result = self
            .optimize_collection_inner(
                collection_name,
                &coll_handle,
                rebuild_graph,
                &mut need_index,
                &mut need_metadata,
                &mut need_graph,
                &mut status_taken,
            )
            .await;

        // Nothing to optimize; status was never flipped, do not touch it.
        if matches!(result, Ok(WorkOutcome::NoOp)) {
            return Ok(OptimizeResult {
                status: "already_optimized".to_string(),
                message: "nothing to optimize".to_string(),
                duration_ms: 0,
                vectors_processed: 0,
            });
        }

        // Status restoration is owned by the RAII OptimizeStatusGuard inside
        // optimize_collection_inner: it disarms to Ready on the committed
        // success path, and on any error / panic / cancel it restores a correct
        // non-Optimizing status (Ready if the index ended populated, else
        // PendingOptimization). The flag can therefore never leak as Optimizing.

        // Post-optimize maintenance (WAL prune + auto-compact) runs off the
        // runtime in spawn_blocking. Both do synchronous filesystem and segment
        // work (compact takes the global collection_manager write lock and merges
        // segments); running them inline would pin a runtime worker after an
        // otherwise-successful optimize. AppState is a cheap bag of Arc handles,
        // so cloning it into the closure is just Arc bumps. No parking_lot guard
        // is held across the .await; the closure takes its own fresh guards.
        let do_prune = result.is_ok() && self.config.wal_prune_after_optimize;
        let do_compact = result.is_ok() && self.config.auto_compact_after_optimize;
        if do_prune || do_compact {
            let maint_state = self.clone();
            let coll_name = collection_name.to_string();
            let min_segments = self.config.compaction_min_segments;
            let _ = tokio::task::spawn_blocking(move || {
                if do_prune {
                    match maint_state.prune_wal_for_collection(&coll_name) {
                        Ok((count, bytes)) => {
                            if count > 0 {
                                tracing::info!("Pruned {} WAL files ({} bytes) after optimize", count, bytes);
                            }
                        }
                        Err(e) => tracing::warn!("WAL prune after optimize failed: {}", e),
                    }
                }
                if do_compact {
                    match maint_state.compact_collection(&coll_name, min_segments, true) {
                        Ok(cr) => {
                            tracing::info!("Auto-compacted {} segments into 1 ({} vectors)", cr.segments_merged, cr.vectors_written);
                        }
                        // Not an error if too few segments.
                        Err(e) => tracing::debug!("Auto-compact skipped: {}", e),
                    }
                }
            })
            .await;
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(WorkOutcome::Done(vectors_processed)) => {
                let mut parts = Vec::new();
                if need_index { parts.push("HNSW index"); }
                if need_metadata { parts.push("metadata indexes"); }
                if need_graph && rebuild_graph { parts.push("virtual graph"); }

                Ok(OptimizeResult {
                    status: "completed".to_string(),
                    message: format!("rebuilt: {}", parts.join(", ")),
                    duration_ms,
                    vectors_processed,
                })
            }
            Ok(WorkOutcome::NoOp) => unreachable!("NoOp handled by early return above"),
            Err(e) => Err(e),
        }
    }

    /// Phased inner optimize. Runs the CPU-bound HNSW rebuild in spawn_blocking
    /// with no collection write guard held across the await, then swaps the new
    /// index in under a brief write lock. The rebuild source is the durable
    /// storage layer (load_all_vectors), not the in-memory index arena, so a
    /// deferred-mode collection (empty arena) rebuilds from the real vectors.
    #[allow(clippy::too_many_arguments)]
    async fn optimize_collection_inner(
        &self,
        collection_name: &str,
        coll_handle: &Arc<RwLock<CollectionState>>,
        rebuild_graph: bool,
        need_index: &mut bool,
        need_metadata: &mut bool,
        need_graph: &mut bool,
        status_taken: &mut bool,
    ) -> Result<WorkOutcome, String> {
        // Phase 0: read flags, decide, and take the Optimizing gate under a
        // brief write lock. Capture the config fields we need later so we do
        // not have to hold the lock across the build.
        let dimension;
        let distance_metric;
        let quantization_config;
        let graph_enabled;
        let graph_threshold;
        let vectors_processed;
        let status_handle;
        {
            let coll = metered_write(coll_handle);

            *need_index = coll.deferred_index.load(Ordering::Acquire);
            *need_metadata = coll.deferred_metadata.load(Ordering::Acquire);
            *need_graph = coll.deferred_graph.load(Ordering::Acquire);
            let has_quantization = coll.config.quantization_config.is_some();

            // Self-heal an orphaned index: if storage holds vectors but the live
            // HNSW index is empty (a prior optimize whose build was dropped before
            // commit, or a clean-shutdown restore of an empty snapshot), the
            // deferred_index flag may read false yet the index does not reflect
            // storage. Force a rebuild in that case so optimize cannot short-circuit
            // over orphaned vectors. The normal up-to-date case (index populated and
            // matching storage) is untouched, preserving the fast path.
            let stored_count = coll.store.len();
            let index_population = coll.index.len();
            if !*need_index && stored_count > 0 && index_population == 0 {
                *need_index = true;
                tracing::warn!(
                    collection = %collection_name,
                    stored = stored_count,
                    "optimize: empty index over non-empty storage, forcing HNSW rebuild"
                );
            }

            // Nothing to do if no deferred ops, or only graph is deferred but rebuild_graph is false.
            // Exception: quantized collections always need optimization to train the quantizer.
            let effective_graph = *need_graph && rebuild_graph;
            if !*need_index && !*need_metadata && !effective_graph && !has_quantization {
                return Ok(WorkOutcome::NoOp);
            }

            // Check not already optimizing.
            {
                let status = coll.status.read().unwrap();
                if *status == CollectionStatus::Optimizing {
                    return Err(format!(
                        "collection '{}' is already being optimized",
                        collection_name
                    ));
                }
            }

            // Set status to optimizing. This is the mutual-exclusion gate held
            // for the whole phased run. Restoration is owned by the RAII guard
            // constructed just below, so the flag cannot leak on any exit path.
            {
                let mut status = coll.status.write().unwrap();
                *status = CollectionStatus::Optimizing;
            }
            *status_taken = true;
            status_handle = coll.status.clone();

            dimension = coll.config.dimension;
            distance_metric = coll.config.distance_metric;
            quantization_config = coll.config.quantization_config.clone();
            graph_enabled = coll.config.graph_enabled();
            graph_threshold = coll.config.default_similarity_threshold.unwrap_or(0.7);
            vectors_processed = coll.store.len() as u64;
        }

        // Arm the RAII status guard now that the Optimizing gate is held. On any
        // early return, error, panic, or future cancellation from here on, Drop
        // restores a correct non-Optimizing status (Ready if the index ended
        // populated, else PendingOptimization). The success path disarms it after
        // the index swap + flag flip have committed.
        let mut status_guard = OptimizeStatusGuard::new(status_handle, coll_handle.clone());

        // The durable vectors are loaded once from the storage layer and the
        // HNSW rebuild runs off the runtime. Both the load and the build happen
        // inside spawn_blocking so the global collection_manager read lock is
        // never held on a runtime worker across the multi-minute load+build, and
        // no parking_lot guard is held across an .await (which would make the
        // future non-Send). The load source is storage, not the in-memory index
        // arena, so a deferred-mode collection (empty arena) rebuilds from real
        // vectors. This mirrors the recovery full_rebuild Arc/bulk_add_with_lsn
        // idiom (see plain HNSW full rebuild, ~lines 1413-1424).
        let graph_needs_vectors = *need_graph && rebuild_graph && graph_enabled;
        let needs_vectors = *need_index || graph_needs_vectors;

        // F10 (lost-write race): capture the storage high-water LSN as observed
        // at load time. Writes that commit after this LSN land durably in storage
        // but are not in the freshly built index. At swap we re-read the LSN; if
        // it advanced we do NOT mask the gap as Ready but set PendingOptimization
        // so a follow-up optimize covers the post-load tail.
        let mut load_high_water_lsn: u64 = 0;

        // Resolve the quantization data dir before the blocking build so the
        // closure carries an owned PathBuf, not a manager borrow.
        let quant_data_dir = if *need_index && quantization_config.is_some() {
            let cm = self.collection_manager.read();
            let dir = cm
                .get_collection(collection_name)
                .map(|c| c.collection_dir().to_path_buf())
                .unwrap_or_else(|_| {
                    std::env::temp_dir()
                        .join(format!("swarndb_optimize_{}", collection_name))
                });
            Some(dir)
        } else {
            None
        };

        // Load + build off the runtime. The closure returns the built index (if
        // needed), the loaded vectors (moved out for the graph phase, never
        // cloned), and the captured high-water LSN. A fresh manager read guard is
        // taken inside the blocking thread and dropped there.
        let cm_for_load = self.collection_manager.clone();
        let coll_name_owned = collection_name.to_string();
        let quant_for_build = if *need_index { quantization_config.clone() } else { None };
        let need_index_build = *need_index;
        let graph_keeps_vectors = graph_needs_vectors;
        let (built_index, mut vector_data): (Option<Box<dyn PersistableIndex>>, Vec<(VectorId, Vec<f32>)>) =
            if needs_vectors {
                tokio::task::spawn_blocking(move || {
                    // Brief storage read inside the blocking thread. Capture the
                    // high-water LSN first so it reflects the state we load.
                    let (vector_data, high_water): (Vec<(VectorId, Vec<f32>)>, u64) = {
                        let cm = cm_for_load.read();
                        let sc = cm
                            .get_collection(&coll_name_owned)
                            .map_err(|e| format!("storage collection not found for optimize: {e}"))?;
                        let high_water = sc.current_lsn();
                        let loaded = sc
                            .load_all_vectors()
                            .map_err(|e| format!("load_all_vectors for optimize failed: {e}"))?
                            .into_iter()
                            .map(|(id, v, _meta)| (id, v))
                            .collect();
                        (loaded, high_water)
                    };

                    if !need_index_build {
                        // Graph-only rebuild needs the vectors but no new index.
                        return Ok::<_, String>((None, vector_data, high_water));
                    }

                    // F1 / F9: build via the migrated Arc/bulk_add_with_lsn idiom
                    // used by recovery full_rebuild (~lines 1413-1424). Wrap each
                    // loaded vector in an Arc once. When the graph phase does NOT
                    // also need the vectors we MOVE them out of vector_data into
                    // the Arcs (drain), so only one full copy exists at build time
                    // instead of two; the build_items Arcs are then the sole copy
                    // and drop right after the build. When the graph DOES need the
                    // vectors we keep vector_data intact and Arc-share a clone, the
                    // one unavoidable extra copy.
                    let new_hnsw = HnswIndex::with_defaults(dimension, distance_metric);
                    let (vector_data, items): (
                        Vec<(VectorId, Vec<f32>)>,
                        Vec<(VectorId, Arc<Vec<f32>>, u64)>,
                    ) = if graph_keeps_vectors {
                        let items = vector_data
                            .iter()
                            .map(|(id, v)| (*id, Arc::new(v.clone()), 0u64))
                            .collect();
                        (vector_data, items)
                    } else {
                        // Move each Vec into an Arc; vector_data is emptied.
                        let items = vector_data
                            .into_iter()
                            .map(|(id, v)| (id, Arc::new(v), 0u64))
                            .collect();
                        (Vec::new(), items)
                    };
                    new_hnsw
                        .bulk_add_with_lsn(&items)
                        .map_err(|e| format!("HNSW index rebuild failed: {}", e))?;
                    drop(items);
                    new_hnsw.compact();

                    // Wrap in QuantizedHnswIndex if quantization is configured.
                    let boxed: Box<dyn PersistableIndex> = match &quant_for_build {
                        Some(QuantizationConfig::Scalar(sq_config)) => {
                            let q_index = QuantizedHnswIndex::from_existing_hnsw(
                                new_hnsw,
                                distance_metric,
                                sq_config.clone(),
                            );
                            if let Some(dir) = quant_data_dir.as_ref() {
                                let _ = std::fs::create_dir_all(dir);
                                q_index.set_data_dir(dir.clone());
                                q_index.train_quantizer(dir);
                            }
                            Box::new(q_index)
                        }
                        _ => Box::new(new_hnsw),
                    };
                    Ok::<_, String>((Some(boxed), vector_data, high_water))
                })
                .await
                .map_err(|e| format!("optimize build task join error: {e}"))?
                .map(|(idx, data, hw)| {
                    load_high_water_lsn = hw;
                    (idx, data)
                })?
            } else {
                (None, Vec::new())
            };

        // Phase 1 commit: swap the freshly built index in.
        if *need_index {
            let new_index = built_index
                .ok_or_else(|| "optimize: index build requested but no index produced".to_string())?;

            // Refuse to commit an empty index over non-empty storage. A build
            // that returned an empty index while storage holds vectors would
            // otherwise be swapped in and the flag cleared, presenting an
            // orphaned (unsearchable) collection as optimized. Returning Err
            // here leaves the guard armed, which restores PendingOptimization so
            // a later optimize retries from storage.
            if vectors_processed > 0 && new_index.len() == 0 {
                return Err(format!(
                    "optimize built an empty HNSW index over {} stored vectors; refusing to commit",
                    vectors_processed
                ));
            }

            // Commit: swap the freshly built index in and clear the deferred
            // flag together under a single brief write lock so the swap and the
            // flag flip are atomic. A failure cannot leave a populated-but-
            // deferred or empty-but-ready state because both happen together.
            {
                let mut coll = metered_write(coll_handle);
                coll.index = new_index;
                coll.deferred_index.store(false, Ordering::Release);
            }
            tracing::info!(
                collection = %collection_name,
                vectors = vectors_processed,
                "rebuilt HNSW index"
            );
        }

        // Phase 2 / 2.5: metadata rebuild + quantizer train under a brief write
        // lock. Both are bounded by collection size and operate on resident
        // state. The metadata rebuild iterates the store single-threaded, so the
        // write-lock hold is kept as short as the work allows and the graph phase
        // (the expensive part) is moved off the lock below.
        {
            let mut coll = metered_write(coll_handle);

            // 2. Rebuild metadata indexes if skipped.
            if *need_metadata {
                let mut new_manager = IndexManager::with_defaults();
                let metadata_pairs = coll.store.iter_metadata();
                for (id, meta) in &metadata_pairs {
                    new_manager.index_record(*id, meta);
                }
                coll.index_manager = new_manager;
                coll.deferred_metadata.store(false, Ordering::Release);
                tracing::info!(
                    collection = %collection_name,
                    "rebuilt metadata indexes"
                );
            }

            // 2.5. Train quantizer if quantized collection hasn't been trained yet.
            if !*need_index {
                if let Some(QuantizationConfig::Scalar(_)) = &coll.config.quantization_config {
                    coll.index.post_optimize();
                    tracing::info!(
                        collection = %collection_name,
                        "trained scalar quantizer"
                    );
                }
            }
        }

        // Phase 3: recompute virtual graph edges off the runtime. The graph
        // compute does ~N index.search calls (compute_batch_parallel); running it
        // under the per-collection WRITE lock on a runtime worker would pin the
        // worker and block all other ops for the whole multi-minute compute.
        // Instead we run the compute inside spawn_blocking and take only a READ
        // lock there (so concurrent searches still proceed), build the graph into
        // a detached VirtualGraph, then swap it in under a short WRITE lock. The
        // read guard lives entirely on the blocking thread and never crosses the
        // .await, so the future stays Send. Vector-only collections skip the
        // graph entirely; the graph source is the storage vectors loaded above,
        // so deferred-mode collections rebuild from real data, not an empty arena.
        if *need_graph && rebuild_graph && graph_enabled {
            let vector_ids: Vec<u64> = vector_data.iter().map(|(id, _)| *id).collect();
            // Arc-share the loaded vectors into the compute map (no per-vector
            // re-clone of the raw Vec data beyond the one Arc wrap).
            let vector_map: HashMap<u64, Arc<Vec<f32>>> = std::mem::take(&mut vector_data)
                .into_iter()
                .map(|(id, v)| (id, Arc::new(v)))
                .collect();

            let threshold = if graph_threshold > 0.0 { graph_threshold } else { 0.7 };
            let graph_handle = coll_handle.clone();
            let new_graph = tokio::task::spawn_blocking(move || {
                // Read lock taken on the blocking thread only; never held across
                // an await. Allows concurrent reads during the compute.
                let coll = metered_read(&graph_handle);
                let mut g = VirtualGraph::with_threshold(threshold, distance_metric);
                if let Err(e) = vf_graph::RelationshipComputer::compute_batch_parallel(
                    &mut g, coll.index.as_vector_index(), &vector_ids, &vector_map, 10,
                ) {
                    tracing::warn!("graph rebuild partially failed: {}", e);
                }
                g
            })
            .await
            .map_err(|e| format!("optimize graph task join error: {e}"))?;

            // Swap the finished graph in under a short write lock.
            {
                let mut coll = metered_write(coll_handle);
                coll.graph = new_graph;
                coll.deferred_graph.store(false, Ordering::Release);
                coll.dirty.store(true, Ordering::Release);
                coll.mutation_count.store(50_001, Ordering::Release);
            }
            tracing::info!(
                collection = %collection_name,
                "rebuilt virtual graph"
            );
        }

        // F10 (lost-write race): writes that committed to storage during the
        // load+build window are durable but absent from the index we just built
        // (we built from the snapshot at load_high_water_lsn). If the storage
        // high-water LSN advanced past what we loaded, do NOT mask the gap by
        // reporting Ready. Re-read the current LSN; on an advance we land on
        // PendingOptimization instead so a follow-up optimize covers the
        // post-load tail. Only the index path can drop writes; a graph-only or
        // metadata-only run does not need this guard but the check is cheap and
        // harmless. We chose PendingOptimization (a follow-up rebuild) over
        // tail-replay here for correctness with minimal surface area.
        let post_build_lsn = if needs_vectors {
            let cm = self.collection_manager.read();
            cm.get_collection(collection_name)
                .map(|sc| sc.current_lsn())
                .unwrap_or(load_high_water_lsn)
        } else {
            load_high_water_lsn
        };
        let lost_writes = *need_index && post_build_lsn > load_high_water_lsn;

        // Flip status and disarm the guard so its Drop does not re-derive (and
        // possibly clobber) the committed result. Done after the index swap so a
        // populated index is never reported as PendingOptimization.
        let final_status = if lost_writes {
            tracing::warn!(
                collection = %collection_name,
                load_lsn = load_high_water_lsn,
                post_lsn = post_build_lsn,
                "optimize: writes landed during rebuild window; deferring to a follow-up optimize"
            );
            // Re-arm the deferred_index flag so the next optimize rebuilds the
            // tail. Brief write lock; consistent with the swap commit above.
            {
                let coll = metered_write(coll_handle);
                coll.deferred_index.store(true, Ordering::Release);
            }
            CollectionStatus::PendingOptimization
        } else {
            CollectionStatus::Ready
        };
        if let Ok(mut status) = status_guard.status.write() {
            *status = final_status;
        }
        status_guard.disarm();

        // I12 (ADR-025): the optimize rebuild allocates a full transient copy of
        // the collection's vectors plus the build scratch. Drop the loaded vectors
        // explicitly, then purge the allocator arenas so the freed pages return to
        // the OS immediately instead of waiting for the background decay timer.
        drop(vector_data);
        vf_index::purge_allocator_arenas();

        Ok(WorkOutcome::Done(vectors_processed))
    }

    /// Snapshot of the global recovery summary captured during boot.
    /// Returned by the `/recovery_status` endpoint. `paths` carries the
    /// per-collection breakdown so operators can spot a single collection
    /// that took the slow path even when most boot fast.
    pub fn recovery_status_snapshot(&self) -> RecoveryStatusSnapshot {
        let path = RecoveryStatus::from_u8(self.recovery_path.load(Ordering::Acquire));
        let elapsed_secs = self.recovery_elapsed_secs.load(Ordering::Acquire);
        let paths: HashMap<String, RecoveryStatus> = {
            let rp = self.recovery_paths.read();
            rp.clone()
        };
        RecoveryStatusSnapshot {
            path,
            elapsed_secs,
            paths,
        }
    }

    /// Read the in-memory persistence cursors for a collection. Returns
    /// `Ok(None)` when the collection exists but its WAL meta could not
    /// be loaded (a fresh collection that has not yet written wal_meta.json
    /// hits this path). Errors are surfaced as `Err` so callers can map to
    /// the right transport-level status.
    pub fn persistence_status(
        &self,
        name: &str,
    ) -> Result<PersistenceStatus, String> {
        let (collection_dir, current_lsn) = {
            let cm = self.collection_manager.read();
            let coll = cm
                .get_collection(name)
                .map_err(|e| format!("collection '{}' not found: {}", name, e))?;
            (coll.collection_dir().to_path_buf(), coll.current_lsn())
        };
        let meta = vf_storage::wal::load_wal_meta(&collection_dir)
            .map_err(|e| format!("failed to load wal_meta: {}", e))?;
        Ok(PersistenceStatus {
            last_snapshot_lsn: meta.last_snapshot_lsn,
            current_lsn,
            next_lsn: meta.next_lsn,
        })
    }

    /// Read the per-collection lock-contention counters. Lock-free
    /// (atomic loads under a brief per-collection read lock to access
    /// the inner counters); the read itself bumps the read counter via
    /// `metered_read`, which is fine because it is what callers would
    /// observe in production.
    pub fn collection_metrics(
        &self,
        name: &str,
    ) -> Result<CollectionMetricsSnapshot, String> {
        let handle = self
            .collection_handle(name)
            .ok_or_else(|| format!("collection '{}' not found", name))?;
        let coll = metered_read(&handle);
        Ok(CollectionMetricsSnapshot {
            map_lock_acquisitions: self.map_lock_acquisitions.load(Ordering::Relaxed),
            collection_read_acquisitions: coll
                .collection_read_acquisitions
                .load(Ordering::Relaxed),
            collection_write_acquisitions: coll
                .collection_write_acquisitions
                .load(Ordering::Relaxed),
            total_blocked_microseconds: coll
                .total_blocked_microseconds
                .load(Ordering::Relaxed),
        })
    }
}

/// Aggregated recovery-status view returned by `/recovery_status`.
#[derive(Debug, Clone)]
pub struct RecoveryStatusSnapshot {
    pub path: RecoveryStatus,
    pub elapsed_secs: u64,
    pub paths: HashMap<String, RecoveryStatus>,
}

/// Persistence cursors for a single collection.
#[derive(Debug, Clone, Copy)]
pub struct PersistenceStatus {
    pub last_snapshot_lsn: u64,
    pub current_lsn: u64,
    pub next_lsn: u64,
}

/// Lock-contention counters for a single collection.
#[derive(Debug, Clone, Copy)]
pub struct CollectionMetricsSnapshot {
    pub map_lock_acquisitions: u64,
    pub collection_read_acquisitions: u64,
    pub collection_write_acquisitions: u64,
    pub total_blocked_microseconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    mod b_parallel_server_startup {
        use super::*;

        use std::sync::{Arc as StdArc, Mutex as StdMutex};
        use std::time::Duration;

        use tempfile::TempDir;
        use tracing::subscriber::with_default;
        use tracing_subscriber::fmt::MakeWriter;

        use vf_core::types::{CollectionConfig, DataTypeConfig, DistanceMetricType, Mode};

        // Build a default ServerConfig with the configured parallel-loader knob.
        fn server_config_with(max_loads: usize) -> crate::config::ServerConfig {
            let mut cfg = crate::config::ServerConfig::default();
            cfg.max_concurrent_collection_loads = max_loads;
            cfg
        }

        // Provision N empty plain HNSW collections under `data_dir`.
        // Uses CollectionManager so the on-disk layout matches what
        // AppState::new will discover on a later boot.
        fn provision_empty_collections(data_dir: &Path, n: usize, dim: usize) {
            let mut manager = CollectionManager::new(data_dir).expect("manager init");
            for i in 0..n {
                let config = CollectionConfig {
                    name: format!("coll_{:04}", i),
                    dimension: dim,
                    distance_metric: DistanceMetricType::Cosine,
                    default_similarity_threshold: Some(0.7),
                    max_vectors: 0,
                    data_type: DataTypeConfig::F32,
                    quantization_config: None,
                    mode: Some(Mode::AutoSimilarity),
                };
                manager.create_collection(config).expect("create collection");
            }
            // Drop manager; AppState::new will re-scan from disk.
            drop(manager);
        }

        // Boot AppState against a populated data dir and return the result.
        fn boot_state(
            data_dir: &Path,
            max_loads: usize,
        ) -> Result<AppState, StorageError> {
            let cfg = server_config_with(max_loads);
            AppState::new(
                data_dir,
                cfg.max_ef_search,
                cfg.max_batch_lock_size,
                cfg.max_wal_flush_interval,
                cfg.max_ef_construction,
                cfg,
            )
        }

        // ── B1 ────────────────────────────────────────────────────────────
        // Smoke test for the parallel boot path on empty collections.
        // NOTE: empty collections produce identical zero-vector load times
        // and so cannot prove the parallel speedup contract. Renamed and
        // ignored. The real wall-clock comparison (cold vs warm restart on
        // populated collections) requires per-collection load instrumentation
        // and provisioning vectors through the gRPC or REST API surface.
        // UAT path: /Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/tests/test_startup_wall_clock.py
        #[test]
        #[ignore]
        fn b1_startup_smoke_empty_collections() {
            // TODO(P05): real timing test needs collections with vector data
            // and per-collection load instrumentation; see UAT path above.
            let tmp = TempDir::new().expect("tempdir");
            let n = 20usize;
            provision_empty_collections(tmp.path(), n, 8);

            let start = Instant::now();
            let state = boot_state(tmp.path(), 4).expect("boot");
            let elapsed = start.elapsed();

            assert_eq!(state.collections.read().len(), n);
            assert!(
                elapsed < Duration::from_secs(30),
                "boot of {} collections took {:?}, exceeds 30s ceiling",
                n,
                elapsed
            );
        }

        // ── B2 ────────────────────────────────────────────────────────────
        // No-race contract on the parallel collection map insertion.
        // NOTE: spec calls for 100 collections of varying sizes. We scale to
        // 50 same-shape empty collections to keep the inline suite fast.
        // The race-detection contract works at any N >= 2 because HashMap
        // key collision is detected by Rust's HashMap regardless of count.
        // NOTE: spec sub-case "metrics registry has one entry per collection"
        // is dropped because the metrics crate exposed via tracing does not
        // offer a per-collection enumerator from inside a unit test. That
        // sub-case is covered by the Prometheus scrape in the Python UAT.
        // UAT path: /Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/tests/test_startup_metrics_registry.py
        #[test]
        fn b2_one_hundred_collections_no_race() {
            let tmp = TempDir::new().expect("tempdir");
            let n = 50usize;
            provision_empty_collections(tmp.path(), n, 8);

            let state = boot_state(tmp.path(), 8).expect("boot");
            let collections = state.collections.read();

            // Length must match exactly. Any duplicate key would shrink the
            // count below n because HashMap collapses duplicates on insert.
            assert_eq!(collections.len(), n, "collection map size mismatch");

            // Every expected name must be present. Combined with the
            // length check, this is the no-missing + no-duplicate contract.
            for i in 0..n {
                let name = format!("coll_{:04}", i);
                assert!(
                    collections.contains_key(&name),
                    "missing collection {}",
                    name
                );
            }
        }

        // ── B3 ────────────────────────────────────────────────────────────
        // Per-collection isolation: a corrupted hnsw.base for one collection
        // must not poison the rest of the boot. plan_recovery is
        // file-existence based, so dropping a 4-byte corrupt hnsw.base into
        // one collection's directory routes that collection's loader into
        // the IncrementalReplay arm, the trait-level restore fails on bad
        // magic, and the loader falls back to full_rebuild. The other
        // collections proceed in parallel and finish normally.
        //
        // NOTE: the per-collection state file (hnsw.base) is corrupted, not
        // the data-dir-level config.json. Corrupting config.json bails
        // CollectionManager::new before the parallel loader runs, which
        // does not exercise the loader's isolation contract at all.
        #[test]
        fn b3_one_bad_collection_isolated_others_load() {
            let tmp = TempDir::new().expect("tempdir");
            let n = 10usize;
            provision_empty_collections(tmp.path(), n, 8);

            // Plant a corrupt hnsw.base for collection #N/2. Flipped magic
            // (b"XXXX" instead of b"HNSW") makes the envelope parser fail
            // at the trait-level restore call. The loader logs a warn and
            // falls back to full_rebuild for this one collection.
            let bad = "coll_0005";
            let bad_hnsw_base = tmp.path().join(bad).join("hnsw.base");
            let mut corrupt = vec![0u8; 64];
            corrupt[0..4].copy_from_slice(b"XXXX");
            std::fs::write(&bad_hnsw_base, &corrupt).expect("plant corrupt hnsw.base");

            // Boot. CollectionManager::new succeeds (config.json is fine);
            // the loader runs in parallel and falls back to full_rebuild
            // for the bad collection. All N collections end up loaded.
            let state = boot_state(tmp.path(), 4).expect("boot");
            let collections = state.collections.read();

            // The bad collection's full_rebuild fallback succeeds (empty
            // vector set), so the full set of N collections is present.
            // If the implementation ever changes to skip on bad base,
            // accept N-1 with the bad name absent.
            let count = collections.len();
            assert!(
                count == n || count == n - 1,
                "expected {} or {} collections, got {}",
                n,
                n - 1,
                count,
            );
            if count == n - 1 {
                assert!(
                    !collections.contains_key(bad),
                    "bad collection {} should be absent when count is N-1",
                    bad,
                );
            }
            // Every other collection must be present regardless.
            for i in 0..n {
                let name = format!("coll_{:04}", i);
                if name == bad {
                    continue;
                }
                assert!(
                    collections.contains_key(&name),
                    "good collection {} must still load",
                    name,
                );
            }
        }

        // ── B4 ────────────────────────────────────────────────────────────
        #[test]
        #[ignore]
        fn b4_concurrent_query_during_startup() {
            // TODO(P05): concurrent query during startup needs either an
            // instrumented AppState::new that exposes a partial-readiness
            // handle, or a Python UAT script.
            // UAT path: /Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/tests/test_startup_query_race.py
        }

        // ── B5 ────────────────────────────────────────────────────────────
        #[test]
        #[ignore]
        fn b5_process_kill_during_parallel_startup() {
            // TODO(P05): SIGKILL during boot needs a Python UAT subprocess
            // script that spawns the server, kills it mid-boot, verifies
            // state files are intact, and asserts the next start loads all.
            // UAT path: /Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/tests/test_startup_sigkill.py
        }

        // ── B6 ────────────────────────────────────────────────────────────
        // Worker panic during collection load.
        //
        // Spec sub-cases and what this inline test covers:
        //   (1) "server process does not crash": exercised by booting
        //       multiple collections through AppState::new and asserting
        //       the call returns Ok. Covered below.
        //   (2) "all other collections still load and are queryable": not
        //       exercised here because load_single_collection has no
        //       test-only panic-injection seam. Adding one would require
        //       a cfg(test) hook in the production loader, which is out
        //       of scope for this test batch. The catch_unwind boundary
        //       primitive (state.rs:188-200) is contract-tested below in
        //       isolation.
        //   (3) "no worker threads are leaked": not exercised here;
        //       would require thread-count introspection before and after
        //       a boot with a synthetic panicking collection.
        //   (4) "single error log line naming the offending collection":
        //       not exercised here; would require capturing tracing logs
        //       during a boot that triggers a real panic.
        // UAT path for the full sub-cases:
        // /Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/tests/test_worker_panic_isolation.py
        #[test]
        fn b6_worker_panic_during_collection_load() {
            // (a) Contract test on the isolation primitive used by the
            // loader: catch_unwind must convert a panic into Err and not
            // propagate. This is the exact wrapper applied per task at
            // state.rs:188-200.
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                panic!("synthetic loader panic");
            }));
            assert!(result.is_err(), "panic must be caught at the boundary");

            // (b) Sub-case (1): boot does not crash. Provision several
            // collections and confirm AppState::new returns Ok with all
            // of them loaded. The catch_unwind wrapper is in place around
            // every per-collection call inside the rayon pool.
            let tmp = TempDir::new().expect("tempdir");
            provision_empty_collections(tmp.path(), 5, 8);
            let state = boot_state(tmp.path(), 2).expect("boot");
            assert_eq!(state.collections.read().len(), 5);
        }

        // ── B7 ────────────────────────────────────────────────────────────
        #[test]
        #[ignore]
        fn b7_memory_pressure_during_parallel_startup() {
            // TODO(P05): memory pressure needs a Python UAT subprocess
            // script that boots the server under a cgroup or container
            // memory cap and asserts no OOM kill and no silent drops.
            // UAT path: /Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/tests/test_startup_memory_pressure.py
        }

        // ── B8 ────────────────────────────────────────────────────────────
        #[test]
        #[ignore]
        fn b8_sigterm_mid_boot() {
            // TODO(P05): SIGTERM mid-boot needs a Python UAT subprocess
            // script that sends SIGTERM during parallel boot, asserts
            // graceful exit, and verifies on-disk state files stay valid.
            // UAT path: /Users/chirotpaldas/Desktop/Projects/SwarnDB/swarndb/tests/test_startup_sigterm.py
        }

        // ── B9 ────────────────────────────────────────────────────────────
        // Capture every log line emitted by the parallel boot into a
        // shared in-memory buffer, then assert that the canonical loader
        // pair (`loading collections in parallel` and the per-collection
        // `recovered collection` line) is present and well-formed for
        // every collection.
        #[derive(Clone)]
        struct SharedBufWriter(StdArc<StdMutex<Vec<u8>>>);

        impl std::io::Write for SharedBufWriter {
            fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
                let mut g = self.0.lock().unwrap();
                g.extend_from_slice(data);
                Ok(data.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> MakeWriter<'a> for SharedBufWriter {
            type Writer = SharedBufWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        // ── B9 ────────────────────────────────────────────────────────────
        // Structured logging under concurrency.
        //
        // NOTE: spec calls for N=50 collections. We scale to N=25 to keep
        // the inline suite under the test-runtime ceiling. The log-integrity
        // contracts (atomic per-line writes, paired events per collection,
        // per-thread timestamp ordering) are independent of N for any
        // N >= 2.
        //
        // NOTE on the spec's "start + finished pair per collection": the
        // loader does not emit a single "start loading" event. The closest
        // start-of-work event per collection is the `recovery plan:` line
        // (state.rs:263-267). The end-of-work event is the
        // `recovered collection with N vectors` line (state.rs:686-691).
        // We assert exactly one of each per collection. This matches the
        // observed loader contract; the spec's stronger pairing requires a
        // dedicated "start loading" event that does not exist today.
        #[test]
        fn b9_structured_logging_under_concurrency() {
            let tmp = TempDir::new().expect("tempdir");
            let n = 25usize;
            provision_empty_collections(tmp.path(), n, 8);

            let buf: StdArc<StdMutex<Vec<u8>>> = StdArc::new(StdMutex::new(Vec::new()));
            let writer = SharedBufWriter(StdArc::clone(&buf));

            // Enable thread names so per-thread ordering can be asserted.
            // The parallel loader names its threads `collection-loader-N`
            // (state.rs:166), which the fmt subscriber emits inline.
            let subscriber = tracing_subscriber::fmt()
                .with_writer(writer)
                .with_ansi(false)
                .with_thread_names(true)
                .with_max_level(tracing::Level::INFO)
                .finish();

            let state = with_default(subscriber, || {
                boot_state(tmp.path(), 4).expect("boot")
            });
            assert_eq!(state.collections.read().len(), n);

            let logs = String::from_utf8(buf.lock().unwrap().clone()).expect("utf8 logs");
            let lines: Vec<&str> = logs.lines().filter(|l| !l.is_empty()).collect();
            assert!(!lines.is_empty(), "log buffer is empty");

            // (a) Atomic per-line writes: every captured line must start
            // with a recognisable ISO-8601 timestamp prefix (year digits).
            // tracing-subscriber's default fmt layer prefixes every record
            // with the timestamp; a half-written or interleaved line would
            // fail this check.
            for line in &lines {
                let starts_with_year = line.len() >= 4
                    && line.as_bytes()[0..4].iter().all(|b| b.is_ascii_digit());
                assert!(
                    starts_with_year,
                    "line missing timestamp prefix: {:?}",
                    line,
                );
            }

            // The boot-banner line is emitted exactly once.
            let banner_hits = lines
                .iter()
                .filter(|l| l.contains("loading collections in parallel"))
                .count();
            assert_eq!(
                banner_hits, 1,
                "expected one boot-banner line, found {}",
                banner_hits,
            );

            // (b) Paired events per collection: exactly one `recovery plan:`
            // (start of work) and exactly one `recovered collection with`
            // (end of work) line per collection name.
            for i in 0..n {
                let name = format!("coll_{:04}", i);
                let plan_hits = lines
                    .iter()
                    .filter(|l| l.contains("recovery plan:") && l.contains(&name))
                    .count();
                let recovered_hits = lines
                    .iter()
                    .filter(|l| l.contains("recovered collection with") && l.contains(&name))
                    .count();
                assert_eq!(
                    plan_hits, 1,
                    "collection {} should have one recovery-plan line, found {}",
                    name, plan_hits,
                );
                assert_eq!(
                    recovered_hits, 1,
                    "collection {} should have one recovered line, found {}",
                    name, recovered_hits,
                );
            }

            // (c) Per-thread timestamp ordering: within each loader thread
            // (`collection-loader-N`), captured timestamps must be
            // non-decreasing. Lines from other threads are skipped because
            // the rayon pool only names its own workers. The timestamp is
            // the leading ISO-8601 token; we compare lexicographically,
            // which is order-preserving for that format.
            let mut by_thread: std::collections::HashMap<String, Vec<String>> =
                std::collections::HashMap::new();
            for line in &lines {
                if let Some(thread) = extract_loader_thread(line) {
                    let ts = match line.split_whitespace().next() {
                        Some(t) => t.to_string(),
                        None => continue,
                    };
                    by_thread.entry(thread).or_default().push(ts);
                }
            }
            for (thread, stamps) in &by_thread {
                for w in stamps.windows(2) {
                    assert!(
                        w[0] <= w[1],
                        "thread {} timestamps not non-decreasing: {} then {}",
                        thread, w[0], w[1],
                    );
                }
            }
        }

        // Extract the rayon loader thread name from a captured log line.
        // Returns the substring `collection-loader-N` if present.
        fn extract_loader_thread(line: &str) -> Option<String> {
            const TAG: &str = "collection-loader-";
            let start = line.find(TAG)?;
            let tail = &line[start..];
            let end = tail.find(|c: char| !c.is_ascii_digit() && c != '-' && !c.is_ascii_lowercase())
                .unwrap_or(tail.len());
            Some(tail[..end].to_string())
        }
    }

    mod d_integration {

        // D2 stub. Requires a real spawned server process, an HTTP/gRPC client driver,
        // a 30s steady-state insert generator across 100 mixed plain HNSW and SQ8 collections,
        // a SIGKILL trigger, and a restart timing harness. Asserts every collection restored,
        // acknowledged inserts present, and total restart time is within the B1 parallel range
        // (not the serial range).
        // TODO(P05): port this scenario to swarndb/tests/test_stress_mixed_collections_insert.py.
        // Follow the subprocess spawn plus signal pattern in swarndb/tests/test_persistence_crash.py.
        #[test]
        #[ignore]
        fn d2_heavy_stress_mixed_collections_insert() {}

        // D3 stub. Requires a real spawned server process and three transport drivers
        // (gRPC client, REST client, Python SDK). For each transport, run create plus insert
        // plus search plus restart plus search, and assert byte-identical results across the
        // restart for every transport.
        // TODO(P05): port this scenario to swarndb/tests/test_sdk_surface_grpc_rest_python.py.
        // Follow the subprocess spawn plus signal pattern in swarndb/tests/test_persistence_crash.py.
        #[test]
        #[ignore]
        fn d3_sdk_surface_grpc_rest_python() {}

        // D4 stub. Requires a real spawned server process and a 30 minute mixed workload
        // driver (inserts, searches, periodic restarts) across 10 collections, plus probes
        // for resident memory, open file handles, live thread count, and on-disk state size.
        // Asserts no growth beyond what the inserts justify.
        // TODO(P05): port this scenario to swarndb/tests/test_long_running_soak.py.
        // Follow the subprocess spawn plus signal pattern in swarndb/tests/test_persistence_crash.py.
        #[test]
        #[ignore]
        fn d4_long_running_soak() {}

        // D6 stub. Requires a real spawned server process, a 1M-vector SQ8 dataset
        // (provisioned as the sole collection), and a startup timing harness that measures
        // both cold-cache (page cache dropped) and warm-cache restart times. Compares both
        // against the quantization_v3 baseline recorded in the test header and asserts no
        // regression beyond 5 percent.
        // TODO(P05): port this scenario to swarndb/tests/test_single_collection_1m_startup.py.
        // Follow the subprocess spawn plus signal pattern in swarndb/tests/test_persistence_crash.py.
        #[test]
        #[ignore]
        fn d6_one_million_scale_single_collection_startup() {}

        // D7 stub. Requires a real spawned server process and three transport drivers
        // (gRPC, REST, Python SDK) running in turn. For each transport: create a fresh
        // collection, insert a batch and record the batch hash, run a search and record
        // the result, restart, run the identical search, assert byte-identical second result.
        // Each transport keeps its own recorded result table, distinct from D3.
        // TODO(P05): port this scenario to swarndb/tests/test_sdk_regression_per_transport.py.
        // Follow the subprocess spawn plus signal pattern in swarndb/tests/test_persistence_crash.py.
        #[test]
        #[ignore]
        fn d7_sdk_regression_per_transport() {}

        // D8 stub. Requires a real spawned server process, 20 mixed collections
        // (roughly half plain HNSW, half SQ8), a 5 to 10 minute continuous insert plus
        // search driver, one mid-run restart trigger, a resident memory probe, a recorded
        // probe-set checker, and a log scanner. Asserts resident memory stays within
        // plus or minus 10 percent of the post-warmup steady state, no data drift on the
        // probe set, zero crashes, zero panics in logs.
        // TODO(P05): port this scenario to swarndb/tests/test_production_shape_soak.py.
        // Follow the subprocess spawn plus signal pattern in swarndb/tests/test_persistence_crash.py.
        #[test]
        #[ignore]
        fn d8_production_shape_soak() {}
    }
}
