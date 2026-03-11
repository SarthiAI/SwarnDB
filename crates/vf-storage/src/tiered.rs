// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Tiered vector storage: hot (RAM), warm (mmap), cold (quantized RAM + disk).
//!
//! Vectors are assigned to tiers based on access patterns. Frequently accessed
//! vectors are promoted to hotter tiers; infrequently accessed ones are demoted.

use std::collections::HashMap;

use dashmap::DashMap;
use parking_lot::Mutex;
use vf_core::types::VectorId;

use crate::disk_ann::DiskAnnStore;
use crate::error::{StorageError, StorageResult};
use crate::mmap;

/// Storage tier for a vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageTier {
    /// Full vectors in RAM (fastest, most memory).
    Hot,
    /// Full vectors on mmap (medium speed, less RAM).
    Warm,
    /// Quantized in RAM + full on disk (slowest for re-rank, least RAM).
    Cold,
}

/// Configuration for tier thresholds and capacities.
#[derive(Clone, Debug)]
pub struct TierConfig {
    /// Maximum number of vectors in the hot tier.
    pub hot_capacity: usize,
    /// Access count threshold to promote warm -> hot.
    pub promotion_threshold: u64,
    /// Access count below which hot -> warm demotion occurs.
    pub demotion_threshold: u64,
}

impl Default for TierConfig {
    fn default() -> Self {
        let hot_capacity = std::env::var("SWARNDB_TIERED_HOT_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100_000);

        Self {
            hot_capacity,
            promotion_threshold: 10,
            demotion_threshold: 2,
        }
    }
}

/// Statistics about the tiered storage.
#[derive(Clone, Debug)]
pub struct TierStats {
    /// Number of vectors in the hot tier.
    pub hot_count: usize,
    /// Number of vectors in the warm tier.
    pub warm_count: usize,
    /// Number of vectors in the cold tier.
    pub cold_count: usize,
    /// Approximate RAM usage for hot tier in bytes.
    pub hot_memory_bytes: usize,
    /// Approximate RAM usage for warm tier (offsets only) in bytes.
    pub warm_memory_bytes: usize,
    /// Approximate RAM usage for cold tier (quantized codes) in bytes.
    pub cold_memory_bytes: usize,
}

/// Tiered vector store with hot (RAM), warm (mmap), and cold (quantized + disk) tiers.
pub struct TieredVectorStore {
    dimension: usize,
    /// Hot tier: full vectors in memory.
    hot: DashMap<VectorId, Vec<f32>>,
    /// Warm tier: mmap'd full vectors.
    warm_mmap: Option<memmap2::Mmap>,
    /// Warm tier offset index: vector_id -> byte offset in warm mmap file.
    warm_offsets: HashMap<VectorId, u64>,
    /// Cold tier: quantized in RAM, full on disk.
    cold_store: Option<DiskAnnStore>,
    /// Tier assignment for each vector.
    tier_map: DashMap<VectorId, StorageTier>,
    /// Thresholds for auto-promotion/demotion.
    config: TierConfig,
    /// Access counters for promotion/demotion decisions.
    access_counts: DashMap<VectorId, u64>,
    /// Task 284: Mutex to serialize promote/demote operations.
    tier_transition_lock: Mutex<()>,
}

impl TieredVectorStore {
    /// Create a new tiered vector store for vectors of the given dimension.
    pub fn new(dimension: usize, config: TierConfig) -> Self {
        Self {
            dimension,
            hot: DashMap::new(),
            warm_mmap: None,
            warm_offsets: HashMap::new(),
            cold_store: None,
            tier_map: DashMap::new(),
            config,
            access_counts: DashMap::new(),
            tier_transition_lock: Mutex::new(()),
        }
    }

    /// Insert a vector into the hot tier.
    pub fn insert_hot(&self, id: VectorId, vector: Vec<f32>) {
        self.hot.insert(id, vector);
        self.tier_map.insert(id, StorageTier::Hot);
        // Initialize access count
        self.access_counts.entry(id).or_insert(0);
    }

    /// Get a vector from the appropriate tier, incrementing its access counter.
    pub fn get(&self, id: VectorId) -> Option<Vec<f32>> {
        // Check existence first, then increment
        let tier = match self.tier_map.get(&id) {
            Some(t) => *t,
            None => return None,
        };
        self.access_counts.entry(id).and_modify(|c| *c += 1).or_insert(1);

        match tier {
            StorageTier::Hot => {
                self.hot.get(&id).map(|v| v.value().clone())
            }
            StorageTier::Warm => self.get_warm_vector(id),
            StorageTier::Cold => {
                self.cold_store
                    .as_ref()
                    .and_then(|store| store.get_full_vector(id))
            }
        }
    }

    /// Batch get vectors from the appropriate tiers.
    pub fn get_batch(&self, ids: &[VectorId]) -> Vec<(VectorId, Vec<f32>)> {
        ids.iter()
            .filter_map(|&id| self.get(id).map(|vec| (id, vec)))
            .collect()
    }

    /// Promote a vector from warm/cold to hot tier.
    pub fn promote(&self, id: VectorId) -> StorageResult<()> {
        let _guard = self.tier_transition_lock.lock();
        let tier = self
            .tier_map
            .get(&id)
            .map(|t| *t)
            .ok_or_else(|| StorageError::SegmentNotFound(format!("vector {} not found", id)))?;

        match tier {
            StorageTier::Hot => {
                // Already hot, nothing to do
                Ok(())
            }
            StorageTier::Warm => {
                let vector = self.get_warm_vector(id).ok_or_else(|| {
                    StorageError::SegmentNotFound(format!(
                        "warm vector {} not found in mmap",
                        id
                    ))
                })?;
                self.hot.insert(id, vector);
                self.tier_map.insert(id, StorageTier::Hot);
                Ok(())
            }
            StorageTier::Cold => {
                let vector = self
                    .cold_store
                    .as_ref()
                    .and_then(|store| store.get_full_vector(id))
                    .ok_or_else(|| {
                        StorageError::SegmentNotFound(format!(
                            "cold vector {} not found on disk",
                            id
                        ))
                    })?;
                self.hot.insert(id, vector);
                self.tier_map.insert(id, StorageTier::Hot);
                Ok(())
            }
        }
    }

    /// Demote a vector from hot to warm tier.
    ///
    /// Note: the vector data remains accessible via the hot DashMap until the
    /// warm tier mmap is set up. In practice, demotion to warm requires a warm
    /// mmap file to already exist with the vector's data written to it. If no
    /// warm mmap is configured, the vector stays in hot.
    pub fn demote(&self, id: VectorId) -> StorageResult<()> {
        let _guard = self.tier_transition_lock.lock();
        let tier = self
            .tier_map
            .get(&id)
            .map(|t| *t)
            .ok_or_else(|| StorageError::SegmentNotFound(format!("vector {} not found", id)))?;

        match tier {
            StorageTier::Warm | StorageTier::Cold => {
                // Already demoted
                Ok(())
            }
            StorageTier::Hot => {
                // Verify the target tier can actually serve the vector before
                // updating tier_map.  A concurrent get() uses tier_map to
                // decide where to look, so the tier_map must only point to a
                // tier that already has the data available.
                //
                // Order: verify data readable in target → update tier_map → remove from hot.
                if self.get_warm_vector(id).is_some() {
                    self.tier_map.insert(id, StorageTier::Warm);
                    self.hot.remove(&id);
                    Ok(())
                } else if self
                    .cold_store
                    .as_ref()
                    .and_then(|s| s.get_full_vector(id))
                    .is_some()
                {
                    // Demote to cold only after verifying the full vector is retrievable.
                    self.tier_map.insert(id, StorageTier::Cold);
                    self.hot.remove(&id);
                    Ok(())
                } else {
                    // No lower tier can serve this vector; keep in hot
                    Ok(())
                }
            }
        }
    }

    /// Check access counts and auto-promote/demote vectors based on thresholds.
    ///
    /// Uses a local `hot_count` variable that is updated after each promotion/demotion
    /// to avoid TOCTOU issues from re-reading `self.hot.len()` between phases.
    pub fn maybe_rebalance(&self) {
        // Snapshot hot count once; track locally to avoid stale re-reads.
        let mut hot_count = self.hot.len();

        // Promote warm/cold vectors with high access counts
        let mut candidates_for_promotion: Vec<(VectorId, u64)> = Vec::new();
        for entry in self.access_counts.iter() {
            let id = *entry.key();
            let count = *entry.value();
            if let Some(tier) = self.tier_map.get(&id) {
                match *tier {
                    StorageTier::Warm | StorageTier::Cold => {
                        if count >= self.config.promotion_threshold {
                            candidates_for_promotion.push((id, count));
                        }
                    }
                    StorageTier::Hot => {}
                }
            }
        }

        // Sort by access count descending for priority promotion
        candidates_for_promotion.sort_by(|a, b| b.1.cmp(&a.1));

        let available_hot_slots = self.config.hot_capacity.saturating_sub(hot_count);
        for (id, _) in candidates_for_promotion.iter().take(available_hot_slots) {
            if self.promote(*id).is_ok() {
                hot_count += 1;
            }
        }

        // Demote hot vectors with low access counts if hot tier is over capacity
        if hot_count > self.config.hot_capacity {
            let mut hot_vectors: Vec<(VectorId, u64)> = Vec::new();
            for entry in self.tier_map.iter() {
                if *entry.value() == StorageTier::Hot {
                    let id = *entry.key();
                    let count = self
                        .access_counts
                        .get(&id)
                        .map(|c| *c)
                        .unwrap_or(0);
                    if count <= self.config.demotion_threshold {
                        hot_vectors.push((id, count));
                    }
                }
            }

            // Sort by access count ascending (demote least accessed first)
            hot_vectors.sort_by(|a, b| a.1.cmp(&b.1));

            let excess = hot_count.saturating_sub(self.config.hot_capacity);
            for (id, _) in hot_vectors.iter().take(excess) {
                if self.demote(*id).is_ok() {
                    hot_count = hot_count.saturating_sub(1);
                }
            }
        }

        // Decay access counts to prevent all vectors migrating to hot
        self.access_counts.iter_mut().for_each(|mut entry| {
            *entry.value_mut() /= 2;
        });
    }

    /// Return statistics about each tier.
    pub fn tier_stats(&self) -> TierStats {
        let mut hot_count = 0usize;
        let mut warm_count = 0usize;
        let mut cold_count = 0usize;

        for entry in self.tier_map.iter() {
            match *entry.value() {
                StorageTier::Hot => hot_count += 1,
                StorageTier::Warm => warm_count += 1,
                StorageTier::Cold => cold_count += 1,
            }
        }

        let hot_memory_bytes =
            hot_count * (self.dimension * std::mem::size_of::<f32>() + std::mem::size_of::<VectorId>());
        let warm_memory_bytes =
            warm_count * (std::mem::size_of::<VectorId>() + std::mem::size_of::<u64>());
        let cold_memory_bytes = self
            .cold_store
            .as_ref()
            .map_or(0, |s| s.ram_usage_bytes());

        TierStats {
            hot_count,
            warm_count,
            cold_count,
            hot_memory_bytes,
            warm_memory_bytes,
            cold_memory_bytes,
        }
    }

    /// Total number of vectors across all tiers.
    pub fn len(&self) -> usize {
        self.tier_map.len()
    }

    /// Returns true if no vectors are stored in any tier.
    pub fn is_empty(&self) -> bool {
        self.tier_map.is_empty()
    }

    /// Set the warm tier mmap and offset index.
    pub fn set_warm_store(
        &mut self,
        path: &std::path::Path,
        offsets: HashMap<VectorId, u64>,
    ) -> StorageResult<()> {
        let mmap_read = mmap::open_read(path)?;
        self.warm_mmap = Some(mmap_read);
        self.warm_offsets = offsets;
        Ok(())
    }

    /// Set the cold tier DiskAnnStore.
    pub fn set_cold_store(&mut self, store: DiskAnnStore) {
        self.cold_store = Some(store);
    }

    /// Returns the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    // ── private helpers ──────────────────────────────────────────────────

    /// Read a full-precision vector from the warm mmap at the recorded offset.
    fn get_warm_vector(&self, id: VectorId) -> Option<Vec<f32>> {
        let &byte_offset = self.warm_offsets.get(&id)?;
        let mmap = self.warm_mmap.as_ref()?;

        // Warm format: at byte_offset we have id(8) + f32*dim
        let data_start = byte_offset as usize + 8;
        let data_end = data_start + self.dimension * 4;

        if mmap.len() < data_end {
            return None;
        }

        let mut vector = Vec::with_capacity(self.dimension);
        for d in 0..self.dimension {
            let start = data_start + d * 4;
            let val =
                f32::from_le_bytes(mmap[start..start + 4].try_into().unwrap());
            vector.push(val);
        }

        Some(vector)
    }
}
