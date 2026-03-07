// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Software prefetching hints for graph traversal.
//!
//! This module provides cross-platform prefetch utilities designed to reduce
//! cache misses during HNSW graph traversal and neighbor list scanning.
//! Prefetching neighbor vectors and adjacency lists ahead of time allows the
//! CPU to overlap memory fetches with computation, significantly improving
//! throughput on large-scale vector search workloads.
//!
//! Supported architectures:
//! - **x86_64**: Uses `_mm_prefetch` SSE intrinsics
//! - **aarch64**: Uses `PRFM` prefetch instructions via inline assembly
//! - **Other**: No-op fallbacks (safe, zero overhead)

use vf_core::types::VectorId;

/// Size of a CPU cache line in bytes (common across x86_64 and aarch64).
const CACHE_LINE_BYTES: usize = 64;

// ---------------------------------------------------------------------------
// x86_64 implementation
// ---------------------------------------------------------------------------

/// Prefetch data for read access with temporal locality (L1 cache hint).
///
/// Signals the processor that the data at `ptr` will be read soon and should
/// be brought into the closest cache level (L1). Use this for data that will
/// be accessed multiple times (e.g., the current candidate vector).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    // SAFETY: _mm_prefetch is a hint instruction. Even if the pointer is
    // invalid or null, the CPU will silently ignore the prefetch without
    // faulting. No memory is actually read by this instruction.
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
}

/// Prefetch data for read access, non-temporal (likely to be evicted soon).
///
/// Signals the processor that the data at `ptr` will be read once and is
/// unlikely to be reused. This avoids polluting the L1/L2 caches with data
/// that will not be needed again (e.g., neighbor lists being scanned linearly).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn prefetch_read_nt<T>(ptr: *const T) {
    // SAFETY: Same as prefetch_read - hint instruction, no actual memory access.
    unsafe {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_NTA);
    }
}

// ---------------------------------------------------------------------------
// aarch64 implementation
// ---------------------------------------------------------------------------

/// Prefetch data for read access with temporal locality (L1 cache hint).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    // SAFETY: The PRFM instruction is a prefetch hint. It does not fault on
    // invalid addresses; the CPU silently drops the request.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags),
        );
    }
}

/// Prefetch data for read access, non-temporal (likely to be evicted soon).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn prefetch_read_nt<T>(ptr: *const T) {
    // SAFETY: Same as prefetch_read - PRFM is a hint, no fault on bad addresses.
    unsafe {
        core::arch::asm!(
            "prfm pldl1strm, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, preserves_flags),
        );
    }
}

// ---------------------------------------------------------------------------
// Fallback (no-op) for unsupported architectures
// ---------------------------------------------------------------------------

/// Prefetch data for read access with temporal locality (no-op fallback).
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub fn prefetch_read<T>(_ptr: *const T) {
    // No prefetch support on this architecture; this is intentionally a no-op.
}

/// Prefetch data for read access, non-temporal (no-op fallback).
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub fn prefetch_read_nt<T>(_ptr: *const T) {
    // No prefetch support on this architecture; this is intentionally a no-op.
}

// ---------------------------------------------------------------------------
// Higher-level prefetch helpers for vector search
// ---------------------------------------------------------------------------

/// Prefetch a slice of `f32` data (e.g., a vector embedding).
///
/// Issues prefetch hints for enough cache lines to cover the entire slice.
/// A cache line holds 16 `f32` values (64 bytes / 4 bytes per f32).
///
/// This is intended for use when the next candidate vector is known ahead of
/// the distance computation, allowing the memory subsystem to start fetching
/// the vector data in parallel.
#[inline(always)]
pub fn prefetch_vector(data: &[f32]) {
    if data.is_empty() {
        return;
    }
    let f32s_per_cache_line = CACHE_LINE_BYTES / core::mem::size_of::<f32>(); // 16
    let ptr = data.as_ptr();
    let mut offset = 0usize;
    while offset < data.len() {
        // SAFETY: ptr.add(offset) stays within or one-past-end of the slice
        // allocation. prefetch_read is a hint and does not dereference.
        prefetch_read(unsafe { ptr.add(offset) });
        offset += f32s_per_cache_line;
    }
}

/// Prefetch a neighbor list (slice of `VectorId`).
///
/// Issues prefetch hints for the neighbor ID array so that the adjacency
/// data is available in cache when the search loop iterates over neighbors.
/// A cache line holds 8 `VectorId` values (64 bytes / 8 bytes per u64).
#[inline(always)]
pub fn prefetch_neighbors(neighbors: &[VectorId]) {
    if neighbors.is_empty() {
        return;
    }
    let ids_per_cache_line = CACHE_LINE_BYTES / core::mem::size_of::<VectorId>(); // 8
    let ptr = neighbors.as_ptr();
    let mut offset = 0usize;
    while offset < neighbors.len() {
        // SAFETY: ptr.add(offset) stays within or one-past-end of the slice
        // allocation. prefetch_read is a hint and does not dereference.
        prefetch_read(unsafe { ptr.add(offset) });
        offset += ids_per_cache_line;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_read_does_not_panic_on_null() {
        // Prefetch hints must be safe to call with any pointer, including null.
        prefetch_read(core::ptr::null::<u8>());
        prefetch_read_nt(core::ptr::null::<u8>());
    }

    #[test]
    fn prefetch_vector_empty_slice() {
        prefetch_vector(&[]);
    }

    #[test]
    fn prefetch_vector_small() {
        let data = vec![1.0f32; 4];
        prefetch_vector(&data);
    }

    #[test]
    fn prefetch_vector_large() {
        let data = vec![0.0f32; 768]; // typical embedding dimension
        prefetch_vector(&data);
    }

    #[test]
    fn prefetch_neighbors_empty() {
        prefetch_neighbors(&[]);
    }

    #[test]
    fn prefetch_neighbors_small() {
        let neighbors: Vec<VectorId> = vec![1, 2, 3, 4, 5];
        prefetch_neighbors(&neighbors);
    }
}
