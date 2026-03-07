// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

use thiserror::Error;

/// Convenience alias for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;

/// Comprehensive error type for the SwarnDB storage engine.
#[derive(Debug, Error)]
pub enum StorageError {
    // ── IO ───────────────────────────────────────────────────────────────
    /// Wraps a standard I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    // ── Corruption ──────────────────────────────────────────────────────
    /// File magic bytes do not match the expected value.
    #[error("bad magic: expected {expected:?}, found {found:?}")]
    BadMagic { expected: [u8; 4], found: [u8; 4] },

    /// CRC-32 checksum does not match.
    #[error("checksum mismatch: expected {expected:#010x}, computed {computed:#010x}")]
    ChecksumMismatch { expected: u32, computed: u32 },

    /// Read fewer bytes than expected (file truncated or corrupted).
    #[error("truncated data: expected {expected} bytes, got {actual}")]
    TruncatedData { expected: usize, actual: usize },

    // ── WAL ─────────────────────────────────────────────────────────────
    /// A WAL write could not be completed.
    #[error("WAL write failed: {0}")]
    WalWriteFailed(String),

    /// WAL log rotation failed.
    #[error("WAL rotation failed: {0}")]
    WalRotationFailed(String),

    /// A WAL entry could not be parsed.
    #[error("WAL invalid entry: {0}")]
    WalInvalidEntry(String),

    // ── Segment ─────────────────────────────────────────────────────────
    /// Segment header is malformed.
    #[error("invalid segment header: {0}")]
    SegmentInvalidHeader(String),

    /// Segment was created with a different vector dimension.
    #[error("segment dimension mismatch: expected {expected}, got {actual}")]
    SegmentDimensionMismatch { expected: u32, actual: u32 },

    /// Referenced segment file does not exist.
    #[error("segment not found: {0}")]
    SegmentNotFound(String),

    // ── Collection ──────────────────────────────────────────────────────
    /// Attempted to create a collection that already exists.
    #[error("collection already exists: {0}")]
    CollectionAlreadyExists(String),

    /// Referenced collection does not exist.
    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    /// Dropping a collection failed.
    #[error("collection drop failed: {0}")]
    CollectionDropFailed(String),

    // ── Serialization ───────────────────────────────────────────────────
    /// Serialization / deserialization failure (e.g. bincode).
    #[error("serialization error: {0}")]
    Serialization(String),

    // ── Mmap ────────────────────────────────────────────────────────────
    /// Memory-mapping a file failed.
    #[error("mmap failed: {0}")]
    MmapFailed(String),

    /// Flushing a memory-mapped region to disk failed.
    #[error("mmap sync failed: {0}")]
    MmapSyncFailed(String),

    /// Resizing a memory-mapped file failed.
    #[error("mmap resize failed: {0}")]
    MmapResizeFailed(String),
}

impl From<bincode::Error> for StorageError {
    fn from(err: bincode::Error) -> Self {
        StorageError::Serialization(err.to_string())
    }
}
