// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Memory-mapped file I/O utilities for SwarnDB storage engine.
//!
//! Provides free functions for creating, opening, syncing, and resizing
//! memory-mapped files using the `memmap2` crate.

use std::fs::OpenOptions;
use std::path::Path;

use crate::error::{StorageError, StorageResult};

// Re-export memmap2 types for convenience.
pub use memmap2::{Mmap, MmapMut};

/// Create a new file at `path` with the given `size` in bytes and return a mutable memory map.
///
/// The file must not already exist (`create_new` semantics). The file is pre-allocated to
/// `size` bytes (filled with zeros) before being memory-mapped.
pub fn create(path: &Path, size: usize) -> StorageResult<MmapMut> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(StorageError::Io)?;

    file.set_len(size as u64).map_err(StorageError::Io)?;

    // SAFETY: The file was just created and sized by us; no other process holds a mapping.
    unsafe { MmapMut::map_mut(&file).map_err(StorageError::Io) }
}

/// Open an existing file at `path` as a read-only memory map.
pub fn open_read(path: &Path) -> StorageResult<Mmap> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(StorageError::Io)?;

    // SAFETY: The caller is responsible for ensuring no concurrent writers invalidate
    // the mapped region while the `Mmap` is alive.
    unsafe { Mmap::map(&file).map_err(StorageError::Io) }
}

/// Open an existing file at `path` as a read-write memory map.
pub fn open_read_write(path: &Path) -> StorageResult<MmapMut> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(StorageError::Io)?;

    // SAFETY: The caller is responsible for ensuring exclusive mutable access.
    unsafe { MmapMut::map_mut(&file).map_err(StorageError::Io) }
}

/// Flush all modified pages of the mutable memory map to the underlying file.
pub fn sync(mmap: &MmapMut) -> StorageResult<()> {
    mmap.flush().map_err(StorageError::Io)
}

/// Resize the file at `path` to `new_size` bytes and return a fresh mutable memory map.
///
/// If `new_size` is larger than the current file size, the file is extended (new bytes
/// are zero-filled by the OS). If smaller, the file is truncated. The previous memory
/// map (if any) should be dropped by the caller before calling this function.
pub fn resize(path: &Path, new_size: usize) -> StorageResult<MmapMut> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(StorageError::Io)?;

    file.set_len(new_size as u64).map_err(StorageError::Io)?;

    // SAFETY: The caller must ensure the old mapping has been dropped before calling resize.
    unsafe { MmapMut::map_mut(&file).map_err(StorageError::Io) }
}

/// Return the size of the file at `path` in bytes.
pub fn file_size(path: &Path) -> StorageResult<u64> {
    let metadata = std::fs::metadata(path).map_err(StorageError::Io)?;
    Ok(metadata.len())
}
