// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Memory-mapped file I/O utilities for SwarnDB storage engine.
//!
//! Provides free functions for creating, opening, syncing, and resizing
//! memory-mapped files using the `memmap2` crate.
//!
//! [`SafeMmap`] wraps a memory-mapped region with a `RwLock` to enforce
//! safe concurrent access: multiple readers OR one exclusive writer.

use std::fs::OpenOptions;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;

use crate::error::{StorageError, StorageResult};
use crate::FilePermissionConfig;

// Re-export memmap2 types for convenience.
pub use memmap2::{Mmap, MmapMut};

/// Default permission config for backward compatibility.
static DEFAULT_PERM_CONFIG: std::sync::LazyLock<FilePermissionConfig> =
    std::sync::LazyLock::new(FilePermissionConfig::default);

/// Create a new file at `path` with the given `size` in bytes and return a mutable memory map.
///
/// The file must not already exist (`create_new` semantics). The file is pre-allocated to
/// `size` bytes (filled with zeros) before being memory-mapped.
/// Uses default file permissions (0o600). Use `create_with_perms` for custom permissions.
pub fn create(path: &Path, size: usize) -> StorageResult<MmapMut> {
    create_with_perms(path, size, &DEFAULT_PERM_CONFIG)
}

/// Create a new file with configurable permissions.
pub fn create_with_perms(path: &Path, size: usize, perms: &FilePermissionConfig) -> StorageResult<MmapMut> {
    // Task 280: Reject size=0.
    if size == 0 {
        return Err(StorageError::MmapFailed(
            "cannot create memory-mapped file with size 0".into(),
        ));
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(StorageError::Io)?;

    file.set_len(size as u64).map_err(StorageError::Io)?;

    // Set file permissions (configurable, defaults to 0o600 on Unix).
    perms.apply_file_permissions(path)?;

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

    // Task 286: Log a warning if truncation would discard data.
    let current_size = file.metadata().map_err(StorageError::Io)?.len();
    if (new_size as u64) < current_size {
        log::warn!(
            "mmap resize: truncating {:?} from {} to {} bytes — data beyond new size will be lost",
            path, current_size, new_size
        );
    }

    file.set_len(new_size as u64).map_err(StorageError::Io)?;

    // SAFETY: The caller must ensure the old mapping has been dropped before calling resize.
    unsafe { MmapMut::map_mut(&file).map_err(StorageError::Io) }
}

/// Return the size of the file at `path` in bytes.
pub fn file_size(path: &Path) -> StorageResult<u64> {
    let metadata = std::fs::metadata(path).map_err(StorageError::Io)?;
    Ok(metadata.len())
}

// ── SafeMmap ────────────────────────────────────────────────────────────────

/// A thread-safe wrapper around a memory-mapped file that uses a `RwLock` to
/// enforce concurrent-access safety: multiple readers OR one exclusive writer.
///
/// The `resize` operation takes an exclusive write lock, ensuring no readers
/// are accessing the mapping while it is being replaced.
pub struct SafeMmap {
    inner: RwLock<MmapMut>,
    path: PathBuf,
}

impl SafeMmap {
    /// Create a new `SafeMmap` by creating a new file at `path`.
    pub fn create_new(path: &Path, size: usize) -> StorageResult<Self> {
        let mmap = create(path, size)?;
        Ok(Self {
            inner: RwLock::new(mmap),
            path: path.to_path_buf(),
        })
    }

    /// Open an existing file as a read-write `SafeMmap`.
    pub fn open(path: &Path) -> StorageResult<Self> {
        let mmap = open_read_write(path)?;
        Ok(Self {
            inner: RwLock::new(mmap),
            path: path.to_path_buf(),
        })
    }

    /// Acquire a read lock and call `f` with a shared reference to the mmap bytes.
    ///
    /// Multiple readers can hold this lock concurrently.
    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        let guard = self.inner.read();
        f(guard.deref())
    }

    /// Acquire a write lock and call `f` with a mutable reference to the mmap bytes.
    ///
    /// This is exclusive: no other readers or writers can proceed.
    pub fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut MmapMut) -> R,
    {
        let mut guard = self.inner.write();
        f(&mut guard)
    }

    /// Resize the underlying file and replace the mmap under an exclusive lock.
    ///
    /// All concurrent readers/writers are blocked until the resize completes.
    pub fn resize(&self, new_size: usize) -> StorageResult<()> {
        let mut guard = self.inner.write();
        // Flush before dropping the old mapping.
        guard.flush().map_err(StorageError::Io)?;
        let new_mmap = resize(&self.path, new_size)?;
        *guard = new_mmap;
        Ok(())
    }

    /// Flush modified pages to disk under a read lock.
    pub fn sync(&self) -> StorageResult<()> {
        let guard = self.inner.read();
        guard.flush().map_err(StorageError::Io)
    }

    /// Returns the length of the mapped region.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Returns true if the mapped region is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
