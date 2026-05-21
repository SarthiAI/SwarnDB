use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use fs2::FileExt;

use crate::error::{StorageError, StorageResult};

/// Process-level file lock using flock().
/// Prevents multiple SwarnDB instances from using the same data directory.
pub struct ProcessLock {
    _file: File,
    path: PathBuf,
}

impl ProcessLock {
    /// Acquires an exclusive lock on `<data_dir>/lock`.
    /// Returns immediately with an error if the lock is held by another process.
    pub fn acquire(data_dir: &Path) -> StorageResult<Self> {
        fs::create_dir_all(data_dir)?;

        let lock_path = data_dir.join("lock");

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;

        file.try_lock_exclusive().map_err(|_| {
            StorageError::LockHeld(format!(
                "Another SwarnDB instance is using data directory: {}",
                data_dir.display()
            ))
        })?;

        Ok(Self {
            _file: file,
            path: lock_path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ProcessLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self._file);
    }
}
