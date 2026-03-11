// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

pub mod error;
pub mod format;
pub mod mmap;
pub mod wal;
pub mod segment;
pub mod collection;
pub mod recovery;
pub mod compaction;
pub mod disk_ann;
pub mod tiered;
pub mod restore;
pub mod backup;
mod util;

pub use error::{StorageError, StorageResult};
pub use format::{SegmentHeader, WalOp, SEGMENT_MAGIC, SEGMENT_VERSION};
pub use compaction::{CompactionOptions, CompactionResult, compact_segments, should_compact};
pub use restore::{RestoreOptions, RestoreResult, restore_backup};
pub use backup::{BackupManifest, BackupOptions, create_backup};

/// Configurable file and directory permissions for storage files.
///
/// Defaults to 0o600 for files and 0o700 for directories (owner-only access).
/// Override these for environments like Docker where admins need to inspect
/// data files with different user/group permissions.
#[derive(Clone, Debug)]
pub struct FilePermissionConfig {
    /// Unix file permission mode (e.g., 0o644 for owner rw, group/other read).
    pub file_mode: u32,
    /// Unix directory permission mode (e.g., 0o755 for owner rwx, group/other rx).
    pub dir_mode: u32,
}

impl Default for FilePermissionConfig {
    fn default() -> Self {
        Self {
            file_mode: 0o600,
            dir_mode: 0o700,
        }
    }
}

impl FilePermissionConfig {
    /// Apply the configured file permissions to a path (Unix only).
    #[cfg(unix)]
    pub fn apply_file_permissions(&self, path: &std::path::Path) -> StorageResult<()> {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(self.file_mode);
        std::fs::set_permissions(path, perms).map_err(StorageError::Io)
    }

    /// Apply the configured directory permissions to a path (Unix only).
    #[cfg(unix)]
    pub fn apply_dir_permissions(&self, path: &std::path::Path) -> StorageResult<()> {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(self.dir_mode);
        std::fs::set_permissions(path, perms).map_err(StorageError::Io)
    }

    /// No-op on non-Unix platforms.
    #[cfg(not(unix))]
    pub fn apply_file_permissions(&self, _path: &std::path::Path) -> StorageResult<()> {
        Ok(())
    }

    /// No-op on non-Unix platforms.
    #[cfg(not(unix))]
    pub fn apply_dir_permissions(&self, _path: &std::path::Path) -> StorageResult<()> {
        Ok(())
    }
}
