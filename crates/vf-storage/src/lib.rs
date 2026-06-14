// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

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
pub mod atomic_write;
pub mod file_lock;
pub mod bulk_checkpoint;
mod util;

pub use error::{StorageError, StorageResult};
pub use format::{SegmentHeader, WalOp, SEGMENT_MAGIC, SEGMENT_VERSION};
pub use compaction::{CompactionOptions, CompactionResult, compact_segments, should_compact};
pub use restore::{RestoreOptions, RestoreResult, restore_backup};
pub use backup::{BackupManifest, BackupOptions, create_backup};
