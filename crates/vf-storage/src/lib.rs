// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

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
