// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

// Bulk insert checkpoint storage.
// On-disk layout (40 bytes, little-endian):
//   0..8    magic "SWBKINS\0"
//   8..12   version u32
//  12..20   collection_id u64
//  20..28   last_completed_batch_idx u64
//  28..36   last_committed_lsn u64
//  36..40   crc32 over bytes 0..36

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

// Magic identifier, NUL terminated so hexdump stays printable.
pub const MAGIC: &[u8; 8] = b"SWBKINS\0";

// Initial format version.
pub const VERSION_V1: u32 = 1;

// Fixed serialized size on disk.
pub const CHECKPOINT_SIZE: usize = 40;

// Size of the CRC-covered prefix.
const PRE_CHECKSUM_LEN: usize = 36;

// Bulk insert checkpoint record persisted between batches.
#[derive(Clone, Debug)]
pub struct BulkCheckpoint {
    pub version: u32,
    pub collection_id: u64,
    pub last_completed_batch_idx: u64,
    pub last_committed_lsn: u64,
}

// Errors surfaced by checkpoint read/write/delete.
#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("bad magic in bulk insert checkpoint")]
    BadMagic,

    #[error("unsupported bulk insert checkpoint version: {0}")]
    UnsupportedVersion(u32),

    #[error("truncated bulk insert checkpoint: expected {expected} bytes, got {actual}")]
    Truncated { expected: usize, actual: usize },

    #[error("checksum mismatch in bulk insert checkpoint: expected {expected:#010x}, actual {actual:#010x}")]
    BadChecksum { expected: u32, actual: u32 },

    #[error("invalid bulk insert checkpoint: {0}")]
    Invalid(String),
}

impl BulkCheckpoint {
    // Build a fresh V1 checkpoint.
    pub fn new(collection_id: u64, last_batch_idx: u64, last_lsn: u64) -> Self {
        BulkCheckpoint {
            version: VERSION_V1,
            collection_id,
            last_completed_batch_idx: last_batch_idx,
            last_committed_lsn: last_lsn,
        }
    }

    // Pack the 36-byte CRC-covered prefix into the buffer.
    fn pack_pre_checksum(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= PRE_CHECKSUM_LEN);
        buf[0..8].copy_from_slice(MAGIC);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..20].copy_from_slice(&self.collection_id.to_le_bytes());
        buf[20..28].copy_from_slice(&self.last_completed_batch_idx.to_le_bytes());
        buf[28..36].copy_from_slice(&self.last_committed_lsn.to_le_bytes());
    }

    // Serialize into the fixed 40-byte on-disk form.
    fn encode(&self) -> [u8; CHECKPOINT_SIZE] {
        let mut buf = [0u8; CHECKPOINT_SIZE];
        self.pack_pre_checksum(&mut buf[..PRE_CHECKSUM_LEN]);
        let crc = crc32fast::hash(&buf[..PRE_CHECKSUM_LEN]);
        buf[PRE_CHECKSUM_LEN..CHECKPOINT_SIZE].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    // Write the checkpoint atomically: temp + rename.
    pub fn write_atomic(&self, path: &Path) -> Result<(), CheckpointError> {
        if self.version != VERSION_V1 {
            return Err(CheckpointError::UnsupportedVersion(self.version));
        }

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let tmp_path = tmp_sibling(path);

        // Clean stale temp from a prior crash.
        let _ = fs::remove_file(&tmp_path);

        let buf = self.encode();

        // atomic write: temp + rename
        {
            let mut file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            if let Err(e) = file.write_all(&buf) {
                let _ = fs::remove_file(&tmp_path);
                return Err(CheckpointError::Io(e));
            }
            if let Err(e) = file.sync_all() {
                let _ = fs::remove_file(&tmp_path);
                return Err(CheckpointError::Io(e));
            }
        }

        if let Err(e) = fs::rename(&tmp_path, path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(CheckpointError::Io(e));
        }

        // fsync parent directory so the rename is durable.
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Ok(dir) = fs::File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
        }

        Ok(())
    }

    // Read and validate the checkpoint from disk.
    pub fn read(path: &Path) -> Result<Self, CheckpointError> {
        let mut file = fs::File::open(path)?;
        let mut buf = [0u8; CHECKPOINT_SIZE];

        let mut read_so_far = 0usize;
        loop {
            match file.read(&mut buf[read_so_far..]) {
                Ok(0) => break,
                Ok(n) => {
                    read_so_far += n;
                    if read_so_far == CHECKPOINT_SIZE {
                        break;
                    }
                }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(CheckpointError::Io(e)),
            }
        }

        if read_so_far != CHECKPOINT_SIZE {
            return Err(CheckpointError::Truncated {
                expected: CHECKPOINT_SIZE,
                actual: read_so_far,
            });
        }

        // Reject any trailing bytes beyond the fixed size.
        let mut extra = [0u8; 1];
        if let Ok(n) = file.read(&mut extra) {
            if n > 0 {
                return Err(CheckpointError::Invalid(
                    "trailing bytes after fixed-size checkpoint record".to_string(),
                ));
            }
        }

        if &buf[0..8] != MAGIC {
            return Err(CheckpointError::BadMagic);
        }

        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        if version != VERSION_V1 {
            return Err(CheckpointError::UnsupportedVersion(version));
        }

        let collection_id = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let last_completed_batch_idx = u64::from_le_bytes(buf[20..28].try_into().unwrap());
        let last_committed_lsn = u64::from_le_bytes(buf[28..36].try_into().unwrap());
        let stored_crc = u32::from_le_bytes(buf[36..40].try_into().unwrap());

        let computed_crc = crc32fast::hash(&buf[..PRE_CHECKSUM_LEN]);
        if stored_crc != computed_crc {
            return Err(CheckpointError::BadChecksum {
                expected: stored_crc,
                actual: computed_crc,
            });
        }

        Ok(BulkCheckpoint {
            version,
            collection_id,
            last_completed_batch_idx,
            last_committed_lsn,
        })
    }

    // Remove the checkpoint. Missing file is not an error.
    pub fn delete(path: &Path) -> Result<(), CheckpointError> {
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(CheckpointError::Io(e)),
        }
    }
}

// Build the sibling temp path: append ".tmp" to the file name.
fn tmp_sibling(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}
