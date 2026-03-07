// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Write-Ahead Log (WAL) writer and reader for crash-safe durability.
//!
//! Every mutation is appended to the WAL before being applied to segments.
//! The WAL can be replayed on startup to recover uncommitted changes.
//!
//! ## On-disk format
//!
//! ```text
//! File header:  [WAL_MAGIC: 4 bytes]
//! Entry:        [entry_len: u32 LE][op: u8][collection_id: u64 LE][payload][crc32: u32 LE]
//! ```
//!
//! `entry_len` = 1 + 8 + payload.len() (does NOT include itself or the trailing CRC).

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{StorageError, StorageResult};
use crate::format::{WalOp, WAL_MAGIC};

// ── WalEntry ────────────────────────────────────────────────────────────────

/// A single deserialized WAL entry (returned by [`WalReader`]).
#[derive(Clone, Debug)]
pub struct WalEntry {
    pub op: WalOp,
    pub collection_id: u64,
    pub payload: Vec<u8>,
}

// ── WalWriter ───────────────────────────────────────────────────────────────

/// Append-only WAL writer with automatic rotation support.
pub struct WalWriter {
    file: BufWriter<File>,
    path: PathBuf,
    bytes_written: u64,
    max_size: u64,
    entry_count: u64,
}

impl WalWriter {
    /// Create a **new** WAL file at `path`, writing the 4-byte magic header.
    pub fn create(path: &Path, max_size: u64) -> StorageResult<Self> {
        let file = File::create(path).map_err(|e| {
            StorageError::WalWriteFailed(format!("failed to create WAL file: {e}"))
        })?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&WAL_MAGIC).map_err(|e| {
            StorageError::WalWriteFailed(format!("failed to write WAL magic: {e}"))
        })?;
        writer.flush().map_err(StorageError::Io)?;

        Ok(Self {
            file: writer,
            path: path.to_path_buf(),
            bytes_written: 4,
            max_size,
            entry_count: 0,
        })
    }

    /// Open an **existing** WAL file for append.
    ///
    /// Verifies the magic bytes and seeks to the end so new entries are
    /// appended after any existing data.
    pub fn open(path: &Path, max_size: u64) -> StorageResult<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| {
                StorageError::WalWriteFailed(format!("failed to open WAL file: {e}"))
            })?;

        // Verify magic bytes.
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                StorageError::BadMagic {
                    expected: WAL_MAGIC,
                    found: [0; 4],
                }
            } else {
                StorageError::Io(e)
            }
        })?;

        if magic != WAL_MAGIC {
            return Err(StorageError::BadMagic {
                expected: WAL_MAGIC,
                found: magic,
            });
        }

        // Seek to end to determine total file size and prepare for appending.
        let size = file.seek(SeekFrom::End(0)).map_err(StorageError::Io)?;

        Ok(Self {
            file: BufWriter::new(file),
            path: path.to_path_buf(),
            bytes_written: size,
            max_size,
            entry_count: 0,
        })
    }

    /// Append an entry to the WAL.
    ///
    /// Wire format written:
    /// ```text
    /// [entry_len: u32 LE][op: u8][collection_id: u64 LE][payload][crc32: u32 LE]
    /// ```
    pub fn append(
        &mut self,
        op: WalOp,
        collection_id: u64,
        payload: &[u8],
    ) -> StorageResult<()> {
        let entry_len = 1u32 + 8u32 + payload.len() as u32;

        // Build the body that is covered by the CRC.
        let mut body = Vec::with_capacity(entry_len as usize);
        body.push(op.as_u8());
        body.extend_from_slice(&collection_id.to_le_bytes());
        body.extend_from_slice(payload);

        let crc = crc32fast::hash(&body);

        // Write: entry_len | body | crc32
        self.file
            .write_all(&entry_len.to_le_bytes())
            .map_err(|e| StorageError::WalWriteFailed(format!("write entry_len: {e}")))?;
        self.file
            .write_all(&body)
            .map_err(|e| StorageError::WalWriteFailed(format!("write body: {e}")))?;
        self.file
            .write_all(&crc.to_le_bytes())
            .map_err(|e| StorageError::WalWriteFailed(format!("write crc: {e}")))?;

        self.file.flush().map_err(StorageError::Io)?;

        // Total bytes on disk for this entry: 4 (entry_len) + body + 4 (crc).
        let total = 4u64 + entry_len as u64 + 4u64;
        self.bytes_written += total;
        self.entry_count += 1;

        Ok(())
    }

    /// Flush the buffer **and** fsync the underlying file to disk.
    pub fn sync(&mut self) -> StorageResult<()> {
        self.file.flush().map_err(StorageError::Io)?;
        self.file.get_ref().sync_all().map_err(StorageError::Io)?;
        Ok(())
    }

    /// Returns `true` when the WAL has grown past the configured maximum size
    /// and should be rotated.
    pub fn should_rotate(&self) -> bool {
        self.bytes_written >= self.max_size
    }

    /// Rotate the current WAL file.
    ///
    /// 1. Syncs the current file.
    /// 2. Renames it to `new_path` (typically a timestamped name).
    /// 3. Creates a fresh WAL at the original `self.path`.
    /// 4. Resets internal counters.
    ///
    /// Returns the path the old WAL was moved to (`new_path`).
    pub fn rotate(&mut self, new_path: &Path) -> StorageResult<PathBuf> {
        // Sync all pending data to disk before rotation.
        self.sync()?;

        let old_path = self.path.clone();

        // Swap out the current file handle with a /dev/null placeholder so we
        // can close the write handle before renaming. Using /dev/null avoids
        // any risk of truncating or locking the original WAL file.
        let old_writer = std::mem::replace(
            &mut self.file,
            BufWriter::new(
                File::open("/dev/null").map_err(|e| {
                    StorageError::WalRotationFailed(format!(
                        "failed to open /dev/null placeholder: {e}"
                    ))
                })?,
            ),
        );

        // Close the write handle so the file is no longer held open for writing.
        drop(old_writer);

        // Rename the old WAL to the archive path.
        fs::rename(&old_path, new_path).map_err(|e| {
            StorageError::WalRotationFailed(format!(
                "failed to rename WAL {:?} -> {:?}: {e}",
                old_path, new_path
            ))
        })?;

        // Create a brand-new WAL at the original path.
        let fresh = Self::create(&old_path, self.max_size)?;
        self.file = fresh.file;
        self.bytes_written = fresh.bytes_written;
        self.entry_count = fresh.entry_count;

        Ok(new_path.to_path_buf())
    }

    /// Path of the current WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total bytes written to the current WAL file (including the magic header).
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Number of entries appended since this writer was created / last rotated.
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }
}

// ── WalReader ───────────────────────────────────────────────────────────────

/// Sequential reader that iterates over entries in a WAL file.
pub struct WalReader {
    reader: BufReader<File>,
    path: PathBuf,
    position: u64,
}

impl WalReader {
    /// Open a WAL file for reading, verifying the magic header.
    pub fn open(path: &Path) -> StorageResult<Self> {
        let file = File::open(path).map_err(StorageError::Io)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                StorageError::BadMagic {
                    expected: WAL_MAGIC,
                    found: [0; 4],
                }
            } else {
                StorageError::Io(e)
            }
        })?;

        if magic != WAL_MAGIC {
            return Err(StorageError::BadMagic {
                expected: WAL_MAGIC,
                found: magic,
            });
        }

        Ok(Self {
            reader,
            path: path.to_path_buf(),
            position: 4,
        })
    }

    /// Read the next WAL entry, or `None` on clean EOF.
    ///
    /// Partial / truncated entries (from a crash mid-write) also return `None`
    /// rather than an error, enabling safe crash recovery.
    pub fn next_entry(&mut self) -> StorageResult<Option<WalEntry>> {
        // Read entry_len (4 bytes).
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(StorageError::Io(e)),
        }
        let entry_len = u32::from_le_bytes(len_buf) as usize;

        // Sanity: entry_len must be at least 9 (1 byte op + 8 bytes collection_id).
        if entry_len < 9 {
            return Err(StorageError::WalInvalidEntry(format!(
                "entry_len too small: {entry_len}"
            )));
        }

        // Read body (entry_len bytes).
        let mut body = vec![0u8; entry_len];
        match self.reader.read_exact(&mut body) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(StorageError::Io(e)),
        }

        // Read stored CRC (4 bytes).
        let mut crc_buf = [0u8; 4];
        match self.reader.read_exact(&mut crc_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(StorageError::Io(e)),
        }
        let stored_crc = u32::from_le_bytes(crc_buf);

        // Verify CRC over body.
        let computed_crc = crc32fast::hash(&body);
        if stored_crc != computed_crc {
            return Err(StorageError::ChecksumMismatch {
                expected: stored_crc,
                computed: computed_crc,
            });
        }

        // Parse fields out of the body.
        let op = WalOp::from_u8(body[0])?;
        let collection_id = u64::from_le_bytes(body[1..9].try_into().unwrap());
        let payload = body[9..].to_vec();

        // Update position: 4 (entry_len) + entry_len + 4 (crc).
        self.position += 4 + entry_len as u64 + 4;

        Ok(Some(WalEntry {
            op,
            collection_id,
            payload,
        }))
    }

    /// Path of the WAL file being read.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Current byte offset into the WAL file.
    pub fn position(&self) -> u64 {
        self.position
    }
}

impl Iterator for WalReader {
    type Item = StorageResult<WalEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_entry() {
            Ok(Some(entry)) => Some(Ok(entry)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
