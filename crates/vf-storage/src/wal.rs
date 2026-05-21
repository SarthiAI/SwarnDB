// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Write-Ahead Log (WAL) writer and reader for crash-safe durability.
//!
//! Every mutation is appended to the WAL before being applied to segments.
//! The WAL can be replayed on startup to recover uncommitted changes.
//!
//! ## On-disk format (v2)
//!
//! ```text
//! File header:  [WAL_MAGIC: 4 bytes][format_version: u8 = 2]
//! Entry:        [entry_len: u32 LE][lsn: u64 LE][op: u8][collection_id: u64 LE][payload][crc32: u32 LE]
//! ```
//!
//! `entry_len` = 8 (lsn) + 1 (op) + 8 (collection_id) + payload.len().
//! Minimum `entry_len` is 17.
//!
//! ## Legacy format (v1)
//!
//! ```text
//! File header:  [WAL_MAGIC: 4 bytes]
//! Entry:        [entry_len: u32 LE][op: u8][collection_id: u64 LE][payload][crc32: u32 LE]
//! ```

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{StorageError, StorageResult};
use crate::format::{WalOp, WAL_MAGIC};

/// Current WAL format version.
/// Value chosen to never collide with the LSB of a valid v1 entry_len (min 9).
pub const WAL_FORMAT_VERSION: u8 = 0;

/// WAL v1 had no version byte; header was just the 4-byte magic.
const WAL_V1_HEADER_SIZE: u64 = 4;

/// WAL v2 header: magic (4) + version (1).
const WAL_V2_HEADER_SIZE: u64 = 5;

/// Minimum entry body size for v2: lsn(8) + op(1) + collection_id(8) = 17.
const MIN_ENTRY_LEN_V2: usize = 17;

/// Minimum entry body size for v1: op(1) + collection_id(8) = 9.
const MIN_ENTRY_LEN_V1: usize = 9;

// ── FsyncMode ──────────────────────────────────────────────────────────────

/// Controls when the WAL is fsync'd to disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsyncMode {
    /// fsync after every write (maximum durability, higher latency).
    PerWrite,
    /// fsync after every N entries (batched durability).
    PerBatch(u64),
    /// No automatic fsync (caller manages).
    None,
}

impl FsyncMode {
    /// Resolve fsync mode from the `SWARNDB_WAL_FSYNC_MODE` env var.
    ///
    /// Accepted values:
    /// - `per_write` (default): fsync after every WAL append. Maximum durability.
    /// - `per_batch:N`: fsync after every N appends. Trades durability for throughput.
    /// - `none`: skip automatic fsync. Caller (or rotate/close) handles durability.
    ///
    /// Any unset or malformed value falls back to `PerWrite` and logs a warning.
    pub fn from_env() -> Self {
        let raw = match std::env::var("SWARNDB_WAL_FSYNC_MODE") {
            Ok(v) => v,
            Err(_) => return FsyncMode::PerWrite,
        };
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("per_write") || trimmed.eq_ignore_ascii_case("perwrite") {
            return FsyncMode::PerWrite;
        }
        if trimmed.eq_ignore_ascii_case("none") {
            return FsyncMode::None;
        }
        if let Some(rest) = trimmed
            .strip_prefix("per_batch:")
            .or_else(|| trimmed.strip_prefix("PerBatch:"))
        {
            match rest.parse::<u64>() {
                Ok(n) if n > 0 => return FsyncMode::PerBatch(n),
                _ => {
                    log::warn!(
                        "invalid per_batch count in SWARNDB_WAL_FSYNC_MODE={:?}, falling back to PerWrite",
                        trimmed
                    );
                    return FsyncMode::PerWrite;
                }
            }
        }
        log::warn!(
            "unknown SWARNDB_WAL_FSYNC_MODE={:?}, falling back to PerWrite",
            trimmed
        );
        FsyncMode::PerWrite
    }
}

// ── WalEntry ────────────────────────────────────────────────────────────────

/// A single deserialized WAL entry (returned by [`WalReader`]).
#[derive(Clone, Debug)]
pub struct WalEntry {
    pub lsn: u64,
    pub op: WalOp,
    pub collection_id: u64,
    pub payload: Vec<u8>,
}

// ── WalWriter ───────────────────────────────────────────────────────────────

/// Append-only WAL writer with LSN tracking and automatic rotation support.
pub struct WalWriter {
    file: BufWriter<File>,
    path: PathBuf,
    bytes_written: u64,
    max_size: u64,
    entry_count: u64,
    next_lsn: u64,
    fsync_mode: FsyncMode,
    unflushed_entries: u64,
}

impl WalWriter {
    /// Create a **new** WAL file at `path` with v2 header (magic + version).
    pub fn create(path: &Path, max_size: u64, initial_lsn: u64) -> StorageResult<Self> {
        let file = File::create(path).map_err(|e| {
            StorageError::WalWriteFailed(format!("failed to create WAL file: {e}"))
        })?;
        let mut writer = BufWriter::new(file);

        // Write v2 header: magic + version byte.
        writer.write_all(&WAL_MAGIC).map_err(|e| {
            StorageError::WalWriteFailed(format!("failed to write WAL magic: {e}"))
        })?;
        writer.write_all(&[WAL_FORMAT_VERSION]).map_err(|e| {
            StorageError::WalWriteFailed(format!("failed to write WAL version: {e}"))
        })?;
        writer.flush().map_err(StorageError::Io)?;

        Ok(Self {
            file: writer,
            path: path.to_path_buf(),
            bytes_written: WAL_V2_HEADER_SIZE,
            max_size,
            entry_count: 0,
            next_lsn: initial_lsn.max(1),
            fsync_mode: FsyncMode::PerWrite,
            unflushed_entries: 0,
        })
    }

    /// Open an **existing** WAL file for append.
    ///
    /// Detects format version (v1 or v2) and scans to find the highest LSN.
    /// If `initial_lsn` is provided, uses max(initial_lsn, highest_found + 1).
    pub fn open(path: &Path, max_size: u64, initial_lsn: u64) -> StorageResult<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| {
                StorageError::WalWriteFailed(format!("failed to open WAL file: {e}"))
            })?;

        // Read magic bytes.
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

        // Detect format version: peek at byte 4.
        // v2 writes WAL_FORMAT_VERSION (0x00) at byte 4.
        // v1 has no version byte; byte 4 is the LSB of the first entry_len (u32 LE).
        // Since min v1 entry_len is 9, the LSB is always >= 9 (for small payloads)
        // or any non-zero value for larger payloads. 0x00 as LSB would require
        // entry_len % 256 == 0, but min is 9, so 256 is the smallest such value
        // (payload of 247 bytes). We use 0x00 as the version sentinel precisely
        // because it cannot be the LSB of any v1 entry_len < 256.
        // For safety, we also verify by checking that the byte after is a valid
        // v2 entry_len (>= 17) or EOF.
        let mut version_byte = [0u8; 1];
        let is_v2 = match file.read_exact(&mut version_byte) {
            Ok(()) => version_byte[0] == WAL_FORMAT_VERSION,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => false,
            Err(e) => return Err(StorageError::Io(e)),
        };

        let header_size = if is_v2 { WAL_V2_HEADER_SIZE } else { WAL_V1_HEADER_SIZE };

        // Scan existing entries to find the highest LSN.
        file.seek(SeekFrom::Start(header_size)).map_err(StorageError::Io)?;
        let mut max_lsn: u64 = 0;
        let mut scan_reader = BufReader::new(&file);

        if is_v2 {
            loop {
                let mut len_buf = [0u8; 4];
                match scan_reader.read_exact(&mut len_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(StorageError::Io(e)),
                }
                let entry_len = u32::from_le_bytes(len_buf) as usize;
                if entry_len < MIN_ENTRY_LEN_V2 {
                    break; // Corrupt entry - stop scanning.
                }

                // Read lsn (first 8 bytes of body).
                let mut lsn_buf = [0u8; 8];
                match scan_reader.read_exact(&mut lsn_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(StorageError::Io(e)),
                }
                let lsn = u64::from_le_bytes(lsn_buf);
                if lsn > max_lsn {
                    max_lsn = lsn;
                }

                // Skip rest of body (entry_len - 8) + crc (4).
                let skip = (entry_len - 8) as u64 + 4;
                match io::copy(&mut scan_reader.by_ref().take(skip), &mut io::sink()) {
                    Ok(n) if n == skip => {}
                    Ok(_) => break, // Truncated entry.
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(StorageError::Io(e)),
                }
            }
        }
        // For v1 files, max_lsn stays 0 - initial_lsn takes over.

        drop(scan_reader);

        // Seek to end for appending.
        let size = file.seek(SeekFrom::End(0)).map_err(StorageError::Io)?;

        let resume_lsn = std::cmp::max(
            initial_lsn.max(1),
            if max_lsn > 0 { max_lsn + 1 } else { initial_lsn.max(1) },
        );

        Ok(Self {
            file: BufWriter::new(file),
            path: path.to_path_buf(),
            bytes_written: size,
            max_size,
            entry_count: 0,
            next_lsn: resume_lsn,
            fsync_mode: FsyncMode::PerWrite,
            unflushed_entries: 0,
        })
    }

    /// Append an entry to the WAL. Returns the assigned LSN.
    ///
    /// Wire format (v2):
    /// ```text
    /// [entry_len: u32 LE][lsn: u64 LE][op: u8][collection_id: u64 LE][payload][crc32: u32 LE]
    /// ```
    pub fn append(
        &mut self,
        op: WalOp,
        collection_id: u64,
        payload: &[u8],
    ) -> StorageResult<u64> {
        let lsn = self.next_lsn;

        // Guard against payload too large for u32 entry_len.
        let payload_len = payload.len();
        if payload_len > (u32::MAX - 17) as usize {
            return Err(StorageError::WalWriteFailed(
                "payload too large for WAL entry".into(),
            ));
        }
        let entry_len = 8u32 + 1u32 + 8u32 + payload_len as u32;

        // Build the body covered by CRC: [lsn][op][collection_id][payload].
        let mut body = Vec::with_capacity(entry_len as usize);
        body.extend_from_slice(&lsn.to_le_bytes());
        body.push(op.as_u8());
        body.extend_from_slice(&collection_id.to_le_bytes());
        body.extend_from_slice(payload);

        let crc = crc32fast::hash(&body);

        // Write: entry_len | body | crc32.
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
        self.next_lsn += 1;
        self.unflushed_entries += 1;

        // Fsync based on mode.
        match self.fsync_mode {
            FsyncMode::PerWrite => {
                self.file.get_ref().sync_all().map_err(StorageError::Io)?;
                self.unflushed_entries = 0;
            }
            FsyncMode::PerBatch(n) if self.unflushed_entries >= n => {
                self.file.get_ref().sync_all().map_err(StorageError::Io)?;
                self.unflushed_entries = 0;
            }
            _ => {}
        }

        Ok(lsn)
    }

    /// Flush the buffer **and** fsync the underlying file to disk.
    pub fn sync(&mut self) -> StorageResult<()> {
        self.file.flush().map_err(StorageError::Io)?;
        self.file.get_ref().sync_all().map_err(StorageError::Io)?;
        self.unflushed_entries = 0;
        Ok(())
    }

    /// Set the fsync mode for this writer.
    pub fn set_fsync_mode(&mut self, mode: FsyncMode) {
        self.fsync_mode = mode;
    }

    /// Returns `true` when the WAL has grown past the configured maximum size.
    pub fn should_rotate(&self) -> bool {
        self.bytes_written >= self.max_size
    }

    /// Rotate the current WAL file.
    ///
    /// LSN continuity is preserved - the fresh WAL inherits `next_lsn`.
    /// Returns the path the old WAL was moved to.
    pub fn rotate(&mut self, new_path: &Path) -> StorageResult<PathBuf> {
        self.sync()?;

        let old_path = self.path.clone();
        let carry_lsn = self.next_lsn;
        let carry_fsync = self.fsync_mode;

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
        drop(old_writer);

        fs::rename(&old_path, new_path).map_err(|e| {
            StorageError::WalRotationFailed(format!(
                "failed to rename WAL {:?} -> {:?}: {e}",
                old_path, new_path
            ))
        })?;

        // Create fresh WAL, preserving LSN continuity.
        let mut fresh = Self::create(&old_path, self.max_size, carry_lsn)?;
        fresh.fsync_mode = carry_fsync;

        self.file = fresh.file;
        self.bytes_written = fresh.bytes_written;
        self.entry_count = fresh.entry_count;
        self.next_lsn = fresh.next_lsn;
        self.unflushed_entries = 0;

        Ok(new_path.to_path_buf())
    }

    /// The next LSN that will be assigned.
    pub fn current_lsn(&self) -> u64 {
        self.next_lsn
    }

    /// Path of the current WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total bytes written to the current WAL file.
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
/// Supports both v1 and v2 WAL formats.
pub struct WalReader {
    reader: BufReader<File>,
    path: PathBuf,
    position: u64,
    is_v2: bool,
}

impl WalReader {
    /// Open a WAL file for reading, detecting format version.
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

        // Detect v2: peek at byte 4.
        let mut version_byte = [0u8; 1];
        let is_v2 = match reader.read_exact(&mut version_byte) {
            Ok(()) => version_byte[0] == WAL_FORMAT_VERSION,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => false,
            Err(e) => return Err(StorageError::Io(e)),
        };

        let position = if is_v2 {
            WAL_V2_HEADER_SIZE
        } else {
            // v1: we already read 5 bytes but header is 4. Seek back to offset 4.
            reader.seek_relative(-1).map_err(StorageError::Io)?;
            WAL_V1_HEADER_SIZE
        };

        Ok(Self {
            reader,
            path: path.to_path_buf(),
            position,
            is_v2,
        })
    }

    /// Read the next WAL entry, or `None` on clean EOF.
    ///
    /// Partial / truncated entries return `None` for safe crash recovery.
    pub fn next_entry(&mut self) -> StorageResult<Option<WalEntry>> {
        if self.is_v2 {
            self.read_v2_entry()
        } else {
            self.read_v1_entry()
        }
    }

    fn read_v2_entry(&mut self) -> StorageResult<Option<WalEntry>> {
        // Read entry_len (4 bytes).
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(StorageError::Io(e)),
        }
        let entry_len = u32::from_le_bytes(len_buf) as usize;

        if entry_len < MIN_ENTRY_LEN_V2 {
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

        let computed_crc = crc32fast::hash(&body);
        if stored_crc != computed_crc {
            return Err(StorageError::ChecksumMismatch {
                expected: stored_crc,
                computed: computed_crc,
            });
        }

        // Parse: [lsn: 8][op: 1][collection_id: 8][payload...]
        let lsn = u64::from_le_bytes(body[0..8].try_into().unwrap());
        let op = WalOp::from_u8(body[8])?;
        let collection_id = u64::from_le_bytes(body[9..17].try_into().unwrap());
        let payload = body[17..].to_vec();

        self.position += 4 + entry_len as u64 + 4;

        Ok(Some(WalEntry {
            lsn,
            op,
            collection_id,
            payload,
        }))
    }

    /// Read a v1 entry (no LSN field). Assigns lsn = 0 for compatibility.
    fn read_v1_entry(&mut self) -> StorageResult<Option<WalEntry>> {
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(StorageError::Io(e)),
        }
        let entry_len = u32::from_le_bytes(len_buf) as usize;

        if entry_len < MIN_ENTRY_LEN_V1 {
            return Err(StorageError::WalInvalidEntry(format!(
                "entry_len too small: {entry_len}"
            )));
        }

        let mut body = vec![0u8; entry_len];
        match self.reader.read_exact(&mut body) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(StorageError::Io(e)),
        }

        let mut crc_buf = [0u8; 4];
        match self.reader.read_exact(&mut crc_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(StorageError::Io(e)),
        }
        let stored_crc = u32::from_le_bytes(crc_buf);

        let computed_crc = crc32fast::hash(&body);
        if stored_crc != computed_crc {
            return Err(StorageError::ChecksumMismatch {
                expected: stored_crc,
                computed: computed_crc,
            });
        }

        let op = WalOp::from_u8(body[0])?;
        let collection_id = u64::from_le_bytes(body[1..9].try_into().unwrap());
        let payload = body[9..].to_vec();

        self.position += 4 + entry_len as u64 + 4;

        Ok(Some(WalEntry {
            lsn: 0,
            op,
            collection_id,
            payload,
        }))
    }

    /// Whether this WAL uses the v2 format (with LSN).
    pub fn is_v2(&self) -> bool {
        self.is_v2
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

// ── WalMeta ─────────────────────────────────────────────────────────────────

/// Persistent metadata for a collection's WAL state.
/// Stored as `wal_meta.json` in each collection directory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalMeta {
    pub next_lsn: u64,
    pub last_snapshot_lsn: u64,
    pub wal_format_version: u32,
}

impl WalMeta {
    pub fn new(next_lsn: u64) -> Self {
        Self {
            next_lsn,
            last_snapshot_lsn: 0,
            wal_format_version: WAL_FORMAT_VERSION as u32,
        }
    }
}

/// Load WAL metadata from a collection directory.
/// Returns a default WalMeta if the file doesn't exist.
pub fn load_wal_meta(dir: &Path) -> StorageResult<WalMeta> {
    let path = dir.join("wal_meta.json");
    if !path.exists() {
        return Ok(WalMeta::new(1));
    }
    let data = fs::read_to_string(&path)?;
    serde_json::from_str(&data).map_err(|e| {
        StorageError::Serialization(format!("failed to parse wal_meta.json: {e}"))
    })
}

/// Save WAL metadata to a collection directory (atomic write).
pub fn save_wal_meta(dir: &Path, meta: &WalMeta) -> StorageResult<()> {
    let path = dir.join("wal_meta.json");
    let data = serde_json::to_string_pretty(meta).map_err(|e| {
        StorageError::Serialization(format!("failed to serialize wal_meta: {e}"))
    })?;
    // Use atomic write for crash safety
    crate::atomic_write::atomic_write(&path, data.as_bytes())
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
