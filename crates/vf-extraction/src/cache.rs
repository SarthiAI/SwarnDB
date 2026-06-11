// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! The extraction cache: a hot-value LRU over an append-only jsonl index on
//! disk, with a full-coverage on-disk offset index so no entry that is on disk
//! is ever reported as a miss. A cache hit returns a prior extraction result for
//! the same normalized text, model, and prompt version, so a re-extraction never
//! re-pays the LLM.
//!
//! Two layers back every lookup:
//!   - The LRU holds the hot VALUES, bounded by `max_entries` (a global budget).
//!   - The offset index maps every on-disk key to the byte offset of its line in
//!     the jsonl file. It holds only key + offset (tens of bytes per key), so its
//!     RAM is proportional to the number of distinct cached chunks, never to the
//!     value sizes. On an LRU miss the index is consulted; if the key is on disk,
//!     its line is seeked, read, deserialized, and promoted into the LRU. Only a
//!     key absent from the index is a true miss. This upholds the invariant that
//!     an entry present on disk is never silently re-paid, even past the LRU cap.
//!
//! The cache is GLOBAL, not per-collection: its key is the triple (chunk hash,
//! model, prompt version) plus an optional custom-prompt digest, with no
//! collection identifier. One shared on-disk index serves every collection, so
//! dropping a collection and re-extracting the same corpus is near-all hits and
//! re-pays the LLM only for genuinely new text or a changed model/prompt.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::adapter::{ExtractionResult, PromptVersion};
use crate::error::ExtractionError;

/// File name of the on-disk jsonl index inside the cache dir.
const INDEX_FILE: &str = "extraction_cache.jsonl";

/// One line of the on-disk jsonl index.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CacheEntry {
    key: String,
    result: ExtractionResult,
}

/// Normalize text for the cache key: trim and collapse internal runs of
/// whitespace to a single space. Two passages that differ only in spacing share
/// a cache key.
pub fn normalize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.trim().chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

/// Derive the hex sha256 cache key from normalized text, model, prompt version,
/// and an optional per-collection custom-prompt digest. The text passed in is
/// expected to already be normalized.
///
/// Cache-key scheme and its backward-compatibility guarantee:
///   - `custom_prompt` is `Some(digest)` only when a collection set a custom
///     `system_prompt` and/or `extra_guidance`; `digest` is the stable hash from
///     `custom_prompt_hash`. It is `None` for the default prompt.
///   - When `custom_prompt` is `None` the hasher input is exactly
///     `text \0 model \0 prompt_version`, byte-for-byte identical to the prior
///     scheme, so caches written for default-prompt collections stay valid and
///     are never invalidated by this change.
///   - When `custom_prompt` is `Some(digest)` a third field `\0 digest` is
///     appended, so any change to the custom prompt yields a distinct key and
///     forces a recompute, while a fixed custom prompt stays stable across
///     restarts (the digest is content-derived, not random).
pub fn cache_key(
    normalized_text: &str,
    model: &str,
    prompt_version: PromptVersion,
    custom_prompt: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(normalized_text.as_bytes());
    hasher.update(b"\0");
    hasher.update(model.as_bytes());
    hasher.update(b"\0");
    hasher.update(prompt_version.as_str().as_bytes());
    // Only the custom-prompt case touches the hasher further, so the default
    // (None) key is byte-identical to the pre-customization scheme above.
    if let Some(digest) = custom_prompt {
        hasher.update(b"\0");
        hasher.update(digest.as_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((b & 0x0f) as u32, 16).unwrap_or('0'));
    }
    out
}

/// Stable digest of a collection's custom-prompt inputs for the cache key, or
/// `None` when none is set (so the cache key keeps its default-prompt form).
///
/// Treats whitespace-only inputs as unset to mirror `prompt::build_system_prompt`,
/// so a blank override does not split the cache from the default. The fields are
/// joined with a unit-separator (`\x1f`) byte so distinct combinations cannot
/// collide by concatenation, then hashed to a fixed-width hex string.
///
/// `link_passages` (ADR-012) is folded in only when true. When it is true the
/// merged ontology gains a passage-linking edge type, which changes the allowed-
/// edge-types section of the extraction prompt, so the LLM response can differ;
/// folding it in keeps such collections from reading stale cached results. When
/// false (and no custom prompt is set) the digest stays `None`, so the cache key
/// is byte-identical to the pre-existing default-prompt form.
pub fn custom_prompt_hash(
    system_prompt: Option<&str>,
    extra_guidance: Option<&str>,
    link_passages: bool,
) -> Option<String> {
    let system = system_prompt.map(str::trim).filter(|s| !s.is_empty());
    let guidance = extra_guidance.map(str::trim).filter(|s| !s.is_empty());
    if system.is_none() && guidance.is_none() && !link_passages {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(system.unwrap_or("").as_bytes());
    hasher.update(b"\x1f");
    hasher.update(guidance.unwrap_or("").as_bytes());
    // Only a set link_passages touches the hasher further, so a custom-prompt
    // collection that does not opt in keeps its byte-identical digest.
    if link_passages {
        hasher.update(b"\x1flp1");
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((b & 0x0f) as u32, 16).unwrap_or('0'));
    }
    Some(out)
}

/// Stable content hash of a chunk: sha256 of the normalized text, hex.
pub fn chunk_content_hash(text: &str) -> String {
    let normalized = normalize_text(text);
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((b & 0x0f) as u32, 16).unwrap_or('0'));
    }
    out
}

/// Name of the global cache directory under the storage data root. Public so the
/// server can reserve this exact name at collection creation: a collection named
/// `_extraction_cache` would map to this dir, and dropping it would delete the
/// shared cache. Referenced by the server's name validation to avoid drift.
pub const GLOBAL_CACHE_DIR: &str = "_extraction_cache";

/// Derive the shared global cache directory from one collection's extraction
/// dir. The extraction dir is `<data_root>/<collection>/extraction`, so its
/// grandparent is the data root; the global cache lives at
/// `<data_root>/_extraction_cache`, shared by every collection and untouched
/// when any single collection is dropped. Falls back to the parent, then the
/// dir itself, if the path is too shallow to have a grandparent, so the cache
/// still opens rather than panicking on an unexpected layout.
pub fn global_cache_dir(extraction_dir: &Path) -> PathBuf {
    let root = extraction_dir
        .parent()
        .and_then(Path::parent)
        .or_else(|| extraction_dir.parent())
        .unwrap_or(extraction_dir);
    root.join(GLOBAL_CACHE_DIR)
}

/// An LRU extraction cache backed by an append-only jsonl index on disk, with a
/// full-coverage offset index so on-disk entries are never false-missed.
pub struct ExtractionCache {
    /// Hot VALUES, bounded by the configured cap.
    mem: LruCache<String, ExtractionResult>,
    /// Every on-disk key -> byte offset of its line's first byte in the jsonl
    /// file. Holds key + offset only (no values), so its RAM grows with the
    /// number of distinct cached chunks (~tens of bytes per key), not value size.
    offsets: HashMap<String, u64>,
    dir: PathBuf,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ExtractionCache {
    /// Open (or create) the cache under `dir`. Builds the full offset index over
    /// the jsonl file (every on-disk key -> line offset) and pre-warms the LRU
    /// with the most recent values up to `max_entries` (at least 1). Reads bytes
    /// line by line so a corrupt or truncated line is skipped without breaking
    /// the offsets of the lines around it.
    pub fn open(dir: &Path, max_entries: usize) -> Result<Self, ExtractionError> {
        std::fs::create_dir_all(dir).map_err(|e| ExtractionError::Io(e.to_string()))?;
        let cap = NonZeroUsize::new(max_entries.max(1))
            .unwrap_or_else(|| NonZeroUsize::new(1).expect("1 is non-zero"));
        let mut mem = LruCache::new(cap);
        let mut offsets: HashMap<String, u64> = HashMap::new();
        let mut write_offset: u64 = 0;

        let index_path = dir.join(INDEX_FILE);
        if index_path.exists() {
            let file = File::open(&index_path).map_err(|e| ExtractionError::Io(e.to_string()))?;
            let mut reader = BufReader::new(file);
            let mut buf: Vec<u8> = Vec::new();
            // Track the byte offset of the current line's first byte. read_until
            // returns the bytes consumed including the delimiter, so the offset
            // stays exact across skipped lines and a missing final newline.
            loop {
                let line_start = write_offset;
                buf.clear();
                let n = reader
                    .read_until(b'\n', &mut buf)
                    .map_err(|e| ExtractionError::Io(e.to_string()))?;
                if n == 0 {
                    break; // EOF
                }
                write_offset += n as u64;

                let line = String::from_utf8_lossy(&buf);
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // Skip a malformed or truncated-tail line rather than failing the
                // whole open; its offset slot simply gets no index entry, and the
                // running offset already advanced past its raw bytes.
                if let Ok(entry) = serde_json::from_str::<CacheEntry>(trimmed) {
                    // A later line for the same key wins (newest offset + value),
                    // mirroring put() overwrite semantics.
                    offsets.insert(entry.key.clone(), line_start);
                    mem.put(entry.key, entry.result);
                }
            }
        }

        Ok(Self {
            mem,
            offsets,
            dir: dir.to_path_buf(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        })
    }

    /// Look up a cached result, bumping the hit or miss counter accordingly.
    ///
    /// LRU hit -> return. LRU miss -> consult the offset index: if the key is on
    /// disk (evicted from the LRU or never warmed), seek+read+deserialize that one
    /// line, promote it into the LRU, and return it as a hit. Only a key absent
    /// from the index is a true miss. This guarantees an on-disk entry is never
    /// silently re-paid.
    pub fn get(&mut self, key: &str) -> Option<ExtractionResult> {
        if let Some(result) = self.mem.get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(result.clone());
        }

        // LRU miss: fall back to the on-disk offset index before declaring a miss.
        if let Some(&offset) = self.offsets.get(key) {
            match self.read_entry_at(offset) {
                Some(result) => {
                    self.mem.put(key.to_string(), result.clone());
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Some(result);
                }
                None => {
                    // The index pointed at a line we could not read or parse
                    // (unexpected on a healthy file). Treat as a true miss rather
                    // than panic; the caller re-extracts and put() re-indexes it.
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Read and deserialize the single jsonl line whose first byte is at
    /// `offset`. Returns None on any IO/parse failure so a damaged line degrades
    /// to a miss instead of a panic.
    fn read_entry_at(&self, offset: u64) -> Option<ExtractionResult> {
        let index_path = self.dir.join(INDEX_FILE);
        let file = File::open(&index_path).ok()?;
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(offset)).ok()?;
        let mut buf: Vec<u8> = Vec::new();
        // read_until stops at this line's own newline (or EOF for a final line
        // missing one), so the read is bounded by the single line, never the
        // whole file. Reading the exact line a put() wrote keeps large but valid
        // entries intact instead of capping them into a parse failure.
        let n = reader.read_until(b'\n', &mut buf).ok()?;
        if n == 0 {
            return None;
        }
        let line = String::from_utf8_lossy(&buf);
        let entry = serde_json::from_str::<CacheEntry>(line.trim()).ok()?;
        Some(entry.result)
    }

    /// Store a result: append it to the jsonl index, record its line offset in the
    /// offset index, and insert into the LRU. The offset enables a later LRU-miss
    /// disk fallback so the entry is never re-paid once on disk.
    pub fn put(&mut self, key: &str, result: &ExtractionResult) -> Result<(), ExtractionError> {
        let entry = CacheEntry {
            key: key.to_string(),
            result: result.clone(),
        };
        // Build the full line (payload + newline) and write it in one call so
        // there is no partial-success split between two writes within a run.
        let mut line = serde_json::to_string(&entry)
            .map_err(|e| ExtractionError::Io(e.to_string()))?;
        line.push('\n');
        let bytes = line.as_bytes();

        let index_path = self.dir.join(INDEX_FILE);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&index_path)
            .map_err(|e| ExtractionError::Io(e.to_string()))?;

        // Derive the new line's start offset from the real file length, so any
        // prior write error (which left write_offset stale) self-corrects here
        // rather than recording a wrong offset for this entry.
        let line_start = file
            .metadata()
            .map(|m| m.len())
            .map_err(|e| ExtractionError::Io(e.to_string()))?;

        file.write_all(bytes)
            .map_err(|e| ExtractionError::Io(e.to_string()))?;

        // Only after a successful write do we publish the offset and value, so a
        // failed write never leaves a dangling index entry pointing at bytes that
        // are not there.
        self.offsets.insert(key.to_string(), line_start);
        self.mem.put(key.to_string(), result.clone());
        Ok(())
    }

    /// Hit rate over the lifetime of this cache instance (0.0 with no lookups).
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Total cache hits since open.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Total cache misses since open.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}
