// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0
// See LICENSE file in the project root for full license text

//! Collection restore from tar.gz backup archives.
//!
//! Extracts a backup archive produced by the [`crate::backup`] module,
//! optionally verifies SHA-256 checksums against the embedded manifest, and
//! places files into the target directory so the collection can be re-opened.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use tar::Archive;

use crate::backup::{BackupManifest, ManifestEntry};
use crate::error::{StorageError, StorageResult};
use crate::util::hex_encode;

/// Default allowed file extensions for restore operations.
const DEFAULT_ALLOWED_EXTENSIONS: &[&str] = &["bin", "wal", "json", "seg", "vfs", "log", "old", "dat"];

// ── Options & Result ────────────────────────────────────────────────────────

/// Options controlling how a backup archive is restored.
#[derive(Debug, Clone)]
pub struct RestoreOptions {
    /// When `true`, compute SHA-256 of every extracted file and compare against
    /// the manifest. Restoration fails if any checksum does not match.
    pub verify_checksums: bool,

    /// Optional override for the target directory. When `None`, files are
    /// extracted into `target_dir` as passed to [`restore_backup`].
    pub target_dir_override: Option<PathBuf>,

    /// Additional file extensions to allow during restore, beyond the defaults
    /// ("bin", "wal", "json", "seg", "vfs", "log", "old", "dat").
    /// Useful for plugins or custom storage backends that use non-standard extensions.
    pub extra_allowed_extensions: Vec<String>,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self {
            verify_checksums: true,
            target_dir_override: None,
            extra_allowed_extensions: Vec::new(),
        }
    }
}

/// Result of a successful restore operation.
#[derive(Debug, Clone)]
pub struct RestoreResult {
    /// Name of the restored collection (from the manifest).
    pub collection_name: String,

    /// Number of files extracted from the archive.
    pub files_restored: usize,

    /// Whether SHA-256 verification was performed and passed.
    pub verified: bool,
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Restore a collection from a tar.gz backup archive.
///
/// Opens the archive, reads the embedded `manifest.json`, extracts all data
/// files to `target_dir` (or the override), and optionally verifies SHA-256
/// checksums.
///
/// # Arguments
/// * `archive_path` - Path to the `.tar.gz` backup file.
/// * `target_dir`   - Directory where the collection files will be placed.
/// * `options`      - [`RestoreOptions`] controlling verification behaviour.
///
/// # Errors
/// Returns [`StorageError::Io`] on I/O failures,
/// [`StorageError::Serialization`] if the manifest cannot be parsed or if
/// a SHA-256 checksum does not match.
pub fn restore_backup(
    archive_path: &Path,
    target_dir: &Path,
    options: &RestoreOptions,
) -> StorageResult<RestoreResult> {
    let effective_dir = options
        .target_dir_override
        .as_deref()
        .unwrap_or(target_dir);

    // Ensure the target directory exists.
    fs::create_dir_all(effective_dir).map_err(StorageError::Io)?;

    let archive_file = fs::File::open(archive_path).map_err(StorageError::Io)?;
    let decoder = GzDecoder::new(archive_file);
    let mut archive = Archive::new(decoder);

    // Single pass: extract all entries, collecting relative paths for verification.
    let mut extracted_files: Vec<PathBuf> = Vec::new();
    let mut manifest: Option<BackupManifest> = None;

    for entry_result in archive.entries().map_err(StorageError::Io)? {
        let mut entry = entry_result.map_err(StorageError::Io)?;
        let entry_path = entry.path().map_err(StorageError::Io)?.into_owned();
        let entry_path_str = entry_path.to_string_lossy().to_string();

        // Strip the leading collection-name prefix the backup module adds.
        // Archive paths look like: "my_collection/segment_0.vfs"
        let rel_path = strip_first_component(&entry_path_str);

        if rel_path == "manifest.json" {
            // Read manifest into memory.
            let mut contents = String::new();
            entry
                .read_to_string(&mut contents)
                .map_err(StorageError::Io)?;
            manifest = Some(serde_json::from_str(&contents).map_err(|e| {
                StorageError::Serialization(format!("failed to parse manifest.json: {e}"))
            })?);
        } else {
            // PATH TRAVERSAL GUARD: reject any relative path containing ".."
            // or starting with "/" to prevent writing outside the target dir.
            if rel_path.contains("..") || rel_path.starts_with('/') {
                return Err(StorageError::Serialization(format!(
                    "path traversal detected: archive entry '{}' contains unsafe path components",
                    entry_path_str,
                )));
            }

            // Task 274: Validate file extension against allowlist.
            let rel_path_obj = Path::new(rel_path);
            let extension = rel_path_obj
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let ext_allowed = DEFAULT_ALLOWED_EXTENSIONS.contains(&extension)
                || options.extra_allowed_extensions.iter().any(|e| e.as_str() == extension);
            if !ext_allowed {
                log::warn!(
                    "Restore: skipping file '{}' with disallowed extension '{}'",
                    rel_path, extension
                );
                continue;
            }

            // Extract the file to the target directory using the relative path.
            let dest = effective_dir.join(rel_path);

            // Ensure parent directories exist.
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(StorageError::Io)?;
            }

            // Second safety check: canonicalize the parent and verify the
            // resolved destination is still within the target directory.
            let canonical_base = effective_dir.canonicalize().map_err(StorageError::Io)?;
            if let Some(parent) = dest.parent() {
                let canonical_parent = parent.canonicalize().map_err(StorageError::Io)?;
                if !canonical_parent.starts_with(&canonical_base) {
                    return Err(StorageError::Serialization(format!(
                        "path traversal detected: archive entry '{}' resolves outside target directory",
                        entry_path_str,
                    )));
                }
            }

            let mut dest_file = fs::File::create(&dest).map_err(StorageError::Io)?;
            std::io::copy(&mut entry, &mut dest_file).map_err(StorageError::Io)?;

            extracted_files.push(PathBuf::from(rel_path));
        }
    }

    let manifest = manifest.ok_or_else(|| {
        StorageError::Serialization("backup archive missing manifest.json".to_string())
    })?;

    let files_restored = extracted_files.len();

    // Build a lookup table from the manifest entries for fast verification.
    let hash_map: HashMap<&str, &ManifestEntry> = manifest
        .files
        .iter()
        .map(|e| (e.path.as_str(), e))
        .collect();

    // Verify checksums if requested.
    let verified = if options.verify_checksums {
        verify_extracted_files(effective_dir, &extracted_files, &hash_map)?;
        true
    } else {
        false
    };

    log::info!(
        "Restored collection '{}': {} files, verified={}",
        manifest.collection_name,
        files_restored,
        verified,
    );

    Ok(RestoreResult {
        collection_name: manifest.collection_name,
        files_restored,
        verified,
    })
}

// ── Internals ───────────────────────────────────────────────────────────────

/// Strip the first path component (the collection-name directory prefix).
///
/// `"my_collection/segment_0.vfs"` -> `"segment_0.vfs"`
/// `"segment_0.vfs"` (no prefix) -> `"segment_0.vfs"`
fn strip_first_component(path: &str) -> &str {
    match path.find('/') {
        Some(idx) => &path[idx + 1..],
        None => path,
    }
}

/// Verify SHA-256 checksums of extracted files against the manifest entries.
fn verify_extracted_files(
    base_dir: &Path,
    extracted: &[PathBuf],
    expected: &HashMap<&str, &ManifestEntry>,
) -> StorageResult<()> {
    for rel_path in extracted {
        let rel_key = rel_path.to_string_lossy();

        if let Some(entry) = expected.get(rel_key.as_ref()) {
            let full_path = base_dir.join(rel_path);
            let actual_hex = sha256_file(&full_path)?;

            if actual_hex != entry.sha256 {
                return Err(StorageError::Serialization(format!(
                    "checksum mismatch for {}: expected {}, computed {}",
                    rel_key, entry.sha256, actual_hex
                )));
            }
        }
        // Task 273: Reject files not listed in the manifest.
        if !expected.contains_key(rel_key.as_ref()) {
            return Err(StorageError::Serialization(format!(
                "restored file '{}' is not listed in the backup manifest",
                rel_key
            )));
        }
    }

    Ok(())
}

/// Compute the hex-encoded SHA-256 digest of a file.
fn sha256_file(path: &Path) -> StorageResult<String> {
    let mut file = fs::File::open(path).map_err(StorageError::Io)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];

    loop {
        let n = file.read(&mut buf).map_err(StorageError::Io)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    Ok(hex_encode(&hasher.finalize()))
}
