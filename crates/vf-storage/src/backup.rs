// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Collection backup: snapshot segments, config, and WAL into a tar.gz archive
//! with a SHA-256 manifest for integrity verification.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::write::GzEncoder;
use flate2::Compression;
use sha2::{Digest, Sha256};
use tar::Builder;

use crate::error::{StorageError, StorageResult};
use crate::util::hex_encode;

/// Options controlling how a backup is created.
#[derive(Debug, Clone)]
pub struct BackupOptions {
    /// Flate2 compression level (0-9). Default: 6.
    pub compression_level: u32,
    /// Whether to include WAL files in the backup. Default: true.
    pub include_wal: bool,
}

impl Default for BackupOptions {
    fn default() -> Self {
        Self {
            compression_level: 6,
            include_wal: true,
        }
    }
}

/// An entry in the backup manifest recording a file and its SHA-256 hash.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ManifestEntry {
    /// Relative path inside the archive.
    pub path: String,
    /// Hex-encoded SHA-256 hash of the file contents.
    pub sha256: String,
    /// File size in bytes.
    pub size: u64,
}

/// Manifest describing the contents of a backup archive.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackupManifest {
    /// Name of the backed-up collection.
    pub collection_name: String,
    /// Unix timestamp (seconds) when the backup was created.
    pub timestamp: u64,
    /// All files included in the archive with their hashes.
    pub files: Vec<ManifestEntry>,
    /// Path to the generated archive file.
    pub archive_path: PathBuf,
}

/// Compute the hex-encoded SHA-256 hash of a byte slice.
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex_encode(&result)
}

/// Collect files from `collection_dir` that should be included in the backup.
///
/// Returns a sorted map of (relative_path -> absolute_path).
fn collect_backup_files(
    collection_dir: &Path,
    include_wal: bool,
) -> StorageResult<BTreeMap<String, PathBuf>> {
    let mut files = BTreeMap::new();

    let entries = fs::read_dir(collection_dir).map_err(StorageError::Io)?;

    for entry in entries {
        let entry = entry.map_err(StorageError::Io)?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let file_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };

        // Include segment files (*.vfs)
        if file_name.ends_with(".vfs") {
            files.insert(file_name, path);
            continue;
        }

        // Include config
        if file_name == "config.json" {
            files.insert(file_name, path);
            continue;
        }

        // Include WAL files if requested
        if include_wal && (file_name.ends_with(".log") || file_name.starts_with("wal")) {
            files.insert(file_name, path);
            continue;
        }
    }

    Ok(files)
}

/// Create a backup of a collection directory.
///
/// Reads all segment files, `config.json`, and optionally WAL files from
/// `collection_dir`. Computes SHA-256 hashes, packages everything into a
/// tar.gz archive in `output_dir`, and returns a [`BackupManifest`].
///
/// The archive also contains a `manifest.json` file with hash information
/// for integrity verification.
pub fn create_backup(
    collection_dir: &Path,
    output_dir: &Path,
    options: &BackupOptions,
) -> StorageResult<BackupManifest> {
    // Validate inputs
    if options.compression_level > 9 {
        return Err(StorageError::Serialization(format!(
            "invalid compression level {}: must be 0-9",
            options.compression_level
        )));
    }

    if !collection_dir.is_dir() {
        return Err(StorageError::CollectionNotFound(
            collection_dir.display().to_string(),
        ));
    }

    // Ensure output directory exists
    fs::create_dir_all(output_dir).map_err(StorageError::Io)?;

    // Derive collection name from directory name
    let collection_name = collection_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Collect files to back up
    let backup_files = collect_backup_files(collection_dir, options.include_wal)?;

    if backup_files.is_empty() {
        return Err(StorageError::CollectionNotFound(format!(
            "no files found in {}",
            collection_dir.display()
        )));
    }

    // Generate timestamp
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Build archive filename
    let archive_name = format!("{collection_name}_{timestamp}.tar.gz");
    let archive_path = output_dir.join(&archive_name);

    // Read all file contents and compute hashes
    let mut file_contents: Vec<(String, Vec<u8>)> = Vec::new();
    let mut manifest_entries: Vec<ManifestEntry> = Vec::new();

    for (rel_path, abs_path) in &backup_files {
        let mut buf = Vec::new();
        let mut f = File::open(abs_path).map_err(StorageError::Io)?;
        f.read_to_end(&mut buf).map_err(StorageError::Io)?;

        let hash = sha256_hex(&buf);
        let size = buf.len() as u64;

        manifest_entries.push(ManifestEntry {
            path: rel_path.clone(),
            sha256: hash,
            size,
        });

        file_contents.push((rel_path.clone(), buf));
    }

    // Build the manifest (without archive_path, which we set after)
    let manifest = BackupManifest {
        collection_name: collection_name.clone(),
        timestamp,
        files: manifest_entries,
        archive_path: archive_path.clone(),
    };

    // Serialize manifest to JSON
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;

    // Create the tar.gz archive
    let gz_file = File::create(&archive_path).map_err(StorageError::Io)?;
    let compression = Compression::new(options.compression_level);
    let gz_encoder = GzEncoder::new(gz_file, compression);
    let mut tar_builder = Builder::new(gz_encoder);

    // Add each collected file to the archive
    for (rel_path, data) in &file_contents {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();

        let archive_entry_path = format!("{collection_name}/{rel_path}");
        tar_builder
            .append_data(&mut header, &archive_entry_path, data.as_slice())
            .map_err(StorageError::Io)?;
    }

    // Add manifest.json to the archive
    {
        let manifest_bytes = manifest_json.as_bytes();
        let mut header = tar::Header::new_gnu();
        header.set_size(manifest_bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();

        let manifest_path = format!("{collection_name}/manifest.json");
        tar_builder
            .append_data(&mut header, &manifest_path, manifest_bytes)
            .map_err(StorageError::Io)?;
    }

    // Finalize the archive
    let gz_encoder = tar_builder
        .into_inner()
        .map_err(StorageError::Io)?;
    gz_encoder.finish().map_err(StorageError::Io)?;

    log::info!(
        "Backup created for collection '{}': {} files, archive at {}",
        collection_name,
        manifest.files.len(),
        archive_path.display()
    );

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_hex() {
        let hash = sha256_hex(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0xab]), "00ffab");
    }

    #[test]
    fn test_backup_options_default() {
        let opts = BackupOptions::default();
        assert_eq!(opts.compression_level, 6);
        assert!(opts.include_wal);
    }
}
