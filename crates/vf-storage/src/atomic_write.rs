use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::StorageResult;

/// Build a sibling path with an extra suffix appended.
/// e.g. "hnsw.base" -> "hnsw.base.tmp", "data" -> "data.tmp"
fn sibling_path(target: &Path, suffix: &str) -> PathBuf {
    let mut name = target
        .file_name()
        .unwrap_or_default()
        .to_os_string();
    name.push(suffix);
    target.with_file_name(name)
}

/// Atomically writes data to a target file.
///
/// Protocol:
/// 1. Write to target.tmp
/// 2. fsync(target.tmp)
/// 3. If target exists, rename to target.prev (crash safety backup)
/// 4. Rename target.tmp → target
/// 5. fsync(parent directory)
pub fn atomic_write(target: &Path, data: &[u8]) -> StorageResult<()> {
    atomic_write_with_callback(target, |file| {
        file.write_all(data)?;
        Ok(())
    })
}

/// Same as `atomic_write` but uses a callback for streaming writes,
/// avoiding buffering the entire content in memory.
pub fn atomic_write_with_callback<F>(target: &Path, write_fn: F) -> StorageResult<()>
where
    F: FnOnce(&mut File) -> StorageResult<()>,
{
    let tmp_path = sibling_path(target, ".tmp");
    let prev_path = sibling_path(target, ".prev");

    // Ensure parent directory exists.
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }

    // Clean up any stale .tmp from a previous crash.
    let _ = fs::remove_file(&tmp_path);

    // 1. Write to .tmp
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)?;

    let result = write_fn(&mut file);
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    // 2. fsync the tmp file.
    file.sync_all()?;
    drop(file);

    // 3. If target exists, rename to .prev
    if target.exists() {
        let _ = fs::remove_file(&prev_path);
        fs::rename(target, &prev_path)?;
    }

    // 4. Rename .tmp → target
    if let Err(e) = fs::rename(&tmp_path, target) {
        // Try to restore .prev
        if prev_path.exists() {
            let _ = fs::rename(&prev_path, target);
        }
        return Err(e.into());
    }

    // 5. fsync parent directory for rename durability.
    if let Some(parent) = target.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

/// Recovers the previous version of a file if the current one is missing or corrupt.
/// Also cleans up stale .tmp files from interrupted writes.
/// Returns true if recovery from .prev happened.
pub fn recover_from_prev(target: &Path) -> StorageResult<bool> {
    let tmp_path = sibling_path(target, ".tmp");
    let prev_path = sibling_path(target, ".prev");

    // Clean up stale .tmp from an interrupted write.
    let _ = fs::remove_file(&tmp_path);

    if !target.exists() && prev_path.exists() {
        fs::rename(&prev_path, target)?;
        if let Some(parent) = target.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        return Ok(true);
    }

    Ok(false)
}
