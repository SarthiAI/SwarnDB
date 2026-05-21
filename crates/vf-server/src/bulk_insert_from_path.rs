// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! TOCTOU-safe path validation and mmap-based file open for the
//! BulkInsertFromPath endpoint (ADR-001, Decisions 4 and 5).
//!
//! The handler in grpc_vector.rs / rest.rs opens every allow-list root
//! once, hands their raw fds to `open_validated`, and gets back an owned
//! File, an Mmap, and a DatasetView describing the dim, count, and
//! header_offset. The handler then casts the mmap'd byte slice to
//! `&[f32]` via bytemuck and slices it into rows. The Mmap must outlive
//! every `&[f32]` borrow; the handler holds it on its stack frame across
//! the full HNSW insert call.

use std::ffi::OsStr;
use std::fs::File;
use std::io::Cursor;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};

use memmap2::Mmap;
use nix::fcntl::{openat, OFlag};
use nix::sys::stat::Mode;
use npyz::NpyFile;

/// Lightweight view returned by `open_validated`. The caller (handler)
/// owns the File + Mmap; this view describes how to slice the mmap'd
/// bytes into f32 rows.
pub struct DatasetView {
    pub dim: usize,
    pub count: usize,
    pub header_offset: usize,
}

/// Owner of the ids-file mapping. Held on the handler's stack so the
/// borrowed `Vec<u64>` returned by `open_ids_validated` keeps its backing
/// file mapped for the duration of the bulk insert call.
#[allow(dead_code)]
pub enum IdsSource {
    Sequential,
    Mmap { _file: File, _mmap: Mmap },
}

/// Errors surfaced by this module. All variants carry inline context; the
/// gRPC / REST handlers map each variant to its transport-appropriate
/// status code via `map_bifp_error` and `map_bifp_error_rest`.
#[derive(Debug)]
pub enum BulkFromPathError {
    Io { source: std::io::Error, context: String },
    PathDenied { path: String },
    RelativePath { path: String },
    TraversalAttempt { path: String, component: String },
    NullByte { path: String },
    BadMagic { reason: String },
    DimensionMismatch { expected: usize, got: usize },
    CountMismatch { expected: u64, got: u64 },
    MmapFailed { source: std::io::Error, path: String },
}

impl std::fmt::Display for BulkFromPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { source, context } => write!(f, "io error ({}): {}", context, source),
            Self::PathDenied { path } => {
                write!(f, "path '{}' is not under any configured allow-list root", path)
            }
            Self::RelativePath { path } => {
                write!(f, "path '{}' must be absolute", path)
            }
            Self::TraversalAttempt { path, component } => write!(
                f,
                "path '{}' contains forbidden component '{}'",
                path, component
            ),
            Self::NullByte { path } => {
                write!(f, "path '{}' contains a NUL byte", path)
            }
            Self::BadMagic { reason } => write!(f, "file header rejected: {}", reason),
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "vector dimension mismatch: expected {}, got {}",
                expected, got
            ),
            Self::CountMismatch { expected, got } => write!(
                f,
                "vector count mismatch: expected {}, got {}",
                expected, got
            ),
            Self::MmapFailed { source, path } => {
                write!(f, "mmap of '{}' failed: {}", path, source)
            }
        }
    }
}

impl std::error::Error for BulkFromPathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::MmapFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Open one allow-list root directory. The returned File keeps the
/// directory fd alive; the handler should hold every root File on its
/// stack for the duration of the openat walk.
pub fn open_root(root_path: &Path) -> Result<File, BulkFromPathError> {
    File::open(root_path).map_err(|e| BulkFromPathError::Io {
        source: e,
        context: format!("open allow-list root '{}'", root_path.display()),
    })
}

/// TOCTOU-safe open + mmap of a user-supplied data file path.
///
/// `expected_dim` of 0 means "trust the .npy header" (only valid when the
/// file is .npy). `expected_count` of 0 means "infer from file size".
pub fn open_validated(
    user_path: &str,
    root_fds: &[RawFd],
    expected_dim: usize,
    expected_count: u64,
) -> Result<(File, Mmap, DatasetView), BulkFromPathError> {
    let file = open_file_under_roots(user_path, root_fds)?;
    let metadata = file.metadata().map_err(|e| BulkFromPathError::Io {
        source: e,
        context: format!("stat '{}'", user_path),
    })?;
    let file_size = metadata.len() as usize;

    // SAFETY: `file` is an open read-only fd we hold for the full
    // lifetime of `mmap`; the underlying file is not concurrently
    // truncated or written while the server is up because the allow-list
    // root is reserved for read-only ingest staging.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| BulkFromPathError::MmapFailed {
        source: e,
        path: user_path.to_string(),
    })?;

    let view = detect_view(&mmap, file_size, expected_dim, expected_count)?;
    Ok((file, mmap, view))
}

/// TOCTOU-safe open + mmap of a user-supplied ids file. Returns the
/// File, the Mmap, and a Vec<u64> parsed from the mapped bytes. The
/// file is expected to be a flat int64 buffer of `expected_count` ids;
/// a .npy magic is accepted and decoded transparently with dtype `<i8`.
pub fn open_ids_validated(
    user_path: &str,
    root_fds: &[RawFd],
    expected_count: usize,
) -> Result<(File, Mmap, Vec<u64>), BulkFromPathError> {
    let file = open_file_under_roots(user_path, root_fds)?;
    let metadata = file.metadata().map_err(|e| BulkFromPathError::Io {
        source: e,
        context: format!("stat '{}'", user_path),
    })?;
    let file_size = metadata.len() as usize;

    // SAFETY: see open_validated above; same invariant on the ids file.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| BulkFromPathError::MmapFailed {
        source: e,
        path: user_path.to_string(),
    })?;

    let header_offset = if mmap.len() >= NPY_MAGIC.len() && &mmap[..NPY_MAGIC.len()] == &NPY_MAGIC[..] {
        let mut cursor = Cursor::new(&mmap[..]);
        let npy = NpyFile::new(&mut cursor).map_err(|e| BulkFromPathError::BadMagic {
            reason: format!("ids .npy parse failed: {}", e),
        })?;
        let dtype_descr = npy.dtype().descr();
        // Accept both the quoted descr form ('<i8') and the bare form (<i8)
        // because npyz Display vs descr() rendering has varied across patch
        // releases; the dtype semantics are identical.
        if dtype_descr != "'<i8'" && dtype_descr != "<i8" && dtype_descr != "'|i8'" && dtype_descr != "|i8" {
            return Err(BulkFromPathError::BadMagic {
                reason: format!("ids .npy dtype must be '<i8' or '|i8', got {}", dtype_descr),
            });
        }
        if npy.order() != npyz::Order::C {
            return Err(BulkFromPathError::BadMagic {
                reason: "ids .npy must be C-order".to_string(),
            });
        }
        let shape = npy.shape().to_vec();
        if shape.len() != 1 {
            return Err(BulkFromPathError::BadMagic {
                reason: format!("ids .npy must be 1-D, got shape {:?}", shape),
            });
        }
        let got_count = shape[0] as usize;
        if got_count != expected_count {
            return Err(BulkFromPathError::CountMismatch {
                expected: expected_count as u64,
                got: got_count as u64,
            });
        }
        drop(npy);
        cursor.position() as usize
    } else {
        let row_bytes = file_size;
        let elem_bytes = std::mem::size_of::<i64>();
        if row_bytes % elem_bytes != 0 {
            return Err(BulkFromPathError::BadMagic {
                reason: format!(
                    "flat ids file size {} not a multiple of {} bytes",
                    row_bytes, elem_bytes
                ),
            });
        }
        let got_count = row_bytes / elem_bytes;
        if got_count != expected_count {
            return Err(BulkFromPathError::CountMismatch {
                expected: expected_count as u64,
                got: got_count as u64,
            });
        }
        0
    };

    let payload = &mmap[header_offset..];
    let needed = expected_count
        .saturating_mul(std::mem::size_of::<i64>());
    if payload.len() < needed {
        return Err(BulkFromPathError::BadMagic {
            reason: format!(
                "ids payload is {} bytes, expected at least {}",
                payload.len(),
                needed
            ),
        });
    }
    let raw: &[i64] = bytemuck::cast_slice(&payload[..needed]);
    let ids: Vec<u64> = raw.iter().map(|&v| v as u64).collect();

    Ok((file, mmap, ids))
}

// -- internals -----------------------------------------------------------

fn detect_view(
    mmap: &Mmap,
    file_size: usize,
    expected_dim: usize,
    expected_count: u64,
) -> Result<DatasetView, BulkFromPathError> {
    if mmap.len() >= NPY_MAGIC.len() && &mmap[..NPY_MAGIC.len()] == &NPY_MAGIC[..] {
        return parse_npy_view(mmap, expected_dim, expected_count);
    }
    parse_flat_view(file_size, expected_dim, expected_count)
}

fn parse_npy_view(
    mmap: &Mmap,
    expected_dim: usize,
    expected_count: u64,
) -> Result<DatasetView, BulkFromPathError> {
    let mut cursor = Cursor::new(&mmap[..]);
    let npy = NpyFile::new(&mut cursor).map_err(|e| BulkFromPathError::BadMagic {
        reason: format!(".npy parse failed: {}", e),
    })?;

    let dtype_descr = npy.dtype().descr();
    // Accept both the quoted descr form ('<f4') and the bare form (<f4)
    // because npyz Display vs descr() rendering has varied across patch
    // releases; the dtype semantics are identical.
    if dtype_descr != "'<f4'" && dtype_descr != "<f4" && dtype_descr != "'|f4'" && dtype_descr != "|f4" {
        return Err(BulkFromPathError::BadMagic {
            reason: format!(".npy dtype must be '<f4' or '|f4', got {}", dtype_descr),
        });
    }
    if npy.order() != npyz::Order::C {
        return Err(BulkFromPathError::BadMagic {
            reason: ".npy must be C-order (not Fortran)".to_string(),
        });
    }
    let shape = npy.shape().to_vec();
    if shape.len() != 2 {
        return Err(BulkFromPathError::BadMagic {
            reason: format!(
                ".npy vector file must be 2-D (rows, dim); got shape {:?}",
                shape
            ),
        });
    }

    let count = shape[0] as usize;
    let dim = shape[1] as usize;

    if expected_dim != 0 && expected_dim != dim {
        return Err(BulkFromPathError::DimensionMismatch {
            expected: expected_dim,
            got: dim,
        });
    }
    if expected_count != 0 && expected_count != count as u64 {
        return Err(BulkFromPathError::CountMismatch {
            expected: expected_count,
            got: count as u64,
        });
    }

    drop(npy);
    let header_offset = cursor.position() as usize;
    Ok(DatasetView {
        dim,
        count,
        header_offset,
    })
}

fn parse_flat_view(
    file_size: usize,
    expected_dim: usize,
    expected_count: u64,
) -> Result<DatasetView, BulkFromPathError> {
    if expected_dim == 0 {
        return Err(BulkFromPathError::BadMagic {
            reason: "flat .f32 file requires non-zero dim in the request".to_string(),
        });
    }
    let row_bytes = expected_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| BulkFromPathError::BadMagic {
            reason: format!("dim {} overflows row size", expected_dim),
        })?;
    if row_bytes == 0 {
        return Err(BulkFromPathError::BadMagic {
            reason: "row size is zero".to_string(),
        });
    }
    if file_size % row_bytes != 0 {
        return Err(BulkFromPathError::BadMagic {
            reason: format!(
                "flat file size {} not a multiple of row size {} ({} * 4)",
                file_size, row_bytes, expected_dim
            ),
        });
    }
    let count = file_size / row_bytes;
    if expected_count != 0 && expected_count != count as u64 {
        return Err(BulkFromPathError::CountMismatch {
            expected: expected_count,
            got: count as u64,
        });
    }
    Ok(DatasetView {
        dim: expected_dim,
        count,
        header_offset: 0,
    })
}

fn open_file_under_roots(
    user_path: &str,
    root_fds: &[RawFd],
) -> Result<File, BulkFromPathError> {
    let path = Path::new(user_path);
    let components = validate_components(user_path, path)?;
    let mut last_err: Option<BulkFromPathError> = None;
    for &root_fd in root_fds {
        match openat_walk(root_fd, &components) {
            Ok(file) => return Ok(file),
            Err(e) => last_err = Some(e),
        }
    }
    if let Some(e) = last_err {
        // If the failure was strictly a not-found / not-permitted on
        // every root, report it as PathDenied so the handler maps it to
        // 403. Genuine IO surprises (EIO, etc) get passed through as-is.
        match e {
            BulkFromPathError::Io { ref source, .. }
                if matches!(
                    source.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
                ) =>
            {
                return Err(BulkFromPathError::PathDenied {
                    path: user_path.to_string(),
                });
            }
            other => return Err(other),
        }
    }
    Err(BulkFromPathError::PathDenied {
        path: user_path.to_string(),
    })
}

fn validate_components<'a>(
    user_path: &str,
    path: &'a Path,
) -> Result<Vec<&'a OsStr>, BulkFromPathError> {
    if !path.is_absolute() {
        return Err(BulkFromPathError::RelativePath {
            path: user_path.to_string(),
        });
    }
    let mut out: Vec<&OsStr> = Vec::new();
    for comp in path.components() {
        match comp {
            Component::RootDir | Component::Prefix(_) => continue,
            Component::CurDir => {
                return Err(BulkFromPathError::TraversalAttempt {
                    path: user_path.to_string(),
                    component: ".".to_string(),
                });
            }
            Component::ParentDir => {
                return Err(BulkFromPathError::TraversalAttempt {
                    path: user_path.to_string(),
                    component: "..".to_string(),
                });
            }
            Component::Normal(name) => {
                let bytes = name.as_bytes();
                if bytes.is_empty() {
                    return Err(BulkFromPathError::TraversalAttempt {
                        path: user_path.to_string(),
                        component: String::new(),
                    });
                }
                if bytes.contains(&0) {
                    return Err(BulkFromPathError::NullByte {
                        path: user_path.to_string(),
                    });
                }
                out.push(name);
            }
        }
    }
    if out.is_empty() {
        return Err(BulkFromPathError::TraversalAttempt {
            path: user_path.to_string(),
            component: "<empty>".to_string(),
        });
    }
    Ok(out)
}

fn openat_walk(root_fd: RawFd, components: &[&OsStr]) -> Result<File, BulkFromPathError> {
    let (leaf, interior) = components.split_last().expect("validated non-empty");

    // Step through every interior directory component with O_NOFOLLOW so
    // a swapped symlink can never escape the allow-list root.
    let mut interior_files: Vec<File> = Vec::with_capacity(interior.len());
    let mut parent_fd: RawFd = root_fd;
    for &comp in interior {
        let comp_path: &Path = Path::new(comp);
        let raw = openat(
            Some(parent_fd),
            comp_path,
            OFlag::O_RDONLY
                | OFlag::O_DIRECTORY
                | OFlag::O_NOFOLLOW
                | OFlag::O_CLOEXEC,
            Mode::empty(),
        )
        .map_err(|errno| io_from_errno(errno, comp))?;
        // SAFETY: `raw` was just returned by openat and is not owned by
        // any other File; wrapping it in File is the standard idiom for
        // taking ownership of the fd so it closes on drop.
        let f = unsafe { File::from_raw_fd(raw) };
        parent_fd = f.as_raw_fd();
        interior_files.push(f);
    }

    let leaf_path: &Path = Path::new(*leaf);
    let leaf_raw = openat(
        Some(parent_fd),
        leaf_path,
        OFlag::O_RDONLY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
        Mode::empty(),
    )
    .map_err(|errno| io_from_errno(errno, *leaf))?;
    // SAFETY: same as above; `leaf_raw` is freshly returned by openat
    // and has no other owner.
    let file = unsafe { File::from_raw_fd(leaf_raw) };
    drop(interior_files);
    Ok(file)
}

fn io_from_errno(errno: nix::errno::Errno, component: &OsStr) -> BulkFromPathError {
    let context = format!("openat component '{}'", component.to_string_lossy());
    BulkFromPathError::Io {
        source: std::io::Error::from_raw_os_error(errno as i32),
        context,
    }
}
