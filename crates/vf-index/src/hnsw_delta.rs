// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! HNSW Delta Log - append-only mutation journal between base snapshots.
//!
//! Records graph mutations (add/remove node, set neighbors, set entry point)
//! as compact binary entries. Recovery = load base snapshot + replay delta.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crc32fast::Hasher as Crc32Hasher;
use vf_core::types::VectorId;

use crate::hnsw_persistence::{HnswTopologySnapshot, TopologyNode};
use crate::traits::IndexError;

// ── Constants ───────────────────────────────────────────────────────────

/// Magic bytes identifying the HNSW delta log format.
const DELTA_MAGIC: &[u8; 4] = b"HDLT";
/// Current delta format version.
const DELTA_VERSION: u8 = 1;
/// Op-type tag for AddNode.
const OP_ADD_NODE: u8 = 0;
/// Op-type tag for RemoveNode.
const OP_REMOVE_NODE: u8 = 1;
/// Op-type tag for SetNeighbors.
const OP_SET_NEIGHBORS: u8 = 2;
/// Op-type tag for SetEntryPoint.
const OP_SET_ENTRY_POINT: u8 = 3;

// ── Delta types ─────────────────────────────────────────────────────────

/// Operations that can be recorded in the HNSW delta log.
#[derive(Clone, Debug)]
pub enum HnswDeltaOp {
    AddNode {
        id: VectorId,
        level: u32,
        vector_slot: u64,
        neighbors_per_layer: Vec<Vec<VectorId>>,
    },
    RemoveNode {
        id: VectorId,
    },
    SetNeighbors {
        node_id: VectorId,
        layer: u32,
        neighbors: Vec<VectorId>,
    },
    SetEntryPoint {
        id: VectorId,
        level: u32,
    },
}

/// A single delta entry with its WAL LSN.
#[derive(Clone, Debug)]
pub struct HnswDeltaEntry {
    pub lsn: u64,
    pub op: HnswDeltaOp,
}

// ── Serialization helpers ───────────────────────────────────────────────

/// Serialize a delta op into its payload bytes.
fn serialize_op(op: &HnswDeltaOp) -> (u8, Vec<u8>) {
    match op {
        HnswDeltaOp::AddNode {
            id,
            level,
            vector_slot,
            neighbors_per_layer,
        } => {
            let num_layers = neighbors_per_layer.len() as u32;
            // Estimate: 8 + 4 + 8 + 4 + per-layer overhead
            let mut buf = Vec::with_capacity(24 + num_layers as usize * 36);
            buf.extend_from_slice(&id.to_le_bytes());
            buf.extend_from_slice(&level.to_le_bytes());
            buf.extend_from_slice(&vector_slot.to_le_bytes());
            buf.extend_from_slice(&num_layers.to_le_bytes());
            for layer in neighbors_per_layer {
                buf.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for &nid in layer {
                    buf.extend_from_slice(&nid.to_le_bytes());
                }
            }
            (OP_ADD_NODE, buf)
        }
        HnswDeltaOp::RemoveNode { id } => {
            let mut buf = Vec::with_capacity(8);
            buf.extend_from_slice(&id.to_le_bytes());
            (OP_REMOVE_NODE, buf)
        }
        HnswDeltaOp::SetNeighbors {
            node_id,
            layer,
            neighbors,
        } => {
            let mut buf = Vec::with_capacity(16 + neighbors.len() * 8);
            buf.extend_from_slice(&node_id.to_le_bytes());
            buf.extend_from_slice(&layer.to_le_bytes());
            buf.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
            for &nid in neighbors {
                buf.extend_from_slice(&nid.to_le_bytes());
            }
            (OP_SET_NEIGHBORS, buf)
        }
        HnswDeltaOp::SetEntryPoint { id, level } => {
            let mut buf = Vec::with_capacity(12);
            buf.extend_from_slice(&id.to_le_bytes());
            buf.extend_from_slice(&level.to_le_bytes());
            (OP_SET_ENTRY_POINT, buf)
        }
    }
}

/// Deserialize a delta op from its type tag and payload bytes.
fn deserialize_op(op_type: u8, payload: &[u8]) -> Result<HnswDeltaOp, IndexError> {
    let mut pos = 0;

    let read_u32 = |pos: &mut usize, payload: &[u8]| -> Result<u32, IndexError> {
        if *pos + 4 > payload.len() {
            return Err(IndexError::Internal(
                "delta payload: unexpected end (u32)".into(),
            ));
        }
        let val = u32::from_le_bytes(payload[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(val)
    };

    let read_u64 = |pos: &mut usize, payload: &[u8]| -> Result<u64, IndexError> {
        if *pos + 8 > payload.len() {
            return Err(IndexError::Internal(
                "delta payload: unexpected end (u64)".into(),
            ));
        }
        let val = u64::from_le_bytes(payload[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        Ok(val)
    };

    match op_type {
        OP_ADD_NODE => {
            let id = read_u64(&mut pos, payload)?;
            let level = read_u32(&mut pos, payload)?;
            let vector_slot = read_u64(&mut pos, payload)?;
            let num_layers = read_u32(&mut pos, payload)? as usize;
            let mut neighbors_per_layer = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let count = read_u32(&mut pos, payload)? as usize;
                let mut layer = Vec::with_capacity(count);
                for _ in 0..count {
                    layer.push(read_u64(&mut pos, payload)?);
                }
                neighbors_per_layer.push(layer);
            }
            Ok(HnswDeltaOp::AddNode {
                id,
                level,
                vector_slot,
                neighbors_per_layer,
            })
        }
        OP_REMOVE_NODE => {
            let id = read_u64(&mut pos, payload)?;
            Ok(HnswDeltaOp::RemoveNode { id })
        }
        OP_SET_NEIGHBORS => {
            let node_id = read_u64(&mut pos, payload)?;
            let layer = read_u32(&mut pos, payload)?;
            let count = read_u32(&mut pos, payload)? as usize;
            let mut neighbors = Vec::with_capacity(count);
            for _ in 0..count {
                neighbors.push(read_u64(&mut pos, payload)?);
            }
            Ok(HnswDeltaOp::SetNeighbors {
                node_id,
                layer,
                neighbors,
            })
        }
        OP_SET_ENTRY_POINT => {
            let id = read_u64(&mut pos, payload)?;
            let level = read_u32(&mut pos, payload)?;
            Ok(HnswDeltaOp::SetEntryPoint { id, level })
        }
        _ => Err(IndexError::Internal(format!(
            "delta log: unknown op_type {}",
            op_type
        ))),
    }
}

// ── Writer ──────────────────────────────────────────────────────────────

/// Append-only writer for HNSW delta entries.
pub struct HnswDeltaWriter {
    file: BufWriter<File>,
    path: PathBuf,
    entry_count: u64,
    last_lsn: u64,
}

impl HnswDeltaWriter {
    /// Create a new delta file with header.
    pub fn create(path: &Path) -> Result<Self, IndexError> {
        let wrap = |e: io::Error| IndexError::Internal(format!("delta create: {}", e));

        let file = File::create(path).map_err(wrap)?;
        let mut writer = BufWriter::new(file);

        // Write header: magic + version.
        writer.write_all(DELTA_MAGIC).map_err(wrap)?;
        writer.write_all(&[DELTA_VERSION]).map_err(wrap)?;
        writer.flush().map_err(wrap)?;

        Ok(Self {
            file: writer,
            path: path.to_path_buf(),
            entry_count: 0,
            last_lsn: 0,
        })
    }

    /// Open existing delta file for append. Scans to find entry_count and last_lsn.
    pub fn open(path: &Path) -> Result<Self, IndexError> {
        let wrap = |e: io::Error| IndexError::Internal(format!("delta open: {}", e));

        // First, scan the file to count entries and find last LSN.
        let (entry_count, last_lsn) = {
            let reader = HnswDeltaReader::open(path)?;
            let mut count = 0u64;
            let mut lsn = 0u64;
            for entry_result in reader {
                let entry = entry_result?;
                lsn = entry.lsn;
                count += 1;
            }
            (count, lsn)
        };

        // Open for append.
        let file = OpenOptions::new().append(true).open(path).map_err(wrap)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            file: writer,
            path: path.to_path_buf(),
            entry_count,
            last_lsn,
        })
    }

    /// Append a delta entry. Returns the entry count after append.
    pub fn append(&mut self, entry: &HnswDeltaEntry) -> Result<u64, IndexError> {
        let wrap = |e: io::Error| IndexError::Internal(format!("delta append: {}", e));

        let (op_type, payload) = serialize_op(&entry.op);
        let payload_len = payload.len() as u32;

        // Build the record bytes for CRC: [lsn][op_type][payload_len][payload]
        let record_size = 8 + 1 + 4 + payload.len();
        let mut record = Vec::with_capacity(record_size);
        record.extend_from_slice(&entry.lsn.to_le_bytes());
        record.push(op_type);
        record.extend_from_slice(&payload_len.to_le_bytes());
        record.extend_from_slice(&payload);

        // CRC32 over the record.
        let mut hasher = Crc32Hasher::new();
        hasher.update(&record);
        let crc = hasher.finalize();

        // Write record + CRC.
        self.file.write_all(&record).map_err(wrap)?;
        self.file.write_all(&crc.to_le_bytes()).map_err(wrap)?;

        self.entry_count += 1;
        self.last_lsn = entry.lsn;

        Ok(self.entry_count)
    }

    /// Sync to disk.
    pub fn sync(&mut self) -> Result<(), IndexError> {
        let wrap = |e: io::Error| IndexError::Internal(format!("delta sync: {}", e));
        self.file.flush().map_err(wrap)?;
        self.file.get_ref().sync_all().map_err(wrap)?;
        Ok(())
    }

    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    pub fn last_lsn(&self) -> u64 {
        self.last_lsn
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the file size.
    pub fn file_size(&self) -> Result<u64, IndexError> {
        let wrap = |e: io::Error| IndexError::Internal(format!("delta file_size: {}", e));
        let metadata = self.file.get_ref().metadata().map_err(wrap)?;
        Ok(metadata.len())
    }
}

// ── Reader ──────────────────────────────────────────────────────────────

/// Sequential reader for HNSW delta entries.
pub struct HnswDeltaReader {
    reader: BufReader<File>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl HnswDeltaReader {
    pub fn open(path: &Path) -> Result<Self, IndexError> {
        let wrap = |e: io::Error| IndexError::Internal(format!("delta reader open: {}", e));

        let file = File::open(path).map_err(wrap)?;
        let mut reader = BufReader::new(file);

        // Validate header.
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(wrap)?;
        if &magic != DELTA_MAGIC {
            return Err(IndexError::Internal(
                "delta log: invalid magic bytes".into(),
            ));
        }

        let mut version = [0u8; 1];
        reader.read_exact(&mut version).map_err(wrap)?;
        if version[0] != DELTA_VERSION {
            return Err(IndexError::Internal(format!(
                "delta log: unsupported version {}",
                version[0]
            )));
        }

        Ok(Self {
            reader,
            path: path.to_path_buf(),
        })
    }

    /// Read the next entry, or None if EOF / truncated entry (crash recovery).
    pub fn next_entry(&mut self) -> Result<Option<HnswDeltaEntry>, IndexError> {
        // Read LSN (8 bytes). If we can't read a full LSN, treat as EOF.
        let mut lsn_buf = [0u8; 8];
        match self.reader.read_exact(&mut lsn_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => {
                return Err(IndexError::Internal(format!("delta read lsn: {}", e)));
            }
        }
        let lsn = u64::from_le_bytes(lsn_buf);

        // Read op_type (1 byte).
        let mut op_type_buf = [0u8; 1];
        match self.reader.read_exact(&mut op_type_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => {
                return Err(IndexError::Internal(format!("delta read op_type: {}", e)));
            }
        }
        let op_type = op_type_buf[0];

        // Read payload_len (4 bytes).
        let mut plen_buf = [0u8; 4];
        match self.reader.read_exact(&mut plen_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => {
                return Err(IndexError::Internal(format!(
                    "delta read payload_len: {}",
                    e
                )));
            }
        }
        let payload_len = u32::from_le_bytes(plen_buf) as usize;

        // Read payload.
        let mut payload = vec![0u8; payload_len];
        match self.reader.read_exact(&mut payload) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => {
                return Err(IndexError::Internal(format!("delta read payload: {}", e)));
            }
        }

        // Read CRC (4 bytes).
        let mut crc_buf = [0u8; 4];
        match self.reader.read_exact(&mut crc_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => {
                return Err(IndexError::Internal(format!("delta read crc: {}", e)));
            }
        }
        let stored_crc = u32::from_le_bytes(crc_buf);

        // Verify CRC over [lsn][op_type][payload_len][payload].
        let mut hasher = Crc32Hasher::new();
        hasher.update(&lsn_buf);
        hasher.update(&op_type_buf);
        hasher.update(&plen_buf);
        hasher.update(&payload);
        let computed_crc = hasher.finalize();

        if stored_crc != computed_crc {
            // Treat CRC mismatch as truncated/corrupt entry (crash recovery).
            return Ok(None);
        }

        // Deserialize the operation.
        let op = deserialize_op(op_type, &payload)?;

        Ok(Some(HnswDeltaEntry { lsn, op }))
    }
}

impl Iterator for HnswDeltaReader {
    type Item = Result<HnswDeltaEntry, IndexError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_entry() {
            Ok(Some(entry)) => Some(Ok(entry)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// ── Replay functions ────────────────────────────────────────────────────

/// Apply a single delta op to a topology snapshot.
fn apply_op(snapshot: &mut HnswTopologySnapshot, op: &HnswDeltaOp) {
    match op {
        HnswDeltaOp::AddNode {
            id,
            level,
            vector_slot,
            neighbors_per_layer,
        } => {
            snapshot.nodes.push(TopologyNode {
                id: *id,
                level: *level as usize,
                vector_slot: *vector_slot as usize,
                neighbors: neighbors_per_layer.clone(),
            });
        }
        HnswDeltaOp::RemoveNode { id } => {
            // Remove the node itself.
            snapshot.nodes.retain(|n| n.id != *id);
            // Remove from all other nodes' neighbor lists.
            for node in &mut snapshot.nodes {
                for layer in &mut node.neighbors {
                    layer.retain(|nid| *nid != *id);
                }
            }
            // Clear entry point if it was this node.
            if snapshot.entry_point == Some(*id) {
                snapshot.entry_point = None;
            }
        }
        HnswDeltaOp::SetNeighbors {
            node_id,
            layer,
            neighbors,
        } => {
            if let Some(node) = snapshot.nodes.iter_mut().find(|n| n.id == *node_id) {
                let layer_idx = *layer as usize;
                // Extend the neighbors vec if needed.
                while node.neighbors.len() <= layer_idx {
                    node.neighbors.push(Vec::new());
                }
                node.neighbors[layer_idx] = neighbors.clone();
            }
        }
        HnswDeltaOp::SetEntryPoint { id, level } => {
            snapshot.entry_point = Some(*id);
            snapshot.max_level = *level;
        }
    }
}

/// Replay all delta entries onto a topology snapshot.
/// Returns the last applied LSN (or 0 if no entries).
pub fn replay_delta(
    snapshot: &mut HnswTopologySnapshot,
    delta_path: &Path,
) -> Result<u64, IndexError> {
    let reader = HnswDeltaReader::open(delta_path)?;
    let mut last_lsn = 0u64;

    for entry_result in reader {
        let entry = entry_result?;
        apply_op(snapshot, &entry.op);
        last_lsn = entry.lsn;
    }

    Ok(last_lsn)
}

/// Replay only entries with LSN > after_lsn.
pub fn replay_delta_after_lsn(
    snapshot: &mut HnswTopologySnapshot,
    delta_path: &Path,
    after_lsn: u64,
) -> Result<u64, IndexError> {
    let reader = HnswDeltaReader::open(delta_path)?;
    let mut last_lsn = 0u64;

    for entry_result in reader {
        let entry = entry_result?;
        if entry.lsn > after_lsn {
            apply_op(snapshot, &entry.op);
            last_lsn = entry.lsn;
        }
    }

    Ok(last_lsn)
}
