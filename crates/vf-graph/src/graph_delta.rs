// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crc32fast::Hasher as Crc32Hasher;

use crate::error::GraphError;
use crate::types::{Edge, VirtualGraph};

// ── Constants ───────────────────────────────────────────────────────
const DELTA_MAGIC: &[u8; 4] = b"GDLT";
const DELTA_VERSION: u8 = 1;
const OP_ADD_NODE: u8 = 0;
const OP_REMOVE_NODE: u8 = 1;
const OP_ADD_EDGE: u8 = 2;
const OP_REMOVE_EDGE: u8 = 3;
const OP_SET_THRESHOLD: u8 = 4;

// ── Types ───────────────────────────────────────────────────────────

/// Operations recorded in the graph delta log.
#[derive(Clone, Debug)]
pub enum GraphDeltaOp {
    AddNode {
        id: u64,
        threshold_override: Option<f32>,
    },
    RemoveNode {
        id: u64,
    },
    AddEdge {
        src: u64,
        tgt: u64,
        similarity: f32,
        created_at: u64,
        refined: bool,
    },
    RemoveEdge {
        src: u64,
        tgt: u64,
    },
    SetThreshold {
        node_id: u64,
        threshold: f32,
    },
}

#[derive(Clone, Debug)]
pub struct GraphDeltaEntry {
    pub lsn: u64,
    pub op: GraphDeltaOp,
}

// ── Serialization helpers ───────────────────────────────────────────

/// Encode a delta op into its payload bytes and return (op_type, payload).
fn encode_op(op: &GraphDeltaOp) -> (u8, Vec<u8>) {
    match op {
        GraphDeltaOp::AddNode {
            id,
            threshold_override,
        } => {
            let mut payload = Vec::with_capacity(13);
            payload.extend_from_slice(&id.to_le_bytes());
            match threshold_override {
                Some(t) => {
                    payload.push(1u8);
                    payload.extend_from_slice(&t.to_le_bytes());
                }
                None => {
                    payload.push(0u8);
                }
            }
            (OP_ADD_NODE, payload)
        }
        GraphDeltaOp::RemoveNode { id } => {
            let mut payload = Vec::with_capacity(8);
            payload.extend_from_slice(&id.to_le_bytes());
            (OP_REMOVE_NODE, payload)
        }
        GraphDeltaOp::AddEdge {
            src,
            tgt,
            similarity,
            created_at,
            refined,
        } => {
            let mut payload = Vec::with_capacity(29);
            payload.extend_from_slice(&src.to_le_bytes());
            payload.extend_from_slice(&tgt.to_le_bytes());
            payload.extend_from_slice(&similarity.to_le_bytes());
            payload.extend_from_slice(&created_at.to_le_bytes());
            payload.push(*refined as u8);
            (OP_ADD_EDGE, payload)
        }
        GraphDeltaOp::RemoveEdge { src, tgt } => {
            let mut payload = Vec::with_capacity(16);
            payload.extend_from_slice(&src.to_le_bytes());
            payload.extend_from_slice(&tgt.to_le_bytes());
            (OP_REMOVE_EDGE, payload)
        }
        GraphDeltaOp::SetThreshold { node_id, threshold } => {
            let mut payload = Vec::with_capacity(12);
            payload.extend_from_slice(&node_id.to_le_bytes());
            payload.extend_from_slice(&threshold.to_le_bytes());
            (OP_SET_THRESHOLD, payload)
        }
    }
}

/// Compute CRC32 over [lsn][op_type][payload_len][payload].
fn compute_entry_crc(lsn: u64, op_type: u8, payload: &[u8]) -> u32 {
    let mut crc = Crc32Hasher::new();
    crc.update(&lsn.to_le_bytes());
    crc.update(&[op_type]);
    crc.update(&(payload.len() as u32).to_le_bytes());
    crc.update(payload);
    crc.finalize()
}

// ── Writer ──────────────────────────────────────────────────────────

pub struct GraphDeltaWriter {
    file: BufWriter<File>,
    path: PathBuf,
    entry_count: u64,
    last_lsn: u64,
}

impl GraphDeltaWriter {
    /// Create a new delta log file, writing the header.
    pub fn create(path: &Path) -> Result<Self, GraphError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header: magic + version
        writer.write_all(DELTA_MAGIC)?;
        writer.write_all(&[DELTA_VERSION])?;
        writer.flush()?;

        Ok(Self {
            file: writer,
            path: path.to_path_buf(),
            entry_count: 0,
            last_lsn: 0,
        })
    }

    /// Open an existing delta log, scanning to the end to recover state.
    pub fn open(path: &Path) -> Result<Self, GraphError> {
        // First, scan to count entries and find last LSN
        let mut reader = GraphDeltaReader::open(path)?;
        let mut entry_count: u64 = 0;
        let mut last_lsn: u64 = 0;

        while let Some(entry) = reader.next_entry()? {
            entry_count += 1;
            last_lsn = entry.lsn;
        }

        drop(reader);

        // Open the file for appending
        let file = OpenOptions::new().append(true).open(path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            file: writer,
            path: path.to_path_buf(),
            entry_count,
            last_lsn,
        })
    }

    /// Append a delta entry. Returns the byte offset after the write.
    pub fn append(&mut self, entry: &GraphDeltaEntry) -> Result<u64, GraphError> {
        let (op_type, payload) = encode_op(&entry.op);
        let payload_len = payload.len() as u32;
        let crc = compute_entry_crc(entry.lsn, op_type, &payload);

        // Write: [lsn: u64][op_type: u8][payload_len: u32][payload][crc32: u32]
        self.file.write_all(&entry.lsn.to_le_bytes())?;
        self.file.write_all(&[op_type])?;
        self.file.write_all(&payload_len.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&crc.to_le_bytes())?;

        self.entry_count += 1;
        self.last_lsn = entry.lsn;

        // Total bytes for this entry
        let entry_bytes = 8 + 1 + 4 + payload.len() as u64 + 4;
        Ok(entry_bytes)
    }

    /// Flush buffered writes to disk.
    pub fn sync(&mut self) -> Result<(), GraphError> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
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

    /// Return the current file size in bytes.
    pub fn file_size(&self) -> Result<u64, GraphError> {
        let metadata = std::fs::metadata(&self.path)?;
        Ok(metadata.len())
    }
}

// ── Reader ──────────────────────────────────────────────────────────

pub struct GraphDeltaReader {
    reader: BufReader<File>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl GraphDeltaReader {
    /// Open an existing delta log for reading. Validates the header.
    pub fn open(path: &Path) -> Result<Self, GraphError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and validate header
        let mut magic = [0u8; 4];
        if reader.read_exact(&mut magic).is_err() {
            return Err(GraphError::InvalidFormat(
                "delta log too short for header".into(),
            ));
        }
        if &magic != DELTA_MAGIC {
            return Err(GraphError::InvalidFormat(
                "invalid delta magic, expected GDLT".into(),
            ));
        }

        let mut ver = [0u8; 1];
        if reader.read_exact(&mut ver).is_err() {
            return Err(GraphError::InvalidFormat(
                "delta log truncated at version byte".into(),
            ));
        }
        if ver[0] != DELTA_VERSION {
            return Err(GraphError::InvalidFormat(format!(
                "unsupported delta version: {}, expected {}",
                ver[0], DELTA_VERSION
            )));
        }

        Ok(Self {
            reader,
            path: path.to_path_buf(),
        })
    }

    /// Read the next entry, or None if at end of file.
    /// Truncated entries return None (crash recovery friendly).
    pub fn next_entry(&mut self) -> Result<Option<GraphDeltaEntry>, GraphError> {
        // Read LSN (8 bytes) -- if we can't read it, we're at EOF or truncated
        let mut buf8 = [0u8; 8];
        match self.reader.read_exact(&mut buf8) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(GraphError::Io(e)),
        }
        let lsn = u64::from_le_bytes(buf8);

        // Read op_type (1 byte)
        let mut op_byte = [0u8; 1];
        if self.reader.read_exact(&mut op_byte).is_err() {
            return Ok(None); // truncated
        }
        let op_type = op_byte[0];

        // Read payload_len (4 bytes)
        let mut buf4 = [0u8; 4];
        if self.reader.read_exact(&mut buf4).is_err() {
            return Ok(None); // truncated
        }
        let payload_len = u32::from_le_bytes(buf4) as usize;

        // Sanity check payload length
        if payload_len > 1024 * 1024 {
            return Ok(None); // unreasonable, treat as corruption
        }

        // Read payload
        let mut payload = vec![0u8; payload_len];
        if self.reader.read_exact(&mut payload).is_err() {
            return Ok(None); // truncated
        }

        // Read CRC32 (4 bytes)
        if self.reader.read_exact(&mut buf4).is_err() {
            return Ok(None); // truncated
        }
        let stored_crc = u32::from_le_bytes(buf4);

        // Verify CRC
        let computed_crc = compute_entry_crc(lsn, op_type, &payload);
        if stored_crc != computed_crc {
            return Err(GraphError::Corrupted(format!(
                "delta entry CRC mismatch at LSN {lsn}: stored={stored_crc:#010x}, computed={computed_crc:#010x}"
            )));
        }

        // Decode op from payload
        let op = decode_op(op_type, &payload)?;

        Ok(Some(GraphDeltaEntry { lsn, op }))
    }
}

impl Iterator for GraphDeltaReader {
    type Item = Result<GraphDeltaEntry, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_entry() {
            Ok(Some(entry)) => Some(Ok(entry)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// ── Payload decoding ────────────────────────────────────────────────

fn decode_op(op_type: u8, payload: &[u8]) -> Result<GraphDeltaOp, GraphError> {
    let mut cursor = 0usize;

    match op_type {
        OP_ADD_NODE => {
            let id = read_u64(payload, &mut cursor)?;
            let has_threshold = read_u8(payload, &mut cursor)?;
            let threshold_override = if has_threshold == 1 {
                Some(read_f32(payload, &mut cursor)?)
            } else {
                None
            };
            Ok(GraphDeltaOp::AddNode {
                id,
                threshold_override,
            })
        }
        OP_REMOVE_NODE => {
            let id = read_u64(payload, &mut cursor)?;
            Ok(GraphDeltaOp::RemoveNode { id })
        }
        OP_ADD_EDGE => {
            let src = read_u64(payload, &mut cursor)?;
            let tgt = read_u64(payload, &mut cursor)?;
            let similarity = read_f32(payload, &mut cursor)?;
            let created_at = read_u64(payload, &mut cursor)?;
            let refined = read_u8(payload, &mut cursor)? != 0;
            Ok(GraphDeltaOp::AddEdge {
                src,
                tgt,
                similarity,
                created_at,
                refined,
            })
        }
        OP_REMOVE_EDGE => {
            let src = read_u64(payload, &mut cursor)?;
            let tgt = read_u64(payload, &mut cursor)?;
            Ok(GraphDeltaOp::RemoveEdge { src, tgt })
        }
        OP_SET_THRESHOLD => {
            let node_id = read_u64(payload, &mut cursor)?;
            let threshold = read_f32(payload, &mut cursor)?;
            Ok(GraphDeltaOp::SetThreshold {
                node_id,
                threshold,
            })
        }
        _ => Err(GraphError::InvalidFormat(format!(
            "unknown delta op type: {op_type}"
        ))),
    }
}

// ── Cursor helpers (same pattern as persistence.rs) ─────────────────

fn ensure_remaining(data: &[u8], cursor: usize, need: usize) -> Result<(), GraphError> {
    if cursor + need > data.len() {
        Err(GraphError::Corrupted(format!(
            "delta payload truncated at offset {cursor}, need {need} bytes but only {} remain",
            data.len() - cursor
        )))
    } else {
        Ok(())
    }
}

fn read_u8(data: &[u8], cursor: &mut usize) -> Result<u8, GraphError> {
    ensure_remaining(data, *cursor, 1)?;
    let v = data[*cursor];
    *cursor += 1;
    Ok(v)
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64, GraphError> {
    ensure_remaining(data, *cursor, 8)?;
    let v = u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_f32(data: &[u8], cursor: &mut usize) -> Result<f32, GraphError> {
    ensure_remaining(data, *cursor, 4)?;
    let v = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(v)
}

// ── Replay functions ────────────────────────────────────────────────

/// Apply a single delta op to a VirtualGraph.
fn apply_op(graph: &mut VirtualGraph, op: &GraphDeltaOp) {
    match op {
        GraphDeltaOp::AddNode {
            id,
            threshold_override,
        } => {
            graph.add_node(*id);
            if let Some(t) = threshold_override {
                graph.set_vector_threshold(*id, *t);
            }
        }
        GraphDeltaOp::RemoveNode { id } => {
            graph.remove_node(*id);
        }
        GraphDeltaOp::AddEdge {
            src,
            tgt,
            similarity,
            created_at,
            refined,
        } => {
            // Add the edge directly to the source node's edge list
            graph.add_node(*src);
            let max_edges = graph.config().max_edges_per_node;
            if let Some(node) = graph.get_node_mut(*src) {
                let edge = Edge {
                    target: *tgt,
                    similarity: *similarity,
                    created_at: *created_at,
                    refined: *refined,
                };
                node.upsert_edge(edge, max_edges);
            }
        }
        GraphDeltaOp::RemoveEdge { src, tgt } => {
            if let Some(node) = graph.get_node_mut(*src) {
                node.remove_edge(*tgt);
            }
        }
        GraphDeltaOp::SetThreshold {
            node_id,
            threshold,
        } => {
            graph.set_vector_threshold(*node_id, *threshold);
        }
    }
}

/// Replay all delta entries onto a VirtualGraph.
/// Returns the number of entries replayed.
pub fn replay_delta(graph: &mut VirtualGraph, delta_path: &Path) -> Result<u64, GraphError> {
    let mut reader = GraphDeltaReader::open(delta_path)?;
    let mut count: u64 = 0;

    while let Some(entry) = reader.next_entry()? {
        apply_op(graph, &entry.op);
        count += 1;
    }

    Ok(count)
}

/// Replay only entries with LSN > after_lsn.
/// Returns the number of entries replayed.
pub fn replay_delta_after_lsn(
    graph: &mut VirtualGraph,
    delta_path: &Path,
    after_lsn: u64,
) -> Result<u64, GraphError> {
    let mut reader = GraphDeltaReader::open(delta_path)?;
    let mut count: u64 = 0;

    while let Some(entry) = reader.next_entry()? {
        if entry.lsn > after_lsn {
            apply_op(graph, &entry.op);
            count += 1;
        }
    }

    Ok(count)
}
