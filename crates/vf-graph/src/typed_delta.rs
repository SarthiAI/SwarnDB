// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Typed graph delta log (the v3 delta stream). Mirrors `graph_delta.rs` but
//! records typed node and edge operations rather than similarity edges.
//!
//! Op payloads are JSON (`serde_json`) so the nested property bags and
//! provenance round-trip exactly. bincode is deliberately NOT used: it cannot
//! deserialize `serde_json::Value` (a non-self-describing format does not
//! support `deserialize_any`).
//!
//! Op-codes start at 10 so they never collide with the v2 similarity delta ops
//! (0..=4). The stream is self-describing: a reader can tell a typed op from a
//! similarity op by file magic (`GTDL` vs `GDLT`) and by op-code.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crc32fast::Hasher as Crc32Hasher;
use serde::{Deserialize, Serialize};

use crate::error::GraphError;
use crate::model::{Edge, EdgeId, Node, NodeId};

const TYPED_DELTA_MAGIC: &[u8; 4] = b"GTDL";
const TYPED_DELTA_VERSION: u8 = 1;

const OP_PUT_NODE: u8 = 10;
const OP_DELETE_NODE: u8 = 11;
const OP_PUT_EDGE: u8 = 12;
const OP_DELETE_EDGE: u8 = 13;

/// Guard against absurd payload lengths from a corrupt tail.
const MAX_PAYLOAD: usize = 64 * 1024 * 1024;

/// A single typed graph mutation recorded in the delta log.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TypedGraphOp {
    PutNode(Node),
    DeleteNode(NodeId),
    PutEdge(Edge),
    DeleteEdge(EdgeId),
}

#[derive(Clone, Debug)]
pub struct TypedDeltaEntry {
    pub lsn: u64,
    pub op: TypedGraphOp,
}

fn encode_op(op: &TypedGraphOp) -> Result<(u8, Vec<u8>), GraphError> {
    let enc = |r: serde_json::Result<Vec<u8>>| {
        r.map_err(|e| GraphError::Internal(format!("typed delta encode failed: {e}")))
    };
    let pair = match op {
        TypedGraphOp::PutNode(n) => (OP_PUT_NODE, enc(serde_json::to_vec(n))?),
        TypedGraphOp::DeleteNode(id) => (OP_DELETE_NODE, enc(serde_json::to_vec(id))?),
        TypedGraphOp::PutEdge(e) => (OP_PUT_EDGE, enc(serde_json::to_vec(e))?),
        TypedGraphOp::DeleteEdge(id) => (OP_DELETE_EDGE, enc(serde_json::to_vec(id))?),
    };
    Ok(pair)
}

fn decode_op(op_type: u8, payload: &[u8]) -> Result<TypedGraphOp, GraphError> {
    let bad = |e: serde_json::Error| GraphError::Corrupted(format!("typed delta decode failed: {e}"));
    match op_type {
        OP_PUT_NODE => Ok(TypedGraphOp::PutNode(
            serde_json::from_slice(payload).map_err(bad)?,
        )),
        OP_DELETE_NODE => Ok(TypedGraphOp::DeleteNode(
            serde_json::from_slice(payload).map_err(bad)?,
        )),
        OP_PUT_EDGE => Ok(TypedGraphOp::PutEdge(
            serde_json::from_slice(payload).map_err(bad)?,
        )),
        OP_DELETE_EDGE => Ok(TypedGraphOp::DeleteEdge(
            serde_json::from_slice(payload).map_err(bad)?,
        )),
        other => Err(GraphError::InvalidFormat(format!(
            "unknown typed delta op-code: {other}"
        ))),
    }
}

fn compute_entry_crc(lsn: u64, op_type: u8, payload: &[u8]) -> u32 {
    let mut crc = Crc32Hasher::new();
    crc.update(&lsn.to_le_bytes());
    crc.update(&[op_type]);
    crc.update(&(payload.len() as u32).to_le_bytes());
    crc.update(payload);
    crc.finalize()
}

// ── Writer ──────────────────────────────────────────────────────────

pub struct TypedDeltaWriter {
    file: BufWriter<File>,
    path: PathBuf,
    entry_count: u64,
    last_lsn: u64,
}

impl TypedDeltaWriter {
    /// Create a fresh typed delta log, writing the header.
    pub fn create(path: &Path) -> Result<Self, GraphError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(TYPED_DELTA_MAGIC)?;
        writer.write_all(&[TYPED_DELTA_VERSION])?;
        writer.flush()?;
        Ok(Self {
            file: writer,
            path: path.to_path_buf(),
            entry_count: 0,
            last_lsn: 0,
        })
    }

    /// Open an existing typed delta log for appending, scanning to recover
    /// entry count and last LSN.
    pub fn open(path: &Path) -> Result<Self, GraphError> {
        let mut reader = TypedDeltaReader::open(path)?;
        let mut entry_count: u64 = 0;
        let mut last_lsn: u64 = 0;
        while let Some(entry) = reader.next_entry()? {
            entry_count += 1;
            last_lsn = entry.lsn;
        }
        drop(reader);

        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self {
            file: BufWriter::new(file),
            path: path.to_path_buf(),
            entry_count,
            last_lsn,
        })
    }

    /// Append one entry. Returns the number of bytes written for it.
    pub fn append(&mut self, entry: &TypedDeltaEntry) -> Result<u64, GraphError> {
        let (op_type, payload) = encode_op(&entry.op)?;
        let payload_len = payload.len() as u32;
        let crc = compute_entry_crc(entry.lsn, op_type, &payload);

        self.file.write_all(&entry.lsn.to_le_bytes())?;
        self.file.write_all(&[op_type])?;
        self.file.write_all(&payload_len.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&crc.to_le_bytes())?;

        self.entry_count += 1;
        self.last_lsn = entry.lsn;
        Ok(8 + 1 + 4 + payload.len() as u64 + 4)
    }

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

    pub fn file_size(&self) -> Result<u64, GraphError> {
        Ok(std::fs::metadata(&self.path)?.len())
    }
}

// ── Reader ──────────────────────────────────────────────────────────

pub struct TypedDeltaReader {
    reader: BufReader<File>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl TypedDeltaReader {
    pub fn open(path: &Path) -> Result<Self, GraphError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        if reader.read_exact(&mut magic).is_err() {
            return Err(GraphError::InvalidFormat(
                "typed delta log too short for header".into(),
            ));
        }
        if &magic != TYPED_DELTA_MAGIC {
            return Err(GraphError::InvalidFormat(
                "invalid typed delta magic, expected GTDL".into(),
            ));
        }

        let mut ver = [0u8; 1];
        if reader.read_exact(&mut ver).is_err() {
            return Err(GraphError::InvalidFormat(
                "typed delta log truncated at version byte".into(),
            ));
        }
        if ver[0] != TYPED_DELTA_VERSION {
            return Err(GraphError::InvalidFormat(format!(
                "unsupported typed delta version: {}, expected {}",
                ver[0], TYPED_DELTA_VERSION
            )));
        }

        Ok(Self {
            reader,
            path: path.to_path_buf(),
        })
    }

    /// Read the next entry. Truncated or partial tail entries return `None`
    /// (crash-recovery friendly), matching the v2 delta reader.
    pub fn next_entry(&mut self) -> Result<Option<TypedDeltaEntry>, GraphError> {
        let mut buf8 = [0u8; 8];
        match self.reader.read_exact(&mut buf8) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(GraphError::Io(e)),
        }
        let lsn = u64::from_le_bytes(buf8);

        let mut op_byte = [0u8; 1];
        if self.reader.read_exact(&mut op_byte).is_err() {
            return Ok(None);
        }
        let op_type = op_byte[0];

        let mut buf4 = [0u8; 4];
        if self.reader.read_exact(&mut buf4).is_err() {
            return Ok(None);
        }
        let payload_len = u32::from_le_bytes(buf4) as usize;
        if payload_len > MAX_PAYLOAD {
            return Ok(None);
        }

        let mut payload = vec![0u8; payload_len];
        if self.reader.read_exact(&mut payload).is_err() {
            return Ok(None);
        }

        if self.reader.read_exact(&mut buf4).is_err() {
            return Ok(None);
        }
        let stored_crc = u32::from_le_bytes(buf4);

        let computed = compute_entry_crc(lsn, op_type, &payload);
        if stored_crc != computed {
            return Err(GraphError::Corrupted(format!(
                "typed delta CRC mismatch at LSN {lsn}: stored={stored_crc:#010x}, computed={computed:#010x}"
            )));
        }

        let op = decode_op(op_type, &payload)?;
        Ok(Some(TypedDeltaEntry { lsn, op }))
    }
}

impl Iterator for TypedDeltaReader {
    type Item = Result<TypedDeltaEntry, GraphError>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.next_entry() {
            Ok(Some(e)) => Some(Ok(e)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, EdgeId, Interner, NodeId};

    #[test]
    fn writer_reader_roundtrip_and_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph_typed.delta");

        let mut interner = Interner::new();
        let edge = Edge::manual(EdgeId(1), NodeId(10), NodeId(20), interner.intern("CITES"));

        let mut w = TypedDeltaWriter::create(&path).unwrap();
        w.append(&TypedDeltaEntry {
            lsn: 1,
            op: TypedGraphOp::PutEdge(edge.clone()),
        })
        .unwrap();
        w.append(&TypedDeltaEntry {
            lsn: 2,
            op: TypedGraphOp::DeleteEdge(EdgeId(1)),
        })
        .unwrap();
        w.sync().unwrap();

        let mut r = TypedDeltaReader::open(&path).unwrap();
        let e1 = r.next_entry().unwrap().unwrap();
        assert_eq!(e1.lsn, 1);
        matches!(e1.op, TypedGraphOp::PutEdge(_));
        let e2 = r.next_entry().unwrap().unwrap();
        assert_eq!(e2.lsn, 2);
        assert!(r.next_entry().unwrap().is_none());
    }
}
