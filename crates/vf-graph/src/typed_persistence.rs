// Copyright (c) 2026 Chirotpal Das
// Licensed under the Elastic License 2.0 (ELv2).
// See the LICENSE file at the repository root for full terms.

//! v3 base-snapshot persistence for the typed graph store (Hybrid mode).
//!
//! ADR-007 R4 note on versioning: the similarity `VirtualGraph` keeps writing
//! and reading the v2 format unchanged (its `BASE_FORMAT_VERSION` stays 2), so
//! AutoSimilarity and legacy collections are untouched (R4(b), R4(c)). The v3
//! format introduced here is written ONLY for Hybrid collections, to a separate
//! `graph_typed.base` file. The on-disk envelope reuses the v2 layout (magic
//! `VGRF`, version field at bytes [4..8], snapshot_lsn at [8..16], node_count at
//! [24..32], edge_count at [32..40]) so the shared `validate_graph_base` reader
//! recognises a v3 header, and `deserialize_base` cleanly routes a v3 file here
//! instead of failing as "unknown version" (R4(a), R4(b)). Only the payload
//! differs: typed nodes and edges, JSON-encoded then LZ4-compressed.

use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crc32fast::Hasher as Crc32Hasher;
use serde::{Deserialize, Serialize};

use crate::error::GraphError;
use crate::model::{Edge, Node};
use crate::store::{GraphStoreConfig, TypedGraphStore};

/// The typed graph base format version. Distinct from the v2 `BASE_FORMAT_VERSION`
/// (which stays 2 for the similarity graph). Only Hybrid collections write v3.
pub const TYPED_BASE_FORMAT_VERSION: u32 = 3;

const MAGIC: &[u8; 4] = b"VGRF";
const ENVELOPE_SIZE: usize = 72;
const COMPRESSION_NONE: u8 = 0;
const COMPRESSION_LZ4: u8 = 1;

#[derive(Serialize, Deserialize)]
struct TypedPayload {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

/// Serialize a typed graph store to the v3 base snapshot format.
pub fn serialize_typed_base(
    store: &TypedGraphStore,
    snapshot_lsn: u64,
    writer: &mut impl IoWrite,
) -> Result<(), GraphError> {
    let payload = TypedPayload {
        nodes: store.nodes_snapshot(),
        edges: store.edges_snapshot(),
    };
    let node_count = payload.nodes.len() as u64;
    let edge_count = payload.edges.len() as u64;

    let raw = serde_json::to_vec(&payload)
        .map_err(|e| GraphError::Internal(format!("typed base encode failed: {e}")))?;
    let uncompressed_size = raw.len() as u64;
    let compressed = lz4_flex::compress_prepend_size(&raw);

    let mut env = [0u8; ENVELOPE_SIZE];
    env[0..4].copy_from_slice(MAGIC);
    env[4..8].copy_from_slice(&TYPED_BASE_FORMAT_VERSION.to_le_bytes());
    env[8..16].copy_from_slice(&snapshot_lsn.to_le_bytes());
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    env[16..24].copy_from_slice(&ts.to_le_bytes());
    env[24..32].copy_from_slice(&node_count.to_le_bytes());
    env[32..40].copy_from_slice(&edge_count.to_le_bytes());
    env[53] = COMPRESSION_LZ4;
    env[54..62].copy_from_slice(&uncompressed_size.to_le_bytes());

    let mut crc = Crc32Hasher::new();
    crc.update(&env);
    crc.update(&compressed);
    let checksum = crc.finalize();

    writer.write_all(&env)?;
    writer.write_all(&compressed)?;
    writer.write_all(&checksum.to_le_bytes())?;
    Ok(())
}

/// Deserialize a v3 typed base snapshot with the default store config.
/// Returns `(snapshot_lsn, store)`.
pub fn deserialize_typed_base(
    reader: &mut impl IoRead,
) -> Result<(u64, TypedGraphStore), GraphError> {
    deserialize_typed_base_with_config(reader, GraphStoreConfig::default())
}

/// Deserialize a v3 typed base snapshot, building the store with `config`.
/// Lets a recovered store honor the env-driven entity-index switch.
/// Returns `(snapshot_lsn, store)`.
pub fn deserialize_typed_base_with_config(
    reader: &mut impl IoRead,
    config: GraphStoreConfig,
) -> Result<(u64, TypedGraphStore), GraphError> {
    let mut env = [0u8; ENVELOPE_SIZE];
    reader.read_exact(&mut env)?;

    if &env[0..4] != MAGIC {
        return Err(GraphError::InvalidFormat(
            "invalid magic bytes, expected VGRF".into(),
        ));
    }
    let version = u32::from_le_bytes(env[4..8].try_into().unwrap());
    if version != TYPED_BASE_FORMAT_VERSION {
        return Err(GraphError::InvalidFormat(format!(
            "expected typed base v{TYPED_BASE_FORMAT_VERSION}, got version {version}"
        )));
    }

    let snapshot_lsn = u64::from_le_bytes(env[8..16].try_into().unwrap());
    let compression = env[53];
    let uncompressed_size = u64::from_le_bytes(env[54..62].try_into().unwrap());

    let mut rest = Vec::new();
    reader.read_to_end(&mut rest)?;
    if rest.len() < 4 {
        return Err(GraphError::Corrupted(
            "typed base too short, missing CRC32 footer".into(),
        ));
    }
    let crc_offset = rest.len() - 4;
    let compressed_payload = &rest[..crc_offset];
    let stored_crc = u32::from_le_bytes(rest[crc_offset..].try_into().unwrap());

    let mut crc = Crc32Hasher::new();
    crc.update(&env);
    crc.update(compressed_payload);
    if crc.finalize() != stored_crc {
        return Err(GraphError::Corrupted(format!(
            "typed base CRC32 mismatch: stored={stored_crc:#010x}"
        )));
    }

    let raw = match compression {
        COMPRESSION_NONE => compressed_payload.to_vec(),
        COMPRESSION_LZ4 => lz4_flex::decompress_size_prepended(compressed_payload)
            .map_err(|e| GraphError::Corrupted(format!("typed base LZ4 failed: {e}")))?,
        other => {
            return Err(GraphError::InvalidFormat(format!(
                "unknown compression type: {other}"
            )));
        }
    };
    if raw.len() as u64 != uncompressed_size {
        return Err(GraphError::Corrupted(format!(
            "typed base size mismatch: envelope says {uncompressed_size}, got {}",
            raw.len()
        )));
    }

    let payload: TypedPayload = serde_json::from_slice(&raw)
        .map_err(|e| GraphError::Corrupted(format!("typed base decode failed: {e}")))?;
    let store = TypedGraphStore::from_parts(payload.nodes, payload.edges, config);
    Ok((snapshot_lsn, store))
}

/// Quick header validation for a v3 typed base. Returns `(snapshot_lsn, node_count, edge_count)`.
pub fn validate_typed_base(path: &Path) -> Result<(u64, u64, u64), GraphError> {
    let data = std::fs::read(path)?;
    if data.len() < ENVELOPE_SIZE + 4 {
        return Err(GraphError::Corrupted(
            "file too short for typed base snapshot".into(),
        ));
    }
    if &data[0..4] != MAGIC {
        return Err(GraphError::InvalidFormat(
            "invalid magic bytes, expected VGRF".into(),
        ));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != TYPED_BASE_FORMAT_VERSION {
        return Err(GraphError::InvalidFormat(format!(
            "expected typed base v{TYPED_BASE_FORMAT_VERSION}, got version {version}"
        )));
    }
    let snapshot_lsn = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let node_count = u64::from_le_bytes(data[24..32].try_into().unwrap());
    let edge_count = u64::from_le_bytes(data[32..40].try_into().unwrap());

    let crc_offset = data.len() - 4;
    let stored_crc = u32::from_le_bytes(data[crc_offset..].try_into().unwrap());
    let mut crc = Crc32Hasher::new();
    crc.update(&data[..crc_offset]);
    if crc.finalize() != stored_crc {
        return Err(GraphError::Corrupted(
            "typed base CRC32 mismatch".into(),
        ));
    }
    Ok((snapshot_lsn, node_count, edge_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, Node, NodeId, NodeSource};
    use crate::store::GraphStore;

    #[test]
    fn typed_base_roundtrip() {
        let mut store = TypedGraphStore::with_defaults();
        let a = store.alloc_node_id();
        let b = store.alloc_node_id();
        store
            .put_node(Node::content(a, Some(vec![0.1, 0.2]), NodeSource::Ingested), 1)
            .unwrap();
        store
            .put_node(Node::entity(b, "Party", NodeSource::Manual), 2)
            .unwrap();
        let label = store.intern("PARTY_TO");
        let eid = store.alloc_edge_id();
        store.put_edge(Edge::manual(eid, b, a, label), 3).unwrap();

        let mut buf = Vec::new();
        serialize_typed_base(&store, 3, &mut buf).unwrap();

        let (lsn, restored) = deserialize_typed_base(&mut buf.as_slice()).unwrap();
        assert_eq!(lsn, 3);
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
        assert_eq!(restored.get_node(b).unwrap().id, NodeId(b.0));
        assert_eq!(restored.get_edge(eid).unwrap().edge_type.as_str(), "PARTY_TO");
    }
}
