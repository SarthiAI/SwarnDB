// Copyright (c) 2026 Chirotpal Das
// Licensed under the Business Source License 1.1
// Change Date: 2030-03-06
// Change License: MIT

//! Segment file format constants, header structures, and WAL entry definitions.
//!
//! ```text
//! Segment file layout:
//! ┌──────────────────────────────────┐
//! │ SegmentHeader (64 bytes)         │
//! │   magic: [u8; 4] = b"VFSG"      │
//! │   version: u16                   │
//! │   vector_count: u64              │
//! │   dimension: u32                 │
//! │   data_type: u8 (0=f32,1=f16..) │
//! │   data_offset: u64              │
//! │   meta_offset: u64              │
//! │   checksum: u32 (CRC32)         │
//! ├──────────────────────────────────┤
//! │ Vector Data Region (mmap'd)      │
//! │   [id: u64 | data: [f32; dim]]  │
//! │   ...                            │
//! ├──────────────────────────────────┤
//! │ Metadata Region                  │
//! │   [id: u64 | meta_len: u32 |    │
//! │    meta_bytes: bincode]          │
//! ├──────────────────────────────────┤
//! │ Footer / index hints             │
//! └──────────────────────────────────┘
//! ```

use std::io::{self, Read, Write};

use crate::error::{StorageError, StorageResult};
use vf_core::vector::DataType;

// ── Constants ────────────────────────────────────────────────────────────────

/// Magic bytes identifying a segment file.
pub const SEGMENT_MAGIC: [u8; 4] = *b"VFSG";

/// Current segment format version.
pub const SEGMENT_VERSION: u16 = 1;

/// Fixed size of the segment header in bytes.
pub const SEGMENT_HEADER_SIZE: usize = 64;

/// Magic bytes identifying a WAL file.
pub const WAL_MAGIC: [u8; 4] = *b"VFWL";

// ── DataTypeFlag ─────────────────────────────────────────────────────────────

/// On-disk representation of the vector element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DataTypeFlag {
    F32 = 0,
    F16 = 1,
    U8 = 2,
}

impl DataTypeFlag {
    /// Convert a raw byte to a `DataTypeFlag`.
    pub fn from_u8(v: u8) -> StorageResult<Self> {
        match v {
            0 => Ok(DataTypeFlag::F32),
            1 => Ok(DataTypeFlag::F16),
            2 => Ok(DataTypeFlag::U8),
            _ => Err(StorageError::SegmentInvalidHeader(format!(
                "unknown data type flag: {v}"
            ))),
        }
    }

    /// Convert to the raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl From<DataType> for DataTypeFlag {
    fn from(dt: DataType) -> Self {
        match dt {
            DataType::F32 => DataTypeFlag::F32,
            DataType::F16 => DataTypeFlag::F16,
            DataType::U8 => DataTypeFlag::U8,
        }
    }
}

impl From<DataTypeFlag> for DataType {
    fn from(flag: DataTypeFlag) -> Self {
        match flag {
            DataTypeFlag::F32 => DataType::F32,
            DataTypeFlag::F16 => DataType::F16,
            DataTypeFlag::U8 => DataType::U8,
        }
    }
}

// ── SegmentHeader ────────────────────────────────────────────────────────────

/// Fixed 64-byte header at the start of every segment file.
///
/// Byte layout (little-endian):
/// ```text
/// Offset  Size  Field
///  0       4    magic
///  4       2    version
///  6       8    vector_count
/// 14       4    dimension
/// 18       1    data_type
/// 19       8    data_offset
/// 27       8    meta_offset
/// 35       4    checksum (CRC-32 of bytes 0..35)
/// 39      25    reserved (zero-padded to 64 bytes)
/// ```
#[derive(Clone, Debug)]
pub struct SegmentHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub vector_count: u64,
    pub dimension: u32,
    pub data_type: DataTypeFlag,
    pub data_offset: u64,
    pub meta_offset: u64,
    pub checksum: u32,
}

impl SegmentHeader {
    /// Create a new header with default magic/version and the given parameters.
    pub fn new(dimension: u32, data_type: DataTypeFlag) -> Self {
        let mut hdr = SegmentHeader {
            magic: SEGMENT_MAGIC,
            version: SEGMENT_VERSION,
            vector_count: 0,
            dimension,
            data_type,
            data_offset: SEGMENT_HEADER_SIZE as u64,
            meta_offset: SEGMENT_HEADER_SIZE as u64,
            checksum: 0,
        };
        hdr.checksum = hdr.compute_checksum();
        hdr
    }

    /// Compute CRC-32 over the header bytes *excluding* the checksum field
    /// (bytes 0..35).
    pub fn compute_checksum(&self) -> u32 {
        let mut buf = [0u8; 35];
        self.pack_pre_checksum(&mut buf);
        crc32fast::hash(&buf)
    }

    /// Validate magic, version, and checksum.
    pub fn validate(&self) -> StorageResult<()> {
        if self.magic != SEGMENT_MAGIC {
            return Err(StorageError::BadMagic {
                expected: SEGMENT_MAGIC,
                found: self.magic,
            });
        }
        let computed = self.compute_checksum();
        if self.checksum != computed {
            return Err(StorageError::ChecksumMismatch {
                expected: self.checksum,
                computed,
            });
        }
        Ok(())
    }

    /// Serialize the header to the given writer (exactly 64 bytes).
    pub fn write_to<W: Write>(&self, w: &mut W) -> StorageResult<()> {
        let mut buf = [0u8; SEGMENT_HEADER_SIZE];
        self.pack_pre_checksum(&mut buf[..35]);

        // checksum at offset 35
        buf[35..39].copy_from_slice(&self.checksum.to_le_bytes());

        // bytes 39..64 are reserved (already zero)
        w.write_all(&buf).map_err(StorageError::Io)
    }

    /// Deserialize a header from the given reader (reads exactly 64 bytes).
    pub fn read_from<R: Read>(r: &mut R) -> StorageResult<Self> {
        let mut buf = [0u8; SEGMENT_HEADER_SIZE];
        r.read_exact(&mut buf).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                StorageError::TruncatedData {
                    expected: SEGMENT_HEADER_SIZE,
                    actual: 0, // exact count unknown at this level
                }
            } else {
                StorageError::Io(e)
            }
        })?;

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);

        let version = u16::from_le_bytes([buf[4], buf[5]]);
        let vector_count = u64::from_le_bytes(buf[6..14].try_into().unwrap());
        let dimension = u32::from_le_bytes(buf[14..18].try_into().unwrap());
        let data_type = DataTypeFlag::from_u8(buf[18])?;
        let data_offset = u64::from_le_bytes(buf[19..27].try_into().unwrap());
        let meta_offset = u64::from_le_bytes(buf[27..35].try_into().unwrap());
        let checksum = u32::from_le_bytes(buf[35..39].try_into().unwrap());

        Ok(SegmentHeader {
            magic,
            version,
            vector_count,
            dimension,
            data_type,
            data_offset,
            meta_offset,
            checksum,
        })
    }

    // ── private helpers ──────────────────────────────────────────────────

    /// Pack header fields (before checksum) into the first 35 bytes of `buf`.
    fn pack_pre_checksum(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= 35);
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..14].copy_from_slice(&self.vector_count.to_le_bytes());
        buf[14..18].copy_from_slice(&self.dimension.to_le_bytes());
        buf[18] = self.data_type.as_u8();
        buf[19..27].copy_from_slice(&self.data_offset.to_le_bytes());
        buf[27..35].copy_from_slice(&self.meta_offset.to_le_bytes());
    }
}

// ── WalOp ────────────────────────────────────────────────────────────────────

/// Operation type encoded in each WAL entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WalOp {
    Insert = 0,
    Update = 1,
    Delete = 2,
    CreateCollection = 3,
    DropCollection = 4,
}

impl WalOp {
    /// Convert a raw byte to a `WalOp`.
    pub fn from_u8(v: u8) -> StorageResult<Self> {
        match v {
            0 => Ok(WalOp::Insert),
            1 => Ok(WalOp::Update),
            2 => Ok(WalOp::Delete),
            3 => Ok(WalOp::CreateCollection),
            4 => Ok(WalOp::DropCollection),
            _ => Err(StorageError::WalInvalidEntry(format!(
                "unknown WAL op: {v}"
            ))),
        }
    }

    /// Convert to the raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

// ── WalEntryHeader ───────────────────────────────────────────────────────────

/// Header parsed from each WAL entry.
///
/// Wire format (little-endian):
/// ```text
/// [entry_len: u32][op: u8][collection_id: u64][payload: bytes][crc32: u32]
/// ```
///
/// `entry_len` covers `op + collection_id + payload` (does *not* include
/// `entry_len` itself or the trailing CRC-32).
#[derive(Clone, Debug)]
pub struct WalEntryHeader {
    /// Length of the entry body (op + collection_id + payload).
    pub entry_len: u32,
    /// The operation type.
    pub op: WalOp,
    /// Target collection identifier.
    pub collection_id: u64,
}

impl WalEntryHeader {
    /// Size of the fixed portion: entry_len(4) + op(1) + collection_id(8).
    pub const FIXED_SIZE: usize = 4 + 1 + 8;

    /// Read the fixed portion of a WAL entry from a reader.
    pub fn read_from<R: Read>(r: &mut R) -> StorageResult<Self> {
        let mut buf = [0u8; Self::FIXED_SIZE];
        r.read_exact(&mut buf).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                StorageError::TruncatedData {
                    expected: Self::FIXED_SIZE,
                    actual: 0,
                }
            } else {
                StorageError::Io(e)
            }
        })?;

        let entry_len = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let op = WalOp::from_u8(buf[4])?;
        let collection_id = u64::from_le_bytes(buf[5..13].try_into().unwrap());

        Ok(WalEntryHeader {
            entry_len,
            op,
            collection_id,
        })
    }

    /// Write the fixed portion of a WAL entry to a writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> StorageResult<()> {
        let mut buf = [0u8; Self::FIXED_SIZE];
        buf[0..4].copy_from_slice(&self.entry_len.to_le_bytes());
        buf[4] = self.op.as_u8();
        buf[5..13].copy_from_slice(&self.collection_id.to_le_bytes());
        w.write_all(&buf).map_err(StorageError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn header_round_trip() {
        let hdr = SegmentHeader::new(128, DataTypeFlag::F32);
        assert_eq!(hdr.magic, SEGMENT_MAGIC);
        assert_eq!(hdr.version, SEGMENT_VERSION);

        let mut buf = Vec::new();
        hdr.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), SEGMENT_HEADER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let hdr2 = SegmentHeader::read_from(&mut cursor).unwrap();
        assert_eq!(hdr2.magic, hdr.magic);
        assert_eq!(hdr2.version, hdr.version);
        assert_eq!(hdr2.vector_count, hdr.vector_count);
        assert_eq!(hdr2.dimension, hdr.dimension);
        assert_eq!(hdr2.data_type, hdr.data_type);
        assert_eq!(hdr2.data_offset, hdr.data_offset);
        assert_eq!(hdr2.meta_offset, hdr.meta_offset);
        assert_eq!(hdr2.checksum, hdr.checksum);
    }

    #[test]
    fn header_validate_ok() {
        let hdr = SegmentHeader::new(64, DataTypeFlag::F16);
        assert!(hdr.validate().is_ok());
    }

    #[test]
    fn header_validate_bad_magic() {
        let mut hdr = SegmentHeader::new(64, DataTypeFlag::F32);
        hdr.magic = *b"BAAD";
        assert!(hdr.validate().is_err());
    }

    #[test]
    fn header_validate_bad_checksum() {
        let mut hdr = SegmentHeader::new(64, DataTypeFlag::F32);
        hdr.checksum = 0xDEADBEEF;
        assert!(hdr.validate().is_err());
    }

    #[test]
    fn data_type_flag_round_trip() {
        for flag in [DataTypeFlag::F32, DataTypeFlag::F16, DataTypeFlag::U8] {
            let byte = flag.as_u8();
            let back = DataTypeFlag::from_u8(byte).unwrap();
            assert_eq!(back, flag);
        }
    }

    #[test]
    fn data_type_flag_core_conversion() {
        assert_eq!(DataType::from(DataTypeFlag::F32), DataType::F32);
        assert_eq!(DataTypeFlag::from(DataType::F16), DataTypeFlag::F16);
        assert_eq!(DataTypeFlag::from(DataType::U8), DataTypeFlag::U8);
    }

    #[test]
    fn wal_op_round_trip() {
        for op in [
            WalOp::Insert,
            WalOp::Update,
            WalOp::Delete,
            WalOp::CreateCollection,
            WalOp::DropCollection,
        ] {
            let byte = op.as_u8();
            let back = WalOp::from_u8(byte).unwrap();
            assert_eq!(back, op);
        }
    }

    #[test]
    fn wal_entry_header_round_trip() {
        let hdr = WalEntryHeader {
            entry_len: 42,
            op: WalOp::Insert,
            collection_id: 7,
        };

        let mut buf = Vec::new();
        hdr.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), WalEntryHeader::FIXED_SIZE);

        let mut cursor = Cursor::new(&buf);
        let hdr2 = WalEntryHeader::read_from(&mut cursor).unwrap();
        assert_eq!(hdr2.entry_len, 42);
        assert_eq!(hdr2.op, WalOp::Insert);
        assert_eq!(hdr2.collection_id, 7);
    }

    #[test]
    fn invalid_data_type_flag() {
        assert!(DataTypeFlag::from_u8(99).is_err());
    }

    #[test]
    fn invalid_wal_op() {
        assert!(WalOp::from_u8(99).is_err());
    }
}
