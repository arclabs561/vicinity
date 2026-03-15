//! Unified persistence format for vicinity indexes.
//!
//! # Design Goals
//!
//! 1. **Cross-crate compatibility**: shared format across dense vector indexes
//! 2. **Incremental writes**: WAL-based for crash recovery
//! 3. **Memory-mapped reads**: Zero-copy for large indices
//! 4. **Versioned**: Forward/backward compatible with magic bytes
//!
//! # File Layout
//!
//! ```text
//! vicinity-index-v1
//! ├── manifest.json          # Index metadata, version, config
//! ├── wal/                    # Write-ahead log for crash recovery
//! │   ├── 00001.wal
//! │   ├── 00002.wal
//! │   └── ...
//! ├── segments/               # Immutable data segments
//! │   ├── 00001.seg           # Vectors + graph
//! │   ├── 00002.seg
//! │   └── ...
//! └── checkpoints/            # Periodic snapshots
//!     ├── 00001.ckpt
//!     └── ...
//! ```
//!
//! # Segment Format
//!
//! Each segment is a self-contained unit with:
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Magic bytes (8B): "VCNT" + version      │
//! ├─────────────────────────────────────────┤
//! │ Header (variable):                      │
//! │   - Segment ID                          │
//! │   - Vector count                        │
//! │   - Dimension                           │
//! │   - Index type (HNSW, DiskANN, etc.)    │
//! │   - Compression type                    │
//! │   - Created timestamp                   │
//! ├─────────────────────────────────────────┤
//! │ Vector data section:                    │
//! │   - Raw f32 vectors (mmap-able)         │
//! │   OR quantized vectors                  │
//! ├─────────────────────────────────────────┤
//! │ Graph section (index-specific):         │
//! │   - HNSW: layers + edges                │
//! │   - DiskANN: adjacency lists            │
//! │   - IVF: centroids + posting lists      │
//! ├─────────────────────────────────────────┤
//! │ ID mapping section:                     │
//! │   - Internal ID -> External ID          │
//! ├─────────────────────────────────────────┤
//! │ Footer:                                 │
//! │   - CRC32 checksum                      │
//! │   - Section offsets                     │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # WAL Entry Format
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Length (4B): Entry length               │
//! │ CRC32 (4B): Checksum                    │
//! │ Type (1B): Insert/Delete/Update         │
//! │ Timestamp (8B): Unix nanos              │
//! │ Payload (variable):                     │
//! │   - Vector ID                           │
//! │   - Vector data (for inserts)           │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Compatibility
//!
//! Version handling:
//! - v1: Initial format (HNSW, DiskANN, IVF-PQ)
//! - Future versions add new fields to header
//! - Unknown fields are ignored for forward compatibility
//!
//! # Usage
//!
//! ```rust,ignore
//! use vicinity::persistence::format::IndexPersistence;
//!
//! // Save/load via IndexPersistence trait
//! index.save(path)?;
//! let index = MyIndex::load(path)?;
//! ```

use serde::{Deserialize, Serialize};

/// Magic bytes for segment files.
pub const SEGMENT_MAGIC: &[u8; 4] = b"VCNT";

/// Magic bytes for checkpoint files.
pub const CHECKPOINT_MAGIC: [u8; 4] = *b"VCKP";

/// Current format version.
pub const FORMAT_VERSION: u32 = 1;

/// Magic bytes for WAL files.
pub const WAL_MAGIC: [u8; 4] = *b"VWAL";

/// Index types supported by persistence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// Hierarchical Navigable Small World
    Hnsw,
    /// DiskANN / Vamana
    DiskAnn,
    /// Inverted File with Product Quantization
    IvfPq,
    /// Anisotropic Vector Quantization with k-means (ScaNN)
    ScaNN,
    /// Simple Neighborhood Graph
    Sng,
    /// Flat (brute force)
    Flat,
}

/// Compression types for vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression (raw f32)
    None,
    /// Product quantization
    ProductQuantization,
    /// Scalar quantization (int8)
    ScalarQuantization,
    /// Binary quantization
    BinaryQuantization,
    /// RaBitQ (randomized binary)
    RaBitQ,
}

/// Segment header metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentHeader {
    /// Segment identifier
    pub segment_id: u64,
    /// Number of vectors
    pub vector_count: u64,
    /// Vector dimension
    pub dimension: u32,
    /// Index type
    pub index_type: IndexType,
    /// Compression type
    pub compression: CompressionType,
    /// Creation timestamp (Unix nanos)
    pub created_at: u64,
    /// Optional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// WAL entry types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalEntryType {
    /// Insert a new vector
    Insert = 1,
    /// Delete a vector
    Delete = 2,
    /// Update a vector
    Update = 3,
    /// Checkpoint marker
    Checkpoint = 4,
}

impl TryFrom<u8> for WalEntryType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(WalEntryType::Insert),
            2 => Ok(WalEntryType::Delete),
            3 => Ok(WalEntryType::Update),
            4 => Ok(WalEntryType::Checkpoint),
            _ => Err(()),
        }
    }
}

/// Manifest for the index directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManifest {
    /// Format version
    pub version: u32,
    /// Index type
    pub index_type: IndexType,
    /// Vector dimension
    pub dimension: u32,
    /// Total vector count
    pub total_vectors: u64,
    /// Active segment IDs
    pub segments: Vec<u64>,
    /// Latest WAL sequence number
    pub wal_sequence: u64,
    /// Latest checkpoint ID
    pub checkpoint_id: Option<u64>,
    /// Index-specific configuration
    pub config: serde_json::Value,
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
}

/// Section offsets in a segment file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SegmentFooter {
    /// Magic bytes
    pub magic: [u8; 4],
    /// Format version
    pub format_version: u32,
    /// Header section offset
    pub header_offset: u64,
    /// Vectors section offset
    pub vectors_offset: u64,
    /// Graph section offset
    pub graph_offset: u64,
    /// ID mapping section offset
    pub ids_offset: u64,
    /// CRC32 of entire segment (excluding footer)
    pub checksum: u32,
}

impl SegmentFooter {
    /// Create a new segment footer.
    pub fn new() -> Self {
        Self {
            magic: *SEGMENT_MAGIC,
            format_version: FORMAT_VERSION,
            header_offset: 0,
            vectors_offset: 0,
            graph_offset: 0,
            ids_offset: 0,
            checksum: 0,
        }
    }

    /// Serialized size in bytes (not mem::size_of due to padding).
    /// 4 (magic) + 4 (version) + 8*4 (u64 fields) + 4 (checksum) = 44
    const SERIALIZED_SIZE: usize = 44;

    /// Read a segment footer from a reader.
    pub fn read<R: std::io::Read>(reader: &mut R) -> super::error::PersistenceResult<Self> {
        let mut buf = vec![0u8; Self::SERIALIZED_SIZE];
        reader.read_exact(&mut buf)?;

        // Simple binary deserialization
        let mut cursor = std::io::Cursor::new(&buf);
        use std::io::Read;

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;

        let mut u32_buf = [0u8; 4];
        cursor.read_exact(&mut u32_buf)?;
        let format_version = u32::from_le_bytes(u32_buf);

        let mut u64_buf = [0u8; 8];
        cursor.read_exact(&mut u64_buf)?;
        let header_offset = u64::from_le_bytes(u64_buf);

        cursor.read_exact(&mut u64_buf)?;
        let vectors_offset = u64::from_le_bytes(u64_buf);

        cursor.read_exact(&mut u64_buf)?;
        let graph_offset = u64::from_le_bytes(u64_buf);

        cursor.read_exact(&mut u64_buf)?;
        let ids_offset = u64::from_le_bytes(u64_buf);

        cursor.read_exact(&mut u32_buf)?;
        let checksum = u32::from_le_bytes(u32_buf);

        Ok(Self {
            magic,
            format_version,
            header_offset,
            vectors_offset,
            graph_offset,
            ids_offset,
            checksum,
        })
    }

    /// Write a segment footer to a writer.
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> super::error::PersistenceResult<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.format_version.to_le_bytes())?;
        writer.write_all(&self.header_offset.to_le_bytes())?;
        writer.write_all(&self.vectors_offset.to_le_bytes())?;
        writer.write_all(&self.graph_offset.to_le_bytes())?;
        writer.write_all(&self.ids_offset.to_le_bytes())?;
        writer.write_all(&self.checksum.to_le_bytes())?;
        Ok(())
    }
}

/// Trait for types that can be persisted.
pub trait Persistable: Sized {
    /// Serialize to bytes.
    fn to_bytes(&self) -> crate::Result<Vec<u8>>;

    /// Deserialize from bytes.
    fn from_bytes(bytes: &[u8]) -> crate::Result<Self>;

    /// Estimated size in bytes.
    fn size_hint(&self) -> usize;
}

/// Trait for index persistence operations.
pub trait IndexPersistence: Sized {
    /// Save index to a directory.
    fn save(&self, path: &std::path::Path) -> crate::Result<()>;

    /// Load index from a directory.
    fn load(path: &std::path::Path) -> crate::Result<Self>;

    /// Check if an index exists at the path.
    fn exists(path: &std::path::Path) -> bool {
        path.join("manifest.json").exists()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_entry_type_roundtrip() {
        assert_eq!(WalEntryType::try_from(1), Ok(WalEntryType::Insert));
        assert_eq!(WalEntryType::try_from(2), Ok(WalEntryType::Delete));
        assert_eq!(WalEntryType::try_from(3), Ok(WalEntryType::Update));
        assert_eq!(WalEntryType::try_from(4), Ok(WalEntryType::Checkpoint));
        assert_eq!(WalEntryType::try_from(99), Err(()));
    }

    #[test]
    fn test_segment_header_serde() {
        let header = SegmentHeader {
            segment_id: 1,
            vector_count: 1000,
            dimension: 128,
            index_type: IndexType::Hnsw,
            compression: CompressionType::None,
            created_at: 1234567890,
            metadata: std::collections::HashMap::new(),
        };

        let json = serde_json::to_string(&header).unwrap();
        let parsed: SegmentHeader = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.segment_id, 1);
        assert_eq!(parsed.index_type, IndexType::Hnsw);
    }

    #[test]
    fn test_manifest_serde() {
        let manifest = IndexManifest {
            version: FORMAT_VERSION,
            index_type: IndexType::DiskAnn,
            dimension: 384,
            total_vectors: 10000,
            segments: vec![1, 2, 3],
            wal_sequence: 42,
            checkpoint_id: Some(5),
            config: serde_json::json!({"M": 16, "ef_construction": 200}),
            created_at: 1234567890,
            modified_at: 1234567899,
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let parsed: IndexManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, FORMAT_VERSION);
        assert_eq!(parsed.segments.len(), 3);
    }
}
