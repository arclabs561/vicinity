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

/// Per-segment metadata tracked in the manifest.
///
/// Each segment carries its own WAL watermark so that multi-segment compaction
/// can replay from `min(segment watermarks)` instead of a single global sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Segment identifier.
    pub segment_id: u64,
    /// WAL sequence number: the last WAL entry that this segment has fully absorbed.
    /// During replay, start from `min(all segment wal_sequences)`.
    pub wal_sequence: u64,
}

/// Graph-level WAL entries for streaming HNSW updates.
///
/// These are finer-grained than the segment-lifecycle entries in `durability::walog::WalEntry`,
/// enabling incremental graph recovery without replaying full segment rebuilds.
///
/// Serialized via postcard through `durability`'s generic `WalWriter<GraphWalEntry>`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GraphWalEntry {
    /// A node was inserted into the HNSW graph.
    InsertNode {
        /// External document ID.
        doc_id: u32,
        /// Layer the node was assigned to (0 = bottom).
        level: u8,
        /// The node's vector data.
        vector: Vec<f32>,
        /// Neighbor lists per level, from level 0 up to `level`.
        /// `neighbors_per_level[i]` is the neighbor list at level `i`.
        neighbors_per_level: Vec<Vec<u32>>,
    },
    /// A node was deleted from the graph.
    DeleteNode {
        /// External document ID.
        doc_id: u32,
    },
    /// Neighbor list for an existing node was updated (e.g., after a neighbor deletion or
    /// re-ranking pass).
    UpdateNeighbors {
        /// Internal node ID.
        node_id: u32,
        /// Graph level whose neighbor list changed.
        level: u8,
        /// New neighbor list (replaces the old one entirely).
        neighbors: Vec<u32>,
    },
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
    /// Active segment IDs (kept for backward compatibility with v1 manifests).
    pub segments: Vec<u64>,
    /// Per-segment metadata with individual WAL watermarks.
    ///
    /// When present, WAL replay starts from `min(segment_info[*].wal_sequence)`.
    /// When absent (legacy manifests), falls back to the global `wal_sequence`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub segment_info: Vec<SegmentInfo>,
    /// Latest WAL sequence number (global watermark, legacy).
    ///
    /// Superseded by per-segment watermarks in `segment_info` when available.
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

impl IndexManifest {
    /// Compute the WAL replay start sequence.
    ///
    /// If per-segment watermarks are available, returns `min(segment watermarks)`.
    /// Otherwise, falls back to the global `wal_sequence`.
    pub fn replay_start_sequence(&self) -> u64 {
        if self.segment_info.is_empty() {
            self.wal_sequence
        } else {
            self.segment_info
                .iter()
                .map(|s| s.wal_sequence)
                .min()
                .unwrap_or(self.wal_sequence)
        }
    }
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
            segment_info: vec![],
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
        // segment_info is empty, so it should be absent from JSON (skip_serializing_if)
        assert!(!json.contains("segment_info"));
    }

    #[test]
    fn test_manifest_backward_compat_no_segment_info() {
        // Simulate a legacy manifest JSON without segment_info field.
        let legacy_json = r#"{
            "version": 1,
            "index_type": "DiskAnn",
            "dimension": 384,
            "total_vectors": 10000,
            "segments": [1, 2, 3],
            "wal_sequence": 42,
            "checkpoint_id": 5,
            "config": {},
            "created_at": 1234567890,
            "modified_at": 1234567899
        }"#;

        let parsed: IndexManifest = serde_json::from_str(legacy_json).unwrap();
        assert!(parsed.segment_info.is_empty());
        // Falls back to global wal_sequence
        assert_eq!(parsed.replay_start_sequence(), 42);
    }

    #[test]
    fn test_manifest_per_segment_watermarks() {
        let manifest = IndexManifest {
            version: FORMAT_VERSION,
            index_type: IndexType::Hnsw,
            dimension: 128,
            total_vectors: 5000,
            segments: vec![1, 2, 3],
            segment_info: vec![
                SegmentInfo {
                    segment_id: 1,
                    wal_sequence: 100,
                },
                SegmentInfo {
                    segment_id: 2,
                    wal_sequence: 50,
                },
                SegmentInfo {
                    segment_id: 3,
                    wal_sequence: 200,
                },
            ],
            wal_sequence: 200,
            checkpoint_id: None,
            config: serde_json::json!({}),
            created_at: 0,
            modified_at: 0,
        };

        // Replay starts from min(segment watermarks) = 50
        assert_eq!(manifest.replay_start_sequence(), 50);

        // Round-trip through JSON
        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: IndexManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.segment_info.len(), 3);
        assert_eq!(parsed.replay_start_sequence(), 50);
    }

    #[test]
    fn test_segment_info_serde() {
        let info = SegmentInfo {
            segment_id: 42,
            wal_sequence: 100,
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: SegmentInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.segment_id, 42);
        assert_eq!(parsed.wal_sequence, 100);
    }

    #[test]
    fn test_graph_wal_entry_postcard_roundtrip() {
        let entries = vec![
            GraphWalEntry::InsertNode {
                doc_id: 42,
                level: 2,
                vector: vec![1.0, 2.0, 3.0],
                neighbors_per_level: vec![vec![1, 2, 3], vec![4, 5], vec![6]],
            },
            GraphWalEntry::DeleteNode { doc_id: 99 },
            GraphWalEntry::UpdateNeighbors {
                node_id: 10,
                level: 0,
                neighbors: vec![20, 30, 40],
            },
        ];

        for entry in &entries {
            let bytes = postcard::to_allocvec(entry).unwrap();
            let decoded: GraphWalEntry = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(&decoded, entry);
        }
    }
}
