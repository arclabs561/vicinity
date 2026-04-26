//! Persistence format types.
//!
//! Contains serializable entry types used by the WAL subsystem, and binary
//! format constants for the HNSW segment persistence path.

// ---------------------------------------------------------------------------
// HNSW segment binary format constants
// ---------------------------------------------------------------------------

/// Magic bytes prepended to `metadata.bin` in HNSW segments written by
/// vicinity 0.7+.
///
/// Files written by 0.6.x have no magic; the loader recognises legacy files
/// by the absence of this marker and applies the v0 decode path.
pub const HNSW_SEGMENT_MAGIC: [u8; 8] = *b"VCNHNSW\x01";

/// Format version stored as a little-endian u32 immediately after the magic.
///
/// Increment this when the on-disk layout changes in a backward-incompatible
/// way. The loader rejects files whose version exceeds this constant.
pub const FORMAT_VERSION: u32 = 1;

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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
