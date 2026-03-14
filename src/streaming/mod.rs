//! Streaming updates for vector indices.
//!
//! # The Problem
//!
//! Real-world vector databases face continuous updates:
//! - Documents added as they're created
//! - Documents deleted for GDPR/legal compliance
//! - Embeddings updated when models change
//!
//! Naive approaches rebuild the entire index, which is expensive.
//! This module provides efficient streaming update primitives.
//!
//! # Architecture
//!
//! ```text
//! Stream of Updates
//!     │
//!     ▼
//! ┌──────────────┐
//! │ StreamBuffer │ ◄── Batches small updates
//! └──────┬───────┘
//!        │
//!        ▼ (periodic merge)
//! ┌──────────────┐
//! │  Main Index  │ ◄── HNSW, DiskANN, etc.
//! └──────────────┘
//! ```
//!
//! # Update Types
//!
//! | Type | Complexity | Notes |
//! |------|------------|-------|
//! | Insert | O(log n) amortized | Buffered, merged periodically |
//! | Delete | O(degree) | Mark-and-repair (IP-DiskANN pattern) |
//! | Update | Delete + Insert | Atomic replacement |
//!
//! # Research Context
//!
//! - **FreshDiskANN** (Singh et al. 2021): StreamingMerge for efficient insert/delete
//!   - Key insight: Maintain a small "fresh" index alongside main index
//!   - Insertions go to fresh index; periodic merge into main
//!   - Deletions: tombstone + lazy cleanup during merge
//!
//! - **IP-DiskANN** (2025): In-neighbor tracking for O(1) delete preparation
//!   - Each node tracks its in-neighbors (who points to it)
//!   - Delete: O(degree) edge repairs instead of O(n) search
//!
//! - **SPFresh** (2024): Streaming vector search with LIRE (Lazy In-place REfresh)
//!   - Lazy updates: defer graph repairs until node is visited
//!   - Achieves 2-5x throughput over eager updates
//!
//! - **LSM-VEC** (2025): LSM-tree approach for streaming vectors
//!   - Multiple levels of indices, periodic compaction
//!   - Write-optimized: O(1) amortized insert
//!
//! # Our Approach
//!
//! We implement a hybrid:
//! 1. **Write buffer**: Absorbs burst writes (like LSM memtable)
//! 2. **Tombstone deletes**: Mark-and-sweep during compaction
//! 3. **Merged search**: Query both buffer and main index
//! 4. **Periodic compaction**: Merge buffer into main index
//!
//! # Example
//!
//! ```rust,ignore
//! use vicinity::streaming::{StreamingIndex, UpdateOp};
//! use vicinity::hnsw::HNSWIndex;
//!
//! let mut index = StreamingIndex::new(HNSWIndex::new(128)?);
//!
//! // Stream of updates
//! index.apply(UpdateOp::Insert { id: 0, vector: vec![0.1; 128] })?;
//! index.apply(UpdateOp::Insert { id: 1, vector: vec![0.2; 128] })?;
//! index.apply(UpdateOp::Delete { id: 0 })?;
//!
//! // Search works across buffered and merged data
//! let results = index.search(&query, 10)?;
//!
//! // Periodic compaction
//! index.compact()?;
//! ```

mod buffer;
mod coordinator;
mod ops;

pub use buffer::{StreamBuffer, StreamBufferConfig};
pub use coordinator::{IndexOps, StreamingConfig, StreamingCoordinator};
pub use ops::{UpdateBatch, UpdateOp, UpdateStats};

/// Statistics about streaming index state.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Number of vectors in main index.
    pub main_index_size: usize,
    /// Number of vectors in buffer.
    pub buffer_size: usize,
    /// Number of pending deletes.
    pub pending_deletes: usize,
    /// Total inserts since creation.
    pub total_inserts: u64,
    /// Total deletes since creation.
    pub total_deletes: u64,
    /// Total compactions performed.
    pub total_compactions: u64,
}
