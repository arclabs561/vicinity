//! Write buffer for streaming updates.
//!
//! Buffers small updates before merging into the main index.
//! This amortizes the cost of index updates and enables batched processing.

use crate::error::{Result, RetrieveError};
use std::collections::{HashMap, HashSet};

/// Configuration for the stream buffer.
#[derive(Debug, Clone)]
pub struct StreamBufferConfig {
    /// Maximum vectors to buffer before forcing a merge.
    pub max_buffer_size: usize,
    /// Maximum pending deletes before forcing a merge.
    pub max_pending_deletes: usize,
    /// Buffer memory limit in bytes (approximate).
    pub max_memory_bytes: usize,
    /// Distance metric for brute-force buffer search.
    pub distance_metric: crate::distance::DistanceMetric,
}

impl Default for StreamBufferConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10_000,
            max_pending_deletes: 1_000,
            max_memory_bytes: 100 * 1024 * 1024, // 100 MB
            // Default to Cosine to match HNSW (the primary index type).
            // Must match the distance metric of the main index to produce
            // consistent merged rankings in StreamingCoordinator::search.
            distance_metric: crate::distance::DistanceMetric::Cosine,
        }
    }
}

/// A buffer for streaming vector updates.
///
/// # Design
///
/// The buffer maintains:
/// 1. **Insert buffer**: New vectors not yet in main index
/// 2. **Delete set**: IDs marked for deletion (applied during compaction)
///
/// During search, both buffer and main index are queried, with
/// delete set filtering applied to results.
#[derive(Debug)]
pub struct StreamBuffer {
    /// Buffered inserts: id -> vector
    inserts: HashMap<u32, Vec<f32>>,
    /// Pending deletes
    deletes: HashSet<u32>,
    /// Configuration
    config: StreamBufferConfig,
    /// Vector dimension (set on first insert)
    dimension: Option<usize>,
    /// Approximate memory usage
    memory_bytes: usize,
}

impl StreamBuffer {
    /// Create a buffer with default configuration.
    pub fn new() -> Self {
        Self::with_config(StreamBufferConfig::default())
    }

    /// Create a buffer with the given configuration.
    pub fn with_config(config: StreamBufferConfig) -> Self {
        Self {
            inserts: HashMap::new(),
            deletes: HashSet::new(),
            config,
            dimension: None,
            memory_bytes: 0,
        }
    }

    /// Insert a vector into the buffer.
    pub fn insert(&mut self, id: u32, vector: Vec<f32>) -> Result<()> {
        // Check/set dimension
        match self.dimension {
            None => self.dimension = Some(vector.len()),
            Some(dim) if dim != vector.len() => {
                return Err(RetrieveError::DimensionMismatch {
                    query_dim: dim,
                    doc_dim: vector.len(),
                });
            }
            _ => {}
        }

        // If this ID was pending delete, remove from delete set
        self.deletes.remove(&id);

        // Update memory tracking
        let vec_bytes = vector.len() * std::mem::size_of::<f32>();
        if let Some(old) = self.inserts.insert(id, vector) {
            // Replacing existing, adjust memory
            let old_bytes = old.len() * std::mem::size_of::<f32>();
            self.memory_bytes = self.memory_bytes.saturating_sub(old_bytes);
        }
        self.memory_bytes += vec_bytes;

        Ok(())
    }

    /// Mark an ID for deletion.
    pub fn delete(&mut self, id: u32) {
        // If in insert buffer, just remove it
        if let Some(vec) = self.inserts.remove(&id) {
            let vec_bytes = vec.len() * std::mem::size_of::<f32>();
            self.memory_bytes = self.memory_bytes.saturating_sub(vec_bytes);
        } else {
            // Not in buffer, mark for deletion from main index
            self.deletes.insert(id);
        }
    }

    /// Check if buffer needs compaction.
    pub fn needs_compaction(&self) -> bool {
        self.inserts.len() >= self.config.max_buffer_size
            || self.deletes.len() >= self.config.max_pending_deletes
            || self.memory_bytes >= self.config.max_memory_bytes
    }

    /// Drain the buffer for compaction.
    pub fn drain(&mut self) -> (HashMap<u32, Vec<f32>>, HashSet<u32>) {
        let inserts = std::mem::take(&mut self.inserts);
        let deletes = std::mem::take(&mut self.deletes);
        self.memory_bytes = 0;
        (inserts, deletes)
    }

    /// Number of buffered inserts.
    pub fn insert_count(&self) -> usize {
        self.inserts.len()
    }

    /// Number of pending deletes.
    pub fn delete_count(&self) -> usize {
        self.deletes.len()
    }

    /// Check if an ID is pending delete.
    pub fn is_deleted(&self, id: u32) -> bool {
        self.deletes.contains(&id)
    }

    /// Get a vector from the buffer (if present).
    pub fn get(&self, id: u32) -> Option<&Vec<f32>> {
        self.inserts.get(&id)
    }

    /// Iterate over buffered vectors.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &Vec<f32>)> {
        self.inserts.iter().map(|(&id, vec)| (id, vec))
    }

    /// Brute-force search in buffer.
    ///
    /// Returns (id, distance) pairs sorted by distance ascending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let metric = self.config.distance_metric;
        let mut results: Vec<(u32, f32)> = self
            .inserts
            .iter()
            .filter(|(id, _)| !self.deletes.contains(id))
            .map(|(&id, vec)| {
                let dist = metric.distance(query, vec);
                (id, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

impl Default for StreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_delete() {
        let mut buffer = StreamBuffer::new();

        buffer.insert(0, vec![1.0, 2.0]).unwrap();
        buffer.insert(1, vec![3.0, 4.0]).unwrap();

        assert_eq!(buffer.insert_count(), 2);

        buffer.delete(0);
        assert_eq!(buffer.insert_count(), 1);
        assert!(!buffer.is_deleted(0)); // Was in buffer, so just removed
    }

    #[test]
    fn test_delete_from_main() {
        let mut buffer = StreamBuffer::new();

        // Delete something not in buffer (i.e., in main index)
        buffer.delete(42);

        assert!(buffer.is_deleted(42));
        assert_eq!(buffer.delete_count(), 1);
    }

    #[test]
    fn test_search() {
        let mut config = StreamBufferConfig::default();
        config.distance_metric = crate::distance::DistanceMetric::L2;
        let mut buffer = StreamBuffer::with_config(config);

        buffer.insert(0, vec![0.0, 0.0]).unwrap();
        buffer.insert(1, vec![1.0, 0.0]).unwrap();
        buffer.insert(2, vec![0.0, 1.0]).unwrap();

        let query = vec![0.1, 0.1];
        let results = buffer.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest to query
    }
}
