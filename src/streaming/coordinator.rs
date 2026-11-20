//! Coordinator for streaming updates to vector indices.
//!
//! Manages the interaction between the write buffer and the main index.

use super::buffer::{StreamBuffer, StreamBufferConfig};
use super::ops::UpdateStats;
use super::StreamingStats;
use crate::error::Result;
use std::time::Instant;

/// Configuration for streaming coordinator.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer configuration.
    pub buffer: StreamBufferConfig,
    /// Auto-compact when buffer is full.
    pub auto_compact: bool,
    /// Merge buffer results with main index results during search.
    pub merge_search_results: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer: StreamBufferConfig::default(),
            auto_compact: true,
            merge_search_results: true,
        }
    }
}

/// Coordinator that wraps an index with streaming update support.
///
/// # Type Parameters
///
/// - `I`: The underlying index type
///
/// # Example
///
/// ```rust,ignore
/// use vicinity::streaming::StreamingCoordinator;
/// use vicinity::hnsw::HNSWIndex;
///
/// let index = HNSWIndex::new(128)?;
/// let mut streaming = StreamingCoordinator::new(index);
///
/// streaming.insert(0, vec![0.1; 128])?;
/// streaming.insert(1, vec![0.2; 128])?;
/// streaming.delete(0)?;
///
/// let results = streaming.search(&query, 10)?;
/// ```
pub struct StreamingCoordinator<I> {
    /// The underlying index.
    index: I,
    /// Write buffer.
    buffer: StreamBuffer,
    /// Configuration.
    config: StreamingConfig,
    /// Statistics.
    total_inserts: u64,
    total_deletes: u64,
    total_compactions: u64,
    /// Ids replaced through `update` since the last compaction. The main index
    /// may still hold their old vector: search hides it, and compaction deletes
    /// it before inserting the new one.
    replaced: std::collections::HashSet<u32>,
}

impl<I> StreamingCoordinator<I> {
    /// Create a new streaming coordinator wrapping an index.
    pub fn new(index: I) -> Self {
        Self::with_config(index, StreamingConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(index: I, config: StreamingConfig) -> Self {
        Self {
            index,
            buffer: StreamBuffer::with_config(config.buffer.clone()),
            config,
            total_inserts: 0,
            total_deletes: 0,
            total_compactions: 0,
            replaced: std::collections::HashSet::new(),
        }
    }

    /// Insert a vector into buffer.
    ///
    /// Note: Call `compact()` periodically to merge buffer into main index.
    /// Use `insert_and_compact()` for auto-compaction (requires `IndexOps`).
    pub fn buffer_insert(&mut self, id: u32, vector: Vec<f32>) -> Result<()> {
        self.buffer.insert(id, vector)?;
        self.total_inserts += 1;
        Ok(())
    }

    /// Mark a vector for deletion.
    ///
    /// If the vector is in the buffer, it's removed immediately.
    /// Otherwise, it's marked for deletion during next compaction.
    pub fn buffer_delete(&mut self, id: u32) {
        self.buffer.delete(id);
        self.total_deletes += 1;
    }

    /// Check if the buffer needs compaction.
    pub fn needs_compaction(&self) -> bool {
        self.buffer.needs_compaction()
    }

    /// Get streaming statistics.
    pub fn stats(&self) -> StreamingStats {
        StreamingStats {
            main_index_size: 0, // Would need index trait to get this
            buffer_size: self.buffer.insert_count(),
            pending_deletes: self.buffer.delete_count(),
            total_inserts: self.total_inserts,
            total_deletes: self.total_deletes,
            total_compactions: self.total_compactions,
        }
    }

    /// Access the underlying index.
    pub fn inner(&self) -> &I {
        &self.index
    }

    /// Access the underlying index mutably.
    pub fn inner_mut(&mut self) -> &mut I {
        &mut self.index
    }

    /// Access the buffer.
    pub fn buffer(&self) -> &StreamBuffer {
        &self.buffer
    }
}

// Methods that require the index to support insert/delete
impl<I: IndexOps> StreamingCoordinator<I> {
    /// Insert a vector with auto-compaction.
    pub fn insert(&mut self, id: u32, vector: Vec<f32>) -> Result<()> {
        self.buffer.insert(id, vector)?;
        self.total_inserts += 1;

        if self.config.auto_compact && self.buffer.needs_compaction() {
            self.compact()?;
        }

        Ok(())
    }

    /// Delete a vector with auto-compaction.
    pub fn delete(&mut self, id: u32) -> Result<()> {
        // Deleting a buffered vector only drops it from the buffer. If that
        // vector came from `update`, the main index may still hold the old
        // copy, which needs a pending delete of its own.
        let buffered = self.buffer.get(id).is_some();
        self.buffer.delete(id);
        if buffered && self.replaced.contains(&id) {
            self.buffer.delete(id);
        }
        self.total_deletes += 1;

        if self.config.auto_compact && self.buffer.needs_compaction() {
            self.compact()?;
        }

        Ok(())
    }

    /// Update a vector (atomic delete + insert) with auto-compaction.
    pub fn update(&mut self, id: u32, vector: Vec<f32>) -> Result<()> {
        // `buffer.insert` clears the pending delete that `buffer.delete` just
        // queued, so the buffer alone no longer records that the main index
        // may hold an older copy. Remember it here.
        self.buffer.delete(id);
        self.buffer.insert(id, vector)?;
        self.replaced.insert(id);
        self.total_inserts += 1;
        self.total_deletes += 1;

        if self.config.auto_compact && self.buffer.needs_compaction() {
            self.compact()?;
        }

        Ok(())
    }

    /// Compact buffered updates into the main index.
    pub fn compact(&mut self) -> Result<UpdateStats> {
        let start = Instant::now();
        let (inserts, deletes) = self.buffer.drain();
        let replaced = std::mem::take(&mut self.replaced);

        let mut stats = UpdateStats::default();

        // Remove the old copy of every updated id that is about to be
        // re-inserted. The id may never have reached the main index (inserted
        // and updated within one buffer generation), so a failed delete here
        // is not an error. Without this, an index that rejects duplicate ids
        // dropped the update, and one that accepts them kept both vectors.
        for id in replaced {
            if inserts.contains_key(&id) && !deletes.contains(&id) {
                let _ = self.index.delete(id);
            }
        }

        // Apply deletes first
        for id in deletes {
            if self.index.delete(id).is_ok() {
                stats.deletes_applied += 1;
            } else {
                stats.errors += 1;
            }
        }

        // Then inserts
        for (id, vector) in inserts {
            if self.index.insert(id, vector).is_ok() {
                stats.inserts_applied += 1;
            } else {
                stats.errors += 1;
            }
        }

        stats.duration_us = start.elapsed().as_micros() as u64;
        self.total_compactions += 1;

        Ok(stats)
    }

    /// Search the index, merging buffer and main index results.
    ///
    /// The buffer's `distance_metric` must match the main index's distance
    /// function for merged rankings to be meaningful. By default, both use
    /// cosine distance (matching HNSW). If you use a different index type
    /// or metric, set `StreamBufferConfig::distance_metric` accordingly.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        // Detect metric mismatch between buffer and main index.
        // This is a runtime check (not debug_assert) because mismatched metrics
        // silently produce wrong merged rankings in release builds.
        if self.config.buffer.distance_metric != self.index.distance_metric() {
            return Err(crate::RetrieveError::InvalidParameter(format!(
                "buffer distance metric ({:?}) does not match index metric ({:?}); \
                 merged search results would have inconsistent rankings",
                self.config.buffer.distance_metric,
                self.index.distance_metric()
            )));
        }

        if !self.config.merge_search_results || self.buffer.insert_count() == 0 {
            // Just search main index, filter deletes
            let results = self.index.search(query, k)?;
            return Ok(results
                .into_iter()
                .filter(|(id, _)| !self.buffer.is_deleted(*id))
                .collect());
        }

        // Search both buffer and main index
        let buffer_results = self.buffer.search(query, k);
        let mut main_results = self.index.search(query, k * 2)?; // Overfetch to account for deletes

        // Filter deletes from main results, and any id the buffer also holds:
        // the buffered vector is the newer one, and the main index's stale
        // copy must not compete with it.
        main_results
            .retain(|(id, _)| !self.buffer.is_deleted(*id) && self.buffer.get(*id).is_none());

        // Merge results
        let mut combined = buffer_results;
        combined.extend(main_results);
        combined.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        // Deduplicate before truncating, so a duplicate cannot take one of
        // the k places.
        let mut seen = std::collections::HashSet::new();
        combined.retain(|(id, _)| seen.insert(*id));
        combined.truncate(k);

        Ok(combined)
    }
}

/// Trait for index operations needed by streaming coordinator.
pub trait IndexOps {
    /// Insert a vector into the index.
    fn insert(&mut self, id: u32, vector: Vec<f32>) -> Result<()>;

    /// Delete a vector from the index.
    fn delete(&mut self, id: u32) -> Result<()>;

    /// Search the index.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>>;

    /// Distance metric used by the index.
    ///
    /// The streaming coordinator checks this against the buffer's metric
    /// to prevent silently merging results from incompatible distance functions.
    fn distance_metric(&self) -> crate::distance::DistanceMetric {
        // Default to Cosine since HNSW (the primary index) uses cosine.
        crate::distance::DistanceMetric::Cosine
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // Mock index for testing
    struct MockIndex {
        vectors: std::collections::HashMap<u32, Vec<f32>>,
    }

    impl MockIndex {
        fn new() -> Self {
            Self {
                vectors: std::collections::HashMap::new(),
            }
        }
    }

    impl IndexOps for MockIndex {
        fn insert(&mut self, id: u32, vector: Vec<f32>) -> Result<()> {
            self.vectors.insert(id, vector);
            Ok(())
        }

        fn delete(&mut self, id: u32) -> Result<()> {
            self.vectors.remove(&id);
            Ok(())
        }

        fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
            let mut results: Vec<_> = self
                .vectors
                .iter()
                .map(|(&id, vec)| {
                    let dist: f32 = query
                        .iter()
                        .zip(vec.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        .sqrt();
                    (id, dist)
                })
                .collect();
            results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            results.truncate(k);
            Ok(results)
        }

        fn distance_metric(&self) -> crate::distance::DistanceMetric {
            crate::distance::DistanceMetric::L2
        }
    }

    /// Create a StreamingConfig with L2 buffer metric to match MockIndex.
    fn mock_config() -> StreamingConfig {
        let mut config = StreamingConfig::default();
        config.buffer.distance_metric = crate::distance::DistanceMetric::L2;
        config
    }

    /// Like `MockIndex`, but refuses an id it already holds, as an index
    /// without overwrite semantics does.
    struct StrictIndex(MockIndex);

    impl IndexOps for StrictIndex {
        fn insert(&mut self, id: u32, vector: Vec<f32>) -> Result<()> {
            if self.0.vectors.contains_key(&id) {
                return Err(crate::RetrieveError::InvalidParameter(format!(
                    "duplicate id {id}"
                )));
            }
            self.0.insert(id, vector)
        }

        fn delete(&mut self, id: u32) -> Result<()> {
            match self.0.vectors.remove(&id) {
                Some(_) => Ok(()),
                None => Err(crate::RetrieveError::InvalidParameter(format!(
                    "unknown id {id}"
                ))),
            }
        }

        fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
            self.0.search(query, k)
        }

        fn distance_metric(&self) -> crate::distance::DistanceMetric {
            self.0.distance_metric()
        }
    }

    #[test]
    fn update_hides_the_old_vector_and_survives_compaction() {
        let mut streaming =
            StreamingCoordinator::with_config(StrictIndex(MockIndex::new()), mock_config());
        streaming.insert(1, vec![1.0, 0.0]).unwrap();
        streaming.insert(2, vec![0.0, 1.0]).unwrap();
        streaming.insert(3, vec![-1.0, 0.0]).unwrap();
        streaming.compact().unwrap();

        streaming.update(1, vec![0.0, -1.0]).unwrap();

        // Before compaction: the main index still holds [1, 0] for id 1.
        let near_old = streaming.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(near_old.len(), 2, "{near_old:?}");
        assert!(
            near_old.iter().all(|(id, dist)| *id != 1 || *dist > 1.0),
            "stale vector for id 1 was returned: {near_old:?}"
        );
        let near_new = streaming.search(&[0.0, -1.0], 1).unwrap();
        assert_eq!(near_new[0].0, 1);
        assert!(near_new[0].1 < 1e-6);

        // Compaction replaces the old copy and reports no error.
        let stats = streaming.compact().unwrap();
        assert_eq!(stats.errors, 0);
        let near_new = streaming.search(&[0.0, -1.0], 1).unwrap();
        assert_eq!(near_new[0].0, 1);
        assert!(near_new[0].1 < 1e-6);
        let near_old = streaming.search(&[1.0, 0.0], 3).unwrap();
        assert_eq!(near_old.len(), 3);
        assert!(near_old.iter().all(|(_, dist)| *dist > 1.0), "{near_old:?}");
    }

    #[test]
    fn update_of_an_id_never_compacted_is_not_an_error() {
        let mut streaming =
            StreamingCoordinator::with_config(StrictIndex(MockIndex::new()), mock_config());
        streaming.insert(1, vec![1.0, 0.0]).unwrap();
        streaming.update(1, vec![0.0, 1.0]).unwrap();
        let stats = streaming.compact().unwrap();
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.inserts_applied, 1);
        assert_eq!(streaming.search(&[0.0, 1.0], 1).unwrap()[0].0, 1);
    }

    #[test]
    fn update_then_delete_removes_the_id_everywhere() {
        let mut streaming =
            StreamingCoordinator::with_config(StrictIndex(MockIndex::new()), mock_config());
        streaming.insert(1, vec![1.0, 0.0]).unwrap();
        streaming.insert(2, vec![0.0, 1.0]).unwrap();
        streaming.compact().unwrap();
        streaming.update(1, vec![0.5, 0.5]).unwrap();
        streaming.delete(1).unwrap();
        streaming.compact().unwrap();
        let all = streaming.search(&[1.0, 0.0], 5).unwrap();
        assert_eq!(all.iter().map(|r| r.0).collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn test_streaming_insert_search() {
        let index = MockIndex::new();
        let mut streaming = StreamingCoordinator::with_config(index, mock_config());

        streaming.insert(0, vec![0.0, 0.0]).unwrap();
        streaming.insert(1, vec![1.0, 0.0]).unwrap();

        // Before compaction, search should find buffered vectors
        let results = streaming.search(&[0.1, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_streaming_delete() {
        let index = MockIndex::new();
        let mut streaming = StreamingCoordinator::with_config(index, mock_config());

        streaming.insert(0, vec![0.0, 0.0]).unwrap();
        streaming.insert(1, vec![1.0, 0.0]).unwrap();
        streaming.delete(0).unwrap();

        // Search should not return deleted vector
        let results = streaming.search(&[0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_compaction() {
        let index = MockIndex::new();
        let mut streaming = StreamingCoordinator::with_config(index, mock_config());

        streaming.insert(0, vec![0.0, 0.0]).unwrap();
        streaming.insert(1, vec![1.0, 0.0]).unwrap();

        let stats = streaming.compact().unwrap();
        assert_eq!(stats.inserts_applied, 2);

        // After compaction, buffer should be empty
        assert_eq!(streaming.buffer().insert_count(), 0);
    }
}
