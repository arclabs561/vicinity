//! Tombstone-based deletions for streaming HNSW updates.
//!
//! Inspired by FreshDiskANN (Singh et al., 2021) and IP-DiskANN (Xu et al., 2025),
//! this module provides soft deletion via tombstones rather than immediate graph repair.
//!
//! # Why Tombstones?
//!
//! Graph-based ANN indices have expensive deletion:
//! - Removing a node breaks edges, potentially disconnecting the graph
//! - Repairing edges requires finding replacement neighbors (O(degree * search))
//! - Naive deletion can degrade recall significantly
//!
//! FreshDiskANN insight: defer expensive repairs via tombstones:
//! 1. Mark deleted nodes as tombstones (O(1))
//! 2. Filter tombstones from search results (small overhead)
//! 3. Batch repair tombstones periodically (amortized cost)
//!
//! # Trade-offs
//!
//! | Approach | Deletion | Search | Memory |
//! |----------|----------|--------|--------|
//! | Immediate repair | O(degree * search) | Optimal | Optimal |
//! | Tombstones | O(1) | Slight overhead | Tombstone bloat |
//! | Rebuild | N/A | Optimal | 2x during rebuild |
//!
//! Tombstones are best when:
//! - Deletions are infrequent relative to searches
//! - Latency-sensitive deletion (can't block on repair)
//! - Deletions can be batched for periodic compaction
//!
//! # References
//!
//! - Singh et al. (2021). "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index
//!   for Streaming Similarity Search." arXiv:2105.09613
//! - Xu et al. (2025). "IP-DiskANN: In-Place Graph Index Updates for Streaming ANN."

use std::collections::HashSet;

/// Manages tombstones for soft deletion in HNSW.
///
/// Tombstones are stored separately from the main graph structure,
/// allowing O(1) deletion with deferred cleanup.
#[derive(Debug)]
pub struct TombstoneSet {
    /// Set of deleted internal node IDs
    deleted: HashSet<usize>,
    /// Count of deleted nodes (for metrics)
    delete_count: usize,
    /// Threshold for triggering compaction (fraction of total nodes)
    compaction_threshold: f32,
}

impl TombstoneSet {
    /// Create a new tombstone set.
    ///
    /// # Arguments
    ///
    /// * `compaction_threshold` - Fraction of nodes that can be tombstoned
    ///   before suggesting compaction (e.g., 0.1 = 10%)
    pub fn new(compaction_threshold: f32) -> Self {
        TombstoneSet {
            deleted: HashSet::new(),
            delete_count: 0,
            compaction_threshold: compaction_threshold.clamp(0.01, 0.5),
        }
    }

    /// Mark a node as deleted (soft delete).
    ///
    /// Returns true if the node was newly deleted, false if already deleted.
    pub fn delete(&mut self, internal_id: usize) -> bool {
        let inserted = self.deleted.insert(internal_id);
        if inserted {
            self.delete_count += 1;
        }
        inserted
    }

    /// Check if a node is deleted.
    #[inline]
    pub fn is_deleted(&self, internal_id: usize) -> bool {
        self.deleted.contains(&internal_id)
    }

    /// Get the number of deleted nodes.
    pub fn len(&self) -> usize {
        self.deleted.len()
    }

    /// Check if there are no tombstones.
    pub fn is_empty(&self) -> bool {
        self.deleted.is_empty()
    }

    /// Check if compaction is recommended.
    ///
    /// Returns true if the fraction of tombstones exceeds the threshold.
    pub fn should_compact(&self, total_nodes: usize) -> bool {
        if total_nodes == 0 {
            return false;
        }
        let deleted = self.deleted.len();
        (deleted as f32 / total_nodes as f32) > self.compaction_threshold
    }

    /// Get all tombstoned node IDs.
    ///
    /// Used during compaction to identify nodes for removal.
    pub fn tombstones(&self) -> impl Iterator<Item = usize> + '_ {
        self.deleted.iter().copied()
    }

    /// Clear all tombstones (call after compaction).
    pub fn clear(&mut self) {
        self.deleted.clear();
        self.delete_count = 0;
    }

    /// Filter tombstones from search results.
    ///
    /// # Arguments
    ///
    /// * `results` - Search results as (internal_id, distance) pairs
    ///
    /// # Returns
    ///
    /// Filtered results with tombstones removed
    pub fn filter_results<'a>(
        &'a self,
        results: impl Iterator<Item = (usize, f32)> + 'a,
    ) -> impl Iterator<Item = (usize, f32)> + 'a {
        results.filter(move |(id, _)| !self.is_deleted(*id))
    }

    /// Get tombstone statistics.
    pub fn stats(&self, total_nodes: usize) -> TombstoneStats {
        let count = self.deleted.len();
        let ratio = if total_nodes > 0 {
            count as f32 / total_nodes as f32
        } else {
            0.0
        };
        TombstoneStats {
            count,
            total_nodes,
            ratio,
            needs_compaction: self.should_compact(total_nodes),
        }
    }
}

impl Default for TombstoneSet {
    fn default() -> Self {
        Self::new(0.1) // 10% default threshold
    }
}

/// Statistics about tombstone state.
#[derive(Debug, Clone, Copy)]
pub struct TombstoneStats {
    /// Number of tombstoned nodes
    pub count: usize,
    /// Total nodes in the index
    pub total_nodes: usize,
    /// Ratio of tombstones to total
    pub ratio: f32,
    /// Whether compaction is recommended
    pub needs_compaction: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tombstone_basic() {
        let mut ts = TombstoneSet::new(0.1);

        assert!(ts.is_empty());
        assert!(ts.delete(5));
        assert!(!ts.delete(5)); // Already deleted
        assert!(ts.is_deleted(5));
        assert!(!ts.is_deleted(10));
        assert_eq!(ts.len(), 1);
    }

    #[test]
    fn test_filter_results() {
        let mut ts = TombstoneSet::new(0.1);
        ts.delete(2);
        ts.delete(4);

        let results = vec![(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4), (5, 0.5)];
        let filtered: Vec<_> = ts.filter_results(results.into_iter()).collect();

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].0, 1);
        assert_eq!(filtered[1].0, 3);
        assert_eq!(filtered[2].0, 5);
    }

    #[test]
    fn test_compaction_threshold() {
        let mut ts = TombstoneSet::new(0.1); // 10% threshold

        // With 100 nodes, need >10 deletions to trigger compaction
        assert!(!ts.should_compact(100));

        for i in 0..11 {
            ts.delete(i);
        }

        assert!(ts.should_compact(100)); // 11% > 10%
        assert!(!ts.should_compact(1000)); // 1.1% < 10%
    }

    #[test]
    fn test_stats() {
        let mut ts = TombstoneSet::new(0.1);
        ts.delete(1);
        ts.delete(2);
        ts.delete(3);

        let stats = ts.stats(30);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_nodes, 30);
        assert!((stats.ratio - 0.1).abs() < 1e-6);
        assert!(!stats.needs_compaction); // Exactly 10%, threshold is >10%

        let stats2 = ts.stats(20);
        assert!(stats2.needs_compaction); // 15% > 10%
    }

    #[test]
    fn test_clear() {
        let mut ts = TombstoneSet::new(0.1);
        ts.delete(1);
        ts.delete(2);
        assert_eq!(ts.len(), 2);

        ts.clear();
        assert!(ts.is_empty());
        assert!(!ts.is_deleted(1));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Deletion is idempotent
        #[test]
        fn prop_delete_idempotent(id in 0..1000_usize) {
            let mut ts = TombstoneSet::new(0.1);
            let first = ts.delete(id);
            let second = ts.delete(id);

            // First delete returns true, subsequent return false
            prop_assert!(first);
            prop_assert!(!second);
            prop_assert!(ts.is_deleted(id));
            prop_assert_eq!(ts.len(), 1);
        }

        /// Filter preserves non-deleted items in order
        #[test]
        fn prop_filter_preserves_order(
            deletions in proptest::collection::vec(0..100_usize, 0..20),
            results in proptest::collection::vec((0..100_usize, 0.0..1.0_f32), 0..50),
        ) {
            let mut ts = TombstoneSet::new(0.1);
            for id in &deletions {
                ts.delete(*id);
            }

            let filtered: Vec<_> = ts.filter_results(results.iter().cloned()).collect();

            // All filtered results should be non-deleted
            for (id, _) in &filtered {
                prop_assert!(!ts.is_deleted(*id));
            }

            // Filtered count + deleted-in-results count = original count
            let deleted_count = results.iter().filter(|(id, _)| ts.is_deleted(*id)).count();
            prop_assert_eq!(filtered.len() + deleted_count, results.len());
        }

        /// Compaction threshold is consistent
        #[test]
        fn prop_compaction_threshold_consistent(
            threshold in 0.01..0.5_f32,
            deletions in 0..50_usize,
            total in 50..500_usize,
        ) {
            let mut ts = TombstoneSet::new(threshold);
            for i in 0..deletions {
                ts.delete(i);
            }

            let ratio = deletions as f32 / total as f32;
            let should_compact = ts.should_compact(total);

            // Should compact iff ratio > threshold
            prop_assert_eq!(should_compact, ratio > threshold);
        }

        /// Stats are consistent with state
        #[test]
        fn prop_stats_consistent(
            deletions in proptest::collection::vec(0..1000_usize, 0..50),
            total in 50..500_usize,
        ) {
            let mut ts = TombstoneSet::new(0.1);
            for id in &deletions {
                ts.delete(*id);
            }

            let unique_deletions = ts.len();
            let stats = ts.stats(total);

            prop_assert_eq!(stats.count, unique_deletions);
            prop_assert_eq!(stats.total_nodes, total);

            let expected_ratio = unique_deletions as f32 / total as f32;
            prop_assert!((stats.ratio - expected_ratio).abs() < 1e-5);
        }
    }
}
