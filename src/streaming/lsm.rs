//! LSM-tiered streaming vector index.
//!
//! Applies LSM-tree storage principles to vector graph indexing:
//!
//! ```text
//! Writes ──► L0 (in-memory buffer, brute-force search)
//!              │
//!              ▼ compaction (build HNSW, merge into L1)
//!            L1 (small HNSW graph)
//!              │
//!              ▼ compaction (merge L1 + L2 → new L2)
//!            L2 (larger HNSW graph)
//!              │
//!              ...
//! ```
//!
//! # Write Amplification
//!
//! With size ratio T (default 10), each vector is rewritten O(log_T(N/B)) times
//! where N is total vectors and B is the L0 buffer size. For T=10, N=100M, B=10K:
//! `log_10(10000) = 4` rewrites per vector over its lifetime.
//!
//! # Search
//!
//! Query searches each level independently and merges results. Total search cost:
//! `O(L * search_per_level)` where L = number of levels (typically 2-4).
//! Each level is an independent HNSW graph searched with its own ef parameter.
//!
//! # Compaction Trigger
//!
//! Level i compacts into level i+1 when `level_i.size >= T * level_{i+1}.size`
//! (or when L0 exceeds buffer capacity). This is size-tiered compaction.
//!
//! # Tombstones
//!
//! Deletes are recorded as tombstones propagated during compaction. A deleted ID
//! is filtered from all search results across all levels. Tombstones are garbage-
//! collected when they reach the deepest level.
//!
//! # References
//!
//! - Inspired by LSM-VEC (2025, arXiv:2505.17152)
//! - O'Neil et al. (1996). "The Log-Structured Merge-Tree (LSM-Tree)."

use crate::distance::DistanceMetric;
use crate::error::{Result, RetrieveError};
use std::collections::HashSet;

/// Configuration for LSM-tiered streaming.
#[derive(Debug, Clone)]
pub struct LsmConfig {
    /// Vector dimension.
    pub dimension: usize,
    /// L0 buffer capacity (number of vectors before first compaction).
    pub buffer_capacity: usize,
    /// Size ratio between adjacent levels (T in LSM literature). Default: 10.
    pub size_ratio: usize,
    /// Maximum number of levels (prevents unbounded growth). Default: 5.
    pub max_levels: usize,
    /// HNSW M parameter for compacted levels.
    pub hnsw_m: usize,
    /// HNSW ef_construction for compacted levels.
    pub hnsw_ef_construction: usize,
    /// ef_search for query on each level.
    pub ef_search: usize,
    /// Distance metric.
    pub distance_metric: DistanceMetric,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            buffer_capacity: 10_000,
            size_ratio: 10,
            max_levels: 5,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            ef_search: 100,
            distance_metric: DistanceMetric::Cosine,
        }
    }
}

/// A single level in the LSM tree.
///
/// L0 is a flat buffer (brute-force search). L1+ are HNSW graphs.
#[derive(Debug)]
struct Level {
    /// Vectors stored at this level: flat `[v0_d0, v0_d1, ..., v1_d0, ...]`.
    vectors: Vec<f32>,
    /// Doc IDs for each vector.
    doc_ids: Vec<u32>,
    /// Number of vectors.
    count: usize,
    /// HNSW index (None for L0, Some for L1+).
    #[cfg(feature = "hnsw")]
    hnsw: Option<crate::hnsw::HNSWIndex>,
}

impl Level {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
            doc_ids: Vec::new(),
            count: 0,
            #[cfg(feature = "hnsw")]
            hnsw: None,
        }
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Brute-force search (used for L0 or when HNSW is not available).
    fn brute_force_search(
        &self,
        query: &[f32],
        k: usize,
        dimension: usize,
        tombstones: &HashSet<u32>,
        metric: DistanceMetric,
    ) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = (0..self.count)
            .filter_map(|i| {
                let doc_id = self.doc_ids[i];
                if tombstones.contains(&doc_id) {
                    return None;
                }
                let start = i * dimension;
                let vec = &self.vectors[start..start + dimension];
                let dist = metric.distance(query, vec);
                Some((doc_id, dist))
            })
            .collect();
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        results
    }
}

/// LSM-tiered streaming vector index.
///
/// Provides O(1) amortized inserts with multi-level search across
/// independently-built HNSW graphs.
pub struct LsmIndex {
    config: LsmConfig,
    /// levels[0] = L0 (buffer), levels[1] = L1, etc.
    levels: Vec<Level>,
    /// Global tombstone set (filtered from all search results).
    tombstones: HashSet<u32>,
    /// Total vectors inserted (including deleted).
    total_inserts: u64,
    /// Total deletes.
    total_deletes: u64,
    /// Total compactions performed.
    total_compactions: u64,
}

impl LsmIndex {
    /// Create a new LSM-tiered index.
    pub fn new(config: LsmConfig) -> Self {
        let mut levels = Vec::with_capacity(config.max_levels);
        levels.push(Level::new()); // L0
        Self {
            config,
            levels,
            tombstones: HashSet::new(),
            total_inserts: 0,
            total_deletes: 0,
            total_compactions: 0,
        }
    }

    /// Insert a vector.
    ///
    /// Appends to L0. When L0 exceeds `buffer_capacity`, triggers compaction.
    pub fn insert(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<()> {
        self.insert_slice(doc_id, &vector)
    }

    /// Insert from a borrowed slice.
    pub fn insert_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.config.dimension,
            });
        }

        // Remove from tombstones if re-inserting a deleted ID
        self.tombstones.remove(&doc_id);

        let stored = self.prepare_vector(vector);
        self.levels[0].vectors.extend_from_slice(&stored);
        self.levels[0].doc_ids.push(doc_id);
        self.levels[0].count += 1;
        self.total_inserts += 1;

        // Auto-compact if L0 is full
        if self.levels[0].count >= self.config.buffer_capacity {
            self.compact()?;
        }

        Ok(())
    }

    /// Mark a vector for deletion.
    pub fn delete(&mut self, doc_id: u32) {
        self.tombstones.insert(doc_id);
        self.total_deletes += 1;
    }

    fn prepare_vector(&self, vector: &[f32]) -> Vec<f32> {
        if self.config.distance_metric != DistanceMetric::Cosine {
            return vector.to_vec();
        }

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            vector.iter().map(|x| x / norm).collect()
        } else {
            vector.to_vec()
        }
    }

    /// Search across all levels, merging results.
    ///
    /// Searches each level independently, filters tombstones, and merges by
    /// distance. Cost: `O(L * search_per_level)` where L = number of levels.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Search across all levels with an explicit per-level HNSW `ef_search`.
    ///
    /// This is useful for benchmark sweeps because the LSM levels can stay in
    /// place while the search beam changes.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u32, f32)>> {
        if query.len() != self.config.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.config.dimension,
            });
        }

        let query_normalized = self.prepare_vector(query);
        let query = query_normalized.as_slice();

        let mut all_results: Vec<(u32, f32)> = Vec::new();

        for (level_idx, level) in self.levels.iter().enumerate() {
            if level.is_empty() {
                continue;
            }

            let level_results = if level_idx == 0 {
                // L0: brute-force
                level.brute_force_search(
                    query,
                    k,
                    self.config.dimension,
                    &self.tombstones,
                    self.config.distance_metric,
                )
            } else {
                // L1+: HNSW search
                #[cfg(feature = "hnsw")]
                {
                    if let Some(ref hnsw) = level.hnsw {
                        let ef = ef_search.max(k);
                        match hnsw.search(query, k, ef) {
                            Ok(results) => results
                                .into_iter()
                                .filter(|(id, _)| !self.tombstones.contains(id))
                                .collect(),
                            Err(_) => Vec::new(),
                        }
                    } else {
                        // Fallback to brute-force if HNSW not built
                        level.brute_force_search(
                            query,
                            k,
                            self.config.dimension,
                            &self.tombstones,
                            self.config.distance_metric,
                        )
                    }
                }
                #[cfg(not(feature = "hnsw"))]
                {
                    level.brute_force_search(
                        query,
                        k,
                        self.config.dimension,
                        &self.tombstones,
                        self.config.distance_metric,
                    )
                }
            };

            all_results.extend(level_results);
        }

        // Deduplicate (keep lowest distance per ID)
        let mut seen = HashSet::new();
        all_results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        all_results.retain(|(id, _)| seen.insert(*id));
        all_results.truncate(k);

        Ok(all_results)
    }

    /// Trigger compaction: merge L0 into L1, and cascade if needed.
    ///
    /// Compaction builds an HNSW graph from the merged vectors and replaces
    /// the target level. Tombstoned vectors are excluded during the merge.
    pub fn compact(&mut self) -> Result<()> {
        if self.levels[0].is_empty() {
            return Ok(());
        }

        // Drain L0
        let l0 = std::mem::replace(&mut self.levels[0], Level::new());

        // Ensure L1 exists
        if self.levels.len() < 2 {
            self.levels.push(Level::new());
        }

        // Merge L0 vectors (excluding tombstones) into L1
        self.merge_into_level(l0, 1)?;

        // Cascade: check if L1 should merge into L2, etc.
        self.cascade_compact(1)?;

        self.total_compactions += 1;
        Ok(())
    }

    /// Merge a source level's vectors into target level, rebuilding the HNSW graph.
    fn merge_into_level(&mut self, source: Level, target_idx: usize) -> Result<()> {
        let dim = self.config.dimension;

        // Collect all vectors for the new level (target + source, minus tombstones)
        let mut merged_vectors: Vec<f32> = Vec::new();
        let mut merged_ids: Vec<u32> = Vec::new();

        // Add existing target level vectors
        if target_idx < self.levels.len() {
            let target = &self.levels[target_idx];
            for i in 0..target.count {
                let doc_id = target.doc_ids[i];
                if !self.tombstones.contains(&doc_id) {
                    let start = i * dim;
                    merged_vectors.extend_from_slice(&target.vectors[start..start + dim]);
                    merged_ids.push(doc_id);
                }
            }
        }

        // Add source vectors
        for i in 0..source.count {
            let doc_id = source.doc_ids[i];
            if !self.tombstones.contains(&doc_id) {
                let start = i * dim;
                merged_vectors.extend_from_slice(&source.vectors[start..start + dim]);
                merged_ids.push(doc_id);
            }
        }

        let merged_count = merged_ids.len();

        // Build HNSW graph for the merged level
        #[cfg(feature = "hnsw")]
        let hnsw = if merged_count > 0 {
            let mut hnsw = crate::hnsw::HNSWIndex::builder(dim)
                .m(self.config.hnsw_m)
                .ef_construction(self.config.hnsw_ef_construction)
                .metric(self.config.distance_metric)
                .auto_normalize(false) // cosine vectors are already normalized
                .build()?;
            for (i, &doc_id) in merged_ids.iter().enumerate() {
                let start = i * dim;
                hnsw.add_slice(doc_id, &merged_vectors[start..start + dim])?;
            }
            hnsw.build()?;
            Some(hnsw)
        } else {
            None
        };

        // Replace the target level
        while self.levels.len() <= target_idx {
            self.levels.push(Level::new());
        }
        self.levels[target_idx] = Level {
            vectors: merged_vectors,
            doc_ids: merged_ids,
            count: merged_count,
            #[cfg(feature = "hnsw")]
            hnsw,
        };

        Ok(())
    }

    /// Cascade compaction: if level i exceeds T * level i+1, merge down.
    fn cascade_compact(&mut self, level_idx: usize) -> Result<()> {
        if level_idx >= self.config.max_levels - 1 {
            return Ok(()); // Don't cascade past max
        }

        let level_size = self.levels.get(level_idx).map_or(0, |l| l.count);
        let next_size = self.levels.get(level_idx + 1).map_or(0, |l| l.count);

        // Compact if this level is T times larger than the next (or next is empty
        // and this level has enough vectors to warrant a new level)
        let should_compact = if next_size == 0 {
            level_size >= self.config.buffer_capacity * self.config.size_ratio
        } else {
            level_size >= self.config.size_ratio * next_size
        };

        if should_compact {
            // Drain current level
            let source = std::mem::replace(&mut self.levels[level_idx], Level::new());

            // Ensure next level exists
            while self.levels.len() <= level_idx + 1 {
                self.levels.push(Level::new());
            }

            self.merge_into_level(source, level_idx + 1)?;

            // Continue cascading
            self.cascade_compact(level_idx + 1)?;
        }

        Ok(())
    }

    /// Number of active (non-tombstoned) vectors across all levels.
    pub fn len(&self) -> usize {
        let total: usize = self.levels.iter().map(|l| l.count).sum();
        // Approximate: subtract tombstones (may overcount if tombstone not in any level)
        total.saturating_sub(self.tombstones.len())
    }

    /// Whether the index has no active vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of levels (including L0).
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Number of vectors at each level.
    pub fn level_sizes(&self) -> Vec<usize> {
        self.levels.iter().map(|l| l.count).collect()
    }

    /// Statistics.
    pub fn stats(&self) -> LsmStats {
        LsmStats {
            total_inserts: self.total_inserts,
            total_deletes: self.total_deletes,
            total_compactions: self.total_compactions,
            num_levels: self.levels.len(),
            level_sizes: self.level_sizes(),
            tombstone_count: self.tombstones.len(),
        }
    }

    /// Force-compact all levels into a single bottom level.
    ///
    /// Useful for read-heavy workloads: eliminates multi-level search overhead.
    /// Expensive: rebuilds the entire graph.
    pub fn force_merge_all(&mut self) -> Result<()> {
        if self.levels.is_empty() {
            return Ok(());
        }

        let dim = self.config.dimension;
        let mut all_vectors: Vec<f32> = Vec::new();
        let mut all_ids: Vec<u32> = Vec::new();

        for level in &self.levels {
            for i in 0..level.count {
                let doc_id = level.doc_ids[i];
                if !self.tombstones.contains(&doc_id) {
                    let start = i * dim;
                    all_vectors.extend_from_slice(&level.vectors[start..start + dim]);
                    all_ids.push(doc_id);
                }
            }
        }

        // Clear all levels
        self.levels.clear();
        self.levels.push(Level::new()); // Fresh L0

        let count = all_ids.len();
        if count == 0 {
            return Ok(());
        }

        // Build single HNSW
        #[cfg(feature = "hnsw")]
        let hnsw = {
            let mut hnsw = crate::hnsw::HNSWIndex::builder(dim)
                .m(self.config.hnsw_m)
                .ef_construction(self.config.hnsw_ef_construction)
                .metric(self.config.distance_metric)
                .auto_normalize(false)
                .build()?;
            for (i, &doc_id) in all_ids.iter().enumerate() {
                let start = i * dim;
                hnsw.add_slice(doc_id, &all_vectors[start..start + dim])?;
            }
            hnsw.build()?;
            Some(hnsw)
        };

        self.levels.push(Level {
            vectors: all_vectors,
            doc_ids: all_ids,
            count,
            #[cfg(feature = "hnsw")]
            hnsw,
        });

        // Clear tombstones (all surviving vectors are in the merged level)
        self.tombstones.clear();

        Ok(())
    }
}

/// Statistics for the LSM index.
#[derive(Debug, Clone)]
pub struct LsmStats {
    /// Total vectors inserted.
    pub total_inserts: u64,
    /// Total deletes.
    pub total_deletes: u64,
    /// Total compactions.
    pub total_compactions: u64,
    /// Number of levels.
    pub num_levels: usize,
    /// Vectors per level.
    pub level_sizes: Vec<usize>,
    /// Active tombstones.
    pub tombstone_count: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_config(dim: usize) -> LsmConfig {
        LsmConfig {
            dimension: dim,
            buffer_capacity: 20,
            size_ratio: 5,
            max_levels: 4,
            hnsw_m: 8,
            hnsw_ef_construction: 50,
            ef_search: 50,
            distance_metric: DistanceMetric::L2,
        }
    }

    fn make_l2_config(dim: usize, buffer_capacity: usize) -> LsmConfig {
        LsmConfig {
            buffer_capacity,
            ..make_config(dim)
        }
    }

    fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| (seed as f32 * 0.1 + i as f32 * 0.01).sin())
            .collect()
    }

    #[test]
    fn insert_and_search_l0() {
        let mut index = LsmIndex::new(make_config(8));

        for i in 0..10u32 {
            index.insert(i, make_vector(8, i)).unwrap();
        }

        // All in L0, no compaction yet
        assert_eq!(index.level_sizes(), vec![10]);

        let results = index.search(&make_vector(8, 0), 3).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Self-match
    }

    #[test]
    fn l2_l0_preserves_vector_magnitude() {
        let mut index = LsmIndex::new(make_l2_config(2, 10));

        index.insert(0, vec![1.0, 0.0]).unwrap();
        index.insert(1, vec![2.0, 0.0]).unwrap();

        let results = index.search(&[2.0, 0.0], 2).unwrap();
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn l2_compacted_level_preserves_vector_magnitude() {
        let mut index = LsmIndex::new(make_l2_config(2, 2));

        index.insert(0, vec![1.0, 0.0]).unwrap();
        index.insert(1, vec![2.0, 0.0]).unwrap();

        assert!(
            index.level_sizes().len() >= 2,
            "expected compaction into L1"
        );
        let results = index.search_with_ef(&[2.0, 0.0], 2, 20).unwrap();
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn compaction_moves_to_l1() {
        let mut index = LsmIndex::new(make_config(8));

        // Insert enough to trigger compaction (buffer_capacity = 20)
        for i in 0..25u32 {
            index.insert(i, make_vector(8, i)).unwrap();
        }

        let sizes = index.level_sizes();
        // L0 should have remaining vectors, L1 should have compacted ones
        assert!(sizes.len() >= 2, "expected at least 2 levels: {sizes:?}");
        assert!(
            sizes[1] > 0,
            "L1 should have vectors after compaction: {sizes:?}"
        );

        // Search should still work across levels
        let results = index.search(&make_vector(8, 5), 3).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn delete_filters_from_search() {
        let mut index = LsmIndex::new(make_config(8));

        for i in 0..10u32 {
            index.insert(i, make_vector(8, i)).unwrap();
        }

        index.delete(0);
        index.delete(1);

        let results = index.search(&make_vector(8, 0), 10).unwrap();
        for (id, _) in &results {
            assert!(*id != 0 && *id != 1, "deleted ID {id} in results");
        }
    }

    #[test]
    fn delete_survives_compaction() {
        let mut index = LsmIndex::new(make_config(8));

        for i in 0..25u32 {
            index.insert(i, make_vector(8, i)).unwrap();
        }
        index.delete(5);

        // Force another compaction
        for i in 25..50u32 {
            index.insert(i, make_vector(8, i)).unwrap();
        }

        let results = index.search(&make_vector(8, 5), 50).unwrap();
        for (id, _) in &results {
            assert_ne!(*id, 5, "deleted ID 5 in results after compaction");
        }
    }

    #[test]
    fn force_merge_all() {
        let mut index = LsmIndex::new(make_config(8));

        for i in 0..50u32 {
            index.insert(i, make_vector(8, i)).unwrap();
        }

        index.delete(10);
        index.force_merge_all().unwrap();

        let sizes = index.level_sizes();
        // After force merge: L0 (empty) + L1 (all non-tombstoned)
        assert_eq!(sizes[0], 0, "L0 should be empty after merge");
        assert_eq!(sizes[1], 49, "L1 should have 49 vectors (50 - 1 deleted)");

        // Tombstones cleared
        assert_eq!(index.stats().tombstone_count, 0);

        // Search still works
        let results = index.search(&make_vector(8, 0), 3).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn reinsert_after_delete() {
        let mut index = LsmIndex::new(make_config(8));

        index.insert(0, make_vector(8, 0)).unwrap();
        index.delete(0);
        index.insert(0, make_vector(8, 100)).unwrap(); // Re-insert with different vector

        let results = index.search(&make_vector(8, 100), 1).unwrap();
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn empty_search() {
        let index = LsmIndex::new(make_config(8));
        let results = index.search(&make_vector(8, 0), 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn stats_tracking() {
        let mut index = LsmIndex::new(make_config(8));

        for i in 0..30u32 {
            index.insert(i, make_vector(8, i)).unwrap();
        }
        index.delete(0);

        let stats = index.stats();
        assert_eq!(stats.total_inserts, 30);
        assert_eq!(stats.total_deletes, 1);
        assert!(stats.total_compactions >= 1);
        assert_eq!(stats.tombstone_count, 1);
    }
}
