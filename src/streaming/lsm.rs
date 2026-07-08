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
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
#[cfg(feature = "serde")]
use std::io::{BufReader, BufWriter, Write};
#[cfg(feature = "serde")]
use std::path::Path;

#[cfg(feature = "serde")]
const LSM_FORMAT_VERSION: u32 = 1;

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

#[cfg(feature = "serde")]
#[derive(Debug, Serialize, Deserialize)]
struct LsmManifest {
    version: u32,
    config: PersistedLsmConfig,
    level_counts: Vec<usize>,
    tombstone_count: usize,
    total_inserts: u64,
    total_deletes: u64,
    total_compactions: u64,
}

#[cfg(feature = "serde")]
#[derive(Debug, Serialize, Deserialize)]
struct PersistedLsmConfig {
    dimension: usize,
    buffer_capacity: usize,
    size_ratio: usize,
    max_levels: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    ef_search: usize,
    distance_metric: DistanceMetric,
}

#[cfg(feature = "serde")]
impl From<&LsmConfig> for PersistedLsmConfig {
    fn from(config: &LsmConfig) -> Self {
        Self {
            dimension: config.dimension,
            buffer_capacity: config.buffer_capacity,
            size_ratio: config.size_ratio,
            max_levels: config.max_levels,
            hnsw_m: config.hnsw_m,
            hnsw_ef_construction: config.hnsw_ef_construction,
            ef_search: config.ef_search,
            distance_metric: config.distance_metric,
        }
    }
}

#[cfg(feature = "serde")]
impl PersistedLsmConfig {
    fn into_config(self) -> Result<LsmConfig> {
        if self.dimension == 0 {
            return Err(RetrieveError::FormatError(
                "LSM manifest has zero dimension".into(),
            ));
        }
        if self.buffer_capacity == 0 {
            return Err(RetrieveError::FormatError(
                "LSM manifest has zero buffer_capacity".into(),
            ));
        }
        if self.size_ratio == 0 {
            return Err(RetrieveError::FormatError(
                "LSM manifest has zero size_ratio".into(),
            ));
        }
        if self.max_levels == 0 {
            return Err(RetrieveError::FormatError(
                "LSM manifest has zero max_levels".into(),
            ));
        }
        Ok(LsmConfig {
            dimension: self.dimension,
            buffer_capacity: self.buffer_capacity,
            size_ratio: self.size_ratio,
            max_levels: self.max_levels,
            hnsw_m: self.hnsw_m,
            hnsw_ef_construction: self.hnsw_ef_construction,
            ef_search: self.ef_search,
            distance_metric: self.distance_metric,
        })
    }
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

    fn memory_usage(&self) -> crate::memory::MemoryReport {
        let mut report = crate::memory::MemoryReport {
            vectors_bytes: self.vectors.capacity() * std::mem::size_of::<f32>(),
            graph_bytes: 0,
            quantized_bytes: 0,
            metadata_bytes: self.doc_ids.capacity() * std::mem::size_of::<u32>(),
        };

        #[cfg(feature = "hnsw")]
        if let Some(hnsw) = &self.hnsw {
            let hnsw_report = hnsw.memory_usage();
            report.vectors_bytes += hnsw_report.vectors_bytes;
            report.graph_bytes += hnsw_report.graph_bytes;
            report.quantized_bytes += hnsw_report.quantized_bytes;
            report.metadata_bytes += hnsw_report.metadata_bytes;
        }

        report
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

    /// Save the current LSM state as a restart snapshot.
    ///
    /// This is not a write-ahead log. It writes the current mutable L0,
    /// compacted levels, tombstones, config, and counters. HNSW graphs in
    /// compacted levels are derived state and are rebuilt by [`Self::load_from_dir`].
    #[cfg(feature = "serde")]
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        self.validate_snapshot_shape()?;

        let mut tombstones: Vec<u32> = self.tombstones.iter().copied().collect();
        tombstones.sort_unstable();
        let manifest = LsmManifest {
            version: LSM_FORMAT_VERSION,
            config: PersistedLsmConfig::from(&self.config),
            level_counts: self.levels.iter().map(|level| level.count).collect(),
            tombstone_count: tombstones.len(),
            total_inserts: self.total_inserts,
            total_deletes: self.total_deletes,
            total_compactions: self.total_compactions,
        };

        write_json_atomic(&output_dir.join("manifest.json"), &manifest)?;
        write_u32_atomic(&output_dir.join("tombstones.bin"), &tombstones)?;
        for (level_idx, level) in self.levels.iter().enumerate() {
            write_f32_atomic(
                &output_dir.join(format!("level_{level_idx}_vectors.bin")),
                &level.vectors,
            )?;
            write_u32_atomic(
                &output_dir.join(format!("level_{level_idx}_doc_ids.bin")),
                &level.doc_ids,
            )?;
        }
        Ok(())
    }

    /// Load an LSM restart snapshot saved by [`Self::save_to_dir`].
    ///
    /// When compiled with `hnsw`, compacted levels rebuild their HNSW graphs
    /// from the saved vectors and doc IDs. Without `hnsw`, all levels remain
    /// searchable through the brute-force fallback.
    #[cfg(feature = "serde")]
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self> {
        let input_dir = input_dir.as_ref();
        let manifest: LsmManifest = read_json(&input_dir.join("manifest.json"))?;
        if manifest.version != LSM_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported LSM format version {}",
                manifest.version
            )));
        }
        if manifest.level_counts.is_empty() {
            return Err(RetrieveError::FormatError(
                "LSM manifest has no levels".into(),
            ));
        }

        let config = manifest.config.into_config()?;
        if manifest.level_counts.len() > config.max_levels {
            return Err(RetrieveError::FormatError(format!(
                "LSM manifest level count {} exceeds max_levels {}",
                manifest.level_counts.len(),
                config.max_levels
            )));
        }

        let tombstones =
            read_u32_exact(&input_dir.join("tombstones.bin"), manifest.tombstone_count)?
                .into_iter()
                .collect();
        let mut levels = Vec::with_capacity(manifest.level_counts.len());
        for (level_idx, &count) in manifest.level_counts.iter().enumerate() {
            let vector_len = count.checked_mul(config.dimension).ok_or_else(|| {
                RetrieveError::FormatError(format!("LSM level {level_idx} vector length overflow"))
            })?;
            let vectors = read_f32_exact(
                &input_dir.join(format!("level_{level_idx}_vectors.bin")),
                vector_len,
            )?;
            let doc_ids = read_u32_exact(
                &input_dir.join(format!("level_{level_idx}_doc_ids.bin")),
                count,
            )?;
            levels.push(Self::level_from_snapshot(
                &config, level_idx, vectors, doc_ids,
            )?);
        }

        Ok(Self {
            config,
            levels,
            tombstones,
            total_inserts: manifest.total_inserts,
            total_deletes: manifest.total_deletes,
            total_compactions: manifest.total_compactions,
        })
    }

    #[cfg(feature = "serde")]
    fn validate_snapshot_shape(&self) -> Result<()> {
        let dim = self.config.dimension;
        if dim == 0 {
            return Err(RetrieveError::InvalidParameter(
                "cannot save LSM index with zero dimension".into(),
            ));
        }
        if self.levels.is_empty() {
            return Err(RetrieveError::InvalidParameter(
                "cannot save LSM index with no levels".into(),
            ));
        }
        if self.levels.len() > self.config.max_levels {
            return Err(RetrieveError::InvalidParameter(format!(
                "LSM level count {} exceeds max_levels {}",
                self.levels.len(),
                self.config.max_levels
            )));
        }
        for (level_idx, level) in self.levels.iter().enumerate() {
            let expected_vectors = level.count.checked_mul(dim).ok_or_else(|| {
                RetrieveError::InvalidParameter(format!(
                    "LSM level {level_idx} vector length overflow"
                ))
            })?;
            if level.vectors.len() != expected_vectors {
                return Err(RetrieveError::InvalidParameter(format!(
                    "LSM level {level_idx} has {} vector scalars, expected {}",
                    level.vectors.len(),
                    expected_vectors
                )));
            }
            if level.doc_ids.len() != level.count {
                return Err(RetrieveError::InvalidParameter(format!(
                    "LSM level {level_idx} has {} doc ids, expected {}",
                    level.doc_ids.len(),
                    level.count
                )));
            }
        }
        Ok(())
    }

    #[cfg(feature = "serde")]
    fn level_from_snapshot(
        config: &LsmConfig,
        level_idx: usize,
        vectors: Vec<f32>,
        doc_ids: Vec<u32>,
    ) -> Result<Level> {
        let count = doc_ids.len();
        let expected_vectors = count.checked_mul(config.dimension).ok_or_else(|| {
            RetrieveError::FormatError(format!("LSM level {level_idx} vector length overflow"))
        })?;
        if vectors.len() != expected_vectors {
            return Err(RetrieveError::FormatError(format!(
                "LSM level {level_idx} has {} vector scalars, expected {}",
                vectors.len(),
                expected_vectors
            )));
        }

        #[cfg(feature = "hnsw")]
        let hnsw = if level_idx > 0 && count > 0 {
            let mut hnsw = crate::hnsw::HNSWIndex::builder(config.dimension)
                .m(config.hnsw_m)
                .ef_construction(config.hnsw_ef_construction)
                .metric(config.distance_metric)
                .auto_normalize(false)
                .build()?;
            for (i, &doc_id) in doc_ids.iter().enumerate() {
                let start = i * config.dimension;
                hnsw.add_slice(doc_id, &vectors[start..start + config.dimension])?;
            }
            hnsw.build()?;
            Some(hnsw)
        } else {
            None
        };

        Ok(Level {
            vectors,
            doc_ids,
            count,
            #[cfg(feature = "hnsw")]
            hnsw,
        })
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

    /// Estimated heap memory used by all levels and tombstone metadata.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let mut report = crate::memory::MemoryReport {
            vectors_bytes: 0,
            graph_bytes: 0,
            quantized_bytes: 0,
            metadata_bytes: self.levels.capacity() * std::mem::size_of::<Level>()
                + self.tombstones.capacity() * std::mem::size_of::<u32>(),
        };

        for level in &self.levels {
            let level_report = level.memory_usage();
            report.vectors_bytes += level_report.vectors_bytes;
            report.graph_bytes += level_report.graph_bytes;
            report.quantized_bytes += level_report.quantized_bytes;
            report.metadata_bytes += level_report.metadata_bytes;
        }

        report
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

#[cfg(feature = "serde")]
fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    write_atomic(path, |writer| {
        serde_json::to_writer_pretty(writer, value)
            .map_err(|e| std::io::Error::other(e.to_string()))
    })
}

#[cfg(feature = "serde")]
fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let file = std::fs::File::open(path)?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
}

#[cfg(feature = "serde")]
fn write_f32_atomic(path: &Path, values: &[f32]) -> Result<()> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

#[cfg(feature = "serde")]
fn read_f32_exact(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| RetrieveError::FormatError("f32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(feature = "serde")]
fn write_u32_atomic(path: &Path, values: &[u32]) -> Result<()> {
    write_atomic(path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    })
}

#[cfg(feature = "serde")]
fn read_u32_exact(path: &Path, expected_len: usize) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path)?;
    let expected_bytes = expected_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| RetrieveError::FormatError("u32 byte length overflow".into()))?;
    if bytes.len() != expected_bytes {
        return Err(RetrieveError::FormatError(format!(
            "{} size mismatch: expected {} bytes, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(feature = "serde")]
fn write_atomic(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<std::fs::File>) -> std::io::Result<()>,
) -> Result<()> {
    let tmp_path = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        writer.flush()?;
    }
    std::fs::rename(&tmp_path, path)?;
    Ok(())
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

    #[cfg(feature = "serde")]
    #[test]
    fn snapshot_roundtrip_preserves_levels_tombstones_and_counters() {
        let mut index = LsmIndex::new(make_config(8));

        for i in 0..35u32 {
            index.insert(i, make_vector(8, i)).unwrap();
        }
        index.delete(5);
        index.delete(7);

        let before_sizes = index.level_sizes();
        let before_stats = index.stats();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();

        let loaded = LsmIndex::load_from_dir(dir.path()).unwrap();
        assert_eq!(loaded.level_sizes(), before_sizes);
        assert_eq!(loaded.stats().total_inserts, before_stats.total_inserts);
        assert_eq!(loaded.stats().total_deletes, before_stats.total_deletes);
        assert_eq!(
            loaded.stats().total_compactions,
            before_stats.total_compactions
        );
        assert_eq!(loaded.stats().tombstone_count, 2);

        let results = loaded.search(&make_vector(8, 5), 20).unwrap();
        for (id, _) in results {
            assert_ne!(id, 5);
            assert_ne!(id, 7);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn load_rejects_future_snapshot_version() {
        let mut index = LsmIndex::new(make_config(4));
        index.insert(0, make_vector(4, 0)).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();

        let manifest_path = dir.path().join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["version"] = serde_json::json!(LSM_FORMAT_VERSION + 1);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let err = match LsmIndex::load_from_dir(dir.path()) {
            Ok(_) => panic!("future LSM snapshot version should be rejected"),
            Err(err) => err,
        };
        match err {
            RetrieveError::FormatError(message) => {
                assert!(message.contains("unsupported LSM format version"));
            }
            other => panic!("expected format error, got {other:?}"),
        }
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
