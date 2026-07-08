//! Cross-Polytope Locality-Sensitive Hashing (LSH) for approximate nearest neighbor search.
//!
//! Implements the cross-polytope LSH family (Andoni et al., NeurIPS 2015) which achieves
//! optimal sensitivity for angular distance. Combined with multiprobe, a single hash table
//! can match the recall of multiple tables with hyperplane LSH.
//!
//! Hashing primitives (rotation generation, Walsh-Hadamard transform, vertex selection)
//! are provided by the [`sketchir`] crate.
//!
//! # When to use
//!
//! - Streaming/online settings: O(1) insertion (no graph reconnection).
//! - Provable guarantees: (1+ε)-approximation with known space/time tradeoffs.
//! - Distributed systems: hash tables partition naturally across nodes.
//! - Candidate generation: LSH as a first-pass filter feeding exact reranking.
//!
//! # References
//!
//! - Andoni, Indyk, Laarhoven, Razenshteyn, Schmidt (2015). "Practical and Optimal LSH
//!   for Angular Distance." NeurIPS. <https://people.csail.mit.edu/ludwigs/papers/nips15_crosspolytopelsh.pdf>

use crate::distance;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use sketchir::cross_polytope::{self, CrossPolytopeHasher};
use std::collections::HashMap;
use std::path::Path;

/// Parameters for cross-polytope LSH.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LSHParams {
    /// Number of hash tables (L). More tables = higher recall, more memory.
    pub num_tables: usize,
    /// Number of buckets to probe per table per query (multiprobe).
    /// 1 = single probe, higher = better recall at cost of more distance computations.
    pub num_probes: usize,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for LSHParams {
    fn default() -> Self {
        Self {
            num_tables: 8,
            num_probes: 4,
            seed: None,
        }
    }
}

/// Cross-polytope LSH index for approximate nearest neighbor search.
///
/// Uses random rotations + cross-polytope vertex selection for angular-distance-sensitive
/// hashing. Supports multiprobe for improved recall with fewer tables.
#[derive(Debug)]
pub struct CrossPolytopeLSHIndex {
    /// Flat vector storage.
    vectors: Vec<f32>,
    /// Vector dimension.
    dimension: usize,
    /// Number of stored vectors.
    num_vectors: usize,
    /// Parameters.
    params: LSHParams,
    /// L independent hashers (one per table), from sketchir.
    hashers: Vec<CrossPolytopeHasher>,
    /// Seed used to build `hashers`. `params.seed` can be `None`, but a built
    /// index needs the realized seed for deterministic reloads.
    hasher_seed: Option<u64>,
    /// L hash tables. Each table maps bucket_id -> list of vector indices.
    tables: Vec<HashMap<u32, Vec<u32>>>,
    /// Whether the index has been built.
    built: bool,
}

impl CrossPolytopeLSHIndex {
    /// Create a new cross-polytope LSH index.
    pub fn new(dimension: usize, params: LSHParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if params.num_tables == 0 {
            return Err(RetrieveError::InvalidParameter(
                "num_tables must be > 0".into(),
            ));
        }
        if params.num_probes == 0 {
            return Err(RetrieveError::InvalidParameter(
                "num_probes must be > 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            params,
            hashers: Vec::new(),
            hasher_seed: None,
            tables: Vec::new(),
            built: false,
        })
    }

    /// Add vectors to the index (flat layout, length must be a multiple of dimension).
    pub fn add_vectors(&mut self, vectors: &[f32]) -> Result<(), RetrieveError> {
        if !vectors.len().is_multiple_of(self.dimension) {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vectors.len(),
                doc_dim: self.dimension,
            });
        }

        self.vectors.extend_from_slice(vectors);
        self.num_vectors += vectors.len() / self.dimension;
        self.built = false;
        Ok(())
    }

    /// Insert a single vector. Returns the assigned index.
    ///
    /// If the index is already built, the vector is hashed into existing tables
    /// immediately (O(1) insert per table, no graph reconnection needed).
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32, RetrieveError> {
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        let id = self.num_vectors as u32;
        self.vectors.extend_from_slice(vector);
        self.num_vectors += 1;

        // If already built, hash into existing tables immediately.
        if self.built {
            for (table_idx, hasher) in self.hashers.iter().enumerate() {
                if let Ok(bucket) = hasher.hash(vector) {
                    self.tables[table_idx].entry(bucket).or_default().push(id);
                }
            }
        }

        Ok(id)
    }

    /// Build the index: generate rotation matrices and hash all vectors.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let base_seed = self.params.seed.or(self.hasher_seed).unwrap_or_else(|| {
            use rand::RngCore;
            rand::rng().next_u64()
        });

        // Generate L independent hashers via sketchir.
        self.hashers =
            cross_polytope::multi_hasher(self.dimension, self.params.num_tables, base_seed)
                .map_err(|e| RetrieveError::InvalidParameter(format!("sketchir: {e}")))?;
        self.hasher_seed = Some(base_seed);

        // Hash all vectors into tables.
        self.tables = vec![HashMap::new(); self.params.num_tables];
        for vec_idx in 0..self.num_vectors {
            let start = vec_idx * self.dimension;
            let vec = &self.vectors[start..start + self.dimension];
            for (table_idx, hasher) in self.hashers.iter().enumerate() {
                // Infallible: dimension was checked at add time.
                if let Ok(bucket) = hasher.hash(vec) {
                    self.tables[table_idx]
                        .entry(bucket)
                        .or_default()
                        .push(vec_idx as u32);
                }
            }
        }

        self.built = true;
        Ok(())
    }

    /// Save a built LSH index to a directory snapshot.
    ///
    /// Hash tables are rebuilt on load from the persisted vectors and realized
    /// hasher seed. This keeps the format small while preserving exact search
    /// behavior for deterministic `sketchir` hash generation.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter("index not built".into()));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        let manifest = LshSnapshotManifest {
            magic: *b"VICLSH01",
            version: 1,
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            params: self.params.clone(),
            hasher_seed: self
                .hasher_seed
                .ok_or_else(|| RetrieveError::FormatError("built LSH index has no seed".into()))?,
        };
        crate::graph_snapshot::write_json_atomic(&output_dir.join("manifest.json"), &manifest)?;
        crate::graph_snapshot::write_f32_atomic(&output_dir.join("vectors.bin"), &self.vectors)?;
        Ok(())
    }

    /// Load an LSH index from a directory snapshot.
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let manifest: LshSnapshotManifest =
            crate::graph_snapshot::read_json(&input_dir.join("manifest.json"))?;
        if manifest.magic != *b"VICLSH01" {
            return Err(RetrieveError::FormatError(
                "invalid LSH snapshot magic".into(),
            ));
        }
        if manifest.version != 1 {
            return Err(RetrieveError::FormatError(format!(
                "unsupported LSH snapshot version {}",
                manifest.version
            )));
        }
        if manifest.dimension == 0 {
            return Err(RetrieveError::FormatError(
                "LSH snapshot has zero dimension".into(),
            ));
        }

        let vector_len = manifest
            .num_vectors
            .checked_mul(manifest.dimension)
            .ok_or_else(|| RetrieveError::FormatError("LSH vector length overflow".into()))?;
        let vectors =
            crate::graph_snapshot::read_f32_exact(&input_dir.join("vectors.bin"), vector_len)?;

        let mut index = Self::new(manifest.dimension, manifest.params)?;
        index.hasher_seed = Some(manifest.hasher_seed);
        index.add_vectors(&vectors)?;
        index.build()?;
        Ok(index)
    }

    /// Search for the k nearest neighbors of `query`.
    ///
    /// Returns up to k results as `(vector_index, distance)` pairs, sorted by distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter("index not built".into()));
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        // Collect candidate set from all tables using multiprobe.
        let mut candidates = Vec::new();
        let mut seen = vec![false; self.num_vectors];

        for (table_idx, hasher) in self.hashers.iter().enumerate() {
            // hash_ranked returns the top-k vertices sorted by coordinate magnitude,
            // which is exactly the multiprobe set.
            let probes = hasher
                .hash_ranked(query, self.params.num_probes)
                .unwrap_or_default();

            for bucket_id in probes {
                if let Some(ids) = self.tables[table_idx].get(&bucket_id) {
                    for &id in ids {
                        if !seen[id as usize] {
                            seen[id as usize] = true;
                            candidates.push(id);
                        }
                    }
                }
            }
        }

        // Exact distance computation on candidates.
        let mut results: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&id| {
                let dist = distance::l2_distance(query, self.get_vector(id as usize));
                (id, dist)
            })
            .collect();

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        Ok(results)
    }

    /// Hash a vector using the hasher for `table_idx`.
    #[cfg(test)]
    fn hash_vector(&self, vector: &[f32], table_idx: usize) -> u32 {
        self.hashers[table_idx].hash(vector).unwrap_or(0)
    }

    /// Get vector by index.
    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Get index statistics.
    pub fn stats(&self) -> LSHStats {
        let total_entries: usize = self
            .tables
            .iter()
            .map(|t| t.values().map(|v| v.len()).sum::<usize>())
            .sum();
        let num_buckets: usize = self.tables.iter().map(|t| t.len()).sum();

        LSHStats {
            num_vectors: self.num_vectors,
            num_tables: self.params.num_tables,
            num_probes: self.params.num_probes,
            num_occupied_buckets: num_buckets,
            avg_bucket_size: if num_buckets > 0 {
                total_entries as f32 / num_buckets as f32
            } else {
                0.0
            },
        }
    }

    /// Estimated heap memory used by this index.
    #[must_use]
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let f32_bytes = std::mem::size_of::<f32>();
        let vectors_bytes = self.vectors.capacity() * f32_bytes;
        let graph_bytes = self.tables.capacity() * std::mem::size_of::<HashMap<u32, Vec<u32>>>()
            + self
                .tables
                .iter()
                .map(|table| {
                    table.capacity()
                        * (std::mem::size_of::<u32>() + std::mem::size_of::<Vec<u32>>())
                        + table
                            .values()
                            .map(|ids| ids.capacity() * std::mem::size_of::<u32>())
                            .sum::<usize>()
                })
                .sum::<usize>();
        let metadata_bytes = self.hashers.capacity() * std::mem::size_of::<CrossPolytopeHasher>()
            + self.hashers.len() * estimated_hasher_rotation_bytes(self.dimension);

        crate::memory::MemoryReport {
            vectors_bytes,
            graph_bytes,
            quantized_bytes: 0,
            metadata_bytes,
        }
    }
}

fn estimated_hasher_rotation_bytes(dim: usize) -> usize {
    let rotation_len = if dim <= 64 {
        dim.saturating_mul(dim)
    } else {
        dim.checked_next_power_of_two()
            .map(|padded| 2usize.saturating_add(3usize.saturating_mul(padded)))
            .unwrap_or(usize::MAX / std::mem::size_of::<f32>())
    };
    rotation_len.saturating_mul(std::mem::size_of::<f32>())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LshSnapshotManifest {
    magic: [u8; 8],
    version: u32,
    dimension: usize,
    num_vectors: usize,
    params: LSHParams,
    hasher_seed: u64,
}

/// Statistics about a cross-polytope LSH index.
#[derive(Debug, Clone)]
pub struct LSHStats {
    /// Number of indexed vectors.
    pub num_vectors: usize,
    /// Number of hash tables.
    pub num_tables: usize,
    /// Number of probes per query.
    pub num_probes: usize,
    /// Total occupied buckets across all tables.
    pub num_occupied_buckets: usize,
    /// Average number of vectors per occupied bucket.
    pub avg_bucket_size: f32,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Helper: create clustered test data.
    fn clustered_data(n_clusters: usize, points_per_cluster: usize, dim: usize) -> Vec<f32> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = Vec::new();

        for c in 0..n_clusters {
            let center: Vec<f32> = (0..dim)
                .map(|_| (c as f32) * 10.0 + rng.random::<f32>())
                .collect();
            for _ in 0..points_per_cluster {
                for val in &center {
                    data.push(val + rng.random::<f32>() * 0.5);
                }
            }
        }
        data
    }

    /// Helper: brute-force k-NN.
    fn brute_force_knn(data: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let n = data.len() / dim;
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                (i, distance::l2_distance(query, v))
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        dists.truncate(k);
        dists
    }

    #[test]
    fn test_build_and_search() {
        let dim = 16;
        let data = clustered_data(5, 40, dim);
        let n = data.len() / dim;

        let params = LSHParams {
            num_tables: 12,
            num_probes: 6,
            seed: Some(42),
        };

        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();
        index.add_vectors(&data).unwrap();
        index.build().unwrap();

        assert_eq!(index.num_vectors, n);

        // Query with a point from the dataset — should find itself.
        let query = &data[0..dim];
        let results = index.search(query, 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0, "should find the query point itself");
        assert!(results[0].1 < 1e-6, "self-distance should be ~0");
    }

    #[test]
    fn test_recall() {
        let dim = 16;
        let data = clustered_data(5, 100, dim);

        let params = LSHParams {
            num_tables: 16,
            num_probes: 8,
            seed: Some(42),
        };

        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();
        index.add_vectors(&data).unwrap();
        index.build().unwrap();

        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(123);
        let n = data.len() / dim;
        let num_queries = 30;
        let k = 10;
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query_idx = rng.random_range(0..n);
            let query = &data[query_idx * dim..(query_idx + 1) * dim];

            let results = index.search(query, k).unwrap();
            let gt = brute_force_knn(&data, dim, query, k);

            let gt_set: std::collections::HashSet<u32> =
                gt.iter().map(|&(id, _)| id as u32).collect();
            let result_set: std::collections::HashSet<u32> =
                results.iter().map(|&(id, _)| id).collect();

            let hits = gt_set.intersection(&result_set).count();
            total_recall += hits as f32 / k as f32;
        }

        let avg_recall = total_recall / num_queries as f32;
        // LSH with 16 tables and 8 probes on clustered 16d data should achieve
        // reasonable recall. The bar is lower than graph methods.
        assert!(
            avg_recall > 0.3,
            "LSH recall too low: {:.2}% (expected >30%)",
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_multiprobe_improves_recall() {
        let dim = 16;
        let data = clustered_data(5, 80, dim);
        let k = 10;

        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(123);
        let n = data.len() / dim;
        let queries: Vec<usize> = (0..20).map(|_| rng.random_range(0..n)).collect();

        // Single probe.
        let params1 = LSHParams {
            num_tables: 8,
            num_probes: 1,
            seed: Some(42),
        };
        let mut idx1 = CrossPolytopeLSHIndex::new(dim, params1).unwrap();
        idx1.add_vectors(&data).unwrap();
        idx1.build().unwrap();

        // Multi probe.
        let params4 = LSHParams {
            num_tables: 8,
            num_probes: 6,
            seed: Some(42),
        };
        let mut idx4 = CrossPolytopeLSHIndex::new(dim, params4).unwrap();
        idx4.add_vectors(&data).unwrap();
        idx4.build().unwrap();

        let mut recall1 = 0.0;
        let mut recall4 = 0.0;

        for &qi in &queries {
            let query = &data[qi * dim..(qi + 1) * dim];
            let gt = brute_force_knn(&data, dim, query, k);
            let gt_set: std::collections::HashSet<u32> =
                gt.iter().map(|&(id, _)| id as u32).collect();

            let r1 = idx1.search(query, k).unwrap();
            let r4 = idx4.search(query, k).unwrap();

            let s1: std::collections::HashSet<u32> = r1.iter().map(|&(id, _)| id).collect();
            let s4: std::collections::HashSet<u32> = r4.iter().map(|&(id, _)| id).collect();

            recall1 += gt_set.intersection(&s1).count() as f32 / k as f32;
            recall4 += gt_set.intersection(&s4).count() as f32 / k as f32;
        }

        recall1 /= queries.len() as f32;
        recall4 /= queries.len() as f32;

        assert!(
            recall4 >= recall1,
            "Multiprobe recall ({:.2}%) should be >= single probe ({:.2}%)",
            recall4 * 100.0,
            recall1 * 100.0
        );
    }

    #[test]
    fn test_online_insert() {
        let dim = 8;
        let params = LSHParams {
            num_tables: 4,
            num_probes: 2,
            seed: Some(42),
        };

        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();

        // Add initial batch and build.
        let initial: Vec<f32> = (0..20).flat_map(|i| vec![i as f32; dim]).collect();
        index.add_vectors(&initial).unwrap();
        index.build().unwrap();

        // Online insert after build.
        let new_vec = vec![5.5; dim];
        let new_id = index.insert(&new_vec).unwrap();

        // Should be searchable immediately.
        let results = index.search(&new_vec, 3).unwrap();
        assert!(
            results.iter().any(|&(id, _)| id == new_id),
            "newly inserted vector should be found"
        );
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let dim = 16;
        let data = clustered_data(4, 30, dim);
        let params = LSHParams {
            num_tables: 10,
            num_probes: 5,
            seed: None,
        };

        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();
        index.add_vectors(&data).unwrap();
        index.build().unwrap();
        let inserted_id = index.insert(&vec![2.5; dim]).unwrap();

        let query = &data[dim..2 * dim];
        let before = index.search(query, 8).unwrap();
        assert!(
            index
                .search(&vec![2.5; dim], 8)
                .unwrap()
                .iter()
                .any(|(id, _)| *id == inserted_id),
            "online insert should be present before snapshot"
        );

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = CrossPolytopeLSHIndex::load_from_dir(dir.path()).unwrap();
        let after = loaded.search(query, 8).unwrap();

        assert_eq!(after, before);
        assert!(
            loaded
                .search(&vec![2.5; dim], 8)
                .unwrap()
                .iter()
                .any(|(id, _)| *id == inserted_id),
            "online insert should survive snapshot"
        );
    }

    #[test]
    fn test_hash_determinism() {
        let dim = 8;
        let params = LSHParams {
            num_tables: 4,
            num_probes: 2,
            seed: Some(999),
        };

        let data: Vec<f32> = (0..80).map(|i| (i as f32) * 0.1).collect();

        let mut idx1 = CrossPolytopeLSHIndex::new(dim, params.clone()).unwrap();
        idx1.add_vectors(&data).unwrap();
        idx1.build().unwrap();

        let mut idx2 = CrossPolytopeLSHIndex::new(dim, params).unwrap();
        idx2.add_vectors(&data).unwrap();
        idx2.build().unwrap();

        // Same seed should produce identical results.
        let query = &data[0..dim];
        let r1 = idx1.search(query, 5).unwrap();
        let r2 = idx2.search(query, 5).unwrap();
        assert_eq!(r1, r2, "same seed should produce identical results");
    }

    #[test]
    fn test_similar_vectors_hash_together() {
        let dim = 32;
        let params = LSHParams {
            num_tables: 1,
            num_probes: 1,
            seed: Some(42),
        };

        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();

        // Two very similar vectors.
        let v1: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let v2: Vec<f32> = (0..dim).map(|i| i as f32 + 0.001).collect();
        // One very different vector.
        let v3: Vec<f32> = (0..dim).map(|i| -(i as f32) * 10.0).collect();

        let mut all = v1.clone();
        all.extend(&v2);
        all.extend(&v3);
        index.add_vectors(&all).unwrap();
        index.build().unwrap();

        // Check that v1 and v2 hash to the same bucket in the single table.
        let h1 = index.hash_vector(&v1, 0);
        let h2 = index.hash_vector(&v2, 0);
        let h3 = index.hash_vector(&v3, 0);

        assert_eq!(h1, h2, "similar vectors should hash to same bucket");
        // h3 may or may not differ, but the similar pair should agree.
        let _ = h3; // used to verify compilation
    }

    #[test]
    fn test_cross_polytope_vertex_basic() {
        // Verify via sketchir hasher: argmax-based vertex selection.
        let hasher = CrossPolytopeHasher::new(3, 0).unwrap();
        // For a small dim, the dense rotation means raw vertex tests
        // don't directly apply, but we can test through the hasher.
        let v = vec![1.0_f32, 0.0, 0.0];
        let bucket = hasher.hash(&v).unwrap();
        assert!(bucket < 6, "bucket should be in 0..2*dim");
    }

    #[test]
    fn test_hadamard_search_recall() {
        // End-to-end: build with Hadamard rotation (dim=128), verify recall.
        let dim = 128;
        let n_clusters = 5;
        let points_per_cluster = 60;

        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = Vec::new();
        for c in 0..n_clusters {
            let center: Vec<f32> = (0..dim)
                .map(|_| (c as f32) * 5.0 + rng.random::<f32>())
                .collect();
            for _ in 0..points_per_cluster {
                for val in &center {
                    data.push(val + rng.random::<f32>() * 0.3);
                }
            }
        }
        let n = data.len() / dim;

        let params = LSHParams {
            num_tables: 16,
            num_probes: 8,
            seed: Some(42),
        };

        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();
        index.add_vectors(&data).unwrap();
        index.build().unwrap();

        let mut total_recall = 0.0;
        let num_queries = 20;
        let k = 10;

        for qi in 0..num_queries {
            let query = &data[qi * dim..(qi + 1) * dim];
            let results = index.search(query, k).unwrap();
            let gt = brute_force_knn(&data, dim, query, k);

            let gt_set: std::collections::HashSet<u32> =
                gt.iter().map(|&(id, _)| id as u32).collect();
            let result_set: std::collections::HashSet<u32> =
                results.iter().map(|&(id, _)| id).collect();
            total_recall += gt_set.intersection(&result_set).count() as f32 / k as f32;
        }

        let avg_recall = total_recall / num_queries as f32;
        assert!(
            avg_recall > 0.3,
            "Hadamard LSH recall too low: {:.1}% on {}d data (n={})",
            avg_recall * 100.0,
            dim,
            n
        );
    }

    #[test]
    fn test_empty_index() {
        let params = LSHParams::default();
        let mut index = CrossPolytopeLSHIndex::new(4, params).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let params = LSHParams::default();
        let mut index = CrossPolytopeLSHIndex::new(4, params).unwrap();
        index.add_vectors(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        index.build().unwrap();

        // Wrong query dimension.
        let result = index.search(&[1.0, 2.0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats() {
        let dim = 8;
        let params = LSHParams {
            num_tables: 4,
            num_probes: 2,
            seed: Some(42),
        };

        let data: Vec<f32> = (0..240).map(|i| (i as f32) * 0.1).collect();
        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();
        index.add_vectors(&data).unwrap();
        index.build().unwrap();

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 30);
        assert_eq!(stats.num_tables, 4);
        assert!(stats.num_occupied_buckets > 0);
        assert!(stats.avg_bucket_size > 0.0);
    }

    #[test]
    fn memory_usage_reports_vectors_hash_tables_and_hashers() {
        let dim = 16;
        let num_tables = 6;
        let data = clustered_data(4, 20, dim);
        let params = LSHParams {
            num_tables,
            num_probes: 3,
            seed: Some(42),
        };

        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();
        index.add_vectors(&data).unwrap();
        index.build().unwrap();

        let report = index.memory_usage();
        assert!(report.vectors_bytes >= data.len() * std::mem::size_of::<f32>());
        assert!(report.graph_bytes > num_tables * std::mem::size_of::<HashMap<u32, Vec<u32>>>());
        assert!(
            report.metadata_bytes
                >= num_tables
                    * (std::mem::size_of::<CrossPolytopeHasher>()
                        + dim * dim * std::mem::size_of::<f32>())
        );
        assert_eq!(report.quantized_bytes, 0);
        assert!(report.total() >= report.vectors_bytes + report.graph_bytes);
    }
}
