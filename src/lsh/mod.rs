//! Cross-Polytope Locality-Sensitive Hashing (LSH) for approximate nearest neighbor search.
//!
//! Implements the cross-polytope LSH family (Andoni et al., NeurIPS 2015) which achieves
//! optimal sensitivity for angular distance. Combined with multiprobe, a single hash table
//! can match the recall of multiple tables with hyperplane LSH.
//!
//! # When to use
//!
//! - Streaming/online settings: O(1) insertion (no graph reconnection).
//! - Provable guarantees: (1+ε)-approximation with known space/time tradeoffs.
//! - Distributed systems: hash tables partition naturally across nodes.
//! - Candidate generation: LSH as a first-pass filter feeding exact reranking.
//!
//! # How it works
//!
//! 1. Apply a random orthogonal rotation to the input vector.
//! 2. Find the coordinate with maximum absolute value — this selects one of 2d
//!    vertices of the cross-polytope (the unit ball of the L1 norm).
//! 3. Repeat with L independent rotations (one per hash table).
//! 4. At query time, probe the matching bucket plus nearby buckets (multiprobe).
//!
//! # References
//!
//! - Andoni, Indyk, Laarhoven, Razenshteyn, Schmidt (2015). "Practical and Optimal LSH
//!   for Angular Distance." NeurIPS. <https://people.csail.mit.edu/ludwigs/papers/nips15_crosspolytopelsh.pdf>

use crate::distance;
use crate::RetrieveError;
use rand::prelude::*;
use std::collections::HashMap;

/// Parameters for cross-polytope LSH.
#[derive(Clone, Debug)]
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
    /// L rotation matrices, each dimension x dimension stored row-major.
    rotations: Vec<Vec<f32>>,
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
            rotations: Vec::new(),
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
            for table_idx in 0..self.tables.len() {
                let bucket = hash_with_rotation(vector, &self.rotations[table_idx], self.dimension);
                self.tables[table_idx].entry(bucket).or_default().push(id);
            }
        }

        Ok(id)
    }

    /// Build the index: generate rotation matrices and hash all vectors.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let mut rng: Box<dyn RngCore> = match self.params.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Generate L random orthogonal rotation matrices via QR decomposition.
        self.rotations.clear();
        for _ in 0..self.params.num_tables {
            self.rotations
                .push(random_orthogonal_matrix(self.dimension, &mut *rng));
        }

        // Hash all vectors into tables.
        self.tables = vec![HashMap::new(); self.params.num_tables];
        for vec_idx in 0..self.num_vectors {
            let start = vec_idx * self.dimension;
            let vec = &self.vectors[start..start + self.dimension];
            for table_idx in 0..self.params.num_tables {
                let bucket = hash_with_rotation(vec, &self.rotations[table_idx], self.dimension);
                self.tables[table_idx]
                    .entry(bucket)
                    .or_default()
                    .push(vec_idx as u32);
            }
        }

        self.built = true;
        Ok(())
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

        for table_idx in 0..self.params.num_tables {
            let probes = self.multiprobe_buckets(query, table_idx);

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

    /// Hash a vector using the rotation matrix for `table_idx`.
    #[cfg(test)]
    fn hash_vector(&self, vector: &[f32], table_idx: usize) -> u32 {
        hash_with_rotation(vector, &self.rotations[table_idx], self.dimension)
    }

    /// Return the top `num_probes` bucket ids for multiprobe.
    ///
    /// Sorts the rotated coordinates by magnitude and returns the corresponding
    /// cross-polytope vertex ids (including both sign variants for coordinates
    /// close in magnitude to the maximum).
    fn multiprobe_buckets(&self, query: &[f32], table_idx: usize) -> Vec<u32> {
        let rotated = apply_rotation(query, &self.rotations[table_idx], self.dimension);

        // Collect (coordinate_index, absolute_value, sign) and sort descending by |value|.
        let mut ranked: Vec<(usize, f32, bool)> = rotated
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs(), v < 0.0))
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Primary probe: the vertex corresponding to the max coordinate.
        let mut buckets = Vec::with_capacity(self.params.num_probes);

        for &(coord_idx, _, is_negative) in ranked.iter().take(self.params.num_probes) {
            let bucket = if is_negative {
                (coord_idx as u32) * 2 + 1
            } else {
                (coord_idx as u32) * 2
            };
            buckets.push(bucket);
        }

        // Also probe the opposite sign of the top coordinate(s) when budget allows.
        let remaining = self.params.num_probes.saturating_sub(buckets.len());
        if remaining > 0 {
            for &(coord_idx, _, is_negative) in ranked.iter().take(remaining) {
                let opposite = if is_negative {
                    (coord_idx as u32) * 2
                } else {
                    (coord_idx as u32) * 2 + 1
                };
                if !buckets.contains(&opposite) {
                    buckets.push(opposite);
                }
            }
        }

        buckets
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

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a rotation matrix (row-major, d×d) to a vector.
fn apply_rotation(vector: &[f32], rotation: &[f32], d: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; d];
    for (i, out) in result.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let row_start = i * d;
        for j in 0..d {
            sum += rotation[row_start + j] * vector[j];
        }
        *out = sum;
    }
    result
}

/// Apply rotation and return the cross-polytope vertex (bucket id).
fn hash_with_rotation(vector: &[f32], rotation: &[f32], d: usize) -> u32 {
    let rotated = apply_rotation(vector, rotation, d);
    cross_polytope_vertex(&rotated)
}

/// Map a rotated vector to a cross-polytope vertex index.
///
/// Returns `2 * argmax_i(|v_i|) + sign_bit` where sign_bit is 1 if v[argmax] < 0.
fn cross_polytope_vertex(rotated: &[f32]) -> u32 {
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    let mut max_neg = false;

    for (i, &v) in rotated.iter().enumerate() {
        let abs_v = v.abs();
        if abs_v > max_abs {
            max_abs = abs_v;
            max_idx = i;
            max_neg = v < 0.0;
        }
    }

    (max_idx as u32) * 2 + u32::from(max_neg)
}

/// Generate a random orthogonal matrix via QR decomposition (Gram-Schmidt).
///
/// Draws a d x d matrix of i.i.d. Gaussian entries, then orthogonalizes.
/// Returns a flat row-major d x d matrix.
fn random_orthogonal_matrix(d: usize, rng: &mut dyn RngCore) -> Vec<f32> {
    // Generate random Gaussian matrix.
    let mut matrix = vec![0.0f32; d * d];
    for val in &mut matrix {
        *val = standard_normal(rng);
    }

    // Modified Gram-Schmidt orthogonalization.
    for i in 0..d {
        // Normalize column i.
        let mut norm = 0.0f32;
        for row in 0..d {
            norm += matrix[row * d + i] * matrix[row * d + i];
        }
        let norm = norm.sqrt();
        if norm > 1e-10 {
            for row in 0..d {
                matrix[row * d + i] /= norm;
            }
        }

        // Subtract projection of column i from all subsequent columns.
        for j in (i + 1)..d {
            let mut dot = 0.0f32;
            for row in 0..d {
                dot += matrix[row * d + i] * matrix[row * d + j];
            }
            for row in 0..d {
                matrix[row * d + j] -= dot * matrix[row * d + i];
            }
        }
    }

    // Convert from column-major orthogonalized to row-major for fast matvec.
    let mut row_major = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            row_major[i * d + j] = matrix[j * d + i];
        }
    }

    row_major
}

/// Box-Muller transform for standard normal samples.
fn standard_normal(rng: &mut dyn RngCore) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Helper: create clustered test data.
    fn clustered_data(n_clusters: usize, points_per_cluster: usize, dim: usize) -> Vec<f32> {
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
        let n = data.len() / dim;

        let params = LSHParams {
            num_tables: 16,
            num_probes: 8,
            seed: Some(42),
        };

        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();
        index.add_vectors(&data).unwrap();
        index.build().unwrap();

        let mut rng = StdRng::seed_from_u64(123);
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
        let n = data.len() / dim;
        let k = 10;

        let mut rng = StdRng::seed_from_u64(123);
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
        // [3, -1, 2] -> argmax_abs = 0 (value 3), positive -> bucket = 0
        assert_eq!(cross_polytope_vertex(&[3.0, -1.0, 2.0]), 0);
        // [1, -5, 2] -> argmax_abs = 1 (value -5), negative -> bucket = 3
        assert_eq!(cross_polytope_vertex(&[1.0, -5.0, 2.0]), 3);
        // [0, 0, 7] -> argmax_abs = 2 (value 7), positive -> bucket = 4
        assert_eq!(cross_polytope_vertex(&[0.0, 0.0, 7.0]), 4);
    }

    #[test]
    fn test_rotation_orthogonality() {
        let dim = 8;
        let mut rng = StdRng::seed_from_u64(42);
        let rot = random_orthogonal_matrix(dim, &mut rng);

        // Check that R * R^T ≈ I.
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += rot[i * dim + k] * rot[j * dim + k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "R*R^T[{},{}] = {} (expected {})",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
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
}
