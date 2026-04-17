//! Graph-based ANN for sparse vectors using maximum inner product search.
//!
//! Designed for sparse vectors such as SPLADE embeddings, BM25 term vectors,
//! and bag-of-words representations. Vectors are stored in sparse format
//! (sorted `(index, value)` pairs) and graph search is optimized for sparsity
//! using inner-product-based similarity.
//!
//! Distance is defined as the negated inner product: lower = more similar. This
//! lets standard min-distance graph search maximize inner product.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.6", features = ["sparse_mips"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::sparse_mips::{SparseMipsIndex, SparseMipsParams, SparseVector};
//!
//! let params = SparseMipsParams::default();
//! let mut index = SparseMipsIndex::new(params);
//!
//! for (id, vec) in data {
//!     index.add(id, vec)?;
//! }
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Construction
//!
//! 1. Find entry point: vector with highest L2 norm (likely most "central" in IP space)
//! 2. Build initial kNN graph (brute-force for n <= 1000, NN-descent for larger)
//! 3. RNG pruning pass with alpha relaxation
//! 4. Connectivity enforcement: DFS from entry, connect unreachable nodes
//!
//! # References
//!
//! - Malkov, Yashunin (2020). "Efficient and robust approximate nearest neighbor
//!   search using Hierarchical Navigable Small World graphs." TPAMI 42(4).
//! - Jayaram Subramanya et al. (2019). "DiskANN: Fast Accurate Billion-point
//!   Nearest Neighbor Search on a Single Node." NeurIPS.

use crate::distance::FloatOrd;
use crate::RetrieveError;
use smallvec::SmallVec;
use std::collections::{BinaryHeap, HashSet};

// ── Public types ───────────────────────────────────────────────────────────────

/// A sparse vector as sorted `(dimension_index, value)` pairs.
///
/// The `indices` array must be sorted in ascending order and have the same
/// length as `values`. Duplicate indices are not allowed.
#[derive(Clone, Debug, Default)]
pub struct SparseVector {
    /// Dimension indices, sorted ascending.
    pub indices: Vec<u32>,
    /// Corresponding values.
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Create a new sparse vector from unsorted (index, value) pairs.
    ///
    /// Sorts by index. Pairs with duplicate indices are silently deduplicated
    /// (last value wins after sorting).
    pub fn from_pairs(mut pairs: Vec<(u32, f32)>) -> Self {
        pairs.sort_unstable_by_key(|&(i, _)| i);
        pairs.dedup_by_key(|p| p.0);
        let indices = pairs.iter().map(|&(i, _)| i).collect();
        let values = pairs.iter().map(|&(_, v)| v).collect();
        Self { indices, values }
    }

    /// L2 norm of the sparse vector.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Number of nonzero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}

/// Construction and search parameters.
#[derive(Clone, Debug)]
pub struct SparseMipsParams {
    /// Maximum out-degree per node. Default: 32.
    pub max_degree: usize,
    /// Construction beam width. Default: 200.
    pub ef_construction: usize,
    /// Search beam width. Default: 100.
    pub ef_search: usize,
    /// RNG pruning relaxation factor (>= 1.0). Default: 1.2.
    ///
    /// A larger value is more permissive, allowing more neighbors to survive
    /// pruning and generally improving recall at the cost of higher degree.
    pub alpha: f32,
}

impl Default for SparseMipsParams {
    fn default() -> Self {
        Self {
            max_degree: 32,
            ef_construction: 200,
            ef_search: 100,
            alpha: 1.2,
        }
    }
}

/// Sparse MIPS graph index.
pub struct SparseMipsIndex {
    params: SparseMipsParams,
    built: bool,

    vectors: Vec<SparseVector>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    neighbors: Vec<SmallVec<[u32; 16]>>,
    entry_point: u32,
}

impl SparseMipsIndex {
    /// Create a new index. No dimension parameter is required because
    /// sparse vectors are dimension-agnostic.
    pub fn new(params: SparseMipsParams) -> Self {
        Self {
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            neighbors: Vec::new(),
            entry_point: 0,
        }
    }

    /// Add a sparse vector.
    ///
    /// Returns an error if the index has already been built.
    pub fn add(&mut self, doc_id: u32, vector: SparseVector) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add after build".into(),
            ));
        }
        self.vectors.push(vector);
        self.doc_ids.push(doc_id);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index.
    ///
    /// Must be called before `search`. After building, no more vectors can be added.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let n = self.num_vectors;

        // Step 1: Entry point -- vector with highest L2 norm.
        self.entry_point = self.find_entry_point();

        // Step 2: Initial kNN graph.
        if n <= 1000 {
            self.build_knn_bruteforce();
        } else {
            self.build_knn_nndescent();
        }

        // Step 3: RNG pruning refinement pass.
        for i in 0..n {
            let candidates = self.beam_search_internal(i, self.params.ef_construction);
            let selected = self.rng_prune(i, &candidates);

            let old = std::mem::replace(
                &mut self.neighbors[i],
                selected.iter().map(|&(id, _)| id).collect(),
            );
            drop(old);

            // Bidirectional edge insertion with capacity-aware pruning.
            let max_deg = self.params.max_degree;
            for &(nb_id, _) in &selected {
                let nid = nb_id as usize;
                if !self.neighbors[nid].contains(&(i as u32)) {
                    if self.neighbors[nid].len() < max_deg {
                        self.neighbors[nid].push(i as u32);
                    } else {
                        // Re-prune the reverse neighbor list to keep degree bounded.
                        let rev_cands: Vec<(u32, f32)> = self.neighbors[nid]
                            .iter()
                            .chain(std::iter::once(&(i as u32)))
                            .map(|&cid| {
                                (
                                    cid,
                                    sparse_distance(
                                        &self.vectors[nid],
                                        &self.vectors[cid as usize],
                                    ),
                                )
                            })
                            .collect();
                        let pruned = self.rng_prune(nid, &rev_cands);
                        self.neighbors[nid] = pruned.iter().map(|&(id, _)| id).collect();
                    }
                }
            }
        }

        // Step 4: Connectivity enforcement.
        self.ensure_connectivity();

        self.built = true;
        Ok(())
    }

    /// Search for the `k` nearest neighbors by inner product.
    ///
    /// Returns `(doc_id, distance)` pairs sorted ascending by distance
    /// (i.e., descending by inner product).
    pub fn search(&self, query: &SparseVector, k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let ef = self.params.ef_search.max(k);
        let results = self.beam_search_query(query, ef);

        Ok(results
            .into_iter()
            .take(k)
            .map(|(id, dist)| (self.doc_ids[id as usize], dist))
            .collect())
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    // ── Internal ───────────────────────────────────────────────────────────────

    /// Entry point: the vector with the highest L2 norm.
    fn find_entry_point(&self) -> u32 {
        let mut best = 0u32;
        let mut best_norm = self.vectors[0].norm();
        for i in 1..self.num_vectors {
            let n = self.vectors[i].norm();
            if n > best_norm {
                best_norm = n;
                best = i as u32;
            }
        }
        best
    }

    /// Brute-force initial kNN graph (used when n <= 1000).
    fn build_knn_bruteforce(&mut self) {
        let n = self.num_vectors;
        let k = (self.params.max_degree / 2).max(1).min(n - 1);
        self.neighbors = vec![SmallVec::new(); n];

        for i in 0..n {
            let mut dists: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    (
                        j as u32,
                        sparse_distance(&self.vectors[i], &self.vectors[j]),
                    )
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            dists.truncate(k);
            self.neighbors[i] = dists.iter().map(|&(id, _)| id).collect();
        }
    }

    /// NN-descent-style kNN graph construction for larger datasets.
    fn build_knn_nndescent(&mut self) {
        let (n, k) = (self.num_vectors, (self.params.max_degree / 2).max(1));
        let vecs = &self.vectors;
        self.neighbors = crate::graph_utils::build_knn_graph_nndescent(n, k, |i, j| {
            sparse_distance(&vecs[i], &vecs[j])
        });
    }

    /// RNG (Relative Neighborhood Graph) pruning with alpha relaxation.
    ///
    /// Keeps candidate `c` unless an already-selected neighbor `s` satisfies:
    /// `dist(query, s) <= alpha * dist(query, c)` AND `dist(s, c) <= alpha * dist(query, c)`.
    ///
    /// This is equivalent to Vamana's robust pruning rule.
    fn rng_prune(&self, query_idx: usize, candidates: &[(u32, f32)]) -> Vec<(u32, f32)> {
        let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
        sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        sorted.dedup_by_key(|c| c.0);
        // Remove self-loops.
        sorted.retain(|&(id, _)| id as usize != query_idx);

        let max_deg = self.params.max_degree;
        let alpha = self.params.alpha;
        let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_deg);

        for &(cand_id, cand_dist) in &sorted {
            if selected.len() >= max_deg {
                break;
            }

            let mut keep = true;
            for &(sel_id, _sel_dist) in &selected {
                // Robust pruning: drop cand if selected neighbor s is already
                // close enough to cand that s would serve as a proxy.
                // Condition: alpha * dist(s, cand) <= dist(query, cand)
                let sc_dist = sparse_distance(
                    &self.vectors[sel_id as usize],
                    &self.vectors[cand_id as usize],
                );
                if alpha * sc_dist <= cand_dist {
                    keep = false;
                    break;
                }
            }

            if keep {
                selected.push((cand_id, cand_dist));
            }
        }

        selected
    }

    /// Beam search from the entry point to find candidates near node `query_idx`.
    /// Used during build; treats `vectors[query_idx]` as the query.
    fn beam_search_internal(&self, query_idx: usize, ef: usize) -> Vec<(u32, f32)> {
        self.beam_search_query(&self.vectors[query_idx].clone(), ef)
    }

    /// Beam search from the entry point toward `query`.
    fn beam_search_query(&self, query: &SparseVector, ef: usize) -> Vec<(u32, f32)> {
        let n = self.num_vectors;
        if n == 0 {
            return Vec::new();
        }

        let mut visited: HashSet<u32> = HashSet::new();
        // Min-heap by distance.
        let mut frontier: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        let entry = self.entry_point;
        let entry_dist = sparse_distance(query, &self.vectors[entry as usize]);
        visited.insert(entry);
        frontier.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
        candidates.push((entry, entry_dist));

        while let Some(std::cmp::Reverse((FloatOrd(current_dist), current_id))) = frontier.pop() {
            // Early termination: if the frontier's best is worse than the ef-th
            // result by a margin, further expansion cannot improve the top-ef.
            if candidates.len() >= ef {
                candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                if current_dist > candidates[ef - 1].1 * 1.5 {
                    break;
                }
            }

            for &neighbor in &self.neighbors[current_id as usize] {
                if visited.insert(neighbor) {
                    let dist = sparse_distance(query, &self.vectors[neighbor as usize]);
                    candidates.push((neighbor, dist));
                    frontier.push(std::cmp::Reverse((FloatOrd(dist), neighbor)));
                }
            }

            // Cap visited set to prevent runaway expansion.
            if visited.len() > ef * 10 {
                break;
            }
        }

        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        candidates.dedup_by_key(|c| c.0);
        candidates
    }

    fn ensure_connectivity(&mut self) {
        let vecs = &self.vectors;
        crate::graph_utils::ensure_connectivity(&mut self.neighbors, self.entry_point, |i, j| {
            sparse_distance(&vecs[i], &vecs[j])
        });
    }
}

// ── Distance ──────────────────────────────────────────────────────────────────

/// Sparse inner product distance: `-dot_product(a, b)`.
///
/// Lower values indicate higher similarity (more positive inner product).
/// Uses a merge-join on sorted index arrays: O(|a| + |b|).
pub fn sparse_distance(a: &SparseVector, b: &SparseVector) -> f32 {
    let mut dot = 0.0f32;
    let mut ai = 0usize;
    let mut bi = 0usize;
    let a_idx = &a.indices;
    let b_idx = &b.indices;
    let a_val = &a.values;
    let b_val = &b.values;

    while ai < a_idx.len() && bi < b_idx.len() {
        match a_idx[ai].cmp(&b_idx[bi]) {
            std::cmp::Ordering::Equal => {
                dot += a_val[ai] * b_val[bi];
                ai += 1;
                bi += 1;
            }
            std::cmp::Ordering::Less => {
                ai += 1;
            }
            std::cmp::Ordering::Greater => {
                bi += 1;
            }
        }
    }

    -dot
}

// ── FloatOrd ──────────────────────────────────────────────────────────────────

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Generate `n` random sparse vectors with `nnz` nonzero entries each,
    /// drawn from dimensions `0..max_dim` with values in `(0, 1)`.
    fn make_sparse(n: usize, nnz: usize, max_dim: u32, seed: u64) -> Vec<SparseVector> {
        let mut rng = seed;
        let lcg = |state: &mut u64| -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        };

        (0..n)
            .map(|_| {
                // Pick `nnz` distinct random indices.
                let mut indices: Vec<u32> = Vec::with_capacity(nnz);
                let mut attempts = 0usize;
                while indices.len() < nnz && attempts < nnz * 16 {
                    attempts += 1;
                    let idx = (lcg(&mut rng) >> 33) as u32 % max_dim;
                    if !indices.contains(&idx) {
                        indices.push(idx);
                    }
                }
                indices.sort_unstable();

                let values: Vec<f32> = indices
                    .iter()
                    .map(|_| {
                        let r = lcg(&mut rng);
                        (r >> 33) as f32 / (1u64 << 31) as f32
                    })
                    .collect();

                SparseVector { indices, values }
            })
            .collect()
    }

    #[test]
    fn build_and_search() {
        let n = 50;
        let vecs = make_sparse(n, 10, 100, 42);

        let mut index = SparseMipsIndex::new(SparseMipsParams {
            max_degree: 16,
            ef_construction: 64,
            ef_search: 32,
            alpha: 1.2,
        });

        for (i, v) in vecs.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        assert_eq!(index.len(), n);

        let results = index.search(&vecs[0], 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // The query vector itself should be in the results.
        assert!(results.iter().any(|&(id, _)| id == 0));
    }

    #[test]
    fn self_search_recall() {
        let n = 50;
        let vecs = make_sparse(n, 15, 200, 7);

        let mut index = SparseMipsIndex::new(SparseMipsParams {
            max_degree: 16,
            ef_construction: 100,
            ef_search: 50,
            alpha: 1.2,
        });

        for (i, v) in vecs.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        let mut hits = 0usize;
        for (i, v) in vecs.iter().enumerate() {
            let results = index.search(v, 1).unwrap();
            if results.first().map(|&(id, _)| id) == Some(i as u32) {
                hits += 1;
            }
        }
        let recall = hits as f64 / n as f64;
        assert!(
            recall > 0.5,
            "self-search recall too low: {recall:.2} ({hits}/{n})"
        );
    }

    #[test]
    fn empty_sparse_vectors() {
        // Vectors with no nonzero entries should not crash; all inner products are 0.
        let mut index = SparseMipsIndex::new(SparseMipsParams::default());
        for i in 0..5u32 {
            index
                .add(
                    i,
                    SparseVector {
                        indices: vec![],
                        values: vec![],
                    },
                )
                .unwrap();
        }
        index.build().unwrap();

        let query = SparseVector {
            indices: vec![],
            values: vec![],
        };
        let results = index.search(&query, 3).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn disjoint_vectors() {
        // Two groups with non-overlapping dimensions: inner product between groups = 0.
        let mut index = SparseMipsIndex::new(SparseMipsParams {
            max_degree: 8,
            ef_construction: 32,
            ef_search: 16,
            alpha: 1.2,
        });

        // Group A: dimensions 0..10
        for i in 0..5u32 {
            let sv = SparseVector {
                indices: (0u32..10).collect(),
                values: vec![1.0; 10],
            };
            index.add(i, sv).unwrap();
        }
        // Group B: dimensions 100..110
        for i in 5..10u32 {
            let sv = SparseVector {
                indices: (100u32..110).collect(),
                values: vec![1.0; 10],
            };
            index.add(i, sv).unwrap();
        }
        index.build().unwrap();

        // Query from group A should preferentially return group A ids.
        let query_a = SparseVector {
            indices: (0u32..10).collect(),
            values: vec![1.0; 10],
        };
        let results = index.search(&query_a, 3).unwrap();
        assert!(!results.is_empty());
        // All returned ids should be in group A (0..5), since inter-group IP = 0
        // and intra-group IP = 10 (distance = -10).
        let all_group_a = results.iter().all(|&(id, _)| id < 5);
        assert!(all_group_a, "expected group A results, got: {results:?}");
    }

    #[test]
    fn empty_index_errors() {
        let mut index = SparseMipsIndex::new(SparseMipsParams::default());
        assert!(index.build().is_err());
        assert!(index.search(&SparseVector::default(), 1).is_err());
    }

    #[test]
    fn add_after_build_errors() {
        let mut index = SparseMipsIndex::new(SparseMipsParams::default());
        index
            .add(
                0,
                SparseVector {
                    indices: vec![0],
                    values: vec![1.0],
                },
            )
            .unwrap();
        index.build().unwrap();
        let err = index.add(1, SparseVector::default());
        assert!(err.is_err());
    }

    #[test]
    fn sparse_distance_correctness() {
        let a = SparseVector {
            indices: vec![0, 2, 4],
            values: vec![1.0, 2.0, 3.0],
        };
        let b = SparseVector {
            indices: vec![1, 2, 3],
            values: vec![5.0, 6.0, 7.0],
        };
        // Only dimension 2 overlaps: 2.0 * 6.0 = 12.0; distance = -12.0
        let d = sparse_distance(&a, &b);
        assert!((d - (-12.0f32)).abs() < 1e-5, "expected -12.0, got {d}");

        // Disjoint vectors: dot = 0, distance = 0
        let c = SparseVector {
            indices: vec![10, 11],
            values: vec![3.0, 4.0],
        };
        let d2 = sparse_distance(&a, &c);
        assert!((d2).abs() < 1e-5, "expected 0.0, got {d2}");
    }

    #[test]
    fn from_pairs_sorts_and_deduplicates() {
        let sv = SparseVector::from_pairs(vec![(3, 1.0), (1, 2.0), (3, 5.0), (0, 0.5)]);
        assert_eq!(sv.indices, vec![0, 1, 3]);
        // dedup_by_key keeps the first occurrence for duplicate indices
        assert_eq!(sv.values, vec![0.5, 2.0, 1.0]);
    }
}
