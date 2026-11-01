//! Random Projection Tree implementation.
//!
//! Single random projection tree (base structure for Random Projection Tree Forest).
//! Uses random hyperplanes to partition space.
//!
//! **Technical Name**: Random Projection Tree
//!
//! Algorithm:
//! - Single tree using random hyperplane splits
//! - Each node splits space with a random hyperplane
//! - Simpler than Random Projection Tree Forest (Annoy)
//! - Good baseline method
//!
//! **Relationships**:
//! - Base structure for Random Projection Tree Forest (Annoy uses multiple RP-Trees)
//! - Similar to KD-Tree but uses random hyperplanes instead of dimension-aligned splits
//! - Complementary to other tree methods
//!
//! # References
//!
//! - Dasgupta & Freund (2008): "Random projection trees and low dimensional manifolds"

use crate::classic::trees::persistence::{read_json, validate_vector_shape, write_json_atomic};
use crate::simd;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::path::Path;

const RPTREE_FORMAT_VERSION: u32 = 1;
const DEFAULT_RPTREE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Tolerance on `|‖v‖² - 1|` for treating a vector as unit norm.
///
/// The k-th-distance bound relies on `1 - q·p = ‖q - p‖² / 2`, which only
/// holds for unit vectors. Pruning is gated on that precondition, and the
/// bound is loosened by this tolerance so it stays a lower bound for data
/// that is unit norm only to within floating-point error.
const UNIT_NORM_TOLERANCE: f32 = 1e-5;

/// Candidate ordered so `BinaryHeap` keeps the worst distance on top.
///
/// "Worst" breaks exact distance ties by the *larger* `doc_id`, because the
/// result order prefers the smaller one: the heap must evict the larger id so
/// the k-th-boundary survivor matches brute force.
#[derive(Clone, Copy, PartialEq)]
struct Candidate(f32, u32);

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.cmp(&other.1))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Random Projection Tree index.
///
/// Single tree using random hyperplanes for space partitioning.
#[derive(Deserialize, Serialize)]
pub struct RPTreeIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    doc_ids: Vec<u32>,
    params: RPTreeParams,
    built: bool,
    root: Option<RPNode>,
    /// Every stored vector is unit norm within [`UNIT_NORM_TOLERANCE`].
    ///
    /// Set by [`RPTreeIndex::build`] and recomputed on load. When `false` the
    /// k-th-distance bound is unavailable and search traverses the whole tree.
    #[serde(default)]
    unit_norm_vectors: bool,
}

/// Random Projection Tree parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RPTreeParams {
    /// Maximum leaf size
    pub max_leaf_size: usize,

    /// Maximum depth
    pub max_depth: usize,
}

impl Default for RPTreeParams {
    fn default() -> Self {
        Self {
            max_leaf_size: 10,
            max_depth: 32,
        }
    }
}

/// Random Projection Tree node.
#[derive(Clone, Deserialize, Serialize)]
enum RPNode {
    /// Internal node: splits with random hyperplane
    Internal {
        hyperplane: Vec<f32>,
        threshold: f32,
        left: Box<RPNode>,
        right: Box<RPNode>,
    },
    /// Leaf node: contains vector indices
    Leaf { indices: Vec<u32> },
}

#[derive(Deserialize, Serialize)]
struct RPTreeSnapshot {
    version: u32,
    index: RPTreeIndex,
}

impl RPTreeIndex {
    /// Create new Random Projection Tree index.
    pub fn new(dimension: usize, params: RPTreeParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidConfig {
                field: "dimension",
                expected: "must be greater than 0",
            });
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            doc_ids: Vec::new(),
            params,
            built: false,
            root: None,
            unit_norm_vectors: false,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u32, embedding: Vec<f32>) -> Result<(), RetrieveError> {
        if embedding.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: embedding.len(),
                doc_dim: self.dimension,
            });
        }

        if self.built {
            return Err(RetrieveError::AlreadyBuilt {
                component: "random-projection tree index",
                operation: "add vectors",
            });
        }

        self.vectors.extend_from_slice(&embedding);
        self.doc_ids.push(doc_id);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the Random Projection Tree.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        self.unit_norm_vectors = self.all_unit_norm();
        let indices: Vec<u32> = (0..self.num_vectors as u32).collect();
        self.root = Some(self.build_tree(&indices, 0, DEFAULT_RPTREE_SEED)?);

        self.built = true;
        Ok(())
    }

    /// Save a built random-projection tree index to a directory.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::NotBuilt {
                component: "random-projection tree index",
                operation: "save",
            });
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        write_json_atomic(
            &output_dir.join("index.json"),
            &RPTreeSnapshot {
                version: RPTREE_FORMAT_VERSION,
                index: self.clone_for_snapshot(),
            },
        )
    }

    /// Load a random-projection tree index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let snapshot: RPTreeSnapshot = read_json(&input_dir.as_ref().join("index.json"))?;
        if snapshot.version != RPTREE_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported random-projection tree format version {}",
                snapshot.version
            )));
        }
        let mut index = snapshot.index;
        validate_vector_shape(
            "random-projection tree",
            index.dimension,
            index.num_vectors,
            &index.vectors,
            &index.doc_ids,
        )?;
        if !index.built || index.root.is_none() {
            return Err(RetrieveError::FormatError(
                "random-projection tree snapshot is not built".into(),
            ));
        }
        // Recompute rather than trust the snapshot: the flag decides whether
        // the bound may be used, so it must describe the vectors actually loaded.
        index.unit_norm_vectors = index.all_unit_norm();
        Ok(index)
    }

    fn all_unit_norm(&self) -> bool {
        (0..self.num_vectors).all(|idx| is_unit_norm(self.get_vector(idx)))
    }

    fn clone_for_snapshot(&self) -> Self {
        Self {
            vectors: self.vectors.clone(),
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            doc_ids: self.doc_ids.clone(),
            params: self.params.clone(),
            built: self.built,
            root: self.root.clone(),
            unit_norm_vectors: self.unit_norm_vectors,
        }
    }

    /// Estimated heap memory used by this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        crate::memory::MemoryReport {
            vectors_bytes: self.vectors.capacity() * std::mem::size_of::<f32>(),
            graph_bytes: self.root.as_ref().map(RPNode::owned_bytes).unwrap_or(0),
            quantized_bytes: 0,
            metadata_bytes: self.doc_ids.capacity() * std::mem::size_of::<u32>(),
        }
    }

    /// Build tree recursively.
    fn build_tree(
        &self,
        indices: &[u32],
        depth: usize,
        seed: u64,
    ) -> Result<RPNode, RetrieveError> {
        if indices.is_empty() {
            return Ok(RPNode::Leaf {
                indices: Vec::new(),
            });
        }

        // Leaf node if small enough or max depth reached
        if indices.len() <= self.params.max_leaf_size || depth >= self.params.max_depth {
            return Ok(RPNode::Leaf {
                indices: indices.to_vec(),
            });
        }

        // Generate random hyperplane
        let hyperplane = self.generate_random_hyperplane(seed);

        // Compute projections and find median
        let mut projections: Vec<(f32, u32)> = indices
            .iter()
            .map(|&idx| {
                let vec = self.get_vector(idx as usize);
                let projection = simd::dot(vec, &hyperplane);
                (projection, idx)
            })
            .collect();

        projections.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let median_idx = projections.len() / 2;
        let threshold = projections[median_idx].0;

        // Split by threshold
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for (proj, idx) in projections {
            if proj < threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Build children
        let left = self.build_tree(&left_indices, depth + 1, child_seed(seed, 0))?;
        let right = self.build_tree(&right_indices, depth + 1, child_seed(seed, 1))?;

        Ok(RPNode::Internal {
            hyperplane,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Generate random hyperplane.
    fn generate_random_hyperplane(&self, seed: u64) -> Vec<f32> {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut hyperplane = Vec::with_capacity(self.dimension);
        let mut norm = 0.0;

        for _ in 0..self.dimension {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            norm += val * val;
            hyperplane.push(val);
        }

        // Normalize
        let norm = norm.sqrt();
        if norm > 0.0 {
            for val in hyperplane.iter_mut() {
                *val /= norm;
            }
        }

        hyperplane
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::NotBuilt {
                component: "random-projection tree index",
                operation: "search",
            });
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let root = self.root.as_ref().ok_or(RetrieveError::NotBuilt {
            component: "random-projection tree index",
            operation: "search",
        })?;

        // The bound assumes both the data and the query are unit vectors. It
        // is not a lower bound of `1 - q·p` otherwise (e.g. a long vector in
        // the query direction scores below 0), so any other input takes the
        // exhaustive walk.
        let prune = self.unit_norm_vectors && is_unit_norm(query);
        let mut heap = BinaryHeap::with_capacity(k.max(1));
        self.search_pruned(root, query, k, &mut heap, prune)?;

        let mut results: Vec<(u32, f32)> = heap
            .into_iter()
            .map(|candidate| (candidate.1, candidate.0))
            .collect();
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Ok(results)
    }

    /// Search with k-th-distance pruning.
    ///
    /// A random-projection split is a hyperplane with unit normal `w`, so for
    /// **unit** vectors (`cos_dist = ‖q-p‖² / 2`) any point on the far side
    /// satisfies `cos_dist(q, p) >= margin² / 2`, where `margin` is the query's
    /// distance from the split plane. Once `k` candidates are held, a far
    /// subtree whose bound exceeds the current worst is skipped.
    ///
    /// `prune` is the precondition check: it is only `true` when every stored
    /// vector and the query are unit norm, in which case the traversal is exact
    /// and only the visit count changes. With `prune == false` both children are
    /// always visited, which is exact for the raw `1 - q·p` scoring used here.
    fn search_pruned(
        &self,
        node: &RPNode,
        query: &[f32],
        k: usize,
        heap: &mut BinaryHeap<Candidate>,
        prune: bool,
    ) -> Result<(), RetrieveError> {
        match node {
            RPNode::Leaf { indices } => {
                for &idx in indices {
                    let distance = crate::distance::cosine_distance_normalized(
                        query,
                        self.get_vector(idx as usize),
                    );
                    let candidate = Candidate(distance, self.doc_ids[idx as usize]);
                    if heap.len() < k {
                        heap.push(candidate);
                    } else if let Some(worst) = heap.peek() {
                        if candidate < *worst {
                            heap.pop();
                            heap.push(candidate);
                        }
                    }
                }
            }
            RPNode::Internal {
                hyperplane,
                threshold,
                left,
                right,
            } => {
                let projection = simd::dot(query, hyperplane);
                let (near, far, margin) = if projection < *threshold {
                    (left, right, *threshold - projection)
                } else {
                    (right, left, projection - *threshold)
                };
                self.search_pruned(near, query, k, heap, prune)?;
                let visit_far = if prune {
                    // `1 - q·p = ‖q-p‖²/2 - (‖q‖²-1 + ‖p‖²-1)/2`, so the
                    // tolerance tightens the bound into a true lower bound.
                    let bound = margin * margin * 0.5 - UNIT_NORM_TOLERANCE;
                    heap.len() < k || heap.peek().is_some_and(|worst| bound <= worst.0)
                } else {
                    true
                };
                if visit_far {
                    self.search_pruned(far, query, k, heap, prune)?;
                }
            }
        }
        Ok(())
    }

    /// Get vector from SoA storage.
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

impl RPNode {
    fn owned_bytes(&self) -> usize {
        match self {
            RPNode::Internal {
                hyperplane,
                left,
                right,
                ..
            } => {
                hyperplane.capacity() * std::mem::size_of::<f32>()
                    + boxed_node_bytes(left)
                    + boxed_node_bytes(right)
            }
            RPNode::Leaf { indices } => indices.capacity() * std::mem::size_of::<u32>(),
        }
    }
}

fn boxed_node_bytes(node: &RPNode) -> usize {
    std::mem::size_of::<RPNode>() + node.owned_bytes()
}

/// `true` when `vector` is unit norm within [`UNIT_NORM_TOLERANCE`].
fn is_unit_norm(vector: &[f32]) -> bool {
    (simd::dot(vector, vector) - 1.0).abs() <= UNIT_NORM_TOLERANCE
}

fn child_seed(seed: u64, branch: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407 ^ branch)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn build_index() -> RPTreeIndex {
        let mut index = RPTreeIndex::new(4, RPTreeParams::default()).unwrap();
        for i in 0..32u32 {
            let mut v = vec![0.0f32; 4];
            v[(i as usize) % 4] = 1.0;
            index.add(4000 + i, v).unwrap();
        }
        index.build().unwrap();
        index
    }

    #[test]
    fn build_is_deterministic() {
        let first = build_index();
        let second = build_index();
        assert_eq!(
            serde_json::to_value(&first.root).unwrap(),
            serde_json::to_value(&second.root).unwrap()
        );
    }

    #[test]
    fn search_returns_external_doc_ids() {
        let index = build_index();
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 6).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|(id, _)| *id >= 4000));
    }

    #[test]
    fn save_load_roundtrip_preserves_sampled_tree() {
        let index = build_index();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = RPTreeIndex::load_from_dir(dir.path()).unwrap();
        let query = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            index.search(&query, 8).unwrap(),
            loaded.search(&query, 8).unwrap()
        );
    }

    /// Pruning must not change results: the k-th-distance bound is exact for
    /// unit vectors, so the pruned walk has to match brute force bit for bit,
    /// including the (distance, doc_id) tie order.
    #[test]
    fn pruned_search_matches_brute_force() {
        let dim = 8;
        let n = 500u32;
        let mut seed = 987_654_321u64;
        let mut vectors = Vec::with_capacity(n as usize * dim);
        for _ in 0..n as usize * dim {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            vectors.push(((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0);
        }
        // The tree's metric assumes unit vectors (cos_dist = ||q-p||^2 / 2),
        // which is also the precondition for the pruning bound.
        for chunk in vectors.chunks_mut(dim) {
            let norm = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
            for value in chunk.iter_mut() {
                *value /= norm;
            }
        }
        let mut index = RPTreeIndex::new(dim, RPTreeParams::default()).unwrap();
        for id in 0..n {
            let start = id as usize * dim;
            index.add(id, vectors[start..start + dim].to_vec()).unwrap();
        }
        index.build().unwrap();
        assert!(
            index.unit_norm_vectors,
            "normalized data must keep k-th-distance pruning enabled"
        );

        for q in 0..20u32 {
            let start = (q as usize * 37) % (n as usize - dim);
            let query: Vec<f32> = vectors[start..start + dim]
                .iter()
                .map(|v| v * 0.5)
                .collect();
            let query: Vec<f32> = {
                let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
                query.iter().map(|v| v / norm).collect()
            };
            let mut brute: Vec<(u32, f32)> = (0..n)
                .map(|id| {
                    let s = id as usize * dim;
                    (
                        id,
                        crate::distance::cosine_distance_normalized(&query, &vectors[s..s + dim]),
                    )
                })
                .collect();
            brute.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            brute.truncate(10);
            assert_eq!(index.search(&query, 10).unwrap(), brute, "query {q}");
        }
    }

    /// Regression for the shrinker's counterexample: the k-th-distance bound is
    /// only sound for unit vectors, so it must not be applied to raw input.
    ///
    /// For this input the query norm is 1.128 and stored norms span
    /// 0.676..1.569. The far subtree holding doc `5005` was skipped because its
    /// `margin²/2 = 0.098` exceeded the current worst -0.272, although `5005`
    /// scores -0.604 (its `‖q-p‖²/2` is 0.263, so the bound was never a bound
    /// on `1 - q·p`). `search` returned `5010` at -0.272 instead.
    #[test]
    fn non_unit_input_does_not_prune_away_the_neighbour() {
        let (dim, n, seed, k) = (3usize, 22usize, 2_060_225_295_894_311_425u64, 1usize);
        let mut state = seed | 1;
        let data: Vec<Vec<f32>> = (0..n).map(|_| random_vector(dim, &mut state)).collect();
        let doc_ids: Vec<u32> = (0..n).map(|i| 5_000 + (n - 1 - i) as u32).collect();
        let mut index = RPTreeIndex::new(
            dim,
            RPTreeParams {
                max_leaf_size: 1,
                max_depth: 2,
            },
        )
        .unwrap();
        for (vector, &doc_id) in data.iter().zip(&doc_ids) {
            index.add(doc_id, vector.clone()).unwrap();
        }
        index.build().unwrap();
        assert!(
            !index.unit_norm_vectors,
            "raw vectors must report that the bound is unavailable"
        );

        let query = data[(seed as usize) % n].clone();
        let expected = vec![(5_005u32, -0.604_336_6f32)];
        assert_eq!(brute_force_top_k(&data, &doc_ids, &query, k), expected);
        assert_eq!(index.search(&query, k).unwrap(), expected);
    }

    // ── Property: pruning must never change the exact top-k ──────────────────

    /// Brute-force top-k scored exactly like `search`: cosine distance over the
    /// unnormalized query, ties broken by ascending doc_id.
    fn brute_force_top_k(
        data: &[Vec<f32>],
        doc_ids: &[u32],
        query: &[f32],
        k: usize,
    ) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = data
            .iter()
            .zip(doc_ids)
            .map(|(vector, &doc_id)| {
                (
                    doc_id,
                    crate::distance::cosine_distance_normalized(query, vector),
                )
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    /// Deterministic pseudo-random vector with components in `[-1, 1)`.
    fn random_vector(dim: usize, state: &mut u64) -> Vec<f32> {
        (0..dim)
            .map(|_| {
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Whatever bound the traversal prunes with must be exact:
        /// `search(q, k)` equals brute force over the same scoring for every
        /// `k` in 1..=20, including the `(distance, doc_id)` tie order. A bound
        /// that drops a candidate whose distance is <= the current k-th
        /// distance fails here, and so does a split that misroutes data points.
        #[test]
        fn search_equals_brute_force_for_every_k(
            dim in 1usize..=8,
            n in 1usize..=60,
            max_leaf_size in 1usize..=8,
            max_depth in 1usize..=10,
            seed in any::<u64>(),
            duplicate_last in any::<bool>(),
            query_from_data in any::<bool>(),
            k in 1usize..=20,
        ) {
            let mut state = seed | 1;
            let mut data: Vec<Vec<f32>> =
                (0..n).map(|_| random_vector(dim, &mut state)).collect();
            if duplicate_last && n > 1 {
                // Exact distance ties: tie order must fall back to doc_id.
                data[n - 1] = data[0].clone();
            }
            // Reversed doc_ids: tie order cannot fall out of insertion order.
            let doc_ids: Vec<u32> = (0..n).map(|i| 5_000 + (n - 1 - i) as u32).collect();

            let mut index =
                RPTreeIndex::new(dim, RPTreeParams { max_leaf_size, max_depth }).unwrap();
            for (vector, &doc_id) in data.iter().zip(&doc_ids) {
                index.add(doc_id, vector.clone()).unwrap();
            }
            index.build().unwrap();

            let query = if query_from_data {
                // A stored point: distance 0, so it must come first.
                data[(seed as usize) % n].clone()
            } else {
                let mut query = random_vector(dim, &mut state);
                query[0] += 1.0; // never the all-zero query
                query
            };

            let expected = brute_force_top_k(&data, &doc_ids, &query, k);
            let actual = index.search(&query, k).unwrap();
            prop_assert_eq!(
                &actual,
                &expected,
                "search(q, {}) disagreed with brute force (n={}, dim={}, \
                 max_leaf_size={}, max_depth={})",
                k,
                n,
                dim,
                max_leaf_size,
                max_depth
            );
        }
    }
}
