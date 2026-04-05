//! Random Projection Tree Forest implementation.
//!
//! Pure Rust implementation of a Random Projection Tree Forest.
//!
//! Note: some ecosystems refer to this family as “Annoy-style”, but this crate uses
//! method-family names (`rptree`) rather than vendor/library names.
//!
//! Algorithm:
//! - Forest of independent random projection trees
//! - Random hyperplane splits at each node
//! - Multiple trees improve recall through ensemble search
//! - Memory-mapped index support (optional)
//! - Thread-safe search
//!
//! **Relationships**:
//! - Uses Random Projection Trees (RP-Trees) as base structure
//! - Forest approach (multiple trees) improves recall vs single tree
//! - Complementary to graph-based methods (HNSW, SNG)
//! - Tree-based space partitioning (different from graph-based proximity)
//!
//! **vs hash-bucket LSH**: if you need near-duplicate detection or set similarity
//! (MinHash/SimHash) rather than ranked k-NN retrieval, see the [`sketchir`](https://docs.rs/sketchir)
//! crate, which provides banding LSH and random-projection LSH over hash buckets.
//!
//! # References
//!
//! - Dasgupta & Freund (2008): "Random projection trees and low dimensional manifolds"
//! - Spotify Engineering Blog: "Annoy: Approximate Nearest Neighbors in C++/Python" (historical name)

use crate::simd;
use crate::RetrieveError;

/// Random Projection Tree Forest index.
///
/// Uses a forest of independent random projection trees for approximate
/// nearest neighbor search. Each tree partitions space using random hyperplanes.
///
pub struct RpForestIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: RpForestParams,
    built: bool,

    /// Forest of random projection trees
    pub(crate) trees: Vec<RPTree>,
}

/// Random Projection Tree Forest parameters.
#[derive(Clone, Debug)]
pub struct RpForestParams {
    /// Number of trees in forest
    pub num_trees: usize,

    /// Tree construction parameters
    pub tree_params: RPTreeParams,
}

impl Default for RpForestParams {
    fn default() -> Self {
        Self {
            num_trees: 10,
            tree_params: RPTreeParams::default(),
        }
    }
}

/// Random projection tree.
pub(crate) struct RPTree {
    root: Option<TreeNode>,
}

/// Tree node.
enum TreeNode {
    Leaf {
        indices: Vec<u32>,
    },
    Internal {
        hyperplane: Vec<f32>, // Random hyperplane
        #[allow(dead_code)]
        threshold: f32, // Reserved for non-zero split thresholds
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Random projection tree parameters.
#[derive(Clone, Debug)]
pub struct RPTreeParams {
    /// Maximum points per leaf
    pub max_leaf_size: usize,
}

impl Default for RPTreeParams {
    fn default() -> Self {
        Self { max_leaf_size: 10 }
    }
}

impl RpForestIndex {
    /// Create a new RP-forest index.
    pub fn new(dimension: usize, params: RpForestParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            params,
            built: false,
            trees: Vec::new(),
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "Cannot add vectors after index is built".to_string(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        self.vectors.extend_from_slice(&vector);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index with random projection tree forest.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Build forest of trees
        self.trees = Vec::new();
        for _ in 0..self.params.num_trees {
            let tree = self.build_tree()?;
            self.trees.push(tree);
        }

        self.built = true;
        Ok(())
    }

    /// Build a single random projection tree.
    fn build_tree(&self) -> Result<RPTree, RetrieveError> {
        let indices: Vec<u32> = (0..self.num_vectors as u32).collect();
        let root = self.build_tree_recursive(&indices)?;
        Ok(RPTree { root })
    }

    /// Build tree recursively. Each call independently samples a fresh random hyperplane.
    fn build_tree_recursive(&self, indices: &[u32]) -> Result<Option<TreeNode>, RetrieveError> {
        if indices.is_empty() {
            return Ok(None);
        }

        // Leaf node if small enough
        if indices.len() <= self.params.tree_params.max_leaf_size {
            return Ok(Some(TreeNode::Leaf {
                indices: indices.to_vec(),
            }));
        }

        // Generate a fresh random hyperplane for this split
        use rand::Rng;
        let mut rng = rand::rng();
        let mut hyperplane = Vec::with_capacity(self.dimension);
        let mut norm = 0.0f32;
        for _ in 0..self.dimension {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            norm += val * val;
            hyperplane.push(val);
        }
        let norm = norm.sqrt();
        if norm > 0.0 {
            for val in &mut hyperplane {
                *val /= norm;
            }
        }

        // Split by hyperplane
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            let vec = self.get_vector(idx as usize);
            let projection = simd::dot(vec, &hyperplane);
            if projection < 0.0 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Degenerate split: all points went to one side (e.g., duplicate or collinear vectors).
        // Fall back to a leaf rather than recursing infinitely.
        if left_indices.is_empty() || right_indices.is_empty() {
            return Ok(Some(TreeNode::Leaf {
                indices: indices.to_vec(),
            }));
        }

        // Each child independently picks its own split hyperplane
        let left = self.build_tree_recursive(&left_indices)?;
        let right = self.build_tree_recursive(&right_indices)?;

        Ok(Some(TreeNode::Internal {
            hyperplane,
            threshold: 0.0,
            left: Box::new(left.unwrap_or(TreeNode::Leaf {
                indices: Vec::new(),
            })),
            right: Box::new(right.unwrap_or(TreeNode::Leaf {
                indices: Vec::new(),
            })),
        }))
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "Index must be built before search".to_string(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        // Search all trees and collect candidates
        let mut candidate_set = std::collections::HashSet::new();

        for tree in &self.trees {
            if let Some(ref root) = tree.root {
                let candidates = self.search_tree(root, query);
                for idx in candidates {
                    candidate_set.insert(idx);
                }
            }
        }

        // Compute exact distances for candidates
        let mut results: Vec<(u32, f32)> = candidate_set
            .iter()
            .map(|&idx| {
                let vec = self.get_vector(idx as usize);
                let dist = 1.0 - simd::dot(query, vec);
                (idx, dist)
            })
            .collect();

        // Sort and return top k
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1)); // Unstable for better performance
        Ok(results.into_iter().take(k).collect())
    }

    /// Search a single tree.
    fn search_tree(&self, node: &TreeNode, query: &[f32]) -> Vec<u32> {
        match node {
            TreeNode::Leaf { indices } => indices.clone(),
            TreeNode::Internal {
                hyperplane,
                threshold: _,
                left,
                right,
            } => {
                let projection = simd::dot(query, hyperplane);
                if projection < 0.0 {
                    self.search_tree(left, query)
                } else {
                    self.search_tree(right, query)
                }
            }
        }
    }

    /// Get vector from SoA storage.
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index(n: usize, dim: usize) -> RpForestIndex {
        let params = RpForestParams {
            num_trees: 5,
            tree_params: RPTreeParams { max_leaf_size: 10 },
        };
        let mut index = RpForestIndex::new(dim, params).unwrap();
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();
        index
    }

    #[test]
    fn test_basic_search_returns_results() {
        let index = build_index(50, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_returns_at_most_k() {
        let index = build_index(50, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        for k in [1, 3, 5, 10] {
            let results = index.search(&query, k).unwrap();
            assert!(results.len() <= k);
        }
    }

    #[test]
    fn test_results_sorted_by_distance() {
        let index = build_index(50, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted: {:?}", results);
        }
    }

    #[test]
    fn test_ids_in_bounds() {
        let n = 50usize;
        let index = build_index(n, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();
        for (id, _) in results {
            assert!((id as usize) < n, "id {} out of bounds (n={})", id, n);
        }
    }

    #[test]
    fn test_multiple_trees_improve_coverage() {
        // With many trees, the candidate set should cover a reasonable fraction of
        // close vectors. Regression guard: ensures each tree uses an independent
        // hyperplane (the bug was both children shared one hyperplane per level).
        let dim = 8;
        let n = 100;
        let params = RpForestParams {
            num_trees: 20,
            tree_params: RPTreeParams { max_leaf_size: 5 },
        };
        let mut index = RpForestIndex::new(dim, params).unwrap();
        // Add clustered data: half near [1,0,...], half near [0,1,0,...]
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            if i < n / 2 {
                v[0] = 1.0;
                v[1] = 0.01 * (i as f32);
            } else {
                v[1] = 1.0;
                v[0] = 0.01 * (i as f32);
            }
            // normalize
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            index.add(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();
        // All top-10 should come from the [1,0,...] cluster (ids 0..50)
        let correct = results
            .iter()
            .filter(|(id, _)| *id < (n / 2) as u32)
            .count();
        assert!(
            correct >= 5,
            "only {correct}/10 results from correct cluster — hyperplane independence may be broken"
        );
    }

    #[test]
    fn test_build_errors_on_empty_index() {
        let mut index = RpForestIndex::new(4, RpForestParams::default()).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn test_add_after_build_errors() {
        let mut index = RpForestIndex::new(4, RpForestParams::default()).unwrap();
        index.add(0, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.build().unwrap();
        assert!(index.add(1, vec![0.0, 1.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let mut index = RpForestIndex::new(4, RpForestParams::default()).unwrap();
        assert!(index.add(0, vec![1.0, 0.0]).is_err());
    }

    #[test]
    fn test_degenerate_split_does_not_recurse_infinitely() {
        // All vectors identical → every random hyperplane sends all points to one side.
        // Regression guard: the degenerate-split guard must fall back to a leaf node
        // rather than recursing until stack overflow.
        let params = RpForestParams {
            num_trees: 3,
            tree_params: RPTreeParams { max_leaf_size: 2 },
        };
        let mut index = RpForestIndex::new(4, params).unwrap();
        for i in 0..20u32 {
            index.add(i, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        }
        // Must complete without stack overflow.
        index.build().unwrap();
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(!results.is_empty());
    }
}
