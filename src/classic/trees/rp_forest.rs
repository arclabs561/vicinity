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
        threshold: f32,
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
        use rand::Rng;
        let mut rng = rand::rng();

        // Generate random hyperplane
        let mut hyperplane = Vec::new();
        let mut norm = 0.0;
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

        // Build tree recursively
        let indices: Vec<u32> = (0..self.num_vectors as u32).collect();
        let root = self.build_tree_recursive(&indices, &hyperplane, 0)?;

        Ok(RPTree { root })
    }

    /// Build tree recursively.
    fn build_tree_recursive(
        &self,
        indices: &[u32],
        hyperplane: &[f32],
        _depth: usize,
    ) -> Result<Option<TreeNode>, RetrieveError> {
        if indices.is_empty() {
            return Ok(None);
        }

        // Leaf node if small enough
        if indices.len() <= self.params.tree_params.max_leaf_size {
            return Ok(Some(TreeNode::Leaf {
                indices: indices.to_vec(),
            }));
        }

        // Split by hyperplane
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            let vec = self.get_vector(idx as usize);
            let projection = simd::dot(vec, hyperplane);

            if projection < 0.0 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Generate new hyperplane for children
        use rand::Rng;
        let mut rng = rand::rng();
        let mut new_hyperplane = Vec::new();
        let mut norm = 0.0;
        for _ in 0..self.dimension {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            norm += val * val;
            new_hyperplane.push(val);
        }
        let norm = norm.sqrt();
        if norm > 0.0 {
            for val in &mut new_hyperplane {
                *val /= norm;
            }
        }

        let left = self.build_tree_recursive(&left_indices, &new_hyperplane, _depth + 1)?;
        let right = self.build_tree_recursive(&right_indices, &new_hyperplane, _depth + 1)?;

        Ok(Some(TreeNode::Internal {
            hyperplane: hyperplane.to_vec(),
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
