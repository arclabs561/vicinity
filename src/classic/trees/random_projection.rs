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

use crate::simd;
use crate::RetrieveError;

/// Random Projection Tree index.
///
/// Single tree using random hyperplanes for space partitioning.
pub struct RPTreeIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: RPTreeParams,
    built: bool,
    root: Option<RPNode>,
}

/// Random Projection Tree parameters.
#[derive(Clone, Debug)]
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

impl RPTreeIndex {
    /// Create new Random Projection Tree index.
    pub fn new(dimension: usize, params: RPTreeParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::Other(
                "Dimension must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            params,
            built: false,
            root: None,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _doc_id: u32, embedding: Vec<f32>) -> Result<(), RetrieveError> {
        if embedding.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Embedding dimension {} != {}",
                embedding.len(),
                self.dimension
            )));
        }

        if self.built {
            return Err(RetrieveError::Other(
                "Cannot add vectors after build".to_string(),
            ));
        }

        self.vectors.extend_from_slice(&embedding);
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

        let indices: Vec<u32> = (0..self.num_vectors as u32).collect();
        self.root = Some(self.build_tree(&indices, 0)?);

        self.built = true;
        Ok(())
    }

    /// Build tree recursively.
    fn build_tree(&self, indices: &[u32], depth: usize) -> Result<RPNode, RetrieveError> {
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
        let hyperplane = self.generate_random_hyperplane();

        // Compute projections and find median
        let mut projections: Vec<(f32, u32)> = indices
            .iter()
            .map(|&idx| {
                let vec = self.get_vector(idx as usize);
                let projection = simd::dot(vec, &hyperplane);
                (projection, idx)
            })
            .collect();

        projections.sort_by(|a, b| a.0.total_cmp(&b.0));
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
        let left = self.build_tree(&left_indices, depth + 1)?;
        let right = self.build_tree(&right_indices, depth + 1)?;

        Ok(RPNode::Internal {
            hyperplane,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Generate random hyperplane.
    fn generate_random_hyperplane(&self) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::rng();

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
            return Err(RetrieveError::Other("Index not built".to_string()));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Query dimension {} != {}",
                query.len(),
                self.dimension
            )));
        }

        let root = self
            .root
            .as_ref()
            .ok_or_else(|| RetrieveError::Other("Tree not built".to_string()))?;

        // Collect candidates from tree traversal
        let mut candidates = Vec::new();
        self.search_recursive(root, query, &mut candidates)?;

        // Compute distances and sort
        let mut results: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&idx| {
                let vec = self.get_vector(idx as usize);
                let dist = crate::distance::cosine_distance_normalized(query, vec);
                (idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);

        Ok(results)
    }

    /// Search recursively.
    fn search_recursive(
        &self,
        node: &RPNode,
        query: &[f32],
        candidates: &mut Vec<u32>,
    ) -> Result<(), RetrieveError> {
        match node {
            RPNode::Leaf { indices } => {
                candidates.extend_from_slice(indices);
            }
            RPNode::Internal {
                hyperplane,
                threshold,
                left,
                right,
            } => {
                let projection = simd::dot(query, hyperplane);

                // Traverse both subtrees (pruning could be added)
                if projection < *threshold {
                    self.search_recursive(left, query, candidates)?;
                    self.search_recursive(right, query, candidates)?;
                } else {
                    self.search_recursive(right, query, candidates)?;
                    self.search_recursive(left, query, candidates)?;
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
