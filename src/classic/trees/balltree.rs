//! Ball Tree implementation.
//!
//! Space-partitioning tree using hyperspheres (balls) instead of hyperplanes.
//! Better than KD-Tree for medium dimensions (20 < d < 100).
//!
//! **Technical Name**: Ball Tree
//!
//! Algorithm:
//! - Recursive space partitioning using hyperspheres
//! - Each node represents a ball (center + radius) containing its vectors
//! - Better for medium dimensions than KD-Tree
//! - More robust to high-dimensional data
//!
//! **Relationships**:
//! - Improvement over KD-Tree for medium dimensions
//! - Uses hyperspheres instead of hyperplanes
//! - Complementary to KD-Tree (KD-Tree better for d < 20, Ball Tree better for 20 < d < 100)
//!
//! # References
//!
//! - Omohundro (1989): "Five balltree construction algorithms"
//! - Liu et al. (2006): "An investigation of practical approximate nearest neighbor algorithms"

use crate::RetrieveError;

/// Ball Tree index.
///
/// Space-partitioning tree using hyperspheres for medium-dimensional data.
pub struct BallTreeIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: BallTreeParams,
    built: bool,
    root: Option<BallNode>,
}

/// Ball Tree parameters.
#[derive(Clone, Debug)]
pub struct BallTreeParams {
    /// Maximum leaf size
    pub max_leaf_size: usize,

    /// Maximum depth
    pub max_depth: usize,
}

impl Default for BallTreeParams {
    fn default() -> Self {
        Self {
            max_leaf_size: 10,
            max_depth: 32,
        }
    }
}

/// Ball Tree node.
enum BallNode {
    /// Internal node: has center, radius, and children
    Internal {
        center: Vec<f32>,
        radius: f32,
        left: Box<BallNode>,
        right: Box<BallNode>,
    },
    /// Leaf node: contains vector indices
    Leaf {
        indices: Vec<u32>,
        center: Vec<f32>,
        radius: f32,
    },
}

impl BallTreeIndex {
    /// Create new Ball Tree index.
    pub fn new(dimension: usize, params: BallTreeParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
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
            return Err(RetrieveError::InvalidParameter(format!(
                "Embedding dimension {} != {}",
                embedding.len(),
                self.dimension
            )));
        }

        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "Cannot add vectors after build".to_string(),
            ));
        }

        self.vectors.extend_from_slice(&embedding);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the Ball Tree.
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
    fn build_tree(&self, indices: &[u32], depth: usize) -> Result<BallNode, RetrieveError> {
        if indices.is_empty() {
            return Err(RetrieveError::InvalidParameter("Empty indices".to_string()));
        }

        // Compute center and radius
        let center = self.compute_center(indices);
        let radius = self.compute_radius(indices, &center);

        // Leaf node if small enough or max depth reached
        if indices.len() <= self.params.max_leaf_size || depth >= self.params.max_depth {
            return Ok(BallNode::Leaf {
                indices: indices.to_vec(),
                center,
                radius,
            });
        }

        // Find two farthest points as seeds for splitting
        let (seed1_idx, seed2_idx) = self.find_farthest_pair(indices);

        // Split indices by distance to seeds
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            let vec = self.get_vector(idx as usize);
            let dist1 = crate::distance::l2_distance(vec, self.get_vector(seed1_idx as usize));
            let dist2 = crate::distance::l2_distance(vec, self.get_vector(seed2_idx as usize));

            if dist1 < dist2 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Ensure both sides have at least one point.
        // Safety: indices.len() >= 2 here (leaf check above requires max_leaf_size >= 1),
        // so at least one side is non-empty after the split loop.
        if left_indices.is_empty() {
            #[allow(clippy::unwrap_used)] // right_indices is non-empty when left is empty
            left_indices.push(right_indices.pop().unwrap());
        }
        if right_indices.is_empty() {
            #[allow(clippy::unwrap_used)] // left_indices is non-empty (just ensured above)
            right_indices.push(left_indices.pop().unwrap());
        }

        // Build children
        let left = self.build_tree(&left_indices, depth + 1)?;
        let right = self.build_tree(&right_indices, depth + 1)?;

        Ok(BallNode::Internal {
            center,
            radius,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Compute center of vectors.
    fn compute_center(&self, indices: &[u32]) -> Vec<f32> {
        let mut center = vec![0.0f32; self.dimension];

        for &idx in indices {
            let vec = self.get_vector(idx as usize);
            for (j, &val) in vec.iter().enumerate() {
                center[j] += val;
            }
        }

        let count = indices.len() as f32;
        for val in center.iter_mut() {
            *val /= count;
        }

        center
    }

    /// Compute radius (max distance from center).
    fn compute_radius(&self, indices: &[u32], center: &[f32]) -> f32 {
        let mut max_radius = 0.0f32;

        for &idx in indices {
            let vec = self.get_vector(idx as usize);
            let dist = crate::distance::l2_distance(vec, center);
            max_radius = max_radius.max(dist);
        }

        max_radius
    }

    /// Find two farthest points.
    fn find_farthest_pair(&self, indices: &[u32]) -> (u32, u32) {
        let mut max_dist = 0.0f32;
        let mut pair = (indices[0], indices[0]);

        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let vec1 = self.get_vector(indices[i] as usize);
                let vec2 = self.get_vector(indices[j] as usize);
                let dist = crate::distance::l2_distance(vec1, vec2);

                if dist > max_dist {
                    max_dist = dist;
                    pair = (indices[i], indices[j]);
                }
            }
        }

        pair
    }

    /// Search for k nearest neighbors.
    ///
    /// Uses ball tree pruning: a ball can be skipped if the minimum possible
    /// distance to any point in the ball (dist_to_center - radius) is greater
    /// than the current k-th best distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "Index not built".to_string(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::InvalidParameter(format!(
                "Query dimension {} != {}",
                query.len(),
                self.dimension
            )));
        }

        let root = self
            .root
            .as_ref()
            .ok_or_else(|| RetrieveError::InvalidParameter("Tree not built".to_string()))?;

        // Use a bounded priority queue for k-nearest neighbors
        // Store (distance, index) pairs, sorted by distance descending (max-heap behavior)
        let mut best_k: Vec<(f32, u32)> = Vec::with_capacity(k);
        let mut best_dist = f32::INFINITY; // Current k-th best distance (pruning threshold)

        self.search_recursive_pruned(root, query, k, &mut best_k, &mut best_dist)?;

        // Convert to output format: (index, distance)
        let mut results: Vec<(u32, f32)> = best_k.iter().map(|&(d, idx)| (idx, d)).collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1));

        Ok(results)
    }

    /// Search with radius-based pruning.
    ///
    /// Pruning rule: if `dist(query, center) - radius > best_dist`, the ball
    /// cannot contain any point closer than our current k-th best, so skip it.
    fn search_recursive_pruned(
        &self,
        node: &BallNode,
        query: &[f32],
        k: usize,
        best_k: &mut Vec<(f32, u32)>,
        best_dist: &mut f32,
    ) -> Result<(), RetrieveError> {
        match node {
            BallNode::Leaf {
                indices,
                center,
                radius,
            } => {
                // Pruning check for leaf: can this leaf contain better results?
                let dist_to_center = crate::distance::l2_distance(query, center);
                let min_possible_dist = (dist_to_center - radius).max(0.0);

                if min_possible_dist > *best_dist {
                    // This leaf can't have better results, skip it
                    return Ok(());
                }

                // Process all vectors in leaf
                for &idx in indices {
                    let vec = self.get_vector(idx as usize);
                    let dist = crate::distance::cosine_distance_normalized(query, vec);

                    if best_k.len() < k {
                        // Not yet k results, add unconditionally
                        best_k.push((dist, idx));
                        if best_k.len() == k {
                            // Now we have k results, find the worst
                            *best_dist = best_k
                                .iter()
                                .map(|&(d, _)| d)
                                .fold(f32::NEG_INFINITY, f32::max);
                        }
                    } else if dist < *best_dist {
                        // Replace the worst result
                        if let Some(worst_idx) = best_k
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
                            .map(|(i, _)| i)
                        {
                            best_k[worst_idx] = (dist, idx);
                            // Update best_dist
                            *best_dist = best_k
                                .iter()
                                .map(|&(d, _)| d)
                                .fold(f32::NEG_INFINITY, f32::max);
                        }
                    }
                }
            }
            BallNode::Internal {
                center,
                radius,
                left,
                right,
            } => {
                // Compute distance from query to ball center
                let dist_to_center = crate::distance::l2_distance(query, center);

                // Pruning: minimum possible distance to any point in this ball
                let min_possible_dist = (dist_to_center - radius).max(0.0);

                if min_possible_dist > *best_dist {
                    // This entire subtree can be pruned
                    return Ok(());
                }

                // Compute distances to children's centers for prioritization
                let (left_center, left_radius) = match left.as_ref() {
                    BallNode::Internal { center, radius, .. } => (center, *radius),
                    BallNode::Leaf { center, radius, .. } => (center, *radius),
                };
                let (right_center, right_radius) = match right.as_ref() {
                    BallNode::Internal { center, radius, .. } => (center, *radius),
                    BallNode::Leaf { center, radius, .. } => (center, *radius),
                };

                let left_dist = crate::distance::l2_distance(query, left_center);
                let right_dist = crate::distance::l2_distance(query, right_center);

                // Visit closer child first (more likely to find good results early)
                let left_min = (left_dist - left_radius).max(0.0);
                let right_min = (right_dist - right_radius).max(0.0);

                if left_min < right_min {
                    self.search_recursive_pruned(left, query, k, best_k, best_dist)?;
                    self.search_recursive_pruned(right, query, k, best_k, best_dist)?;
                } else {
                    self.search_recursive_pruned(right, query, k, best_k, best_dist)?;
                    self.search_recursive_pruned(left, query, k, best_k, best_dist)?;
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
