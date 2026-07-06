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

use crate::classic::trees::persistence::{read_json, validate_vector_shape, write_json_atomic};
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::path::Path;

const BALLTREE_FORMAT_VERSION: u32 = 1;

/// Ball Tree index.
///
/// Space-partitioning tree using hyperspheres for medium-dimensional data.
#[derive(Deserialize, Serialize)]
pub struct BallTreeIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    doc_ids: Vec<u32>,
    params: BallTreeParams,
    built: bool,
    root: Option<BallNode>,
}

/// Ball Tree parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
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
#[derive(Clone, Deserialize, Serialize)]
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

#[derive(Deserialize, Serialize)]
struct BallTreeSnapshot {
    version: u32,
    index: BallTreeIndex,
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
            doc_ids: Vec::new(),
            params,
            built: false,
            root: None,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u32, embedding: Vec<f32>) -> Result<(), RetrieveError> {
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
        self.doc_ids.push(doc_id);
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

    /// Save a built Ball tree index to a directory.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt Ball tree index".into(),
            ));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        write_json_atomic(
            &output_dir.join("index.json"),
            &BallTreeSnapshot {
                version: BALLTREE_FORMAT_VERSION,
                index: self.clone_for_snapshot(),
            },
        )
    }

    /// Load a Ball tree index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let snapshot: BallTreeSnapshot = read_json(&input_dir.as_ref().join("index.json"))?;
        if snapshot.version != BALLTREE_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported Ball tree format version {}",
                snapshot.version
            )));
        }
        let index = snapshot.index;
        validate_vector_shape(
            "Ball tree",
            index.dimension,
            index.num_vectors,
            &index.vectors,
            &index.doc_ids,
        )?;
        if !index.built || index.root.is_none() {
            return Err(RetrieveError::FormatError(
                "Ball tree snapshot is not built".into(),
            ));
        }
        Ok(index)
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
        }
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
        let mut results: Vec<(u32, f32)> = best_k
            .iter()
            .map(|&(d, idx)| (self.doc_ids[idx as usize], d))
            .collect();
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn build_index() -> BallTreeIndex {
        let mut index = BallTreeIndex::new(3, BallTreeParams::default()).unwrap();
        for i in 0..16u32 {
            index
                .add(2000 + i, vec![i as f32, (i * 2) as f32, 1.0])
                .unwrap();
        }
        index.build().unwrap();
        index
    }

    #[test]
    fn search_returns_external_doc_ids() {
        let index = build_index();
        let results = index.search(&[4.0, 8.0, 1.0], 3).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|(id, _)| *id >= 2000));
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let index = build_index();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = BallTreeIndex::load_from_dir(dir.path()).unwrap();
        let query = [4.0, 8.0, 1.0];
        assert_eq!(
            index.search(&query, 5).unwrap(),
            loaded.search(&query, 5).unwrap()
        );
    }
}
