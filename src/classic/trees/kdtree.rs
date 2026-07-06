//! KD-Tree (K-Dimensional Tree) implementation.
//!
//! Classic space-partitioning tree for low-dimensional data (d < 20).
//! Excellent for exact nearest neighbor search in low dimensions.
//!
//! **Technical Name**: KD-Tree (K-Dimensional Tree)
//!
//! Algorithm:
//! - Recursive space partitioning by alternating dimensions
//! - Each node splits space along one dimension
//! - Best for low-dimensional data (d < 20)
//! - Can provide exact nearest neighbors in low dimensions
//!
//! **Relationships**:
//! - Classic tree-based method (predecessor to modern methods)
//! - Complementary to Ball Tree (better for medium dimensions)
//! - Foundation for many tree-based ANN methods
//!
//! # References
//!
//! - Bentley (1975): "Multidimensional binary search trees used for associative searching"
//! - Friedman et al. (1977): "An algorithm for finding best matches in logarithmic expected time"

use crate::classic::trees::persistence::{read_json, validate_vector_shape, write_json_atomic};
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::path::Path;

const KDTREE_FORMAT_VERSION: u32 = 1;

/// KD-Tree index.
///
/// Space-partitioning tree for low-dimensional approximate nearest neighbor search.
#[derive(Deserialize, Serialize)]
pub struct KDTreeIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    doc_ids: Vec<u32>,
    params: KDTreeParams,
    built: bool,
    root: Option<KDNode>,
}

/// KD-Tree parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KDTreeParams {
    /// Maximum leaf size (stop splitting when leaf has this many vectors)
    pub max_leaf_size: usize,

    /// Maximum depth (prevent excessive recursion)
    pub max_depth: usize,
}

impl Default for KDTreeParams {
    fn default() -> Self {
        Self {
            max_leaf_size: 10,
            max_depth: 32,
        }
    }
}

/// KD-Tree node.
#[derive(Clone, Deserialize, Serialize)]
enum KDNode {
    /// Internal node: splits along a dimension
    Internal {
        dimension: usize,
        split_value: f32,
        left: Box<KDNode>,
        right: Box<KDNode>,
    },
    /// Leaf node: contains vector indices
    Leaf { indices: Vec<u32> },
}

#[derive(Deserialize, Serialize)]
struct KDTreeSnapshot {
    version: u32,
    index: KDTreeIndex,
}

impl KDTreeIndex {
    /// Create new KD-Tree index.
    pub fn new(dimension: usize, params: KDTreeParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "Dimension must be greater than 0".to_string(),
            ));
        }

        if dimension > 50 {
            return Err(RetrieveError::InvalidParameter(
                "KD-Tree not recommended for dimensions > 50. Use Ball Tree or modern methods."
                    .to_string(),
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

    /// Build the KD-Tree.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let indices: Vec<u32> = (0..self.num_vectors as u32).collect();
        self.root = Some(self.build_tree(&indices, 0, 0)?);

        self.built = true;
        Ok(())
    }

    /// Save a built KD-tree index to a directory.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt KD-tree index".into(),
            ));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        write_json_atomic(
            &output_dir.join("index.json"),
            &KDTreeSnapshot {
                version: KDTREE_FORMAT_VERSION,
                index: self.clone_for_snapshot(),
            },
        )
    }

    /// Load a KD-tree index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let snapshot: KDTreeSnapshot = read_json(&input_dir.as_ref().join("index.json"))?;
        if snapshot.version != KDTREE_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported KD-tree format version {}",
                snapshot.version
            )));
        }
        let index = snapshot.index;
        validate_vector_shape(
            "KD-tree",
            index.dimension,
            index.num_vectors,
            &index.vectors,
            &index.doc_ids,
        )?;
        if !index.built || index.root.is_none() {
            return Err(RetrieveError::FormatError(
                "KD-tree snapshot is not built".into(),
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
    fn build_tree(
        &self,
        indices: &[u32],
        depth: usize,
        dimension: usize,
    ) -> Result<KDNode, RetrieveError> {
        if indices.is_empty() {
            return Ok(KDNode::Leaf {
                indices: Vec::new(),
            });
        }

        // Leaf node if small enough or max depth reached
        if indices.len() <= self.params.max_leaf_size || depth >= self.params.max_depth {
            return Ok(KDNode::Leaf {
                indices: indices.to_vec(),
            });
        }

        // Select dimension to split (alternate)
        let split_dim = dimension % self.dimension;

        // Find median value in this dimension
        let mut values: Vec<(f32, u32)> = indices
            .iter()
            .map(|&idx| {
                let vec = self.get_vector(idx as usize);
                (vec[split_dim], idx)
            })
            .collect();

        values.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let median_idx = values.len() / 2;
        let split_value = values[median_idx].0;

        // Split indices by median
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for (val, idx) in values {
            if val < split_value {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Build children
        let left = self.build_tree(&left_indices, depth + 1, split_dim + 1)?;
        let right = self.build_tree(&right_indices, depth + 1, split_dim + 1)?;

        Ok(KDNode::Internal {
            dimension: split_dim,
            split_value,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Search for k nearest neighbors.
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

        let dist_fn = crate::distance::cosine_distance_normalized;

        // Use a max-heap of size k to track the k best results.
        // worst_dist starts at infinity so the first k points are always accepted.
        let mut results: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut worst_dist = f32::INFINITY;
        self.search_recursive(root, query, k, &mut results, &mut worst_dist, dist_fn);

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);

        Ok(results)
    }

    /// Search recursively with pruning.
    ///
    /// Prune the far subtree when the squared coordinate difference along the
    /// split dimension exceeds `worst_dist`. This is exact for L2; for cosine
    /// on normalized vectors it is a valid lower bound (coordinate difference
    /// <= L2 distance <= cosine distance for unit vectors).
    fn search_recursive(
        &self,
        node: &KDNode,
        query: &[f32],
        k: usize,
        results: &mut Vec<(u32, f32)>,
        worst_dist: &mut f32,
        dist_fn: fn(&[f32], &[f32]) -> f32,
    ) {
        match node {
            KDNode::Leaf { indices } => {
                for &idx in indices {
                    let vec = self.get_vector(idx as usize);
                    let dist = dist_fn(query, vec);
                    if results.len() < k {
                        results.push((self.doc_ids[idx as usize], dist));
                        if results.len() == k {
                            // Find worst distance among the k results
                            *worst_dist = results
                                .iter()
                                .map(|(_, d)| *d)
                                .fold(f32::NEG_INFINITY, f32::max);
                        }
                    } else if dist < *worst_dist {
                        // Replace the worst element
                        if let Some(pos) = results
                            .iter()
                            .position(|(_, d)| (*d - *worst_dist).abs() < f32::EPSILON)
                        {
                            results[pos] = (self.doc_ids[idx as usize], dist);
                            *worst_dist = results
                                .iter()
                                .map(|(_, d)| *d)
                                .fold(f32::NEG_INFINITY, f32::max);
                        }
                    }
                }
            }
            KDNode::Internal {
                dimension,
                split_value,
                left,
                right,
            } => {
                let query_val = query[*dimension];
                let diff = query_val - split_value;

                // Traverse near subtree first
                let (near, far) = if diff < 0.0 {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                self.search_recursive(near, query, k, results, worst_dist, dist_fn);

                // Prune: skip far subtree if the split-plane distance exceeds worst_dist.
                // diff^2 is a lower bound on the L2 distance to any point in the far subtree.
                let split_dist = diff * diff;
                if split_dist < *worst_dist || results.len() < k {
                    self.search_recursive(far, query, k, results, worst_dist, dist_fn);
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn build_index() -> KDTreeIndex {
        let mut index = KDTreeIndex::new(3, KDTreeParams::default()).unwrap();
        for i in 0..16u32 {
            index
                .add(1000 + i, vec![i as f32, (i * 2) as f32, 1.0])
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
        assert!(results.iter().all(|(id, _)| *id >= 1000));
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let index = build_index();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = KDTreeIndex::load_from_dir(dir.path()).unwrap();
        let query = [4.0, 8.0, 1.0];
        assert_eq!(
            index.search(&query, 5).unwrap(),
            loaded.search(&query, 5).unwrap()
        );
    }
}
