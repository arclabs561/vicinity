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
use std::path::Path;

const RPTREE_FORMAT_VERSION: u32 = 1;

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

    /// Save a built random-projection tree index to a directory.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt random-projection tree index".into(),
            ));
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
        let index = snapshot.index;
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

        // Collect candidates from tree traversal
        let mut candidates = Vec::new();
        self.search_recursive(root, query, &mut candidates)?;

        // Compute distances and sort
        let mut results: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&idx| {
                let vec = self.get_vector(idx as usize);
                let dist = crate::distance::cosine_distance_normalized(query, vec);
                (self.doc_ids[idx as usize], dist)
            })
            .collect();

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

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
}
