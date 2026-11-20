//! KD-Tree (K-Dimensional Tree) implementation.
//!
//! Classic space-partitioning tree for low-dimensional data (d < 20).
//! Useful as a low-dimensional baseline. This implementation is bounded by
//! leaf size and maximum depth, so benchmark rows should be treated as
//! implementation-specific rather than a mathematical exact-search oracle.
//!
//! **Technical Name**: KD-Tree (K-Dimensional Tree)
//!
//! Algorithm:
//! - Recursive space partitioning by alternating dimensions
//! - Each node splits space along one dimension
//! - Best for low-dimensional data (d < 20)
//! - Can approach exact nearest neighbors in low dimensions when configured
//!   with enough depth and small leaves
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
            return Err(RetrieveError::InvalidConfig {
                field: "dimension",
                expected: "must be greater than 0",
            });
        }

        if dimension > 50 {
            return Err(RetrieveError::Unsupported {
                feature: "KD-tree with dimension > 50",
                hint: "use BallTree or an HNSW index for high-dimensional data",
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
                component: "KD-tree index",
                operation: "add vectors",
            });
        }

        // Distances are `1 - dot`, which is cosine distance only for unit
        // vectors, and the pruning bounds assume unit vectors too. Store the
        // normalized vector, as the other cosine indexes do.
        self.vectors
            .extend_from_slice(&crate::distance::normalize(&embedding));
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
            return Err(RetrieveError::NotBuilt {
                component: "KD-tree index",
                operation: "save",
            });
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

    /// Estimated heap memory used by this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        crate::memory::MemoryReport {
            vectors_bytes: self.vectors.capacity() * std::mem::size_of::<f32>(),
            graph_bytes: self.root.as_ref().map(KDNode::owned_bytes).unwrap_or(0),
            quantized_bytes: 0,
            metadata_bytes: self.doc_ids.capacity() * std::mem::size_of::<u32>(),
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
            return Err(RetrieveError::NotBuilt {
                component: "KD-tree index",
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
            component: "KD-tree index",
            operation: "search",
        })?;

        let dist_fn = crate::distance::cosine_distance_normalized;
        let query = &crate::distance::normalize(query)[..];

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
    /// Every point `x` in the far subtree differs from the query by at least
    /// `diff` along the split dimension, so `||q - x||^2 >= diff^2`. For unit
    /// vectors `||q - x||^2 = 2 * (1 - q.x)`, twice the cosine distance the
    /// results are ranked by. The far subtree can therefore be skipped only
    /// when `diff^2 >= 2 * worst_dist`. Comparing `diff^2` with `worst_dist`
    /// directly prunes subtrees that can still hold a nearer point.
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

                // Prune the far subtree only when it cannot hold a nearer point
                // (see the doc comment for the factor of 2).
                let split_dist = diff * diff;
                if split_dist < 2.0 * *worst_dist || results.len() < k {
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

impl KDNode {
    fn owned_bytes(&self) -> usize {
        match self {
            KDNode::Internal { left, right, .. } => {
                boxed_node_bytes(left) + boxed_node_bytes(right)
            }
            KDNode::Leaf { indices } => indices.capacity() * std::mem::size_of::<u32>(),
        }
    }
}

fn boxed_node_bytes(node: &KDNode) -> usize {
    std::mem::size_of::<KDNode>() + node.owned_bytes()
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

    fn lcg_vectors(n: usize, dim: usize, seed: u64, unit: bool) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| next()).collect();
                if unit {
                    crate::distance::normalize(&v)
                } else {
                    // Lengths from about 0.1 to 10.
                    let scale = 10f32.powf(next());
                    v.iter().map(|x| x * scale).collect()
                }
            })
            .collect()
    }

    fn brute_force(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<f32> {
        let q = crate::distance::normalize(query);
        let mut d: Vec<f32> = data
            .iter()
            .map(|v| {
                1.0 - crate::distance::normalize(v)
                    .iter()
                    .zip(&q)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            })
            .collect();
        d.sort_unstable_by(|a, b| a.total_cmp(b));
        d.truncate(k);
        d
    }

    /// The tree is documented as exact. With small leaves the pruning bound
    /// decides most of the search; comparing `diff^2` with the cosine
    /// distance (half the squared L2 distance) pruned true neighbours.
    #[test]
    fn search_matches_brute_force_on_unit_vectors() {
        for (n, dim, seed) in [(200usize, 3usize, 2u64), (600, 6, 7), (400, 4, 11)] {
            let data = lcg_vectors(n, dim, seed, true);
            let params = KDTreeParams {
                max_leaf_size: 2,
                ..KDTreeParams::default()
            };
            let mut index = KDTreeIndex::new(dim, params).unwrap();
            for (i, v) in data.iter().enumerate() {
                index.add(i as u32, v.clone()).unwrap();
            }
            index.build().unwrap();
            for query in lcg_vectors(40, dim, seed + 100, true) {
                let got: Vec<f32> = index
                    .search(&query, 10)
                    .unwrap()
                    .iter()
                    .map(|r| r.1)
                    .collect();
                let want = brute_force(&data, &query, 10);
                for (g, w) in got.iter().zip(&want) {
                    assert!(
                        (g - w).abs() < 1e-5,
                        "n={n} dim={dim}: got {got:?}, want {want:?}"
                    );
                }
            }
        }
    }

    /// Vectors of any length are ranked by angle, and an exact match comes
    /// first with distance 0.
    #[test]
    fn non_unit_vectors_are_ranked_by_cosine() {
        let mut index = KDTreeIndex::new(2, KDTreeParams::default()).unwrap();
        index.add(0, vec![1.0, 0.0]).unwrap();
        index.add(1, vec![10.0, 1.0]).unwrap();
        index.add(2, vec![0.0, 1.0]).unwrap();
        index.build().unwrap();
        let results = index.search(&[5.0, 0.0], 3).unwrap();
        assert_eq!(
            results.iter().map(|r| r.0).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(results[0].1.abs() < 1e-6);
        assert!(results.iter().all(|r| r.1 >= -1e-6), "{results:?}");

        let data = lcg_vectors(300, 4, 5, false);
        let mut index = KDTreeIndex::new(4, KDTreeParams::default()).unwrap();
        for (i, v) in data.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();
        for query in lcg_vectors(20, 4, 99, false) {
            let got: Vec<f32> = index
                .search(&query, 5)
                .unwrap()
                .iter()
                .map(|r| r.1)
                .collect();
            let want = brute_force(&data, &query, 5);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-5, "got {got:?}, want {want:?}");
            }
        }
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
