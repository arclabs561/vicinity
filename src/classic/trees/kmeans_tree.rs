//! K-Means Tree implementation.
//!
//! Hierarchical clustering structure for fast similarity search.
//! Uses k-means clustering at each level to partition the data space.
//!
//! **Technical Name**: K-Means Tree
//!
//! Algorithm:
//! - Recursive k-means clustering
//! - Each node represents a cluster (center + vectors)
//! - Search follows branches to closest clusters
//! - Good for medium to high dimensions
//!
//! **Relationships**:
//! - Tree-based ANN method
//! - Uses clustering instead of space partitioning
//! - Complementary to KD-Tree and Ball Tree
//!
//! # References
//!
//! - Survey: Section III-B2
//! - Ponomarenko et al. (2021): "K-means tree: an optimal clustering tree for unsupervised learning"

use crate::classic::trees::persistence::{read_json, validate_vector_shape, write_json_atomic};
use crate::distance::FloatOrd;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::path::Path;

const KMEANS_TREE_FORMAT_VERSION: u32 = 1;

/// K-Means Tree index.
///
/// Hierarchical clustering tree for approximate nearest neighbor search.
#[derive(Debug, Deserialize, Serialize)]
pub struct KMeansTreeIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    doc_ids: Vec<u32>,
    params: KMeansTreeParams,
    built: bool,
    root: Option<KMeansNode>,
}

/// K-Means Tree parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KMeansTreeParams {
    /// Number of clusters per node (k in k-means)
    pub num_clusters: usize,

    /// Maximum leaf size (stop clustering when leaf has this many vectors)
    pub max_leaf_size: usize,

    /// Maximum depth (prevent excessive recursion)
    pub max_depth: usize,

    /// Maximum iterations for k-means clustering
    pub max_iterations: usize,
}

impl Default for KMeansTreeParams {
    fn default() -> Self {
        Self {
            num_clusters: 16,
            max_leaf_size: 50,
            max_depth: 10,
            max_iterations: 10,
        }
    }
}

/// K-Means Tree node.
#[derive(Clone, Debug, Deserialize, Serialize)]
enum KMeansNode {
    /// Internal node: has cluster centers and children
    Internal {
        centers: Vec<Vec<f32>>,    // Cluster centers
        children: Vec<KMeansNode>, // Child nodes for each cluster
        #[allow(dead_code)]
        cluster_assignments: Vec<usize>, // Vector index -> cluster index (reserved for online updates)
    },
    /// Leaf node: contains vector indices
    Leaf {
        indices: Vec<u32>,
        #[allow(dead_code)]
        center: Vec<f32>, // Cluster center (reserved for re-clustering)
    },
}

struct KMeansQueueEntry<'a> {
    distance: FloatOrd,
    sequence: usize,
    node: &'a KMeansNode,
}

impl Eq for KMeansQueueEntry<'_> {}

impl PartialEq for KMeansQueueEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.sequence == other.sequence
    }
}

impl Ord for KMeansQueueEntry<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

impl PartialOrd for KMeansQueueEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Deserialize, Serialize)]
struct KMeansTreeSnapshot {
    version: u32,
    index: KMeansTreeIndex,
}

impl KMeansTreeIndex {
    /// Create new K-Means Tree index.
    pub fn new(dimension: usize, params: KMeansTreeParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "Dimension must be greater than 0".to_string(),
            ));
        }

        if params.num_clusters == 0 {
            return Err(RetrieveError::InvalidParameter(
                "Number of clusters must be greater than 0".to_string(),
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

    /// Build the K-Means Tree.
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

    /// Save a built K-means tree index to a directory.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt K-means tree index".into(),
            ));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        write_json_atomic(
            &output_dir.join("index.json"),
            &KMeansTreeSnapshot {
                version: KMEANS_TREE_FORMAT_VERSION,
                index: self.clone_for_snapshot(),
            },
        )
    }

    /// Load a K-means tree index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let snapshot: KMeansTreeSnapshot = read_json(&input_dir.as_ref().join("index.json"))?;
        if snapshot.version != KMEANS_TREE_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported K-means tree format version {}",
                snapshot.version
            )));
        }
        let index = snapshot.index;
        validate_vector_shape(
            "K-means tree",
            index.dimension,
            index.num_vectors,
            &index.vectors,
            &index.doc_ids,
        )?;
        if !index.built || index.root.is_none() {
            return Err(RetrieveError::FormatError(
                "K-means tree snapshot is not built".into(),
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
            graph_bytes: self.root.as_ref().map(KMeansNode::owned_bytes).unwrap_or(0),
            quantized_bytes: 0,
            metadata_bytes: self.doc_ids.capacity() * std::mem::size_of::<u32>(),
        }
    }

    /// Build tree recursively using k-means clustering.
    fn build_tree(&self, indices: &[u32], depth: usize) -> Result<KMeansNode, RetrieveError> {
        // Base case: create leaf if small enough or max depth reached
        if indices.len() <= self.params.max_leaf_size || depth >= self.params.max_depth {
            let center = self.compute_center(indices);
            return Ok(KMeansNode::Leaf {
                indices: indices.to_vec(),
                center,
            });
        }

        // Perform k-means clustering
        let (centers, assignments) = self.kmeans_cluster(indices)?;

        // Group indices by cluster
        let mut cluster_groups: Vec<Vec<u32>> = vec![Vec::new(); centers.len()];
        for (idx, &cluster_idx) in indices.iter().zip(assignments.iter()) {
            cluster_groups[cluster_idx].push(*idx);
        }

        // Recursively build children
        let mut active_centers = Vec::new();
        let mut children = Vec::new();
        for (cluster_idx, cluster_indices) in cluster_groups.into_iter().enumerate() {
            if !cluster_indices.is_empty() {
                active_centers.push(centers[cluster_idx].clone());
                children.push(self.build_tree(&cluster_indices, depth + 1)?);
            }
        }

        Ok(KMeansNode::Internal {
            centers: active_centers,
            children,
            cluster_assignments: assignments,
        })
    }

    /// Perform k-means clustering on vectors.
    fn kmeans_cluster(
        &self,
        indices: &[u32],
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>), RetrieveError> {
        let k = self.params.num_clusters.min(indices.len());

        // Initialize centers using k-means++ (better than random)
        let mut centers = self.kmeans_plus_plus_init(indices, k)?;

        // K-means iterations
        let mut assignments = vec![0; indices.len()];

        for _iteration in 0..self.params.max_iterations {
            // Assign vectors to nearest centers
            let mut changed = false;
            for (i, &idx) in indices.iter().enumerate() {
                let vec = get_vector(&self.vectors, self.dimension, idx as usize);
                let mut best_cluster = 0;
                let mut best_dist = f32::INFINITY;

                for (cluster_idx, center) in centers.iter().enumerate() {
                    let dist = euclidean_distance(vec, center);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = cluster_idx;
                    }
                }

                if assignments[i] != best_cluster {
                    changed = true;
                    assignments[i] = best_cluster;
                }
            }

            // Update centers
            self.update_centers(indices, &assignments, &mut centers);

            // Early termination if no changes
            if !changed {
                break;
            }
        }

        Ok((centers, assignments))
    }

    /// Initialize centers using k-means++.
    fn kmeans_plus_plus_init(
        &self,
        indices: &[u32],
        k: usize,
    ) -> Result<Vec<Vec<f32>>, RetrieveError> {
        let mut centers = Vec::new();

        // First center: random vector
        let first_idx = indices[0];
        let first_vec = get_vector(&self.vectors, self.dimension, first_idx as usize);
        centers.push(first_vec.to_vec());

        // Subsequent centers: weighted by distance to nearest center
        for _ in 1..k {
            let mut distances = Vec::new();
            for &idx in indices {
                let vec = get_vector(&self.vectors, self.dimension, idx as usize);
                let min_dist = centers
                    .iter()
                    .map(|center| euclidean_distance(vec, center))
                    .fold(f32::INFINITY, f32::min);
                distances.push(min_dist * min_dist); // Square for probability
            }

            // Select center with probability proportional to distance^2
            let total: f32 = distances.iter().sum();
            // Use simple deterministic selection for k-means++ (can be improved with rand if available)
            // In production, use proper random selection
            let mut rng = {
                // Deterministic selection based on index (good enough for initialization)
                (indices.len() as f32 * 0.618_034) % total // Golden ratio for spread
            };
            let mut selected_idx = 0;
            for (i, &dist) in distances.iter().enumerate() {
                rng -= dist;
                if rng <= 0.0 {
                    selected_idx = i;
                    break;
                }
            }

            let vec = get_vector(
                &self.vectors,
                self.dimension,
                indices[selected_idx] as usize,
            );
            centers.push(vec.to_vec());
        }

        Ok(centers)
    }

    /// Update cluster centers based on assignments.
    fn update_centers(&self, indices: &[u32], assignments: &[usize], centers: &mut [Vec<f32>]) {
        let k = centers.len();
        let mut counts = vec![0; k];

        // Reset centers
        for center in centers.iter_mut() {
            center.fill(0.0);
        }

        // Sum vectors in each cluster
        for (i, &idx) in indices.iter().enumerate() {
            let cluster = assignments[i];
            let vec = get_vector(&self.vectors, self.dimension, idx as usize);

            for (j, &val) in vec.iter().enumerate() {
                centers[cluster][j] += val;
            }
            counts[cluster] += 1;
        }

        // Average to get new centers
        for (cluster, count) in counts.iter().enumerate() {
            if *count > 0 {
                for val in centers[cluster].iter_mut() {
                    *val /= *count as f32;
                }
            }
        }
    }

    /// Compute center of vectors.
    fn compute_center(&self, indices: &[u32]) -> Vec<f32> {
        let mut center = vec![0.0; self.dimension];

        for &idx in indices {
            let vec = get_vector(&self.vectors, self.dimension, idx as usize);
            for (i, &val) in vec.iter().enumerate() {
                center[i] += val;
            }
        }

        let n = indices.len() as f32;
        for val in center.iter_mut() {
            *val /= n;
        }

        center
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_branch_budget(query, k, 1)
    }

    /// Search for k nearest neighbors while visiting multiple child clusters per internal node.
    pub fn search_with_branch_budget(
        &self,
        query: &[f32],
        k: usize,
        branch_budget: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "Index must be built before search".to_string(),
            ));
        }
        if branch_budget == 0 {
            return Err(RetrieveError::InvalidParameter(
                "branch_budget must be greater than 0".to_string(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::InvalidParameter(format!(
                "Query dimension {} != {}",
                query.len(),
                self.dimension
            )));
        }

        let root = self.root.as_ref().ok_or(RetrieveError::EmptyIndex)?;
        let mut candidates = Vec::new();

        if branch_budget == 1 {
            self.search_node(root, query, &mut candidates);
        } else {
            self.search_node_with_branch_budget(root, query, branch_budget, &mut candidates);
        }

        Ok(self.rank_candidates(query, k, &candidates))
    }

    /// Search for k nearest neighbors while visiting a global best-first budget of leaf clusters.
    pub fn search_with_leaf_budget(
        &self,
        query: &[f32],
        k: usize,
        leaf_budget: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "Index must be built before search".to_string(),
            ));
        }
        if leaf_budget == 0 {
            return Err(RetrieveError::InvalidParameter(
                "leaf_budget must be greater than 0".to_string(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::InvalidParameter(format!(
                "Query dimension {} != {}",
                query.len(),
                self.dimension
            )));
        }

        let root = self.root.as_ref().ok_or(RetrieveError::EmptyIndex)?;
        let mut candidates = Vec::new();
        self.search_node_with_leaf_budget(root, query, leaf_budget, &mut candidates);
        Ok(self.rank_candidates(query, k, &candidates))
    }

    fn rank_candidates(&self, query: &[f32], k: usize, candidates: &[u32]) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&idx| {
                let vec = get_vector(&self.vectors, self.dimension, idx as usize);
                let dist = euclidean_distance(query, vec);
                (self.doc_ids[idx as usize], dist)
            })
            .collect();

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);
        results
    }

    /// Search node recursively.
    fn search_node(&self, node: &KMeansNode, query: &[f32], candidates: &mut Vec<u32>) {
        match node {
            KMeansNode::Leaf { indices, .. } => {
                candidates.extend_from_slice(indices);
            }
            KMeansNode::Internal {
                centers, children, ..
            } => {
                // Find closest cluster
                let mut best_cluster = 0;
                let mut best_dist = f32::INFINITY;

                for (i, center) in centers.iter().enumerate() {
                    let dist = euclidean_distance(query, center);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = i;
                    }
                }

                // Search closest cluster's subtree
                if best_cluster < children.len() {
                    self.search_node(&children[best_cluster], query, candidates);
                }
            }
        }
    }

    fn search_node_with_branch_budget(
        &self,
        node: &KMeansNode,
        query: &[f32],
        branch_budget: usize,
        candidates: &mut Vec<u32>,
    ) {
        match node {
            KMeansNode::Leaf { indices, .. } => {
                candidates.extend_from_slice(indices);
            }
            KMeansNode::Internal {
                centers, children, ..
            } => {
                let mut ranked: Vec<(usize, f32)> = centers
                    .iter()
                    .zip(children.iter())
                    .enumerate()
                    .map(|(idx, (center, _))| (idx, euclidean_distance(query, center)))
                    .collect();
                ranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

                for (idx, _) in ranked.into_iter().take(branch_budget) {
                    self.search_node_with_branch_budget(
                        &children[idx],
                        query,
                        branch_budget,
                        candidates,
                    );
                }
            }
        }
    }

    fn search_node_with_leaf_budget(
        &self,
        root: &KMeansNode,
        query: &[f32],
        leaf_budget: usize,
        candidates: &mut Vec<u32>,
    ) {
        let mut frontier = BinaryHeap::new();
        frontier.push(Reverse(KMeansQueueEntry {
            distance: FloatOrd(0.0),
            sequence: 0,
            node: root,
        }));

        let mut sequence = 1;
        let mut leaves_visited = 0;
        while leaves_visited < leaf_budget {
            let Some(Reverse(entry)) = frontier.pop() else {
                break;
            };

            match entry.node {
                KMeansNode::Leaf { indices, .. } => {
                    candidates.extend_from_slice(indices);
                    leaves_visited += 1;
                }
                KMeansNode::Internal {
                    centers, children, ..
                } => {
                    for (center, child) in centers.iter().zip(children.iter()) {
                        frontier.push(Reverse(KMeansQueueEntry {
                            distance: FloatOrd(euclidean_distance(query, center)),
                            sequence,
                            node: child,
                        }));
                        sequence += 1;
                    }
                }
            }
        }
    }
}

impl KMeansNode {
    fn owned_bytes(&self) -> usize {
        match self {
            KMeansNode::Internal {
                centers,
                children,
                cluster_assignments,
            } => {
                centers.capacity() * std::mem::size_of::<Vec<f32>>()
                    + centers
                        .iter()
                        .map(|center| center.capacity() * std::mem::size_of::<f32>())
                        .sum::<usize>()
                    + children.capacity() * std::mem::size_of::<KMeansNode>()
                    + children.iter().map(KMeansNode::owned_bytes).sum::<usize>()
                    + cluster_assignments.capacity() * std::mem::size_of::<usize>()
            }
            KMeansNode::Leaf { indices, center } => {
                indices.capacity() * std::mem::size_of::<u32>()
                    + center.capacity() * std::mem::size_of::<f32>()
            }
        }
    }
}

use crate::distance::l2_distance as euclidean_distance;

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn build_index() -> KMeansTreeIndex {
        let mut tree = KMeansTreeIndex::new(3, KMeansTreeParams::default()).unwrap();
        for i in 0..100u32 {
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
            tree.add(3000 + i, vec).unwrap();
        }
        tree.build().unwrap();
        tree
    }

    fn brute_force(tree: &KMeansTreeIndex, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut results: Vec<_> = (0..tree.num_vectors)
            .map(|idx| {
                let vector = get_vector(&tree.vectors, tree.dimension, idx);
                (tree.doc_ids[idx], euclidean_distance(query, vector))
            })
            .collect();
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);
        results
    }

    #[test]
    fn test_kmeans_tree_basic() {
        let tree = build_index();

        // Search
        let query = vec![50.0, 100.0, 150.0];
        let results = tree.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        assert!((3000..3100).contains(&results[0].0));
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let tree = build_index();
        let dir = tempfile::tempdir().unwrap();
        tree.save_to_dir(dir.path()).unwrap();
        let loaded = KMeansTreeIndex::load_from_dir(dir.path()).unwrap();
        let query = vec![50.0, 100.0, 150.0];
        assert_eq!(
            tree.search(&query, 8).unwrap(),
            loaded.search(&query, 8).unwrap()
        );
    }

    #[test]
    fn save_load_roundtrip_preserves_leaf_budget_search() {
        let tree = build_index();
        let dir = tempfile::tempdir().unwrap();
        tree.save_to_dir(dir.path()).unwrap();
        let loaded = KMeansTreeIndex::load_from_dir(dir.path()).unwrap();
        let query = vec![50.0, 100.0, 150.0];
        assert_eq!(
            tree.search_with_leaf_budget(&query, 8, 8).unwrap(),
            loaded.search_with_leaf_budget(&query, 8, 8).unwrap()
        );
    }

    #[test]
    fn internal_centers_match_non_empty_children() {
        let params = KMeansTreeParams {
            num_clusters: 8,
            max_leaf_size: 1,
            max_depth: 3,
            max_iterations: 10,
        };
        let mut tree = KMeansTreeIndex::new(2, params).unwrap();
        for i in 0..6u32 {
            tree.add(i, vec![0.0, 0.0]).unwrap();
        }
        for i in 6..12u32 {
            tree.add(i, vec![100.0, 100.0]).unwrap();
        }
        tree.build().unwrap();

        fn assert_aligned(node: &KMeansNode) {
            match node {
                KMeansNode::Internal {
                    centers, children, ..
                } => {
                    assert_eq!(centers.len(), children.len());
                    for child in children {
                        assert_aligned(child);
                    }
                }
                KMeansNode::Leaf { .. } => {}
            }
        }

        assert_aligned(tree.root.as_ref().unwrap());
    }

    #[test]
    fn branch_budget_one_matches_default_search() {
        let tree = build_index();
        let query = vec![50.0, 100.0, 150.0];
        assert_eq!(
            tree.search(&query, 8).unwrap(),
            tree.search_with_branch_budget(&query, 8, 1).unwrap()
        );
    }

    #[test]
    fn branch_budget_must_be_positive() {
        let tree = build_index();
        let err = tree
            .search_with_branch_budget(&[50.0, 100.0, 150.0], 8, 0)
            .unwrap_err();
        assert!(matches!(err, RetrieveError::InvalidParameter(_)));
    }

    #[test]
    fn leaf_budget_must_be_positive() {
        let tree = build_index();
        let err = tree
            .search_with_leaf_budget(&[50.0, 100.0, 150.0], 8, 0)
            .unwrap_err();
        assert!(matches!(err, RetrieveError::InvalidParameter(_)));
    }

    #[test]
    fn large_leaf_budget_matches_brute_force() {
        let tree = build_index();
        let query = vec![50.0, 100.0, 150.0];
        assert_eq!(
            tree.search_with_leaf_budget(&query, 8, 128).unwrap(),
            brute_force(&tree, &query, 8)
        );
    }
}
