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

use crate::RetrieveError;

/// K-Means Tree index.
///
/// Hierarchical clustering tree for approximate nearest neighbor search.
#[derive(Debug)]
pub struct KMeansTreeIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: KMeansTreeParams,
    built: bool,
    root: Option<KMeansNode>,
}

/// K-Means Tree parameters.
#[derive(Clone, Debug)]
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
#[derive(Debug)]
enum KMeansNode {
    /// Internal node: has cluster centers and children
    Internal {
        centers: Vec<Vec<f32>>,          // Cluster centers
        children: Vec<KMeansNode>,       // Child nodes for each cluster
        cluster_assignments: Vec<usize>, // Vector index -> cluster index
    },
    /// Leaf node: contains vector indices
    Leaf {
        indices: Vec<u32>,
        center: Vec<f32>, // Cluster center
    },
}

impl KMeansTreeIndex {
    /// Create new K-Means Tree index.
    pub fn new(dimension: usize, params: KMeansTreeParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::Other(
                "Dimension must be greater than 0".to_string(),
            ));
        }

        if params.num_clusters == 0 {
            return Err(RetrieveError::Other(
                "Number of clusters must be greater than 0".to_string(),
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
        let mut cluster_groups: Vec<Vec<u32>> = vec![Vec::new(); self.params.num_clusters];
        for (idx, &cluster_idx) in indices.iter().zip(assignments.iter()) {
            cluster_groups[cluster_idx].push(*idx);
        }

        // Recursively build children
        let mut children = Vec::new();
        for cluster_indices in cluster_groups {
            if !cluster_indices.is_empty() {
                children.push(self.build_tree(&cluster_indices, depth + 1)?);
            }
        }

        Ok(KMeansNode::Internal {
            centers,
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
        if !self.built {
            return Err(RetrieveError::Other(
                "Index must be built before search".to_string(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Query dimension {} != {}",
                query.len(),
                self.dimension
            )));
        }

        let root = self.root.as_ref().ok_or(RetrieveError::EmptyIndex)?;
        let mut candidates = Vec::new();

        // Search tree
        self.search_node(root, query, &mut candidates);

        // Compute exact distances and sort
        let mut results: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&idx| {
                let vec = get_vector(&self.vectors, self.dimension, idx as usize);
                let dist = euclidean_distance(query, vec);
                (idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);

        Ok(results)
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

    #[test]
    fn test_kmeans_tree_basic() {
        let mut tree = KMeansTreeIndex::new(3, KMeansTreeParams::default()).unwrap();

        // Add test vectors
        for i in 0..100 {
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
            tree.add(i, vec).unwrap();
        }

        tree.build().unwrap();

        // Search
        let query = vec![50.0, 100.0, 150.0];
        let results = tree.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        assert!(results[0].0 < 100);
    }
}
