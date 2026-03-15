//! Dynamic Edge Navigation Graph (DEG) for bimodal data hybrid search.
//!
//! DEG is optimized for scenarios where data has bimodal structure -
//! dense clusters with sparse inter-cluster connectivity. It adapts
//! graph structure dynamically based on local density.
//!
//! # Key Features
//!
//! - **Adaptive edge counts**: More edges in sparse regions, fewer in dense
//! - **Density-aware navigation**: Uses local density for search decisions
//! - **Efficient updates**: Dynamic edge maintenance without full rebuild
//! - **Hybrid search**: Combines graph navigation with density estimation
//!
//! # Algorithm
//!
//! 1. During construction:
//!    - Estimate local density for each node
//!    - Assign edge budget based on density (more edges for isolated nodes)
//!    - Use alpha-pruning but with density-weighted distances
//!
//! 2. During search:
//!    - Navigate using greedy search
//!    - Expand exploration in sparse regions
//!    - Contract in dense regions to reduce computation
//!
//! # References
//!
//! - DEG (2025): "Dynamic Edge Navigation Graph for Hybrid Vector Search"

use crate::RetrieveError;
use std::collections::{BinaryHeap, HashSet};

/// DEG configuration.
#[derive(Clone, Debug)]
pub struct DEGConfig {
    /// Base number of edges per node
    pub base_edges: usize,
    /// Maximum edges per node
    pub max_edges: usize,
    /// Minimum edges per node
    pub min_edges: usize,
    /// Density estimation radius (k neighbors)
    pub density_k: usize,
    /// Alpha for diversity pruning
    pub alpha: f32,
    /// Expansion factor during search
    pub ef_search: usize,
}

impl Default for DEGConfig {
    fn default() -> Self {
        Self {
            base_edges: 16,
            max_edges: 32,
            min_edges: 8,
            density_k: 10,
            alpha: 1.2,
            ef_search: 100,
        }
    }
}

/// Node density information.
#[derive(Clone, Debug)]
pub struct DensityInfo {
    /// Local density score (higher = denser region)
    pub density: f32,
    /// Assigned edge budget
    pub edge_budget: usize,
    /// Average distance to k nearest neighbors
    pub avg_neighbor_dist: f32,
}

/// DEG index for hybrid vector search.
pub struct DEGIndex {
    config: DEGConfig,
    dim: usize,
    /// Vectors stored by ID
    vectors: Vec<Vec<f32>>,
    /// Graph edges per node
    edges: Vec<Vec<u32>>,
    /// Density information per node
    density: Vec<DensityInfo>,
    /// Entry point for search
    entry_point: Option<u32>,
}

impl DEGIndex {
    /// Create new DEG index.
    pub fn new(dim: usize, config: DEGConfig) -> Self {
        Self {
            config,
            dim,
            vectors: Vec::new(),
            edges: Vec::new(),
            density: Vec::new(),
            entry_point: None,
        }
    }

    /// Add vector to index.
    pub fn add(&mut self, vector: Vec<f32>) -> Result<u32, RetrieveError> {
        if vector.len() != self.dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.dim,
                doc_dim: vector.len(),
            });
        }

        let id = self.vectors.len() as u32;
        self.vectors.push(vector);
        self.edges.push(Vec::new());
        self.density.push(DensityInfo {
            density: 0.0,
            edge_budget: self.config.base_edges,
            avg_neighbor_dist: 0.0,
        });

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
        }

        Ok(id)
    }

    /// Build index with density-aware edge assignment.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        let n = self.vectors.len();

        // Step 1: Estimate density for each node
        self.estimate_densities()?;

        // Step 2: Assign edge budgets based on density
        self.assign_edge_budgets();

        // Step 3: Build edges with density-aware pruning
        for i in 0..n {
            self.connect_node(i as u32)?;
        }

        // Step 4: Select best entry point (medoid)
        self.select_entry_point();

        Ok(())
    }

    /// Estimate local density for each node.
    fn estimate_densities(&mut self) -> Result<(), RetrieveError> {
        let n = self.vectors.len();
        let k = self.config.density_k.min(n - 1);

        for i in 0..n {
            // Find k nearest neighbors (brute force for now)
            let mut distances: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u32, self.distance(i as u32, j as u32)))
                .collect();

            distances.sort_by(|a, b| a.1.total_cmp(&b.1));

            // Compute average distance to k nearest
            let k_neighbors: Vec<_> = distances.iter().take(k).collect();
            let avg_dist = if k_neighbors.is_empty() {
                1.0
            } else {
                k_neighbors.iter().map(|(_, d)| d).sum::<f32>() / k_neighbors.len() as f32
            };

            // Density is inverse of average distance (with smoothing)
            let density = 1.0 / (avg_dist + 0.1);

            self.density[i] = DensityInfo {
                density,
                edge_budget: self.config.base_edges,
                avg_neighbor_dist: avg_dist,
            };
        }

        Ok(())
    }

    /// Assign edge budgets based on density.
    fn assign_edge_budgets(&mut self) {
        // Find density range
        let min_density = self
            .density
            .iter()
            .map(|d| d.density)
            .fold(f32::INFINITY, f32::min);
        let max_density = self
            .density
            .iter()
            .map(|d| d.density)
            .fold(f32::NEG_INFINITY, f32::max);

        let density_range = (max_density - min_density).max(0.1);

        for info in &mut self.density {
            // Normalize density to [0, 1]
            let normalized = (info.density - min_density) / density_range;

            // Low density (sparse) -> more edges
            // High density -> fewer edges
            let edge_range = (self.config.max_edges - self.config.min_edges) as f32;
            let budget = self.config.max_edges - (normalized * edge_range) as usize;

            info.edge_budget = budget.clamp(self.config.min_edges, self.config.max_edges);
        }
    }

    /// Connect a node using density-aware pruning.
    fn connect_node(&mut self, node_id: u32) -> Result<(), RetrieveError> {
        let budget = self.density[node_id as usize].edge_budget;

        // Find candidates (all other nodes for now)
        let mut candidates: Vec<(u32, f32)> = (0..self.vectors.len() as u32)
            .filter(|&j| j != node_id)
            .map(|j| (j, self.distance(node_id, j)))
            .collect();

        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Select neighbors with alpha-pruning
        let mut neighbors = Vec::new();

        for (candidate, dist) in candidates {
            if neighbors.len() >= budget {
                break;
            }

            // Alpha-pruning: skip if too close to existing neighbor
            let is_diverse = neighbors.iter().all(|&n| {
                let neighbor_dist = self.distance(candidate, n);
                neighbor_dist > dist * self.config.alpha || neighbor_dist > dist
            });

            if is_diverse {
                neighbors.push(candidate);
            }
        }

        // Add bidirectional edges
        for &neighbor in &neighbors {
            let neighbor_edges = &mut self.edges[neighbor as usize];
            if !neighbor_edges.contains(&node_id) {
                let neighbor_budget = self.density[neighbor as usize].edge_budget;
                if neighbor_edges.len() < neighbor_budget {
                    neighbor_edges.push(node_id);
                }
            }
        }

        self.edges[node_id as usize] = neighbors;

        Ok(())
    }

    /// Select entry point (approximate medoid).
    fn select_entry_point(&mut self) {
        if self.vectors.is_empty() {
            return;
        }

        // Use node with highest density as entry point (central region)
        let best = self
            .density
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.density.total_cmp(&b.1.density))
            .map(|(i, _)| i as u32);

        if let Some(entry) = best {
            self.entry_point = Some(entry);
        }
    }

    /// Search for k nearest neighbors with density-aware navigation.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if query.len() != self.dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dim,
            });
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let entry = self.entry_point.unwrap_or(0);

        // Greedy search with density-aware expansion
        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        // Initialize
        let entry_dist = self.query_distance(entry, query);
        candidates.push(Candidate {
            id: entry,
            distance: -entry_dist,
        }); // Min-heap
        results.push(Candidate {
            id: entry,
            distance: entry_dist,
        });
        visited.insert(entry);

        while let Some(Candidate {
            id: current,
            distance: neg_dist,
        }) = candidates.pop()
        {
            let current_dist = -neg_dist;

            // Get worst result distance
            let worst_result = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);

            if current_dist > worst_result && results.len() >= k {
                break;
            }

            // Get local density for adaptive exploration
            let local_density = self.density[current as usize].density;
            let expansion = if local_density < 0.5 {
                2 // Sparse region: explore more
            } else {
                1 // Dense region: normal exploration
            };

            // Expand neighbors
            for &neighbor in &self.edges[current as usize] {
                if visited.insert(neighbor) {
                    let dist = self.query_distance(neighbor, query);

                    // Add to results
                    if results.len() < k || dist < worst_result {
                        results.push(Candidate {
                            id: neighbor,
                            distance: dist,
                        });
                        while results.len() > k {
                            results.pop();
                        }
                    }

                    // Add to candidates (with expansion factor)
                    for _ in 0..expansion {
                        if candidates.len() < self.config.ef_search {
                            candidates.push(Candidate {
                                id: neighbor,
                                distance: -dist,
                            });
                        }
                    }
                }
            }

            if visited.len() >= self.config.ef_search {
                break;
            }
        }

        // Convert results
        let mut result_vec: Vec<(u32, f32)> =
            results.into_iter().map(|c| (c.id, c.distance)).collect();
        result_vec.sort_by(|a, b| a.1.total_cmp(&b.1));
        result_vec.truncate(k);

        Ok(result_vec)
    }

    /// Compute distance between two vectors in index.
    fn distance(&self, a: u32, b: u32) -> f32 {
        euclidean_distance(&self.vectors[a as usize], &self.vectors[b as usize])
    }

    /// Compute distance from query to vector in index.
    fn query_distance(&self, id: u32, query: &[f32]) -> f32 {
        euclidean_distance(&self.vectors[id as usize], query)
    }

    /// Number of vectors in index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get density info for a node.
    pub fn get_density(&self, id: u32) -> Option<&DensityInfo> {
        self.density.get(id as usize)
    }

    /// Get edge count for a node.
    pub fn edge_count(&self, id: u32) -> usize {
        self.edges.get(id as usize).map(|e| e.len()).unwrap_or(0)
    }
}

/// Search candidate.
#[derive(Clone, Copy)]
struct Candidate {
    id: u32,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance = higher priority (for results pruning)
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance)
    }
}

/// Euclidean distance.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn create_clustered_data(
        num_clusters: usize,
        points_per_cluster: usize,
        dim: usize,
    ) -> Vec<Vec<f32>> {
        let mut data = Vec::new();

        for c in 0..num_clusters {
            let center_offset = c as f32 * 10.0;

            for p in 0..points_per_cluster {
                let mut point = vec![0.0; dim];
                for (d, val) in point.iter_mut().enumerate() {
                    *val = center_offset + ((p * d) % 10) as f32 * 0.1;
                }
                data.push(point);
            }
        }

        data
    }

    #[test]
    fn test_deg_basic() {
        let mut index = DEGIndex::new(4, DEGConfig::default());

        // Add vectors
        for i in 0..10 {
            let v = vec![i as f32 * 0.1; 4];
            index.add(v).unwrap();
        }

        // Build
        index.build().unwrap();

        assert_eq!(index.len(), 10);
        assert!(index.entry_point.is_some());
    }

    #[test]
    fn test_deg_search() {
        let mut index = DEGIndex::new(
            4,
            DEGConfig {
                density_k: 3,
                base_edges: 4,
                ..Default::default()
            },
        );

        // Add clustered data
        let data = create_clustered_data(3, 10, 4);
        for v in data {
            index.add(v).unwrap();
        }

        index.build().unwrap();

        // Search
        let query = vec![0.0; 4]; // Near first cluster
        let results = index.search(&query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_density_estimation() {
        let mut index = DEGIndex::new(2, DEGConfig::default());

        // Create data with varying density
        // Dense cluster at origin
        for i in 0..10 {
            index.add(vec![i as f32 * 0.1, i as f32 * 0.1]).unwrap();
        }

        // Isolated point
        index.add(vec![100.0, 100.0]).unwrap();

        index.build().unwrap();

        // Isolated point should have lower density
        let isolated_density = index.get_density(10).unwrap().density;
        let cluster_density = index.get_density(5).unwrap().density;

        assert!(isolated_density < cluster_density);
    }

    #[test]
    fn test_adaptive_edge_budget() {
        let mut index = DEGIndex::new(
            2,
            DEGConfig {
                min_edges: 2,
                max_edges: 8,
                base_edges: 4,
                ..Default::default()
            },
        );

        // Dense cluster
        for i in 0..20 {
            index.add(vec![i as f32 * 0.1, i as f32 * 0.05]).unwrap();
        }

        // Isolated points
        index.add(vec![50.0, 50.0]).unwrap();
        index.add(vec![60.0, 60.0]).unwrap();

        index.build().unwrap();

        // Isolated points should have more edges (higher budget for sparse regions)
        let isolated_budget = index.get_density(20).unwrap().edge_budget;
        let cluster_budget = index.get_density(10).unwrap().edge_budget;

        assert!(isolated_budget >= cluster_budget);
    }

    #[test]
    fn test_config_defaults() {
        let config = DEGConfig::default();

        assert_eq!(config.base_edges, 16);
        assert_eq!(config.max_edges, 32);
        assert_eq!(config.min_edges, 8);
        assert_eq!(config.density_k, 10);
    }
}
