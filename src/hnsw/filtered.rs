//! ACORN-style filtered HNSW search.
//!
//! Implements filter-aware traversal strategies for HNSW that maintain
//! navigability when filters eliminate nodes from consideration.
//!
//! # Algorithm Overview
//!
//! Standard HNSW with post-filtering fails when filters are restrictive
//! because the search may land in regions where few nodes match the filter.
//! ACORN solves this with:
//!
//! 1. **Two-hop expansion**: When visiting a node, examine both direct
//!    neighbors AND neighbors-of-neighbors (2-hop)
//! 2. **Filter-first**: Evaluate filter predicate before computing distances
//! 3. **Adaptive behavior**: Use standard traversal when dense matches exist,
//!    expand to 2-hop when sparse
//!
//! # References
//!
//! - Patel et al. (2024): "ACORN: Performant and Predicate-Agnostic Search
//!   Over Vector Embeddings and Structured Data"
//! - Weaviate blog: "Speed Up Filtered Vector Search"

use crate::RetrieveError;
use std::collections::{BinaryHeap, HashSet};

/// Filter predicate for ACORN search.
pub trait FilterPredicate: Sync {
    /// Check if a node passes the filter.
    fn matches(&self, node_id: u32) -> bool;
}

/// Simple function-based filter.
pub struct FnFilter<F: Fn(u32) -> bool + Sync>(pub F);

impl<F: Fn(u32) -> bool + Sync> FilterPredicate for FnFilter<F> {
    fn matches(&self, node_id: u32) -> bool {
        self.0(node_id)
    }
}

/// Always-pass filter (no filtering).
pub struct NoFilter;

impl FilterPredicate for NoFilter {
    fn matches(&self, _node_id: u32) -> bool {
        true
    }
}

/// ACORN search configuration.
#[derive(Clone, Debug)]
pub struct AcornConfig {
    /// Enable two-hop expansion when filter is selective
    pub enable_two_hop: bool,
    /// Threshold for switching to two-hop (ratio of filtered/visited)
    pub two_hop_threshold: f32,
    /// Maximum two-hop neighbors to examine per node
    pub max_two_hop_neighbors: usize,
    /// Expansion factor for candidate pool
    pub ef_search: usize,
}

impl Default for AcornConfig {
    fn default() -> Self {
        Self {
            enable_two_hop: true,
            two_hop_threshold: 0.3, // Switch to 2-hop if <30% pass filter
            max_two_hop_neighbors: 32,
            ef_search: 100,
        }
    }
}

/// Search state tracking for adaptive behavior.
struct SearchState {
    visited: HashSet<u32>,
    filtered_count: usize,
    visited_count: usize,
}

impl SearchState {
    fn new() -> Self {
        Self {
            visited: HashSet::new(),
            filtered_count: 0,
            visited_count: 0,
        }
    }

    fn visit(&mut self, node_id: u32, passes_filter: bool) -> bool {
        if self.visited.insert(node_id) {
            self.visited_count += 1;
            if passes_filter {
                self.filtered_count += 1;
            }
            true
        } else {
            false
        }
    }

    fn filter_ratio(&self) -> f32 {
        if self.visited_count == 0 {
            1.0
        } else {
            self.filtered_count as f32 / self.visited_count as f32
        }
    }
}

/// Candidate for search (ordered by distance, reversed for max-heap).
#[derive(Clone, Copy)]
struct Candidate {
    node_id: u32,
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
        // Max-heap: larger distance = higher priority (pops largest first).
        // The frontier stores negated distances so pop() returns the closest node.
        // The results heap stores positive distances so pop() evicts the farthest.
        // Use total_cmp for IEEE 754 total ordering (NaN-safe).
        self.distance.total_cmp(&other.distance)
    }
}

/// ACORN-style filtered search on HNSW graph.
///
/// This function performs filtered k-NN search with adaptive two-hop expansion.
///
/// # Arguments
/// * `query` - Query vector
/// * `k` - Number of results to return
/// * `config` - ACORN configuration
/// * `filter` - Filter predicate
/// * `get_neighbors` - Function to get neighbors of a node
/// * `compute_distance` - Function to compute distance between query and node
/// * `entry_point` - Starting node for search
///
/// # Returns
/// Vector of (node_id, distance) pairs for nodes passing the filter.
pub fn acorn_search<F, N, D>(
    k: usize,
    config: &AcornConfig,
    filter: &F,
    get_neighbors: N,
    compute_distance: D,
    entry_point: u32,
) -> Result<Vec<(u32, f32)>, RetrieveError>
where
    F: FilterPredicate,
    N: Fn(u32) -> Vec<u32>,
    D: Fn(u32) -> f32,
{
    let mut state = SearchState::new();

    // Result candidates (filtered nodes only)
    let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

    // Search frontier
    let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();

    // Initialize with entry point
    let entry_passes = filter.matches(entry_point);
    state.visit(entry_point, entry_passes);
    let entry_dist = compute_distance(entry_point);

    frontier.push(Candidate {
        node_id: entry_point,
        distance: -entry_dist, // Negate for min-heap behavior
    });

    if entry_passes {
        results.push(Candidate {
            node_id: entry_point,
            distance: entry_dist,
        });
    }

    // Track worst result distance for pruning
    let mut worst_result_dist = f32::INFINITY;

    while let Some(current) = frontier.pop() {
        let current_dist = -current.distance; // Un-negate

        // Early termination: only if we have enough results AND frontier is worse
        // Be more conservative about stopping when filter is selective
        let can_stop = results.len() >= k && current_dist > worst_result_dist * 1.5; // 50% margin
        if can_stop && state.filter_ratio() > 0.3 {
            break;
        }

        // Get direct neighbors
        let neighbors = get_neighbors(current.node_id);

        // Decide whether to use two-hop expansion based on filter selectivity
        let use_two_hop = config.enable_two_hop
            && (state.filter_ratio() < config.two_hop_threshold || results.len() < k); // Always use two-hop if we don't have k results

        // Process direct neighbors
        for &neighbor in &neighbors {
            let neighbor_passes = filter.matches(neighbor);
            if !state.visit(neighbor, neighbor_passes) {
                continue; // Already visited
            }

            // Compute distance for all neighbors (needed for navigation)
            let dist = compute_distance(neighbor);

            if neighbor_passes {
                // Add to results
                results.push(Candidate {
                    node_id: neighbor,
                    distance: dist,
                });

                // Keep only top k
                while results.len() > k {
                    results.pop();
                }

                // Update worst distance
                if let Some(worst) = results.peek() {
                    worst_result_dist = worst.distance;
                }
            }

            // Add to frontier for exploration (even if doesn't pass filter)
            // This is critical: we need to navigate through non-matching nodes
            if dist < worst_result_dist * 2.0 || results.len() < k {
                frontier.push(Candidate {
                    node_id: neighbor,
                    distance: -dist,
                });
            }

            // Two-hop expansion for non-matching nodes
            if !neighbor_passes && use_two_hop {
                let two_hop_neighbors = get_neighbors(neighbor);
                let mut two_hop_count = 0;

                for &two_hop in &two_hop_neighbors {
                    if two_hop_count >= config.max_two_hop_neighbors {
                        break;
                    }

                    let two_hop_passes = filter.matches(two_hop);
                    if !state.visit(two_hop, two_hop_passes) {
                        continue;
                    }

                    let two_hop_dist = compute_distance(two_hop);

                    if two_hop_passes {
                        results.push(Candidate {
                            node_id: two_hop,
                            distance: two_hop_dist,
                        });

                        while results.len() > k {
                            results.pop();
                        }

                        if let Some(worst) = results.peek() {
                            worst_result_dist = worst.distance;
                        }
                    }

                    // Also add two-hop to frontier
                    if two_hop_dist < worst_result_dist * 2.0 || results.len() < k {
                        frontier.push(Candidate {
                            node_id: two_hop,
                            distance: -two_hop_dist,
                        });
                    }

                    two_hop_count += 1;
                }
            }
        }

        // Limit exploration
        if state.visited_count >= config.ef_search * 10 {
            break;
        }
    }

    // Convert results to sorted vector
    let mut result_vec: Vec<(u32, f32)> = results
        .into_iter()
        .map(|c| (c.node_id, c.distance))
        .collect();

    result_vec.sort_by(|a, b| a.1.total_cmp(&b.1));
    result_vec.truncate(k);

    Ok(result_vec)
}

/// Filter execution strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterStrategy {
    /// Evaluate filter before computing distance (default for ACORN)
    PreFilter,
    /// Compute distance, then filter results
    PostFilter,
    /// Adaptive: start with pre-filter, switch based on selectivity
    Adaptive,
}

/// Estimate optimal filter strategy based on selectivity.
///
/// # Arguments
/// * `estimated_selectivity` - Fraction of nodes expected to pass filter (0.0 to 1.0)
/// * `k` - Number of results needed
/// * `total_nodes` - Total number of nodes in index
///
/// # Returns
/// Recommended filter strategy.
pub fn recommend_strategy(
    estimated_selectivity: f32,
    k: usize,
    total_nodes: usize,
) -> FilterStrategy {
    // If very selective (few pass), use ACORN with pre-filter
    if estimated_selectivity < 0.1 {
        return FilterStrategy::PreFilter;
    }

    // If high selectivity (most pass), post-filter is fine
    if estimated_selectivity > 0.8 {
        return FilterStrategy::PostFilter;
    }

    // For k much smaller than filtered set, post-filter works
    let estimated_filtered = (total_nodes as f32 * estimated_selectivity) as usize;
    if k * 10 < estimated_filtered {
        return FilterStrategy::PostFilter;
    }

    // Otherwise, use adaptive
    FilterStrategy::Adaptive
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn mock_graph() -> (Vec<Vec<u32>>, Vec<f32>) {
        // 10-node fully connected graph for reliable navigation
        // Each node connects to its neighbors within distance 2
        let neighbors = vec![
            vec![1, 2, 3],          // 0
            vec![0, 2, 3, 4],       // 1
            vec![0, 1, 3, 4, 5],    // 2
            vec![0, 1, 2, 4, 5, 6], // 3
            vec![1, 2, 3, 5, 6, 7], // 4
            vec![2, 3, 4, 6, 7, 8], // 5
            vec![3, 4, 5, 7, 8, 9], // 6
            vec![4, 5, 6, 8, 9],    // 7
            vec![5, 6, 7, 9],       // 8
            vec![6, 7, 8],          // 9
        ];

        // Distances from query - lower is better
        let distances = vec![0.5, 0.3, 0.6, 0.4, 0.7, 0.2, 0.8, 0.1, 0.9, 0.35];

        (neighbors, distances)
    }

    #[test]
    fn test_acorn_no_filter() {
        let (neighbors, distances) = mock_graph();

        let config = AcornConfig {
            enable_two_hop: true,
            two_hop_threshold: 0.3,
            max_two_hop_neighbors: 32,
            ef_search: 100,
        };

        let results = acorn_search(
            5, // Get more results to ensure we find best ones
            &config,
            &NoFilter,
            |id| neighbors[id as usize].clone(),
            |id| distances[id as usize],
            0,
        )
        .unwrap();

        assert!(!results.is_empty(), "Should find some results");
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1, "Results should be sorted");
        }
    }

    #[test]
    fn test_acorn_with_filter() {
        let (neighbors, distances) = mock_graph();

        // Filter: only even nodes (0, 2, 4, 6, 8)
        let filter = FnFilter(|id: u32| id % 2 == 0);

        let results = acorn_search(
            3,
            &AcornConfig::default(),
            &filter,
            |id| neighbors[id as usize].clone(),
            |id| distances[id as usize],
            0,
        )
        .unwrap();

        // All results should be even
        for (id, _) in &results {
            assert_eq!(id % 2, 0, "Node {} should be even", id);
        }
    }

    #[test]
    fn test_acorn_selective_filter() {
        let (neighbors, distances) = mock_graph();

        // Filter: only nodes 8 or 9 (far end of graph)
        let filter = FnFilter(|id: u32| id >= 8);

        let config = AcornConfig {
            enable_two_hop: true,
            two_hop_threshold: 0.8, // High threshold to force two-hop early
            max_two_hop_neighbors: 32,
            ef_search: 100,
        };

        let results = acorn_search(
            2,
            &config,
            &filter,
            |id| neighbors[id as usize].clone(),
            |id| distances[id as usize],
            0,
        )
        .unwrap();

        // Should find nodes 8 and/or 9 through expansion
        assert!(!results.is_empty(), "Should find at least one node >= 8");
        for (id, _) in &results {
            assert!(*id >= 8, "Node {} should be >= 8", id);
        }
    }

    #[test]
    fn test_recommend_strategy() {
        // Very selective -> PreFilter
        assert_eq!(
            recommend_strategy(0.05, 10, 10000),
            FilterStrategy::PreFilter
        );

        // High selectivity -> PostFilter
        assert_eq!(
            recommend_strategy(0.9, 10, 10000),
            FilterStrategy::PostFilter
        );

        // Medium with small k -> PostFilter
        assert_eq!(
            recommend_strategy(0.5, 10, 10000),
            FilterStrategy::PostFilter
        );

        // Medium with large k -> Adaptive
        assert_eq!(
            recommend_strategy(0.3, 1000, 10000),
            FilterStrategy::Adaptive
        );
    }
}
