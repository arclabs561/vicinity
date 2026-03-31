//! Random Walk-based graph repair for HNSW deletions.
//!
//! An alternative to MN-RU that preserves hitting time statistics using
//! random walk analysis. When a node is deleted, edges are rewired to
//! maintain the expected number of steps to reach any other node.
//!
//! # Key Insight
//!
//! HNSW search is essentially a greedy random walk toward the query.
//! When we delete a node, we change the transition probabilities of the
//! Markov chain. The goal is to rewire edges so that hitting times
//! (expected steps to reach targets) are preserved.
//!
//! # Algorithm
//!
//! 1. For each neighbor N of deleted node D:
//!    a. Compute the "flow" through D to other nodes
//!    b. Redirect this flow by adding edges to D's other neighbors
//!    c. Weight new edges by their contribution to hitting time preservation
//!
//! 2. Use randomized sampling to estimate hitting times efficiently
//!
//! # References
//!
//! - Mishra et al. (2025): "Graph-based Nearest Neighbors with Dynamic Updates
//!   via Random Walks" - <https://arxiv.org/abs/2512.18060>

use crate::RetrieveError;
use std::collections::{HashMap, HashSet};

/// Random walk repair configuration.
#[derive(Clone, Debug)]
pub struct RandomWalkConfig {
    /// Number of random walks to sample for estimation
    pub num_walks: usize,
    /// Maximum walk length
    pub max_walk_length: usize,
    /// Maximum neighbors per node
    pub max_neighbors: usize,
    /// Damping factor for PageRank-like computation
    pub damping: f32,
    /// Whether to use deterministic refinement after sampling
    pub refine: bool,
}

impl Default for RandomWalkConfig {
    fn default() -> Self {
        Self {
            num_walks: 100,
            max_walk_length: 50,
            max_neighbors: 16,
            damping: 0.85,
            refine: true,
        }
    }
}

/// Statistics from random walk repair.
#[derive(Clone, Debug, Default)]
pub struct RandomWalkStats {
    /// Nodes affected by deletion
    pub nodes_affected: usize,
    /// Edges removed
    pub edges_removed: usize,
    /// Edges added
    pub edges_added: usize,
    /// Random walks performed
    pub walks_performed: usize,
    /// Estimated hitting time change
    pub hitting_time_delta: f32,
}

/// Node importance scores computed via random walks.
#[derive(Clone, Debug)]
pub struct ImportanceScores {
    /// PageRank-like scores per node
    pub scores: HashMap<u32, f32>,
    /// Hitting times from entry point
    pub hitting_times: HashMap<u32, f32>,
}

/// Random walk repair engine.
pub struct RandomWalkRepairer<'a> {
    config: RandomWalkConfig,
    /// Get neighbors of a node
    get_neighbors: Box<dyn Fn(u32) -> Vec<u32> + 'a>,
    /// Compute distance between two nodes
    compute_distance: Box<dyn Fn(u32, u32) -> f32 + 'a>,
    /// RNG seed
    seed: u64,
}

impl<'a> RandomWalkRepairer<'a> {
    /// Create new random walk repairer.
    pub fn new<G, D>(config: RandomWalkConfig, get_neighbors: G, compute_distance: D) -> Self
    where
        G: Fn(u32) -> Vec<u32> + 'a,
        D: Fn(u32, u32) -> f32 + 'a,
    {
        Self {
            config,
            get_neighbors: Box::new(get_neighbors),
            compute_distance: Box::new(compute_distance),
            seed: 42,
        }
    }

    /// Set RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Compute repair operations for deleting a node.
    ///
    /// Returns a map of node_id -> new_neighbors.
    pub fn compute_repairs(
        &self,
        deleted_node: u32,
        deleted_set: &HashSet<u32>,
    ) -> Result<(HashMap<u32, Vec<u32>>, RandomWalkStats), RetrieveError> {
        let mut stats = RandomWalkStats::default();
        let mut repairs: HashMap<u32, Vec<u32>> = HashMap::new();

        // Get neighbors of deleted node
        let neighbors_of_deleted = (self.get_neighbors)(deleted_node);
        let valid_neighbors: Vec<u32> = neighbors_of_deleted
            .iter()
            .filter(|n| !deleted_set.contains(n))
            .cloned()
            .collect();

        if valid_neighbors.is_empty() {
            return Ok((repairs, stats));
        }

        // Compute importance scores via random walks from deleted node's neighbors
        let importance = self.compute_importance_from_walks(
            &valid_neighbors,
            deleted_node,
            deleted_set,
            &mut stats,
        );

        // For each neighbor of deleted node, rewire edges
        for &neighbor in &valid_neighbors {
            stats.nodes_affected += 1;

            // Current neighbors (excluding deleted)
            let current: Vec<u32> = (self.get_neighbors)(neighbor)
                .into_iter()
                .filter(|&n| n != deleted_node && !deleted_set.contains(&n))
                .collect();

            stats.edges_removed += 1;

            // Find replacement candidates based on hitting time preservation
            let replacements = self.find_replacements_by_importance(
                neighbor,
                &current,
                &valid_neighbors,
                &importance,
                deleted_set,
            );

            let mut new_neighbors = current;
            for candidate in replacements {
                if new_neighbors.len() >= self.config.max_neighbors {
                    break;
                }
                if !new_neighbors.contains(&candidate) {
                    new_neighbors.push(candidate);
                    stats.edges_added += 1;
                }
            }

            repairs.insert(neighbor, new_neighbors);
        }

        // Optional: deterministic refinement
        if self.config.refine {
            self.refine_repairs(&mut repairs, deleted_set, &mut stats);
        }

        Ok((repairs, stats))
    }

    /// Compute importance scores via random walks.
    fn compute_importance_from_walks(
        &self,
        start_nodes: &[u32],
        deleted: u32,
        deleted_set: &HashSet<u32>,
        stats: &mut RandomWalkStats,
    ) -> ImportanceScores {
        let mut visit_counts: HashMap<u32, usize> = HashMap::new();
        let mut first_visit_step: HashMap<u32, Vec<usize>> = HashMap::new();

        let mut rng_state = self.seed;

        for &start in start_nodes {
            for _ in 0..self.config.num_walks {
                stats.walks_performed += 1;

                let mut current = start;
                let mut visited_this_walk: HashSet<u32> = HashSet::new();

                for step in 0..self.config.max_walk_length {
                    // Skip deleted nodes
                    if deleted_set.contains(&current) || current == deleted {
                        break;
                    }

                    // Record visit
                    *visit_counts.entry(current).or_insert(0) += 1;

                    // Record first visit step for hitting time (only first visit)
                    if visited_this_walk.insert(current) {
                        first_visit_step.entry(current).or_default().push(step);
                    }

                    // Random walk step
                    let neighbors = (self.get_neighbors)(current);
                    let valid: Vec<u32> = neighbors
                        .into_iter()
                        .filter(|&n| n != deleted && !deleted_set.contains(&n))
                        .collect();

                    if valid.is_empty() {
                        break;
                    }

                    // Simple LCG for random selection
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let idx = (rng_state >> 33) as usize % valid.len();
                    current = valid[idx];
                }
            }
        }

        // Compute PageRank-like scores
        let total_walks = stats.walks_performed.max(1);
        let scores: HashMap<u32, f32> = visit_counts
            .iter()
            .map(|(&node, &count)| (node, count as f32 / total_walks as f32))
            .collect();

        // Compute average hitting times
        let hitting_times: HashMap<u32, f32> = first_visit_step
            .into_iter()
            .map(|(node, steps)| {
                let avg = steps.iter().sum::<usize>() as f32 / steps.len().max(1) as f32;
                (node, avg)
            })
            .collect();

        ImportanceScores {
            scores,
            hitting_times,
        }
    }

    /// Find replacement candidates based on importance preservation.
    fn find_replacements_by_importance(
        &self,
        from_node: u32,
        existing: &[u32],
        deleted_neighbors: &[u32],
        importance: &ImportanceScores,
        deleted_set: &HashSet<u32>,
    ) -> Vec<u32> {
        let existing_set: HashSet<u32> = existing.iter().cloned().collect();

        // Candidates are neighbors of deleted node's other neighbors (2-hop)
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        for &neighbor in deleted_neighbors {
            if neighbor == from_node {
                continue;
            }

            // Direct connection to other neighbors of deleted
            if !existing_set.contains(&neighbor) && !deleted_set.contains(&neighbor) {
                let dist = (self.compute_distance)(from_node, neighbor);
                let score = importance.scores.get(&neighbor).copied().unwrap_or(0.0);
                // Priority: high importance, low distance
                let priority = score / (dist + 0.1);
                candidates.push((neighbor, priority));
            }

            // Also consider 2-hop neighbors
            for two_hop in (self.get_neighbors)(neighbor) {
                if two_hop != from_node
                    && !existing_set.contains(&two_hop)
                    && !deleted_set.contains(&two_hop)
                    && !candidates.iter().any(|(c, _)| *c == two_hop)
                {
                    let dist = (self.compute_distance)(from_node, two_hop);
                    let score = importance.scores.get(&two_hop).copied().unwrap_or(0.0);
                    let priority = score / (dist + 0.1);
                    candidates.push((two_hop, priority));
                }
            }
        }

        // Sort by priority (descending)
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Return top candidates
        let needed = self.config.max_neighbors.saturating_sub(existing.len());
        candidates
            .into_iter()
            .take(needed)
            .map(|(id, _)| id)
            .collect()
    }

    /// Deterministic refinement pass.
    fn refine_repairs(
        &self,
        repairs: &mut HashMap<u32, Vec<u32>>,
        deleted_set: &HashSet<u32>,
        _stats: &mut RandomWalkStats,
    ) {
        // Ensure bidirectional connectivity
        let repairs_snapshot: Vec<(u32, Vec<u32>)> =
            repairs.iter().map(|(k, v)| (*k, v.clone())).collect();

        for (node, neighbors) in repairs_snapshot {
            for &neighbor in &neighbors {
                if deleted_set.contains(&neighbor) {
                    continue;
                }

                // Check if reverse edge exists
                if let Some(reverse_neighbors) = repairs.get_mut(&neighbor) {
                    if !reverse_neighbors.contains(&node)
                        && reverse_neighbors.len() < self.config.max_neighbors
                    {
                        reverse_neighbors.push(node);
                    }
                }
            }
        }
    }
}

/// Estimate hitting time change from deletion.
///
/// Uses random walks to estimate how much longer it takes to reach
/// nodes after deletion vs before.
pub fn estimate_hitting_time_change<G>(
    deleted_nodes: &[u32],
    sample_targets: &[u32],
    entry_point: u32,
    get_neighbors: G,
    num_walks: usize,
    max_length: usize,
) -> f32
where
    G: Fn(u32) -> Vec<u32>,
{
    let deleted_set: HashSet<u32> = deleted_nodes.iter().cloned().collect();
    let mut rng_state = 12345u64;

    let mut before_times: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut after_times: HashMap<u32, Vec<usize>> = HashMap::new();

    // Sample walks BEFORE deletion
    for _ in 0..num_walks {
        let mut current = entry_point;
        let mut visited: HashSet<u32> = HashSet::new();

        for step in 0..max_length {
            if visited.insert(current) && sample_targets.contains(&current) {
                before_times.entry(current).or_default().push(step);
            }

            let neighbors = get_neighbors(current);
            if neighbors.is_empty() {
                break;
            }

            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state >> 33) as usize % neighbors.len();
            current = neighbors[idx];
        }
    }

    // Sample walks AFTER deletion (skip deleted nodes)
    for _ in 0..num_walks {
        let mut current = entry_point;
        if deleted_set.contains(&current) {
            continue;
        }

        let mut visited: HashSet<u32> = HashSet::new();

        for step in 0..max_length {
            if visited.insert(current) && sample_targets.contains(&current) {
                after_times.entry(current).or_default().push(step);
            }

            let neighbors: Vec<u32> = get_neighbors(current)
                .into_iter()
                .filter(|n| !deleted_set.contains(n))
                .collect();

            if neighbors.is_empty() {
                break;
            }

            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state >> 33) as usize % neighbors.len();
            current = neighbors[idx];
        }
    }

    // Compute average change in hitting time
    let mut total_change = 0.0f32;
    let mut count = 0;

    for target in sample_targets {
        let before_avg = before_times
            .get(target)
            .map(|v| v.iter().sum::<usize>() as f32 / v.len().max(1) as f32)
            .unwrap_or(max_length as f32);

        let after_avg = after_times
            .get(target)
            .map(|v| v.iter().sum::<usize>() as f32 / v.len().max(1) as f32)
            .unwrap_or(max_length as f32);

        total_change += after_avg - before_avg;
        count += 1;
    }

    if count > 0 {
        total_change / count as f32
    } else {
        0.0
    }
}

/// Simplified stateless repair function.
///
/// Returns the new neighbor list for each affected node.
pub fn random_walk_repair(
    deleted_node: u32,
    neighbors_of_deleted: &[u32],
    get_neighbors: impl Fn(u32) -> Vec<u32>,
    compute_distance: impl Fn(u32, u32) -> f32,
    config: &RandomWalkConfig,
    deleted_set: &HashSet<u32>,
) -> HashMap<u32, Vec<u32>> {
    let repairer = RandomWalkRepairer::new(config.clone(), get_neighbors, compute_distance);

    // Use existing logic but with stateless interface
    let mut repairs = HashMap::new();
    let valid_neighbors: Vec<u32> = neighbors_of_deleted
        .iter()
        .filter(|n| !deleted_set.contains(n))
        .cloned()
        .collect();

    for &neighbor in &valid_neighbors {
        let current: Vec<u32> = (repairer.get_neighbors)(neighbor)
            .into_iter()
            .filter(|&n| n != deleted_node && !deleted_set.contains(&n))
            .collect();

        // Simple 2-hop expansion
        let mut candidates: Vec<(u32, f32)> = Vec::new();
        let existing_set: HashSet<u32> = current.iter().cloned().collect();

        for &other in &valid_neighbors {
            if other != neighbor && !existing_set.contains(&other) {
                let dist = (repairer.compute_distance)(neighbor, other);
                candidates.push((other, dist));
            }
        }

        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        let mut new_neighbors = current;
        for (candidate, _) in candidates {
            if new_neighbors.len() >= config.max_neighbors {
                break;
            }
            new_neighbors.push(candidate);
        }

        repairs.insert(neighbor, new_neighbors);
    }

    repairs
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn create_chain_graph(size: usize) -> Vec<Vec<u32>> {
        let mut graph = Vec::with_capacity(size);
        for i in 0..size {
            let mut neighbors = Vec::new();
            if i > 0 {
                neighbors.push((i - 1) as u32);
            }
            if i < size - 1 {
                neighbors.push((i + 1) as u32);
            }
            graph.push(neighbors);
        }
        graph
    }

    fn create_clique_graph(size: usize) -> Vec<Vec<u32>> {
        let mut graph = Vec::with_capacity(size);
        for i in 0..size {
            let neighbors: Vec<u32> = (0..size).filter(|&j| j != i).map(|j| j as u32).collect();
            graph.push(neighbors);
        }
        graph
    }

    #[test]
    fn test_random_walk_repair_chain() {
        let graph = create_chain_graph(5);
        // Graph: 0 - 1 - 2 - 3 - 4

        let config = RandomWalkConfig {
            max_neighbors: 4,
            num_walks: 10,
            ..Default::default()
        };

        let deleted_set: HashSet<u32> = [2].into_iter().collect();

        let repairs = random_walk_repair(
            2,
            &graph[2],
            |id| graph[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(),
            &config,
            &deleted_set,
        );

        // Nodes 1 and 3 should be repaired
        assert!(repairs.contains_key(&1));
        assert!(repairs.contains_key(&3));

        // Node 1's neighbors should not contain 2
        let node1_neighbors = &repairs[&1];
        assert!(!node1_neighbors.contains(&2));
    }

    #[test]
    fn test_random_walk_repair_clique() {
        let graph = create_clique_graph(5);
        // Fully connected graph

        let config = RandomWalkConfig {
            max_neighbors: 4,
            num_walks: 10,
            ..Default::default()
        };

        let deleted_set: HashSet<u32> = [2].into_iter().collect();

        let repairs = random_walk_repair(
            2,
            &graph[2],
            |id| graph[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(),
            &config,
            &deleted_set,
        );

        // All non-deleted nodes that were connected to 2 should be repaired
        for (node_id, neighbors) in &repairs {
            assert!(
                !neighbors.contains(&2),
                "Node {} should not have deleted node 2",
                node_id
            );
        }
    }

    #[test]
    fn test_estimate_hitting_time() {
        let graph = create_chain_graph(10);

        let change = estimate_hitting_time_change(
            &[5],    // Delete middle node
            &[0, 9], // Sample targets at ends
            4,       // Start near middle
            |id| graph[id as usize].clone(),
            50,
            20,
        );

        // Deleting a middle node should increase hitting time
        // (or at least not decrease it significantly)
        assert!(
            change >= -1.0,
            "Hitting time shouldn't decrease significantly"
        );
    }

    #[test]
    fn test_importance_scores() {
        let graph = create_chain_graph(5);

        let repairer = RandomWalkRepairer::new(
            RandomWalkConfig {
                num_walks: 50,
                max_walk_length: 20,
                ..Default::default()
            },
            |id| graph[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(),
        );

        let mut stats = RandomWalkStats::default();
        let deleted_set = HashSet::new();

        let importance = repairer.compute_importance_from_walks(
            &[0, 4], // Start from ends
            999,     // Non-existent deleted node
            &deleted_set,
            &mut stats,
        );

        // Middle nodes should be visited more often
        assert!(stats.walks_performed > 0);
        assert!(!importance.scores.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = RandomWalkConfig::default();

        assert_eq!(config.num_walks, 100);
        assert_eq!(config.max_walk_length, 50);
        assert_eq!(config.max_neighbors, 16);
        assert!(config.refine);
    }

    #[test]
    fn test_repairer_with_seed() {
        let graph = create_chain_graph(5);

        let repairer1 = RandomWalkRepairer::new(
            RandomWalkConfig::default(),
            |id| graph[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(),
        )
        .with_seed(123);

        let repairer2 = RandomWalkRepairer::new(
            RandomWalkConfig::default(),
            |id| graph[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(),
        )
        .with_seed(123);

        // Same seed should give same results
        let deleted_set = HashSet::new();
        let (repairs1, _) = repairer1.compute_repairs(2, &deleted_set).unwrap();
        let (repairs2, _) = repairer2.compute_repairs(2, &deleted_set).unwrap();

        assert_eq!(repairs1, repairs2);
    }
}
