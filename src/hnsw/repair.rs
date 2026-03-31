//! MN-RU (Missing Neighbor Repair Update) algorithm for HNSW graph repair.
//!
//! When nodes are deleted from an HNSW graph, edges pointing to deleted nodes
//! become invalid, potentially causing:
//! - Reduced recall (can't reach certain regions)
//! - Disconnected subgraphs
//! - Query failures
//!
//! MN-RU repairs the graph by reconnecting orphaned edges to nearby valid nodes.
//!
//! # Algorithm
//!
//! For each neighbor N of deleted node D:
//! 1. Remove D from N's neighbor list
//! 2. Find replacement candidates via search from N
//! 3. Add best non-redundant candidates to N's neighbors
//! 4. Optionally add bidirectional edges for better connectivity
//!
//! # References
//!
//! - omendb: mark_deleted with MN-RU repair
//! - FreshDiskANN: StreamingMerge delete handling

use crate::RetrieveError;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for graph repair operations.
#[derive(Clone, Debug)]
pub struct RepairConfig {
    /// Maximum candidates to consider for replacement
    pub max_candidates: usize,
    /// Maximum neighbors per node (M parameter)
    pub max_neighbors: usize,
    /// Enable bidirectional edge repair
    pub bidirectional: bool,
    /// Alpha for diversity pruning (1.0 = no pruning, higher = more diverse)
    pub alpha: f32,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_candidates: 64,
            max_neighbors: 16,
            bidirectional: true,
            alpha: 1.2,
        }
    }
}

/// Statistics from a repair operation.
#[derive(Clone, Debug, Default)]
pub struct RepairStats {
    /// Nodes processed
    pub nodes_processed: usize,
    /// Edges removed (to deleted nodes)
    pub edges_removed: usize,
    /// Edges added (replacements)
    pub edges_added: usize,
    /// Bidirectional edges added
    pub bidirectional_edges: usize,
}

/// Graph repair engine for HNSW indices.
pub struct GraphRepairer<'a> {
    config: RepairConfig,
    /// Deleted node IDs
    deleted: HashSet<u32>,
    /// Get neighbors of a node
    get_neighbors: Box<dyn Fn(u32) -> Vec<u32> + 'a>,
    /// Compute distance between two nodes
    compute_distance: Box<dyn Fn(u32, u32) -> f32 + 'a>,
    /// Set neighbors for a node
    set_neighbors: Box<dyn FnMut(u32, Vec<u32>) + 'a>,
}

impl<'a> GraphRepairer<'a> {
    /// Create new graph repairer.
    pub fn new<G, D, S>(
        config: RepairConfig,
        get_neighbors: G,
        compute_distance: D,
        set_neighbors: S,
    ) -> Self
    where
        G: Fn(u32) -> Vec<u32> + 'a,
        D: Fn(u32, u32) -> f32 + 'a,
        S: FnMut(u32, Vec<u32>) + 'a,
    {
        Self {
            config,
            deleted: HashSet::new(),
            get_neighbors: Box::new(get_neighbors),
            compute_distance: Box::new(compute_distance),
            set_neighbors: Box::new(set_neighbors),
        }
    }

    /// Mark a node as deleted and repair its neighbors.
    pub fn mark_deleted(&mut self, node_id: u32) -> Result<RepairStats, RetrieveError> {
        self.deleted.insert(node_id);

        let mut stats = RepairStats::default();

        // Get neighbors of the deleted node (these need repair)
        let neighbors_to_repair = (self.get_neighbors)(node_id);

        for &neighbor in &neighbors_to_repair {
            if self.deleted.contains(&neighbor) {
                continue;
            }

            let repair_result = self.repair_single_node(neighbor, node_id)?;
            stats.nodes_processed += 1;
            stats.edges_removed += repair_result.edges_removed;
            stats.edges_added += repair_result.edges_added;
            stats.bidirectional_edges += repair_result.bidirectional_edges;
        }

        Ok(stats)
    }

    /// Batch mark multiple nodes as deleted.
    pub fn mark_deleted_batch(&mut self, node_ids: &[u32]) -> Result<RepairStats, RetrieveError> {
        // First mark all as deleted
        for &id in node_ids {
            self.deleted.insert(id);
        }

        // Collect all neighbors that need repair
        let mut neighbors_to_repair: HashSet<u32> = HashSet::new();
        for &id in node_ids {
            for neighbor in (self.get_neighbors)(id) {
                if !self.deleted.contains(&neighbor) {
                    neighbors_to_repair.insert(neighbor);
                }
            }
        }

        // Repair each affected neighbor
        let mut stats = RepairStats::default();
        for neighbor in neighbors_to_repair {
            let repair_result = self.repair_node_full(neighbor)?;
            stats.nodes_processed += 1;
            stats.edges_removed += repair_result.edges_removed;
            stats.edges_added += repair_result.edges_added;
            stats.bidirectional_edges += repair_result.bidirectional_edges;
        }

        Ok(stats)
    }

    /// Repair a single node after one of its neighbors was deleted.
    fn repair_single_node(
        &mut self,
        node_id: u32,
        deleted_neighbor: u32,
    ) -> Result<RepairStats, RetrieveError> {
        let mut stats = RepairStats::default();

        // Get current neighbors, remove deleted one
        let mut neighbors: Vec<u32> = (self.get_neighbors)(node_id)
            .into_iter()
            .filter(|&n| n != deleted_neighbor && !self.deleted.contains(&n))
            .collect();
        stats.edges_removed = 1;

        // If we're under capacity, find replacements
        if neighbors.len() < self.config.max_neighbors {
            let needed = self.config.max_neighbors - neighbors.len();
            let candidates = self.find_replacement_candidates(node_id, &neighbors, needed)?;

            for candidate in candidates {
                if neighbors.len() >= self.config.max_neighbors {
                    break;
                }
                neighbors.push(candidate);
                stats.edges_added += 1;

                // Add bidirectional edge
                if self.config.bidirectional {
                    self.add_bidirectional_edge(candidate, node_id)?;
                    stats.bidirectional_edges += 1;
                }
            }
        }

        (self.set_neighbors)(node_id, neighbors);

        Ok(stats)
    }

    /// Full repair of a node (remove all deleted neighbors, find replacements).
    fn repair_node_full(&mut self, node_id: u32) -> Result<RepairStats, RetrieveError> {
        let mut stats = RepairStats::default();

        let original = (self.get_neighbors)(node_id);
        let mut neighbors: Vec<u32> = original
            .into_iter()
            .filter(|n| !self.deleted.contains(n))
            .collect();

        let removed = (self.get_neighbors)(node_id).len() - neighbors.len();
        stats.edges_removed = removed;

        if neighbors.len() < self.config.max_neighbors {
            let needed = self.config.max_neighbors - neighbors.len();
            let candidates = self.find_replacement_candidates(node_id, &neighbors, needed)?;

            for candidate in candidates {
                if neighbors.len() >= self.config.max_neighbors {
                    break;
                }
                neighbors.push(candidate);
                stats.edges_added += 1;

                if self.config.bidirectional {
                    self.add_bidirectional_edge(candidate, node_id)?;
                    stats.bidirectional_edges += 1;
                }
            }
        }

        (self.set_neighbors)(node_id, neighbors);

        Ok(stats)
    }

    /// Find replacement candidates via local search.
    fn find_replacement_candidates(
        &self,
        from_node: u32,
        existing_neighbors: &[u32],
        needed: usize,
    ) -> Result<Vec<u32>, RetrieveError> {
        // BFS from existing neighbors to find candidates
        let mut visited: HashSet<u32> = HashSet::new();
        visited.insert(from_node);
        visited.extend(existing_neighbors.iter().cloned());
        visited.extend(self.deleted.iter().cloned());

        #[derive(Clone, Copy)]
        struct Candidate {
            id: u32,
            dist: f32,
        }

        impl PartialEq for Candidate {
            fn eq(&self, other: &Self) -> bool {
                self.dist == other.dist
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
                // Min-heap: smaller distance = higher priority
                // Use total_cmp for IEEE 754 total ordering (NaN-safe)
                self.dist.total_cmp(&other.dist).reverse()
            }
        }

        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();

        // Explore from existing neighbors
        for &neighbor in existing_neighbors {
            for two_hop in (self.get_neighbors)(neighbor) {
                if visited.insert(two_hop) {
                    let dist = (self.compute_distance)(from_node, two_hop);
                    candidates.push(Candidate { id: two_hop, dist });

                    if candidates.len() > self.config.max_candidates {
                        // Trim to keep best candidates
                        let mut temp: Vec<_> = candidates.drain().collect();
                        temp.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
                        temp.truncate(self.config.max_candidates / 2);
                        for c in temp {
                            candidates.push(c);
                        }
                    }
                }
            }
        }

        // Select best non-redundant candidates with alpha-pruning
        let mut selected = Vec::new();
        let sorted: Vec<_> = candidates.into_sorted_vec();

        'outer: for candidate in sorted.iter().rev() {
            // Skip if would be redundant (too close to existing neighbor)
            for &existing in existing_neighbors.iter().chain(selected.iter()) {
                let dist_to_existing = (self.compute_distance)(candidate.id, existing);
                if dist_to_existing < candidate.dist * self.config.alpha {
                    continue 'outer; // Redundant
                }
            }

            selected.push(candidate.id);
            if selected.len() >= needed {
                break;
            }
        }

        Ok(selected)
    }

    /// Add bidirectional edge (add from_node to to_node's neighbors if room).
    fn add_bidirectional_edge(
        &mut self,
        to_node: u32,
        from_node: u32,
    ) -> Result<(), RetrieveError> {
        let mut neighbors = (self.get_neighbors)(to_node);

        // Skip if already connected or at capacity
        if neighbors.contains(&from_node) || neighbors.len() >= self.config.max_neighbors {
            return Ok(());
        }

        neighbors.push(from_node);
        (self.set_neighbors)(to_node, neighbors);

        Ok(())
    }

    /// Check if a node is deleted.
    pub fn is_deleted(&self, node_id: u32) -> bool {
        self.deleted.contains(&node_id)
    }

    /// Count deleted nodes.
    pub fn deleted_count(&self) -> usize {
        self.deleted.len()
    }
}

/// Simplified graph repair without mutable closures.
///
/// This is a standalone function that returns the repair operations
/// to be applied by the caller.
pub fn compute_repair_operations(
    deleted_node: u32,
    neighbors_of_deleted: &[u32],
    get_neighbors: impl Fn(u32) -> Vec<u32>,
    compute_distance: impl Fn(u32, u32) -> f32,
    config: &RepairConfig,
    deleted_set: &HashSet<u32>,
) -> HashMap<u32, Vec<u32>> {
    let mut operations: HashMap<u32, Vec<u32>> = HashMap::new();

    for &neighbor in neighbors_of_deleted {
        if deleted_set.contains(&neighbor) {
            continue;
        }

        // Get current neighbors, remove deleted
        let current: Vec<u32> = get_neighbors(neighbor)
            .into_iter()
            .filter(|&n| n != deleted_node && !deleted_set.contains(&n))
            .collect();

        if current.len() >= config.max_neighbors {
            operations.insert(neighbor, current);
            continue;
        }

        // Find replacements via 2-hop
        let mut candidates: Vec<(u32, f32)> = Vec::new();
        let mut visited: HashSet<u32> = current.iter().cloned().collect();
        visited.insert(neighbor);
        visited.extend(deleted_set.iter().cloned());

        for &n in &current {
            for two_hop in get_neighbors(n) {
                if visited.insert(two_hop) {
                    let dist = compute_distance(neighbor, two_hop);
                    candidates.push((two_hop, dist));
                }
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

        operations.insert(neighbor, new_neighbors);
    }

    operations
}

/// Validate graph connectivity after deletions.
///
/// Returns (reachable_count, orphan_count) from a BFS starting at entry_point.
pub fn validate_connectivity(
    entry_point: u32,
    total_nodes: usize,
    get_neighbors: impl Fn(u32) -> Vec<u32>,
    is_deleted: impl Fn(u32) -> bool,
) -> (usize, usize) {
    let mut visited: HashSet<u32> = HashSet::new();
    let mut queue: Vec<u32> = vec![entry_point];

    while let Some(node) = queue.pop() {
        if is_deleted(node) || !visited.insert(node) {
            continue;
        }

        for neighbor in get_neighbors(node) {
            if !visited.contains(&neighbor) && !is_deleted(neighbor) {
                queue.push(neighbor);
            }
        }
    }

    let reachable = visited.len();
    let expected_valid = total_nodes - (0..total_nodes as u32).filter(|&n| is_deleted(n)).count();
    let orphans = expected_valid.saturating_sub(reachable);

    (reachable, orphans)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn test_repair_config_default() {
        let config = RepairConfig::default();
        assert_eq!(config.max_candidates, 64);
        assert_eq!(config.max_neighbors, 16);
        assert!(config.bidirectional);
    }

    #[test]
    fn test_compute_repair_operations() {
        // Simple 5-node graph: 0 - 1 - 2 - 3 - 4
        let adjacency: Vec<Vec<u32>> = vec![
            vec![1],    // 0
            vec![0, 2], // 1
            vec![1, 3], // 2
            vec![2, 4], // 3
            vec![3],    // 4
        ];

        let config = RepairConfig {
            max_candidates: 10,
            max_neighbors: 4,
            bidirectional: true,
            alpha: 1.0,
        };

        // Delete node 2
        let deleted_set: HashSet<u32> = [2].into_iter().collect();
        let neighbors_of_deleted = &adjacency[2];

        let ops = compute_repair_operations(
            2,
            neighbors_of_deleted,
            |id| adjacency[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(), // Simple distance
            &config,
            &deleted_set,
        );

        // Node 1 should have been repaired
        assert!(ops.contains_key(&1));
        let node1_new = &ops[&1];
        // Should not contain deleted node 2
        assert!(!node1_new.contains(&2));
    }

    #[test]
    fn test_validate_connectivity() {
        let adjacency: Vec<Vec<u32>> = vec![
            vec![1, 2],    // 0
            vec![0, 2, 3], // 1
            vec![0, 1, 3], // 2
            vec![1, 2],    // 3
        ];

        let (reachable, orphans) = validate_connectivity(
            0,
            4,
            |id| adjacency[id as usize].clone(),
            |_| false, // Nothing deleted
        );

        assert_eq!(reachable, 4);
        assert_eq!(orphans, 0);
    }

    #[test]
    fn test_validate_connectivity_with_deletion() {
        // Graph with a bridge node
        let adjacency: Vec<Vec<u32>> = vec![
            vec![1],    // 0
            vec![0, 2], // 1 - bridge
            vec![1, 3], // 2
            vec![2],    // 3
        ];

        // Delete node 1 (bridge)
        let (reachable, orphans) =
            validate_connectivity(0, 4, |id| adjacency[id as usize].clone(), |id| id == 1);

        // Only node 0 reachable, nodes 2 and 3 orphaned
        assert_eq!(reachable, 1);
        assert_eq!(orphans, 2);
    }

    #[test]
    fn test_graph_repairer() {
        let adjacency = RefCell::new(vec![
            vec![1, 2],    // 0
            vec![0, 2, 3], // 1
            vec![0, 1, 3], // 2
            vec![1, 2],    // 3
        ]);

        let config = RepairConfig {
            max_candidates: 10,
            max_neighbors: 4,
            bidirectional: true,
            alpha: 1.0,
        };

        let mut repairer = GraphRepairer::new(
            config,
            |id| adjacency.borrow()[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(),
            |id, neighbors| {
                adjacency.borrow_mut()[id as usize] = neighbors;
            },
        );

        // Delete node 1
        let stats = repairer.mark_deleted(1).unwrap();

        // Should have processed nodes 0, 2, 3
        assert!(stats.nodes_processed > 0);
        assert!(stats.edges_removed > 0);

        // Node 1 should be marked deleted
        assert!(repairer.is_deleted(1));
    }

    #[test]
    fn test_batch_deletion() {
        let adjacency = RefCell::new(vec![
            vec![1, 2],       // 0
            vec![0, 2, 3, 4], // 1
            vec![0, 1, 3],    // 2
            vec![1, 2, 4],    // 3
            vec![1, 3],       // 4
        ]);

        let config = RepairConfig::default();

        let mut repairer = GraphRepairer::new(
            config,
            |id| adjacency.borrow()[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(),
            |id, neighbors| {
                adjacency.borrow_mut()[id as usize] = neighbors;
            },
        );

        // Delete nodes 1 and 3
        let stats = repairer.mark_deleted_batch(&[1, 3]).unwrap();

        assert!(repairer.is_deleted(1));
        assert!(repairer.is_deleted(3));
        assert_eq!(repairer.deleted_count(), 2);
        assert!(stats.nodes_processed > 0);
    }
}
