//! HNSW Index Merging Algorithms
//!
//! Implements algorithms for merging separate HNSW graphs, essential for:
//! - Distributed indexing (merge shards)
//! - Incremental indexing (merge incremental updates)
//! - Database compaction
//!
//! References:
//! - Ponomarenko (2025): "Three Algorithms for Merging HNSW Graphs"
//!
//! Algorithms:
//! - NGM (Naive Graph Merge): Simple edge union
//! - IGTM (Intra Graph Traversal Merge): Traverse within each graph
//! - CGTM (Cross Graph Traversal Merge): Traverse across both graphs

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for HNSW merging.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Maximum edges per node at layer 0.
    pub max_edges_l0: usize,
    /// Maximum edges per node at higher layers.
    pub max_edges: usize,
    /// ef_construction parameter for refinement.
    pub ef_construction: usize,
    /// Whether to perform edge refinement after merge.
    pub refine_edges: bool,
    /// Alpha for neighbor selection (diversity pruning).
    pub alpha: f32,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            max_edges_l0: 32,
            max_edges: 16,
            ef_construction: 100,
            refine_edges: true,
            alpha: 1.0,
        }
    }
}

/// A simple graph node for merging.
#[derive(Debug, Clone)]
pub struct MergeNode {
    /// Node ID.
    pub id: u32,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Neighbors at each layer.
    pub neighbors: Vec<Vec<u32>>,
    /// Maximum layer this node appears in.
    pub max_layer: usize,
}

/// Graph structure for merging.
#[derive(Debug)]
pub struct MergeGraph {
    /// All nodes.
    nodes: HashMap<u32, MergeNode>,
    /// Entry point ID.
    entry_point: Option<u32>,
    /// Maximum layer in the graph.
    max_layer: usize,
    /// Configuration.
    config: MergeConfig,
}

impl MergeGraph {
    /// Create a new empty graph.
    pub fn new(config: MergeConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            config,
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: MergeNode) {
        if node.max_layer > self.max_layer {
            self.max_layer = node.max_layer;
        }
        let should_update_entry = match self.entry_point {
            None => true,
            Some(ep) => {
                let current_max = self.nodes.get(&ep).map(|n| n.max_layer).unwrap_or(0);
                node.max_layer > current_max
            }
        };
        if should_update_entry {
            self.entry_point = Some(node.id);
        }
        self.nodes.insert(node.id, node);
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: u32) -> Option<&MergeNode> {
        self.nodes.get(&id)
    }

    /// Get mutable node by ID.
    pub fn get_node_mut(&mut self, id: u32) -> Option<&mut MergeNode> {
        self.nodes.get_mut(&id)
    }

    /// Get all node IDs.
    pub fn node_ids(&self) -> impl Iterator<Item = &u32> {
        self.nodes.keys()
    }

    /// Get entry point.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Get number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get max layer.
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }
}

/// Statistics from merge operation.
#[derive(Debug, Default, Clone)]
pub struct MergeStats {
    /// Nodes from graph A.
    pub nodes_from_a: usize,
    /// Nodes from graph B.
    pub nodes_from_b: usize,
    /// Total nodes in merged graph.
    pub total_nodes: usize,
    /// Edges added during merge.
    pub edges_added: usize,
    /// Edges pruned during refinement.
    pub edges_pruned: usize,
    /// Distance computations.
    pub distance_computations: u64,
}

/// Naive Graph Merge (NGM).
///
/// Simply unions the edge lists from both graphs.
/// Fast but may exceed edge limits.
pub fn naive_graph_merge(
    graph_a: &MergeGraph,
    graph_b: &MergeGraph,
    config: &MergeConfig,
) -> (MergeGraph, MergeStats) {
    let mut stats = MergeStats::default();
    let mut merged = MergeGraph::new(config.clone());

    // Add all nodes from graph A
    for node in graph_a.nodes.values() {
        let mut new_node = node.clone();
        // Ensure neighbor lists are properly sized
        while new_node.neighbors.len() <= new_node.max_layer {
            new_node.neighbors.push(Vec::new());
        }
        merged.add_node(new_node);
        stats.nodes_from_a += 1;
    }

    // Add all nodes from graph B (may overlap)
    for node in graph_b.nodes.values() {
        if let Some(existing) = merged.nodes.get_mut(&node.id) {
            // Node exists in both - merge neighbor lists
            for (layer, neighbors) in node.neighbors.iter().enumerate() {
                if layer < existing.neighbors.len() {
                    for &neighbor in neighbors {
                        if !existing.neighbors[layer].contains(&neighbor) {
                            existing.neighbors[layer].push(neighbor);
                            stats.edges_added += 1;
                        }
                    }
                }
            }
        } else {
            merged.add_node(node.clone());
            stats.nodes_from_b += 1;
        }
    }

    // Prune edges if exceeding limits
    if config.refine_edges {
        for node in merged.nodes.values_mut() {
            for (layer, neighbors) in node.neighbors.iter_mut().enumerate() {
                let max_edges = if layer == 0 {
                    config.max_edges_l0
                } else {
                    config.max_edges
                };
                if neighbors.len() > max_edges {
                    stats.edges_pruned += neighbors.len() - max_edges;
                    neighbors.truncate(max_edges);
                }
            }
        }
    }

    stats.total_nodes = merged.len();
    (merged, stats)
}

/// Candidate for search during merge.
#[derive(Debug, Clone)]
struct MergeCandidate {
    id: u32,
    distance: f32,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

/// Intra Graph Traversal Merge (IGTM).
///
/// For each node in A, find nearest neighbors in B using B's graph structure.
/// More accurate than NGM but still limited to single graph traversal.
pub fn intra_graph_traversal_merge(
    graph_a: &MergeGraph,
    graph_b: &MergeGraph,
    config: &MergeConfig,
) -> (MergeGraph, MergeStats) {
    let mut stats = MergeStats::default();

    // Start with NGM as base
    let (base_merged, base_stats) = naive_graph_merge(graph_a, graph_b, config);
    let mut merged = base_merged;
    stats.nodes_from_a = base_stats.nodes_from_a;
    stats.nodes_from_b = base_stats.nodes_from_b;
    stats.edges_added = base_stats.edges_added;

    // For each node in A, search in B's graph
    if let Some(entry_b) = graph_b.entry_point() {
        for node_a in graph_a.nodes.values() {
            let neighbors_from_b = search_in_graph(
                &node_a.vector,
                graph_b,
                entry_b,
                config.max_edges_l0,
                &mut stats.distance_computations,
            );

            // Add edges to merged graph
            if let Some(merged_node) = merged.get_node_mut(node_a.id) {
                if !merged_node.neighbors.is_empty() {
                    for (neighbor_id, _dist) in neighbors_from_b {
                        if !merged_node.neighbors[0].contains(&neighbor_id) {
                            merged_node.neighbors[0].push(neighbor_id);
                            stats.edges_added += 1;
                        }
                    }
                }
            }
        }
    }

    // For each node in B, search in A's graph
    if let Some(entry_a) = graph_a.entry_point() {
        for node_b in graph_b.nodes.values() {
            let neighbors_from_a = search_in_graph(
                &node_b.vector,
                graph_a,
                entry_a,
                config.max_edges_l0,
                &mut stats.distance_computations,
            );

            if let Some(merged_node) = merged.get_node_mut(node_b.id) {
                if !merged_node.neighbors.is_empty() {
                    for (neighbor_id, _dist) in neighbors_from_a {
                        if !merged_node.neighbors[0].contains(&neighbor_id) {
                            merged_node.neighbors[0].push(neighbor_id);
                            stats.edges_added += 1;
                        }
                    }
                }
            }
        }
    }

    // Refine edges using select_neighbors
    if config.refine_edges {
        refine_all_edges(&mut merged, config, &mut stats);
    }

    stats.total_nodes = merged.len();
    (merged, stats)
}

/// Cross Graph Traversal Merge (CGTM).
///
/// Most accurate: uses both graphs simultaneously during neighbor search.
/// For each node, searches in combined graph space.
pub fn cross_graph_traversal_merge(
    graph_a: &MergeGraph,
    graph_b: &MergeGraph,
    config: &MergeConfig,
) -> (MergeGraph, MergeStats) {
    let mut stats = MergeStats::default();
    let mut merged = MergeGraph::new(config.clone());

    // Collect all nodes
    let mut all_nodes: HashMap<u32, MergeNode> = HashMap::new();

    for (id, node) in &graph_a.nodes {
        all_nodes.insert(*id, node.clone());
        stats.nodes_from_a += 1;
    }

    for (id, node) in &graph_b.nodes {
        if !all_nodes.contains_key(id) {
            all_nodes.insert(*id, node.clone());
            stats.nodes_from_b += 1;
        }
    }

    // For each node, find neighbors using cross-graph search
    let all_vectors: Vec<(u32, Vec<f32>)> = all_nodes
        .iter()
        .map(|(id, node)| (*id, node.vector.clone()))
        .collect();

    for (id, node) in &all_nodes {
        let neighbors = cross_graph_search(
            &node.vector,
            *id,
            graph_a,
            graph_b,
            config.ef_construction,
            &mut stats.distance_computations,
        );

        // Select best neighbors
        let selected = select_neighbors_heuristic(
            &node.vector,
            &neighbors,
            &all_vectors,
            if node.max_layer == 0 {
                config.max_edges_l0
            } else {
                config.max_edges
            },
            config.alpha,
            &mut stats.distance_computations,
        );

        let mut new_node = node.clone();
        if !new_node.neighbors.is_empty() {
            new_node.neighbors[0] = selected.into_iter().map(|(id, _)| id).collect();
        }

        merged.add_node(new_node);
        stats.edges_added += merged
            .get_node(*id)
            .map(|n| n.neighbors.first().map(|v| v.len()).unwrap_or(0))
            .unwrap_or(0);
    }

    stats.total_nodes = merged.len();
    (merged, stats)
}

/// Search in a single graph.
fn search_in_graph(
    query: &[f32],
    graph: &MergeGraph,
    entry: u32,
    k: usize,
    dist_count: &mut u64,
) -> Vec<(u32, f32)> {
    let mut visited: HashSet<u32> = HashSet::new();
    let mut candidates: BinaryHeap<MergeCandidate> = BinaryHeap::new();
    let mut results: Vec<(u32, f32)> = Vec::new();

    if let Some(entry_node) = graph.get_node(entry) {
        *dist_count += 1;
        let entry_dist = euclidean_distance(query, &entry_node.vector);
        candidates.push(MergeCandidate {
            id: entry,
            distance: entry_dist,
        });
        visited.insert(entry);
    }

    while let Some(current) = candidates.pop() {
        results.push((current.id, current.distance));

        if let Some(node) = graph.get_node(current.id) {
            if !node.neighbors.is_empty() {
                for &neighbor_id in &node.neighbors[0] {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);
                        if let Some(neighbor_node) = graph.get_node(neighbor_id) {
                            *dist_count += 1;
                            let dist = euclidean_distance(query, &neighbor_node.vector);
                            candidates.push(MergeCandidate {
                                id: neighbor_id,
                                distance: dist,
                            });
                        }
                    }
                }
            }
        }

        if results.len() >= k * 2 {
            break;
        }
    }

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(k);
    results
}

/// Search across both graphs simultaneously.
fn cross_graph_search(
    query: &[f32],
    exclude_id: u32,
    graph_a: &MergeGraph,
    graph_b: &MergeGraph,
    ef: usize,
    dist_count: &mut u64,
) -> Vec<(u32, f32)> {
    let mut visited: HashSet<u32> = HashSet::new();
    let mut candidates: BinaryHeap<MergeCandidate> = BinaryHeap::new();
    let mut results: Vec<(u32, f32)> = Vec::new();

    visited.insert(exclude_id);

    // Initialize with entry points from both graphs
    if let Some(entry_a) = graph_a.entry_point() {
        if let Some(node) = graph_a.get_node(entry_a) {
            if entry_a != exclude_id {
                *dist_count += 1;
                let dist = euclidean_distance(query, &node.vector);
                candidates.push(MergeCandidate {
                    id: entry_a,
                    distance: dist,
                });
                visited.insert(entry_a);
            }
        }
    }

    if let Some(entry_b) = graph_b.entry_point() {
        if let Some(node) = graph_b.get_node(entry_b) {
            if entry_b != exclude_id && !visited.contains(&entry_b) {
                *dist_count += 1;
                let dist = euclidean_distance(query, &node.vector);
                candidates.push(MergeCandidate {
                    id: entry_b,
                    distance: dist,
                });
                visited.insert(entry_b);
            }
        }
    }

    while let Some(current) = candidates.pop() {
        results.push((current.id, current.distance));

        // Get neighbors from both graphs
        let mut neighbor_ids: Vec<u32> = Vec::new();

        if let Some(node) = graph_a.get_node(current.id) {
            if !node.neighbors.is_empty() {
                neighbor_ids.extend(&node.neighbors[0]);
            }
        }

        if let Some(node) = graph_b.get_node(current.id) {
            if !node.neighbors.is_empty() {
                for &n in &node.neighbors[0] {
                    if !neighbor_ids.contains(&n) {
                        neighbor_ids.push(n);
                    }
                }
            }
        }

        for neighbor_id in neighbor_ids {
            if neighbor_id != exclude_id && !visited.contains(&neighbor_id) {
                visited.insert(neighbor_id);

                // Try to get vector from either graph
                let vector = graph_a
                    .get_node(neighbor_id)
                    .or_else(|| graph_b.get_node(neighbor_id))
                    .map(|n| &n.vector);

                if let Some(v) = vector {
                    *dist_count += 1;
                    let dist = euclidean_distance(query, v);
                    candidates.push(MergeCandidate {
                        id: neighbor_id,
                        distance: dist,
                    });
                }
            }
        }

        if results.len() >= ef {
            break;
        }
    }

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results
}

/// Select neighbors using diversity-promoting heuristic.
fn select_neighbors_heuristic(
    _query: &[f32],
    candidates: &[(u32, f32)],
    all_vectors: &[(u32, Vec<f32>)],
    max_neighbors: usize,
    alpha: f32,
    dist_count: &mut u64,
) -> Vec<(u32, f32)> {
    if candidates.len() <= max_neighbors {
        return candidates.to_vec();
    }

    let vectors_map: HashMap<u32, &Vec<f32>> = all_vectors.iter().map(|(id, v)| (*id, v)).collect();

    let mut selected: Vec<(u32, f32)> = Vec::new();
    let mut remaining: Vec<(u32, f32)> = candidates.to_vec();

    while selected.len() < max_neighbors && !remaining.is_empty() {
        // Find candidate with best score considering diversity
        let mut best_idx = 0;
        let mut best_score = f32::INFINITY;

        for (i, (cand_id, cand_dist)) in remaining.iter().enumerate() {
            let mut score = *cand_dist;

            // Penalize if too close to already selected
            for (sel_id, _) in &selected {
                if let (Some(cand_vec), Some(sel_vec)) =
                    (vectors_map.get(cand_id), vectors_map.get(sel_id))
                {
                    *dist_count += 1;
                    let inter_dist = euclidean_distance(cand_vec, sel_vec);
                    if inter_dist < *cand_dist * alpha {
                        score += *cand_dist; // Penalty
                    }
                }
            }

            if score < best_score {
                best_score = score;
                best_idx = i;
            }
        }

        selected.push(remaining.remove(best_idx));
    }

    selected
}

/// Refine all edges in merged graph.
fn refine_all_edges(graph: &mut MergeGraph, config: &MergeConfig, stats: &mut MergeStats) {
    let all_vectors: Vec<(u32, Vec<f32>)> = graph
        .nodes
        .iter()
        .map(|(id, node)| (*id, node.vector.clone()))
        .collect();

    let node_ids: Vec<u32> = graph.nodes.keys().copied().collect();

    for id in node_ids {
        let Some(node) = graph.nodes.get(&id) else {
            continue;
        };
        let (query, candidates) = {
            let query = node.vector.clone();
            let candidates: Vec<(u32, f32)> = node
                .neighbors
                .first()
                .map(|neighbors| {
                    neighbors
                        .iter()
                        .filter_map(|&n_id| {
                            graph.get_node(n_id).map(|n| {
                                stats.distance_computations += 1;
                                (n_id, euclidean_distance(&query, &n.vector))
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            (query, candidates)
        };

        let max_edges = config.max_edges_l0;
        let selected = select_neighbors_heuristic(
            &query,
            &candidates,
            &all_vectors,
            max_edges,
            config.alpha,
            &mut stats.distance_computations,
        );

        let original_count = graph
            .nodes
            .get(&id)
            .and_then(|n| n.neighbors.first())
            .map(|v| v.len())
            .unwrap_or(0);

        if let Some(node) = graph.nodes.get_mut(&id) {
            if !node.neighbors.is_empty() {
                node.neighbors[0] = selected.into_iter().map(|(id, _)| id).collect();
                let new_count = node.neighbors[0].len();
                if original_count > new_count {
                    stats.edges_pruned += original_count - new_count;
                }
            }
        }
    }
}

use crate::distance::l2_distance as euclidean_distance;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| ((seed as f32 * 0.1 + i as f32) * 0.01).sin())
            .collect()
    }

    fn make_test_graph(start_id: u32, count: usize, dim: usize) -> MergeGraph {
        let config = MergeConfig::default();
        let mut graph = MergeGraph::new(config);

        for i in 0..count {
            let id = start_id + i as u32;
            let neighbors = if i > 0 {
                vec![vec![start_id + (i as u32 - 1)]]
            } else {
                vec![vec![]]
            };

            graph.add_node(MergeNode {
                id,
                vector: make_vector(dim, id),
                neighbors,
                max_layer: 0,
            });
        }

        graph
    }

    #[test]
    fn test_naive_merge() {
        let graph_a = make_test_graph(0, 10, 64);
        let graph_b = make_test_graph(10, 10, 64);
        let config = MergeConfig::default();

        let (merged, stats) = naive_graph_merge(&graph_a, &graph_b, &config);

        assert_eq!(merged.len(), 20);
        assert_eq!(stats.nodes_from_a, 10);
        assert_eq!(stats.nodes_from_b, 10);
    }

    #[test]
    fn test_naive_merge_overlapping() {
        let graph_a = make_test_graph(0, 10, 64);
        let graph_b = make_test_graph(5, 10, 64); // Overlaps IDs 5-9
        let config = MergeConfig::default();

        let (merged, _stats) = naive_graph_merge(&graph_a, &graph_b, &config);

        // Should have 15 unique nodes (0-14)
        assert_eq!(merged.len(), 15);
    }

    #[test]
    fn test_igtm_merge() {
        let graph_a = make_test_graph(0, 10, 64);
        let graph_b = make_test_graph(10, 10, 64);
        let config = MergeConfig::default();

        let (merged, stats) = intra_graph_traversal_merge(&graph_a, &graph_b, &config);

        assert_eq!(merged.len(), 20);
        assert!(stats.distance_computations > 0);
    }

    #[test]
    fn test_cgtm_merge() {
        let graph_a = make_test_graph(0, 10, 64);
        let graph_b = make_test_graph(10, 10, 64);
        let config = MergeConfig::default();

        let (merged, stats) = cross_graph_traversal_merge(&graph_a, &graph_b, &config);

        assert_eq!(merged.len(), 20);
        assert!(stats.distance_computations > 0);
    }

    #[test]
    fn test_select_neighbors_diversity() {
        let query = make_vector(64, 0);
        let candidates: Vec<(u32, f32)> = (1..20).map(|i| (i, i as f32 * 0.1)).collect();

        let all_vectors: Vec<(u32, Vec<f32>)> = (0..20).map(|i| (i, make_vector(64, i))).collect();

        let mut dist_count = 0;
        let selected =
            select_neighbors_heuristic(&query, &candidates, &all_vectors, 5, 1.0, &mut dist_count);

        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_empty_graph_merge() {
        let graph_a = MergeGraph::new(MergeConfig::default());
        let graph_b = make_test_graph(0, 10, 64);
        let config = MergeConfig::default();

        let (merged, stats) = naive_graph_merge(&graph_a, &graph_b, &config);

        assert_eq!(merged.len(), 10);
        assert_eq!(stats.nodes_from_a, 0);
        assert_eq!(stats.nodes_from_b, 10);
    }
}
