//! Vamana graph construction algorithm with alpha-pruning.
//!
//! Vamana (from DiskANN) builds high-quality navigable graphs using:
//! - Greedy search for candidate neighbors
//! - Alpha-pruning for diversity (avoids redundant edges)
//! - Multiple refinement passes
//! - Symmetrization (bidirectional edges)
//!
//! # Key Parameters
//!
//! - `max_degree` (R): Maximum neighbors per node
//! - `alpha`: Pruning aggressiveness (1.0 = none, 1.2 = moderate, 2.0 = aggressive)
//! - `build_beam_width` (L): Search width during construction
//!
//! # Algorithm
//!
//! For each vector v:
//! 1. Find candidates via greedy search from medoid/random seeds
//! 2. Alpha-prune candidates to get diverse neighbors
//! 3. Symmetrize edges (if u->v, add v->u if room)
//! 4. Repeat refinement passes for quality
//!
//! # References
//!
//! - Subramanya et al. (2019): "DiskANN: Fast Accurate Billion-point Nearest
//!   Neighbor Search on a Single Node"
//! - jianshu93/rust-diskann: Vamana implementation

use crate::RetrieveError;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// Configuration for Vamana graph construction.
#[derive(Clone, Debug)]
pub struct VamanaConfig {
    /// Maximum degree (R) - neighbors per node
    pub max_degree: usize,
    /// Build beam width (L) - search width during construction
    pub build_beam_width: usize,
    /// Alpha pruning factor (1.0 = no pruning, 1.2-2.0 typical)
    pub alpha: f32,
    /// Number of refinement passes
    pub num_passes: usize,
    /// Extra random seeds for multi-start search
    pub extra_seeds: usize,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            max_degree: 64,
            build_beam_width: 128,
            alpha: 1.2,
            num_passes: 2,
            extra_seeds: 2,
        }
    }
}

impl VamanaConfig {
    /// Fast config (lower quality, faster build)
    pub fn fast() -> Self {
        Self {
            max_degree: 32,
            build_beam_width: 64,
            alpha: 1.0,
            num_passes: 1,
            extra_seeds: 1,
        }
    }

    /// High quality config
    pub fn high_quality() -> Self {
        Self {
            max_degree: 96,
            build_beam_width: 256,
            alpha: 1.5,
            num_passes: 3,
            extra_seeds: 4,
        }
    }
}

/// Result of graph construction.
pub struct VamanaGraph {
    /// Adjacency lists (node -> neighbors)
    pub adjacency: Vec<Vec<u32>>,
    /// Medoid (entry point) ID
    pub medoid_id: u32,
}

/// Build Vamana graph from vectors.
///
/// # Arguments
/// * `vectors` - Flat array of vectors (num_vectors * dimension)
/// * `num_vectors` - Number of vectors
/// * `dimension` - Vector dimension
/// * `config` - Build configuration
/// * `distance_fn` - Distance function between two vector indices
///
/// # Returns
/// VamanaGraph with adjacency lists and medoid ID.
pub fn build_vamana_graph<F>(
    num_vectors: usize,
    config: &VamanaConfig,
    distance_fn: F,
) -> Result<VamanaGraph, RetrieveError>
where
    F: Fn(u32, u32) -> f32 + Sync,
{
    if num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    // Initialize random adjacency
    let mut graph = initialize_random_graph(num_vectors, config.max_degree);

    // Find medoid (approximate centroid)
    let medoid_id = find_medoid(num_vectors, &distance_fn);

    // Refinement passes
    for pass in 0..config.num_passes {
        let visit_order = shuffled_order(num_vectors, pass as u64);

        for &node in &visit_order {
            // Gather candidates from greedy search
            let candidates = gather_candidates(node, &graph, medoid_id, config, &distance_fn);

            // Alpha-prune to get final neighbors
            let neighbors = alpha_prune(
                node,
                &candidates,
                config.max_degree,
                config.alpha,
                &distance_fn,
            );

            graph[node as usize] = neighbors;
        }

        // Symmetrize after each pass
        symmetrize_graph(&mut graph, config.max_degree, &distance_fn, config.alpha);
    }

    Ok(VamanaGraph {
        adjacency: graph,
        medoid_id,
    })
}

/// Initialize graph with random edges.
fn initialize_random_graph(num_vectors: usize, max_degree: usize) -> Vec<Vec<u32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let target = (max_degree / 2).max(2).min(num_vectors.saturating_sub(1));

    let mut graph = vec![Vec::new(); num_vectors];
    let mut state = 12345u64;

    for (i, node_neighbors) in graph.iter_mut().enumerate() {
        let mut neighbors: HashSet<u32> = HashSet::new();
        while neighbors.len() < target {
            let mut hasher = DefaultHasher::new();
            state.hash(&mut hasher);
            state = hasher.finish();
            let nb = (state % num_vectors as u64) as u32;
            if nb != i as u32 {
                neighbors.insert(nb);
            }
        }
        *node_neighbors = neighbors.into_iter().collect();
    }

    graph
}

/// Find approximate medoid (node closest to random pivot set).
fn find_medoid<F>(num_vectors: usize, distance_fn: &F) -> u32
where
    F: Fn(u32, u32) -> f32,
{
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let k = 8.min(num_vectors);
    let mut state = 42u64;
    let mut pivots = Vec::new();

    for _ in 0..k {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        pivots.push((state % num_vectors as u64) as u32);
    }

    let mut best_id = 0u32;
    let mut best_score = f32::INFINITY;

    for i in 0..num_vectors {
        let score: f32 = pivots.iter().map(|&p| distance_fn(i as u32, p)).sum();
        if score < best_score {
            best_score = score;
            best_id = i as u32;
        }
    }

    best_id
}

/// Generate shuffled visit order.
fn shuffled_order(n: usize, seed: u64) -> Vec<u32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut state = seed;

    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        let j = (state % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }

    order
}

/// Gather candidates via greedy search from multiple seeds.
fn gather_candidates<F>(
    query_id: u32,
    graph: &[Vec<u32>],
    medoid_id: u32,
    config: &VamanaConfig,
    distance_fn: &F,
) -> Vec<(u32, f32)>
where
    F: Fn(u32, u32) -> f32,
{
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut all_candidates: Vec<(u32, f32)> = Vec::new();

    // Include current neighbors
    for &nb in &graph[query_id as usize] {
        let dist = distance_fn(query_id, nb);
        all_candidates.push((nb, dist));
    }

    // Generate seeds
    let mut seeds = vec![medoid_id];
    let mut state = query_id as u64;
    for _ in 0..config.extra_seeds {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        seeds.push((state % graph.len() as u64) as u32);
    }

    // Greedy search from each seed
    for start in seeds {
        let results = greedy_search(query_id, start, graph, config.build_beam_width, distance_fn);
        all_candidates.extend(results);
    }

    // Deduplicate by keeping best distance per ID
    all_candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    all_candidates.dedup_by(|a, b| {
        if a.0 == b.0 {
            if a.1 < b.1 {
                *b = *a;
            }
            true
        } else {
            false
        }
    });

    // Remove self
    all_candidates.retain(|(id, _)| *id != query_id);

    all_candidates
}

/// Greedy beam search.
fn greedy_search<F>(
    query_id: u32,
    start_id: u32,
    graph: &[Vec<u32>],
    beam_width: usize,
    distance_fn: &F,
) -> Vec<(u32, f32)>
where
    F: Fn(u32, u32) -> f32,
{
    #[derive(Clone, Copy)]
    struct Candidate {
        dist: f32,
        id: u32,
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
            // Max-heap: larger distance = higher priority
            // Use total_cmp for IEEE 754 total ordering (NaN-safe)
            self.dist.total_cmp(&other.dist)
        }
    }

    let mut visited: HashSet<u32> = HashSet::new();
    let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    let mut working_set: BinaryHeap<Candidate> = BinaryHeap::new();

    let start_dist = distance_fn(query_id, start_id);
    let start = Candidate {
        dist: start_dist,
        id: start_id,
    };
    frontier.push(Reverse(start));
    working_set.push(start);
    visited.insert(start_id);

    while let Some(Reverse(best)) = frontier.peek().copied() {
        if working_set.len() >= beam_width {
            if let Some(worst) = working_set.peek() {
                if best.dist >= worst.dist {
                    break;
                }
            }
        }

        // SAFETY: We just peeked successfully, so pop will succeed
        let Some(Reverse(current)) = frontier.pop() else {
            break;
        };

        for &nb in &graph[current.id as usize] {
            if !visited.insert(nb) {
                continue;
            }

            let d = distance_fn(query_id, nb);
            let cand = Candidate { dist: d, id: nb };

            if working_set.len() < beam_width {
                working_set.push(cand);
                frontier.push(Reverse(cand));
            } else if let Some(worst) = working_set.peek() {
                if d < worst.dist {
                    working_set.pop();
                    working_set.push(cand);
                    frontier.push(Reverse(cand));
                }
            }
        }
    }

    working_set
        .into_vec()
        .into_iter()
        .map(|c| (c.id, c.dist))
        .collect()
}

/// Alpha-pruning for neighbor selection.
///
/// Keeps neighbors that are not "redundant" - a candidate is redundant if
/// it's closer to an already-selected neighbor than to the query node,
/// scaled by alpha.
fn alpha_prune<F>(
    query_id: u32,
    candidates: &[(u32, f32)],
    max_degree: usize,
    alpha: f32,
    distance_fn: &F,
) -> Vec<u32>
where
    F: Fn(u32, u32) -> f32,
{
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort by distance
    let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected: Vec<u32> = Vec::new();

    'outer: for &(cand_id, cand_dist) in &sorted {
        if cand_id == query_id {
            continue;
        }

        // Check if redundant with any selected neighbor
        for &sel in &selected {
            let dist_to_sel = distance_fn(cand_id, sel);
            // Redundant if closer to selected neighbor than scaled distance to query
            if dist_to_sel < alpha * cand_dist {
                continue 'outer;
            }
        }

        selected.push(cand_id);
        if selected.len() >= max_degree {
            break;
        }
    }

    // Fill with closest if not full
    for &(cand_id, _) in &sorted {
        if selected.len() >= max_degree {
            break;
        }
        if cand_id != query_id && !selected.contains(&cand_id) {
            selected.push(cand_id);
        }
    }

    selected
}

/// Symmetrize graph (ensure bidirectional edges where possible).
fn symmetrize_graph<F>(graph: &mut [Vec<u32>], max_degree: usize, distance_fn: &F, alpha: f32)
where
    F: Fn(u32, u32) -> f32,
{
    let n = graph.len();

    // Collect incoming edges
    let mut incoming: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (u, neighbors) in graph.iter().enumerate() {
        for &v in neighbors {
            incoming[v as usize].push(u as u32);
        }
    }

    // For each node, merge outgoing with incoming and re-prune
    for u in 0..n {
        let mut pool: Vec<(u32, f32)> = Vec::new();

        // Add outgoing
        for &v in &graph[u] {
            let d = distance_fn(u as u32, v);
            pool.push((v, d));
        }

        // Add incoming
        for &v in &incoming[u] {
            if v != u as u32 {
                let d = distance_fn(u as u32, v);
                pool.push((v, d));
            }
        }

        // Deduplicate
        pool.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        pool.dedup_by(|a, b| a.0 == b.0);

        // Re-prune
        graph[u] = alpha_prune(u as u32, &pool, max_degree, alpha, distance_fn);
    }
}

/// Search the Vamana graph for k nearest neighbors.
pub fn search_vamana<F>(
    query_id: u32,
    graph: &VamanaGraph,
    k: usize,
    ef_search: usize,
    distance_fn: &F,
) -> Vec<(u32, f32)>
where
    F: Fn(u32, u32) -> f32,
{
    let results = greedy_search(
        query_id,
        graph.medoid_id,
        &graph.adjacency,
        ef_search,
        distance_fn,
    );

    let mut sorted = results;
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    sorted.truncate(k);
    sorted
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn simple_l2_distance(vectors: &[Vec<f32>]) -> impl Fn(u32, u32) -> f32 + '_ {
        move |a, b| {
            vectors[a as usize]
                .iter()
                .zip(vectors[b as usize].iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        }
    }

    #[test]
    fn test_vamana_config() {
        let default = VamanaConfig::default();
        assert_eq!(default.max_degree, 64);
        assert_eq!(default.alpha, 1.2);

        let fast = VamanaConfig::fast();
        assert!(fast.max_degree < default.max_degree);

        let hq = VamanaConfig::high_quality();
        assert!(hq.max_degree > default.max_degree);
    }

    #[test]
    fn test_alpha_prune_basic() {
        let distance_fn = |a: u32, b: u32| (a as f32 - b as f32).abs();

        // Candidates: 1, 2, 3, 4, 5 with distances from 0
        let candidates: Vec<(u32, f32)> = vec![(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0)];

        let pruned = alpha_prune(0, &candidates, 3, 1.0, &distance_fn);
        assert_eq!(pruned.len(), 3);
        assert!(pruned.contains(&1)); // Closest always included
    }

    #[test]
    fn test_alpha_prune_with_alpha() {
        // With alpha > 1, should keep more diverse neighbors
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.1, 0.0], // Very close to 1
            vec![0.0, 1.0], // Different direction
        ];
        let distance_fn = simple_l2_distance(&vectors);

        let candidates: Vec<(u32, f32)> = vec![(1, 1.0), (2, 1.1), (3, 1.0)];

        // With alpha=1.5, node 2 should be pruned (too close to 1)
        let pruned = alpha_prune(0, &candidates, 3, 1.5, &distance_fn);
        assert!(pruned.contains(&1));
        assert!(pruned.contains(&3)); // Different direction, not pruned
    }

    #[test]
    fn test_build_vamana_small() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let distance_fn = simple_l2_distance(&vectors);

        let config = VamanaConfig {
            max_degree: 3,
            build_beam_width: 10,
            alpha: 1.2,
            num_passes: 1,
            extra_seeds: 1,
        };

        let graph = build_vamana_graph(vectors.len(), &config, distance_fn).unwrap();

        // Check basic properties
        assert_eq!(graph.adjacency.len(), 5);
        assert!(graph.medoid_id < 5);

        // Each node should have <= max_degree neighbors
        for neighbors in &graph.adjacency {
            assert!(neighbors.len() <= config.max_degree);
        }
    }

    #[test]
    fn test_greedy_search() {
        // Simple line graph: 0 - 1 - 2 - 3 - 4
        let graph: Vec<Vec<u32>> = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];

        let distance_fn = |a: u32, b: u32| (a as f32 - b as f32).abs();

        // Search for node 4 starting from 0
        let results = greedy_search(4, 0, &graph, 10, &distance_fn);

        // Should find node 4
        assert!(results.iter().any(|(id, _)| *id == 4));
    }

    #[test]
    fn test_find_medoid() {
        let distance_fn = |a: u32, b: u32| (a as f32 - b as f32).abs();
        let medoid = find_medoid(10, &distance_fn);
        assert!(medoid < 10);
    }

    #[test]
    fn test_symmetrization() {
        // Asymmetric graph
        let mut graph: Vec<Vec<u32>> = vec![
            vec![1, 2], // 0 -> 1, 2
            vec![2],    // 1 -> 2
            vec![],     // 2 -> nothing
        ];

        let distance_fn = |a: u32, b: u32| (a as f32 - b as f32).abs();

        symmetrize_graph(&mut graph, 3, &distance_fn, 1.0);

        // After symmetrization, edges should be bidirectional
        // Node 1 should now have edge to 0
        assert!(graph[1].contains(&0));
        // Node 2 should have edges to 0 and 1
        assert!(graph[2].contains(&0) || graph[2].contains(&1));
    }
}
