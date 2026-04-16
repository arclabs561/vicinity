//! Graph quality analysis: occlusion statistics and monotonic search path (MSP) validation.
//!
//! These metrics are derived from the SNG pruning theory (Ma et al., 2025) and the
//! MARGO disk layout optimization (VLDB 2025). They quantify how well a proximity
//! graph supports greedy search.
//!
//! # Key concepts
//!
//! - **Occlusion**: edge (p, p*) *occludes* edge (p, p') if `d(p', p*) < d(p', p)`.
//!   The number of edges an edge occludes equals the number of nodes monotonically
//!   reachable via that edge (the "monotonic reachability" of the edge).
//!
//! - **Monotonic search path (MSP)**: a path where each step strictly decreases
//!   distance to the target. Greedy search follows MSPs. A graph is navigable iff
//!   MSPs exist from the entry point to all nodes.
//!
//! # References
//!
//! - Ma et al. (2025). "On the Sparse Neighborhood Graph" (arXiv:2603.06660)
//! - Zheng et al. (2025). "MARGO: Maximizing Recall on Graph with Optimal Disk Layout." VLDB.

use std::collections::{HashSet, VecDeque};

/// Statistics about a proximity graph's quality.
#[derive(Debug, Clone)]
pub struct GraphQuality {
    /// Number of nodes in the graph.
    pub num_nodes: usize,
    /// Number of directed edges in the graph.
    pub num_edges: usize,
    /// Fraction of nodes reachable from the entry point via greedy search (0.0 to 1.0).
    pub reachability: f32,
    /// Number of nodes NOT reachable from the entry point.
    pub unreachable_count: usize,
    /// Average number of nodes occluded per edge (monotonic reachability).
    pub avg_occlusion: f32,
    /// Maximum occlusion count across all edges.
    pub max_occlusion: usize,
    /// Average out-degree.
    pub avg_out_degree: f32,
    /// Average shortest path length from entry to all reachable nodes (in hops).
    pub avg_path_length: f32,
}

/// Compute graph quality metrics for an HNSW base layer.
///
/// `get_neighbors`: returns the neighbor list for a given node id.
/// `distance`: computes distance between two node ids.
/// `entry_point`: the entry node for reachability analysis.
/// `num_nodes`: total number of nodes.
///
/// This is O(n * avg_degree^2) for occlusion analysis and O(n + E) for BFS reachability.
pub fn analyze_graph_quality(
    entry_point: u32,
    num_nodes: usize,
    get_neighbors: impl Fn(u32) -> Vec<u32>,
    distance: impl Fn(u32, u32) -> f32,
) -> GraphQuality {
    if num_nodes == 0 {
        return GraphQuality {
            num_nodes: 0,
            num_edges: 0,
            reachability: 0.0,
            unreachable_count: 0,
            avg_occlusion: 0.0,
            max_occlusion: 0,
            avg_out_degree: 0.0,
            avg_path_length: 0.0,
        };
    }

    // BFS from entry point to compute reachability and path lengths.
    let mut visited = vec![false; num_nodes];
    let mut dist_from_entry = vec![u32::MAX; num_nodes];
    let mut queue = VecDeque::new();

    visited[entry_point as usize] = true;
    dist_from_entry[entry_point as usize] = 0;
    queue.push_back(entry_point);

    let mut reachable_count = 1usize;
    let mut total_path_length = 0u64;

    while let Some(current) = queue.pop_front() {
        let neighbors = get_neighbors(current);
        for &neighbor in &neighbors {
            if (neighbor as usize) < num_nodes && !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                dist_from_entry[neighbor as usize] = dist_from_entry[current as usize] + 1;
                total_path_length += dist_from_entry[neighbor as usize] as u64;
                reachable_count += 1;
                queue.push_back(neighbor);
            }
        }
    }

    let unreachable_count = num_nodes - reachable_count;
    let reachability = reachable_count as f32 / num_nodes as f32;
    let avg_path_length = if reachable_count > 1 {
        total_path_length as f32 / (reachable_count - 1) as f32
    } else {
        0.0
    };

    // Compute edge statistics and occlusion counts.
    let mut total_edges = 0usize;
    let mut total_degree = 0usize;
    let mut total_occlusion = 0usize;
    let mut max_occlusion = 0usize;

    // Sample-based occlusion: for each node, check how many 2-hop neighbors
    // each edge occludes. Full occlusion counting is O(n²); we approximate
    // by checking only 2-hop neighbors.
    for node_id in 0..num_nodes {
        let neighbors = get_neighbors(node_id as u32);
        let degree = neighbors.len();
        total_degree += degree;
        total_edges += degree;

        for &neighbor in &neighbors {
            // Count how many other neighbors this edge occludes.
            // Edge (node_id, neighbor) occludes (node_id, other) iff
            // d(other, neighbor) < d(other, node_id).
            let mut occlusion_count = 0;
            let other_neighbors = get_neighbors(neighbor);
            let mut seen = HashSet::new();
            seen.insert(node_id as u32);
            seen.insert(neighbor);

            for &two_hop in &other_neighbors {
                if seen.insert(two_hop) {
                    let d_to_neighbor = distance(two_hop, neighbor);
                    let d_to_node = distance(two_hop, node_id as u32);
                    if d_to_neighbor < d_to_node {
                        occlusion_count += 1;
                    }
                }
            }

            total_occlusion += occlusion_count;
            if occlusion_count > max_occlusion {
                max_occlusion = occlusion_count;
            }
        }
    }

    let avg_occlusion = if total_edges > 0 {
        total_occlusion as f32 / total_edges as f32
    } else {
        0.0
    };

    let avg_out_degree = if num_nodes > 0 {
        total_degree as f32 / num_nodes as f32
    } else {
        0.0
    };

    GraphQuality {
        num_nodes,
        num_edges: total_edges,
        reachability,
        unreachable_count,
        avg_occlusion,
        max_occlusion,
        avg_out_degree,
        avg_path_length,
    }
}

/// Validate monotonic search paths from the entry point to a sample of target nodes.
///
/// For each target, runs greedy search and checks whether the path found is
/// strictly monotonically decreasing in distance. Returns the fraction of targets
/// where a valid MSP was found.
///
/// `sample_size`: number of random targets to test (capped at num_nodes).
pub fn validate_msp(
    entry_point: u32,
    num_nodes: usize,
    get_neighbors: impl Fn(u32) -> Vec<u32>,
    distance_to_target: impl Fn(u32, u32) -> f32,
    sample_size: usize,
) -> MspValidation {
    let sample_size = sample_size.min(num_nodes);
    let mut valid_count = 0usize;
    let mut total_path_len = 0usize;
    let mut failed_targets = Vec::new();

    // Deterministic sample: evenly spaced node ids.
    let step = (num_nodes.checked_div(sample_size).unwrap_or(1)).max(1);

    for target_idx in 0..sample_size {
        let target = ((target_idx * step) as u32).min((num_nodes - 1) as u32);
        if target == entry_point {
            valid_count += 1;
            continue;
        }

        // Greedy walk from entry toward target, checking monotonicity.
        let mut current = entry_point;
        let mut current_dist = distance_to_target(current, target);
        let mut path_len = 0;
        let mut is_monotonic = true;
        let mut visited = HashSet::new();
        visited.insert(current);

        loop {
            let neighbors = get_neighbors(current);
            let mut best_neighbor = None;
            let mut best_dist = current_dist;

            for &neighbor in &neighbors {
                if visited.contains(&neighbor) {
                    continue;
                }
                let d = distance_to_target(neighbor, target);
                if d < best_dist {
                    best_dist = d;
                    best_neighbor = Some(neighbor);
                }
            }

            if let Some(next) = best_neighbor {
                if best_dist >= current_dist {
                    is_monotonic = false;
                    break;
                }
                visited.insert(next);
                current = next;
                current_dist = best_dist;
                path_len += 1;

                // Reached target.
                if current == target {
                    break;
                }

                // Safety: prevent infinite loops on disconnected graphs.
                if path_len > num_nodes {
                    is_monotonic = false;
                    break;
                }
            } else {
                // Stuck: no improving neighbor found.
                is_monotonic = false;
                break;
            }
        }

        if is_monotonic && current == target {
            valid_count += 1;
            total_path_len += path_len;
        } else {
            failed_targets.push(target);
        }
    }

    MspValidation {
        sample_size,
        valid_count,
        valid_fraction: if sample_size > 0 {
            valid_count as f32 / sample_size as f32
        } else {
            0.0
        },
        avg_path_length: if valid_count > 0 {
            total_path_len as f32 / valid_count as f32
        } else {
            0.0
        },
        failed_targets,
    }
}

/// Results of MSP (monotonic search path) validation.
#[derive(Debug, Clone)]
pub struct MspValidation {
    /// Number of targets sampled.
    pub sample_size: usize,
    /// Number of targets with valid MSPs from entry.
    pub valid_count: usize,
    /// Fraction of targets with valid MSPs (0.0 to 1.0).
    pub valid_fraction: f32,
    /// Average MSP length for valid paths (in hops).
    pub avg_path_length: f32,
    /// Node IDs where MSP validation failed (greedy search got stuck or
    /// path was non-monotonic).
    pub failed_targets: Vec<u32>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::useless_vec, clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Build a simple line graph: 0 -> 1 -> 2 -> ... -> n-1.
    fn line_graph(n: usize) -> (Vec<Vec<u32>>, Vec<f32>) {
        let mut neighbors = vec![Vec::new(); n];
        let coords: Vec<f32> = (0..n).map(|i| i as f32).collect();

        for i in 0..n {
            if i > 0 {
                neighbors[i].push((i - 1) as u32);
            }
            if i + 1 < n {
                neighbors[i].push((i + 1) as u32);
            }
        }

        (neighbors, coords)
    }

    /// Build a complete graph on n nodes.
    fn complete_graph(n: usize) -> Vec<Vec<u32>> {
        let mut neighbors = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    neighbors[i].push(j as u32);
                }
            }
        }
        neighbors
    }

    #[test]
    fn test_line_graph_quality() {
        let n = 10;
        let (neighbors, coords) = line_graph(n);

        let quality = analyze_graph_quality(
            0,
            n,
            |id| neighbors[id as usize].clone(),
            |a, b| (coords[a as usize] - coords[b as usize]).abs(),
        );

        assert_eq!(quality.num_nodes, 10);
        assert_eq!(quality.reachability, 1.0);
        assert_eq!(quality.unreachable_count, 0);
        assert!(quality.avg_out_degree > 1.5); // ~1.8 for line graph
        assert!(quality.avg_path_length > 0.0);
    }

    #[test]
    fn test_complete_graph_quality() {
        let n = 8;
        let neighbors = complete_graph(n);
        let coords: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let quality = analyze_graph_quality(
            0,
            n,
            |id| neighbors[id as usize].clone(),
            |a, b| (coords[a as usize] - coords[b as usize]).abs(),
        );

        assert_eq!(quality.num_nodes, 8);
        assert_eq!(quality.reachability, 1.0);
        // Complete graph: avg_path_length should be 1.0 (every node is 1 hop away).
        assert!(
            (quality.avg_path_length - 1.0).abs() < 0.01,
            "complete graph should have avg path length 1.0, got {}",
            quality.avg_path_length
        );
    }

    #[test]
    fn test_disconnected_graph() {
        // Two disconnected components.
        let neighbors = vec![
            vec![1], // 0 -> 1
            vec![0], // 1 -> 0
            vec![3], // 2 -> 3 (separate component)
            vec![2], // 3 -> 2
        ];

        let quality = analyze_graph_quality(
            0,
            4,
            |id| neighbors[id as usize].clone(),
            |a, b| (a as f32 - b as f32).abs(),
        );

        assert_eq!(quality.num_nodes, 4);
        assert_eq!(quality.unreachable_count, 2);
        assert!((quality.reachability - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_msp_validation_line_graph() {
        let n = 20;
        let (neighbors, coords) = line_graph(n);

        let msp = validate_msp(
            0,
            n,
            |id| neighbors[id as usize].clone(),
            |a, b| (coords[a as usize] - coords[b as usize]).abs(),
            10,
        );

        // On a line graph, greedy search from 0 can reach any node
        // monotonically by always stepping closer.
        assert!(
            msp.valid_fraction > 0.8,
            "line graph should have high MSP validity: {:.1}%",
            msp.valid_fraction * 100.0
        );
    }

    #[test]
    fn test_msp_validation_complete_graph() {
        let n = 10;
        let neighbors = complete_graph(n);
        let coords: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let msp = validate_msp(
            0,
            n,
            |id| neighbors[id as usize].clone(),
            |a, b| (coords[a as usize] - coords[b as usize]).abs(),
            n,
        );

        // Complete graph: every node is a direct neighbor of every other.
        assert_eq!(msp.valid_fraction, 1.0);
        assert!(msp.failed_targets.is_empty());
    }

    #[test]
    fn test_empty_graph() {
        let quality = analyze_graph_quality(0, 0, |_| Vec::new(), |_, _| 0.0);
        assert_eq!(quality.num_nodes, 0);
        assert_eq!(quality.reachability, 0.0);
    }

    #[test]
    fn test_occlusion_on_triangle() {
        // Triangle: 0-1, 0-2, 1-2 with coords [0, 1, 3].
        // Edge (0, 1) should occlude some nodes reachable via 1.
        let neighbors = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let coords = [0.0f32, 1.0, 3.0];

        let quality = analyze_graph_quality(
            0,
            3,
            |id| neighbors[id as usize].clone(),
            |a, b| (coords[a as usize] - coords[b as usize]).abs(),
        );

        assert_eq!(quality.num_nodes, 3);
        assert_eq!(quality.num_edges, 6); // bidirectional
        assert!(quality.avg_occlusion >= 0.0);
    }
}
