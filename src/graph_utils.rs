//! Shared graph construction utilities used by multiple ANN index modules.
//!
//! - [`build_knn_graph_nndescent`]: NN-descent (Dong et al., 2011) kNN graph construction.
//! - [`ensure_connectivity`]: repair so every node is reachable from the entry point.
//!
//! Both functions are conditionally used depending on which algorithm features
//! are enabled. Allow dead_code at the module level to avoid per-feature-combo lint noise.
#![allow(dead_code)]

use smallvec::{Array, SmallVec};

/// Build a kNN graph using NN-descent (Dong et al., 2011).
///
/// Returns neighbor lists where each entry contains
/// the `k` approximate nearest neighbor IDs for that node, sorted by distance (closest first).
///
/// # Arguments
/// - `n`: number of nodes
/// - `k`: target neighbor count per node
/// - `dist_fn`: `dist_fn(i, j)` returns the distance between nodes `i` and `j`
#[allow(clippy::needless_range_loop)]
pub fn build_knn_graph_nndescent<A, F>(n: usize, k: usize, dist_fn: F) -> Vec<SmallVec<A>>
where
    A: Array<Item = u32>,
    F: Fn(usize, usize) -> f32,
{
    let k = k.min(n.saturating_sub(1));
    if k == 0 {
        return vec![SmallVec::<A>::new(); n];
    }

    let mut nn: Vec<Vec<(f32, u32)>> = vec![Vec::with_capacity(k + 1); n];

    // LCG RNG for deterministic random initialization.
    let mut rng: u64 = 0xdeadbeef_cafebabe;
    let lcg_next = |state: &mut u64| -> usize {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as usize
    };

    // Initialize each node with k random distinct neighbors.
    for i in 0..n {
        let mut added = 0usize;
        let mut attempts = 0usize;
        while added < k && attempts < n * 4 {
            attempts += 1;
            let j = lcg_next(&mut rng) % n;
            if j == i {
                continue;
            }
            if nn[i].iter().any(|&(_, id)| id == j as u32) {
                continue;
            }
            let d = dist_fn(i, j);
            nn[i].push((d, j as u32));
            added += 1;
        }
        nn[i].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }

    fn try_insert(list: &mut Vec<(f32, u32)>, k: usize, dist: f32, id: u32) -> bool {
        if list.len() >= k && dist >= list[list.len() - 1].0 {
            return false;
        }
        if list.iter().any(|&(_, nid)| nid == id) {
            return false;
        }
        let pos = list.partition_point(|&(d, _)| d <= dist);
        list.insert(pos, (dist, id));
        if list.len() > k {
            list.pop();
        }
        true
    }

    let max_iters = 10usize;
    let early_stop_threshold = (0.001 * (n * k) as f64) as usize;

    for _iter in 0..max_iters {
        let mut updates = 0usize;

        for u in 0..n {
            let neighbors_u: Vec<(f32, u32)> = nn[u].clone();
            let len = neighbors_u.len();

            for a in 0..len {
                let (_, v1) = neighbors_u[a];
                for b in (a + 1)..len {
                    let (_, v2) = neighbors_u[b];
                    if v1 == v2 {
                        continue;
                    }
                    let d12 = dist_fn(v1 as usize, v2 as usize);
                    if try_insert(&mut nn[v1 as usize], k, d12, v2) {
                        updates += 1;
                    }
                    if try_insert(&mut nn[v2 as usize], k, d12, v1) {
                        updates += 1;
                    }
                }
            }
        }

        if updates <= early_stop_threshold {
            break;
        }
    }

    nn.into_iter()
        .map(|list| list.into_iter().map(|(_, id)| id).collect())
        .collect()
}

/// Ensure every node is reachable from `entry_point` along directed edges.
///
/// Graph search starts at `entry_point` and only follows out-edges, so the
/// guarantee that matters is directed reachability, not undirected
/// connectivity (a node with only out-edges is connected but never visited).
/// This marks the nodes reachable from the entry, then, for each unreached
/// node, adds an out-edge from a nearby reached node (NSG, Fu et al. 2019,
/// Alg. 2: DFS from the navigating node and link unreached nodes into the
/// tree) and marks everything that edge makes reachable.
/// Bridge candidates come from a local neighborhood scan (O(k^2) per bridge)
/// rather than an O(n) brute-force search.
///
/// # Arguments
/// - `neighbors`: mutable neighbor adjacency lists
/// - `entry_point`: the root/medoid node index
/// - `dist_fn`: `dist_fn(i, j)` returns the distance between nodes `i` and `j`
pub fn ensure_connectivity<A, F>(neighbors: &mut [SmallVec<A>], entry_point: u32, dist_fn: F)
where
    A: Array<Item = u32>,
    F: Fn(usize, usize) -> f32,
{
    let n = neighbors.len();
    if n <= 1 {
        return;
    }

    // Mark every node reachable from `start` along out-edges.
    fn mark_reachable<A: Array<Item = u32>>(
        neighbors: &[SmallVec<A>],
        reached: &mut [bool],
        start: u32,
    ) {
        if reached[start as usize] {
            return;
        }
        reached[start as usize] = true;
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            for &v in &neighbors[u as usize] {
                if !reached[v as usize] {
                    reached[v as usize] = true;
                    stack.push(v);
                }
            }
        }
    }

    let mut reached = vec![false; n];
    mark_reachable(neighbors, &mut reached, entry_point);

    for i in 0..n {
        if reached[i] {
            continue;
        }

        // Pick the closest reached node among a few cheap candidate sets:
        // i's two-hop out-neighborhood, the entry point's neighbors, and
        // their neighbors. The entry point itself is the fallback.
        let mut best_id = entry_point;
        let mut best_dist = dist_fn(i, entry_point as usize);
        let consider = |c: u32, best_id: &mut u32, best_dist: &mut f32| {
            if reached[c as usize] {
                let d = dist_fn(i, c as usize);
                if d < *best_dist {
                    *best_dist = d;
                    *best_id = c;
                }
            }
        };

        for &nb in &neighbors[i] {
            for &nb2 in &neighbors[nb as usize] {
                consider(nb2, &mut best_id, &mut best_dist);
            }
        }
        for &enb in &neighbors[entry_point as usize] {
            consider(enb, &mut best_id, &mut best_dist);
            for &enb2 in &neighbors[enb as usize] {
                consider(enb2, &mut best_id, &mut best_dist);
            }
        }

        // The edge best_id -> i is what makes i reachable; the reverse edge
        // keeps the bridge usable from i's side as well.
        neighbors[best_id as usize].push(i as u32);
        neighbors[i].push(best_id);
        mark_reachable(neighbors, &mut reached, i as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nndescent_basic() {
        // 2D points in a line: 0, 1, 2, 3, 4
        let points: Vec<[f32; 2]> = (0..5).map(|i| [i as f32, 0.0]).collect();
        let dist_fn = |i: usize, j: usize| {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            (dx * dx + dy * dy).sqrt()
        };

        let neighbors: Vec<SmallVec<[u32; 16]>> = build_knn_graph_nndescent(5, 2, dist_fn);
        assert_eq!(neighbors.len(), 5);
        // Each node should have 2 neighbors
        for nb in &neighbors {
            assert_eq!(nb.len(), 2);
        }
        // Node 0's nearest should be node 1
        assert!(neighbors[0].contains(&1));
        // Node 4's nearest should be node 3
        assert!(neighbors[4].contains(&3));
    }

    #[test]
    fn test_ensure_connectivity_already_connected() {
        let mut neighbors: Vec<SmallVec<[u32; 16]>> = vec![
            SmallVec::from_slice(&[1]),
            SmallVec::from_slice(&[0, 2]),
            SmallVec::from_slice(&[1]),
        ];
        let dist_fn = |i: usize, j: usize| (i as f32 - j as f32).abs();
        ensure_connectivity(&mut neighbors, 0, dist_fn);
        // Should be unchanged (already connected)
        assert_eq!(neighbors[0].len(), 1);
    }

    #[test]
    fn test_ensure_connectivity_disconnected() {
        // Two components: {0, 1} and {2, 3}
        let mut neighbors: Vec<SmallVec<[u32; 16]>> = vec![
            SmallVec::from_slice(&[1]),
            SmallVec::from_slice(&[0]),
            SmallVec::from_slice(&[3]),
            SmallVec::from_slice(&[2]),
        ];
        let dist_fn = |i: usize, j: usize| (i as f32 - j as f32).abs();
        ensure_connectivity(&mut neighbors, 0, dist_fn);

        // Verify all nodes reachable from entry_point 0
        let mut visited = [false; 4];
        let mut stack = vec![0usize];
        visited[0] = true;
        while let Some(node) = stack.pop() {
            for &nb in &neighbors[node] {
                let nb = nb as usize;
                if !visited[nb] {
                    visited[nb] = true;
                    stack.push(nb);
                }
            }
        }
        assert!(visited.iter().all(|&v| v), "All nodes should be reachable");
    }

    #[test]
    fn test_ensure_connectivity_uses_edge_direction() {
        // Node 2 points at 0 but nothing points at 2: undirected-connected,
        // yet unreachable from entry 0 along out-edges.
        let mut neighbors: Vec<SmallVec<[u32; 16]>> = vec![
            SmallVec::from_slice(&[1]),
            SmallVec::from_slice(&[0]),
            SmallVec::from_slice(&[0]),
        ];
        let dist_fn = |i: usize, j: usize| (i as f32 - j as f32).abs();
        ensure_connectivity(&mut neighbors, 0, dist_fn);
        assert!(
            neighbors[0].contains(&2) || neighbors[1].contains(&2),
            "node 2 must get an in-edge from a reachable node: {neighbors:?}"
        );
    }
}
