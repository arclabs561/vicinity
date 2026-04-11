//! BFS graph reordering for cache-friendly traversal.
//!
//! Computes a permutation that places BFS-adjacent nodes contiguously in memory.
//! Applied after graph construction to reduce L2/L3 cache misses during search.
//!
//! The permutation is index-agnostic: it only needs an adjacency list and an
//! entry point. Each index type applies the permutation to its own fields.

use std::collections::VecDeque;

/// Result of BFS reordering: the permutation to apply.
pub struct ReorderPermutation {
    /// `new_order[new_idx] = old_idx` — which old node goes to each new position.
    pub new_order: Vec<u32>,
    /// `old_to_new[old_idx] = new_idx` — where each old node ends up.
    pub old_to_new: Vec<u32>,
}

/// Compute a BFS-order permutation from an adjacency list.
///
/// `neighbors_fn(node) -> &[u32]` returns the neighbor list for a given node.
/// `entry_point` is the BFS start (typically the medoid or highest-layer node).
/// `n` is the total number of nodes.
pub fn compute_bfs_permutation<F>(
    entry_point: usize,
    n: usize,
    neighbors_fn: F,
) -> ReorderPermutation
where
    F: Fn(usize) -> Vec<u32>,
{
    let mut new_order: Vec<u32> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut queue = VecDeque::with_capacity(n);

    queue.push_back(entry_point);
    visited[entry_point] = true;

    while let Some(node) = queue.pop_front() {
        new_order.push(node as u32);
        for &nb in &neighbors_fn(node) {
            let nb = nb as usize;
            if nb < n && !visited[nb] {
                visited[nb] = true;
                queue.push_back(nb);
            }
        }
    }

    // Append unreachable nodes
    for (i, &v) in visited.iter().enumerate().take(n) {
        if !v {
            new_order.push(i as u32);
        }
    }

    // Build inverse permutation
    let mut old_to_new = vec![0u32; n];
    for (new_idx, &old_idx) in new_order.iter().enumerate() {
        old_to_new[old_idx as usize] = new_idx as u32;
    }

    ReorderPermutation {
        new_order,
        old_to_new,
    }
}

/// Permute a flat vector array (stride = dim) according to the permutation.
pub fn permute_vectors(vectors: &[f32], dim: usize, perm: &ReorderPermutation) -> Vec<f32> {
    let n = perm.new_order.len();
    let mut new_vectors = vec![0.0f32; n * dim];
    for (new_idx, &old_idx) in perm.new_order.iter().enumerate() {
        let src = old_idx as usize * dim;
        let dst = new_idx * dim;
        new_vectors[dst..dst + dim].copy_from_slice(&vectors[src..src + dim]);
    }
    new_vectors
}

/// Permute a slice of T according to the permutation.
pub fn permute_slice<T: Clone>(items: &[T], perm: &ReorderPermutation) -> Vec<T> {
    perm.new_order
        .iter()
        .map(|&old| items[old as usize].clone())
        .collect()
}
