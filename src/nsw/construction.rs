//! Flat NSW graph construction.
//!
//! Inserts nodes one at a time. For each new node, greedy search finds
//! `ef_construction` candidates from the existing partial graph, then
//! selects the `m` best neighbors. Bidirectional edges are added with
//! degree trimming to `m_max`. Construction is O(n · ef · log n) rather
//! than the O(n²) brute-force that a full-scan candidate set would require.

use crate::simd;
use crate::RetrieveError;
use smallvec::SmallVec;

use super::graph::NSWIndex;
use super::search::greedy_search;

/// Construct flat NSW graph via incremental insertion.
pub fn construct_graph(index: &mut NSWIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    let n = index.num_vectors;
    let m = index.params.m;
    let m_max = index.params.m_max;
    let ef_construction = index.params.ef_construction.max(m);

    // Initialize neighbor lists for all nodes.
    index.neighbors = vec![SmallVec::new(); n];
    index.entry_point = Some(0);

    // Node 0 is the entry point; no neighbors to add yet.
    for current_id in 1..n {
        let entry_point = index.entry_point.unwrap_or(0);

        // Greedy search in the partial graph to get ef_construction candidates.
        // Nodes current_id..n-1 have empty neighbor lists so the search stays
        // in 0..current_id naturally.
        let candidates = greedy_search(
            index.get_vector(current_id),
            entry_point,
            &index.neighbors,
            &index.vectors,
            index.dimension,
            ef_construction,
        )?;

        // Select best m neighbors from the candidate set.
        let selected = select_neighbors(&candidates, m);

        for &neighbor_id in &selected {
            let j = neighbor_id as usize;

            // Forward edge: current -> neighbor
            if !index.neighbors[current_id].contains(&neighbor_id) {
                index.neighbors[current_id].push(neighbor_id);
            }

            // Reverse edge: neighbor -> current
            let current_u32 = current_id as u32;
            if !index.neighbors[j].contains(&current_u32) {
                index.neighbors[j].push(current_u32);
            }

            // Prune reverse edge list to m_max if over capacity.
            if index.neighbors[j].len() > m_max {
                prune_neighbors(
                    j,
                    m_max,
                    &index.vectors,
                    index.dimension,
                    &mut index.neighbors,
                );
            }
        }
    }

    Ok(())
}

/// Select the `m` closest candidates (deterministic, no randomness).
fn select_neighbors(candidates: &[(u32, f32)], m: usize) -> Vec<u32> {
    // candidates are already sorted by distance (greedy_search returns sorted)
    candidates.iter().take(m).map(|&(id, _)| id).collect()
}

/// Prune a node's neighbor list to `m_max` by keeping the closest neighbors.
fn prune_neighbors(
    node_id: usize,
    m_max: usize,
    vectors: &[f32],
    dimension: usize,
    neighbors: &mut [SmallVec<[u32; 16]>],
) {
    let node_vec_start = node_id * dimension;
    let node_vec = &vectors[node_vec_start..node_vec_start + dimension];

    // Compute distances from this node to all its current neighbors.
    let mut with_dist: Vec<(u32, f32)> = neighbors[node_id]
        .iter()
        .map(|&nb_id| {
            let start = nb_id as usize * dimension;
            let nb_vec = &vectors[start..start + dimension];
            let dist = 1.0 - simd::dot(node_vec, nb_vec);
            (nb_id, dist)
        })
        .collect();

    with_dist.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    with_dist.truncate(m_max);

    neighbors[node_id] = with_dist.into_iter().map(|(id, _)| id).collect();
}
