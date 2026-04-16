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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

/// Parallel NSW construction using batched insertion.
#[cfg(feature = "parallel")]
pub fn construct_graph_parallel(
    index: &mut NSWIndex,
    batch_size: usize,
) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    let n = index.num_vectors;
    let m = index.params.m;
    let m_max = index.params.m_max;
    let ef_construction = index.params.ef_construction.max(m);
    let dim = index.dimension;

    index.neighbors = vec![SmallVec::new(); n];
    index.entry_point = Some(0);

    // Bootstrap: sequential first batch.
    let sequential_count = (ef_construction * 2).min(n);
    for current_id in 1..sequential_count {
        let entry_point = index.entry_point.unwrap_or(0);
        let candidates = greedy_search(
            index.get_vector(current_id),
            entry_point,
            &index.neighbors,
            &index.vectors,
            dim,
            ef_construction,
        )?;
        let selected = select_neighbors(&candidates, m);
        for &neighbor_id in &selected {
            let j = neighbor_id as usize;
            if !index.neighbors[current_id].contains(&neighbor_id) {
                index.neighbors[current_id].push(neighbor_id);
            }
            let current_u32 = current_id as u32;
            if !index.neighbors[j].contains(&current_u32) {
                index.neighbors[j].push(current_u32);
            }
        }
    }

    // Batched parallel insertion.
    let batch_sz = batch_size.max(1);
    for batch_start in (sequential_count..n).step_by(batch_sz) {
        let batch_end = (batch_start + batch_sz).min(n);
        let batch_ids: Vec<usize> = (batch_start..batch_end).collect();
        let entry_point = index.entry_point.unwrap_or(0);

        // Phase 1: parallel search.
        let results: Vec<(usize, Vec<u32>)> = batch_ids
            .par_iter()
            .map(|&current_id| {
                let candidates = greedy_search(
                    index.get_vector(current_id),
                    entry_point,
                    &index.neighbors,
                    &index.vectors,
                    dim,
                    ef_construction,
                )
                .unwrap_or_default();
                (current_id, select_neighbors(&candidates, m))
            })
            .collect();

        // Phase 2: sequential edge commit.
        for (current_id, selected) in &results {
            for &neighbor_id in selected {
                let j = neighbor_id as usize;
                if !index.neighbors[*current_id].contains(&neighbor_id) {
                    index.neighbors[*current_id].push(neighbor_id);
                }
                let current_u32 = *current_id as u32;
                if !index.neighbors[j].contains(&current_u32) {
                    index.neighbors[j].push(current_u32);
                }
            }
        }

        // Phase 3: parallel prune of overweight nodes.
        let overweight: Vec<usize> = (0..n)
            .filter(|&id| index.neighbors[id].len() > m_max)
            .collect();
        if !overweight.is_empty() {
            let pruned: Vec<(usize, SmallVec<[u32; 16]>)> = overweight
                .par_iter()
                .map(|&node_id| {
                    let node_start = node_id * dim;
                    let node_vec = &index.vectors[node_start..node_start + dim];
                    let mut with_dist: Vec<(u32, f32)> = index.neighbors[node_id]
                        .iter()
                        .map(|&nb_id| {
                            let start = nb_id as usize * dim;
                            let nb_vec = &index.vectors[start..start + dim];
                            (nb_id, 1.0 - simd::dot(node_vec, nb_vec))
                        })
                        .collect();
                    with_dist.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    with_dist.truncate(m_max);
                    (node_id, with_dist.into_iter().map(|(id, _)| id).collect())
                })
                .collect();
            for (node_id, new_neighbors) in pruned {
                index.neighbors[node_id] = new_neighbors;
            }
        }
    }

    Ok(())
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
