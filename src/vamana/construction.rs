//! Vamana graph construction algorithm (two-pass with RRND + RND).

use crate::distance as hnsw_distance;
use crate::hnsw::construction::select_neighbors;
use crate::hnsw::graph::NeighborhoodDiversification;
use crate::vamana::graph::VamanaIndex;
use crate::RetrieveError;
use rand::seq::IndexedRandom;
use smallvec::SmallVec;

/// Initialize random graph with minimum degree >= log(n).
///
/// Creates initial graph structure by connecting each node to log(n) random neighbors.
fn initialize_random_graph(index: &mut VamanaIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    let min_degree = (index.num_vectors as f64).ln().ceil() as usize;
    let mut rng = rand::rng();

    // Pre-allocate all_ids once (optimized: avoid recreating for each node)
    let all_ids: Vec<u32> = (0..index.num_vectors as u32).collect();

    // For each vector, connect to min_degree random neighbors
    for i in 0..index.num_vectors {
        let mut neighbors: SmallVec<[u32; 16]> = SmallVec::with_capacity(min_degree);

        // Sample random neighbors (excluding self) - optimized: filter inline
        let candidate_ids: Vec<u32> = all_ids
            .iter()
            .filter(|&&id| id != i as u32)
            .copied()
            .collect();

        let num_neighbors = min_degree.min(candidate_ids.len());
        let selected: Vec<u32> = candidate_ids
            .choose_multiple(&mut rng, num_neighbors)
            .copied()
            .collect();

        neighbors.extend(selected);
        index.neighbors[i] = neighbors;
    }

    Ok(())
}

/// First pass: Refine graph using RRND (Relaxed Relative Neighborhood Diversification).
///
/// Formula: dist(X_q, X_j) < α · dist(X_i, X_j) with α ≥ 1.5
#[cfg(feature = "vamana")]
fn refine_with_rrnd(index: &mut VamanaIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    // For each vector, refine its neighbors using RRND
    for current_id in 0..index.num_vectors {
        let current_vector = index.get_vector(current_id);

        // Collect all current neighbors and their distances
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(index.params.ef_construction);
        for &neighbor_id in &index.neighbors[current_id] {
            let neighbor_vec = index.get_vector(neighbor_id as usize);
            let dist = hnsw_distance::cosine_distance_normalized(current_vector, neighbor_vec);
            candidates.push((neighbor_id, dist));
        }

        // Also explore neighbors of neighbors (up to ef_construction)
        // Use VecDeque for O(1) pop_front instead of O(n) remove(0)
        use std::collections::VecDeque;
        let mut to_explore: VecDeque<u32> = index.neighbors[current_id].iter().copied().collect();
        let mut visited = std::collections::HashSet::with_capacity(index.params.ef_construction);
        visited.insert(current_id as u32);

        while let Some(explore_id) = to_explore.pop_front() {
            if visited.contains(&explore_id) {
                continue;
            }
            visited.insert(explore_id);

            if candidates.len() >= index.params.ef_construction {
                break;
            }

            let explore_vec = index.get_vector(explore_id as usize);
            let dist = hnsw_distance::cosine_distance_normalized(current_vector, explore_vec);
            candidates.push((explore_id, dist));

            // Add neighbors to explore
            for &neighbor_id in &index.neighbors[explore_id as usize] {
                if !visited.contains(&neighbor_id) {
                    to_explore.push_back(neighbor_id);
                }
            }
        }

        // Select neighbors using RRND
        // Note: vamana feature requires hnsw feature, so we can always use select_neighbors
        let selected = select_neighbors(
            current_vector,
            &candidates,
            index.params.max_degree,
            &index.vectors,
            index.dimension,
            &NeighborhoodDiversification::RelaxedRelative {
                alpha: index.params.alpha,
            },
        );

        // Update outgoing neighbors
        index.neighbors[current_id] = SmallVec::from_vec(selected.clone());

        // Add reverse (bidirectional) edges. Paper: when p selects q as
        // neighbor, add q->p edge. Prune if q's list exceeds max_degree.
        let dim = index.dimension;
        let max_deg = index.params.max_degree;
        for &neighbor_id in &selected {
            let reverse = &mut index.neighbors[neighbor_id as usize];
            if !reverse.contains(&(current_id as u32)) {
                reverse.push(current_id as u32);
                if reverse.len() > max_deg {
                    let nstart = neighbor_id as usize * dim;
                    let node_vec = &index.vectors[nstart..nstart + dim];
                    let mut scored: Vec<(u32, f32)> = reverse
                        .iter()
                        .map(|&nid| {
                            let s = nid as usize * dim;
                            let v = &index.vectors[s..s + dim];
                            (
                                nid,
                                crate::distance::cosine_distance_normalized(node_vec, v),
                            )
                        })
                        .collect();
                    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                    scored.truncate(max_deg);
                    *reverse = scored.iter().map(|(id, _)| *id).collect();
                }
            }
        }
    }

    Ok(())
}

/// Second pass: Further refine using RND (Relative Neighborhood Diversification).
///
/// Formula: dist(X_q, X_j) < dist(X_i, X_j) for all neighbors X_i
#[cfg(feature = "vamana")]
fn refine_with_rnd(index: &mut VamanaIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    // For each vector, refine its neighbors using RND
    for current_id in 0..index.num_vectors {
        let current_vector = index.get_vector(current_id);

        // Collect all current neighbors and their distances
        // Avoid clone: use reference to neighbors
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(index.params.ef_construction);
        for &neighbor_id in &index.neighbors[current_id] {
            let neighbor_vec = index.get_vector(neighbor_id as usize);
            let dist = hnsw_distance::cosine_distance_normalized(current_vector, neighbor_vec);
            candidates.push((neighbor_id, dist));
        }

        // Also explore neighbors of neighbors (up to ef_construction)
        // Use VecDeque for O(1) pop_front instead of O(n) remove(0)
        use std::collections::VecDeque;
        let mut to_explore: VecDeque<u32> = index.neighbors[current_id].iter().copied().collect();
        let mut visited = std::collections::HashSet::with_capacity(index.params.ef_construction);
        visited.insert(current_id as u32);

        while let Some(explore_id) = to_explore.pop_front() {
            if visited.contains(&explore_id) {
                continue;
            }
            visited.insert(explore_id);

            if candidates.len() >= index.params.ef_construction {
                break;
            }

            let explore_vec = index.get_vector(explore_id as usize);
            let dist = hnsw_distance::cosine_distance_normalized(current_vector, explore_vec);
            candidates.push((explore_id, dist));

            // Add neighbors to explore (avoid clone: use reference)
            for &neighbor_id in &index.neighbors[explore_id as usize] {
                if !visited.contains(&neighbor_id) {
                    to_explore.push_back(neighbor_id);
                }
            }
        }

        // Select neighbors using RND
        // Note: vamana feature requires hnsw feature, so we can always use select_neighbors
        let selected = select_neighbors(
            current_vector,
            &candidates,
            index.params.max_degree,
            &index.vectors,
            index.dimension,
            &NeighborhoodDiversification::RelativeNeighborhood,
        );

        // Update outgoing neighbors
        index.neighbors[current_id] = SmallVec::from_vec(selected.clone());

        // Add reverse (bidirectional) edges with pruning
        let dim = index.dimension;
        let max_deg = index.params.max_degree;
        for &neighbor_id in &selected {
            let reverse = &mut index.neighbors[neighbor_id as usize];
            if !reverse.contains(&(current_id as u32)) {
                reverse.push(current_id as u32);
                if reverse.len() > max_deg {
                    let nstart = neighbor_id as usize * dim;
                    let node_vec = &index.vectors[nstart..nstart + dim];
                    let mut scored: Vec<(u32, f32)> = reverse
                        .iter()
                        .map(|&nid| {
                            let s = nid as usize * dim;
                            let v = &index.vectors[s..s + dim];
                            (
                                nid,
                                crate::distance::cosine_distance_normalized(node_vec, v),
                            )
                        })
                        .collect();
                    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                    scored.truncate(max_deg);
                    *reverse = scored.iter().map(|(id, _)| *id).collect();
                }
            }
        }
    }

    Ok(())
}

/// Compute the medoid: the vector closest to the centroid of all vectors.
///
/// Returns the index of the medoid vector.
fn compute_medoid(index: &VamanaIndex) -> u32 {
    let n = index.num_vectors;
    let dim = index.dimension;

    // Compute centroid
    let mut centroid = vec![0.0_f32; dim];
    for i in 0..n {
        let vec = index.get_vector(i);
        for (c, &v) in centroid.iter_mut().zip(vec.iter()) {
            *c += v;
        }
    }
    let inv_n = 1.0 / n as f32;
    for c in centroid.iter_mut() {
        *c *= inv_n;
    }

    // Normalize centroid so cosine_distance_normalized is valid
    centroid = crate::distance::normalize(&centroid);

    // Find vector closest to centroid
    let mut best_id: u32 = 0;
    let mut best_dist = f32::INFINITY;
    for i in 0..n {
        let vec = index.get_vector(i);
        let dist = hnsw_distance::cosine_distance_normalized(&centroid, vec);
        if dist < best_dist {
            best_dist = dist;
            best_id = i as u32;
        }
    }

    best_id
}

/// Construct Vamana graph using two-pass algorithm.
///
/// 1. Compute medoid (entry point for search)
/// 2. Initialize random graph with degree >= log(n)
/// 3. First pass: Refine using RND (alpha=1.0, strict diversity)
/// 4. Second pass: Further refine using RRND (alpha>1.0, relaxed)
///
/// The paper (DiskANN/Vamana, Algorithm 2) specifies: first pass with
/// alpha=1.0, second pass with alpha>1.0. The first pass builds a basic
/// graph with strict diversity; the second pass enriches it with the
/// relaxed criterion to add longer-range edges.
pub fn construct_graph(index: &mut VamanaIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    // Step 0: Compute medoid entry point
    index.medoid = compute_medoid(index);

    // Step 1: Initialize random graph
    initialize_random_graph(index)?;

    // Step 2: First pass - strict diversity (alpha=1.0)
    refine_with_rnd(index)?;

    // Step 3: Second pass - relaxed diversity (alpha>1.0)
    refine_with_rrnd(index)?;

    Ok(())
}
