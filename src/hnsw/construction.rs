//! HNSW graph construction algorithm.

use crate::distance;
use crate::hnsw::graph::{HNSWIndex, Layer};
use crate::hnsw::search::greedy_search_layer;
use crate::RetrieveError;
use smallvec::SmallVec;

/// Select neighbors using RND (Relative Neighborhood Diversification).
///
/// Criterion: include candidate \(X_j\) if it is closer to the query than it is to
/// every already-selected neighbor \(X_i\) (i.e. dist(q, j) < dist(i, j) for all selected i).
fn select_neighbors_rnd(
    _query_vector: &[f32],
    candidates: &[(u32, f32)],
    m: usize,
    vectors: &[f32],
    dimension: usize,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort by distance to query
    let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected = Vec::with_capacity(m.min(sorted.len()));

    // Start with closest candidate
    if let Some((id, _)) = sorted.first() {
        selected.push(*id);
    }

    // RND: Add candidate X_j if dist(X_q, X_j) < dist(X_i, X_j) for all selected neighbors X_i
    for (candidate_id, query_to_candidate_dist) in sorted.iter().skip(1) {
        if selected.len() >= m {
            break;
        }

        let candidate_vec = get_vector(vectors, dimension, *candidate_id as usize);
        let mut can_add = true;

        // Check RND condition: dist(X_q, X_j) < dist(X_i, X_j) for all X_i in selected
        for &selected_id in &selected {
            let selected_vec = get_vector(vectors, dimension, selected_id as usize);
            let inter_distance = distance::cosine_distance_normalized(selected_vec, candidate_vec);

            // RND formula: query_to_candidate_dist must be < inter_distance
            if *query_to_candidate_dist >= inter_distance {
                can_add = false;
                break;
            }
        }

        if can_add {
            selected.push(*candidate_id);
        }
    }

    // If we still need more neighbors, add closest remaining
    while selected.len() < m && selected.len() < sorted.len() {
        for (id, _) in &sorted {
            if !selected.contains(id) {
                selected.push(*id);
                break;
            }
        }
    }

    selected
}

/// Select neighbors using MOND (Maximum-Oriented Neighborhood Diversification).
///
/// Maximizes angles between neighbors. Formula: ∠(X_j X_q X_i) > θ for all selected X_i.
/// Second-best ND strategy with moderate pruning (2-4%).
fn select_neighbors_mond(
    query_vector: &[f32],
    candidates: &[(u32, f32)],
    m: usize,
    vectors: &[f32],
    dimension: usize,
    min_angle_degrees: f32,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let min_angle_rad = min_angle_degrees.to_radians();
    let min_cos = min_angle_rad.cos();

    // Sort by distance to query
    let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected = Vec::with_capacity(m.min(sorted.len()));

    // Start with closest candidate
    if let Some((id, _)) = sorted.first() {
        selected.push(*id);
    }

    // Precompute dot(query, query) once -- loop-invariant across all candidates and selected neighbors
    use crate::simd;
    let dot_qq = simd::dot(query_vector, query_vector);

    // MOND: Add candidate if angle with all selected neighbors > min_angle
    for (candidate_id, _) in sorted.iter().skip(1) {
        if selected.len() >= m {
            break;
        }

        let candidate_vec = get_vector(vectors, dimension, *candidate_id as usize);
        let mut can_add = true;

        // Compute angle between query->candidate and query->selected for each selected neighbor
        // Precompute candidate-invariant dot products (loop-invariant hoisting)
        let dot_cq = simd::dot(candidate_vec, query_vector);
        let dot_cc_self = simd::dot(candidate_vec, candidate_vec);

        for &selected_id in &selected {
            let selected_vec = get_vector(vectors, dimension, selected_id as usize);

            // Use identity: dot(a-b, c-b) = dot(a,c) - dot(a,b) - dot(c,b) + dot(b,b)
            // For our case: dot(q_to_c, q_to_s) = dot(candidate_vec, selected_vec)
            //                - dot(candidate_vec, query) - dot(selected_vec, query) + dot(query, query)
            let dot_cc = simd::dot(candidate_vec, selected_vec);
            let dot_sq = simd::dot(selected_vec, query_vector);
            let dot_qc_qs = dot_cc - dot_cq - dot_sq + dot_qq;

            // Compute norms: norm(a-b)^2 = norm(a)^2 + norm(b)^2 - 2*dot(a,b)
            let norm_c_sq = dot_cc_self + dot_qq - 2.0 * dot_cq;
            let norm_s_sq = simd::dot(selected_vec, selected_vec) + dot_qq - 2.0 * dot_sq;

            if norm_c_sq > 0.0 && norm_s_sq > 0.0 {
                let norm_c = norm_c_sq.sqrt();
                let norm_s = norm_s_sq.sqrt();
                let cos_angle = dot_qc_qs / (norm_c * norm_s);
                // Angle > min_angle means cos(angle) < cos(min_angle) (since cosine is decreasing)
                if cos_angle >= min_cos {
                    can_add = false;
                    break;
                }
            }
        }

        if can_add {
            selected.push(*candidate_id);
        }
    }

    // If we still need more neighbors, add closest remaining
    while selected.len() < m && selected.len() < sorted.len() {
        for (id, _) in &sorted {
            if !selected.contains(id) {
                selected.push(*id);
                break;
            }
        }
    }

    selected
}

/// Select neighbors using RRND (Relaxed Relative Neighborhood Diversification).
///
/// Formula: dist(X_q, X_j) < α · dist(X_i, X_j) with α ≥ 1.5.
/// Less effective than RND, creates larger graphs.
fn select_neighbors_rrnd(
    _query_vector: &[f32],
    candidates: &[(u32, f32)],
    m: usize,
    vectors: &[f32],
    dimension: usize,
    alpha: f32,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort by distance to query
    let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected = Vec::with_capacity(m.min(sorted.len()));

    // Start with closest candidate
    if let Some((id, _)) = sorted.first() {
        selected.push(*id);
    }

    // RRND: Add candidate X_j if dist(X_q, X_j) < α · dist(X_i, X_j) for all selected X_i
    for (candidate_id, query_to_candidate_dist) in sorted.iter().skip(1) {
        if selected.len() >= m {
            break;
        }

        let candidate_vec = get_vector(vectors, dimension, *candidate_id as usize);
        let mut can_add = true;

        for &selected_id in &selected {
            let selected_vec = get_vector(vectors, dimension, selected_id as usize);
            let inter_distance = distance::cosine_distance_normalized(selected_vec, candidate_vec);

            // RRND formula: query_to_candidate_dist < alpha * inter_distance
            if *query_to_candidate_dist >= alpha * inter_distance {
                can_add = false;
                break;
            }
        }

        if can_add {
            selected.push(*candidate_id);
        }
    }

    // If we still need more neighbors, add closest remaining
    while selected.len() < m && selected.len() < sorted.len() {
        for (id, _) in &sorted {
            if !selected.contains(id) {
                selected.push(*id);
                break;
            }
        }
    }

    selected
}

/// Select neighbors based on configured diversification strategy.
pub fn select_neighbors(
    query_vector: &[f32],
    candidates: &[(u32, f32)],
    m: usize,
    vectors: &[f32],
    dimension: usize,
    strategy: &crate::hnsw::graph::NeighborhoodDiversification,
) -> Vec<u32> {
    match strategy {
        crate::hnsw::graph::NeighborhoodDiversification::RelativeNeighborhood => {
            select_neighbors_rnd(query_vector, candidates, m, vectors, dimension)
        }
        crate::hnsw::graph::NeighborhoodDiversification::MaximumOriented { min_angle_degrees } => {
            select_neighbors_mond(
                query_vector,
                candidates,
                m,
                vectors,
                dimension,
                *min_angle_degrees,
            )
        }
        crate::hnsw::graph::NeighborhoodDiversification::RelaxedRelative { alpha } => {
            select_neighbors_rrnd(query_vector, candidates, m, vectors, dimension, *alpha)
        }
    }
}

/// Get vector from SoA storage.
pub fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}

/// Construct HNSW graph layers.
///
/// Implements the insertion algorithm from the HNSW paper (Malkov & Yashunin, 2018).
///
/// Key insight: When descending through layers, we use the closest node found
/// in the layer above as the entry point for the next layer. This ensures
/// we start searching from a good position, not an arbitrary node.
pub fn construct_graph(index: &mut HNSWIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    // Find maximum layer
    let max_layer = index.layer_assignments.iter().max().copied().unwrap_or(0) as usize;

    // Initialize layers with uncompressed storage
    index.layers = (0..=max_layer)
        .map(|_| Layer::new_uncompressed(vec![SmallVec::new(); index.num_vectors]))
        .collect();

    // Global entry point for the already-inserted subgraph.
    //
    // IMPORTANT: we are doing an *offline* build from stored vectors, but we still must
    // respect insertion order: the entry point must be a node that has already been
    // inserted (otherwise early nodes route through "future" nodes that have no edges yet).
    let mut global_entry_point = 0u32;
    let mut global_entry_layer = index.layer_assignments[0];

    // Insert each vector into the graph
    for current_id in 0..index.num_vectors {
        let current_layer = index.layer_assignments[current_id] as usize;
        let current_vector = get_vector(&index.vectors, index.dimension, current_id);

        // First node initializes the entry point.
        if current_id == 0 {
            global_entry_point = 0;
            global_entry_layer = index.layer_assignments[0];
            continue;
        }

        // Track closest node found while descending through layers.
        // We propagate the best entry point down (standard HNSW insertion).
        let mut layer_entry_point = global_entry_point;

        let entry_layer = (global_entry_layer as usize).min(max_layer);

        // 1) Descend from the current entry layer down to (current_layer + 1) using ef=1 greedy search.
        // We do NOT add edges here; we only refine the entry point.
        if entry_layer > current_layer {
            for layer_idx in ((current_layer + 1)..=entry_layer).rev() {
                let layer = &index.layers[layer_idx];
                let results = greedy_search_layer(
                    current_vector,
                    layer_entry_point,
                    layer,
                    &index.vectors,
                    index.dimension,
                    1,
                );
                if let Some((best_id, _)) = results.first() {
                    layer_entry_point = *best_id;
                }
            }
        }

        // 2) For layers <= min(current_layer, entry_layer), run ef_construction search and connect.
        for layer_idx in (0..=current_layer.min(entry_layer)).rev() {
            let candidates = greedy_search_layer(
                current_vector,
                layer_entry_point,
                &index.layers[layer_idx],
                &index.vectors,
                index.dimension,
                index.params.ef_construction,
            );

            // Update entry point for the next lower layer.
            if let Some((best_id, _)) = candidates.first() {
                layer_entry_point = *best_id;
            }

            // Select neighbors using configured diversification strategy
            let m_actual = if layer_idx == 0 {
                index.params.m_max
            } else {
                index.params.m
            };

            let mut selected = select_neighbors(
                current_vector,
                &candidates,
                m_actual,
                &index.vectors,
                index.dimension,
                &index.params.neighborhood_diversification,
            );

            // Enforce layer membership + insertion order invariants.
            // We are inserting `current_id` now, so only nodes < current_id exist in the graph.
            selected.retain(|&id| {
                let id_usize = id as usize;
                id_usize < current_id && (index.layer_assignments[id_usize] as usize) >= layer_idx
            });

            // Pre-compute neighbor distances (before any mutable borrows).
            // Vectors are looked up via get_vector(&index.vectors, ...) when needed,
            // avoiding a Vec<f32> heap allocation per neighbor.
            let neighbor_data: Vec<(u32, f32)> = selected
                .iter()
                .map(|&id| {
                    let vec = get_vector(&index.vectors, index.dimension, id as usize);
                    let dist = distance::cosine_distance_normalized(current_vector, vec);
                    (id, dist)
                })
                .collect();

            // Pre-compute existing neighbors of current_id
            let current_existing_neighbors: Vec<u32> = if layer_idx < index.layers.len() {
                index.layers[layer_idx]
                    .get_neighbors(current_id as u32)
                    .iter()
                    .copied()
                    .collect()
            } else {
                Vec::new()
            };

            // Pre-compute distances from current to ALL its neighbors (existing + selected)
            // Use HashMap for O(1) lookup during pruning
            let mut all_current_distances: std::collections::HashMap<u32, f32> =
                std::collections::HashMap::with_capacity(
                    current_existing_neighbors.len() + selected.len(),
                );

            // Add distances to existing neighbors
            for &id in &current_existing_neighbors {
                let vec = get_vector(&index.vectors, index.dimension, id as usize);
                let dist = distance::cosine_distance_normalized(current_vector, vec);
                all_current_distances.insert(id, dist);
            }

            // Add distances to selected neighbors
            for &(nid, dist) in &neighbor_data {
                all_current_distances.insert(nid, dist);
            }

            // Pre-compute existing neighbor data for reverse connections
            let existing_neighbor_lists: Vec<Vec<u32>> = selected
                .iter()
                .map(|&neighbor_id| {
                    if layer_idx < index.layers.len() {
                        index.layers[layer_idx]
                            .get_neighbors(neighbor_id)
                            .iter()
                            .copied()
                            .collect()
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            // Pre-compute distances for each selected neighbor to ALL its potential neighbors
            // This includes: existing neighbors of that node + current_id
            let mut all_reverse_distances: Vec<std::collections::HashMap<u32, f32>> = Vec::new();
            for (idx, &(neighbor_id, dist_to_current)) in neighbor_data.iter().enumerate() {
                let neighbor_vec =
                    get_vector(&index.vectors, index.dimension, neighbor_id as usize);
                let mut distances = std::collections::HashMap::new();

                // Distance to current_id (already computed)
                distances.insert(current_id as u32, dist_to_current);

                // Distances to existing neighbors
                for &existing_id in &existing_neighbor_lists[idx] {
                    let existing_vec =
                        get_vector(&index.vectors, index.dimension, existing_id as usize);
                    let dist = distance::cosine_distance_normalized(neighbor_vec, existing_vec);
                    distances.insert(existing_id, dist);
                }

                all_reverse_distances.push(distances);
            }

            // Now do all mutable operations
            let layer = &mut index.layers[layer_idx];
            let neighbors_vec = layer.get_neighbors_mut();

            // First pass: add all edges without pruning
            for &neighbor_id in &selected {
                let neighbors = &mut neighbors_vec[current_id];
                if !neighbors.contains(&neighbor_id) {
                    neighbors.push(neighbor_id);
                }

                let reverse_neighbors = &mut neighbors_vec[neighbor_id as usize];
                if !reverse_neighbors.contains(&(current_id as u32)) {
                    reverse_neighbors.push(current_id as u32);
                }
            }

            // Second pass: prune current_id's neighbors if needed
            {
                let neighbors = &mut neighbors_vec[current_id];
                if neighbors.len() > m_actual {
                    let mut neighbor_candidates: Vec<(u32, f32)> = neighbors
                        .iter()
                        .map(|&id| {
                            let dist =
                                all_current_distances.get(&id).copied().unwrap_or_else(|| {
                                    // Compute distance on the fly if somehow missing
                                    let vec =
                                        get_vector(&index.vectors, index.dimension, id as usize);
                                    distance::cosine_distance_normalized(current_vector, vec)
                                });
                            (id, dist)
                        })
                        .collect();

                    neighbor_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
                    neighbor_candidates.truncate(m_actual);
                    *neighbors = neighbor_candidates.iter().map(|(id, _)| *id).collect();
                }
            }

            // Third pass: prune each selected neighbor's reverse list if needed
            for (idx, &neighbor_id) in selected.iter().enumerate() {
                let reverse_neighbors = &mut neighbors_vec[neighbor_id as usize];
                if reverse_neighbors.len() > m_actual {
                    let distances = &all_reverse_distances[idx];
                    let neighbor_vec =
                        get_vector(&index.vectors, index.dimension, neighbor_id as usize);

                    let mut reverse_candidates: Vec<(u32, f32)> = reverse_neighbors
                        .iter()
                        .map(|&id| {
                            let dist = distances.get(&id).copied().unwrap_or_else(|| {
                                // Compute distance on the fly if somehow missing
                                let vec = get_vector(&index.vectors, index.dimension, id as usize);
                                distance::cosine_distance_normalized(neighbor_vec, vec)
                            });
                            (id, dist)
                        })
                        .collect();

                    reverse_candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
                    reverse_candidates.truncate(m_actual);
                    *reverse_neighbors = reverse_candidates.iter().map(|(id, _)| *id).collect();
                }
            }
        }

        // 3) Update global entry point if this node reaches a new top layer.
        if current_layer > (global_entry_layer as usize) {
            global_entry_point = current_id as u32;
            global_entry_layer = index.layer_assignments[current_id];
        }
    }

    Ok(())
}
