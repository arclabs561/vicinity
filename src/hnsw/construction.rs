//! HNSW graph construction algorithm.

use crate::distance;
use crate::hnsw::graph::{HNSWIndex, Layer};
use crate::hnsw::search::greedy_search_layer;
use crate::RetrieveError;
use smallvec::SmallVec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Return a plain function pointer for the configured distance metric.
#[inline(always)]
fn dist_fn_for(index: &HNSWIndex) -> fn(&[f32], &[f32]) -> f32 {
    use crate::distance::DistanceMetric;
    match index.params.metric {
        DistanceMetric::L2 => distance::l2_distance,
        DistanceMetric::Cosine => distance::cosine_distance_normalized,
        DistanceMetric::Angular => distance::angular_distance,
        DistanceMetric::InnerProduct => distance::inner_product_distance,
    }
}

/// Select neighbors using RND (Relative Neighborhood Diversification).
///
/// Criterion: include candidate \(X_j\) if it is closer to the query than it is to
/// every already-selected neighbor \(X_i\) (i.e. dist(q, j) < dist(i, j) for all selected i).
fn select_neighbors_rnd(
    _query_vector: &[f32],
    candidates: &mut [(u32, f32)],
    m: usize,
    vectors: &[f32],
    dimension: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort in place (caller passes owned/mutable slice)
    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected = Vec::with_capacity(m.min(candidates.len()));
    // Collect rejected candidates in distance order for O(1) backfill
    let mut rejected = Vec::new();

    // Start with closest candidate
    selected.push(candidates[0].0);

    // RND: Add candidate X_j if dist(X_q, X_j) < dist(X_i, X_j) for all selected neighbors X_i
    for &(candidate_id, query_to_candidate_dist) in candidates.iter().skip(1) {
        if selected.len() >= m {
            break;
        }

        let candidate_vec = get_vector(vectors, dimension, candidate_id as usize);
        let mut can_add = true;

        for &selected_id in &selected {
            let selected_vec = get_vector(vectors, dimension, selected_id as usize);
            let inter_distance = dist_fn(selected_vec, candidate_vec);

            if query_to_candidate_dist >= inter_distance {
                can_add = false;
                break;
            }
        }

        if can_add {
            selected.push(candidate_id);
        } else {
            rejected.push(candidate_id);
        }
    }

    // Backfill from rejected candidates (already in distance order)
    let mut ri = 0;
    while selected.len() < m && ri < rejected.len() {
        selected.push(rejected[ri]);
        ri += 1;
    }

    selected
}

/// Select neighbors using MOND (Maximum-Oriented Neighborhood Diversification).
///
/// Maximizes angles between neighbors. Formula: ∠(X_j X_q X_i) > θ for all selected X_i.
/// Second-best ND strategy with moderate pruning (2-4%).
fn select_neighbors_mond(
    query_vector: &[f32],
    candidates: &mut [(u32, f32)],
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

    // Sort in place
    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected = Vec::with_capacity(m.min(candidates.len()));
    let mut rejected = Vec::new();

    // Start with closest candidate
    selected.push(candidates[0].0);

    // Precompute dot(query, query) once
    use crate::simd;
    let dot_qq = simd::dot(query_vector, query_vector);

    // MOND: Add candidate if angle with all selected neighbors > min_angle
    for &(candidate_id, _) in candidates.iter().skip(1) {
        if selected.len() >= m {
            break;
        }

        let candidate_vec = get_vector(vectors, dimension, candidate_id as usize);
        let mut can_add = true;

        let dot_cq = simd::dot(candidate_vec, query_vector);
        let dot_cc_self = simd::dot(candidate_vec, candidate_vec);

        for &selected_id in &selected {
            let selected_vec = get_vector(vectors, dimension, selected_id as usize);

            let dot_cc = simd::dot(candidate_vec, selected_vec);
            let dot_sq = simd::dot(selected_vec, query_vector);
            let dot_qc_qs = dot_cc - dot_cq - dot_sq + dot_qq;

            let norm_c_sq = dot_cc_self + dot_qq - 2.0 * dot_cq;
            let norm_s_sq = simd::dot(selected_vec, selected_vec) + dot_qq - 2.0 * dot_sq;

            if norm_c_sq > 0.0 && norm_s_sq > 0.0 {
                let norm_c = norm_c_sq.sqrt();
                let norm_s = norm_s_sq.sqrt();
                let cos_angle = dot_qc_qs / (norm_c * norm_s);
                if cos_angle >= min_cos {
                    can_add = false;
                    break;
                }
            }
        }

        if can_add {
            selected.push(candidate_id);
        } else {
            rejected.push(candidate_id);
        }
    }

    // Backfill from rejected (already in distance order)
    let mut ri = 0;
    while selected.len() < m && ri < rejected.len() {
        selected.push(rejected[ri]);
        ri += 1;
    }

    selected
}

/// Select neighbors using RRND (Relaxed Relative Neighborhood Diversification).
///
/// Formula: dist(X_q, X_j) < α · dist(X_i, X_j) with α ≥ 1.5.
/// Less effective than RND, creates larger graphs.
fn select_neighbors_rrnd(
    _query_vector: &[f32],
    candidates: &mut [(u32, f32)],
    m: usize,
    vectors: &[f32],
    dimension: usize,
    alpha: f32,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort in place
    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected = Vec::with_capacity(m.min(candidates.len()));
    let mut rejected = Vec::new();

    // Start with closest candidate
    selected.push(candidates[0].0);

    // RRND: Add candidate X_j if dist(X_q, X_j) < α · dist(X_i, X_j) for all selected X_i
    for &(candidate_id, query_to_candidate_dist) in candidates.iter().skip(1) {
        if selected.len() >= m {
            break;
        }

        let candidate_vec = get_vector(vectors, dimension, candidate_id as usize);
        let mut can_add = true;

        for &selected_id in &selected {
            let selected_vec = get_vector(vectors, dimension, selected_id as usize);
            let inter_distance = dist_fn(selected_vec, candidate_vec);

            if query_to_candidate_dist >= alpha * inter_distance {
                can_add = false;
                break;
            }
        }

        if can_add {
            selected.push(candidate_id);
        } else {
            rejected.push(candidate_id);
        }
    }

    // Backfill from rejected (already in distance order)
    let mut ri = 0;
    while selected.len() < m && ri < rejected.len() {
        selected.push(rejected[ri]);
        ri += 1;
    }

    selected
}

/// Select neighbors based on configured diversification strategy.
///
/// `candidates` is sorted in place to avoid allocation.
pub fn select_neighbors(
    query_vector: &[f32],
    candidates: &mut [(u32, f32)],
    m: usize,
    vectors: &[f32],
    dimension: usize,
    strategy: &crate::hnsw::graph::NeighborhoodDiversification,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<u32> {
    match strategy {
        crate::hnsw::graph::NeighborhoodDiversification::RelativeNeighborhood => {
            select_neighbors_rnd(query_vector, candidates, m, vectors, dimension, dist_fn)
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
            select_neighbors_rrnd(
                query_vector,
                candidates,
                m,
                vectors,
                dimension,
                *alpha,
                dist_fn,
            )
        }
    }
}

/// Get vector from SoA storage.
#[inline]
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

    // Capture the distance function once so later mutable borrows of `index` don't conflict.
    let dist_fn = dist_fn_for(index);

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
                    dist_fn,
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
                dist_fn,
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

            let mut candidates = candidates;
            let mut selected = select_neighbors(
                current_vector,
                &mut candidates,
                m_actual,
                &index.vectors,
                index.dimension,
                &index.params.neighborhood_diversification,
                dist_fn,
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
                    let dist = dist_fn(current_vector, vec);
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

            // Pre-compute distances from current to ALL its neighbors (existing + selected).
            // Uses a flat Vec instead of HashMap: for m ≤ 32 entries, linear scan avoids
            // hashing overhead and repeated allocations that dominated 7-8% of construction.
            let mut all_current_distances: Vec<(u32, f32)> =
                Vec::with_capacity(current_existing_neighbors.len() + selected.len());

            // Add distances to existing neighbors
            for &id in &current_existing_neighbors {
                let vec = get_vector(&index.vectors, index.dimension, id as usize);
                let dist = dist_fn(current_vector, vec);
                all_current_distances.push((id, dist));
            }

            // Add distances to selected neighbors
            for &(nid, dist) in &neighbor_data {
                all_current_distances.push((nid, dist));
            }

            // Pre-compute distances for each selected neighbor to ALL its potential neighbors.
            // Inlines neighbor list iteration to avoid intermediate Vec<Vec<u32>> allocation.
            let mut all_reverse_distances: Vec<Vec<(u32, f32)>> =
                Vec::with_capacity(neighbor_data.len());
            for &(neighbor_id, dist_to_current) in &neighbor_data {
                let neighbor_vec =
                    get_vector(&index.vectors, index.dimension, neighbor_id as usize);
                let existing: Vec<u32> = if layer_idx < index.layers.len() {
                    index.layers[layer_idx]
                        .get_neighbors(neighbor_id)
                        .iter()
                        .copied()
                        .collect()
                } else {
                    Vec::new()
                };
                let mut distances = Vec::with_capacity(1 + existing.len());

                // Distance to current_id (already computed)
                distances.push((current_id as u32, dist_to_current));

                // Distances to existing neighbors
                for &existing_id in &existing {
                    let existing_vec =
                        get_vector(&index.vectors, index.dimension, existing_id as usize);
                    let dist = dist_fn(neighbor_vec, existing_vec);
                    distances.push((existing_id, dist));
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

            // Second pass: prune current_id's neighbors using diversity heuristic
            // Paper (Algorithm 4): pruning should use the same heuristic as selection
            {
                let neighbors = &mut neighbors_vec[current_id];
                if neighbors.len() > m_actual {
                    let mut neighbor_candidates: Vec<(u32, f32)> = neighbors
                        .iter()
                        .map(|&id| {
                            let dist = all_current_distances
                                .iter()
                                .find(|(k, _)| *k == id)
                                .map(|(_, v)| *v)
                                .unwrap_or_else(|| {
                                    let vec =
                                        get_vector(&index.vectors, index.dimension, id as usize);
                                    dist_fn(current_vector, vec)
                                });
                            (id, dist)
                        })
                        .collect();

                    let pruned = select_neighbors(
                        current_vector,
                        &mut neighbor_candidates,
                        m_actual,
                        &index.vectors,
                        index.dimension,
                        &index.params.neighborhood_diversification,
                        dist_fn,
                    );
                    *neighbors = pruned.into_iter().collect();
                }
            }

            // Third pass: prune each selected neighbor's reverse list using diversity heuristic
            for (idx, &neighbor_id) in selected.iter().enumerate() {
                let reverse_neighbors = &mut neighbors_vec[neighbor_id as usize];
                if reverse_neighbors.len() > m_actual {
                    let distances = &all_reverse_distances[idx];
                    let neighbor_vec =
                        get_vector(&index.vectors, index.dimension, neighbor_id as usize);

                    let mut reverse_candidates: Vec<(u32, f32)> = reverse_neighbors
                        .iter()
                        .map(|&id| {
                            let dist = distances
                                .iter()
                                .find(|(k, _)| *k == id)
                                .map(|(_, v)| *v)
                                .unwrap_or_else(|| {
                                    let vec =
                                        get_vector(&index.vectors, index.dimension, id as usize);
                                    dist_fn(neighbor_vec, vec)
                                });
                            (id, dist)
                        })
                        .collect();

                    let pruned = select_neighbors(
                        neighbor_vec,
                        &mut reverse_candidates,
                        m_actual,
                        &index.vectors,
                        index.dimension,
                        &index.params.neighborhood_diversification,
                        dist_fn,
                    );
                    *reverse_neighbors = pruned.into_iter().collect();
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

// ─── Parallel construction ──────────────────────────────────────────────────

/// Result of searching the graph for one vector (computed in parallel).
#[cfg(feature = "parallel")]
struct SearchResult {
    /// The node being inserted.
    current_id: usize,
    /// Per-layer selected neighbors (layer_idx -> selected neighbor IDs).
    /// Only layers where this node has edges.
    per_layer: Vec<(usize, Vec<u32>)>,
}

/// Parallel HNSW graph construction using batched insertion.
///
/// Processes vectors in batches: within each batch, the neighbor search
/// (the expensive part) runs in parallel via rayon, then edges are committed
/// sequentially. Vectors in the same batch don't see each other's edges,
/// but with batch_size << n the quality impact is negligible.
///
/// Typical speedup: 3-6x on 8+ core machines for datasets > 100K vectors.
#[cfg(feature = "parallel")]
pub fn construct_graph_parallel(
    index: &mut HNSWIndex,
    batch_size: usize,
) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    let max_layer = index.layer_assignments.iter().max().copied().unwrap_or(0) as usize;
    index.layers = (0..=max_layer)
        .map(|_| Layer::new_uncompressed(vec![SmallVec::new(); index.num_vectors]))
        .collect();

    let dist_fn = dist_fn_for(index);

    // Phase 0: sequential bootstrap. Insert enough vectors to form a connected
    // base graph before switching to parallel batches. We need at least
    // ef_construction nodes so searches have enough candidates.
    let sequential_count = (index.params.ef_construction * 2).min(index.num_vectors);
    let mut global_entry_point = 0u32;
    let mut global_entry_layer = index.layer_assignments[0];

    for current_id in 0..sequential_count {
        insert_single(
            index,
            current_id,
            &mut global_entry_point,
            &mut global_entry_layer,
            dist_fn,
        );
    }

    // Phase 1+2: batched parallel insertion for remaining vectors.
    let batch_sz = batch_size.max(1);
    let remaining = sequential_count..index.num_vectors;

    for batch_start in remaining.step_by(batch_sz) {
        let batch_end = (batch_start + batch_sz).min(index.num_vectors);
        let batch_ids: Vec<usize> = (batch_start..batch_end).collect();

        // Phase 1: parallel neighbor search (read-only on graph).
        // All vectors in this batch see the graph as it existed before the batch.
        let ep = global_entry_point;
        let ep_layer = global_entry_layer as usize;
        let search_results: Vec<SearchResult> = batch_ids
            .par_iter()
            .map(|&current_id| search_for_neighbors(index, current_id, ep, ep_layer, dist_fn))
            .collect();

        // Phase 2: sequential edge commit (fast: just edge insertion + forward prune).
        for result in search_results {
            commit_edges(index, &result, dist_fn);

            // Update global entry point if this node reaches a new top layer.
            let current_layer = index.layer_assignments[result.current_id] as usize;
            if current_layer > (global_entry_layer as usize) {
                global_entry_point = result.current_id as u32;
                global_entry_layer = index.layer_assignments[result.current_id];
            }
        }

        // Phase 3: parallel pruning of overweight reverse neighbors.
        prune_overweight_nodes(index, dist_fn);
    }

    Ok(())
}

/// Insert a single vector into the graph (sequential path).
/// Used for the bootstrap phase and as a fallback.
#[cfg(feature = "parallel")]
fn insert_single(
    index: &mut HNSWIndex,
    current_id: usize,
    global_entry_point: &mut u32,
    global_entry_layer: &mut u8,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) {
    let current_layer = index.layer_assignments[current_id] as usize;
    let current_vector = get_vector(&index.vectors, index.dimension, current_id);
    let max_layer = index.layers.len().saturating_sub(1);

    if current_id == 0 {
        *global_entry_point = 0;
        *global_entry_layer = index.layer_assignments[0];
        return;
    }

    let mut layer_entry_point = *global_entry_point;
    let entry_layer = (*global_entry_layer as usize).min(max_layer);

    // Greedy descent through upper layers.
    if entry_layer > current_layer {
        for layer_idx in ((current_layer + 1)..=entry_layer).rev() {
            if layer_idx >= index.layers.len() {
                continue;
            }
            let results = greedy_search_layer(
                current_vector,
                layer_entry_point,
                &index.layers[layer_idx],
                &index.vectors,
                index.dimension,
                1,
                dist_fn,
            );
            if let Some((best_id, _)) = results.first() {
                layer_entry_point = *best_id;
            }
        }
    }

    // Search and connect at each layer.
    for layer_idx in (0..=current_layer.min(entry_layer)).rev() {
        let candidates = greedy_search_layer(
            current_vector,
            layer_entry_point,
            &index.layers[layer_idx],
            &index.vectors,
            index.dimension,
            index.params.ef_construction,
            dist_fn,
        );
        if let Some((best_id, _)) = candidates.first() {
            layer_entry_point = *best_id;
        }

        let m_actual = if layer_idx == 0 {
            index.params.m_max
        } else {
            index.params.m
        };

        let mut candidates = candidates;
        let mut selected = select_neighbors(
            current_vector,
            &mut candidates,
            m_actual,
            &index.vectors,
            index.dimension,
            &index.params.neighborhood_diversification,
            dist_fn,
        );
        selected.retain(|&id| {
            let id_usize = id as usize;
            id_usize < current_id && (index.layer_assignments[id_usize] as usize) >= layer_idx
        });

        // Simple edge commit (no reverse pruning for bootstrap speed).
        let layer = &mut index.layers[layer_idx];
        let neighbors_vec = layer.get_neighbors_mut();
        for &neighbor_id in &selected {
            let neighbors = &mut neighbors_vec[current_id];
            if !neighbors.contains(&neighbor_id) {
                neighbors.push(neighbor_id);
            }
            let reverse = &mut neighbors_vec[neighbor_id as usize];
            if !reverse.contains(&(current_id as u32)) {
                reverse.push(current_id as u32);
            }
        }
    }

    if current_layer > (*global_entry_layer as usize) {
        *global_entry_point = current_id as u32;
        *global_entry_layer = index.layer_assignments[current_id];
    }
}

/// Search phase: find neighbors for a vector (read-only on graph).
#[cfg(feature = "parallel")]
fn search_for_neighbors(
    index: &HNSWIndex,
    current_id: usize,
    entry_point: u32,
    entry_layer: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> SearchResult {
    let current_layer = index.layer_assignments[current_id] as usize;
    let current_vector = get_vector(&index.vectors, index.dimension, current_id);
    let max_layer = index.layers.len().saturating_sub(1);

    let mut layer_entry_point = entry_point;

    // Greedy descent through upper layers.
    if entry_layer > current_layer {
        for layer_idx in ((current_layer + 1)..=entry_layer.min(max_layer)).rev() {
            let results = greedy_search_layer(
                current_vector,
                layer_entry_point,
                &index.layers[layer_idx],
                &index.vectors,
                index.dimension,
                1,
                dist_fn,
            );
            if let Some((best_id, _)) = results.first() {
                layer_entry_point = *best_id;
            }
        }
    }

    let mut per_layer = Vec::new();

    for layer_idx in (0..=current_layer.min(entry_layer.min(max_layer))).rev() {
        let candidates = greedy_search_layer(
            current_vector,
            layer_entry_point,
            &index.layers[layer_idx],
            &index.vectors,
            index.dimension,
            index.params.ef_construction,
            dist_fn,
        );
        if let Some((best_id, _)) = candidates.first() {
            layer_entry_point = *best_id;
        }

        let m_actual = if layer_idx == 0 {
            index.params.m_max
        } else {
            index.params.m
        };

        let mut candidates = candidates;
        let mut selected = select_neighbors(
            current_vector,
            &mut candidates,
            m_actual,
            &index.vectors,
            index.dimension,
            &index.params.neighborhood_diversification,
            dist_fn,
        );
        // Only connect to nodes that exist (< current batch start, approximately).
        // We use current_id as the cutoff -- nodes after us in the batch haven't
        // committed edges yet, but their vectors are in the index.
        selected.retain(|&id| (index.layer_assignments[id as usize] as usize) >= layer_idx);

        per_layer.push((layer_idx, selected));
    }

    SearchResult {
        current_id,
        per_layer,
    }
}

/// Commit phase: add edges from search results to the graph (sequential).
/// Only adds edges and prunes the current node -- reverse pruning is deferred.
#[cfg(feature = "parallel")]
fn commit_edges(index: &mut HNSWIndex, result: &SearchResult, dist_fn: fn(&[f32], &[f32]) -> f32) {
    let current_id = result.current_id;
    let current_vector = get_vector(&index.vectors, index.dimension, current_id);

    for &(layer_idx, ref selected) in &result.per_layer {
        let m_actual = if layer_idx == 0 {
            index.params.m_max
        } else {
            index.params.m
        };

        let layer = &mut index.layers[layer_idx];
        let neighbors_vec = layer.get_neighbors_mut();

        // Add forward and reverse edges.
        for &neighbor_id in selected {
            let fwd = &mut neighbors_vec[current_id];
            if !fwd.contains(&neighbor_id) {
                fwd.push(neighbor_id);
            }
            let rev = &mut neighbors_vec[neighbor_id as usize];
            if !rev.contains(&(current_id as u32)) {
                rev.push(current_id as u32);
            }
        }

        // Prune current node's forward neighbors if over budget.
        // (Reverse pruning is deferred to prune_overweight_nodes.)
        let neighbors = &mut neighbors_vec[current_id];
        if neighbors.len() > m_actual {
            let mut neighbor_candidates: Vec<(u32, f32)> = neighbors
                .iter()
                .map(|&id| {
                    let vec = get_vector(&index.vectors, index.dimension, id as usize);
                    (id, dist_fn(current_vector, vec))
                })
                .collect();
            let pruned = select_neighbors(
                current_vector,
                &mut neighbor_candidates,
                m_actual,
                &index.vectors,
                index.dimension,
                &index.params.neighborhood_diversification,
                dist_fn,
            );
            *neighbors = pruned.into_iter().collect();
        }
    }
}

/// Prune all overweight neighbor lists after a batch of insertions.
/// This is done in parallel since each node's pruning is independent.
#[cfg(feature = "parallel")]
fn prune_overweight_nodes(index: &mut HNSWIndex, dist_fn: fn(&[f32], &[f32]) -> f32) {
    let num_vectors = index.num_vectors;
    let dimension = index.dimension;
    let diversification = index.params.neighborhood_diversification.clone();
    let m = index.params.m;
    let m_max = index.params.m_max;

    for layer_idx in 0..index.layers.len() {
        let m_actual = if layer_idx == 0 { m_max } else { m };
        let layer = &mut index.layers[layer_idx];
        let neighbors_vec = layer.get_neighbors_mut();

        // Collect IDs of overweight nodes.
        let overweight: Vec<usize> = (0..num_vectors)
            .filter(|&id| neighbors_vec[id].len() > m_actual)
            .collect();

        if overweight.is_empty() {
            continue;
        }

        // Compute pruned neighbor lists in parallel.
        let pruned_lists: Vec<(usize, SmallVec<[u32; 16]>)> = overweight
            .par_iter()
            .map(|&node_id| {
                let node_vec = get_vector(&index.vectors, dimension, node_id);
                let mut candidates: Vec<(u32, f32)> = neighbors_vec[node_id]
                    .iter()
                    .map(|&id| {
                        let vec = get_vector(&index.vectors, dimension, id as usize);
                        (id, dist_fn(node_vec, vec))
                    })
                    .collect();
                let pruned = select_neighbors(
                    node_vec,
                    &mut candidates,
                    m_actual,
                    &index.vectors,
                    dimension,
                    &diversification,
                    dist_fn,
                );
                (node_id, pruned.into_iter().collect::<SmallVec<_>>())
            })
            .collect();

        // Apply pruned lists.
        for (node_id, pruned) in pruned_lists {
            neighbors_vec[node_id] = pruned;
        }
    }
}
