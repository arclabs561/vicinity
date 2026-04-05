//! Vamana graph construction algorithm (two-pass with RRND + RND).

use crate::distance as hnsw_distance;
use crate::hnsw::construction::select_neighbors;
use crate::hnsw::graph::NeighborhoodDiversification;
use crate::vamana::graph::VamanaIndex;
use crate::RetrieveError;
use rand::Rng;
use smallvec::SmallVec;

/// Initialize random graph with minimum degree >= log(n).
///
/// Creates initial graph structure by connecting each node to log(n) random neighbors.
/// Uses O(k) rejection sampling per node (expected), where k = log(n).
fn initialize_random_graph(index: &mut VamanaIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    let n = index.num_vectors;
    let min_degree = (n as f64).ln().ceil() as usize;
    let mut rng = rand::rng();

    for i in 0..n {
        let k = min_degree.min(n - 1);
        // Rejection sampling: draw random IDs until we have k distinct ones != i.
        // Expected draws per slot ≈ n/(n-k) ≈ 1 for k << n.  O(k) expected total.
        let mut selected = std::collections::HashSet::with_capacity(k + 1);
        selected.insert(i as u32); // sentinel: never pick self

        let mut neighbors: SmallVec<[u32; 16]> = SmallVec::with_capacity(k);
        while neighbors.len() < k {
            let j = rng.random_range(0u32..n as u32);
            if selected.insert(j) {
                neighbors.push(j);
            }
        }
        index.neighbors[i] = neighbors;
    }

    Ok(())
}

/// Greedy beam search on the flat Vamana neighbor graph.
///
/// Starting from `entry_point` (typically the medoid), explores the graph
/// closest-first using a min-heap until `ef` candidates are accumulated or
/// no candidate closer than the worst result remains.
///
/// Returns up to `ef` candidates sorted by distance (closest first),
/// with `query_id` always excluded (self is not a valid neighbor).
#[cfg(feature = "vamana")]
fn greedy_search_vamana(
    query: &[f32],
    entry_point: u32,
    query_id: u32,
    neighbors: &[SmallVec<[u32; 16]>],
    vectors: &[f32],
    dimension: usize,
    ef: usize,
) -> Vec<(u32, f32)> {
    use std::collections::BinaryHeap;

    // Min-heap: always expand the closest unexplored candidate.
    #[derive(PartialEq)]
    struct MinCand {
        id: u32,
        dist: f32,
    }
    impl Eq for MinCand {}
    impl Ord for MinCand {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other.dist.total_cmp(&self.dist)
        }
    }
    impl PartialOrd for MinCand {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    // Max-heap: track the worst result so we can prune distant candidates.
    #[derive(PartialEq)]
    struct MaxRes {
        id: u32,
        dist: f32,
    }
    impl Eq for MaxRes {}
    impl Ord for MaxRes {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.dist.total_cmp(&other.dist)
        }
    }
    impl PartialOrd for MaxRes {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let get_vec = |id: u32| -> &[f32] {
        let s = id as usize * dimension;
        &vectors[s..s + dimension]
    };

    let mut visited = std::collections::HashSet::<u32>::with_capacity(ef * 4);
    let mut candidates: BinaryHeap<MinCand> = BinaryHeap::with_capacity(ef * 2);
    let mut results: BinaryHeap<MaxRes> = BinaryHeap::with_capacity(ef + 1);

    // Mark self as visited so it is never returned as a candidate.
    visited.insert(query_id);

    // Seed the search from the entry point.
    if entry_point != query_id {
        let d = hnsw_distance::cosine_distance_normalized(query, get_vec(entry_point));
        visited.insert(entry_point);
        candidates.push(MinCand {
            id: entry_point,
            dist: d,
        });
        results.push(MaxRes {
            id: entry_point,
            dist: d,
        });
    } else {
        // Entry point is the query node itself; seed with its current neighbors.
        visited.insert(entry_point);
        for &nb in &neighbors[entry_point as usize] {
            if visited.insert(nb) {
                let d = hnsw_distance::cosine_distance_normalized(query, get_vec(nb));
                candidates.push(MinCand { id: nb, dist: d });
                results.push(MaxRes { id: nb, dist: d });
            }
        }
    }

    while let Some(MinCand {
        id: curr_id,
        dist: curr_dist,
    }) = candidates.pop()
    {
        // Early termination: best unexplored candidate is worse than worst result.
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if curr_dist > worst.dist {
                    break;
                }
            }
        }

        for &nb_id in &neighbors[curr_id as usize] {
            if visited.insert(nb_id) {
                let nb_dist = hnsw_distance::cosine_distance_normalized(query, get_vec(nb_id));

                let should_add =
                    results.len() < ef || results.peek().map_or(true, |w| nb_dist < w.dist);

                if should_add {
                    candidates.push(MinCand {
                        id: nb_id,
                        dist: nb_dist,
                    });
                    results.push(MaxRes {
                        id: nb_id,
                        dist: nb_dist,
                    });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
    }

    let mut result_vec: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.dist)).collect();
    result_vec.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    result_vec
}

/// First pass: Refine graph using RRND (Relaxed Relative Neighborhood Diversification).
///
/// Formula: dist(X_q, X_j) < α · dist(X_i, X_j) with α ≥ 1.5
#[cfg(feature = "vamana")]
fn refine_with_rrnd(index: &mut VamanaIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    let medoid = index.medoid;

    for current_id in 0..index.num_vectors {
        // Greedy beam search from medoid to collect candidates for this node.
        // Clone the query vector to release the borrow on index.vectors before
        // we pass &index.neighbors and &index.vectors to greedy_search_vamana.
        let current_vector: Vec<f32> = index.get_vector(current_id).to_vec();

        let candidates = greedy_search_vamana(
            &current_vector,
            medoid,
            current_id as u32,
            &index.neighbors,
            &index.vectors,
            index.dimension,
            index.params.ef_construction,
        );

        // Select neighbors using RRND
        let selected = select_neighbors(
            &current_vector,
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
                    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
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

    let medoid = index.medoid;

    for current_id in 0..index.num_vectors {
        let current_vector: Vec<f32> = index.get_vector(current_id).to_vec();

        let candidates = greedy_search_vamana(
            &current_vector,
            medoid,
            current_id as u32,
            &index.neighbors,
            &index.vectors,
            index.dimension,
            index.params.ef_construction,
        );

        // Select neighbors using RND
        let selected = select_neighbors(
            &current_vector,
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
                    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
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

#[cfg(all(test, feature = "vamana"))]
mod tests {
    use super::*;
    use crate::distance;
    use crate::vamana::graph::{VamanaIndex, VamanaParams};

    fn normalized_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let raw: Vec<f32> = (0..dim)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
                    })
                    .collect();
                distance::normalize(&raw)
            })
            .collect()
    }

    /// Regression test: initialize_random_graph must not add any node as its own neighbor.
    ///
    /// The old O(n²) implementation filtered correctly; this test guards against
    /// a regression where rejection sampling fails to exclude self.
    #[test]
    fn test_init_no_self_loops() {
        let n = 200;
        let dim = 8;
        let vecs = normalized_vecs(n, dim, 42);
        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.3,
            ef_construction: 50,
            ef_search: 20,
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        initialize_random_graph(&mut index).unwrap();

        for (i, nbrs) in index.neighbors.iter().enumerate() {
            assert!(
                !nbrs.contains(&(i as u32)),
                "Node {} is its own neighbor after initialization",
                i
            );
        }
    }

    /// Regression test: each node gets exactly ceil(ln(n)) neighbors from initialization.
    #[test]
    fn test_init_degree_is_log_n() {
        let n = 200;
        let dim = 8;
        let expected = (n as f64).ln().ceil() as usize;
        let vecs = normalized_vecs(n, dim, 17);
        let params = VamanaParams {
            max_degree: 32,
            alpha: 1.3,
            ef_construction: 50,
            ef_search: 20,
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        initialize_random_graph(&mut index).unwrap();

        for (i, nbrs) in index.neighbors.iter().enumerate() {
            assert_eq!(
                nbrs.len(),
                expected.min(n - 1),
                "Node {} has {} neighbors, expected {}",
                i,
                nbrs.len(),
                expected
            );
        }
    }

    /// Regression test: greedy_search_vamana must never return query_id in results.
    ///
    /// This tests the `visited.insert(query_id)` sentinel added to fix self-loops.
    #[test]
    fn test_greedy_search_excludes_self() {
        let n = 80;
        let dim = 8;
        let vecs = normalized_vecs(n, dim, 55);
        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.3,
            ef_construction: 40,
            ef_search: 20,
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.neighbors = vec![smallvec::SmallVec::new(); n]; // reset
        let medoid = compute_medoid(&index);
        index.medoid = medoid;
        initialize_random_graph(&mut index).unwrap();

        // Test for first node, the medoid, and the last node.
        for &qid in &[0u32, medoid, (n - 1) as u32] {
            let query: Vec<f32> = index.get_vector(qid as usize).to_vec();
            let results = greedy_search_vamana(
                &query,
                medoid,
                qid,
                &index.neighbors,
                &index.vectors,
                index.dimension,
                20,
            );
            assert!(
                results.iter().all(|&(id, _)| id != qid),
                "greedy_search returned query_id {} in results: {:?}",
                qid,
                results
            );
        }
    }

    /// Regression test: greedy_search_vamana results must be sorted closest-first.
    #[test]
    fn test_greedy_search_sorted() {
        let n = 100;
        let dim = 8;
        let vecs = normalized_vecs(n, dim, 99);
        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.3,
            ef_construction: 50,
            ef_search: 20,
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        let medoid = compute_medoid(&index);
        index.medoid = medoid;
        initialize_random_graph(&mut index).unwrap();

        let query: Vec<f32> = index.get_vector(0).to_vec();
        let results = greedy_search_vamana(
            &query,
            medoid,
            0,
            &index.neighbors,
            &index.vectors,
            index.dimension,
            20,
        );

        assert!(!results.is_empty(), "greedy_search returned no results");
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1,
                "Results not sorted: dist[{}]={} > dist[{}]={}",
                i - 1,
                results[i - 1].1,
                i,
                results[i].1
            );
        }
    }
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
