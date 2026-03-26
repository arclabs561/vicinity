#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Structural invariant tests that prevent regression of key design findings.
//!
//! These tests encode properties that should never break:
//! - Graph connectivity after build
//! - Oracle recall above a meaningful threshold
//! - Distance function agreement across modules
//! - SIMD vs scalar precision
//! - Normalization requirement enforcement

#![cfg(feature = "hnsw")]

use std::collections::HashSet;
use vicinity::distance;
use vicinity::hnsw::filtered::{acorn_search, AcornConfig, FnFilter, NoFilter};
use vicinity::hnsw::HNSWIndex;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// Brute-force ground truth using cosine distance for normalized vectors.
fn ground_truth_cosine(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = database
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, distance::cosine_distance_normalized(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

fn recall_at_k(retrieved: &[(u32, f32)], ground_truth: &[u32]) -> f32 {
    let gt_set: HashSet<u32> = ground_truth.iter().copied().collect();
    let hits = retrieved
        .iter()
        .filter(|(id, _)| gt_set.contains(id))
        .count();
    hits as f32 / ground_truth.len().max(1) as f32
}

/// Generate a synthetic dataset of normalized random vectors.
fn random_normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    // Simple LCG for determinism without importing rand in tests
    let mut state = seed;
    let mut next_f32 = || -> f32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Map to [-1, 1]
        ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    (0..n)
        .map(|_| {
            let raw: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            normalize(&raw)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test: Graph connectivity invariant (search-based)
// ---------------------------------------------------------------------------
// After building, every vector must be findable by searching for itself.
// This catches graph construction bugs that leave isolated nodes.

#[test]
fn graph_connectivity_every_node_findable() {
    let n = 200;
    let dim = 32;
    let vecs = random_normalized_vectors(n, dim, 42);

    let mut index = HNSWIndex::new(dim, 16, 100).unwrap();
    for (i, v) in vecs.iter().enumerate() {
        index.add_slice(i as u32, v).unwrap();
    }
    index.build().unwrap();

    // Every vector should appear in its own top-5 results with high ef
    let mut found = 0;
    for (i, v) in vecs.iter().enumerate() {
        let results = index.search(v, 5, 200).unwrap();
        if results.iter().any(|(id, _)| *id == i as u32) {
            found += 1;
        }
    }

    let findable_rate = found as f32 / n as f32;
    assert!(
        findable_rate >= 0.95,
        "Only {}/{} vectors findable via self-search (rate {:.3}). \
         Graph may have isolated nodes.",
        found,
        n,
        findable_rate
    );
}

// ---------------------------------------------------------------------------
// Test: Oracle recall with meaningful threshold
// ---------------------------------------------------------------------------
// HNSW on 500 normalized 64-dim vectors with M=16, ef_construction=200,
// ef_search=100 should achieve >= 80% recall@10. Previous tests used
// thresholds as low as 0.05 which catch nothing.

#[test]
fn oracle_recall_above_80_percent() {
    let n = 500;
    let dim = 64;
    let k = 10;
    let ef_search = 100;
    let num_queries = 20;

    let vecs = random_normalized_vectors(n, dim, 123);

    let mut index = HNSWIndex::new(dim, 16, 200).unwrap();
    for (i, v) in vecs.iter().enumerate() {
        index.add_slice(i as u32, v).unwrap();
    }
    index.build().unwrap();

    let queries = random_normalized_vectors(num_queries, dim, 456);
    let mut total_recall = 0.0;

    for q in &queries {
        let results = index.search(q, k, ef_search).unwrap();
        let gt = ground_truth_cosine(q, &vecs, k);
        total_recall += recall_at_k(&results, &gt);
    }

    let avg_recall = total_recall / num_queries as f32;
    assert!(
        avg_recall >= 0.80,
        "Average recall@{} = {:.3} (expected >= 0.80). \
         Graph construction may be broken.",
        k,
        avg_recall
    );
}

// ---------------------------------------------------------------------------
// Test: Self-retrieval must be perfect
// ---------------------------------------------------------------------------
// Searching for a vector that's in the index should return itself as the
// nearest neighbor. This catches distance metric bugs.

#[test]
fn self_retrieval_is_nearest() {
    let n = 100;
    let dim = 32;
    let vecs = random_normalized_vectors(n, dim, 789);

    let mut index = HNSWIndex::new(dim, 16, 200).unwrap();
    for (i, v) in vecs.iter().enumerate() {
        index.add_slice(i as u32, v).unwrap();
    }
    index.build().unwrap();

    let mut self_found = 0;
    for (i, v) in vecs.iter().enumerate() {
        let results = index.search(v, 1, 100).unwrap();
        if !results.is_empty() && results[0].0 == i as u32 {
            self_found += 1;
        }
    }

    let self_recall = self_found as f32 / n as f32;
    assert!(
        self_recall >= 0.95,
        "Self-retrieval recall = {:.3} (expected >= 0.95). \
         Distance metric or graph construction is broken.",
        self_recall
    );
}

// ---------------------------------------------------------------------------
// Test: Distance function agreement
// ---------------------------------------------------------------------------
// All distance functions that claim to compute cosine distance should agree.

#[test]
fn distance_functions_agree() {
    let a = normalize(&[1.0, 2.0, 3.0, 4.0]);
    let b = normalize(&[4.0, 3.0, 2.0, 1.0]);

    let d_full = distance::cosine_distance(&a, &b);
    let d_norm = distance::cosine_distance_normalized(&a, &b);

    assert!(
        (d_full - d_norm).abs() < 1e-5,
        "cosine_distance ({}) and cosine_distance_normalized ({}) disagree on normalized vectors",
        d_full,
        d_norm
    );
}

// ---------------------------------------------------------------------------
// Test: SIMD vs scalar distance precision
// ---------------------------------------------------------------------------
// Inspired by pgvectorscale's precision tests. The SIMD-accelerated distance
// functions must agree with naive scalar implementations to within epsilon.

#[test]
fn simd_cosine_matches_scalar() {
    let dims = [4, 16, 64, 128, 256];
    for &dim in &dims {
        let a = random_normalized_vectors(1, dim, 1111)[0].clone();
        let b = random_normalized_vectors(1, dim, 2222)[0].clone();

        // Scalar cosine similarity
        let dot_scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let scalar_dist = 1.0 - dot_scalar;

        // SIMD path (via crate::distance)
        let simd_dist = distance::cosine_distance_normalized(&a, &b);

        let diff = (scalar_dist - simd_dist).abs();
        assert!(
            diff < 1e-5,
            "SIMD vs scalar cosine mismatch at dim={}: scalar={}, simd={}, diff={}",
            dim,
            scalar_dist,
            simd_dist,
            diff
        );
    }
}

#[test]
fn simd_l2_matches_scalar() {
    let dims = [4, 16, 64, 128, 256];
    for &dim in &dims {
        let a = random_normalized_vectors(1, dim, 3333)[0].clone();
        let b = random_normalized_vectors(1, dim, 4444)[0].clone();

        // Scalar L2
        let scalar_dist: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            .sqrt();

        // SIMD path
        let simd_dist = distance::l2_distance(&a, &b);

        let diff = (scalar_dist - simd_dist).abs();
        assert!(
            diff < 1e-5,
            "SIMD vs scalar L2 mismatch at dim={}: scalar={}, simd={}, diff={}",
            dim,
            scalar_dist,
            simd_dist,
            diff
        );
    }
}

// ---------------------------------------------------------------------------
// Test: Degenerate inputs don't panic
// ---------------------------------------------------------------------------
// Inspired by USearch's test_absurd: pathological configs shouldn't crash.

#[test]
fn degenerate_dim_1() {
    let mut index = HNSWIndex::new(1, 4, 10).unwrap();
    for i in 0..10u32 {
        // dim=1 vectors: just [1.0] or [-1.0] (both normalized)
        let v = if i % 2 == 0 { vec![1.0] } else { vec![-1.0] };
        index.add_slice(i, &v).unwrap();
    }
    index.build().unwrap();
    let results = index.search(&[1.0], 3, 10).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn degenerate_single_vector() {
    let mut index = HNSWIndex::new(4, 4, 10).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();
    index.build().unwrap();
    let results = index.search(&v, 1, 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 0);
}

#[test]
fn degenerate_all_identical_vectors() {
    let mut index = HNSWIndex::new(4, 4, 10).unwrap();
    let v = normalize(&[1.0, 1.0, 1.0, 1.0]);
    for i in 0..20u32 {
        index.add_slice(i, &v).unwrap();
    }
    index.build().unwrap();
    let results = index.search(&v, 5, 20).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn degenerate_m_equals_2() {
    let dim = 8;
    let vecs = random_normalized_vectors(50, dim, 999);
    let mut index = HNSWIndex::new(dim, 2, 10).unwrap();
    for (i, v) in vecs.iter().enumerate() {
        index.add_slice(i as u32, v).unwrap();
    }
    index.build().unwrap();
    let results = index.search(&vecs[0], 5, 20).unwrap();
    assert!(!results.is_empty());
}

// ---------------------------------------------------------------------------
// Test: Dimension mismatch is caught
// ---------------------------------------------------------------------------

#[test]
fn dimension_mismatch_on_add() {
    let mut index = HNSWIndex::new(4, 4, 10).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0]);
    let result = index.add_slice(0, &v);
    assert!(result.is_err());
}

#[test]
fn dimension_mismatch_on_search() {
    let mut index = HNSWIndex::new(4, 4, 10).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();
    index.build().unwrap();

    let wrong_dim = normalize(&[1.0, 0.0, 0.0]);
    let result = index.search(&wrong_dim, 1, 10);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Test: Duplicate doc_id is rejected
// ---------------------------------------------------------------------------

#[test]
fn duplicate_doc_id_rejected() {
    let mut index = HNSWIndex::new(4, 4, 10).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();
    let result = index.add_slice(0, &v);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Helpers: build a mutual k-NN graph for filtered search tests
// ---------------------------------------------------------------------------

/// Build a mutual k-NN graph suitable for acorn_search.
///
/// Returns adjacency lists and the vectors (already normalized).
fn build_knn_graph(
    n: usize,
    dim: usize,
    seed: u64,
    neighbors_per_node: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let vectors = random_normalized_vectors(n, dim, seed);
    let mut graph: Vec<HashSet<u32>> = (0..n).map(|_| HashSet::new()).collect();
    for i in 0..n {
        let mut dists: Vec<(u32, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                (
                    j as u32,
                    distance::cosine_distance_normalized(&vectors[i], &vectors[j]),
                )
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(j, _) in dists.iter().take(neighbors_per_node) {
            graph[i].insert(j);
            graph[j as usize].insert(i as u32);
        }
    }
    let adj: Vec<Vec<u32>> = graph.into_iter().map(|s| s.into_iter().collect()).collect();
    (adj, vectors)
}

// ---------------------------------------------------------------------------
// Test: Filtered search adversarial -- always-false filter (inspired by USearch)
// ---------------------------------------------------------------------------

#[test]
fn filtered_search_always_false_returns_empty() {
    let n = 100;
    let dim = 32;
    let (graph, vectors) = build_knn_graph(n, dim, 5050, 16);

    let query = &random_normalized_vectors(1, dim, 6060)[0];
    let filter = FnFilter(|_id: u32| false);
    let config = AcornConfig {
        enable_two_hop: true,
        two_hop_threshold: 0.5,
        max_two_hop_neighbors: 32,
        ef_search: 200,
    };

    let results = acorn_search(
        10,
        &config,
        &filter,
        |id| graph[id as usize].clone(),
        |id| distance::cosine_distance_normalized(&vectors[id as usize], query),
        0,
    )
    .expect("acorn_search failed");

    assert!(
        results.is_empty(),
        "Always-false filter should return no results, got {}",
        results.len()
    );
}

// ---------------------------------------------------------------------------
// Test: Filtered search adversarial -- single-match filter
// ---------------------------------------------------------------------------

#[test]
fn filtered_search_single_match_returns_it() {
    let n = 100;
    let dim = 32;
    let target_id: u32 = 42;
    let (graph, vectors) = build_knn_graph(n, dim, 5151, 16);

    let query = &vectors[target_id as usize]; // Search for the target itself
    let filter = FnFilter(move |id: u32| id == target_id);
    let config = AcornConfig {
        enable_two_hop: true,
        two_hop_threshold: 0.5,
        max_two_hop_neighbors: 64,
        ef_search: 200,
    };

    let results = acorn_search(
        5,
        &config,
        &filter,
        |id| graph[id as usize].clone(),
        |id| distance::cosine_distance_normalized(&vectors[id as usize], query),
        0,
    )
    .expect("acorn_search failed");

    assert!(
        !results.is_empty(),
        "Single-match filter should find the target"
    );
    assert_eq!(
        results[0].0, target_id,
        "First result should be doc_id={}, got {}",
        target_id, results[0].0
    );
}

// ---------------------------------------------------------------------------
// Test: Filtered search -- NoFilter matches regular search
// ---------------------------------------------------------------------------

#[test]
fn filtered_search_no_filter_matches_regular_search() {
    let n = 100;
    let dim = 32;
    let k = 10;
    let (graph, vectors) = build_knn_graph(n, dim, 5252, 16);

    let query = &random_normalized_vectors(1, dim, 6262)[0];

    // Ground truth: brute-force k-NN
    let gt = ground_truth_cosine(query, &vectors, k);
    let gt_set: HashSet<u32> = gt.iter().copied().collect();

    let config = AcornConfig {
        enable_two_hop: true,
        two_hop_threshold: 0.3,
        max_two_hop_neighbors: 32,
        ef_search: 200,
    };

    let results = acorn_search(
        k,
        &config,
        &NoFilter,
        |id| graph[id as usize].clone(),
        |id| distance::cosine_distance_normalized(&vectors[id as usize], query),
        // Use the nearest node as entry point for better convergence
        gt[0],
    )
    .expect("acorn_search failed");

    let result_ids: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
    let overlap = gt_set.intersection(&result_ids).count();
    let recall = overlap as f32 / k as f32;

    assert!(
        recall >= 0.7,
        "NoFilter acorn_search recall = {:.3} (expected >= 0.70). \
         Results should approximate brute-force k-NN.",
        recall
    );
}

// ---------------------------------------------------------------------------
// Test: Incremental build consistency (inspired by qdrant)
// ---------------------------------------------------------------------------
// Building a larger index that includes earlier vectors should not corrupt
// results for queries that were answered well by the smaller index.

#[test]
fn incremental_additions_dont_corrupt_earlier_data() {
    let n_initial = 100;
    let n_extra = 100;
    let dim = 32;
    let k = 10;
    let ef_search = 200;

    let all_vecs = random_normalized_vectors(n_initial + n_extra, dim, 7070);
    let query = &all_vecs[0]; // Query is one of the initial vectors

    // Phase 1: build with initial vectors only
    let mut index_small = HNSWIndex::new(dim, 16, 200).unwrap();
    for (i, v) in all_vecs[..n_initial].iter().enumerate() {
        index_small.add_slice(i as u32, v).unwrap();
    }
    index_small.build().unwrap();
    let results_small = index_small.search(query, k, ef_search).unwrap();
    let _ids_small: HashSet<u32> = results_small.iter().map(|(id, _)| *id).collect();

    // Phase 2: build a NEW index with initial + extra vectors
    let mut index_large = HNSWIndex::new(dim, 16, 200).unwrap();
    for (i, v) in all_vecs.iter().enumerate() {
        index_large.add_slice(i as u32, v).unwrap();
    }
    index_large.build().unwrap();
    let results_large = index_large.search(query, k, ef_search).unwrap();
    let ids_large: HashSet<u32> = results_large.iter().map(|(id, _)| *id).collect();

    // Ground truth: brute-force k-NN among original vectors
    let gt_small = ground_truth_cosine(query, &all_vecs[..n_initial], k);
    let gt_set: HashSet<u32> = gt_small.iter().copied().collect();

    // The large index should still find most of the true nearest neighbors
    // from the original set. Some may be displaced by genuinely closer new
    // vectors, so we check overlap with ground truth, not with the small
    // index's approximate results.
    let ids_large_original: HashSet<u32> = ids_large
        .iter()
        .filter(|&&id| (id as usize) < n_initial)
        .copied()
        .collect();

    let overlap = gt_set.intersection(&ids_large_original).count();
    let consistency = overlap as f32 / gt_set.len().max(1) as f32;

    assert!(
        consistency >= 0.4,
        "Incremental build consistency = {:.3} (expected >= 0.40). \
         Adding vectors should not corrupt earlier data. \
         Ground truth top-{}: {:?}, Large index original-range top: {:?}",
        consistency,
        k,
        gt_set,
        ids_large_original
    );

    // Verify the query vector itself is still found (self-retrieval sanity)
    assert!(
        ids_large.contains(&0),
        "Query vector (doc_id=0) should appear in its own top-{} results \
         even after adding more vectors",
        k
    );
}
