#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end tests validating HNSW actually works.
//!
//! These tests verify that the real HNSWIndex achieves reasonable recall,
//! not just that the code compiles.
//!
//! HNSW uses cosine distance internally, so vectors must be normalized
//! and ground truth must use cosine distance.

#![cfg(feature = "hnsw")]

use std::collections::HashSet;

use vicinity::hnsw::{HNSWIndex, HNSWParams};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = norm(v);
    if n < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// Cosine distance = 1 - cosine_similarity
/// For normalized vectors: cosine_similarity = dot product
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

fn compute_ground_truth(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut distances: Vec<(u32, f32)> = database
        .iter()
        .enumerate()
        .map(|(i, vec)| (i as u32, cosine_distance(query, vec)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).map(|(id, _)| id).collect()
}

fn recall_at_k(ground_truth: &[u32], retrieved: &[u32], k: usize) -> f32 {
    let gt_set: HashSet<u32> = ground_truth.iter().take(k).copied().collect();
    let ret_set: HashSet<u32> = retrieved.iter().take(k).copied().collect();
    gt_set.intersection(&ret_set).count() as f32 / k as f32
}

/// Create clustered dataset with normalized vectors (required for cosine distance).
fn create_clustered_dataset(
    n_clusters: usize,
    points_per_cluster: usize,
    dim: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let mut vectors = Vec::with_capacity(n_clusters * points_per_cluster);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 10.0 - 5.0).collect();
            normalize(&v)
        })
        .collect();

    // Generate points around each center (with small perturbation)
    for center in &centers {
        for _ in 0..points_per_cluster {
            let point: Vec<f32> = center
                .iter()
                .map(|&c| c + rng.random::<f32>() * 0.2 - 0.1)
                .collect();
            vectors.push(normalize(&point));
        }
    }

    vectors
}

#[test]
fn test_hnsw_achieves_reasonable_recall() {
    // Uses smaller dataset for fast CI; see examples/ for large-scale benchmarks.
    let _n_vectors = 1000;
    let _n_queries = 20;
    let dim = 32;
    let k = 10;

    // Create clustered dataset (easier for ANN than uniform random)
    let database = create_clustered_dataset(50, 20, dim, 42); // 1000 vectors
    let queries = create_clustered_dataset(2, 10, dim, 123); // 20 queries

    // Build HNSW index with higher params for better graph quality
    let params = HNSWParams {
        m: 32,
        m_max: 64,
        ef_construction: 400,
        ef_search: 200,
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).expect("Failed to create index");

    for (i, vec) in database.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }
    index.build().expect("Failed to build index");

    // Compute ground truth and measure recall
    let mut total_recall = 0.0;
    for query in &queries {
        let gt = compute_ground_truth(query, &database, k);
        let results = index.search(query, k, 100).expect("Search failed");
        let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        total_recall += recall_at_k(&gt, &retrieved, k);
    }

    let mean_recall = total_recall / queries.len() as f32;

    assert!(
        mean_recall >= 0.90,
        "HNSW recall too low: {:.1}% (expected >= 90%)",
        mean_recall * 100.0
    );

    eprintln!("HNSW recall@{} with ef=100: {:.1}%", k, mean_recall * 100.0);
}

#[test]
fn test_hnsw_recall_increases_with_ef() {
    // Note: Using smaller dataset due to scaling issues
    let _n_vectors = 1000;
    let dim = 32;
    let k = 10;

    let database = create_clustered_dataset(50, 20, dim, 42); // 1000 vectors
    let queries = create_clustered_dataset(2, 5, dim, 999); // 10 queries

    // Use higher params for better graph quality
    let params = HNSWParams {
        m: 32,
        m_max: 64,
        ef_construction: 400,
        ef_search: 50,
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).expect("Failed to create index");

    for (i, vec) in database.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }
    index.build().expect("Failed to build index");

    // Compute ground truth once
    let ground_truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| compute_ground_truth(q, &database, k))
        .collect();

    // Measure recall at different ef values
    let ef_values = [32, 64, 128, 256, 512];
    let mut recalls = Vec::new();

    for &ef in &ef_values {
        let mut total_recall = 0.0;
        for (query, gt) in queries.iter().zip(&ground_truths) {
            let results = index.search(query, k, ef).expect("Search failed");
            let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
            total_recall += recall_at_k(gt, &retrieved, k);
        }
        let mean_recall = total_recall / queries.len() as f32;
        recalls.push(mean_recall);
        eprintln!("ef={}: recall@{}={:.1}%", ef, k, mean_recall * 100.0);
    }

    // Recall should generally increase with ef (not strictly monotonic due to randomness)
    // At minimum, high ef should be better than low ef
    assert!(
        recalls[4] >= recalls[0],
        "Recall at ef=256 ({:.1}%) should be >= recall at ef=16 ({:.1}%)",
        recalls[4] * 100.0,
        recalls[0] * 100.0
    );

    assert!(
        recalls[4] >= 0.90,
        "Recall at ef=256 too low: {:.1}%",
        recalls[4] * 100.0
    );
}

#[test]
fn test_hnsw_search_returns_sorted_results() {
    let dim = 16;
    let database = create_clustered_dataset(5, 20, dim, 42);
    let query = vec![0.0; dim];

    let mut index = HNSWIndex::new(dim, 8, 8).expect("Failed to create index");
    for (i, vec) in database.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }
    index.build().expect("Failed to build index");

    let results = index.search(&query, 10, 50).expect("Search failed");

    // Results should be sorted by distance (ascending)
    for i in 1..results.len() {
        assert!(
            results[i - 1].1 <= results[i].1,
            "Results not sorted: {} > {} at position {}",
            results[i - 1].1,
            results[i].1,
            i
        );
    }
}

#[test]
fn test_hnsw_handles_single_vector() {
    let dim = 8;
    let vec = normalize(&vec![1.0; dim]);

    let mut index = HNSWIndex::new(dim, 8, 8).expect("Failed to create index");
    index.add(0, vec.clone()).expect("Failed to add vector");
    index.build().expect("Failed to build index");

    let results = index.search(&vec, 1, 10).expect("Search failed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 0);
}

/// Test that searching for an existing vector returns that vector first.
/// This is a fundamental correctness test - a vector should be its own nearest neighbor.
#[test]
fn test_self_retrieval() {
    let dim = 32;
    let database = create_clustered_dataset(10, 10, dim, 42); // 100 vectors

    let mut index = HNSWIndex::new(dim, 16, 16).expect("Failed to create index");
    for (i, vec) in database.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }
    index.build().expect("Failed to build index");

    // Search for each vector in the database - it should find itself
    let mut self_found = 0;
    for (i, query) in database.iter().enumerate() {
        let results = index.search(query, 1, 50).expect("Search failed");
        if !results.is_empty() && results[0].0 == i as u32 {
            self_found += 1;
        }
    }

    let self_rate = self_found as f32 / database.len() as f32;
    assert!(
        self_rate >= 0.95,
        "Self-retrieval rate too low: {:.1}% (expected >= 95%)",
        self_rate * 100.0
    );
    eprintln!("Self-retrieval rate: {:.1}%", self_rate * 100.0);
}

#[test]
fn test_scaling_recall() {
    let dim = 32;
    let k = 10;
    let ef = 100;

    for n in [100, 500, 1000, 2000] {
        let database = create_clustered_dataset(n / 20, 20, dim, 42);
        let queries = create_clustered_dataset(2, 5, dim, 999);

        let params = HNSWParams {
            m: 16,
            m_max: 16,
            ef_construction: 200,
            ef_search: ef,
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).expect("Failed to create index");

        for (i, vec) in database.iter().enumerate() {
            index
                .add(i as u32, vec.clone())
                .expect("Failed to add vector");
        }
        index.build().expect("Failed to build index");

        let ground_truths: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| compute_ground_truth(q, &database, k))
            .collect();

        let mut total_recall = 0.0;
        for (query, gt) in queries.iter().zip(&ground_truths) {
            let results = index.search(query, k, ef).expect("Search failed");
            let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
            total_recall += recall_at_k(gt, &retrieved, k);
        }
        let mean_recall = total_recall / queries.len() as f32;
        eprintln!("n={}: recall@{}={:.1}%", n, k, mean_recall * 100.0);
    }
}

#[test]
fn test_compare_neighbor_selection() {
    use vicinity::hnsw::NeighborhoodDiversification;

    let dim = 32;
    let k = 10;
    let ef = 100;
    let _n = 1000; // 50 clusters * 20 points = 1000 vectors

    let database = create_clustered_dataset(50, 20, dim, 42);
    let queries = create_clustered_dataset(2, 5, dim, 999);

    let ground_truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| compute_ground_truth(q, &database, k))
        .collect();

    // Test different diversification strategies
    let strategies = [
        ("RND", NeighborhoodDiversification::RelativeNeighborhood),
        (
            "MOND_60",
            NeighborhoodDiversification::MaximumOriented {
                min_angle_degrees: 60.0,
            },
        ),
        (
            "RRND_1.3",
            NeighborhoodDiversification::RelaxedRelative { alpha: 1.3 },
        ),
    ];

    for (name, strategy) in strategies {
        let params = HNSWParams {
            m: 16,
            m_max: 32,            // More connections in base layer
            ef_construction: 400, // Higher ef_construction
            ef_search: ef,
            neighborhood_diversification: strategy,
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).expect("Failed to create index");

        for (i, vec) in database.iter().enumerate() {
            index
                .add(i as u32, vec.clone())
                .expect("Failed to add vector");
        }
        index.build().expect("Failed to build index");

        let mut total_recall = 0.0;
        for (query, gt) in queries.iter().zip(&ground_truths) {
            let results = index.search(query, k, ef).expect("Search failed");
            let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
            total_recall += recall_at_k(gt, &retrieved, k);
        }
        let mean_recall = total_recall / queries.len() as f32;
        eprintln!("{}: recall@{}={:.1}%", name, k, mean_recall * 100.0);
    }
}

/// Test with uniform random data (harder than clustered).
/// This is a stress test - uniform data has no structure for HNSW to exploit.
#[test]
fn test_uniform_random_data() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let dim = 32;
    let n = 500;
    let k = 10;

    let mut rng = StdRng::seed_from_u64(12345);

    // Uniform random normalized vectors (worst case for ANN)
    let database: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            normalize(&v)
        })
        .collect();

    let queries: Vec<Vec<f32>> = (0..10)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            normalize(&v)
        })
        .collect();

    let params = HNSWParams {
        m: 32,
        m_max: 64,
        ef_construction: 400,
        ef_search: 200,
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).expect("Failed to create index");

    for (i, vec) in database.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }
    index.build().expect("Failed to build index");

    let ground_truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| compute_ground_truth(q, &database, k))
        .collect();

    let mut total_recall = 0.0;
    for (query, gt) in queries.iter().zip(&ground_truths) {
        let results = index.search(query, k, 200).expect("Search failed");
        let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        total_recall += recall_at_k(gt, &retrieved, k);
    }
    let mean_recall = total_recall / queries.len() as f32;

    // Uniform data is harder - expect lower recall than clustered
    // But still should achieve something (random baseline is k/n = 2%)
    assert!(
        mean_recall >= 0.20,
        "Recall on uniform data too low: {:.1}% (expected >= 20%)",
        mean_recall * 100.0
    );

    eprintln!("Uniform random recall@{}: {:.1}%", k, mean_recall * 100.0);
}

/// Test that returned distances are actually correct (not just IDs).
/// Validates that the distance values match our independent computation.
#[test]
fn test_returned_distances_correct() {
    let dim = 16;
    let database = create_clustered_dataset(5, 20, dim, 42); // 100 vectors

    let mut index = HNSWIndex::new(dim, 16, 16).expect("Failed to create index");
    for (i, vec) in database.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }
    index.build().expect("Failed to build index");

    let query = &database[0];
    let results = index.search(query, 10, 50).expect("Search failed");

    // Verify each returned distance matches our computation
    for (id, returned_dist) in &results {
        let vec = &database[*id as usize];
        let expected_dist = cosine_distance(query, vec);

        assert!(
            (returned_dist - expected_dist).abs() < 1e-5,
            "Distance mismatch for id {}: returned {}, expected {}",
            id,
            returned_dist,
            expected_dist
        );
    }
}

#[test]
fn test_high_ef_search() {
    let dim = 32;
    let k = 10;

    let database = create_clustered_dataset(50, 20, dim, 42);
    let queries = create_clustered_dataset(2, 5, dim, 999);

    let ground_truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| compute_ground_truth(q, &database, k))
        .collect();

    let params = HNSWParams {
        m: 32,
        m_max: 64,
        ef_construction: 500,
        ef_search: 50,
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).expect("Failed to create index");

    for (i, vec) in database.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }
    index.build().expect("Failed to build index");

    for ef in [50, 100, 200, 400, 800] {
        let mut total_recall = 0.0;
        for (query, gt) in queries.iter().zip(&ground_truths) {
            let results = index.search(query, k, ef).expect("Search failed");
            let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
            total_recall += recall_at_k(gt, &retrieved, k);
        }
        let mean_recall = total_recall / queries.len() as f32;
        eprintln!("ef={}: recall@{}={:.1}%", ef, k, mean_recall * 100.0);
    }
}

// =============================================================================
// Streaming / IndexOps E2E Tests
// =============================================================================

use vicinity::hnsw::{InPlaceConfig, InPlaceIndex, MappedInPlaceIndex};
use vicinity::streaming::{IndexOps, StreamingCoordinator};

/// End-to-end test: streaming updates via IndexOps trait
#[test]
fn test_streaming_inplace_insert_search_delete() {
    let dim = 8;
    let k = 5;

    // Create normalized test vectors
    let vectors: Vec<Vec<f32>> = (0..20)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|j| ((i * 7 + j) % 13) as f32).collect();
            normalize(&v)
        })
        .collect();

    // Create MappedInPlaceIndex with external ID tracking
    let mut index = MappedInPlaceIndex::new(dim, InPlaceConfig::default());

    // Insert vectors via IndexOps trait
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u32, v.clone()).expect("Insert failed");
    }

    // Search - should find inserted vectors
    let query = &vectors[0];
    let results = index.search(query, k).expect("Search failed");

    // Verify we get results back
    assert!(!results.is_empty(), "Should find vectors after insert");

    // The closest result should be the query vector itself (id=0)
    let ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&0), "Query vector should be in top-k results");

    // Delete the first vector
    index.delete(0).expect("Delete failed");

    // Search again - should NOT find deleted vector
    let results_after_delete = index.search(query, k).expect("Search failed");
    let ids_after: Vec<u32> = results_after_delete.iter().map(|(id, _)| *id).collect();
    assert!(
        !ids_after.contains(&0),
        "Deleted vector should not appear in results"
    );
}

/// End-to-end test: StreamingCoordinator wrapping InPlaceIndex
#[test]
fn test_streaming_coordinator_with_inplace() {
    let dim = 8;

    let vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|j| ((i * 11 + j) % 17) as f32).collect();
            normalize(&v)
        })
        .collect();

    // StreamingCoordinator wraps any IndexOps implementation
    let inner = InPlaceIndex::new(dim, InPlaceConfig::default());
    let mut coordinator = StreamingCoordinator::new(inner);

    // Batch inserts through coordinator
    for (i, v) in vectors.iter().enumerate() {
        coordinator
            .insert(i as u32, v.clone())
            .expect("Insert failed");
    }

    // Search through coordinator
    let query = &vectors[25];
    let results = coordinator.search(query, 10).expect("Search failed");

    assert!(
        !results.is_empty(),
        "Should find vectors through coordinator"
    );

    // Note: With raw InPlaceIndex, external IDs are ignored (it generates its own)
    // The test verifies the pipeline works, not ID preservation
    // For ID preservation, use MappedInPlaceIndex
}

/// End-to-end test: recall quality after streaming updates
#[test]
fn test_streaming_recall_after_updates() {
    let dim = 16;
    let n_initial = 100;
    let n_added = 50;
    let k = 10;

    // Create two batches of vectors
    let initial: Vec<Vec<f32>> = (0..n_initial)
        .map(|i| {
            normalize(
                &(0..dim)
                    .map(|j| ((i * 7 + j) % 19) as f32)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    let added: Vec<Vec<f32>> = (0..n_added)
        .map(|i| {
            normalize(
                &(0..dim)
                    .map(|j| ((i * 13 + j + 100) % 23) as f32)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    let mut index = MappedInPlaceIndex::new(dim, InPlaceConfig::default());

    // Insert initial batch
    for (i, v) in initial.iter().enumerate() {
        index.insert(i as u32, v.clone()).unwrap();
    }

    // Add more vectors (simulating streaming updates)
    for (i, v) in added.iter().enumerate() {
        index.insert((n_initial + i) as u32, v.clone()).unwrap();
    }

    // Combine all vectors for ground truth computation
    let all_vectors: Vec<Vec<f32>> = initial.iter().chain(added.iter()).cloned().collect();

    // Test recall on a few queries
    let mut total_recall = 0.0;
    let n_queries = 10;

    for query_idx in 0..n_queries {
        let query = &all_vectors[query_idx * 10];

        // Compute ground truth
        let gt = compute_ground_truth(query, &all_vectors, k);

        // Search via index
        let results = index.search(query, k).unwrap();
        let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();

        let recall = recall_at_k(&gt, &retrieved, k);
        total_recall += recall;
    }

    let mean_recall = total_recall / n_queries as f32;
    eprintln!("Streaming HNSW recall@{}: {:.1}%", k, mean_recall * 100.0);

    // InPlaceIndex should achieve reasonable recall (>50%)
    assert!(
        mean_recall > 0.5,
        "Streaming HNSW recall should be >50%, got {:.1}%",
        mean_recall * 100.0
    );
}
