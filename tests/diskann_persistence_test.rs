#![cfg(feature = "diskann")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration test for DiskANN persistence.
//!
//! Tests the full cycle: build -> save -> load -> search

use vicinity::diskann::{DiskANNIndex, DiskANNParams, DiskANNSearcher};

/// Generate random vectors for testing.
fn generate_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    (0..n).map(|_| (0..d).map(|_| next()).collect()).collect()
}

/// L2 distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Brute force k-NN for ground truth.
fn brute_force_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut dists: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, l2_distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

/// Compute recall@k
fn compute_recall(results: &[(u32, f32)], ground_truth: &[(u32, f32)], k: usize) -> f64 {
    let gt_ids: std::collections::HashSet<_> =
        ground_truth.iter().take(k).map(|(id, _)| *id).collect();
    let result_ids: std::collections::HashSet<_> =
        results.iter().take(k).map(|(id, _)| *id).collect();
    let intersection = gt_ids.intersection(&result_ids).count();
    intersection as f64 / k as f64
}

#[test]
fn test_diskann_save_load_roundtrip() {
    // Setup
    let n = 1000;
    let d = 32;
    let k = 10;

    let vectors = generate_vectors(n, d, 42);

    // Build index
    let params = DiskANNParams {
        m: 16,
        ef_construction: 50,
        alpha: 1.2,
        ef_search: 50,
        seed: None,
        ..DiskANNParams::default()
    };

    let mut index = DiskANNIndex::new(d, params.clone()).expect("Failed to create index");

    for (i, vec) in vectors.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }

    index.build().expect("Failed to build index");

    // Save to temp directory
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let index_path = temp_dir.path().join("diskann_test");

    index.save(&index_path).expect("Failed to save index");

    // Verify files exist
    assert!(
        index_path.join("vectors.bin").exists(),
        "vectors.bin missing"
    );
    assert!(
        index_path.join("graph.index").exists(),
        "graph.index missing"
    );
    assert!(
        index_path.join("metadata.json").exists(),
        "metadata.json missing"
    );
    assert!(
        index_path.join("doc_ids.bin").exists(),
        "doc_ids.bin missing"
    );

    // Load searcher
    let mut searcher = DiskANNSearcher::load(&index_path).expect("Failed to load index");

    // Test search quality
    let queries = generate_vectors(10, d, 123);
    let mut total_recall = 0.0;

    for query in &queries {
        let ground_truth = brute_force_knn(&vectors, query, k);
        let results = searcher
            .search(query, k, params.ef_search)
            .expect("Search failed");

        let recall = compute_recall(&results, &ground_truth, k);
        total_recall += recall;
    }

    let avg_recall = total_recall / queries.len() as f64;

    println!("DiskANN persistence test:");
    println!("  Vectors: {}", n);
    println!("  Dimension: {}", d);
    println!("  Average Recall@{}: {:.2}%", k, avg_recall * 100.0);

    // Expect reasonable recall (> 50% for this simple test)
    assert!(
        avg_recall > 0.5,
        "Recall too low: {:.2}%",
        avg_recall * 100.0
    );
}

#[test]
fn test_diskann_metadata_roundtrip() {
    let n = 100;
    let d = 16;

    let vectors = generate_vectors(n, d, 42);

    let params = DiskANNParams {
        m: 8,
        ef_construction: 20,
        alpha: 1.1,
        ef_search: 20,
        seed: None,
        ..DiskANNParams::default()
    };

    let mut index = DiskANNIndex::new(d, params.clone()).expect("Failed to create index");

    for (i, vec) in vectors.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }

    index.build().expect("Failed to build index");

    // Save
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let index_path = temp_dir.path().join("diskann_meta_test");
    index.save(&index_path).expect("Failed to save index");

    // Load and verify metadata
    let metadata_path = index_path.join("metadata.json");
    let metadata: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&metadata_path).expect("Failed to open metadata"),
    )
    .expect("Failed to parse metadata");

    assert_eq!(metadata["dimension"], d);
    assert_eq!(metadata["num_vectors"], n);
    assert_eq!(metadata["params"]["m"], params.m);
    assert_eq!(
        metadata["params"]["ef_construction"],
        params.ef_construction
    );
}
