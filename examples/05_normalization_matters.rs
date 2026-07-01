#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Why L2-normalization matters for HNSW cosine search.
//!
//! HNSW uses `1 - dot(a, b)` as its distance function, which equals cosine
//! distance only when vectors are L2-normalized. This example shows the
//! public API contract: raw cosine vectors are rejected by default, manual
//! normalization works, and `auto_normalize(true)` applies the same transform
//! on add and search.
//!
//! ```bash
//! cargo run --example 05_normalization_matters --release
//! ```

use std::collections::HashSet;
use vicinity::hnsw::HNSWIndex;

fn main() -> vicinity::Result<()> {
    let dim = 128;
    let n = 2000;
    let k = 10;
    let ef = 100;
    let n_queries = 100;

    // Generate random vectors with varying magnitudes (not normalized).
    let raw_vectors: Vec<Vec<f32>> = (0..n).map(|i| random_vec(dim, i)).collect();
    let norm_vectors: Vec<Vec<f32>> = raw_vectors.iter().map(|v| normalize(v)).collect();

    println!("L2-normalization impact on HNSW cosine search");
    println!("  n={}, dim={}, k={}, ef={}", n, dim, k, ef);
    println!();

    // The default cosine path rejects clearly un-normalized input. This is
    // better than silently ranking by vector magnitude.
    let mut invalid_index = HNSWIndex::builder(dim).m(16).m_max(100).build()?;
    match invalid_index.add_slice(0, &raw_vectors[0]) {
        Ok(()) => println!("  raw input without normalization: unexpectedly accepted"),
        Err(err) => println!("  raw input without normalization: rejected ({err})"),
    }
    println!();

    let mut manual_index = HNSWIndex::builder(dim)
        .m(16)
        .m_max(100)
        .auto_normalize(false)
        .build()?;
    for (id, vec) in norm_vectors.iter().enumerate() {
        manual_index.add_slice(id as u32, vec)?;
    }
    manual_index.build()?;

    let mut auto_index = HNSWIndex::builder(dim)
        .m(16)
        .m_max(100)
        .auto_normalize(true)
        .build()?;
    for (id, vec) in raw_vectors.iter().enumerate() {
        auto_index.add_slice(id as u32, vec)?;
    }
    auto_index.build()?;

    let manual_recall = recall_against_cosine_truth(
        &manual_index,
        &norm_vectors,
        &norm_vectors,
        k,
        ef,
        n_queries,
    )?;
    let auto_recall =
        recall_against_cosine_truth(&auto_index, &raw_vectors, &norm_vectors, k, ef, n_queries)?;

    println!("{:>28}  {:>10}", "Path", "Recall@10");
    println!("{:->28}  {:->10}", "", "");
    println!(
        "{:>28}  {:>9.1}%",
        "manual normalize()",
        manual_recall * 100.0
    );
    println!(
        "{:>28}  {:>9.1}%",
        "auto_normalize(true)",
        auto_recall * 100.0
    );
    println!();
    println!("Contract: HNSW's cosine fast path needs unit-norm vectors.");
    println!("Use `auto_normalize(true)` when callers may pass raw embeddings.");
    println!("Manual normalization remains useful when vectors are already prepared upstream.");

    Ok(())
}

// --- Helpers ---

fn random_vec(dim: usize, seed: usize) -> Vec<f32> {
    // Varying magnitudes: some vectors much larger than others
    let scale = 1.0 + (seed % 10) as f32 * 2.0;
    (0..dim)
        .map(|i| ((seed * 31 + i * 17) as f32 * 0.001).sin() * scale)
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn recall_against_cosine_truth(
    index: &HNSWIndex,
    query_vectors: &[Vec<f32>],
    truth_vectors: &[Vec<f32>],
    k: usize,
    ef: usize,
    n_queries: usize,
) -> vicinity::Result<f32> {
    let mut recall = 0.0;

    for q in 0..n_queries {
        let query_idx = (q * 19) % query_vectors.len();
        let truth_query = &truth_vectors[query_idx];
        let gt = brute_force_cosine_knn(truth_query, truth_vectors, k);

        let results = index.search(&query_vectors[query_idx], k, ef)?;
        let result_ids: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
        recall += gt.intersection(&result_ids).count() as f32 / k as f32;
    }

    Ok(recall / n_queries as f32)
}

fn brute_force_cosine_knn(query: &[f32], data: &[Vec<f32>], k: usize) -> HashSet<u32> {
    let mut dists: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v).map(|(a, b)| a * b).sum();
            (i as u32, 1.0 - dot) // cosine distance for normalized vectors
        })
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists.into_iter().take(k).map(|(id, _)| id).collect()
}
