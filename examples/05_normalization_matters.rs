#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Why L2-normalization matters for HNSW cosine search.
//!
//! HNSW uses `1 - dot(a, b)` as its distance function, which equals cosine
//! distance only when vectors are L2-normalized. This example shows the
//! recall difference between normalized and un-normalized input.
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

    // Generate random vectors with varying magnitudes (NOT normalized)
    let raw_vectors: Vec<Vec<f32>> = (0..n).map(|i| random_vec(dim, i)).collect();
    // Normalized versions
    let norm_vectors: Vec<Vec<f32>> = raw_vectors.iter().map(|v| normalize(v)).collect();

    // Build index with UN-NORMALIZED vectors
    let mut raw_index = HNSWIndex::new(dim, 16, 100)?;
    for (id, vec) in raw_vectors.iter().enumerate() {
        raw_index.add(id as u32, vec.clone())?;
    }
    raw_index.build()?;

    // Build index with NORMALIZED vectors
    let mut norm_index = HNSWIndex::new(dim, 16, 100)?;
    for (id, vec) in norm_vectors.iter().enumerate() {
        norm_index.add(id as u32, vec.clone())?;
    }
    norm_index.build()?;

    // Measure recall against true cosine-distance nearest neighbors
    let mut recall_raw = 0.0;
    let mut recall_norm = 0.0;

    for q in 0..n_queries {
        let query_idx = (q * 19) % n;
        let raw_query = &raw_vectors[query_idx];
        let norm_query = &norm_vectors[query_idx];

        // Ground truth: brute-force cosine distance (handles any magnitude)
        let gt = brute_force_cosine_knn(norm_query, &norm_vectors, k);

        // HNSW with un-normalized input
        let raw_results = raw_index.search(raw_query, k, ef)?;
        let raw_ids: HashSet<u32> = raw_results.iter().map(|(id, _)| *id).collect();
        recall_raw += gt.intersection(&raw_ids).count() as f32 / k as f32;

        // HNSW with normalized input
        let norm_results = norm_index.search(norm_query, k, ef)?;
        let norm_ids: HashSet<u32> = norm_results.iter().map(|(id, _)| *id).collect();
        recall_norm += gt.intersection(&norm_ids).count() as f32 / k as f32;
    }

    recall_raw /= n_queries as f32;
    recall_norm /= n_queries as f32;

    println!("L2-normalization impact on HNSW recall");
    println!("  n={}, dim={}, k={}, ef={}", n, dim, k, ef);
    println!();
    println!(
        "  Un-normalized vectors: recall@{} = {:.1}%",
        k,
        recall_raw * 100.0
    );
    println!(
        "  L2-normalized vectors: recall@{} = {:.1}%",
        k,
        recall_norm * 100.0
    );
    println!();
    if recall_norm - recall_raw > 0.05 {
        println!(
            "  --> Normalization improved recall by {:.1} percentage points.",
            (recall_norm - recall_raw) * 100.0
        );
        println!("      HNSW's distance function assumes normalized input.");
        println!("      Always normalize vectors before adding them to the index.");
    } else {
        println!("  --> Recall was similar (vectors may have had uniform magnitude).");
        println!("      Normalization is still recommended for correctness.");
    }

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
