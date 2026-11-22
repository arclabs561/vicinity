#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Measuring Recall
//!
//! How to verify your index is working correctly.
//!
//! ```bash
//! cargo run --example 02_measure_recall --release
//! ```

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashSet;
use vicinity::hnsw::HNSWIndex;

fn main() -> vicinity::Result<()> {
    let dim = 64;
    let n = 1000;
    let k = 10;

    // Draw queries separately so a query cannot match itself in the index.
    let mut rng = StdRng::seed_from_u64(42);
    let mut sample = || {
        let values: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        normalize(&values)
    };
    let vectors: Vec<_> = (0..n).map(|_| sample()).collect();
    let queries: Vec<_> = (0..50).map(|_| sample()).collect();
    let truth: Vec<HashSet<u32>> = queries
        .iter()
        .map(|q| brute_force_knn(q, &vectors, k).into_iter().collect())
        .collect();

    // Build index
    let mut index = HNSWIndex::builder(dim).m(16).seed(42).build()?;
    for (id, vec) in vectors.iter().enumerate() {
        index.add(id as u32, vec.clone())?;
    }
    index.build()?;

    // Test recall across different ef values
    println!("Testing recall with {} vectors, k={}\n", n, k);
    println!("{:>8}  {:>10}", "ef", "Recall@k");
    println!("{}", "-".repeat(22));

    for ef in [10, 20, 50, 100, 200] {
        let mut total_recall = 0.0;
        for (query, gt_ids) in queries.iter().zip(&truth) {
            // Approximate: HNSW
            let approx = index.search(query, k, ef)?;
            let approx_ids: HashSet<u32> = approx.iter().map(|(id, _)| *id).collect();

            // Recall = |intersection| / k
            let recall = gt_ids.intersection(&approx_ids).count() as f32 / k as f32;
            total_recall += recall;
        }

        let avg_recall = total_recall / queries.len() as f32;
        println!("{:>8}  {:>9.1}%", ef, avg_recall * 100.0);
    }

    println!("\nLarger ef explores more candidates. Once recall reaches 100%, it cannot improve further.");

    Ok(())
}

// --- Helpers ---

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn brute_force_knn(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine_distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists.into_iter().take(k).map(|(id, _)| id).collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (norm_a * norm_b + f32::EPSILON)
}
