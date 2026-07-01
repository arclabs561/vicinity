#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Measuring Recall
//!
//! How to verify your index is working correctly.
//!
//! ```bash
//! cargo run --example 02_measure_recall --release
//! ```

use std::collections::HashSet;
use vicinity::hnsw::HNSWIndex;

fn main() -> vicinity::Result<()> {
    let dim = 64;
    let n = 1000;
    let k = 10;

    // Generate normalized random vectors
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| normalize(&random_vec(dim, i))).collect();

    // Build index
    let mut index = HNSWIndex::new(dim, 16, 32)?;
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
        let n_queries = 50;

        for q in 0..n_queries {
            let query = &vectors[(q * 17) % n]; // Sample queries

            // Ground truth: brute force
            let gt = brute_force_knn(query, &vectors, k);
            let gt_ids: HashSet<u32> = gt.into_iter().collect();

            // Approximate: HNSW
            let approx = index.search(query, k, ef)?;
            let approx_ids: HashSet<u32> = approx.iter().map(|(id, _)| *id).collect();

            // Recall = |intersection| / k
            let recall = gt_ids.intersection(&approx_ids).count() as f32 / k as f32;
            total_recall += recall;
        }

        let avg_recall = total_recall / n_queries as f32;
        println!("{:>8}  {:>9.1}%", ef, avg_recall * 100.0);
    }

    println!("\nIn this run, ef controls the recall/speed tradeoff.");
    println!("Higher ef = better recall, but slower queries.");

    Ok(())
}

// --- Helpers ---

fn random_vec(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 31 + i * 17) as f32 * 0.001).sin())
        .collect()
}

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
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.into_iter().take(k).map(|(id, _)| id).collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (norm_a * norm_b + f32::EPSILON)
}
