#![allow(clippy::unwrap_used, clippy::expect_used)]
//! HNSW vs Brute Force Benchmark
//!
//! Measures HNSW search against brute force on deterministic normalized
//! vectors and reports QPS, speedup, and recall.
//!
//! ```bash
//! cargo run --example hnsw_benchmark --release
//! ```

use std::time::Instant;
use vicinity::hnsw::HNSWIndex;

fn main() -> vicinity::Result<()> {
    println!("HNSW vs Brute Force Benchmark");
    println!("==============================\n");
    println!("Note: HNSW uses cosine distance internally, vectors are L2-normalized.\n");

    let dim = 128;
    let sizes = [1_000, 5_000, 10_000, 25_000];

    for &n in &sizes {
        benchmark_size(n, dim)?;
    }

    Ok(())
}

fn benchmark_size(n: usize, dim: usize) -> vicinity::Result<()> {
    println!("Dataset: {} vectors, {} dimensions", n, dim);
    println!("{}", "-".repeat(50));

    // Generate random normalized vectors (for cosine similarity)
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| normalize(&generate_vector(i, dim)))
        .collect();

    // Generate query vectors (normalized perturbations of random database vectors)
    let n_queries = 100;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| {
            let base_idx = (i * 7) % n;
            let perturbed: Vec<f32> = vectors[base_idx]
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let noise = ((i * dim + j) as f32 * 0.0001).sin() * 0.1;
                    v + noise
                })
                .collect();
            normalize(&perturbed)
        })
        .collect();

    let k = 10;

    // Build HNSW index with reasonable parameters
    let build_start = Instant::now();
    let mut index = HNSWIndex::new(dim, 16, 32)?;
    for (i, vec) in vectors.iter().enumerate() {
        index.add(i as u32, vec.clone())?;
    }
    index.build()?;
    let build_time = build_start.elapsed();
    println!("HNSW build time: {:?}", build_time);

    // HNSW search
    let ef = 100; // higher ef for better recall
    let hnsw_start = Instant::now();
    let mut hnsw_results = Vec::with_capacity(n_queries);
    for query in &queries {
        let results = index.search(query, k, ef)?;
        hnsw_results.push(results);
    }
    let hnsw_time = hnsw_start.elapsed();

    // Brute force search using COSINE DISTANCE (same as HNSW)
    let brute_start = Instant::now();
    let mut brute_results = Vec::with_capacity(n_queries);
    for query in &queries {
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_distance(query, v)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        brute_results.push(distances.into_iter().take(k).collect::<Vec<_>>());
    }
    let brute_time = brute_start.elapsed();

    // Calculate recall
    let mut total_recall = 0.0;
    for (hnsw, brute) in hnsw_results.iter().zip(brute_results.iter()) {
        let hnsw_ids: std::collections::HashSet<u32> = hnsw.iter().map(|r| r.0).collect();
        let brute_ids: std::collections::HashSet<u32> = brute.iter().map(|r| r.0 as u32).collect();
        let intersection = hnsw_ids.intersection(&brute_ids).count();
        total_recall += intersection as f64 / k as f64;
    }
    let avg_recall = total_recall / n_queries as f64;

    // Report results
    let hnsw_qps = n_queries as f64 / hnsw_time.as_secs_f64();
    let brute_qps = n_queries as f64 / brute_time.as_secs_f64();
    let speedup = brute_time.as_secs_f64() / hnsw_time.as_secs_f64();

    println!("HNSW:        {:>8.1} QPS ({:?} total)", hnsw_qps, hnsw_time);
    println!(
        "Brute force: {:>8.1} QPS ({:?} total)",
        brute_qps, brute_time
    );
    println!("Speedup:     {:>8.1}x", speedup);
    println!("Recall@{}:   {:>8.1}%", k, avg_recall * 100.0);
    println!();

    Ok(())
}

/// Generate a deterministic pseudo-random vector
fn generate_vector(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| {
            let x = (seed * dim + j) as f32;
            (x * 0.618_034).fract() * 2.0 - 1.0 // Golden ratio hash
        })
        .collect()
}

/// L2-normalize a vector
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Cosine distance: 1 - cosine_similarity
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (norm_a * norm_b + f32::EPSILON)
}
