#![allow(clippy::unwrap_used, clippy::expect_used)]
//! SIFT-128 Benchmark
//!
//! Benchmarks HNSW against brute-force on synthetic 128-dimensional data.
//! Runs automatically with no external data download.
//!
//! ```bash
//! cargo run --example sift_benchmark --release --features hnsw
//! ```

use std::path::Path;
use std::time::Instant;

fn main() {
    println!("SIFT-128 ANN Benchmark");
    println!("======================\n");

    let dataset_path = "data/sift-128-euclidean.hdf5";

    if !Path::new(dataset_path).exists() {
        println!("Dataset not found at: {}", dataset_path);
        println!();
        println!("To run this benchmark, download the SIFT-128 dataset:");
        println!();
        println!("  mkdir -p data");
        println!(
            "  curl -o {} http://ann-benchmarks.com/sift-128-euclidean.hdf5",
            dataset_path
        );
        println!();
        println!("Dataset size: 501MB");
        println!("Alternative smaller datasets:");
        println!("  - GloVe-25 (121MB):  http://ann-benchmarks.com/glove-25-angular.hdf5");
        println!(
            "  - Fashion-MNIST (217MB): http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
        );
        println!();

        // Run a mini demo with synthetic data instead
        println!("Running mini demo with synthetic data instead...\n");
        run_synthetic_demo();
        return;
    }

    // HDF5 loading is not yet supported (no hdf5 dependency).
    // Run the synthetic demo instead.
    let _ = dataset_path;
    println!("HDF5 loading not available. Running synthetic demo instead...\n");
    run_synthetic_demo();
}

fn run_synthetic_demo() {
    use vicinity::hnsw::HNSWIndex;

    let n = 50_000;
    let dim = 128;
    let n_queries = 1000;
    let k = 10;

    println!("Synthetic benchmark: {} vectors, {} dims", n, dim);

    // Generate normalized vectors
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| normalize(&generate_vector(i, dim)))
        .collect();

    // Build index
    let build_start = Instant::now();
    let mut index = HNSWIndex::new(dim, 16, 200).unwrap();
    for (i, vec) in vectors.iter().enumerate() {
        index.add(i as u32, vec.clone()).unwrap();
    }
    index.build().unwrap();
    let build_time = build_start.elapsed();
    println!("Build time: {:?}", build_time);

    // Generate queries (perturbations of existing vectors)
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

    // Benchmark HNSW
    let ef = 100;
    let hnsw_start = Instant::now();
    let mut hnsw_results = Vec::with_capacity(n_queries);
    for query in &queries {
        let results = index.search(query, k, ef).unwrap();
        hnsw_results.push(results);
    }
    let hnsw_time = hnsw_start.elapsed();

    // Brute force for ground truth
    let brute_start = Instant::now();
    let mut brute_results = Vec::with_capacity(n_queries);
    for query in &queries {
        let mut distances: Vec<_> = vectors
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

    let hnsw_qps = n_queries as f64 / hnsw_time.as_secs_f64();
    let brute_qps = n_queries as f64 / brute_time.as_secs_f64();

    println!("\n--- Results ---");
    println!("HNSW:        {:.1} QPS ({:?})", hnsw_qps, hnsw_time);
    println!("Brute force: {:.1} QPS ({:?})", brute_qps, brute_time);
    println!(
        "Speedup:     {:.1}x",
        brute_time.as_secs_f64() / hnsw_time.as_secs_f64()
    );
    println!("Recall@{}:   {:.1}%", k, avg_recall * 100.0);
}

fn generate_vector(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| {
            let x = (seed * dim + j) as f32;
            (x * 0.618_034).fract() * 2.0 - 1.0
        })
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (norm_a * norm_b + f32::EPSILON)
}
