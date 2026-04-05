#![allow(clippy::unwrap_used, clippy::expect_used)]
//! GloVe-25 Real ANN Benchmark
//!
//! Benchmarks HNSW against the GloVe-25-angular dataset from ann-benchmarks.com.
//! This is a standard benchmark used across all ANN implementations.
//!
//! Dataset: http://ann-benchmarks.com/glove-25-angular.hdf5
//! Vectors: 1,183,514 train + 10,000 test queries
//! Dimensions: 25
//! Distance: Angular (cosine) - requires L2-normalized vectors
//!
//! # Expected Performance (ann-benchmarks.com reference)
//!
//! HNSW with M=16, ef_construction=500 should achieve:
//! - Recall@10 ~98-99% at ef_search=200-400
//! - QPS: 10,000-20,000 queries/second (depending on hardware)
//!
//! If you see significantly lower recall, check:
//! 1. Vectors are L2-normalized (crucial for angular/cosine distance)
//! 2. ef_search is high enough (higher = better recall, lower QPS)
//! 3. ef_construction was sufficient during index build
//!
//! Run:
//! ```bash
//! # Quick demo (10K vectors)
//! cargo run --example glove_benchmark --release
//!
//! # Full benchmark (1.18M vectors, takes ~30 min to build)
//! cargo run --example glove_benchmark --release -- --full
//! ```

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

use vicinity::hnsw::HNSWIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let full_mode = std::env::args().any(|a| a == "--full");

    println!("GloVe-25 ANN Benchmark (Real Data)");
    println!("==================================\n");

    // Try multiple possible locations for data
    let data_dir = find_data_dir().unwrap_or_else(|| {
        eprintln!("GloVe-25 data not found. To generate:");
        eprintln!();
        eprintln!(
            "  1. Download: curl -o glove.hdf5 http://ann-benchmarks.com/glove-25-angular.hdf5"
        );
        eprintln!("  2. Convert: uv run scripts/download_ann_benchmarks.py glove-25-angular");
        eprintln!();
        eprintln!("Or run with synthetic data: cargo run --example hnsw_benchmark --release");
        std::process::exit(1);
    });

    println!("Data directory: {}", data_dir);
    println!(
        "Mode: {}\n",
        if full_mode {
            "FULL (1.18M)"
        } else {
            "QUICK (10K)"
        }
    );

    // Load train vectors
    let train_file = if full_mode {
        "glove25_train.bin"
    } else {
        "glove25_train_10k.bin"
    };
    let train_path = format!("{}/{}", data_dir, train_file);
    let (train, dim) = load_vectors(&train_path)?;
    println!("Train: {} vectors x {} dims", train.len(), dim);

    // Load test queries
    let test_path = format!("{}/glove25_test.bin", data_dir);
    let (test, _) = load_vectors(&test_path)?;
    let n_queries = if full_mode { test.len() } else { 1000 };
    println!("Test:  {} queries", n_queries);

    // Load ground truth neighbors
    let neighbors_path = format!("{}/glove25_neighbors.bin", data_dir);
    let (neighbors, k_gt) = load_neighbors(&neighbors_path)?;
    println!("Ground truth: {} neighbors per query\n", k_gt);

    // Build HNSW index with standard ann-benchmarks parameters
    // M=16 is standard, ef_construction=500 matches hnswlib benchmark config
    let m = 16;
    let ef_build = 500; // Standard for ann-benchmarks (higher = better graph quality)

    println!(
        "Building HNSW index (M={}, ef_construction={})...",
        m, ef_build
    );
    println!("  (using ann-benchmarks.com standard parameters)");
    let build_start = Instant::now();

    let mut index = HNSWIndex::new(dim, m, ef_build)?;
    for (i, vec) in train.iter().enumerate() {
        // L2-normalize for cosine similarity
        let normalized = normalize(vec);
        index.add(i as u32, normalized)?;
        if (i + 1) % 100_000 == 0 {
            print!("\r  Added {} vectors...", i + 1);
            use std::io::Write;
            std::io::stdout().flush()?;
        }
    }
    index.build()?;

    let build_time = build_start.elapsed();
    println!(
        "\nBuild time: {:.2}s ({:.0} vectors/sec)\n",
        build_time.as_secs_f64(),
        train.len() as f64 / build_time.as_secs_f64()
    );

    // For quick mode, compute brute-force ground truth against the subset
    let local_gt: Vec<Vec<usize>> = if !full_mode {
        println!("Computing local ground truth (brute force)...");
        let gt_start = Instant::now();
        let gt: Vec<Vec<usize>> = test
            .iter()
            .take(n_queries)
            .map(|query| {
                let mut dists: Vec<(usize, f32)> = train
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, angular_distance(query, v)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                dists.iter().take(100).map(|(i, _)| *i).collect()
            })
            .collect();
        println!(
            "Ground truth computed in {:.2}s\n",
            gt_start.elapsed().as_secs_f64()
        );
        gt
    } else {
        vec![]
    };

    // Benchmark at different ef values
    let k = 10;
    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "ef", "QPS", "Latency", "Recall@10"
    );
    println!("{}", "-".repeat(50));

    let ef_values = if full_mode {
        vec![50, 100, 200, 400]
    } else {
        vec![20, 50, 100, 200]
    };

    for ef in ef_values {
        let search_start = Instant::now();
        let mut total_recall = 0.0;

        for (i, query) in test.iter().take(n_queries).enumerate() {
            let normalized_query = normalize(query);
            let results = index.search(&normalized_query, k, ef)?;

            let gt_set: HashSet<usize> = if full_mode {
                neighbors[i].iter().take(k).map(|&n| n as usize).collect()
            } else {
                local_gt[i].iter().take(k).copied().collect()
            };

            let found: HashSet<usize> = results.iter().map(|r| r.0 as usize).collect();
            let intersection = gt_set.intersection(&found).count();
            total_recall += intersection as f64 / k as f64;
        }

        let search_time = search_start.elapsed();
        let qps = n_queries as f64 / search_time.as_secs_f64();
        let latency_us = search_time.as_micros() as f64 / n_queries as f64;
        let recall = total_recall / n_queries as f64 * 100.0;

        println!(
            "{:>8} {:>12.0} {:>10.0}us {:>9.1}%",
            ef, qps, latency_us, recall
        );
    }

    // Analysis and expected baseline comparison
    println!("\n--- Analysis ---");
    println!("Expected baseline (ann-benchmarks.com, hnswlib M=16, ef_construction=500):");
    println!("  ef=200:  ~98% recall, ~15K QPS");
    println!("  ef=400:  ~99% recall, ~10K QPS");
    println!("  ef=800:  ~99.5% recall, ~6K QPS");

    if !full_mode {
        println!("\nNote: Quick mode uses 10K subset with local ground truth.");
        println!("Results may differ from full benchmark due to smaller graph size.");
        println!("Run with --full for benchmark against all 1.18M vectors.");
    }

    println!("\nKey factors affecting recall:");
    println!("  1. Vectors MUST be L2-normalized (cosine distance = 1 - dot product)");
    println!("  2. ef_search controls recall/speed tradeoff (higher = better recall)");
    println!("  3. ef_construction affects graph quality (500 is standard)");

    Ok(())
}

fn find_data_dir() -> Option<String> {
    let paths = [
        "data",
        "vicinity/data",
        "../vicinity/data",
        &format!("{}/data", env!("CARGO_MANIFEST_DIR")),
    ];

    paths
        .iter()
        .map(|s| s.to_string())
        .find(|p| Path::new(p).join("glove25_train.bin").exists())
}

/// Load vectors from binary format: VEC1 + n(u32) + d(u32) + data(f32)
fn load_vectors(path: &str) -> Result<(Vec<Vec<f32>>, usize), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"VEC1" {
        return Err("Invalid vector file format".into());
    }

    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let d = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let mut data = vec![0u8; n * d * 4];
    reader.read_exact(&mut data)?;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..d)
                .map(|j| {
                    let offset = (i * d + j) * 4;
                    f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ])
                })
                .collect()
        })
        .collect();

    Ok((vectors, d))
}

/// Load neighbors from binary format: NBR1 + n(u32) + k(u32) + data(i32)
fn load_neighbors(path: &str) -> Result<(Vec<Vec<i32>>, usize), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"NBR1" {
        return Err("Invalid neighbors file format".into());
    }

    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let k = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let mut data = vec![0u8; n * k * 4];
    reader.read_exact(&mut data)?;

    let neighbors: Vec<Vec<i32>> = (0..n)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let offset = (i * k + j) * 4;
                    i32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ])
                })
                .collect()
        })
        .collect();

    Ok((neighbors, k))
}

/// Angular distance (1 - cosine_similarity)
fn angular_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (norm_a * norm_b + f32::EPSILON)
}

/// L2 normalize a vector
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}
