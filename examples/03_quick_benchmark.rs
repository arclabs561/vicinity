#![allow(clippy::expect_used)]
//! Quick Benchmark with Bundled or Synthetic Data
//!
//! Runs a benchmark using pre-generated sample data if available,
//! otherwise generates synthetic data inline (no downloads required).
//!
//! ```bash
//! cargo run --example 03_quick_benchmark --release
//! JIN_DATASET=quick cargo run --example 03_quick_benchmark --release
//! ```
//!
//! Datasets:
//! - `bench` (default): 10K x 384 dims (from pre-generated files, or synthetic fallback)
//! - `quick`: 2K x 128 dims (always synthetic, for CI)

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

use std::collections::HashMap;
use vicinity::hnsw::HNSWIndex;
use vicinity::hnsw::HNSWParams;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = std::env::var("JIN_DATASET").unwrap_or_else(|_| "bench".to_string());
    let variant = std::env::var("JIN_TEST_VARIANT").unwrap_or_default();

    println!("Quick Benchmark");
    println!("===============\n");

    // Try to load pre-generated data; fall back to synthetic
    let (train, test, neighbors, dim, k_gt, source) = match find_data_dir(&dataset) {
        Ok(data_dir) => {
            let test_file = if variant.is_empty() {
                format!("{}/{}_test.bin", data_dir, dataset)
            } else {
                format!("{}/{}_test_{}.bin", data_dir, dataset, variant)
            };
            let nbr_file = if variant.is_empty() {
                format!("{}/{}_neighbors.bin", data_dir, dataset)
            } else {
                format!("{}/{}_neighbors_{}.bin", data_dir, dataset, variant)
            };

            let (train, dim) = load_vectors(&format!("{}/{}_train.bin", data_dir, dataset))?;
            let (test, _) = load_vectors(&test_file)?;
            let (neighbors, k_gt) = load_neighbors(&nbr_file)?;
            (
                train,
                test,
                neighbors,
                dim,
                k_gt,
                format!("file ({})", data_dir),
            )
        }
        Err(_) => {
            // Generate synthetic data
            let (n, dim) = if dataset == "quick" {
                (2_000, 128)
            } else {
                (10_000, 384)
            };
            let n_queries = 100;
            let k_gt = 100;

            let (train, test, neighbors) = generate_synthetic_benchmark(n, dim, n_queries, k_gt);
            (train, test, neighbors, dim, k_gt, "synthetic".to_string())
        }
    };

    println!("Dataset: {} ({})", dataset, source);
    println!("Train:   {} vectors x {} dims", train.len(), dim);
    println!("Test:    {} queries", test.len());
    println!("Ground truth: {} neighbors per query\n", k_gt);

    // Build HNSW index
    let m = 16;
    let ef_construction = 200;
    let m_max = m * 2;

    println!(
        "Test variant: {}\n",
        if variant.is_empty() { "base" } else { &variant }
    );

    let is_filter = dataset == "hard" && variant == "filter";

    println!(
        "Building HNSW (M={}, m_max={}, ef_construction={})...",
        m, m_max, ef_construction
    );
    let build_start = Instant::now();

    let mut index = if is_filter {
        HNSWIndex::with_filtering(dim, m, m_max, "topic")?
    } else {
        let params = HNSWParams {
            m,
            m_max,
            ef_construction,
            ..Default::default()
        };
        HNSWIndex::with_params(dim, params)?
    };

    let data_dir = find_data_dir(&dataset).ok();
    let train_topics = if is_filter {
        data_dir
            .as_ref()
            .and_then(|d| load_labels(&format!("{}/hard_train_topics.bin", d)).ok())
    } else {
        None
    };

    for (i, vec) in train.iter().enumerate() {
        let doc_id = i as u32;
        if let Some(ref topics) = train_topics {
            let mut md: HashMap<String, u32> = HashMap::new();
            md.insert("topic".to_string(), topics[i]);
            index.add_metadata(doc_id, md)?;
        }
        index.add(doc_id, vec.clone())?;
    }
    index.build()?;

    let build_time = build_start.elapsed();
    println!(
        "Build: {:.2}s ({:.0} vectors/sec)\n",
        build_time.as_secs_f64(),
        train.len() as f64 / build_time.as_secs_f64()
    );

    // Benchmark at different ef values
    let k = 10;
    println!(
        "{:>8} {:>10} {:>12} {:>10}",
        "ef", "Recall@10", "Latency", "QPS"
    );
    println!("{}", "-".repeat(45));

    for ef in [20, 50, 100, 200] {
        let search_start = Instant::now();
        let mut total_recall = 0.0;

        let filter_topics = if is_filter {
            data_dir
                .as_ref()
                .and_then(|d| load_labels(&format!("{}/hard_test_filter_topics.bin", d)).ok())
        } else {
            None
        };

        for (i, query) in test.iter().enumerate() {
            let results = if is_filter {
                let topic = filter_topics.as_ref().expect("filter topics loaded")[i];
                let filter = vicinity::filtering::MetadataFilter::equals("topic", topic);
                index.search_with_filter(query, k, ef, &filter)?
            } else {
                index.search(query, k, ef)?
            };

            let gt_set: HashSet<u32> = neighbors[i].iter().take(k).map(|&n| n as u32).collect();
            let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
            let intersection = gt_set.intersection(&found).count();
            total_recall += intersection as f64 / k as f64;
        }

        let search_time = search_start.elapsed();
        let qps = test.len() as f64 / search_time.as_secs_f64();
        let latency_us = search_time.as_micros() as f64 / test.len() as f64;
        let recall = total_recall / test.len() as f64 * 100.0;

        println!(
            "{:>8} {:>9.1}% {:>10.0}us {:>9.0}",
            ef, recall, latency_us, qps
        );
    }

    println!();

    Ok(())
}

// ─── Synthetic data generation ───────────────────────────────────────────────

/// Generate a self-contained benchmark dataset with ground truth.
///
/// Creates clustered, L2-normalized vectors and computes exact brute-force
/// k-NN ground truth using cosine distance.
fn generate_synthetic_benchmark(
    n: usize,
    dim: usize,
    n_queries: usize,
    k: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<i32>>) {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);

    let num_clusters = 10;
    let vectors_per_cluster = n / num_clusters;

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            normalize(&v)
        })
        .collect();

    // Generate clustered vectors with Gaussian noise
    let mut train = Vec::with_capacity(n);
    for center in &centers {
        for _ in 0..vectors_per_cluster {
            let noisy: Vec<f32> = center
                .iter()
                .map(|&c| c + (rng.random::<f32>() - 0.5) * 0.3)
                .collect();
            train.push(normalize(&noisy));
        }
    }
    // Fill remainder if n not divisible by num_clusters
    while train.len() < n {
        let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        train.push(normalize(&v));
    }

    // Generate queries as perturbations of existing vectors
    let test: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| {
            let base = &train[(i * 7) % n];
            let perturbed: Vec<f32> = base
                .iter()
                .map(|&v| v + (rng.random::<f32>() - 0.5) * 0.15)
                .collect();
            normalize(&perturbed)
        })
        .collect();

    // Compute brute-force ground truth (cosine distance = 1 - dot for normalized vecs)
    let neighbors: Vec<Vec<i32>> = test
        .iter()
        .map(|query| {
            let mut dists: Vec<(usize, f32)> = train
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    (i, 1.0 - dot)
                })
                .collect();
            dists.sort_by(|a, b| a.1.total_cmp(&b.1));
            dists.iter().take(k).map(|(i, _)| *i as i32).collect()
        })
        .collect();

    (train, test, neighbors)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        v.iter().map(|x| x / n).collect()
    } else {
        vec![0.0; v.len()]
    }
}

// ─── File-based data loading ─────────────────────────────────────────────────

fn find_data_dir(dataset: &str) -> Result<String, Box<dyn std::error::Error>> {
    let paths = [
        "data/sample",
        "vicinity/data/sample",
        "../vicinity/data/sample",
        &format!("{}/data/sample", env!("CARGO_MANIFEST_DIR")),
    ];

    let train_file = format!("{}_train.bin", dataset);
    for path in &paths {
        if Path::new(path).join(&train_file).exists() {
            return Ok(path.to_string());
        }
    }

    Err(format!("Sample data not found (looked for {})", train_file).into())
}

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

fn load_labels(path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"LBL1" {
        return Err("Invalid label file format".into());
    }

    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;
    let n = u32::from_le_bytes(header) as usize;

    let mut data = vec![0u8; n * 4];
    reader.read_exact(&mut data)?;

    let labels: Vec<u32> = (0..n)
        .map(|i| {
            let offset = i * 4;
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    Ok(labels)
}
