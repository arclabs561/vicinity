#![allow(clippy::expect_used, clippy::unwrap_used)]
//! ann-benchmarks compatible benchmark runner.
//!
//! Loads datasets converted by `scripts/download_ann_benchmarks.py` and
//! benchmarks HNSW (and optionally NSW) at various ef_search values.
//!
//! ```bash
//! # Download dataset first
//! uv run scripts/download_ann_benchmarks.py glove-25-angular
//!
//! # Run benchmark
//! cargo run --example ann_benchmark --release --features hnsw,nsw -- data/ann-benchmarks/glove-25-angular
//! ```

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("data/ann-benchmarks/glove-25-angular");

    if !Path::new(data_dir).join("train.bin").exists() {
        eprintln!("Dataset not found at: {}/train.bin", data_dir);
        eprintln!("Run: uv run scripts/download_ann_benchmarks.py <dataset>");
        std::process::exit(1);
    }

    println!("ANN Benchmark");
    println!("=============");
    println!("Data: {}\n", data_dir);

    let (train, dim) = load_vectors(&format!("{}/train.bin", data_dir))?;
    let (test, _) = load_vectors(&format!("{}/test.bin", data_dir))?;
    let (neighbors, k_gt) = load_neighbors(&format!("{}/neighbors.bin", data_dir))?;

    println!("Train: {} vectors x {} dims", train.len(), dim);
    println!("Test:  {} queries", test.len());
    println!("Ground truth: {} neighbors per query\n", k_gt);

    // ─── HNSW ────────────────────────────────────────────────────────────────
    {
        use vicinity::hnsw::{HNSWIndex, HNSWParams};

        for &m in &[16, 32] {
            let params = HNSWParams {
                m,
                m_max: m,
                ef_construction: 200,
                ..Default::default()
            };

            println!(
                "--- HNSW (M={}, ef_construction={}) ---",
                m, params.ef_construction
            );
            let build_start = Instant::now();
            let mut index = HNSWIndex::with_params(dim, params)?;
            for (i, vec) in train.iter().enumerate() {
                index.add_slice(i as u32, vec)?;
            }
            index.build()?;
            let build_time = build_start.elapsed();
            println!(
                "Build: {:.2}s ({:.0} vectors/sec)\n",
                build_time.as_secs_f64(),
                train.len() as f64 / build_time.as_secs_f64()
            );

            let k = 10;
            println!(
                "{:>8} {:>10} {:>12} {:>10}",
                "ef", "Recall@10", "Latency", "QPS"
            );
            println!("{}", "-".repeat(45));

            for ef in [10, 20, 50, 100, 200, 400] {
                let (recall, qps, latency_us) = evaluate(&index, &test, &neighbors, k, ef);
                println!(
                    "{:>8} {:>9.1}% {:>10.0}us {:>9.0}",
                    ef,
                    recall * 100.0,
                    latency_us,
                    qps
                );
            }
            println!();
        }
    }

    // ─── NSW ─────────────────────────────────────────────────────────────────
    #[cfg(feature = "nsw")]
    {
        use vicinity::nsw::NSWIndex;

        println!("--- NSW (M=16) ---");
        let build_start = Instant::now();
        let mut index = NSWIndex::new(dim, 16, 16)?;
        for (i, vec) in train.iter().enumerate() {
            index.add_slice(i as u32, vec)?;
        }
        index.build()?;
        let build_time = build_start.elapsed();
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time.as_secs_f64(),
            train.len() as f64 / build_time.as_secs_f64()
        );

        let k = 10;
        println!(
            "{:>8} {:>10} {:>12} {:>10}",
            "ef", "Recall@10", "Latency", "QPS"
        );
        println!("{}", "-".repeat(45));

        for ef in [10, 20, 50, 100, 200, 400] {
            let (recall, qps, latency_us) = evaluate_nsw(&index, &test, &neighbors, k, ef);
            println!(
                "{:>8} {:>9.1}% {:>10.0}us {:>9.0}",
                ef,
                recall * 100.0,
                latency_us,
                qps
            );
        }
        println!();
    }

    Ok(())
}

fn evaluate(
    index: &vicinity::hnsw::HNSWIndex,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64) {
    let start = Instant::now();
    let mut total_recall = 0.0;

    for (i, query) in test.iter().enumerate() {
        let results = index.search(query, k, ef).unwrap();
        let gt_set: HashSet<u32> = neighbors[i].iter().take(k).map(|&n| n as u32).collect();
        let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
        total_recall += gt_set.intersection(&found).count() as f64 / k as f64;
    }

    let elapsed = start.elapsed();
    let recall = total_recall / test.len() as f64;
    let qps = test.len() as f64 / elapsed.as_secs_f64();
    let latency_us = elapsed.as_micros() as f64 / test.len() as f64;
    (recall, qps, latency_us)
}

#[cfg(feature = "nsw")]
fn evaluate_nsw(
    index: &vicinity::nsw::NSWIndex,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64) {
    let start = Instant::now();
    let mut total_recall = 0.0;

    for (i, query) in test.iter().enumerate() {
        let results = index.search(query, k, ef).unwrap();
        let gt_set: HashSet<u32> = neighbors[i].iter().take(k).map(|&n| n as u32).collect();
        let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
        total_recall += gt_set.intersection(&found).count() as f64 / k as f64;
    }

    let elapsed = start.elapsed();
    let recall = total_recall / test.len() as f64;
    let qps = test.len() as f64 / elapsed.as_secs_f64();
    let latency_us = elapsed.as_micros() as f64 / test.len() as f64;
    (recall, qps, latency_us)
}

// ─── File loading ────────────────────────────────────────────────────────────

fn load_vectors(path: &str) -> Result<(Vec<Vec<f32>>, usize), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    assert_eq!(&magic, b"VEC1", "Invalid vector file format");

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
    assert_eq!(&magic, b"NBR1", "Invalid neighbors file format");

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
