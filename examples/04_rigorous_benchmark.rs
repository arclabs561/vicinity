#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Rigorous ANN Benchmark with Statistical Rigor
//!
//! Based on research best practices from:
//! - ann-benchmarks: Recall/QPS evaluation conventions (arXiv:1807.05614)
//! - BigANN (NeurIPS challenge series): workload variants at scale (e.g. "Results of the Big ANN: NeurIPS'23 competition", arXiv:2409.17424)
//! - NeurIPS/ICML paper checklists: Confidence intervals, multiple runs
//!
//! Features:
//! - Multi-run statistical rigor (5 runs, mean/std/CI)
//! - Brute-force baseline comparison
//! - Memory tracking (build and search)
//! - LID-stratified recall reporting
//! - Pareto frontier data export (JSON)
//!
//! ```bash
//! # Generate data first:
//! uvx --with numpy python scripts/generate_multiscale_data.py --scale B
//!
//! # Run benchmark:
//! cargo run --example 04_rigorous_benchmark --release -- --scale B
//! cargo run --example 04_rigorous_benchmark --release -- --scale T
//! cargo run --example 04_rigorous_benchmark --release -- --scale P
//! ```

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use vicinity::hnsw::{HNSWIndex, HNSWParams};

// =============================================================================
// Configuration
// =============================================================================

const N_RUNS: usize = 5;
const WARMUP_QUERIES: usize = 50;
const K: usize = 10;
#[allow(dead_code)]
const K_GT: usize = 100;

// =============================================================================
// Statistics utilities
// =============================================================================

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn confidence_interval_95(values: &[f64]) -> (f64, f64) {
    let m = mean(values);
    let s = std_dev(values);
    let se = s / (values.len() as f64).sqrt();
    let margin = 1.96 * se; // 95% CI
    (m - margin, m + margin)
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() - 1) as f64 * p / 100.0) as usize;
    values[idx.min(values.len() - 1)]
}

// =============================================================================
// Memory tracking (approximate)
// =============================================================================

fn get_process_memory_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<u64>() {
                    return Some(pages * 4); // 4KB pages
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // On macOS, use ps to get RSS
        use std::process::Command;
        let pid = std::process::id();
        if let Ok(output) = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
        {
            if let Ok(rss_str) = String::from_utf8(output.stdout) {
                if let Ok(rss_kb) = rss_str.trim().parse::<u64>() {
                    return Some(rss_kb);
                }
            }
        }
    }

    None
}

// =============================================================================
// Data loading
// =============================================================================

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

fn load_neighbors(path: &str) -> Result<Vec<Vec<i32>>, Box<dyn std::error::Error>> {
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

    Ok(neighbors)
}

#[allow(dead_code)]
fn load_f32_array(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"F32A" {
        return Err("Invalid f32 array file format".into());
    }

    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;
    let n = u32::from_le_bytes(header) as usize;

    let mut data = vec![0u8; n * 4];
    reader.read_exact(&mut data)?;

    let values: Vec<f32> = (0..n)
        .map(|i| {
            let offset = i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    Ok(values)
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

// =============================================================================
// Brute force baseline
// =============================================================================

fn brute_force_search(train: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scores: Vec<(u32, f32)> = train
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let sim: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            (i as u32, sim)
        })
        .collect();

    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    scores.truncate(k);
    scores
}

// =============================================================================
// Recall computation
// =============================================================================

/// Compute recall, returning None if ground truth is invalid.
///
/// Returns None when ground truth has no valid neighbors (all -1),
/// so these queries can be excluded from averaging.
fn compute_recall(results: &[(u32, f32)], ground_truth: &[i32], k: usize) -> Option<f64> {
    let gt_set: HashSet<u32> = ground_truth
        .iter()
        .take(k)
        .filter(|&&n| n >= 0)
        .map(|&n| n as u32)
        .collect();

    if gt_set.is_empty() {
        return None; // No valid ground truth - exclude from average
    }

    let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
    let intersection = gt_set.intersection(&found).count();
    Some(intersection as f64 / gt_set.len() as f64)
}

// =============================================================================
// Benchmark result structures
// =============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RunResult {
    recall: f64,
    latency_us: f64,
    qps: f64,
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    ef: usize,
    recall_mean: f64,
    recall_std: f64,
    recall_ci_low: f64,
    recall_ci_high: f64,
    latency_mean_us: f64,
    latency_p50_us: f64,
    latency_p99_us: f64,
    qps_mean: f64,
    qps_std: f64,

    // LID-stratified recall
    recall_easy: f64,
    recall_medium: f64,
    recall_hard: f64,
}

#[derive(Debug)]
struct FullBenchmark {
    scale: String,
    n_train: usize,
    n_test: usize,
    dim: usize,
    build_time_s: f64,
    memory_kb: Option<u64>,

    // Baselines
    brute_force_qps: f64,

    // HNSW results at different ef values
    results: Vec<BenchmarkResult>,
}

// =============================================================================
// Main benchmark logic
// =============================================================================

fn run_benchmark(
    scale: &str,
    data_dir: &Path,
) -> Result<FullBenchmark, Box<dyn std::error::Error>> {
    let scale_dir = data_dir.join(scale);

    // Load data
    println!("Loading data for scale {}...", scale);
    let (train, dim) = load_vectors(&scale_dir.join("train.bin").to_string_lossy())?;
    let (test, _) = load_vectors(&scale_dir.join("test.bin").to_string_lossy())?;
    let neighbors = load_neighbors(&scale_dir.join("neighbors.bin").to_string_lossy())?;

    // Load LID-based difficulty labels (0=easy, 1=medium, 2=hard)
    let difficulty_labels = load_labels(&scale_dir.join("test_difficulty.bin").to_string_lossy())
        .unwrap_or_else(|_| vec![1; test.len()]); // Default to medium if not available

    println!("  Train: {} vectors x {} dims", train.len(), dim);
    println!("  Test: {} queries", test.len());

    // Memory before build
    let mem_before = get_process_memory_kb();

    // Build HNSW index
    let m = 24; // Increased from 16 for better recall on difficult data
    let m_max = m * 2;
    let ef_construction = 400; // Increased for better graph quality

    println!(
        "\nBuilding HNSW (M={}, ef_construction={})...",
        m, ef_construction
    );
    let build_start = Instant::now();

    let params = HNSWParams {
        m,
        m_max,
        ef_construction,
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params)?;

    for (i, vec) in train.iter().enumerate() {
        index.add(i as u32, vec.clone())?;
    }
    index.build()?;

    let build_time = build_start.elapsed();
    let mem_after = get_process_memory_kb();
    let memory_used = match (mem_before, mem_after) {
        (Some(before), Some(after)) => Some(after.saturating_sub(before)),
        _ => None,
    };

    println!(
        "  Build time: {:.2}s ({:.0} vec/s)",
        build_time.as_secs_f64(),
        train.len() as f64 / build_time.as_secs_f64()
    );
    if let Some(mem) = memory_used {
        println!("  Memory delta: {} KB", mem);
    }

    // Brute force baseline (sample for large datasets)
    println!("\nRunning brute-force baseline...");

    // Estimate complexity: N * Q * D
    let total_ops = train.len() as u64 * test.len() as u64 * dim as u64;
    let bf_sample_size = if total_ops < 1_000_000_000 {
        // If < 1B operations, run full dataset (approx < 1-2s)
        test.len()
    } else {
        // Otherwise sample 100 queries
        100
    };

    let bf_start = Instant::now();
    for query in test.iter().take(bf_sample_size) {
        let _ = brute_force_search(&train, query, K);
    }
    let bf_time = bf_start.elapsed();
    let bf_qps = bf_sample_size as f64 / bf_time.as_secs_f64();
    println!(
        "  Brute-force: {:.1} QPS (sampled {} queries)",
        bf_qps, bf_sample_size
    );

    // Warmup
    println!("\nWarming up...");
    for query in test.iter().take(WARMUP_QUERIES) {
        let _ = index.search(query, K, 100);
    }

    // Benchmark at different ef values
    println!("\nBenchmarking (N_RUNS={}, K={})...", N_RUNS, K);
    println!(
        "{:>6} {:>12} {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8}",
        "ef", "Recall", "95% CI", "QPS", "p99 lat", "Easy", "Med", "Hard"
    );
    println!("{}", "-".repeat(90));

    let ef_values = [20, 50, 100, 200, 400];
    let mut all_results = Vec::new();

    for &ef in &ef_values {
        let mut run_recalls: Vec<f64> = Vec::with_capacity(N_RUNS);
        let mut run_latencies: Vec<f64> = Vec::with_capacity(N_RUNS * test.len());
        let mut run_qps: Vec<f64> = Vec::with_capacity(N_RUNS);

        // Per-difficulty recalls (aggregated across runs)
        let mut easy_recalls: Vec<f64> = Vec::new();
        let mut medium_recalls: Vec<f64> = Vec::new();
        let mut hard_recalls: Vec<f64> = Vec::new();

        for _run in 0..N_RUNS {
            let run_start = Instant::now();
            let mut valid_recalls: Vec<f64> = Vec::with_capacity(test.len());
            let mut query_latencies: Vec<Duration> = Vec::with_capacity(test.len());

            for (i, query) in test.iter().enumerate() {
                let q_start = Instant::now();
                let results = index.search(query, K, ef)?;
                let q_elapsed = q_start.elapsed();
                query_latencies.push(q_elapsed);

                // Only include queries with valid ground truth
                if let Some(recall) = compute_recall(&results, &neighbors[i], K) {
                    valid_recalls.push(recall);

                    // Track per-difficulty
                    match difficulty_labels.get(i).unwrap_or(&1) {
                        0 => easy_recalls.push(recall),
                        1 => medium_recalls.push(recall),
                        _ => hard_recalls.push(recall),
                    }
                }
            }

            let run_elapsed = run_start.elapsed();
            let run_recall = if valid_recalls.is_empty() {
                0.0
            } else {
                valid_recalls.iter().sum::<f64>() / valid_recalls.len() as f64
            };
            run_recalls.push(run_recall);
            run_qps.push(test.len() as f64 / run_elapsed.as_secs_f64());

            for lat in query_latencies {
                run_latencies.push(lat.as_micros() as f64);
            }
        }

        let recall_mean = mean(&run_recalls);
        let recall_std = std_dev(&run_recalls);
        let (recall_ci_low, recall_ci_high) = confidence_interval_95(&run_recalls);
        let qps_mean = mean(&run_qps);
        let qps_std = std_dev(&run_qps);
        let latency_mean = mean(&run_latencies);
        let latency_p50 = percentile(&mut run_latencies.clone(), 50.0);
        let latency_p99 = percentile(&mut run_latencies.clone(), 99.0);

        let recall_easy = if easy_recalls.is_empty() {
            0.0
        } else {
            mean(&easy_recalls)
        };
        let recall_medium = if medium_recalls.is_empty() {
            0.0
        } else {
            mean(&medium_recalls)
        };
        let recall_hard = if hard_recalls.is_empty() {
            0.0
        } else {
            mean(&hard_recalls)
        };

        let result = BenchmarkResult {
            ef,
            recall_mean,
            recall_std,
            recall_ci_low,
            recall_ci_high,
            latency_mean_us: latency_mean,
            latency_p50_us: latency_p50,
            latency_p99_us: latency_p99,
            qps_mean,
            qps_std,
            recall_easy,
            recall_medium,
            recall_hard,
        };

        println!(
            "{:>6} {:>11.1}% {:>10} {:>9.0} {:>9.0}us | {:>7.1}% {:>7.1}% {:>7.1}%",
            ef,
            recall_mean * 100.0,
            format!(
                "[{:.1},{:.1}]",
                recall_ci_low * 100.0,
                recall_ci_high * 100.0
            ),
            qps_mean,
            latency_p99,
            recall_easy * 100.0,
            recall_medium * 100.0,
            recall_hard * 100.0
        );

        all_results.push(result);
    }

    Ok(FullBenchmark {
        scale: scale.to_string(),
        n_train: train.len(),
        n_test: test.len(),
        dim,
        build_time_s: build_time.as_secs_f64(),
        memory_kb: memory_used,
        brute_force_qps: bf_qps,
        results: all_results,
    })
}

fn save_results(
    benchmark: &FullBenchmark,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Write as JSON for easy parsing by visualization scripts
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"scale\": \"{}\",", benchmark.scale)?;
    writeln!(writer, "  \"n_train\": {},", benchmark.n_train)?;
    writeln!(writer, "  \"n_test\": {},", benchmark.n_test)?;
    writeln!(writer, "  \"dim\": {},", benchmark.dim)?;
    writeln!(writer, "  \"build_time_s\": {:.3},", benchmark.build_time_s)?;
    writeln!(
        writer,
        "  \"memory_kb\": {},",
        benchmark.memory_kb.unwrap_or(0)
    )?;
    writeln!(
        writer,
        "  \"brute_force_qps\": {:.1},",
        benchmark.brute_force_qps
    )?;
    writeln!(writer, "  \"pareto_points\": [")?;

    for (i, r) in benchmark.results.iter().enumerate() {
        let comma = if i < benchmark.results.len() - 1 {
            ","
        } else {
            ""
        };
        writeln!(writer, "    {{")?;
        writeln!(writer, "      \"ef\": {},", r.ef)?;
        writeln!(writer, "      \"recall_mean\": {:.4},", r.recall_mean)?;
        writeln!(writer, "      \"recall_std\": {:.4},", r.recall_std)?;
        writeln!(writer, "      \"recall_ci_low\": {:.4},", r.recall_ci_low)?;
        writeln!(writer, "      \"recall_ci_high\": {:.4},", r.recall_ci_high)?;
        writeln!(writer, "      \"qps_mean\": {:.1},", r.qps_mean)?;
        writeln!(writer, "      \"qps_std\": {:.1},", r.qps_std)?;
        writeln!(
            writer,
            "      \"latency_mean_us\": {:.1},",
            r.latency_mean_us
        )?;
        writeln!(writer, "      \"latency_p50_us\": {:.1},", r.latency_p50_us)?;
        writeln!(writer, "      \"latency_p99_us\": {:.1},", r.latency_p99_us)?;
        writeln!(writer, "      \"recall_easy\": {:.4},", r.recall_easy)?;
        writeln!(writer, "      \"recall_medium\": {:.4},", r.recall_medium)?;
        writeln!(writer, "      \"recall_hard\": {:.4}", r.recall_hard)?;
        writeln!(writer, "    }}{}", comma)?;
    }

    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")?;

    Ok(())
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let scale = args
        .iter()
        .position(|a| a == "--scale")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("S");

    println!("Rigorous ANN Benchmark");
    println!("======================");
    println!("Scale: {} (S=10K, M=100K, L=1M, XL=10M)", scale);
    println!("Runs: {}, K: {}", N_RUNS, K);
    println!();

    // Find data directory
    let data_dir = find_data_dir()?;
    println!("Data directory: {}", data_dir.display());

    // Run benchmark
    let benchmark = run_benchmark(scale, &data_dir)?;

    // Save results
    let output_path = data_dir.join(format!("results_{}.json", scale));
    save_results(&benchmark, &output_path)?;
    println!("\nResults saved to: {}", output_path.display());

    // Summary
    println!("\n--- Summary ---");
    println!(
        "Scale {}: {} train, {} test, {}d",
        benchmark.scale, benchmark.n_train, benchmark.n_test, benchmark.dim
    );
    println!("Build: {:.2}s", benchmark.build_time_s);
    println!("Brute-force baseline: {:.1} QPS", benchmark.brute_force_qps);

    // Find 90% recall point
    if let Some(r90) = benchmark.results.iter().find(|r| r.recall_mean >= 0.90) {
        println!(
            "90% recall achieved at ef={}: {:.1} QPS",
            r90.ef, r90.qps_mean
        );
        println!(
            "  Speedup vs brute-force: {:.1}x",
            r90.qps_mean / benchmark.brute_force_qps
        );
    }

    Ok(())
}

fn find_data_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let paths = [
        "data/multiscale",
        "vicinity/data/multiscale",
        "../vicinity/data/multiscale",
        &format!("{}/data/multiscale", env!("CARGO_MANIFEST_DIR")),
    ];

    for path in &paths {
        let p = Path::new(path);
        if p.exists() && p.is_dir() {
            return Ok(p.to_path_buf());
        }
    }

    Err("Multiscale data not found. Run: uvx --with numpy python scripts/generate_multiscale_data.py".into())
}
