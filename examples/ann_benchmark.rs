#![allow(clippy::expect_used, clippy::unwrap_used)]
//! ann-benchmarks compatible benchmark runner.
//!
//! Loads datasets converted by `scripts/download_ann_benchmarks.py` and
//! benchmarks multiple algorithms at various search parameters.
//!
//! ```bash
//! # Download dataset first
//! uv run scripts/download_ann_benchmarks.py glove-25-angular
//!
//! # Run benchmark (HNSW default)
//! cargo run --example ann_benchmark --release --features hnsw -- data/ann-benchmarks/glove-25-angular
//!
//! # Multiple algorithms (results auto-saved to data/ann-benchmarks/results/<dataset>-all-algos.jsonl)
//! cargo run --example ann_benchmark --release --features hnsw,nsw,ivf_pq -- \
//!   data/ann-benchmarks/glove-25-angular --algo hnsw --algo nsw --algo ivfpq --algo brute
//!
//! # Resume after interruption (completed algorithms are skipped automatically)
//! cargo run --example ann_benchmark --release --all-features -- \
//!   data/ann-benchmarks/glove-25-angular --algo hnsw --algo nsw --json
//!
//! # Force re-run (delete existing results)
//! cargo run --example ann_benchmark --release --all-features -- \
//!   data/ann-benchmarks/glove-25-angular --fresh --json
//!
//! # Custom results file
//! cargo run --example ann_benchmark --release --features hnsw -- \
//!   data/ann-benchmarks/glove-25-angular --results /tmp/my-results.jsonl --json
//!
//! # Custom parameters
//! cargo run --example ann_benchmark --release --features hnsw -- \
//!   data/ann-benchmarks/glove-25-angular --m 32 --ef-construction 400 --ef-search 10,50,200
//! ```

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ─── CLI config ──────────────────────────────────────────────────────────────

struct Config {
    data_dir: String,
    algos: Vec<String>,
    m: usize,
    ef_construction: usize,
    ef_search_values: Vec<usize>,
    json: bool,
    results_path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: "data/ann-benchmarks/glove-25-angular".into(),
            algos: vec!["hnsw".into()],
            m: 16,
            ef_construction: 200,
            ef_search_values: vec![10, 20, 50, 100, 200, 400],
            json: false,
            results_path: PathBuf::new(), // derived from data_dir after parsing
        }
    }
}

/// Derive results file path from dataset directory name.
fn default_results_path(data_dir: &str) -> PathBuf {
    let dataset = Path::new(data_dir)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let results_dir = Path::new(data_dir)
        .parent()
        .unwrap_or(Path::new("."))
        .join("results");
    std::fs::create_dir_all(&results_dir).ok();
    results_dir.join(format!("{}-all-algos.jsonl", dataset))
}

/// Load set of algorithm names that already have results.
fn load_completed_algos(path: &Path) -> HashSet<String> {
    let mut completed = HashSet::new();
    let Ok(file) = File::open(path) else {
        return completed;
    };
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        // Extract "algorithm":"<name>" from JSON without pulling in serde
        if let Some(start) = line.find("\"algorithm\":\"") {
            let rest = &line[start + 13..];
            if let Some(end) = rest.find('"') {
                completed.insert(rest[..end].to_string());
            }
        }
    }
    completed
}

/// Append a JSON result line to the results file (and print to stdout).
fn emit_result(results_path: &Path, line: &str) {
    println!("{}", line);
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(results_path)
    {
        let _ = writeln!(f, "{}", line);
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config::default();
    let mut algos_set = false;
    let mut results_override = false;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--algo" => {
                i += 1;
                if !algos_set {
                    cfg.algos.clear();
                    algos_set = true;
                }
                if i < args.len() {
                    cfg.algos.push(args[i].to_lowercase());
                }
            }
            "--m" => {
                i += 1;
                if i < args.len() {
                    cfg.m = args[i].parse().unwrap_or(16);
                }
            }
            "--ef-construction" => {
                i += 1;
                if i < args.len() {
                    cfg.ef_construction = args[i].parse().unwrap_or(200);
                }
            }
            "--ef-search" => {
                i += 1;
                if i < args.len() {
                    cfg.ef_search_values = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
            "--json" => {
                cfg.json = true;
            }
            "--results" => {
                i += 1;
                if i < args.len() {
                    cfg.results_path = PathBuf::from(&args[i]);
                    results_override = true;
                }
            }
            "--fresh" => {
                // Delete existing results to force full re-run
                // (handled after results_path is finalized)
                cfg.results_path = PathBuf::from("__fresh__");
            }
            arg if !arg.starts_with("--") => {
                cfg.data_dir = arg.to_string();
            }
            _ => {
                eprintln!("Unknown flag: {}", args[i]);
            }
        }
        i += 1;
    }

    let fresh_sentinel: PathBuf = PathBuf::from("__fresh__");
    let is_fresh = cfg.results_path == fresh_sentinel;
    if !results_override && !is_fresh {
        cfg.results_path = default_results_path(&cfg.data_dir);
    } else if is_fresh {
        cfg.results_path = default_results_path(&cfg.data_dir);
        std::fs::remove_file(&cfg.results_path).ok();
    }

    cfg
}

// ─── Benchmark result ────────────────────────────────────────────────────────

struct BenchResult {
    recall_at_k: f64,
    qps: f64,
    latency_us: f64,
}

/// Format a single result as a JSON line.
fn json_line(
    algorithm: &str,
    params: &str,
    build_time_s: f64,
    rss_kb: Option<u64>,
    result: &BenchResult,
) -> String {
    let mut s = format!(
        "{{\"algorithm\":\"{}\",\"params\":{},\"recall_at_10\":{:.4},\"qps\":{:.1},\"build_time_s\":{:.2},\"latency_us\":{:.1}",
        algorithm, params, result.recall_at_k, result.qps, build_time_s, result.latency_us
    );
    if let Some(kb) = rss_kb {
        s.push_str(&format!(",\"rss_kb\":{}", kb));
    }
    s.push('}');
    s
}

// ─── Generic evaluate ────────────────────────────────────────────────────────

const WARMUP_QUERIES: usize = 50;

fn evaluate(
    search_fn: &dyn Fn(&[f32], usize) -> Vec<(u32, f32)>,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    k: usize,
) -> BenchResult {
    // Warmup
    let warmup_count = WARMUP_QUERIES.min(test.len());
    for query in test.iter().take(warmup_count) {
        let _ = search_fn(query, k);
    }

    // Timed run
    let start = Instant::now();
    let mut total_recall = 0.0;

    for (i, query) in test.iter().enumerate() {
        let results = search_fn(query, k);
        let gt_set: HashSet<u32> = neighbors[i].iter().take(k).map(|&n| n as u32).collect();
        let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
        total_recall += gt_set.intersection(&found).count() as f64 / k as f64;
    }

    let elapsed = start.elapsed();
    BenchResult {
        recall_at_k: total_recall / test.len() as f64,
        qps: test.len() as f64 / elapsed.as_secs_f64(),
        latency_us: elapsed.as_micros() as f64 / test.len() as f64,
    }
}

// ─── Peak RSS ────────────────────────────────────────────────────────────────

fn current_rss_kb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // mach_task_basic_info via libc-style syscall is complex; use ps instead.
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&output.stdout);
        s.trim().parse::<u64>().ok()
    }
    #[cfg(target_os = "linux")]
    {
        // /proc/self/status -> VmRSS
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb_str = rest.trim().trim_end_matches(" kB").trim();
                return kb_str.parse::<u64>().ok();
            }
        }
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

// ─── Brute force ─────────────────────────────────────────────────────────────

fn brute_force_search(train: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut dists: Vec<(u32, f32)> = train
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, vicinity::distance::cosine_distance(query, v)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.truncate(k);
    dists
}

// ─── Table printing ──────────────────────────────────────────────────────────

fn print_header() {
    println!(
        "{:>10} {:>10} {:>12} {:>10}",
        "param", "Recall@10", "Latency", "QPS"
    );
    println!("{}", "-".repeat(47));
}

fn print_row(param_label: &str, result: &BenchResult) {
    println!(
        "{:>10} {:>9.1}% {:>10.0}us {:>9.0}",
        param_label,
        result.recall_at_k * 100.0,
        result.latency_us,
        result.qps
    );
}

// ─── Algorithm runners ───────────────────────────────────────────────────────

#[cfg(feature = "hnsw")]
fn run_hnsw(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::{HNSWIndex, HNSWParams};

    let params = HNSWParams {
        m: cfg.m,
        m_max: cfg.m,
        ef_construction: cfg.ef_construction,
        ..Default::default()
    };

    if !cfg.json {
        println!(
            "--- HNSW (M={}, ef_construction={}) ---",
            cfg.m, cfg.ef_construction
        );
    }

    let build_start = Instant::now();
    let mut index = HNSWIndex::with_params(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(&|q, k| index.search(q, k, ef).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{}}}",
                cfg.m, cfg.ef_construction, ef
            );
            emit_result(
                &cfg.results_path,
                &json_line("hnsw", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "nsw")]
fn run_nsw(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::nsw::NSWIndex;

    if !cfg.json {
        println!("--- NSW (M={}) ---", cfg.m);
    }

    let build_start = Instant::now();
    let mut index = NSWIndex::new(dim, cfg.m, cfg.m).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(&|q, k| index.search(q, k, ef).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!("{{\"m\":{},\"ef_search\":{}}}", cfg.m, ef);
            emit_result(
                &cfg.results_path,
                &json_line("nsw", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "ivf_pq")]
fn run_ivfpq(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

    let num_clusters = 256;
    // num_codebooks must divide dim evenly. Pick the largest divisor <= 8.
    let num_codebooks = (1..=8.min(dim)).rev().find(|&c| dim % c == 0).unwrap_or(1);

    if !cfg.json {
        println!(
            "--- IVF-PQ (clusters={}, codebooks={}) ---",
            num_clusters, num_codebooks
        );
    }

    let params = IVFPQParams {
        num_clusters,
        num_codebooks,
        codebook_size: 256,
        nprobe: 1, // will be swept
        ..Default::default()
    };

    let build_start = Instant::now();
    let mut index = IVFPQIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    // Sweep nprobe values (analogous to ef_search for graph methods)
    let nprobe_values = [1, 2, 5, 10, 20, 50, 100];
    for &nprobe in &nprobe_values {
        if nprobe > num_clusters {
            continue;
        }
        index.set_nprobe(nprobe);
        let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!(
                "{{\"num_clusters\":{},\"num_codebooks\":{},\"nprobe\":{}}}",
                num_clusters, num_codebooks, nprobe
            );
            emit_result(
                &cfg.results_path,
                &json_line("ivfpq", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("np={}", nprobe), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "emg")]
fn run_emg(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::emg::{EmgIndex, EmgParams};

    let params = EmgParams {
        max_degree: 32,
        candidate_size: 64,
        scale_t: 32,
        iterations: 2,
        alpha: 1.5,
        ef_search: 100,
        ..Default::default()
    };

    if !cfg.json {
        println!("--- EMG (max_degree=32) ---");
    }

    let build_start = Instant::now();
    let mut index = EmgIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line("emg", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "nsg")]
fn run_nsg(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::nsg::{NsgIndex, NsgParams};

    let n = train.len().min(50_000);
    if train.len() > 50_000 {
        eprintln!(
            "NSG: capping at 50,000 vectors (got {}); O(n^2) construction",
            train.len()
        );
    }
    let train = &train[..n];

    if !cfg.json {
        println!("--- NSG (max_degree=32, n={}) ---", n);
    }

    let params = NsgParams::default();

    let build_start = Instant::now();
    let mut index = NsgIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line("nsg", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "pipnn")]
fn run_pipnn(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::pipnn::{PipnnIndex, PipnnParams};

    let params = PipnnParams {
        max_leaf_size: 2048,
        max_degree: 32,
        num_hash_bits: 12,
        final_prune: true,
        alpha: 1.2,
        ef_search: 100,
        ..Default::default()
    };

    if !cfg.json {
        println!("--- PiPNN (max_degree=32, max_leaf_size=2048) ---");
    }

    let build_start = Instant::now();
    let mut index = PipnnIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"max_degree\":32,\"max_leaf_size\":2048,\"ef_search\":{}}}",
                ef
            );
            emit_result(
                &cfg.results_path,
                &json_line("pipnn", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "sng")]
fn run_sng(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::sng::SNGIndex;
    use vicinity::sng::SNGParams;

    let n = train.len().min(50_000);
    if train.len() > 50_000 {
        eprintln!(
            "SNG: capping at 50,000 vectors (got {}); O(n^2) construction",
            train.len()
        );
    }
    let train = &train[..n];

    if !cfg.json {
        println!("--- SNG (n={}) ---", n);
    }

    let params = SNGParams::default();

    let build_start = Instant::now();
    let mut index = SNGIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add(i as u32, vec.clone()).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
    if cfg.json {
        let params_json = "{}";
        emit_result(
            &cfg.results_path,
            &json_line("sng", params_json, build_time_s, rss, &result),
        );
    } else {
        print_row("--", &result);
        println!();
    }
}

#[cfg(feature = "vamana")]
fn run_vamana(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::vamana::VamanaIndex;
    use vicinity::vamana::VamanaParams;

    if !cfg.json {
        println!("--- Vamana ---");
    }

    let params = VamanaParams::default();

    let build_start = Instant::now();
    let mut index = VamanaIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add(i as u32, vec.clone()).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(&|q, k| index.search(q, k, ef).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!("{{\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line("vamana", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "ivf_rabitq")]
fn run_ivf_rabitq(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams};

    let num_clusters = 256;

    if !cfg.json {
        println!(
            "--- IVF-RaBitQ (clusters={}, total_bits=4) ---",
            num_clusters
        );
    }

    let params = IVFRaBitQParams {
        num_clusters,
        nprobe: 10,
        total_bits: 4,
        seed: 42,
        ..Default::default()
    };

    let build_start = Instant::now();
    let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    let nprobe_values = [1, 2, 5, 10, 20, 50, 100];
    for &nprobe in &nprobe_values {
        if nprobe > num_clusters {
            continue;
        }
        index.set_nprobe(nprobe);
        let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!(
                "{{\"num_clusters\":{},\"total_bits\":4,\"nprobe\":{}}}",
                num_clusters, nprobe
            );
            emit_result(
                &cfg.results_path,
                &json_line("ivf_rabitq", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("np={}", nprobe), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "finger")]
fn run_finger(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::finger::{FingerIndex, FingerParams};

    let params = FingerParams {
        max_degree: 32,
        ef_construction: 200,
        ef_search: 100,
        alpha: 1.2,
    };

    if !cfg.json {
        println!("--- FINGER (max_degree=32) ---");
    }

    let build_start = Instant::now();
    let mut index = FingerIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line("finger", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "fresh_graph")]
fn run_fresh_graph(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};

    let params = FreshGraphParams {
        max_degree: 32,
        ef_construction: 200,
        ef_search: 100,
        alpha: 1.2,
    };

    if !cfg.json {
        println!("--- FreshGraph (max_degree=32) ---");
    }

    let build_start = Instant::now();
    let mut index = FreshGraphIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line("fresh_graph", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "filtered_graph")]
fn run_filtered_graph(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use std::collections::HashMap;
    use vicinity::filtered_graph::{FilteredGraphIndex, FilteredGraphParams};

    let params = FilteredGraphParams {
        max_degree: 32,
        ef_construction: 200,
        ef_search: 100,
        alpha: 1.2,
    };

    if !cfg.json {
        println!("--- FilteredGraph (max_degree=32, unfiltered) ---");
    }

    let build_start = Instant::now();
    let mut index = FilteredGraphIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec, HashMap::new()).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line("filtered_graph", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "sparse_mips")]
fn run_sparse_mips(_cfg: &Config, _train: &[Vec<f32>], _test: &[Vec<f32>], _neighbors: &[Vec<i32>]) {
    eprintln!(
        "sparse_mips: skipped -- index requires sparse vectors (SparseVector); \
         the dense benchmark dataset (f32 slices) is incompatible. \
         Use a sparse dataset such as SPLADE or BM25 embeddings instead."
    );
}

#[cfg(feature = "rp_quant")]
fn run_rp_quant(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::rp_quant::{RpQuantIndex, RpQuantParams};

    let projected_dim = 64.min(dim);

    if !cfg.json {
        println!(
            "--- RpQuant (projected_dim={}, rerank=10) ---",
            projected_dim
        );
    }

    let params = RpQuantParams {
        projected_dim,
        rerank_factor: 10,
        seed: 42,
    };

    let build_start = Instant::now();
    let mut index = RpQuantIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
    if cfg.json {
        let params_json = format!(
            "{{\"projected_dim\":{},\"rerank_factor\":10}}",
            projected_dim
        );
        emit_result(
            &cfg.results_path,
            &json_line("rp_quant", &params_json, build_time_s, rss, &result),
        );
    } else {
        print_row("--", &result);
        println!();
    }
}

fn run_brute(cfg: &Config, train: &[Vec<f32>], test: &[Vec<f32>], neighbors: &[Vec<i32>]) {
    if !cfg.json {
        println!("--- Brute Force (linear scan) ---");
    }

    let build_time_s = 0.0; // no build step
    let rss = current_rss_kb();

    if !cfg.json {
        println!("Build: N/A (no index)\n");
        print_header();
    }

    let result = evaluate(&|q, k| brute_force_search(train, q, k), test, neighbors, 10);

    if cfg.json {
        let params_json = "{}";
        emit_result(
            &cfg.results_path,
            &json_line("brute", params_json, build_time_s, rss, &result),
        );
    } else {
        print_row("--", &result);
        println!();
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args();

    if !Path::new(&cfg.data_dir).join("train.bin").exists() {
        eprintln!("Dataset not found at: {}/train.bin", cfg.data_dir);
        eprintln!("Run: uv run scripts/download_ann_benchmarks.py <dataset>");
        std::process::exit(1);
    }

    if !cfg.json {
        println!("ANN Benchmark");
        println!("=============");
        println!("Data: {}\n", cfg.data_dir);
    }

    let (train, dim) = load_vectors(&format!("{}/train.bin", cfg.data_dir))?;
    let (test, _) = load_vectors(&format!("{}/test.bin", cfg.data_dir))?;
    let (neighbors, k_gt) = load_neighbors(&format!("{}/neighbors.bin", cfg.data_dir))?;

    if !cfg.json {
        println!("Train: {} vectors x {} dims", train.len(), dim);
        println!("Test:  {} queries", test.len());
        println!("Ground truth: {} neighbors per query\n", k_gt);
    }

    let completed = load_completed_algos(&cfg.results_path);
    if !completed.is_empty() {
        eprintln!(
            "Resuming: skipping {} completed algorithm(s): {}",
            completed.len(),
            completed.iter().cloned().collect::<Vec<_>>().join(", ")
        );
        eprintln!("Results file: {}\n", cfg.results_path.display());
    } else {
        eprintln!("Results file: {}\n", cfg.results_path.display());
    }

    for algo in &cfg.algos {
        if completed.contains(algo.as_str()) {
            continue;
        }
        match algo.as_str() {
            #[cfg(feature = "hnsw")]
            "hnsw" => run_hnsw(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "hnsw"))]
            "hnsw" => {
                eprintln!("HNSW not available (compile with --features hnsw)");
            }

            #[cfg(feature = "nsw")]
            "nsw" => run_nsw(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "nsw"))]
            "nsw" => {
                eprintln!("NSW not available (compile with --features nsw)");
            }

            #[cfg(feature = "ivf_pq")]
            "ivfpq" => run_ivfpq(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "ivf_pq"))]
            "ivfpq" => {
                eprintln!("IVF-PQ not available (compile with --features ivf_pq)");
            }

            #[cfg(feature = "emg")]
            "emg" => run_emg(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "emg"))]
            "emg" => {
                eprintln!("EMG not available (compile with --features emg)");
            }

            #[cfg(feature = "nsg")]
            "nsg" => run_nsg(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "nsg"))]
            "nsg" => {
                eprintln!("NSG not available (compile with --features nsg)");
            }

            #[cfg(feature = "pipnn")]
            "pipnn" => run_pipnn(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "pipnn"))]
            "pipnn" => {
                eprintln!("PiPNN not available (compile with --features pipnn)");
            }

            #[cfg(feature = "sng")]
            "sng" => run_sng(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "sng"))]
            "sng" => {
                eprintln!("SNG not available (compile with --features sng)");
            }

            #[cfg(feature = "vamana")]
            "vamana" => run_vamana(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "vamana"))]
            "vamana" => {
                eprintln!("Vamana not available (compile with --features vamana)");
            }

            #[cfg(feature = "ivf_rabitq")]
            "ivf_rabitq" => run_ivf_rabitq(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "ivf_rabitq"))]
            "ivf_rabitq" => {
                eprintln!("IVF-RaBitQ not available (compile with --features ivf_rabitq)");
            }

            #[cfg(feature = "finger")]
            "finger" => run_finger(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "finger"))]
            "finger" => {
                eprintln!("FINGER not available (compile with --features finger)");
            }

            #[cfg(feature = "fresh_graph")]
            "fresh_graph" => run_fresh_graph(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "fresh_graph"))]
            "fresh_graph" => {
                eprintln!("FreshGraph not available (compile with --features fresh_graph)");
            }

            #[cfg(feature = "filtered_graph")]
            "filtered_graph" => run_filtered_graph(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "filtered_graph"))]
            "filtered_graph" => {
                eprintln!("FilteredGraph not available (compile with --features filtered_graph)");
            }

            #[cfg(feature = "rp_quant")]
            "rp_quant" => run_rp_quant(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "rp_quant"))]
            "rp_quant" => {
                eprintln!("RpQuant not available (compile with --features rp_quant)");
            }

            #[cfg(feature = "sparse_mips")]
            "sparse_mips" => run_sparse_mips(&cfg, &train, &test, &neighbors),

            #[cfg(not(feature = "sparse_mips"))]
            "sparse_mips" => {
                eprintln!("sparse_mips not available (compile with --features sparse_mips)");
            }

            "brute" => run_brute(&cfg, &train, &test, &neighbors),

            other => {
                eprintln!(
                    "Unknown algorithm: {}. Options: hnsw, nsw, ivfpq, emg, nsg, pipnn, sng, vamana, ivf_rabitq, finger, fresh_graph, filtered_graph, rp_quant, sparse_mips, brute",
                    other
                );
            }
        }
    }

    Ok(())
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
