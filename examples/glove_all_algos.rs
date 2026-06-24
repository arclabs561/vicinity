#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_update)]
//! Multi-algorithm GloVe-25 benchmark. Outputs JSONL for plot_comparison.py.
//!
//! ```bash
//! # All algorithms
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,ivf_avq,diskann,kdtree,balltree,rptree" -- --algo all
//!
//! # Individual algorithms
//! cargo run --example glove_all_algos --release --features "hnsw,sq4" -- --algo sq4u
//! cargo run --example glove_all_algos --release --features "hnsw,sq8" -- --algo sq8u
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,ivf_avq,diskann" -- --algo hnsw
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,ivf_avq,diskann" -- --algo nsw
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,ivf_avq,diskann" -- --algo ivfpq
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,ivf_avq,diskann" -- --algo vamana
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,ivf_avq,diskann" -- --algo ivf_avq
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,ivf_avq,diskann" -- --algo diskann
//! cargo run --example glove_all_algos --release --features "kdtree" -- --algo kdtree
//! cargo run --example glove_all_algos --release --features "balltree" -- --algo balltree
//! cargo run --example glove_all_algos --release --features "rptree" -- --algo rptree
//!
//! # Output appended to docs/glove-25-angular.jsonl; regenerate plot with:
//! # uv run scripts/plot_comparison.py docs/glove-25-angular.jsonl
//! ```

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::time::Instant;

use vicinity::adsampling::{ADSamplingParams, ADSamplingState};
#[cfg(feature = "balltree")]
use vicinity::classic::trees::balltree::{BallTreeIndex, BallTreeParams};
#[cfg(feature = "kdtree")]
use vicinity::classic::trees::kdtree::{KDTreeIndex, KDTreeParams};
#[cfg(feature = "rptree")]
use vicinity::classic::trees::rp_forest::{RPTreeParams, RpForestIndex, RpForestParams};
#[cfg(feature = "diskann")]
use vicinity::diskann::{DiskANNIndex, DiskANNParams};
#[cfg(feature = "emg")]
use vicinity::emg::{EmgIndex, EmgParams};
#[cfg(feature = "sq4")]
use vicinity::hnsw::sq4u::HNSWSq4Index;
#[cfg(feature = "sq8")]
use vicinity::hnsw::sq8u::HNSWSq8Index;
#[cfg(feature = "ivf_rabitq")]
use vicinity::hnsw::symphony_qg::SymphonyQGIndex;
use vicinity::hnsw::{HNSWIndex, HNSWParams};
#[cfg(feature = "ivf_avq")]
use vicinity::ivf_avq::{IVFAVQIndex, IVFAVQParams};
#[cfg(feature = "ivf_pq")]
use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};
#[cfg(feature = "ivf_rabitq")]
use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams};
#[cfg(feature = "nsg")]
use vicinity::nsg::{NsgIndex, NsgParams};
#[cfg(feature = "nsw")]
use vicinity::nsw::NSWIndex;
use vicinity::prt::ProbabilisticRoutingTest;
#[cfg(feature = "vamana")]
use vicinity::vamana::{VamanaIndex, VamanaParams};

#[path = "common/mod.rs"]
mod common;

const DEFAULT_DATA_DIR: &str = "data/ann-benchmarks/glove-25-angular";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let algo = args
        .windows(2)
        .find(|w| w[0] == "--algo")
        .map(|w| w[1].as_str())
        .unwrap_or("all");

    let data_dir = args
        .windows(2)
        .find(|w| w[0] == "--data")
        .map(|w| w[1].as_str())
        .unwrap_or(DEFAULT_DATA_DIR);

    // Derive dataset name from data dir (e.g. "sift-128-euclidean" from path)
    let dataset_name = std::path::Path::new(data_dir)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let jsonl_out = format!("docs/{dataset_name}.jsonl");

    println!("Loading {dataset_name} data...");
    let (train, dim) = common::load_vectors(&format!("{data_dir}/train.bin"))?;
    let (test, _) = common::load_vectors(&format!("{data_dir}/test.bin"))?;
    let (gt, k_gt) = common::load_neighbors(&format!("{data_dir}/neighbors.bin"))?;
    let k = 10;

    println!(
        "  train: {}×{dim}  test: {}  gt: top-{k_gt}",
        train.len(),
        test.len()
    );
    println!("  output → {jsonl_out}\n");

    // Set output path and dataset name for benchmark records
    std::env::set_var("VICINITY_JSONL_OUT", &jsonl_out);
    std::env::set_var("VICINITY_DATASET", dataset_name);

    // Detect distance metric from dataset name
    let metric = if dataset_name.contains("euclidean") {
        vicinity::distance::DistanceMetric::L2
    } else {
        vicinity::distance::DistanceMetric::Cosine
    };
    std::env::set_var(
        "VICINITY_METRIC",
        if matches!(metric, vicinity::distance::DistanceMetric::L2) {
            "l2"
        } else {
            "cosine"
        },
    );

    // Initialize benchmark context (dataset, dim, host, timestamp, etc.)
    init_benchmark_context(dim, train.len());

    // Load cached results so we can skip already-measured algorithms.
    let cached = existing_algorithms();
    if !cached.is_empty() && algo == "all" {
        println!(
            "Found {} cached algorithm(s). Set VICINITY_FORCE=1 to re-run all.",
            cached.len()
        );
    }

    match algo {
        "hnsw" => run_hnsw(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "nsw")]
        "nsw" => run_nsw(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "ivf_pq")]
        "ivfpq" => run_ivfpq(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "vamana")]
        "vamana" => run_vamana(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "ivf_avq")]
        "ivf_avq" => run_ivf_avq(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "diskann")]
        "diskann" => run_diskann(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "kdtree")]
        "kdtree" => run_kdtree(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "balltree")]
        "balltree" => run_balltree(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "rptree")]
        "rptree" => run_rptree(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "nsg")]
        "nsg" => run_nsg(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "emg")]
        "emg" => run_emg(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "ivf_rabitq")]
        "ivf_rabitq" => run_ivf_rabitq(&train, &test, &gt, k, dim)?,
        "adsampling" => run_adsampling(&train, &test, &gt, k, dim)?,
        "prt" => run_prt(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "sq4")]
        "sq4u" => run_sq4u(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "sq8")]
        "sq8u" => run_sq8u(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "ivf_rabitq")]
        "symphonyqg" => run_symphonyqg(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "ivf_rabitq")]
        "symphonyqg-vr" => run_symphonyqg_vr(&train, &test, &gt, k, dim)?,
        "all" => {
            macro_rules! run_if_not_cached {
                ($key:expr, $call:expr) => {
                    if !should_skip($key, &cached) {
                        $call?;
                    }
                };
            }
            #[cfg(feature = "ivf_pq")]
            run_if_not_cached!("ivfpq", run_ivfpq(&train, &test, &gt, k, dim));
            #[cfg(feature = "nsw")]
            run_if_not_cached!("nsw", run_nsw(&train, &test, &gt, k, dim));
            run_if_not_cached!("hnsw-m16", run_hnsw(&train, &test, &gt, k, dim));
            #[cfg(feature = "vamana")]
            run_if_not_cached!("vamana", run_vamana(&train, &test, &gt, k, dim));
            #[cfg(feature = "ivf_avq")]
            run_if_not_cached!("ivf_avq", run_ivf_avq(&train, &test, &gt, k, dim));
            #[cfg(feature = "diskann")]
            run_if_not_cached!("diskann", run_diskann(&train, &test, &gt, k, dim));
            #[cfg(feature = "kdtree")]
            run_if_not_cached!("kdtree", run_kdtree(&train, &test, &gt, k, dim));
            #[cfg(feature = "balltree")]
            run_if_not_cached!("balltree", run_balltree(&train, &test, &gt, k, dim));
            #[cfg(feature = "rptree")]
            run_if_not_cached!("rptree", run_rptree(&train, &test, &gt, k, dim));
            #[cfg(feature = "nsg")]
            run_if_not_cached!("nsg", run_nsg(&train, &test, &gt, k, dim));
            #[cfg(feature = "emg")]
            run_if_not_cached!("emg", run_emg(&train, &test, &gt, k, dim));
            #[cfg(feature = "ivf_rabitq")]
            run_if_not_cached!("ivf-rabitq-np1", run_ivf_rabitq(&train, &test, &gt, k, dim));
            run_if_not_cached!("adsampling", run_adsampling(&train, &test, &gt, k, dim));
            run_if_not_cached!("prt", run_prt(&train, &test, &gt, k, dim));
            #[cfg(feature = "sq4")]
            run_if_not_cached!("sq4u", run_sq4u(&train, &test, &gt, k, dim));
            #[cfg(feature = "sq8")]
            run_if_not_cached!("sq8u", run_sq8u(&train, &test, &gt, k, dim));
            #[cfg(feature = "ivf_rabitq")]
            run_if_not_cached!("symphonyqg", run_symphonyqg(&train, &test, &gt, k, dim));
        }
        other => {
            eprintln!(
                "Unknown algorithm: {other}. Use: hnsw | nsw | ivfpq | vamana | ivf_avq | diskann | \
                 kdtree | balltree | rptree | nsg | emg | ivf_rabitq | adsampling | prt | sq4u | sq8u | symphonyqg | all"
            );
            std::process::exit(1);
        }
    }

    Ok(())
}

// ─── Algorithm runners ────────────────────────────────────────────────────────

fn run_hnsw(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HNSW ===");
    for (m, m_max, ef_construction) in [(16, 32, 200), (32, 64, 200)] {
        print!("  Building (M={m}, ef_construction={ef_construction})... ");
        let _ = std::io::stdout().flush();
        let metric = if std::env::var("VICINITY_METRIC").as_deref() == Ok("l2") {
            vicinity::distance::DistanceMetric::L2
        } else {
            vicinity::distance::DistanceMetric::Cosine
        };
        let params = HNSWParams {
            m,
            m_max,
            ef_construction,
            metric,
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params)?;
        for (i, v) in train.iter().enumerate() {
            index.add(i as u32, v.clone())?;
        }
        let t0 = Instant::now();
        index.build()?;
        let build_secs = t0.elapsed().as_secs_f64();
        println!("{build_secs:.0}s");

        let algo_name = format!("hnsw-m{m}");
        let params_json =
            format!("\"m\":{m},\"m_max\":{m_max},\"ef_construction\":{ef_construction}");
        for ef in [10, 20, 50, 100, 200, 400] {
            let (recall, qps) = measure(&index, test, gt, k, ef);
            println!(
                "    ef={ef:4}  recall={:.1}%  qps={:.0}",
                recall * 100.0,
                qps
            );
            write_record(
                &algo_name,
                recall,
                qps,
                Some(ef),
                Some(build_secs),
                Some(&params_json),
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "nsw")]
fn run_nsw(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NSW ===");
    let m = 16;
    print!("  Building (M={m})... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let mut index = NSWIndex::new(dim, m, m * 2)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_nsw(&index, test, gt, k, ef);
        println!("  ef={ef:4}  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
        append_jsonl_ef("nsw", recall, qps, Some(ef))?;
    }
    Ok(())
}

#[cfg(feature = "ivf_pq")]
fn run_ivfpq(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IVF-PQ ===");
    // Two configs: 5 codebooks (5-d subspaces, low memory) and 25 codebooks
    // (1-d subspaces, ~SQ8-equivalent, much better recall on low-dim data).
    for (num_clusters, num_codebooks) in [(1024, 5), (1024, 25)] {
        print!("  Building (clusters={num_clusters}, codebooks={num_codebooks})... ");
        let _ = std::io::stdout().flush();
        let t0 = Instant::now();
        let params = IVFPQParams {
            num_clusters,
            nprobe: 1, // will be overridden per measurement
            num_codebooks,
            codebook_size: 256,
            ..Default::default()
        };
        let mut index = IVFPQIndex::new(dim, params)?;
        for (i, v) in train.iter().enumerate() {
            index.add(i as u32, v.clone())?;
        }
        index.build()?;
        println!("{:.0}s", t0.elapsed().as_secs_f64());

        let algo_name = format!("ivfpq-{num_clusters}L-cb{num_codebooks}");
        for nprobe in [4, 8, 16, 32, 64, 128, 256] {
            index.set_nprobe(nprobe);
            let (recall, qps) = measure_ivfpq(&index, test, gt, k);
            println!(
                "  nprobe={nprobe:4}  recall={:.1}%  qps={:.0}",
                recall * 100.0,
                qps
            );
            append_jsonl_ef(&algo_name, recall, qps, Some(nprobe))?;
        }
    }
    Ok(())
}

#[cfg(feature = "vamana")]
fn run_vamana(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vamana ===");
    let params = VamanaParams {
        max_degree: 64,
        alpha: 1.3,
        ef_construction: 200,
        ef_search: 50,
        seed: None,
        ..VamanaParams::default()
    };
    print!("  Building (max_degree=64, alpha=1.3, ef_construction=200)... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let mut index = VamanaIndex::new(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    #[cfg(feature = "parallel")]
    {
        let batch = std::env::var("VICINITY_BUILD_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        index.build_parallel(batch)?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        index.build()?;
    }
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    let algo_name = if cfg!(feature = "parallel") {
        "vamana-par"
    } else {
        "vamana"
    };
    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_vamana(&index, test, gt, k, ef);
        println!("  ef={ef:4}  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
        append_jsonl_ef(algo_name, recall, qps, Some(ef))?;
    }
    Ok(())
}

#[cfg(feature = "ivf_avq")]
fn run_ivf_avq(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ScaNN ===");
    // dim=25: num_codebooks must divide 25; use 5 (5×5-d subspaces)
    let params = IVFAVQParams {
        num_partitions: 512,
        nprobe: 1,
        num_reorder: 500,
        num_codebooks: 5,
        codebook_size: 256,
        seed: 42,
    };
    print!("  Building (partitions=512, codebooks=5, reorder=500)... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let mut index = IVFAVQIndex::new(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    for nprobe in [4, 8, 16, 32, 64, 128, 256] {
        index.set_nprobe(nprobe);
        let (recall, qps) = measure_ivf_avq(&index, test, gt, k);
        println!(
            "  nprobe={nprobe:4}  recall={:.1}%  qps={:.0}",
            recall * 100.0,
            qps
        );
        append_jsonl_ef("ivf_avq", recall, qps, Some(nprobe))?;
    }
    Ok(())
}

// ─── Measurement helpers ──────────────────────────────────────────────────────

fn measure(
    index: &HNSWIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    // Warmup: populate caches before timed run
    for q in test.iter().take(50) {
        let _ = index.search(q, k, ef);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k, ef).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "nsw")]
fn measure_nsw(
    index: &NSWIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search(q, k, ef);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k, ef).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "ivf_pq")]
fn measure_ivfpq(index: &IVFPQIndex, test: &[Vec<f32>], gt: &[Vec<i32>], k: usize) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search(q, k);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "vamana")]
fn measure_vamana(
    index: &VamanaIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search(q, k, ef);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k, ef).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "diskann")]
fn run_diskann(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DiskANN ===");
    let params = DiskANNParams {
        m: 64,
        alpha: 1.3,
        ef_construction: 200,
        ef_search: 50,
        seed: None,
        ..DiskANNParams::default()
    };
    print!("  Building (m=64, alpha=1.3, ef_construction=200)... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let mut index = DiskANNIndex::new(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_diskann(&index, test, gt, k, ef);
        println!("  ef={ef:4}  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
        append_jsonl_ef("diskann", recall, qps, Some(ef))?;
    }
    Ok(())
}

#[cfg(feature = "diskann")]
fn measure_diskann(
    index: &DiskANNIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search(q, k, ef);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k, ef).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "ivf_avq")]
fn measure_ivf_avq(
    index: &IVFAVQIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search(q, k);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "kdtree")]
fn run_kdtree(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== KD-Tree ===");
    print!("  Building... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let mut index = KDTreeIndex::new(dim, KDTreeParams::default())?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    let (recall, qps) = measure_kdtree(&index, test, gt, k);
    println!("  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
    append_jsonl_ef("kdtree", recall, qps, None)?;
    Ok(())
}

#[cfg(feature = "kdtree")]
fn measure_kdtree(index: &KDTreeIndex, test: &[Vec<f32>], gt: &[Vec<i32>], k: usize) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search(q, k);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "balltree")]
fn run_balltree(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ball Tree ===");
    print!("  Building... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let mut index = BallTreeIndex::new(dim, BallTreeParams::default())?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    let (recall, qps) = measure_balltree(&index, test, gt, k);
    println!("  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
    append_jsonl_ef("balltree", recall, qps, None)?;
    Ok(())
}

#[cfg(feature = "balltree")]
fn measure_balltree(
    index: &BallTreeIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search(q, k);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "rptree")]
fn run_rptree(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RP-Forest ===");
    for num_trees in [5, 10, 20, 50] {
        let params = RpForestParams {
            num_trees,
            tree_params: RPTreeParams::default(),
        };
        print!("  Building (num_trees={num_trees})... ");
        let _ = std::io::stdout().flush();
        let t0 = Instant::now();
        let mut index = RpForestIndex::new(dim, params)?;
        for (i, v) in train.iter().enumerate() {
            index.add(i as u32, v.clone())?;
        }
        index.build()?;
        println!("{:.0}s", t0.elapsed().as_secs_f64());

        let (recall, qps) = measure_rptree(&index, test, gt, k);
        println!(
            "  num_trees={num_trees:3}  recall={:.1}%  qps={:.0}",
            recall * 100.0,
            qps
        );
        append_jsonl_ef("rptree", recall, qps, None)?;
    }
    Ok(())
}

#[cfg(feature = "rptree")]
fn measure_rptree(
    index: &RpForestIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search(q, k);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

// ─── NSG ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "nsg")]
fn run_nsg(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NSG ===");
    let params = NsgParams {
        max_degree: 32,
        pool_size: 64,
        knn_degree: 32,
        ..NsgParams::default()
    };
    print!("  Building... ");
    let _ = std::io::stdout().flush();
    let mut index = NsgIndex::new(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    let t0 = Instant::now();
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_nsg(&index, test, gt, k, ef);
        println!("  ef={ef:4}  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
        append_jsonl_ef("nsg", recall, qps, Some(ef))?;
    }
    Ok(())
}

#[cfg(feature = "nsg")]
fn measure_nsg(
    index: &NsgIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search_with_ef(q, k, ef);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search_with_ef(q, k, ef).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

// ─── EMG ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "emg")]
fn run_emg(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== EMG ===");
    let params = EmgParams {
        max_degree: 32,
        candidate_size: 100,
        ..EmgParams::default()
    };
    print!("  Building... ");
    let _ = std::io::stdout().flush();
    let mut index = EmgIndex::new(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    let t0 = Instant::now();
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_emg(&index, test, gt, k, ef);
        println!("  ef={ef:4}  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
        append_jsonl_ef("emg", recall, qps, Some(ef))?;
    }
    Ok(())
}

#[cfg(feature = "emg")]
fn measure_emg(
    index: &EmgIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search_with_ef(q, k, ef);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search_with_ef(q, k, ef).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

// ─── IVF-RaBitQ ──────────────────────────────────────────────────────────────

#[cfg(feature = "ivf_rabitq")]
fn run_ivf_rabitq(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IVF-RaBitQ ===");
    let params = IVFRaBitQParams {
        num_clusters: 256,
        nprobe: 10,
        ..IVFRaBitQParams::default()
    };
    print!("  Building... ");
    let _ = std::io::stdout().flush();
    let mut index = IVFRaBitQIndex::new(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    let t0 = Instant::now();
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    for nprobe in [1, 5, 10, 20, 50, 100] {
        let (recall, qps) = measure_ivf_rabitq(&index, test, gt, k, nprobe);
        println!(
            "  nprobe={nprobe:4}  recall={:.1}%  qps={:.0}",
            recall * 100.0,
            qps
        );
        append_jsonl_ef(&format!("ivf-rabitq-np{nprobe}"), recall, qps, Some(nprobe))?;
    }
    Ok(())
}

#[cfg(feature = "ivf_rabitq")]
fn measure_ivf_rabitq(
    index: &IVFRaBitQIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    nprobe: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search_with_ef(q, k, nprobe);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search_with_ef(q, k, nprobe).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

// ─── ADSampling + HNSW ──────────────────────────────────────────────────────

fn run_adsampling(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ADSampling+HNSW ===");
    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        ..Default::default()
    };
    print!("  Building HNSW... ");
    let _ = std::io::stdout().flush();
    let mut index = HNSWIndex::with_params(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    let t0 = Instant::now();
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    print!("  Building ADSampling state... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let state = ADSamplingState::from_hnsw(&index, ADSamplingParams::default());
    println!("{:.1}s", t0.elapsed().as_secs_f64());

    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_adsampling(&state, &index, test, gt, k, ef);
        println!("  ef={ef:4}  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
        append_jsonl_ef("adsampling", recall, qps, Some(ef))?;
    }
    Ok(())
}

fn measure_adsampling(
    state: &ADSamplingState,
    index: &HNSWIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = state.search_hnsw(index, q, k, ef);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = state.search_hnsw(index, q, k, ef).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

// ─── PRT (Probabilistic Routing Test) ────────────────────────────────────────

fn run_prt(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PRT (Projection-Augmented HNSW) ===");
    let metric = if std::env::var("VICINITY_METRIC").as_deref() == Ok("l2") {
        vicinity::distance::DistanceMetric::L2
    } else {
        vicinity::distance::DistanceMetric::Cosine
    };
    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        metric,
        ..Default::default()
    };
    print!("  Building HNSW... ");
    let _ = std::io::stdout().flush();
    let mut index = HNSWIndex::with_params(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    let t0 = Instant::now();
    index.build()?;
    let hnsw_build_secs = t0.elapsed().as_secs_f64();
    println!("{hnsw_build_secs:.0}s");

    // Build PRT with k = dim/4 projections, uncapped for high-d datasets.
    // Previous cap at 128 caused 81% recall ceiling on GIST-960 (dim=960).
    let num_proj = (dim / 4).max(8);
    print!("  Building PRT (k={num_proj})... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let mut prt = ProbabilisticRoutingTest::new(dim, num_proj, Some(42));
    prt.project_database(index.raw_vectors());
    let prt_build_secs = t0.elapsed().as_secs_f64();
    println!("{prt_build_secs:.1}s");

    let total_build = hnsw_build_secs + prt_build_secs;
    let params_json = format!(
        "\"m\":16,\"m_max\":32,\"ef_construction\":200,\"num_projections\":{num_proj},\
         \"initial_ratio\":1.5,\"decay\":0.95"
    );
    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps, avg_full_ratio) = measure_prt(&index, &prt, test, gt, k, ef);
        println!(
            "    ef={ef:4}  recall={:.1}%  qps={:.0}  full_dist={:.0}%",
            recall * 100.0,
            qps,
            avg_full_ratio * 100.0,
        );
        write_record(
            "prt",
            recall,
            qps,
            Some(ef),
            Some(total_build),
            Some(&params_json),
        )?;
    }
    Ok(())
}

fn measure_prt(
    index: &HNSWIndex,
    prt: &ProbabilisticRoutingTest,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64) {
    // Warmup
    for q in test.iter().take(50) {
        let _ = index.search_prt(q, k, ef, prt, 1.5, 0.95);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    let mut full_dist_sum = 0u64;
    for (i, q) in test.iter().enumerate() {
        let (res, full_dists) = index
            .search_prt(q, k, ef, prt, 1.5, 0.95)
            .unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
        full_dist_sum += full_dists as u64;
    }
    let elapsed = t.elapsed().as_secs_f64();
    // Estimate what fraction of candidate distances used full computation
    // (vs filtered by PRT). Rough estimate: full_dists / (test.len() * ef * avg_neighbors).
    let avg_full_per_query = full_dist_sum as f64 / test.len() as f64;
    // Normalize against the total candidates that would be visited at this ef
    // (heuristic: ~ef * 5 for typical HNSW graph connectivity)
    let estimated_total = ef as f64 * 5.0;
    let full_ratio = (avg_full_per_query / estimated_total).min(1.0);
    (
        recall_sum / test.len() as f64,
        test.len() as f64 / elapsed,
        full_ratio,
    )
}

// ─── SQ4U (4-bit quantized HNSW traversal) ───────────────────────────────────

#[cfg(feature = "sq4")]
fn run_sq4u(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SQ4U (4-bit quantized HNSW) ===");
    let metric = if std::env::var("VICINITY_METRIC").as_deref() == Ok("l2") {
        vicinity::distance::DistanceMetric::L2
    } else {
        vicinity::distance::DistanceMetric::Cosine
    };
    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        metric,
        ..Default::default()
    };
    print!("  Building (M=16, M_max=32)... ");
    let _ = std::io::stdout().flush();
    let mut index = HNSWSq4Index::with_params(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add_slice(i as u32, v)?;
    }
    let t0 = Instant::now();
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    // Reranked search: sweep ef with rerank_pool = ef (all candidates get reranked)
    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_sq4u_reranked(&index, test, gt, k, ef, ef);
        println!(
            "    ef={ef:4}  recall={:.1}%  qps={:.0}",
            recall * 100.0,
            qps
        );
        append_jsonl_ef("sq4u", recall, qps, Some(ef))?;
    }
    Ok(())
}

#[cfg(feature = "sq4")]
fn measure_sq4u_reranked(
    index: &HNSWSq4Index,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
    rerank_pool: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search_reranked(q, k, ef, rerank_pool);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index
            .search_reranked(q, k, ef, rerank_pool)
            .unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

// ─── SQ8U (8-bit quantized HNSW traversal) ───────────────────────────────────

#[cfg(feature = "sq8")]
fn run_sq8u(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SQ8U (8-bit quantized HNSW) ===");
    let metric = if std::env::var("VICINITY_METRIC").as_deref() == Ok("l2") {
        vicinity::distance::DistanceMetric::L2
    } else {
        vicinity::distance::DistanceMetric::Cosine
    };
    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        metric,
        ..Default::default()
    };
    print!("  Building (M=16, M_max=32)... ");
    let _ = std::io::stdout().flush();
    let mut index = HNSWSq8Index::with_params(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add_slice(i as u32, v)?;
    }
    let t0 = Instant::now();
    #[cfg(feature = "parallel")]
    {
        let batch = std::env::var("VICINITY_BUILD_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        index.build_parallel(batch)?;
    }
    #[cfg(not(feature = "parallel"))]
    {
        index.build()?;
    }
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    // Reranked search: sweep ef with rerank_pool = ef
    let algo_name = if cfg!(feature = "parallel") {
        "sq8u-par"
    } else {
        "sq8u"
    };
    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_sq8u_reranked(&index, test, gt, k, ef, ef);
        println!(
            "    ef={ef:4}  recall={:.1}%  qps={:.0}",
            recall * 100.0,
            qps
        );
        append_jsonl_ef(algo_name, recall, qps, Some(ef))?;
    }
    Ok(())
}

#[cfg(feature = "sq8")]
fn measure_sq8u_reranked(
    index: &HNSWSq8Index,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
    rerank_pool: usize,
) -> (f64, f64) {
    for q in test.iter().take(50) {
        let _ = index.search_reranked(q, k, ef, rerank_pool);
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index
            .search_reranked(q, k, ef, rerank_pool)
            .unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

// ─── SymphonyQG-VR (vertex-relative RaBitQ) ─────────────────────────────────

#[cfg(feature = "ivf_rabitq")]
fn run_symphonyqg_vr(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use vicinity::hnsw::symphony_qg::SymphonyQGVRIndex;

    println!("=== SymphonyQG-VR (vertex-relative RaBitQ) ===");
    let metric = if std::env::var("VICINITY_METRIC").as_deref() == Ok("l2") {
        vicinity::distance::DistanceMetric::L2
    } else {
        vicinity::distance::DistanceMetric::Cosine
    };
    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        metric,
        ..Default::default()
    };
    print!("  Building HNSW + per-edge RaBitQ codes... ");
    let _ = std::io::stdout().flush();
    let mut index = SymphonyQGVRIndex::new(dim, params, qntz::rabitq::RaBitQConfig::bits4(), 42)?;
    for (i, v) in train.iter().enumerate() {
        index.add_slice(i as u32, v)?;
    }
    let t0 = Instant::now();
    index.build()?;
    let build_secs = t0.elapsed().as_secs_f64();
    println!("{build_secs:.0}s");

    let params_json =
        "\"m\":16,\"m_max\":32,\"ef_construction\":200,\"rabitq_bits\":4,\"vertex_relative\":true";

    // Reranked search (primary path)
    for ef in [10, 20, 50, 100, 200, 400] {
        // Warmup
        for q in test.iter().take(50) {
            let _ = index.search_reranked(q, k, ef, ef * 2);
        }
        let t = Instant::now();
        let mut recall_sum = 0.0;
        for (i, q) in test.iter().enumerate() {
            let res = index.search_reranked(q, k, ef, ef * 2).unwrap_or_default();
            recall_sum += recall_at_k(&res, &gt[i], k);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let recall = recall_sum / test.len() as f64;
        let qps = test.len() as f64 / elapsed;
        println!("    ef={ef:4}  recall={:.1}%  qps={qps:.0}", recall * 100.0);
        write_record(
            "symphonyqg-vr",
            recall,
            qps,
            Some(ef),
            Some(build_secs),
            Some(params_json),
        )?;
    }
    Ok(())
}

// ─── Utilities ───────────────────────────────────────────────────────────────

fn recall_at_k(results: &[(u32, f32)], ground_truth: &[i32], k: usize) -> f64 {
    let gt: HashSet<u32> = ground_truth.iter().take(k).map(|&i| i as u32).collect();
    let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
    gt.intersection(&found).count() as f64 / k as f64
}

// ─── SymphonyQG (RaBitQ quantized graph) ────────────────────────────────────

#[cfg(feature = "ivf_rabitq")]
fn run_symphonyqg(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SymphonyQG (RaBitQ graph traversal) ===");
    let metric = if std::env::var("VICINITY_METRIC").as_deref() == Ok("l2") {
        vicinity::distance::DistanceMetric::L2
    } else {
        vicinity::distance::DistanceMetric::Cosine
    };
    print!("  Building HNSW + RaBitQ codes... ");
    let _ = std::io::stdout().flush();
    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        metric,
        ..Default::default()
    };
    let mut index =
        SymphonyQGIndex::with_hnsw_params(dim, params, qntz::rabitq::RaBitQConfig::bits4(), 42)?;
    for (i, v) in train.iter().enumerate() {
        index.add_slice(i as u32, v)?;
    }
    let t0 = Instant::now();
    index.build()?;
    let build_secs = t0.elapsed().as_secs_f64();
    println!("{build_secs:.0}s");

    let params_json = "\"m\":16,\"m_max\":32,\"ef_construction\":200,\"rabitq_bits\":4";

    // Quantized search (no rerank) -- tests pure RaBitQ approximation
    println!("  --- quantized (no rerank) ---");
    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_symphonyqg(&index, test, gt, k, ef, false);
        println!("    ef={ef:4}  recall={:.1}%  qps={qps:.0}", recall * 100.0);
        write_record(
            "symphonyqg-raw",
            recall,
            qps,
            Some(ef),
            Some(build_secs),
            Some(params_json),
        )?;
    }

    // Reranked search -- recommended path
    println!("  --- reranked ---");
    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_symphonyqg(&index, test, gt, k, ef, true);
        println!("    ef={ef:4}  recall={:.1}%  qps={qps:.0}", recall * 100.0);
        write_record(
            "symphonyqg",
            recall,
            qps,
            Some(ef),
            Some(build_secs),
            Some(params_json),
        )?;
    }
    Ok(())
}

#[cfg(feature = "ivf_rabitq")]
fn measure_symphonyqg(
    index: &SymphonyQGIndex,
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    ef: usize,
    rerank: bool,
) -> (f64, f64) {
    // Warmup
    for q in test.iter().take(50) {
        if rerank {
            let _ = index.search_reranked(q, k, ef, ef * 2);
        } else {
            let _ = index.search(q, k, ef);
        }
    }
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = if rerank {
            index.search_reranked(q, k, ef, ef * 2).unwrap_or_default()
        } else {
            index.search(q, k, ef).unwrap_or_default()
        };
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

thread_local! {
    static SHA: String = git_sha();
    static HOST: String = hostname();
    static TIMESTAMP: String = timestamp_iso8601();
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn timestamp_iso8601() -> String {
    // UTC timestamp without external deps -- shell fallback
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Session-level context shared by all records in a benchmark run.
struct BenchmarkContext {
    dataset: String,
    dim: usize,
    n_vectors: usize,
    git_sha: String,
    host: String,
    timestamp: String,
    metric: String,
}

impl BenchmarkContext {
    fn from_env(dim: usize, n_vectors: usize) -> Self {
        let dataset = std::env::var("VICINITY_DATASET").unwrap_or_else(|_| "unknown".into());
        let metric = std::env::var("VICINITY_METRIC").unwrap_or_else(|_| "cosine".into());
        Self {
            dataset,
            dim,
            n_vectors,
            git_sha: SHA.with(|s| s.clone()),
            host: HOST.with(|s| s.clone()),
            timestamp: TIMESTAMP.with(|s| s.clone()),
            metric,
        }
    }

    fn to_json_prefix(&self) -> String {
        format!(
            "\"dataset\":\"{}\",\"dim\":{},\"n_vectors\":{},\"metric\":\"{}\",\
             \"git_sha\":\"{}\",\"host\":\"{}\",\"timestamp\":\"{}\"",
            self.dataset,
            self.dim,
            self.n_vectors,
            self.metric,
            self.git_sha,
            self.host,
            self.timestamp,
        )
    }
}

thread_local! {
    static CTX: std::cell::RefCell<Option<BenchmarkContext>> = const { std::cell::RefCell::new(None) };
}

fn init_benchmark_context(dim: usize, n_vectors: usize) {
    CTX.with(|c| {
        *c.borrow_mut() = Some(BenchmarkContext::from_env(dim, n_vectors));
    });
}

/// Write a benchmark record. `params` is a pre-formatted JSON object body (no braces).
fn write_record(
    algorithm: &str,
    recall: f64,
    qps: f64,
    ef_search: Option<usize>,
    build_time_secs: Option<f64>,
    params: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ctx_json = CTX.with(|c| {
        c.borrow()
            .as_ref()
            .map(|ctx| ctx.to_json_prefix())
            .unwrap_or_default()
    });

    let mut fields =
        format!("\"algorithm\":\"{algorithm}\",\"recall_at_10\":{recall:.4},\"qps\":{qps:.1}");
    if let Some(ef) = ef_search {
        fields.push_str(&format!(",\"ef_search\":{ef}"));
    }
    if let Some(bt) = build_time_secs {
        fields.push_str(&format!(",\"build_time_secs\":{bt:.1}"));
    }
    if let Some(p) = params {
        fields.push_str(&format!(",\"params\":{{{p}}}"));
    }
    if !ctx_json.is_empty() {
        fields.push_str(&format!(",{ctx_json}"));
    }

    let line = format!("{{{fields}}}\n");
    let out_path =
        std::env::var("VICINITY_JSONL_OUT").unwrap_or_else(|_| "docs/results.jsonl".into());
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)?;
    let mut w = BufWriter::new(file);
    w.write_all(line.as_bytes())?;
    Ok(())
}

/// Convenience: write a record with just ef_search (most common case).
fn append_jsonl_ef(
    algorithm: &str,
    recall: f64,
    qps: f64,
    ef: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    write_record(algorithm, recall, qps, ef, None, None)
}

/// Convenience: write a record without ef_search.
/// Check if results already exist for an algorithm in the output file.
/// Returns a map of algorithm name -> git SHA of the most recent result.
fn existing_algorithms() -> std::collections::HashMap<String, String> {
    let out_path =
        std::env::var("VICINITY_JSONL_OUT").unwrap_or_else(|_| "docs/results.jsonl".into());
    let mut algos = std::collections::HashMap::new();
    if let Ok(content) = std::fs::read_to_string(&out_path) {
        for line in content.lines() {
            let algo = extract_json_str(line, "algorithm");
            let sha = extract_json_str(line, "git_sha");
            if let Some(a) = algo {
                algos.insert(a, sha.unwrap_or_default());
            }
        }
    }
    algos
}

fn extract_json_str(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", key);
    let start = line.find(&needle)?;
    let rest = &line[start + needle.len()..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Returns true if this algorithm should be skipped (already has results from this SHA).
fn should_skip(algo_name: &str, cached: &std::collections::HashMap<String, String>) -> bool {
    if std::env::var("VICINITY_FORCE").is_ok() {
        return false;
    }
    if let Some(cached_sha) = cached.get(algo_name) {
        let current_sha = SHA.with(|s| s.clone());
        if !current_sha.is_empty() && !cached_sha.is_empty() && *cached_sha != current_sha {
            println!(
                "  [stale] {algo_name} -- cached from {cached_sha}, current is {current_sha}; re-running"
            );
            return false;
        }
        println!("  [cached] {algo_name} -- skipping (set VICINITY_FORCE=1 to re-run)");
        true
    } else {
        false
    }
}
