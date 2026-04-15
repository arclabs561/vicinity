//! Multi-algorithm GloVe-25 benchmark — outputs JSONL for plot_comparison.py.
//!
//! ```bash
//! # All algorithms
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,ivf_avq,diskann,kdtree,balltree,rptree" -- --algo all
//!
//! # Individual algorithms
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

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
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
#[cfg(feature = "vamana")]
use vicinity::vamana::{VamanaIndex, VamanaParams};

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
    let (train, dim) = load_vectors(&format!("{data_dir}/train.bin"))?;
    let (test, _) = load_vectors(&format!("{data_dir}/test.bin"))?;
    let (gt, k_gt) = load_neighbors(&format!("{data_dir}/neighbors.bin"))?;
    let k = 10;

    println!(
        "  train: {}×{dim}  test: {}  gt: top-{k_gt}",
        train.len(),
        test.len()
    );
    println!("  output → {jsonl_out}\n");

    // Set output path for append_jsonl (used by all runners)
    std::env::set_var("VICINITY_JSONL_OUT", &jsonl_out);

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
        }
        other => {
            eprintln!(
                "Unknown algorithm: {other}. Use: hnsw | nsw | ivfpq | vamana | ivf_avq | diskann | \
                 kdtree | balltree | rptree | nsg | emg | ivf_rabitq | adsampling | all"
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
        println!("{:.0}s", t0.elapsed().as_secs_f64());

        let algo_name = format!("hnsw-m{m}");
        for ef in [10, 20, 50, 100, 200, 400] {
            let (recall, qps) = measure(&index, test, gt, k, ef);
            println!(
                "    ef={ef:4}  recall={:.1}%  qps={:.0}",
                recall * 100.0,
                qps
            );
            append_jsonl(&algo_name, recall, qps)?;
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
        append_jsonl("nsw", recall, qps)?;
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
            append_jsonl(&algo_name, recall, qps)?;
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
    };
    print!("  Building (max_degree=64, alpha=1.3, ef_construction=200)... ");
    let _ = std::io::stdout().flush();
    let t0 = Instant::now();
    let mut index = VamanaIndex::new(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    for ef in [10, 20, 50, 100, 200, 400] {
        let (recall, qps) = measure_vamana(&index, test, gt, k, ef);
        println!("  ef={ef:4}  recall={:.1}%  qps={:.0}", recall * 100.0, qps);
        append_jsonl("vamana", recall, qps)?;
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
        append_jsonl("ivf_avq", recall, qps)?;
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
        append_jsonl("diskann", recall, qps)?;
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
    append_jsonl("kdtree", recall, qps)?;
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
    append_jsonl("balltree", recall, qps)?;
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
        append_jsonl("rptree", recall, qps)?;
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
        append_jsonl("nsg", recall, qps)?;
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
        append_jsonl("emg", recall, qps)?;
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
        append_jsonl(&format!("ivf-rabitq-np{nprobe}"), recall, qps)?;
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
        append_jsonl("adsampling", recall, qps)?;
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

// ─── Utilities ───────────────────────────────────────────────────────────────

fn recall_at_k(results: &[(u32, f32)], ground_truth: &[i32], k: usize) -> f64 {
    let gt: HashSet<u32> = ground_truth.iter().take(k).map(|&i| i as u32).collect();
    let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
    gt.intersection(&found).count() as f64 / k as f64
}

fn append_jsonl(algorithm: &str, recall: f64, qps: f64) -> Result<(), Box<dyn std::error::Error>> {
    let line = format!(
        "{{\"algorithm\":\"{algorithm}\",\"recall_at_10\":{recall:.4},\"qps\":{qps:.1}}}\n"
    );
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

/// Check if results already exist for an algorithm in the output file.
/// Returns the set of algorithm names that have at least one result.
fn existing_algorithms() -> HashSet<String> {
    let out_path =
        std::env::var("VICINITY_JSONL_OUT").unwrap_or_else(|_| "docs/results.jsonl".into());
    let mut algos = HashSet::new();
    if let Ok(content) = std::fs::read_to_string(&out_path) {
        for line in content.lines() {
            // Parse "algorithm":"<name>" from JSONL
            if let Some(start) = line.find("\"algorithm\":\"") {
                let rest = &line[start + 13..];
                if let Some(end) = rest.find('"') {
                    algos.insert(rest[..end].to_string());
                }
            }
        }
    }
    algos
}

/// Returns true if this algorithm should be skipped (already has results).
fn should_skip(algo_name: &str, cached: &HashSet<String>) -> bool {
    if std::env::var("VICINITY_FORCE").is_ok() {
        return false;
    }
    if cached.contains(algo_name) {
        println!("  [cached] {algo_name} -- skipping (set VICINITY_FORCE=1 to re-run)");
        true
    } else {
        false
    }
}

// ─── Data loading (same format as glove_benchmark.rs) ────────────────────────

fn load_vectors(path: &str) -> Result<(Vec<Vec<f32>>, usize), Box<dyn std::error::Error>> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let n = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
    let d = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let mut buf = vec![0u8; n * d * 4];
    f.read_exact(&mut buf)?;
    let vecs = (0..n)
        .map(|i| {
            (0..d)
                .map(|j| {
                    let o = (i * d + j) * 4;
                    f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
                })
                .collect()
        })
        .collect();
    Ok((vecs, d))
}

fn load_neighbors(path: &str) -> Result<(Vec<Vec<i32>>, usize), Box<dyn std::error::Error>> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let n = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
    let k = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let mut buf = vec![0u8; n * k * 4];
    f.read_exact(&mut buf)?;
    let nbrs = (0..n)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let o = (i * k + j) * 4;
                    i32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
                })
                .collect()
        })
        .collect();
    Ok((nbrs, k))
}
