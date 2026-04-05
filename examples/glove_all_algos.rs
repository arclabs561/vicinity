//! Multi-algorithm GloVe-25 benchmark — outputs JSONL for plot_comparison.py.
//!
//! ```bash
//! # All algorithms
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,scann,diskann" -- --algo all
//!
//! # Individual algorithms
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,scann,diskann" -- --algo hnsw
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,scann,diskann" -- --algo nsw
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,scann,diskann" -- --algo ivfpq
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,scann,diskann" -- --algo vamana
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,scann,diskann" -- --algo scann
//! cargo run --example glove_all_algos --release --features "hnsw,nsw,ivf_pq,vamana,scann,diskann" -- --algo diskann
//!
//! # Output appended to doc/glove-25-angular.jsonl; regenerate plot with:
//! # uv run scripts/plot_comparison.py doc/glove-25-angular.jsonl
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

#[cfg(feature = "diskann")]
use vicinity::diskann::{DiskANNIndex, DiskANNParams};
use vicinity::hnsw::{HNSWIndex, HNSWParams};
#[cfg(feature = "ivf_pq")]
use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};
#[cfg(feature = "nsw")]
use vicinity::nsw::NSWIndex;
#[cfg(feature = "scann")]
use vicinity::scann::{SCANNIndex, SCANNParams};
#[cfg(feature = "vamana")]
use vicinity::vamana::{VamanaIndex, VamanaParams};

const DATA_DIR: &str = "data/ann-benchmarks/glove-25-angular";
const JSONL_OUT: &str = "doc/glove-25-angular.jsonl";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let algo = args
        .windows(2)
        .find(|w| w[0] == "--algo")
        .map(|w| w[1].as_str())
        .unwrap_or("all");

    println!("Loading GloVe-25 data...");
    let (train, dim) = load_vectors(&format!("{DATA_DIR}/train.bin"))?;
    let (test, _) = load_vectors(&format!("{DATA_DIR}/test.bin"))?;
    let (gt, k_gt) = load_neighbors(&format!("{DATA_DIR}/neighbors.bin"))?;
    let k = 10;

    println!(
        "  train: {}×{dim}  test: {}  gt: top-{k_gt}",
        train.len(),
        test.len()
    );
    println!("  output → {JSONL_OUT}\n");

    match algo {
        "hnsw" => run_hnsw(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "nsw")]
        "nsw" => run_nsw(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "ivf_pq")]
        "ivfpq" => run_ivfpq(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "vamana")]
        "vamana" => run_vamana(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "scann")]
        "scann" => run_scann(&train, &test, &gt, k, dim)?,
        #[cfg(feature = "diskann")]
        "diskann" => run_diskann(&train, &test, &gt, k, dim)?,
        "all" => {
            #[cfg(feature = "ivf_pq")]
            run_ivfpq(&train, &test, &gt, k, dim)?;
            #[cfg(feature = "nsw")]
            run_nsw(&train, &test, &gt, k, dim)?;
            run_hnsw(&train, &test, &gt, k, dim)?;
            #[cfg(feature = "vamana")]
            run_vamana(&train, &test, &gt, k, dim)?;
            #[cfg(feature = "scann")]
            run_scann(&train, &test, &gt, k, dim)?;
            #[cfg(feature = "diskann")]
            run_diskann(&train, &test, &gt, k, dim)?;
        }
        other => {
            eprintln!(
                "Unknown algorithm: {other}. Use: hnsw | nsw | ivfpq | vamana | scann | diskann | all"
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
        let params = HNSWParams {
            m,
            m_max,
            ef_construction,
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
        ..Default::default()
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

#[cfg(feature = "scann")]
fn run_scann(
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    gt: &[Vec<i32>],
    k: usize,
    dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ScaNN ===");
    // dim=25: num_codebooks must divide 25; use 5 (5×5-d subspaces)
    let params = SCANNParams {
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
    let mut index = SCANNIndex::new(dim, params)?;
    for (i, v) in train.iter().enumerate() {
        index.add(i as u32, v.clone())?;
    }
    index.build()?;
    println!("{:.0}s", t0.elapsed().as_secs_f64());

    for nprobe in [4, 8, 16, 32, 64, 128, 256] {
        index.set_nprobe(nprobe);
        let (recall, qps) = measure_scann(&index, test, gt, k);
        println!(
            "  nprobe={nprobe:4}  recall={:.1}%  qps={:.0}",
            recall * 100.0,
            qps
        );
        append_jsonl("scann", recall, qps)?;
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
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k, ef).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

#[cfg(feature = "scann")]
fn measure_scann(index: &SCANNIndex, test: &[Vec<f32>], gt: &[Vec<i32>], k: usize) -> (f64, f64) {
    let t = Instant::now();
    let mut recall_sum = 0.0;
    for (i, q) in test.iter().enumerate() {
        let res = index.search(q, k).unwrap_or_default();
        recall_sum += recall_at_k(&res, &gt[i], k);
    }
    let elapsed = t.elapsed().as_secs_f64();
    (recall_sum / test.len() as f64, test.len() as f64 / elapsed)
}

fn recall_at_k(results: &[(u32, f32)], ground_truth: &[i32], k: usize) -> f64 {
    let gt: HashSet<u32> = ground_truth.iter().take(k).map(|&i| i as u32).collect();
    let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
    gt.intersection(&found).count() as f64 / k as f64
}

fn append_jsonl(algorithm: &str, recall: f64, qps: f64) -> Result<(), Box<dyn std::error::Error>> {
    let line = format!(
        "{{\"algorithm\":\"{algorithm}\",\"recall_at_10\":{recall:.4},\"qps\":{qps:.1}}}\n"
    );
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(JSONL_OUT)?;
    let mut w = BufWriter::new(file);
    w.write_all(line.as_bytes())?;
    Ok(())
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
