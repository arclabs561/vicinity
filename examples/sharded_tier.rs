//! Sharded, file-backed tier: build `K` HNSW segments and query `nprobe` of them.
//!
//! The question is whether sharding makes a tier affordable without destroying
//! recall. It bounds the build peak to one shard (a 100M index needs ~199 GiB as
//! one graph and ~13 GiB per shard at K = 16), and it requires the shard to be
//! queried from disk, because the sum of the shards is still far larger than RAM.
//!
//! Run:
//!
//! ```text
//! cargo run --release --features "hnsw,persistence" --example sharded_tier -- \
//!   --points 1m --shards 16 --nprobe 1,2,4 --ef 64 --json /tmp/sharded-1m.jsonl
//! ```
//!
//! Every number this prints is against an exact top-10 scan of the same corpus.
//! Nothing here is estimated.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::time::Instant;
use vicinity::distance::DistanceMetric;
use vicinity::hnsw::{FileCacheConfig, HNSWIndex, HNSWParams, HNSWSegmentSearcher};
use vicinity::persistence::directory::FsDirectory;
use vicinity::persistence::hnsw::HNSWSegmentWriter;

const K: usize = 10;
const QUERIES: usize = 200;

fn parse_points(text: &str) -> usize {
    match text {
        "10k" => 10_000,
        "100k" => 100_000,
        "1m" | "1M" => 1_000_000,
        "3m" | "3M" => 3_000_000,
        "10m" | "10M" => 10_000_000,
        other => other.parse().expect("--points takes 100k, 1m or a number"),
    }
}

fn corpus(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n * dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn row(vectors: &[f32], dim: usize, id: usize) -> &[f32] {
    &vectors[id * dim..(id + 1) * dim]
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// k-means++-style seeding on a sample, then Lloyd iterations on the sample.
///
/// The partitioner only has to route queries and balance shards; it does not
/// have to be a good quantiser, so the sample is bounded and the iteration count
/// is small. The centroids it produces are what `nprobe` is measured against.
fn train_centroids(vectors: &[f32], n: usize, dim: usize, k: usize, seed: u64) -> Vec<f32> {
    let sample = n.min(20_000);
    let mut state = seed ^ 0x5EED_1234;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 33) as usize) % n
    };

    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(row(vectors, dim, next()).to_vec());
    while centroids.len() < k {
        // Pick the sample point farthest from any chosen centroid, weighted by
        // squared distance: the classic k-means++ draw, done with a linear scan
        // because the sample is small.
        let mut best = (0usize, f32::NEG_INFINITY);
        for _ in 0..64 {
            let candidate = next();
            let point = row(vectors, dim, candidate);
            let nearest = centroids
                .iter()
                .map(|c| sq_dist(point, c))
                .fold(f32::INFINITY, f32::min);
            if nearest > best.1 {
                best = (candidate, nearest);
            }
        }
        centroids.push(row(vectors, dim, best.0).to_vec());
    }

    let mut flat: Vec<f32> = centroids.concat();
    for _ in 0..4 {
        let mut sums = vec![0f64; k * dim];
        let mut counts = vec![0usize; k];
        for _ in 0..sample {
            let id = next();
            let point = row(vectors, dim, id);
            let (mut best, mut best_dist) = (0usize, f32::INFINITY);
            for c in 0..k {
                let d = sq_dist(point, &flat[c * dim..(c + 1) * dim]);
                if d < best_dist {
                    best = c;
                    best_dist = d;
                }
            }
            counts[best] += 1;
            for d in 0..dim {
                sums[best * dim + d] += f64::from(point[d]);
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                continue;
            }
            for d in 0..dim {
                flat[c * dim + d] = (sums[c * dim + d] / counts[c] as f64) as f32;
            }
        }
    }
    flat
}

fn assign(vectors: &[f32], n: usize, dim: usize, centroids: &[f32], k: usize) -> Vec<u32> {
    (0..n)
        .map(|id| {
            let point = row(vectors, dim, id);
            let (mut best, mut best_dist) = (0u32, f32::INFINITY);
            for c in 0..k {
                let d = sq_dist(point, &centroids[c * dim..(c + 1) * dim]);
                if d < best_dist {
                    best = c as u32;
                    best_dist = d;
                }
            }
            best
        })
        .collect()
}

fn rss_mib() -> Option<f64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kib: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(kib / 1024.0)
}

fn exact_truth(vectors: &[f32], n: usize, dim: usize, queries: &[f32]) -> Vec<Vec<u32>> {
    (0..QUERIES)
        .map(|q| {
            let query = row(queries, dim, q);
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|id| (sq_dist(query, row(vectors, dim, id)), id as u32))
                .collect();
            scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            scored.iter().take(K).map(|(_, id)| *id).collect()
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut points = 100_000usize;
    let mut shards = 16usize;
    let mut nprobes: Vec<usize> = vec![1, 2, 4];
    let mut ef = 64usize;
    let mut json: Option<PathBuf> = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--points" => {
                index += 1;
                points = parse_points(&args[index]);
            }
            "--shards" => {
                index += 1;
                shards = args[index].parse().expect("--shards");
            }
            "--nprobe" => {
                index += 1;
                nprobes = args[index]
                    .split(',')
                    .map(|v| v.parse().expect("--nprobe"))
                    .collect();
            }
            "--ef" => {
                index += 1;
                ef = args[index].parse().expect("--ef");
            }
            "--json" => {
                index += 1;
                json = Some(PathBuf::from(&args[index]));
            }
            other => panic!("unknown argument {other}"),
        }
        index += 1;
    }

    let dim = 96usize;
    let vectors = corpus(points, dim, 0x5348415244);
    let queries = corpus(QUERIES, dim, 0x51554552);
    let truth = exact_truth(&vectors, points, dim, &queries);
    println!("corpus {points} x {dim}, {QUERIES} queries, exact top-{K}");

    let centroids = train_centroids(&vectors, points, dim, shards, 0xCE17);
    let assignment = assign(&vectors, points, dim, &centroids, shards);
    let mut sizes = vec![0usize; shards];
    for &shard in &assignment {
        sizes[shard as usize] += 1;
    }
    println!(
        "partition sizes: min {}, max {}, mean {}",
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap(),
        points / shards
    );

    let root = std::env::temp_dir().join(format!("vicinity-sharded-{points}-{shards}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    let baseline_rss = rss_mib().unwrap_or(f64::NAN);
    let build_start = Instant::now();
    let mut peak_rss: f64 = 0.0;
    let mut paths: Vec<PathBuf> = Vec::with_capacity(shards);
    for (shard, &shard_size) in sizes.iter().enumerate() {
        let params = HNSWParams {
            m: 16,
            ef_construction: 200,
            metric: DistanceMetric::L2,
            seed: Some(42),
            ..Default::default()
        };
        let mut graph = HNSWIndex::with_params(dim, params).unwrap();
        for (id, &owner) in assignment.iter().enumerate() {
            if owner as usize == shard {
                graph
                    .add(id as u32, row(&vectors, dim, id).to_vec())
                    .unwrap();
            }
        }
        graph.build().unwrap();

        let dir = root.join(format!("shard_{shard}"));
        FsDirectory::new(&dir).unwrap();
        let mut writer =
            HNSWSegmentWriter::new(Box::new(FsDirectory::new(&dir).unwrap()), shard as u64);
        writer.write_hnsw_index(&graph).unwrap();
        paths.push(dir.join(format!("segments/segment_hnsw_{shard}")));
        drop(graph);

        let rss = rss_mib().unwrap_or(f64::NAN);
        peak_rss = peak_rss.max(rss - baseline_rss);
        println!(
            "shard {shard}: {} vectors, rss +{:.0} MiB",
            shard_size,
            rss - baseline_rss
        );
    }
    let build_s = build_start.elapsed().as_secs_f64();
    println!("build: {build_s:.1}s for {shards} shards, peak rss +{peak_rss:.0} MiB");

    let mut searchers: Vec<HNSWSegmentSearcher> = Vec::with_capacity(shards);
    for path in &paths {
        searchers
            .push(HNSWSegmentSearcher::open(path, FileCacheConfig::with_budget(4 << 20)).unwrap());
    }

    let mut rows = String::new();
    use std::fmt::Write as _;
    for &nprobe in &nprobes {
        let mut hits = 0usize;
        let mut latencies: Vec<f64> = Vec::with_capacity(QUERIES);
        for (q, want) in truth.iter().enumerate() {
            let query = row(&queries, dim, q);
            let started = Instant::now();

            let mut nearest: Vec<(f32, usize)> = (0..shards)
                .map(|c| (sq_dist(query, &centroids[c * dim..(c + 1) * dim]), c))
                .collect();
            nearest.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

            let mut merged: Vec<(u32, f32)> = Vec::new();
            for &(_, shard) in nearest.iter().take(nprobe.min(shards)) {
                merged.extend(searchers[shard].search(query, K, ef).unwrap());
            }
            merged.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            merged.truncate(K);

            latencies.push(started.elapsed().as_secs_f64() * 1e6);
            hits += merged.iter().filter(|(id, _)| want.contains(id)).count();
        }
        latencies.sort_by(|a, b| a.total_cmp(b));
        let recall = hits as f64 / (QUERIES * K) as f64;
        let p50 = latencies[latencies.len() / 2];
        println!("nprobe={nprobe} ef={ef}: recall {recall:.4}, p50 {p50:.0} us");
        let _ = writeln!(
            rows,
            "{{\"family\":\"hnsw-sharded\",\"storage_mode\":\"file\",\"dataset\":\"uniform@{points}\",\
             \"knob\":\"nprobe\",\"knob_value\":{nprobe},\"recall\":{recall:.4},\"p50_us\":{p50:.1},\
             \"qps\":{:.1},\"build_s\":{build_s:.1},\"shards\":{shards},\"ef\":{ef},\
             \"peak_shard_rss_mib\":{peak_rss:.0},\"n\":{points},\"dim\":{dim}}}",
            1e6 / p50
        );
    }

    if let Some(path) = json {
        let meta = format!(
            "{{\"_meta\":true,\"schema\":1,\"harness\":\"sharded_tier\",\"n\":{points},\"dim\":{dim},\
             \"k\":{K},\"queries\":{QUERIES},\"shards\":{shards},\"ef\":{ef},\
             \"partition\":\"kmeans on a sample\",\"cache_mib_per_shard\":4}}\n"
        );
        std::fs::write(&path, meta + &rows).unwrap();
        println!("wrote rows to {}", path.display());
    }
    let _ = Path::new(&root);
}
