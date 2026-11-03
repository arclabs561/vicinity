//! Filtered search at equal selectivity but different alignment.
//!
//! Selectivity is not difficulty. With 10% of the corpus matching, the filter is
//! easy when the matches sit next to the query and brutal when they sit away from
//! it: the number of candidates a post-filter must scan to fill `k` changes by an
//! order of magnitude at the same selectivity. arXiv:2606.14193 calls the real
//! parameter hardness; this harness measures its cheapest observable,
//! `overfetch`, the unfiltered candidates consumed per returned hit.
//!
//! Run:
//!
//! ```text
//! cargo run --release --features hnsw,filtered_graph --example filtered_alignment -- \
//!   --points 100k --json benchmarks/filtered-alignment-100k.jsonl
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::time::Instant;
use vicinity::filtered_graph::{AttrValue, Filter, FilteredGraphIndex, FilteredGraphParams};
use vicinity::hnsw::{HNSWIndex, HNSWParams};

const K: usize = 10;
const QUERIES: usize = 100;
/// Partitions, one per sampled query used as a centre. A query evaluates the
/// bucket of its nearest centre ("aligned") or its farthest one ("anti").
const BUCKETS: usize = 20;

fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed | 1;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    // Sixteen cluster centres, sigma 0.25, matching the `clustered` domain.
    let centres: Vec<Vec<f32>> = (0..16)
        .map(|_| (0..dim).map(|_| next()).collect())
        .collect();
    // Unit-normalized, because the index normalizes on insert and normalizes the
    // query: ground truth computed on raw vectors would compare two spaces.
    (0..n)
        .map(|_| {
            let centre = &centres[(next().abs() * 1e6) as usize % centres.len()];
            let mut v: Vec<f32> = centre.iter().map(|c| c + 0.25 * next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

/// Assign every point to its nearest centre.
///
/// A query's aligned bucket is its nearest centre and its anti bucket is its
/// farthest one. Same selectivity, opposite geometry: this is the pair a
/// selectivity sweep cannot distinguish.
fn assign(points: &[Vec<f32>], centres: &[Vec<f32>]) -> Vec<i64> {
    points
        .iter()
        .map(|p| {
            let mut best = (f32::INFINITY, 0i64);
            for (c, centre) in centres.iter().enumerate() {
                let d: f32 = p.iter().zip(centre).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best.0 {
                    best = (d, c as i64);
                }
            }
            best.1
        })
        .collect()
}

/// Nearest and farthest centre for a query, as (distance, bucket) pairs.
fn query_buckets(query: &[f32], centres: &[Vec<f32>]) -> (i64, i64) {
    let mut all: Vec<(f32, i64)> = centres
        .iter()
        .enumerate()
        .map(|(c, centre)| {
            let d: f32 = query
                .iter()
                .zip(centre)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (d, c as i64)
        })
        .collect();
    all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    (all[0].1, all[all.len() - 1].1)
}

fn exact_filtered(points: &[Vec<f32>], buckets: &[i64], bucket: i64, query: &[f32]) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| buckets[*i] == bucket)
        .map(|(i, p)| {
            let d: f32 = p.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (d, i as u32)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    scored.iter().take(K).map(|(_, id)| *id).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut points_n = 100_000usize;
    let mut json: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--points" => {
                i += 1;
                let text = args[i].trim_end_matches(['k', 'K', 'm', 'M']);
                let scale = if args[i].ends_with(['k', 'K']) {
                    1_000
                } else if args[i].ends_with(['m', 'M']) {
                    1_000_000
                } else {
                    1
                };
                points_n = text.parse::<usize>().unwrap() * scale;
            }
            "--json" => {
                i += 1;
                json = Some(args[i].clone());
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }

    let dim = 96;
    let points = corpus(points_n, dim, 0x46494c54);
    let queries = corpus(QUERIES, dim, 0x51554552);
    println!("corpus {points_n} x {dim}, {QUERIES} queries, k={K}, {BUCKETS} buckets");

    let params = FilteredGraphParams {
        max_degree: 16,
        ef_construction: 100,
        ef_search: 100,
        alpha: 1.2,
    };
    let hnsw_params = HNSWParams {
        m: 16,
        ef_construction: 200,
        metric: vicinity::distance::DistanceMetric::L2,
        seed: Some(7),
        ..HNSWParams::default()
    };

    let mut rows = String::new();
    use std::fmt::Write as _;

    let centres: Vec<Vec<f32>> = queries[..BUCKETS].to_vec();
    let buckets = assign(&points, &centres);

    let mut index = FilteredGraphIndex::new(dim, params.clone()).unwrap();
    for (id, p) in points.iter().enumerate() {
        let mut attrs = HashMap::new();
        attrs.insert("bucket".to_string(), AttrValue::Int(buckets[id]));
        index.add_slice(id as u32, p, attrs).unwrap();
    }
    index.build().unwrap();

    let mut graph = HNSWIndex::with_params(dim, hnsw_params.clone()).unwrap();
    for (id, p) in points.iter().enumerate() {
        graph.add(id as u32, p.clone()).unwrap();
    }
    graph.build().unwrap();

    for mode in ["aligned", "anti"] {
        for ef in [40usize, 100, 200, 400] {
            let mut filtered_hits = 0usize;
            let mut post_hits = 0usize;
            let mut overfetch = 0f64;
            let mut filtered_us = Vec::new();
            let mut post_us = Vec::new();
            let mut bucket_total = 0usize;
            let mut exact_us: Vec<f64> = Vec::new();
            let mut used = 0usize;

            for query in queries.iter() {
                let (nearest, farthest) = query_buckets(query, &centres);
                let bucket = if mode == "aligned" { nearest } else { farthest };
                let bucket_size = buckets.iter().filter(|&&b| b == bucket).count();
                // A bucket with fewer than k points cannot express recall: it
                // would report the bucket size, not the search. Skip it.
                if bucket_size < K {
                    continue;
                }
                bucket_total += bucket_size;
                used += 1;
                let scan_start = Instant::now();
                let want = exact_filtered(&points, &buckets, bucket, query);
                exact_us.push(scan_start.elapsed().as_secs_f64() * 1e6);
                let filter = Filter::Clause(vicinity::filtered_graph::Predicate::Eq(
                    "bucket".to_string(),
                    AttrValue::Int(bucket),
                ));

                let started = Instant::now();
                let got = index.search_filtered(query, K, &filter).unwrap();
                filtered_us.push(started.elapsed().as_secs_f64() * 1e6);
                filtered_hits += got.iter().filter(|(id, _)| want.contains(id)).count();

                let started = Instant::now();
                let candidates = graph.search(query, ef, ef).unwrap();
                let kept: Vec<(u32, f32)> = candidates
                    .iter()
                    .copied()
                    .filter(|(id, _)| buckets[*id as usize] == bucket)
                    .take(K)
                    .collect();
                post_us.push(started.elapsed().as_secs_f64() * 1e6);
                post_hits += kept.iter().filter(|(id, _)| want.contains(id)).count();

                let matching_in_beam = candidates
                    .iter()
                    .filter(|(id, _)| buckets[*id as usize] == bucket)
                    .count();
                if matching_in_beam > 0 {
                    overfetch += ef as f64 / matching_in_beam as f64;
                }
            }

            if used == 0 {
                continue;
            }
            let denom = (used * K) as f64;
            let filtered_recall = filtered_hits as f64 / denom;
            let post_recall = post_hits as f64 / denom;
            let mean_overfetch = overfetch / QUERIES as f64;
            let mean_bucket = bucket_total as f64 / used as f64;
            filtered_us.sort_by(|a, b| a.total_cmp(b));
            post_us.sort_by(|a, b| a.total_cmp(b));
            let filtered_p50 = filtered_us[filtered_us.len() / 2];
            exact_us.sort_by(|a, b| a.total_cmp(b));
            let exact_p50 = exact_us[exact_us.len() / 2];
            let post_p50 = post_us[post_us.len() / 2];
            println!(
                "{mode} ef={ef}: filtered_graph {filtered_recall:.3} ({filtered_p50:.0} us) | \
                 post-filter {post_recall:.3} ({post_p50:.0} us) | own scan {exact_p50:.0} us | \
                 overfetch {mean_overfetch:.1}"
            );
            let _ = writeln!(
                rows,
                "{{\"family\":\"filtered_graph\",\"dataset\":\"{mode}@{points_n}\",\"knob\":\"ef\",\
                 \"knob_value\":{ef},\"recall\":{filtered_recall:.4},\"p50_us\":{filtered_p50:.1},\
                 \"overfetch\":{mean_overfetch:.2},\"bucket_size\":{mean_bucket:.0},\
                 \"selectivity\":{:.4}}}",
                mean_bucket / points_n as f64
            );
            let _ = writeln!(
                rows,
                "{{\"family\":\"postfilter\",\"dataset\":\"{mode}@{points_n}\",\"knob\":\"ef\",\
                 \"knob_value\":{ef},\"recall\":{post_recall:.4},\"p50_us\":{post_p50:.1},\
                 \"overfetch\":{mean_overfetch:.2},\"bucket_size\":{mean_bucket:.0},\
                 \"selectivity\":{:.4}}}",
                mean_bucket / points_n as f64
            );
        }
    }

    if let Some(path) = json {
        let meta = format!(
            "{{\"_meta\":true,\"schema\":1,\"harness\":\"filtered_alignment\",\"n\":{points_n},\
             \"dim\":{dim},\"k\":{K},\"queries\":{QUERIES},\"buckets\":{BUCKETS},\
             \"clustered\":true,\"generator\":\"clustered16 sigma 0.25\"}}\n"
        );
        std::fs::write(&path, meta + &rows).unwrap();
        println!("wrote rows to {path}");
    }
}
