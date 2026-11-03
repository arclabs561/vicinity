#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Compare `InPlaceIndex` construction against HNSW at equal search budget.
//!
//! The question this answers is not "which family is faster" but "does the
//! in-place builder, which has to keep every node mutable after the fact,
//! produce a graph worth searching". Both sides use the same corpus, the same
//! exact top-10 ground truth and the same search budget, so the only difference
//! is the build.
//!
//! Run:
//!
//! ```text
//! cargo run --release --features hnsw --example inplace_construction -- --json
//! ```
//!
//! The default output is a markdown table; `--json` writes one row per line in
//! the schema `scripts/ratchet.py` and `docs/benchmark-atlas.md` read.

use std::time::Instant;
use vicinity::distance::DistanceMetric;
use vicinity::hnsw::{HNSWIndex, HNSWParams};
use vicinity::hnsw::{InPlaceConfig, InPlaceIndex};

const QUERIES: usize = 30;
const K: usize = 10;
const SEARCH_BEAM: usize = 64;

fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

/// Exact top-10 per query, recomputed once and reused for both families.
fn ground_truth(data: &[Vec<f32>]) -> Vec<Vec<u32>> {
    (0..QUERIES)
        .map(|q| {
            let query = &data[(q * 37) % data.len()];
            let mut scored: Vec<(f32, u32)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, i as u32)
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            scored.iter().take(K).map(|(_, id)| *id).collect()
        })
        .collect()
}

fn median_us(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.total_cmp(b));
    times[times.len() / 2]
}

fn score(returned: &[(u32, f32)], want: &[u32]) -> usize {
    returned.iter().filter(|(id, _)| want.contains(id)).count()
}

fn main() {
    let json = std::env::args().any(|a| a == "--json");
    let tiers = [
        (2_000usize, 16usize),
        (2_000, 96),
        (20_000, 16),
        (20_000, 32),
        (20_000, 96),
    ];

    if !json {
        println!("| tier | family | recall@10 | p50 µs | build s | µs/insert |");
        println!("| --- | --- | ---: | ---: | ---: | ---: |");
    }

    for (n, dim) in tiers {
        let data = corpus(n, dim, 5);
        let truth = ground_truth(&data);

        let cfg = InPlaceConfig::default();
        let started = Instant::now();
        let mut inplace = InPlaceIndex::new(dim, cfg.clone());
        for vector in &data {
            inplace.insert(vector.clone()).unwrap();
        }
        let inplace_build = started.elapsed().as_secs_f64();
        let mut hits = 0usize;
        let mut times = Vec::with_capacity(QUERIES);
        for (q, want) in truth.iter().enumerate() {
            let query = &data[(q * 37) % n];
            let started = Instant::now();
            let got = inplace.search_with_beam(query, K, SEARCH_BEAM).unwrap();
            times.push(started.elapsed().as_secs_f64() * 1e6);
            hits += score(&got, want);
        }
        let inplace_recall = hits as f64 / (QUERIES * K) as f64;
        let inplace_p50 = median_us(times);

        // The second in-place point uses its own default search beam, which is
        // what a caller who never passes an override gets.
        let mut hits_default = 0usize;
        let mut times = Vec::with_capacity(QUERIES);
        for (q, want) in truth.iter().enumerate() {
            let query = &data[(q * 37) % n];
            let started = Instant::now();
            let got = inplace.search(query, K).unwrap();
            times.push(started.elapsed().as_secs_f64() * 1e6);
            hits_default += score(&got, want);
        }
        let inplace_recall_default = hits_default as f64 / (QUERIES * K) as f64;
        let inplace_p50_default = median_us(times);

        let params = HNSWParams {
            m: 16,
            ef_construction: 200,
            metric: DistanceMetric::L2,
            seed: Some(3),
            ..Default::default()
        };
        let started = Instant::now();
        let mut hnsw = HNSWIndex::with_params(dim, params).unwrap();
        for (i, vector) in data.iter().enumerate() {
            hnsw.add(i as u32, vector.clone()).unwrap();
        }
        hnsw.build().unwrap();
        let hnsw_build = started.elapsed().as_secs_f64();
        let mut hits = 0usize;
        let mut times = Vec::with_capacity(QUERIES);
        for (q, want) in truth.iter().enumerate() {
            let query = &data[(q * 37) % n];
            let started = Instant::now();
            let got = hnsw.search(query, K, SEARCH_BEAM).unwrap();
            times.push(started.elapsed().as_secs_f64() * 1e6);
            hits += score(&got, want);
        }
        let hnsw_recall = hits as f64 / (QUERIES * K) as f64;
        let hnsw_p50 = median_us(times);

        for (family, recall, build, beam, p50) in [
            (
                "inplace",
                inplace_recall,
                inplace_build,
                SEARCH_BEAM,
                inplace_p50,
            ),
            (
                "inplace",
                inplace_recall_default,
                inplace_build,
                cfg.beam_width,
                inplace_p50_default,
            ),
            ("hnsw", hnsw_recall, hnsw_build, SEARCH_BEAM, hnsw_p50),
        ] {
            if json {
                println!(
                    "{{\"family\":\"{family}\",\"dataset\":\"synthetic@{n}\",\"knob\":\"dim\",\
                     \"knob_value\":{dim},\"recall\":{recall:.4},\"p50_us\":{p50:.1},\
                     \"qps\":{:.1},\"build_s\":{build:.3},\"us_per_insert\":{:.1},\"n\":{n},\"dim\":{dim},\
                     \"k\":{K},\"queries\":{QUERIES},\"search_beam\":{beam}}}",
                    1e6 / p50,
                    build * 1e6 / n as f64
                );
            } else {
                println!(
                    "| {n} x {dim} | {family} (beam {beam}) | {recall:.3} | {p50:.0} | {build:.2} | {:.1} |",
                    build * 1e6 / n as f64
                );
            }
        }
    }
}
