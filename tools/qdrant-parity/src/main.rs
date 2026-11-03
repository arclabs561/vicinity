//! In-process HNSW measured on the same corpus a Qdrant server was measured on.
//!
//! `scripts/qdrant_parity.py` generates a corpus and queries, loads them into a
//! local Qdrant container, and times HTTP round trips. This binary reads the
//! same two `.f32` files, so both sides see byte-identical vectors rather than
//! two generators that are meant to agree.
//!
//! Usage: `cargo run --release -- <corpus.f32> <queries.f32> [rows.jsonl]`
//!
//! Rows use the harness schema plus `category = "embedded"`, which is what
//! keeps them from being diffed against the server rows.

use std::time::Instant;

use vicinity::distance::DistanceMetric;
use vicinity::hnsw::{HNSWIndex, HNSWParams};

const K: usize = 10;
const EF: [usize; 6] = [10, 20, 50, 100, 200, 400];

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert!(bytes.len() % 4 == 0, "{path} is not a float array");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (corpus_path, queries_path) = match args.as_slice() {
        [c, q, ..] => (c.as_str(), q.as_str()),
        _ => {
            eprintln!("usage: qdrant-parity <corpus.f32> <queries.f32> [rows.jsonl]");
            std::process::exit(2);
        }
    };
    let corpus = read_f32(corpus_path);
    let queries = read_f32(queries_path);

    // The dimension is not recoverable from two flat float files, so the
    // generator passes it explicitly through PARITY_DIM.
    let dim: usize = std::env::var("PARITY_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(96);
    assert_eq!(
        corpus.len() % dim,
        0,
        "corpus is not a whole number of vectors"
    );
    let n = corpus.len() / dim;
    let nq = queries.len() / dim;
    eprintln!("corpus {n} x {dim}, {nq} queries");

    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        ef_search: 50,
        metric: DistanceMetric::Cosine,
        seed: Some(0x4641_4D49_4C59),
        ..HNSWParams::default()
    };

    let build_start = Instant::now();
    let mut index = HNSWIndex::with_params(dim, params).expect("index");
    for i in 0..n {
        index
            .add_slice(i as u32, &corpus[i * dim..(i + 1) * dim])
            .expect("add");
    }
    index.build().expect("build");
    let build_s = build_start.elapsed().as_secs_f64();

    // Exact top-k ground truth on the same bytes, cosine on unit vectors is the
    // dot product, identical to what the server computes.
    let mut truth: Vec<Vec<u32>> = Vec::with_capacity(nq);
    for q in 0..nq {
        let query = &queries[q * dim..(q + 1) * dim];
        let mut scored: Vec<(f32, u32)> = (0..n)
            .map(|i| {
                let dot: f32 = query
                    .iter()
                    .zip(&corpus[i * dim..(i + 1) * dim])
                    .map(|(a, b)| a * b)
                    .sum();
                (dot, i as u32)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        truth.push(scored.into_iter().take(K).map(|(_, i)| i).collect());
    }

    let mut out = String::new();
    for ef in EF {
        let mut lat: Vec<u128> = Vec::with_capacity(nq);
        let mut hits = 0usize;
        for q in 0..nq {
            let query = &queries[q * dim..(q + 1) * dim];
            let t = Instant::now();
            let got = index.search(query, K, ef).expect("search");
            lat.push(t.elapsed().as_micros());
            let ids: Vec<u32> = got.iter().map(|(id, _)| *id).collect();
            hits += ids.iter().filter(|id| truth[q].contains(id)).count();
        }
        lat.sort_unstable();
        let pick = |p: f64| lat[((lat.len() as f64 - 1.0) * p) as usize];
        let total_us: u128 = lat.iter().sum();
        let row = format!(
            "{{\"family\":\"vicinity_hnsw\",\"dataset\":\"uniform@{n}\",\"knob\":\"ef_search\",\
             \"knob_value\":{ef},\"recall\":{:.4},\"p50_us\":{:.1},\"p95_us\":{:.1},\
             \"p99_us\":{:.1},\"qps\":{:.1},\"build_s\":{build_s:.2},\"dim\":{dim},\
             \"category\":\"embedded\"}}\n",
            hits as f64 / (nq * K) as f64,
            pick(0.50) as f64,
            pick(0.95) as f64,
            pick(0.99) as f64,
            (nq as f64) / (total_us as f64 / 1e6),
        );
        eprint!("{row}");
        out.push_str(&row);
    }

    if let Some(path) = args.get(2) {
        std::fs::write(path, out).expect("write rows");
    }
}
