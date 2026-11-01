#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Recall@10 and latency for the RP-forest family at n=10k, d=96.
//!
//! Two distributions (uniform on the sphere, and 100 tight clusters) so the
//! numbers are not a single-point sample. Runs under `cargo run --release`,
//! not criterion: this is a comparison harness, not a regression bench.
//!
//! ```text
//! cargo run --release --example rp_forest_search_k --features rptree
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use vicinity::classic::trees::rp_forest::{
    RPTreeParams, RpForestIndex, RpForestParams, RpForestSearchStats,
};

const N: usize = 10_000;
const DIM: usize = 96;
const N_QUERIES: usize = 100;
const K: usize = 10;
const N_TREES: usize = 10;
const MAX_LEAF: usize = 10;

/// Budgets to compare. `K * N_TREES` is Annoy's default and what
/// `RpForestIndex::search` uses.
fn budgets() -> Vec<usize> {
    vec![K * N_TREES, 200, 400, 800, 2_000, N]
}

fn unit(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn gaussian(rng: &mut StdRng) -> f32 {
    // Irwin-Hall: sum of 12 uniforms minus 6 is ~N(0, 1) without an extra dep.
    (0..12).map(|_| rng.random::<f32>()).sum::<f32>() - 6.0
}

fn uniform_set(rng: &mut StdRng, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            unit(&mut v);
            v
        })
        .collect()
}

fn clustered_set(rng: &mut StdRng, n: usize, n_clusters: usize, sigma: f32) -> Vec<Vec<f32>> {
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| uniform_set(rng, 1).remove(0))
        .collect();
    (0..n)
        .map(|i| {
            let center = &centers[i % n_clusters];
            let mut v: Vec<f32> = (0..DIM)
                .map(|d| center[d] + sigma * gaussian(rng))
                .collect();
            unit(&mut v);
            v
        })
        .collect()
}

/// Exact top-k by the same ranking the index uses: `1 - dot`, ties by doc id.
fn brute_force(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
            (i as u32, 1.0 - dot)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

fn recall_at_k(gt: &[u32], got: &[(u32, f32)]) -> f32 {
    let hits = got.iter().filter(|(id, _)| gt.contains(id)).count();
    hits as f32 / gt.len() as f32
}

fn p50_nanos(mut xs: Vec<u128>) -> f64 {
    xs.sort_unstable();
    let mid = xs.len() / 2;
    xs[mid] as f64 / 1000.0
}

/// Repeats per query. The machine is shared, so each query keeps its fastest
/// run: a contention spike can only add time, never remove it.
const REPEATS: usize = 5;

fn run(label: &str, vectors: &[Vec<f32>], queries: &[Vec<f32>]) {
    let params = RpForestParams {
        num_trees: N_TREES,
        tree_params: RPTreeParams {
            max_leaf_size: MAX_LEAF,
        },
    };
    let mut build_ms = f64::INFINITY;
    let mut index = None;
    for _ in 0..3 {
        let build_start = Instant::now();
        let mut candidate = RpForestIndex::new(DIM, params.clone()).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            candidate.add(i as u32, v.clone()).unwrap();
        }
        candidate.build().unwrap();
        build_ms = build_ms.min(build_start.elapsed().as_secs_f64() * 1000.0);
        index = Some(candidate);
    }
    let index = index.expect("built");

    let gt: Vec<Vec<u32>> = queries.iter().map(|q| brute_force(vectors, q, K)).collect();

    // Warm up the branch predictor / page cache, then time every query once.
    for q in queries.iter().take(10) {
        let _ = index.search(q, K).unwrap();
    }

    println!("{label}: build {build_ms:.1} ms");
    println!(
        "  {:>8}  {:>9}  {:>10}  {:>12}  {:>9}  {:>8}",
        "search_k", "recall@10", "p50 (us)", "candidates", "nodes", "leaves"
    );
    for budget in budgets() {
        let mut recall = 0.0f32;
        let mut collected = 0usize;
        let mut nodes = 0usize;
        let mut leaves = 0usize;
        let mut latencies = Vec::with_capacity(queries.len());
        for (q, g) in queries.iter().zip(&gt) {
            let mut best = u128::MAX;
            let mut last = (Vec::new(), RpForestSearchStats::default());
            for _ in 0..REPEATS {
                let start = Instant::now();
                let outcome = index.search_with_budget_stats(q, K, budget).unwrap();
                best = best.min(start.elapsed().as_nanos());
                last = outcome;
            }
            latencies.push(best);
            collected += last.1.candidates;
            nodes += last.1.nodes_expanded;
            leaves += last.1.leaves_visited;
            recall += recall_at_k(g, &last.0);
        }
        recall /= queries.len() as f32;
        let per_query = queries.len() as f64;
        println!(
            "  {:>8}  {:>9.3}  {:>10.2}  {:>12.0}  {:>9.0}  {:>8.1}",
            budget,
            recall,
            p50_nanos(latencies),
            collected as f64 / per_query,
            nodes as f64 / per_query,
            leaves as f64 / per_query
        );
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0xA24B_AED4_963E_E407);
    let uniform = uniform_set(&mut rng, N);
    let uniform_queries = uniform_set(&mut rng, N_QUERIES);
    let clustered = clustered_set(&mut rng, N, 100, 0.35);
    let clustered_queries = clustered_set(&mut rng, N_QUERIES, 100, 0.35);

    println!("n={N} d={DIM} queries={N_QUERIES} k={K} trees={N_TREES} leaf={MAX_LEAF}");
    run("uniform", &uniform, &uniform_queries);
    run(
        "clustered (100 centres, sigma=0.35)",
        &clustered,
        &clustered_queries,
    );
}
