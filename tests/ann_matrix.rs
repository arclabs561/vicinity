//! ANN comparison matrix: several sizes, dimensions and distributions.
//!
//! Run with `--features hnsw` (usearch is a dev-dependency of the crate, so use
//! a test-style invocation or add it to your own harness). Sizes are chosen by
//! `ANN_MATRIX_TIERS` (comma separated: 10k,100k,1m) so CI can run the cheap
//! tiers and a workstation can run the rest.

use std::collections::HashSet;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vicinity::distance::{cosine_distance, normalize};

#[derive(Clone, Copy)]
enum Shape {
    Uniform,
    Clustered,
}

struct Case {
    n: usize,
    dim: usize,
    shape: Shape,
}

fn dataset(case: &Case, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = match case.shape {
        Shape::Uniform => Vec::new(),
        Shape::Clustered => (0..16)
            .map(|_| normalize(&(0..case.dim).map(|_| rng.random::<f32>() - 0.5).collect::<Vec<f32>>()))
            .collect(),
    };
    (0..case.n)
        .map(|_| {
            let mut v: Vec<f32> = (0..case.dim).map(|_| rng.random::<f32>() - 0.5).collect();
            if let Shape::Clustered = case.shape {
                let centre = &centres[rng.random_range(0..centres.len())];
                for (value, offset) in v.iter_mut().zip(centre.iter()) {
                    *value += offset * 2.0;
                }
            }
            normalize(&v)
        })
        .collect()
}

fn truth(queries: &[Vec<f32>], db: &[Vec<f32>], k: usize) -> Vec<Vec<(u32, f32)>> {
    queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(u32, f32)> = db
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u32, cosine_distance(q, v)))
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            scored.into_iter().take(k).collect()
        })
        .collect()
}

fn score(
    name: &str,
    truth: &[Vec<(u32, f32)>],
    queries: &[Vec<f32>],
    db: &[Vec<f32>],
    search: impl Fn(&[f32]) -> Vec<u32>,
) {
    let k = truth[0].len();
    let mut latencies = Vec::with_capacity(queries.len());
    let mut hit = 0usize;
    let mut ratio = 0.0f64;
    for (q, expected) in queries.iter().zip(truth) {
        let started = Instant::now();
        let got = search(q);
        latencies.push(started.elapsed().as_secs_f64());
        let expected_ids: HashSet<u32> = expected.iter().map(|(i, _)| *i).collect();
        let returned: HashSet<u32> = got.iter().take(k).copied().collect();
        hit += expected_ids.intersection(&returned).count();
        // 1/Ratio@k: true distance mass over retrieved distance mass, averaged
        // over queries. Judge-free and computable from the same ground truth.
        let true_mass: f32 = expected.iter().map(|(_, d)| d).sum();
        let got_mass: f32 = got
            .iter()
            .take(k)
            .map(|id| cosine_distance(q, &db[*id as usize]))
            .sum();
        if got_mass > 0.0 {
            ratio += f64::from(true_mass / got_mass);
        }
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| latencies[((latencies.len() as f64 * p) as usize).min(latencies.len() - 1)];
    println!(
        "{name:34} recall@{k}={:.3} 1/ratio={:.3} p50={:.0}us p95={:.0}us p99={:.0}us qps={:.0}",
        hit as f32 / (queries.len() * k) as f32,
        ratio / queries.len() as f64,
        pct(0.50) * 1e6,
        pct(0.95) * 1e6,
        pct(0.99) * 1e6,
        queries.len() as f64 / latencies.iter().sum::<f64>(),
    );
}

fn tier(name: &str) -> usize {
    match name {
        "10k" => 10_000,
        "100k" => 100_000,
        "1m" => 1_000_000,
        "10m" => 10_000_000,
        _ => 0,
    }
}

#[test]
#[ignore = "measurement; run with --ignored --features hnsw"]
fn ann_matrix() {
    let tiers: Vec<usize> = std::env::var("ANN_MATRIX_TIERS")
        .unwrap_or_else(|_| "10k".into())
        .split(',')
        .filter_map(|t| {
            let n = tier(t.trim());
            (n > 0).then_some(n)
        })
        .collect();
    for n in tiers {
        // Dimension scales down as n grows: 10M x 96 f32 is ~3.8 GB before the
        // index, and mega tiers exist to test scale behaviour, not width.
        let dims: Vec<usize> = if n >= 10_000_000 {
            vec![32, 96]
        } else {
            vec![96, 128]
        };
        for dim in dims {
            for shape in [Shape::Uniform, Shape::Clustered] {
                let case = Case { n, dim, shape };
                let db = dataset(&case, 11);
                let queries = dataset(&Case { n: 200, dim, shape }, 12);
                let expected = truth(&queries, &db, 10);
                let label = format!(
                    "n={n} d={dim} {}",
                    match shape {
                        Shape::Uniform => "uniform",
                        Shape::Clustered => "clustered",
                    }
                );
                use vicinity::hnsw::HNSWIndex;
                let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
                for (i, v) in db.iter().enumerate() {
                    index.add(i as u32, v.clone()).unwrap();
                }
                index.build().unwrap();
                for ef in [64usize, 128, 256] {
                    score(
                        &format!("vicinity {label} ef={ef}"),
                        &expected,
                        &queries,
                        &db,
                        |q| index.search(q, 10, ef).unwrap().into_iter().map(|(i, _)| i).collect(),
                    );
                }
                let options = usearch::IndexOptions {
                    dimensions: dim,
                    metric: usearch::MetricKind::Cos,
                    connectivity: 16,
                    expansion_add: 128,
                    expansion_search: 128,
                    ..Default::default()
                };
                let uindex = usearch::Index::new(&options).unwrap();
                uindex.reserve(n).unwrap();
                for (i, v) in db.iter().enumerate() {
                    uindex.add(i as u64, v).unwrap();
                }
                for ef in [64usize, 128, 256] {
                    uindex.change_expansion_search(ef);
                    score(
                        &format!("usearch  {label} ef={ef}"),
                        &expected,
                        &queries,
                        &db,
                        |q| uindex.search(q, 10).unwrap().keys.iter().map(|k| *k as u32).collect(),
                    );
                }
            }
        }
    }
}
