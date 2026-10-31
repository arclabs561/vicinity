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

fn truth(queries: &[Vec<f32>], db: &[Vec<f32>], k: usize) -> Vec<HashSet<u32>> {
    queries
        .iter()
        .map(|q| {
            let mut scored: Vec<(u32, f32)> = db
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u32, cosine_distance(q, v)))
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1));
            scored.into_iter().take(k).map(|(i, _)| i).collect()
        })
        .collect()
}

fn score(name: &str, truth: &[HashSet<u32>], queries: &[Vec<f32>], search: impl Fn(&[f32]) -> Vec<u32>) {
    let k = truth[0].len();
    let started = Instant::now();
    let mut hit = 0usize;
    for (q, expected) in queries.iter().zip(truth) {
        let got: HashSet<u32> = search(q).into_iter().take(k).collect();
        hit += expected.intersection(&got).count();
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{name:34} recall@{k}={:.3} qps={:.0}",
        hit as f32 / (queries.len() * k) as f32,
        queries.len() as f64 / elapsed
    );
}

fn tier(name: &str) -> usize {
    match name {
        "10k" => 10_000,
        "100k" => 100_000,
        "1m" => 1_000_000,
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
        for dim in [96usize, 128] {
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
                        |q| uindex.search(q, 10).unwrap().keys.iter().map(|k| *k as u32).collect(),
                    );
                }
            }
        }
    }
}
