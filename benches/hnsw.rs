#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Benchmarks for HNSW index construction and search.
//!
//! These benchmarks measure end-to-end performance on synthetic data
//! using the real `HNSWIndex` implementation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use vicinity::hnsw::HNSWIndex;

// === Synthetic Data Generation ===

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
        .collect()
}

// === Benchmarks ===

fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_construction");

    let dim = 128;

    for n in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*n as u64));

        let vectors = random_vectors(*n, dim, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |bench, _| {
            bench.iter(|| {
                let mut index = HNSWIndex::new(dim, 16, 16).unwrap();
                for (i, v) in vectors.iter().enumerate() {
                    index.add_slice(i as u32, black_box(v)).unwrap();
                }
                index.build().unwrap();
                index
            });
        });
    }

    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    let dim = 128;
    let n_vectors = 10000;
    let n_queries = 100;

    // Build index once
    let vectors = random_vectors(n_vectors, dim, 42);
    let queries = random_vectors(n_queries, dim, 123);

    let mut index = HNSWIndex::new(dim, 16, 16).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        index.add_slice(i as u32, v).unwrap();
    }
    index.build().unwrap();

    for ef in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(n_queries as u64));

        group.bench_with_input(BenchmarkId::new("ef", ef), ef, |bench, &ef| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| index.search(black_box(q), 10, ef).unwrap())
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

fn bench_hnsw_search_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_k");

    let dim = 128;
    let n_vectors = 10000;
    let n_queries = 100;

    // Build index once
    let vectors = random_vectors(n_vectors, dim, 42);
    let queries = random_vectors(n_queries, dim, 123);

    let mut index = HNSWIndex::new(dim, 16, 16).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        index.add_slice(i as u32, v).unwrap();
    }
    index.build().unwrap();

    for k in [1, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(n_queries as u64));

        group.bench_with_input(BenchmarkId::new("k", k), k, |bench, &k| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| index.search(black_box(q), k, 100).unwrap())
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_construction,
    bench_hnsw_search,
    bench_hnsw_search_k,
);
criterion_main!(benches);
