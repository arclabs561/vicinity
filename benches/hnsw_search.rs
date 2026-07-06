#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Search-only HNSW benchmarks.
//!
//! Keep this target separate from `benches/hnsw.rs` so profiling filtered
//! search runs does not include setup for construction benchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use vicinity::hnsw::HNSWIndex;

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            vicinity::distance::normalize(&v)
        })
        .collect()
}

fn build_index(n_vectors: usize, dim: usize) -> (HNSWIndex, Vec<Vec<f32>>) {
    let vectors = random_vectors(n_vectors, dim, 42);
    let mut index = HNSWIndex::new(dim, 16, 16).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        index.add_slice(i as u32, v).unwrap();
    }
    index.build().unwrap();
    (index, vectors)
}

fn bench_hnsw_search_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_only");

    let dim = 128;
    let n_vectors = 10_000;
    let n_queries = 100;
    let queries = random_vectors(n_queries, dim, 123);
    let (index, _vectors) = build_index(n_vectors, dim);

    for ef in [10, 50, 100, 200] {
        group.throughput(Throughput::Elements(n_queries as u64));
        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |bench, &ef| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| index.search(black_box(q), 10, ef).unwrap().len())
                    .sum::<usize>()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hnsw_search_only);
criterion_main!(benches);
