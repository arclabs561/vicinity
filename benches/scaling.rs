#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Scaling benchmarks: how performance changes with dataset size and dimensions.
//!
//! Key questions:
//! - How does construction time scale with n?
//! - How does query time scale with n?
//! - How does query time scale with dimension?
//! - At what n does brute force become slower than ANN?

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vicinity::hnsw::HNSWIndex;

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Brute force k-NN
fn brute_force_knn(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<(u32, f32)> {
    let mut distances: Vec<(u32, f32)> = database
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, l2_distance_squared(query, v)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances
}

fn create_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            vicinity::distance::normalize(&v)
        })
        .collect()
}

fn build_index(database: &[Vec<f32>], dim: usize, m: usize) -> HNSWIndex {
    let mut index = HNSWIndex::new(dim, m, m).unwrap();
    for (i, vec) in database.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    index
}

/// Benchmark query time scaling with dataset size.
fn bench_query_scaling_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scaling_n");
    group.sample_size(20);

    let dimension = 128;
    let k = 10;
    let ef = 64;
    let m = 16;

    for n in [1_000usize, 2_000, 5_000, 10_000, 20_000] {
        let database = create_dataset(n, dimension, 42);
        let queries = create_dataset(10, dimension, 123);

        let index = build_index(&database, dimension, m);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.search(query, k, ef).unwrap());
                }
            })
        });

        // Only benchmark brute force for smaller sizes
        if n <= 10_000 {
            group.bench_with_input(BenchmarkId::new("brute", n), &n, |b, _| {
                b.iter(|| {
                    for query in &queries {
                        black_box(brute_force_knn(query, &database, k));
                    }
                })
            });
        }
    }

    group.finish();
}

/// Benchmark query time scaling with dimension.
fn bench_query_scaling_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scaling_dim");
    group.sample_size(20);

    let n = 10_000;
    let k = 10;
    let ef = 64;
    let m = 16;

    for dim in [32, 64, 128, 256, 512] {
        let database = create_dataset(n, dim, 42);
        let queries = create_dataset(10, dim, 123);

        let index = build_index(&database, dim, m);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("hnsw", dim), &dim, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.search(query, k, ef).unwrap());
                }
            })
        });
    }

    group.finish();
}

/// Benchmark construction time scaling.
fn bench_construction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_scaling");
    group.sample_size(10);

    let dimension = 128;
    let m = 16;

    for n in [500usize, 1_000, 2_000, 5_000] {
        let database = create_dataset(n, dimension, 42);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |b, _| {
            b.iter(|| {
                let mut index = HNSWIndex::new(dimension, m, m).unwrap();
                for (i, vec) in database.iter().enumerate() {
                    index.add_slice(i as u32, vec).unwrap();
                }
                index.build().unwrap();
                index
            })
        });
    }

    group.finish();
}

/// Find crossover point where HNSW beats brute force.
fn bench_crossover_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover_point");
    group.sample_size(20);

    let dimension = 128;
    let k = 10;
    let ef = 64;
    let m = 16;

    for n in [100usize, 200, 500, 1_000, 2_000, 5_000] {
        let database = create_dataset(n, dimension, 42);
        let queries = create_dataset(10, dimension, 123);

        let index = build_index(&database, dimension, m);

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.search(query, k, ef).unwrap());
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("brute", n), &n, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(brute_force_knn(query, &database, k));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_query_scaling_n,
    bench_query_scaling_dim,
    bench_construction_scaling,
    bench_crossover_point,
);
criterion_main!(benches);
