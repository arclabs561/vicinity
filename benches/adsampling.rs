#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Benchmarks for ADSampling rotation and search.
//!
//! Measures: rotate_query (O(D^2) per query), dist_comp (partial distance),
//! and search_hnsw end-to-end at multiple dimensionalities.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use vicinity::adsampling::{ADSamplingParams, ADSamplingState};
use vicinity::hnsw::HNSWIndex;

fn random_vectors_flat(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut flat = vec![0.0f32; n * dim];
    for v in flat.iter_mut() {
        *v = rng.random::<f32>() - 0.5;
    }
    // L2-normalize each vector
    for i in 0..n {
        let start = i * dim;
        let end = start + dim;
        let norm: f32 = flat[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut flat[start..end] {
                *x /= norm;
            }
        }
    }
    flat
}

/// Scalar column-strided rotation (the OLD approach for comparison).
/// result[i] = sum_j Q[j*dim+i] * v[j]  -- column access pattern.
fn rotate_scalar_column_strided(v: &[f32], q: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += q[j * dim + i] * v[j];
        }
        result[i] = sum;
    }
    result
}

fn bench_rotate_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("adsampling_rotate_query");

    let n = 1000;
    let params = ADSamplingParams::default();

    for dim in [64, 128, 256, 512, 960] {
        let vectors = random_vectors_flat(n, dim, 42);
        let state = ADSamplingState::new(&vectors, dim, params.clone());

        let mut rng = StdRng::seed_from_u64(99);
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();

        // New: transpose + innr::dot
        group.bench_with_input(BenchmarkId::new("simd/dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                black_box(state.rotate_query(black_box(&query)));
            });
        });

        // Old: scalar column-strided (Q not transposed)
        // Generate the original (non-transposed) Q for fair comparison
        let q_original = {
            let mut q = vec![0.0f32; dim * dim];
            // Transpose Q_T back to Q for the old-style benchmark
            for i in 0..dim {
                for j in 0..dim {
                    // state stores Q_T; Q[i][j] = Q_T[j][i]
                    // We just need *some* D x D matrix; use random for isolation
                    q[i * dim + j] = (i * dim + j) as f32 * 1e-6;
                }
            }
            q
        };
        group.bench_with_input(BenchmarkId::new("scalar/dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                black_box(rotate_scalar_column_strided(
                    black_box(&query),
                    &q_original,
                    dim,
                ));
            });
        });
    }
    group.finish();
}

fn bench_dist_comp(c: &mut Criterion) {
    let mut group = c.benchmark_group("adsampling_dist_comp");

    let n = 1000;
    let params = ADSamplingParams::default();

    for dim in [128, 256, 960] {
        let vectors = random_vectors_flat(n, dim, 42);
        let state = ADSamplingState::new(&vectors, dim, params.clone());

        let mut rng = StdRng::seed_from_u64(99);
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
        let rotated_query = state.rotate_query(&query);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                // Threshold inf = no early exit (measures full distance path)
                black_box(state.dist_comp(black_box(&rotated_query), 0, f32::INFINITY));
            });
        });
    }
    group.finish();
}

fn bench_search_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("adsampling_search_hnsw");
    group.sample_size(30); // search is expensive

    let n = 10_000;
    let params = ADSamplingParams::default();

    for dim in [128, 256] {
        let vectors = random_vectors_flat(n, dim, 42);

        // Build HNSW
        let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Build ADSampling state from reordered vectors
        let state = ADSamplingState::from_hnsw(&index, params.clone());

        let mut rng = StdRng::seed_from_u64(99);
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                black_box(state.search_hnsw(&index, black_box(&query), 10, 64)).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rotate_query,
    bench_dist_comp,
    bench_search_hnsw
);
criterion_main!(benches);
