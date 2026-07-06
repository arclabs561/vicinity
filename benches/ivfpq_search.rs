#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Search-only IVF-PQ benchmarks.
//!
//! Keep this target separate from the ANN dataset example so profiling search
//! does not include HDF5 loading or k-means training.

#[cfg(feature = "ivf_pq")]
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
#[cfg(feature = "ivf_pq")]
use rand::prelude::*;
#[cfg(feature = "ivf_pq")]
use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

#[cfg(feature = "ivf_pq")]
fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            vicinity::distance::normalize(&v)
        })
        .collect()
}

#[cfg(feature = "ivf_pq")]
fn build_index(n_vectors: usize, dim: usize) -> (IVFPQIndex, Vec<Vec<f32>>) {
    let vectors = random_vectors(n_vectors, dim, 42);
    let params = IVFPQParams {
        num_clusters: 256,
        nprobe: 32,
        num_codebooks: dim,
        codebook_size: 256,
        seed: 42,
        ..IVFPQParams::default()
    };
    let mut index = IVFPQIndex::new(dim, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        index.add(i as u32, v.clone()).unwrap();
    }
    index.build_with_training_options(Some(5_000), 5).unwrap();
    (index, vectors)
}

#[cfg(feature = "ivf_pq")]
fn bench_ivfpq_search_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivfpq_search_only");

    let dim = 25;
    let n_vectors = 20_000;
    let n_queries = 100;
    let queries = random_vectors(n_queries, dim, 123);
    let (index, _vectors) = build_index(n_vectors, dim);

    group.throughput(Throughput::Elements(n_queries as u64));
    group.bench_function("nprobe32_k10", |bench| {
        bench.iter(|| {
            queries
                .iter()
                .map(|q| index.search(black_box(q), 10).unwrap().len())
                .sum::<usize>()
        });
    });

    group.bench_function("nprobe32_rerank500_k10", |bench| {
        bench.iter(|| {
            queries
                .iter()
                .map(|q| index.search_reranked(black_box(q), 10, 500).unwrap().len())
                .sum::<usize>()
        });
    });

    group.finish();
}

#[cfg(feature = "ivf_pq")]
criterion_group!(benches, bench_ivfpq_search_only);
#[cfg(feature = "ivf_pq")]
criterion_main!(benches);

#[cfg(not(feature = "ivf_pq"))]
fn main() {
    eprintln!("ivfpq_search bench requires --features ivf_pq");
}
