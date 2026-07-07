#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Search-only DiskANN benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

#[cfg(feature = "diskann")]
use rand::prelude::*;
#[cfg(feature = "diskann")]
use std::cell::RefCell;
#[cfg(feature = "diskann")]
use vicinity::diskann::{DiskANNIndex, DiskANNPageSearcher, DiskANNParams, DiskANNSearcher};

#[cfg(feature = "diskann")]
fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
        .collect()
}

#[cfg(feature = "diskann")]
fn build_index(n_vectors: usize, dim: usize, ef_search: usize) -> (DiskANNIndex, Vec<Vec<f32>>) {
    let vectors = random_vectors(n_vectors, dim, 42);
    let params = DiskANNParams {
        m: 32,
        ef_construction: 80,
        alpha: 1.2,
        ef_search,
        seed: Some(42),
    };
    let mut index = DiskANNIndex::new(dim, params).unwrap();
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).unwrap();
    }
    index.build().unwrap();
    (index, vectors)
}

#[cfg(feature = "diskann")]
fn bench_diskann_search_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("diskann_search_only");

    let dim = 64;
    let n_vectors = 5_000;
    let n_queries = 100;
    let k = 10;
    let queries = random_vectors(n_queries, dim, 123);
    let (index, _vectors) = build_index(n_vectors, dim, 75);
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let index_dir = temp_dir.path().join("diskann");
    index.save(&index_dir).expect("save DiskANN index");
    index
        .save_page_layout(&index_dir)
        .expect("save DiskANN page layout");
    let file_searcher = RefCell::new(DiskANNSearcher::load(&index_dir).unwrap());
    let mmap_searcher = RefCell::new(DiskANNSearcher::load_mmap(&index_dir).unwrap());
    let page_searcher = RefCell::new(DiskANNPageSearcher::load(&index_dir).unwrap());
    let page_mmap_searcher = RefCell::new(DiskANNPageSearcher::load_mmap(&index_dir).unwrap());

    group.throughput(Throughput::Elements(n_queries as u64));
    for ef_search in [50, 75, 250] {
        group.bench_function(format!("memory_ef{ef_search}"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|query| index.search(black_box(query), k, ef_search).unwrap().len())
                    .sum::<usize>()
            });
        });
        group.bench_function(format!("file_ef{ef_search}"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|query| {
                        file_searcher
                            .borrow_mut()
                            .search(black_box(query), k, ef_search)
                            .unwrap()
                            .len()
                    })
                    .sum::<usize>()
            });
        });
        group.bench_function(format!("mmap_ef{ef_search}"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|query| {
                        mmap_searcher
                            .borrow_mut()
                            .search(black_box(query), k, ef_search)
                            .unwrap()
                            .len()
                    })
                    .sum::<usize>()
            })
        });
        group.bench_function(format!("page_file_ef{ef_search}"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|query| {
                        page_searcher
                            .borrow_mut()
                            .search(black_box(query), k, ef_search)
                            .unwrap()
                            .len()
                    })
                    .sum::<usize>()
            })
        });
        group.bench_function(format!("page_mmap_ef{ef_search}"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|query| {
                        page_mmap_searcher
                            .borrow_mut()
                            .search(black_box(query), k, ef_search)
                            .unwrap()
                            .len()
                    })
                    .sum::<usize>()
            })
        });
    }

    group.finish();
}

#[cfg(not(feature = "diskann"))]
fn bench_diskann_search_only(_c: &mut Criterion) {}

criterion_group!(benches, bench_diskann_search_only);
criterion_main!(benches);
