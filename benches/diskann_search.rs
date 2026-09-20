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

// Use an independent f64 scalar scan over the unnormalized L2 fixture.
// Neither this oracle nor the storage-parity checks belongs in the timed loop.
#[cfg(feature = "diskann")]
fn exact_neighbors(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            let mut scored: Vec<_> = vectors
                .iter()
                .enumerate()
                .map(|(id, vector)| {
                    let distance: f64 = query
                        .iter()
                        .zip(vector)
                        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                        .sum();
                    (id as u32, distance)
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            scored.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

#[cfg(feature = "diskann")]
fn checked_ids(results: &[(u32, f32)], k: usize, n_vectors: usize) -> Vec<u32> {
    assert_eq!(results.len(), k);
    assert!(results.iter().all(|(id, distance)| {
        (*id as usize) < n_vectors && distance.is_finite() && *distance >= 0.0
    }));
    assert!(results.windows(2).all(|pair| pair[0].1 <= pair[1].1));
    let mut ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), k, "search must return distinct IDs");
    ids
}

#[cfg(feature = "diskann")]
fn bench_diskann_search_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("diskann_search_only");

    let dim = 64;
    let n_vectors = 5_000;
    let n_queries = 100;
    let k = 10;
    let queries = random_vectors(n_queries, dim, 123);
    let (index, vectors) = build_index(n_vectors, dim, 75);
    let ground_truth = exact_neighbors(&vectors, &queries, k);
    let mut exact_ids = ground_truth[0].clone();
    exact_ids.sort_unstable();
    assert_eq!(
        checked_ids(
            &index.search(&queries[0], k, n_vectors).unwrap(),
            k,
            n_vectors
        ),
        exact_ids,
        "full-exploration control must match the scalar L2 oracle"
    );
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
        let mut hits = 0;
        for (query, truth) in queries.iter().zip(&ground_truth) {
            let expected = index.search(query, k, ef_search).unwrap();
            let expected_ids = checked_ids(&expected, k, n_vectors);
            hits += truth.iter().filter(|id| expected_ids.contains(id)).count();
            for (mode, results) in [
                (
                    "file",
                    file_searcher
                        .borrow_mut()
                        .search(query, k, ef_search)
                        .unwrap(),
                ),
                (
                    "mmap",
                    mmap_searcher
                        .borrow_mut()
                        .search(query, k, ef_search)
                        .unwrap(),
                ),
                (
                    "page_file",
                    page_searcher
                        .borrow_mut()
                        .search(query, k, ef_search)
                        .unwrap(),
                ),
                (
                    "page_mmap",
                    page_mmap_searcher
                        .borrow_mut()
                        .search(query, k, ef_search)
                        .unwrap(),
                ),
            ] {
                assert_eq!(
                    checked_ids(&results, k, n_vectors),
                    expected_ids,
                    "{mode} must preserve the in-memory neighbor set at ef={ef_search}"
                );
            }
        }
        eprintln!(
            "diskann quality: ef={ef_search} recall@{k}={:.4} storage_parity=5 modes cache=warm",
            hits as f64 / (n_queries * k) as f64
        );
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
