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
use vicinity::{FileCacheConfig, DEFAULT_FILE_CACHE_BLOCK_SIZE, DEFAULT_FILE_CACHE_BYTES};

/// Cache config for a budget, with the block size overridable through
/// `VICINITY_BENCH_CACHE_BLOCK` for block-size comparisons.
#[cfg(feature = "diskann")]
fn cache_config(budget_bytes: usize) -> FileCacheConfig {
    let mut config = FileCacheConfig::with_budget(budget_bytes);
    config.block_size = std::env::var("VICINITY_BENCH_CACHE_BLOCK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_FILE_CACHE_BLOCK_SIZE);
    config
}

#[cfg(feature = "diskann")]
fn file_len(path: &std::path::Path) -> usize {
    std::fs::metadata(path).unwrap().len() as usize
}

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
    let standard_bytes =
        file_len(&index_dir.join("graph.index")) + file_len(&index_dir.join("vectors.bin"));
    let page_bytes = file_len(&index_dir.join("nodes.page"));
    let load = |budget| DiskANNSearcher::load_with_cache(&index_dir, cache_config(budget));
    let load_page = |budget| DiskANNPageSearcher::load_with_cache(&index_dir, cache_config(budget));
    let file_searcher = RefCell::new(load(DEFAULT_FILE_CACHE_BYTES).unwrap());
    let page_searcher = RefCell::new(load_page(DEFAULT_FILE_CACHE_BYTES).unwrap());
    // Uncached rows show the raw positional-read cost; quarter-budget rows show
    // a cache smaller than the index.
    let nocache_searcher = RefCell::new(load(0).unwrap());
    let page_nocache_searcher = RefCell::new(load_page(0).unwrap());
    let quarter_searcher = RefCell::new(load(standard_bytes / 4).unwrap());
    let page_quarter_searcher = RefCell::new(load_page(page_bytes / 4).unwrap());

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
                    "page_file",
                    page_searcher
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
            "diskann quality: ef={ef_search} recall@{k}={:.4} storage_parity=3 modes cache=warm",
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
        if ef_search == 50 {
            for (name, searcher) in [
                ("file_nocache", &nocache_searcher),
                ("file_cache25", &quarter_searcher),
            ] {
                group.bench_function(format!("{name}_ef{ef_search}"), |bench| {
                    bench.iter(|| {
                        queries
                            .iter()
                            .map(|query| {
                                searcher
                                    .borrow_mut()
                                    .search(black_box(query), k, ef_search)
                                    .unwrap()
                                    .len()
                            })
                            .sum::<usize>()
                    })
                });
            }
            for (name, searcher) in [
                ("page_file_nocache", &page_nocache_searcher),
                ("page_file_cache25", &page_quarter_searcher),
            ] {
                group.bench_function(format!("{name}_ef{ef_search}"), |bench| {
                    bench.iter(|| {
                        queries
                            .iter()
                            .map(|query| {
                                searcher
                                    .borrow_mut()
                                    .search(black_box(query), k, ef_search)
                                    .unwrap()
                                    .len()
                            })
                            .sum::<usize>()
                    })
                });
            }
        }
    }

    for (name, stats) in [
        ("file", file_searcher.borrow().cache_stats()),
        ("file_cache25", quarter_searcher.borrow().cache_stats()),
        ("page_file", page_searcher.borrow().cache_stats()),
        (
            "page_file_cache25",
            page_quarter_searcher.borrow().cache_stats(),
        ),
    ] {
        eprintln!(
            "diskann cache: {name} budget={} resident={} hits={} misses={} hit_rate={:.4} block={}",
            stats.budget_bytes,
            stats.resident_bytes,
            stats.hits,
            stats.misses,
            stats.hit_rate(),
            cache_config(0).block_size,
        );
    }
    eprintln!("diskann index bytes: standard={standard_bytes} page={page_bytes}");

    group.finish();
}

#[cfg(not(feature = "diskann"))]
fn bench_diskann_search_only(_c: &mut Criterion) {}

criterion_group!(benches, bench_diskann_search_only);
criterion_main!(benches);
