#![allow(clippy::expect_used, clippy::unwrap_used, unsafe_code)]
//! Search-only HNSW benchmarks.
//!
//! Keep this target separate from `benches/hnsw.rs` so profiling filtered
//! search runs does not include setup for construction benchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "benchmark")]
use vicinity::hnsw::{reset_search_counters, take_search_counters, HnswSearchCounters};
use vicinity::hnsw::{HNSWIndex, HNSWParams};

static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

// SAFETY: This wrapper delegates allocation behavior to `System` unchanged and
// only records relaxed diagnostic counters for this benchmark binary.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` are forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        // SAFETY: `ptr`, `layout`, and `new_size` are forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[derive(Clone, Copy, Debug, Default)]
struct AllocationProfile {
    calls: usize,
    bytes: usize,
}

impl AllocationProfile {
    fn record_search<T>(f: impl FnOnce() -> T) -> (T, Self) {
        ALLOC_CALLS.store(0, Ordering::Relaxed);
        ALLOC_BYTES.store(0, Ordering::Relaxed);
        let out = f();
        let calls = ALLOC_CALLS.load(Ordering::Relaxed);
        let bytes = ALLOC_BYTES.load(Ordering::Relaxed);
        (out, Self { calls, bytes })
    }

    fn add_assign(&mut self, other: Self) {
        self.calls += other.calls;
        self.bytes += other.bytes;
    }
}

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            vicinity::distance::normalize(&v)
        })
        .collect()
}

fn build_index_with_params(
    n_vectors: usize,
    dim: usize,
    m: usize,
    m_max: usize,
    seed: u64,
) -> (HNSWIndex, Vec<Vec<f32>>) {
    let vectors = random_vectors(n_vectors, dim, 42);
    let mut index = HNSWIndex::with_params(
        dim,
        HNSWParams {
            m,
            m_max,
            seed: Some(seed),
            ..Default::default()
        },
    )
    .unwrap();
    for (i, v) in vectors.iter().enumerate() {
        index.add_slice(i as u32, v).unwrap();
    }
    index.build().unwrap();
    (index, vectors)
}

fn graph_seed() -> u64 {
    match std::env::var("VICINITY_BENCH_SEED") {
        Ok(value) => value.parse().expect("VICINITY_BENCH_SEED must be a u64"),
        Err(std::env::VarError::NotPresent) => 42,
        Err(error) => panic!("invalid VICINITY_BENCH_SEED: {error}"),
    }
}

// The fixture contains unit vectors and uses cosine distance. Keep the exact
// scan outside both the allocation diagnostic and Criterion's timed closure.
fn exact_neighbors(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            let mut scored: Vec<_> = vectors
                .iter()
                .enumerate()
                .map(|(id, vector)| {
                    (
                        id as u32,
                        vicinity::distance::cosine_distance_normalized(query, vector),
                    )
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            scored.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

fn print_search_summary(
    label: &str,
    index: &HNSWIndex,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    k: usize,
    ef: usize,
    seed: u64,
) {
    let mut alloc_total = AllocationProfile::default();
    #[cfg(feature = "benchmark")]
    let mut search_total = HnswSearchCounters::default();
    let mut result_count = 0usize;
    let mut hits = 0usize;
    assert_eq!(queries.len(), ground_truth.len());

    for (query, truth) in queries.iter().zip(ground_truth) {
        #[cfg(feature = "benchmark")]
        reset_search_counters();
        let (results, alloc_profile) =
            AllocationProfile::record_search(|| index.search(query, k, ef).unwrap());
        #[cfg(feature = "benchmark")]
        {
            let counters = take_search_counters();
            search_total.distance_evals += counters.distance_evals;
            search_total.result_insertions += counters.result_insertions;
            search_total.result_replacements += counters.result_replacements;
            search_total.result_rejections += counters.result_rejections;
            search_total.candidate_pushes += counters.candidate_pushes;
            search_total.candidate_pops += counters.candidate_pops;
            search_total.frontier_retain_calls += counters.frontier_retain_calls;
            search_total.frontier_pruned_candidates += counters.frontier_pruned_candidates;
            search_total.max_frontier_len =
                search_total.max_frontier_len.max(counters.max_frontier_len);
        }
        alloc_total.add_assign(alloc_profile);
        assert_eq!(results.len(), k);
        assert!(results.iter().all(|(_, distance)| distance.is_finite()));
        assert!(results.windows(2).all(|pair| pair[0].1 <= pair[1].1));
        assert!(results
            .iter()
            .enumerate()
            .all(|(i, (id, _))| { results[..i].iter().all(|(previous, _)| previous != id) }));
        // Count each true neighbor at most once, even if search returns duplicates.
        hits += truth
            .iter()
            .filter(|id| results.iter().any(|(found, _)| found == *id))
            .count();
        result_count += results.len();
    }

    let queries_len = queries.len().max(1) as f64;
    eprintln!(
        "hnsw quality {label}: graph_seed={seed} ef={ef} recall@{k}={:.4}",
        hits as f64 / (queries.len() * k) as f64,
    );
    eprintln!(
        "hnsw alloc {label}: ef={ef} alloc_calls={:.1}/query alloc_bytes={:.1}/query results={result_count}",
        alloc_total.calls as f64 / queries_len,
        alloc_total.bytes as f64 / queries_len,
    );
    #[cfg(feature = "benchmark")]
    eprintln!(
        "hnsw frontier {label}: ef={ef} distance_evals={:.1}/query result_insertions={:.1}/query result_replacements={:.1}/query result_rejections={:.1}/query candidate_pushes={:.1}/query candidate_pops={:.1}/query retain_calls={:.1}/query pruned={:.1}/query max_frontier={}",
        search_total.distance_evals as f64 / queries_len,
        search_total.result_insertions as f64 / queries_len,
        search_total.result_replacements as f64 / queries_len,
        search_total.result_rejections as f64 / queries_len,
        search_total.candidate_pushes as f64 / queries_len,
        search_total.candidate_pops as f64 / queries_len,
        search_total.frontier_retain_calls as f64 / queries_len,
        search_total.frontier_pruned_candidates as f64 / queries_len,
        search_total.max_frontier_len,
    );
    black_box(result_count);
}

fn bench_hnsw_search_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_only");

    let dim = 128;
    let n_vectors = 10_000;
    let n_queries = 100;
    let queries = random_vectors(n_queries, dim, 123);
    let seed = graph_seed();
    let (index, vectors) = build_index_with_params(n_vectors, dim, 16, 16, seed);
    let ground_truth = exact_neighbors(&vectors, &queries, 10);
    assert_eq!(
        index
            .search(&queries[0], 10, n_vectors)
            .unwrap()
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        ground_truth[0],
        "full-exploration control must match exact neighbors"
    );

    for ef in [10, 50, 100, 200] {
        print_search_summary("dim128", &index, &queries, &ground_truth, 10, ef, seed);
        group.throughput(Throughput::Elements(n_queries as u64));
        group.bench_with_input(
            BenchmarkId::new("ef", format!("{ef}/seed_{seed}")),
            &ef,
            |bench, &ef| {
                bench.iter(|| {
                    queries
                        .iter()
                        .map(|q| index.search(black_box(q), 10, ef).unwrap().len())
                        .sum::<usize>()
                });
            },
        );
    }

    group.finish();
}

fn bench_hnsw_search_mmax32(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_mmax32");

    let dim = 128;
    let n_vectors = 10_000;
    let n_queries = 100;
    let queries = random_vectors(n_queries, dim, 123);
    let seed = graph_seed();
    let (index, vectors) = build_index_with_params(n_vectors, dim, 16, 32, seed);
    let ground_truth = exact_neighbors(&vectors, &queries, 10);
    assert_eq!(
        index
            .search(&queries[0], 10, n_vectors)
            .unwrap()
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        ground_truth[0],
        "full-exploration control must match exact neighbors"
    );

    for ef in [10, 50, 100, 200] {
        print_search_summary(
            "dim128_m16_mmax32",
            &index,
            &queries,
            &ground_truth,
            10,
            ef,
            seed,
        );
        group.throughput(Throughput::Elements(n_queries as u64));
        group.bench_with_input(
            BenchmarkId::new("ef", format!("{ef}/seed_{seed}")),
            &ef,
            |bench, &ef| {
                bench.iter(|| {
                    queries
                        .iter()
                        .map(|q| index.search(black_box(q), 10, ef).unwrap().len())
                        .sum::<usize>()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_hnsw_search_only, bench_hnsw_search_mmax32);
criterion_main!(benches);
