#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Search-only IVF-PQ benchmarks.
//!
//! Keep this target separate from the ANN dataset example so profiling search
//! does not include HDF5 loading or k-means training.

#[cfg(feature = "ivf_pq")]
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
#[cfg(feature = "ivf_pq")]
use rand::prelude::*;
#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
use vicinity::ivf_pq::IVFPQSearchProfile;
#[cfg(feature = "ivf_pq")]
use vicinity::ivf_pq::{IVFPQFileSearcher, IVFPQIndex, IVFPQParams};

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
struct CountingAlloc;

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
// SAFETY: This wrapper delegates allocation behavior to `System` unchanged and
// only records relaxed diagnostic counters for this benchmark binary.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: `layout` is provided by the allocator caller and is forwarded
        // unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` are the pair previously returned to the
        // caller; this wrapper does not alter either value.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        // SAFETY: `ptr`, `layout`, and `new_size` are forwarded unchanged to
        // the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

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
fn build_index(n_vectors: usize, dim: usize, num_codebooks: usize) -> (IVFPQIndex, Vec<Vec<f32>>) {
    let vectors = random_vectors(n_vectors, dim, 42);
    let params = IVFPQParams {
        num_clusters: 256,
        nprobe: 32,
        num_codebooks,
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

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
fn duration_us_per_query(duration: std::time::Duration, queries: usize) -> f64 {
    duration.as_secs_f64() * 1_000_000.0 / queries.max(1) as f64
}

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
fn count_per_query(count: usize, queries: usize) -> f64 {
    count as f64 / queries.max(1) as f64
}

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
#[derive(Clone, Copy, Debug, Default)]
struct AllocationProfile {
    calls: usize,
    bytes: usize,
}

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
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

#[cfg(all(feature = "ivf_pq", feature = "benchmark"))]
fn print_profile_summary(label: &str, index: &IVFPQIndex, queries: &[Vec<f32>], k: usize) {
    let mut total = IVFPQSearchProfile::default();
    let mut alloc_total = AllocationProfile::default();
    let mut result_count = 0usize;

    for query in queries {
        let ((results, profile), alloc_profile) =
            AllocationProfile::record_search(|| index.search_profiled(query, k).unwrap());
        alloc_total.add_assign(alloc_profile);
        result_count += results.len();
        total.add_assign(&profile);
    }

    eprintln!(
        concat!(
            "ivfpq profile {label}: ",
            "normalize={normalize:.3}us/query ",
            "centroid={centroid:.3}us/query ",
            "residual={residual:.3}us/query ",
            "adc_table={adc_table:.3}us/query ",
            "code_copy={code_copy:.3}us/query ",
            "adc_dispatch={adc_dispatch:.3}us/query ",
            "finalizer={finalizer:.3}us/query ",
            "clusters={clusters:.1}/query ",
            "simd_clusters={simd_clusters:.1}/query ",
            "scalar_clusters={scalar_clusters:.1}/query ",
            "scanned={scanned:.1}/query ",
            "candidates={candidates:.1}/query ",
            "copy_bytes={copy_bytes:.1}/query ",
            "alloc_calls={alloc_calls:.1}/query ",
            "alloc_bytes={alloc_bytes:.1}/query ",
            "results={results}"
        ),
        label = label,
        normalize = duration_us_per_query(total.normalize, queries.len()),
        centroid = duration_us_per_query(total.centroid_lookup, queries.len()),
        residual = duration_us_per_query(total.residual, queries.len()),
        adc_table = duration_us_per_query(total.adc_table, queries.len()),
        code_copy = duration_us_per_query(total.code_copy, queries.len()),
        adc_dispatch = duration_us_per_query(total.adc_dispatch, queries.len()),
        finalizer = duration_us_per_query(total.finalizer, queries.len()),
        clusters = count_per_query(total.probed_clusters, queries.len()),
        simd_clusters = count_per_query(total.simd_clusters, queries.len()),
        scalar_clusters = count_per_query(total.scalar_clusters, queries.len()),
        scanned = count_per_query(total.scanned_vectors, queries.len()),
        candidates = count_per_query(total.candidate_count, queries.len()),
        copy_bytes = count_per_query(total.code_copy_bytes, queries.len()),
        alloc_calls = count_per_query(alloc_total.calls, queries.len()),
        alloc_bytes = count_per_query(alloc_total.bytes, queries.len()),
        results = result_count
    );
    black_box(result_count);
}

#[cfg(feature = "ivf_pq")]
fn bench_ivfpq_search_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivfpq_search_only");

    let dim = 25;
    let n_vectors = 20_000;
    let n_queries = 100;
    let queries = random_vectors(n_queries, dim, 123);
    let configs = [("m25_one_dim", dim), ("m5_runner_default", 5)];

    for (label, num_codebooks) in configs {
        let (index, _vectors) = build_index(n_vectors, dim, num_codebooks);
        let snapshot_dir = tempfile::tempdir().expect("create IVF-PQ snapshot dir");
        index.save_to_dir(snapshot_dir.path()).unwrap();
        let loaded = IVFPQIndex::load_from_dir(snapshot_dir.path()).unwrap();
        let mut file_searcher = IVFPQFileSearcher::load(snapshot_dir.path()).unwrap();
        #[cfg(feature = "persistence")]
        let mut mmap_searcher = IVFPQFileSearcher::load_mmap(snapshot_dir.path()).unwrap();

        #[cfg(feature = "benchmark")]
        {
            print_profile_summary(&format!("{label}_nprobe32_k10"), &index, &queries, 10);
            print_profile_summary(&format!("{label}_nprobe32_pool500"), &index, &queries, 500);
        }

        group.throughput(Throughput::Elements(n_queries as u64));
        group.bench_function(format!("{label}_nprobe32_k10"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| index.search(black_box(q), 10).unwrap().len())
                    .sum::<usize>()
            });
        });

        group.bench_function(format!("{label}_snapshot_loaded_nprobe32_k10"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| loaded.search(black_box(q), 10).unwrap().len())
                    .sum::<usize>()
            });
        });

        group.bench_function(format!("{label}_file_nprobe32_k10"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| file_searcher.search(black_box(q), 10).unwrap().len())
                    .sum::<usize>()
            });
        });

        #[cfg(feature = "persistence")]
        group.bench_function(format!("{label}_mmap_nprobe32_k10"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| mmap_searcher.search(black_box(q), 10).unwrap().len())
                    .sum::<usize>()
            });
        });

        group.bench_function(format!("{label}_nprobe32_rerank500_k10"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| index.search_reranked(black_box(q), 10, 500).unwrap().len())
                    .sum::<usize>()
            });
        });

        group.bench_function(
            format!("{label}_snapshot_loaded_nprobe32_rerank500_k10"),
            |bench| {
                bench.iter(|| {
                    queries
                        .iter()
                        .map(|q| loaded.search_reranked(black_box(q), 10, 500).unwrap().len())
                        .sum::<usize>()
                });
            },
        );

        group.bench_function(format!("{label}_file_nprobe32_rerank500_k10"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| {
                        file_searcher
                            .search_reranked(black_box(q), 10, 500)
                            .unwrap()
                            .len()
                    })
                    .sum::<usize>()
            });
        });

        #[cfg(feature = "persistence")]
        group.bench_function(format!("{label}_mmap_nprobe32_rerank500_k10"), |bench| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| {
                        mmap_searcher
                            .search_reranked(black_box(q), 10, 500)
                            .unwrap()
                            .len()
                    })
                    .sum::<usize>()
            });
        });
    }

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
