#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Side-by-side search-latency comparison: `vicinity` HNSW vs the `usearch` crate.
//!
//! The earlier single-shot comparison in `examples/ann_benchmark` reported a QPS
//! column that swung ~8x between runs. This target exists to make the speed
//! comparison trustworthy:
//!
//! * both indexes are built from the same seeded corpus, and the ground truth is
//!   computed once, outside every timed region;
//! * every `(backend, ef)` pair is warmed up, then timed over several full passes
//!   of the query set, so a single slow first call cannot move the result;
//! * criterion reports steady-state wall time per pass (its `elem/s` column is
//!   QPS, because the throughput unit is one query), and this file additionally
//!   prints per-query median/p95 and recall@10 for the same configuration.
//!
//! Knobs: `N_VECTORS=10_000`, `DIM=96`, `N_QUERIES=200`, `k=10`,
//! `ef ∈ {64, 128, 256}`, `M=16`, `ef_construction=200`. That is small enough to
//! finish in a couple of minutes, large enough that the graph is not a toy.
//!
//! Distance semantics: vectors are L2-normalised once at generation, and both
//! backends then rank by cosine distance. `vicinity`'s cosine fast path is
//! `1 - dot(a, b)`. `usearch`'s `MetricKind::Cos` recomputes both norms on every
//! distance call, so it is the same ranking at extra cost; `MetricKind::IP` is
//! literally `1 - dot(a, b)` and therefore the arithmetic-identical comparison.
//! Both usearch kernels are measured so the difference is visible rather than
//! hidden inside a headline number.

use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use vicinity::distance::cosine_distance_normalized;
use vicinity::hnsw::{HNSWIndex, HNSWParams};

const N_VECTORS: usize = 10_000;
const DIM: usize = 96;
const N_QUERIES: usize = 200;
const K: usize = 10;
const M: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const EFS: [usize; 3] = [64, 128, 256];
const CORPUS_SEED: u64 = 42;
const QUERY_SEED: u64 = 123;
const WARMUP_PASSES: usize = 2;
const TIMED_PASSES: usize = 5;

/// A full pass over the query set is one criterion iteration.
const fn queries_per_iteration() -> u64 {
    N_QUERIES as u64
}

fn normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            vicinity::distance::normalize(&v)
        })
        .collect()
}

/// Exact cosine top-k, computed once and never timed.
fn exact_top_k(corpus: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            let mut scored: Vec<(u32, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(id, vector)| (id as u32, cosine_distance_normalized(query, vector)))
                .collect();
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            scored.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

fn recall_at_k(found: &[u32], truth: &[u32], k: usize) -> f64 {
    let truth = &truth[..k.min(truth.len())];
    let found = &found[..k.min(found.len())];
    truth.iter().filter(|id| found.contains(id)).count() as f64 / truth.len().max(1) as f64
}

fn build_vicinity(corpus: &[Vec<f32>]) -> HNSWIndex {
    let mut index = HNSWIndex::with_params(
        DIM,
        HNSWParams {
            m: M,
            // Layer 0 keeps 2*M edges, matching both the paper's `m_max0 = 2*M`
            // and usearch's default `connectivity_base = 2 * connectivity`. With
            // `m_max = M` the graph is strictly sparser than usearch's and the
            // recall columns become a comparison of two different graphs.
            m_max: 2 * M,
            ef_construction: EF_CONSTRUCTION,
            // Inputs are pre-normalised, so the extra per-query normalisation
            // pass would only add work that usearch does not do.
            auto_normalize: false,
            seed: Some(CORPUS_SEED),
            ..Default::default()
        },
    )
    .expect("create vicinity index");
    for (id, vector) in corpus.iter().enumerate() {
        index.add_slice(id as u32, vector).expect("vicinity add");
    }
    index.build().expect("vicinity build");
    index
}

fn build_usearch(corpus: &[Vec<f32>], metric: MetricKind) -> Index {
    let index = Index::new(&IndexOptions {
        dimensions: DIM,
        metric,
        quantization: ScalarKind::F32,
        connectivity: M,
        expansion_add: EF_CONSTRUCTION,
        ..Default::default()
    })
    .expect("create usearch index");
    // One reserved thread: this target compares sequential single-threaded
    // latency, and usearch's per-search thread-pool checkout is a mutex either
    // way, so a wider pool would only add contention of its own.
    index
        .reserve_capacity_and_threads(corpus.len(), 1)
        .expect("usearch reserve");
    for (id, vector) in corpus.iter().enumerate() {
        index.add(id as u64, vector).expect("usearch add");
    }
    index
}

#[derive(Clone, Copy, Debug)]
struct QueryStats {
    recall_at_k: f64,
    p50_us: f64,
    p95_us: f64,
    mean_us: f64,
    qps: f64,
    /// `1e6 / p50_us`: the QPS a quiet machine would show, ignoring the
    /// preemption tail that inflates the mean.
    qps_from_p50: f64,
    /// Slowest and fastest whole-pass QPS across the timed passes. A wide spread
    /// here means the machine, not the index, is moving the number.
    pass_qps_min: f64,
    pass_qps_max: f64,
}

/// Warm up, then measure. Recall is computed in its own untimed pass; latency
/// samples are one `Instant` pair per query, pooled across `TIMED_PASSES`.
fn measure(
    label: &str,
    search: &dyn Fn(&[f32]) -> Vec<u32>,
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
) -> QueryStats {
    for _ in 0..WARMUP_PASSES {
        for query in queries {
            black_box(search(query));
        }
    }

    let mut hits = 0.0;
    for (query, expected) in queries.iter().zip(truth) {
        let found = search(query);
        hits += recall_at_k(&found, expected, K);
    }
    let recall = hits / queries.len() as f64;

    let mut latencies_us = Vec::with_capacity(queries.len() * TIMED_PASSES);
    let mut pass_qps = Vec::with_capacity(TIMED_PASSES);
    for _ in 0..TIMED_PASSES {
        let pass_start = Instant::now();
        for query in queries {
            let q_start = Instant::now();
            let found = search(query);
            latencies_us.push(q_start.elapsed().as_nanos() as f64 / 1000.0);
            black_box(found);
        }
        let pass_s = pass_start.elapsed().as_secs_f64();
        pass_qps.push(queries.len() as f64 / pass_s);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = latencies_us.len();
    let total_us: f64 = latencies_us.iter().sum();
    let qps = n as f64 / (total_us / 1_000_000.0);

    let stats = QueryStats {
        recall_at_k: recall,
        p50_us: latencies_us[n / 2],
        p95_us: latencies_us[(((n as f64) * 0.95) as usize).min(n - 1)],
        mean_us: total_us / n as f64,
        qps,
        qps_from_p50: 1_000_000.0 / latencies_us[n / 2],
        pass_qps_min: pass_qps.iter().copied().fold(f64::INFINITY, f64::min),
        pass_qps_max: pass_qps.iter().copied().fold(0.0, f64::max),
    };
    eprintln!(
        "compare {label}: recall@{K}={:.4} p50={:.1}us p95={:.1}us mean={:.1}us qps={:.0} (pass qps {:.0}..{:.0})",
        stats.recall_at_k,
        stats.p50_us,
        stats.p95_us,
        stats.mean_us,
        stats.qps,
        stats.pass_qps_min,
        stats.pass_qps_max,
    );
    stats
}

/// Alternating per-query pairing, reporting the latency ratio A/B.
///
/// Absolute QPS moves with whatever else is running on the machine, so the
/// defensible claim is a ratio measured inside the same drifting window. Each
/// pair times the same query on both backends back to back and flips the order
/// every repetition, so neither a warm-up bias nor a steady drift can favour one
/// side consistently. The median over every pair is reported with a p10..p90
/// band: a wide band is the machine being noisy, not the index.
fn paired_latency_ratio(
    label: &str,
    ef: usize,
    a: &dyn Fn(&[f32]) -> usize,
    b: &dyn Fn(&[f32]) -> usize,
    queries: &[Vec<f32>],
) {
    let mut ratios = Vec::with_capacity(queries.len() * TIMED_PASSES);
    for rep in 0..TIMED_PASSES {
        for query in queries {
            let (a_us, b_us) = if rep % 2 == 0 {
                let a_us = time_one_query(a, query);
                let b_us = time_one_query(b, query);
                (a_us, b_us)
            } else {
                let b_us = time_one_query(b, query);
                let a_us = time_one_query(a, query);
                (a_us, b_us)
            };
            if a_us > 0.0 && b_us > 0.0 {
                ratios.push(a_us / b_us);
            }
        }
    }
    ratios.sort_unstable_by(|x, y| x.total_cmp(y));
    let n = ratios.len();
    println!(
        "paired ef={ef} {label} latency_ratio median={:.3} p10={:.3} p90={:.3} (n={n} query pairs)",
        ratios[n / 2],
        ratios[n / 10],
        ratios[n * 9 / 10]
    );
}

fn time_one_query(search: &dyn Fn(&[f32]) -> usize, query: &[f32]) -> f64 {
    let start = Instant::now();
    black_box(search(query));
    start.elapsed().as_nanos() as f64 / 1000.0
}

fn usearch_search(index: &Index, query: &[f32]) -> Vec<u32> {
    let matches = index.search(query, K).expect("usearch search");
    matches
        .keys
        .into_iter()
        .map(|key| u32::try_from(key).expect("usearch key fits in u32"))
        .collect()
}

/// Does `change_expansion_search` rebuild or re-warm the graph?
///
/// In usearch 2.26.0 it only assigns `config_.expansion_search`
/// (`include/usearch/index_dense.hpp`), so the expectation is no rebuild. This
/// probe measures the first two searches after each change against the
/// steady-state percentile, and re-runs the 256 -> 64 transition to show whether
/// the effect (if any) is order-dependent.
fn probe_expansion_search_transitions(index: &Index, queries: &[Vec<f32>]) {
    println!(
        "usearch probe: hardware_acceleration={} reserved_threads=1",
        index.hardware_acceleration()
    );
    println!("usearch probe: change_expansion_search(ef) first/second call vs steady state");

    let first_us = |index: &Index, query: &[f32]| {
        let start = Instant::now();
        black_box(index.search(query, K).expect("usearch search"));
        start.elapsed().as_nanos() as f64 / 1000.0
    };

    for ef in EFS {
        index.change_expansion_search(ef);
        let first = first_us(index, &queries[0]);
        let second = first_us(index, &queries[0]);

        let mut steady: Vec<f64> = queries
            .iter()
            .map(|query| {
                let start = Instant::now();
                black_box(index.search(query, K).expect("usearch search"));
                start.elapsed().as_nanos() as f64 / 1000.0
            })
            .collect();
        steady.sort_unstable_by(|a, b| a.total_cmp(b));
        println!(
            "usearch probe: ef={ef} first={first:.1}us second={second:.1}us steady_p50={:.1}us",
            steady[steady.len() / 2]
        );
    }

    // Repeat the 256 -> 64 transition: if the first call after a *lower* ef were
    // slow, the graph would be rebuilding or re-warming.
    index.change_expansion_search(256);
    black_box(index.search(&queries[0], K).expect("usearch search"));
    index.change_expansion_search(64);
    let first = first_us(index, &queries[0]);
    let second = first_us(index, &queries[0]);
    println!("usearch probe: 256->64 first={first:.1}us second={second:.1}us");
    index.change_expansion_search(64);
}

fn compare(c: &mut Criterion) {
    let corpus = normalized_vectors(N_VECTORS, DIM, CORPUS_SEED);
    let queries = normalized_vectors(N_QUERIES, DIM, QUERY_SEED);
    let truth = exact_top_k(&corpus, &queries, K);

    println!(
        "compare: n={N_VECTORS} dim={DIM} queries={N_QUERIES} k={K} m={M} ef_construction={EF_CONSTRUCTION} corpus_seed={CORPUS_SEED} query_seed={QUERY_SEED} warmup_passes={WARMUP_PASSES} timed_passes={TIMED_PASSES}"
    );

    let build_start = Instant::now();
    let vicinity = build_vicinity(&corpus);
    let vicinity_build_s = build_start.elapsed().as_secs_f64();

    let usearch_ip_build_start = Instant::now();
    let usearch_ip = build_usearch(&corpus, MetricKind::IP);
    let usearch_ip_build_s = usearch_ip_build_start.elapsed().as_secs_f64();

    let usearch_cos_build_start = Instant::now();
    let usearch_cos = build_usearch(&corpus, MetricKind::Cos);
    let usearch_cos_build_s = usearch_cos_build_start.elapsed().as_secs_f64();

    println!(
        "compare: build_s vicinity={vicinity_build_s:.2} usearch_ip={usearch_ip_build_s:.2} usearch_cos={usearch_cos_build_s:.2}"
    );

    probe_expansion_search_transitions(&usearch_ip, &queries);

    // Sanity checks that the comparison is apples-to-apples before any timing.
    //
    // The two backends build different graphs, so their approximate result sets
    // are not expected to be identical. What must match is the *space*: the exact
    // scan must be reachable by vicinity under full exploration (as in
    // `benches/hnsw_search.rs`), and the distance usearch reports for a hit must
    // equal vicinity's normalised-cosine distance for that pair.
    for (query, expected) in queries.iter().zip(&truth) {
        let found: Vec<u32> = vicinity
            .search(query, K, N_VECTORS)
            .expect("vicinity search")
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert_eq!(
            &found, expected,
            "full-exploration vicinity search must match the exact neighbours"
        );
    }

    let probe_query = &queries[0];
    let usearch_hits = usearch_ip.search(probe_query, K).expect("usearch search");
    for (key, distance) in usearch_hits.keys.iter().zip(&usearch_hits.distances) {
        let id = *key as usize;
        let expected_distance = cosine_distance_normalized(probe_query, &corpus[id]);
        assert!(
            (distance - expected_distance).abs() < 1e-4,
            "usearch IP distance {distance} for id {id} must equal normalised cosine {expected_distance}"
        );
    }

    let mut group = c.benchmark_group("hnsw_vs_usearch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_millis(500));
    group.throughput(Throughput::Elements(queries_per_iteration()));

    for ef in EFS {
        usearch_ip.change_expansion_search(ef);
        let label = format!("usearch_ip/ef{ef}");
        let stats = measure(
            &label,
            &|q| usearch_search(&usearch_ip, q),
            &queries,
            &truth,
        );
        println!(
            "table ef={ef} backend=usearch_ip recall@10={:.4} p50_us={:.1} p95_us={:.1} qps_mean={:.0} qps_p50={:.0}",
            stats.recall_at_k, stats.p50_us, stats.p95_us, stats.qps, stats.qps_from_p50
        );
        group.bench_with_input(BenchmarkId::new("usearch_ip", ef), &ef, |bench, _| {
            bench.iter(|| {
                let mut found = 0usize;
                for query in &queries {
                    found += black_box(usearch_search(&usearch_ip, query)).len();
                }
                found
            });
        });

        let stats = measure(
            &format!("vicinity/ef{ef}"),
            &|q| {
                vicinity
                    .search(q, K, ef)
                    .expect("vicinity search")
                    .iter()
                    .map(|(id, _)| *id)
                    .collect()
            },
            &queries,
            &truth,
        );
        println!(
            "table ef={ef} backend=vicinity recall@10={:.4} p50_us={:.1} p95_us={:.1} qps_mean={:.0} qps_p50={:.0}",
            stats.recall_at_k, stats.p50_us, stats.p95_us, stats.qps, stats.qps_from_p50
        );
        group.bench_with_input(BenchmarkId::new("vicinity", ef), &ef, |bench, _| {
            bench.iter(|| {
                let mut found = 0usize;
                for query in &queries {
                    found += black_box(
                        vicinity
                            .search(query, K, ef)
                            .expect("vicinity search")
                            .len(),
                    );
                }
                found
            });
        });

        usearch_cos.change_expansion_search(ef);
        let stats = measure(
            &format!("usearch_cos/ef{ef}"),
            &|q| usearch_search(&usearch_cos, q),
            &queries,
            &truth,
        );
        println!(
            "table ef={ef} backend=usearch_cos recall@10={:.4} p50_us={:.1} p95_us={:.1} qps_mean={:.0} qps_p50={:.0}",
            stats.recall_at_k, stats.p50_us, stats.p95_us, stats.qps, stats.qps_from_p50
        );
        group.bench_with_input(BenchmarkId::new("usearch_cos", ef), &ef, |bench, _| {
            bench.iter(|| {
                let mut found = 0usize;
                for query in &queries {
                    found += black_box(usearch_search(&usearch_cos, query)).len();
                }
                found
            });
        });

        paired_latency_ratio(
            "vicinity/usearch_ip",
            ef,
            &|q| vicinity.search(q, K, ef).expect("vicinity search").len(),
            &|q| usearch_search(&usearch_ip, q).len(),
            &queries,
        );
    }

    group.finish();
}

criterion_group!(benches, compare);
criterion_main!(benches);
