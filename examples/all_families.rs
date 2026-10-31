//! One protocol, every index family in the crate.
//!
//! This harness puts each feature-gated index family behind a single trait and
//! runs the same dataset, the same ground truth, the same metric and the same
//! query-timing loop for all of them, so the results are comparable and
//! rankable. It is deliberately not the `ann_benchmark` runner: no HDF5
//! datasets, no external baselines, no per-algorithm CLI flags. Build and run:
//!
//! ```bash
//! cargo run --release --example all_families --features hnsw,nsw,nsg,finger,\
//! fresh_graph,filtered_graph,vamana,diskann,ivf_pq,ivf_avq,ivf_rabitq,emg,\
//! curator,lsh,pipnn,rp_quant,sq4,binary_index,sng,range_filtered,kdtree,\
//! balltree,kmeans_tree,rptree
//! ```
//!
//! # Protocol
//!
//! - `n = 10_000` base vectors, `dim = 96`, `k = 10`, `200` held-out queries.
//! - Two datasets: `uniform` (i.i.d. uniform in `[-1, 1]^96`) and `clustered`
//!   (16 uniform centres, `sigma = 0.25` noise). Both are L2-normalised, so
//!   cosine, angular and L2 ranking agree and every family can be fed the same
//!   vectors even though some are cosine-only (Vamana, EMG, RaBitQ) and some
//!   rank by L2 (IVF-PQ, k-d tree).
//! - Ground truth: exact top-10 by inner product on the normalised base set,
//!   computed once per dataset before any index is built or timed.
//! - Per query: recall@10, and one wall-clock sample. `p50/p95/p99` are over
//!   those per-query samples; `QPS` is queries divided by the whole timed loop.
//!   Five untimed warm-up queries run before the loop.
//! - `1/ratio@10` is the time-based definition used by ann-benchmarks: exact
//!   linear-scan p50 divided by the family's p50. Brute force is timed in the
//!   same process, single-threaded. Higher is faster than exact search.
//! - Build wall-clock time excludes dataset generation and ground truth.
//!
//! # Parameter sweeps
//!
//! A family exposes either a search-time knob (a per-call `ef`, `nprobe`, ...)
//! or only a construction-time knob. Search-time families are built once per
//! dataset and swept in place. Construction-time families are rebuilt for each
//! sweep point; their `build s` column is the build that produced that row.
//! Families with no search-time knob at all (`sng`) contribute one row.
//!
//! A family whose build or query fails is reported with the error instead of
//! being dropped silently. Pass a family name as the first argument to rerun
//! just that family (`cargo run --release --example all_families -- ivf_pq`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

/// Recall is compared at three decimals so a 0.99958 and a 0.99999 tie and the
/// faster row wins; below that, differences are noise at 200 queries.
fn recall3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

const N: usize = 10_000;
const DIM: usize = 96;
const K: usize = 10;
const N_QUERIES: usize = 200;
const CENTRES: usize = 16;
const SEED: u64 = 0x4641_4d49_4c59; // "FAMILY"

const EF_VALUES: &[usize] = &[10, 20, 50, 100, 200];
const NPROBE_VALUES: &[usize] = &[1, 2, 4, 8, 16, 32];
const CURATOR_EF_VALUES: &[usize] = &[50, 100, 200, 400, 800];
const RERANK_VALUES: &[usize] = &[1, 2, 5, 10, 20];
const PROBE_VALUES: &[usize] = &[1, 2, 4, 8, 16];
const LEAF_VALUES: &[usize] = &[5, 10, 25, 50, 100];
const BRANCH_VALUES: &[usize] = &[1, 2, 4, 8, 16];
const NO_KNOB: &[usize] = &[0];

/// A built index that can answer one query.
trait Searcher {
    /// `knob` is the family's swept search parameter. Families without one
    /// ignore it.
    fn search(&mut self, query: &[f32], k: usize, knob: usize) -> Result<Vec<(u32, f32)>, String>;
}

/// Adapts a closure to [`Searcher`] so each family needs one builder function
/// instead of its own struct.
struct FnIndex<F>(F)
where
    F: FnMut(&[f32], usize, usize) -> Result<Vec<(u32, f32)>, String>;

impl<F> Searcher for FnIndex<F>
where
    F: FnMut(&[f32], usize, usize) -> Result<Vec<(u32, f32)>, String>,
{
    fn search(&mut self, query: &[f32], k: usize, knob: usize) -> Result<Vec<(u32, f32)>, String> {
        (self.0)(query, k, knob)
    }
}

type BuildFn = fn(&[Vec<f32>], usize, usize) -> Result<Box<dyn Searcher>, String>;

struct Family {
    name: &'static str,
    knob: &'static str,
    values: &'static [usize],
    /// `true` when the knob only affects construction, so each sweep point
    /// needs a fresh index.
    rebuild: bool,
    /// Fixed construction configuration and search path taken.
    note: &'static str,
    build: BuildFn,
}

struct Row {
    dataset: &'static str,
    family: &'static str,
    knob: &'static str,
    knob_value: usize,
    recall: f64,
    ratio: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    qps: f64,
    build_s: f64,
    errors: usize,
    first_error: Option<String>,
}

struct Stats {
    recall: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    qps: f64,
    errors: usize,
    first_error: Option<String>,
}

struct Dataset {
    name: &'static str,
    base: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    truth: Vec<Vec<u32>>,
    exact_p50_us: f64,
}

struct Skipped {
    family: &'static str,
    reason: &'static str,
}

fn main() {
    // Optional positional filter: `all_families ivf_pq` reruns one family.
    let only = std::env::args().nth(1);
    let mut datasets = build_datasets();
    let families: Vec<Family> = families()
        .into_iter()
        .filter(|f| only.as_deref().is_none_or(|name| name == f.name))
        .collect();
    let mut rows: Vec<Row> = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();

    println!("# All-family ANN benchmark");
    println!();
    println!(
        "n={N} dim={DIM} k={K} queries={N_QUERIES} seed=0x{SEED:016X} \
         datasets=uniform,clustered(16 centres) metric=cosine (L2-normalised)"
    );
    println!();

    for dataset in &mut datasets {
        println!(
            "exact linear scan p50 on `{}`: {:.1} us/query",
            dataset.name, dataset.exact_p50_us
        );
    }
    println!();
    println!(
        "1/ratio@10 = exact-scan p50 / family p50 (time-based, as in ann-benchmarks); \
         QPS = queries / total timed-loop seconds; p50/p95/p99 are per-query wall clock."
    );
    println!();

    for dataset in &datasets {
        for family in &families {
            match run_family(family, dataset) {
                Ok(mut produced) => rows.append(&mut produced),
                Err(reason) => {
                    if !failures.iter().any(|(name, _)| name == family.name) {
                        failures.push((family.name.to_string(), reason));
                    }
                }
            }
        }
    }

    print_rows(&rows);
    print_ranking(&rows);
    print_at_recall(&rows, 0.95);
    print_sweeps(&families);
    print_skipped(&failures);
}

// ---------------------------------------------------------------------------
// Protocol: datasets, ground truth, timing
// ---------------------------------------------------------------------------

fn build_datasets() -> Vec<Dataset> {
    let uniform = make_dataset("uniform", SEED, None);
    let clustered = make_dataset("clustered", SEED ^ 0xC1_0000, Some(0.25));
    vec![uniform, clustered]
}

fn make_dataset(name: &'static str, seed: u64, spread: Option<f32>) -> Dataset {
    use rand::{rngs::StdRng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let centres = sample_centres(&mut rng, spread.is_some());
    let base = sample_points(&mut rng, &centres, spread, N);
    let queries = sample_points(&mut rng, &centres, spread, N_QUERIES);
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| exact_top_k(&base, q, K)).collect();

    // Time the exact scan the same way queries are timed below, so `1/ratio`
    // uses a like-for-like p50.
    for q in queries.iter().take(5) {
        let _ = exact_top_k(&base, q, K);
    }
    let mut lat = Vec::with_capacity(queries.len());
    for q in &queries {
        let t = Instant::now();
        let _ = exact_top_k(&base, q, K);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }

    Dataset {
        name,
        base,
        queries,
        truth,
        exact_p50_us: percentile(&mut lat.clone(), 50.0),
    }
}

fn sample_centres(rng: &mut rand::rngs::StdRng, needed: bool) -> Vec<Vec<f32>> {
    if !needed {
        return Vec::new();
    }
    (0..CENTRES).map(|_| normalize(sample_raw(rng))).collect()
}

fn sample_points(
    rng: &mut rand::rngs::StdRng,
    centres: &[Vec<f32>],
    spread: Option<f32>,
    count: usize,
) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let noise = sample_raw(rng);
            match spread {
                None => normalize(noise),
                Some(sigma) => {
                    let centre = &centres[i % centres.len()];
                    let mixed: Vec<f32> = centre
                        .iter()
                        .zip(noise.iter())
                        .map(|(c, n)| c + sigma * n)
                        .collect();
                    normalize(mixed)
                }
            }
        })
        .collect()
}

fn sample_raw(rng: &mut rand::rngs::StdRng) -> Vec<f32> {
    use rand::Rng;
    (0..DIM).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Exact top-k by inner product. This is both the ground truth and the
/// brute-force baseline whose p50 defines `1/ratio`.
fn exact_top_k(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (dot(v, query), i as u32))
        .collect();
    if k < scored.len() {
        scored.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
    }
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().map(|(_, id)| id).collect()
}

fn evaluate(index: &mut dyn Searcher, dataset: &Dataset, knob: usize) -> Stats {
    for query in dataset.queries.iter().take(5) {
        let _ = index.search(query, K, knob);
    }

    let mut lat = Vec::with_capacity(dataset.queries.len());
    let mut hits = 0usize;
    let mut errors = 0usize;
    let mut first_error = None;

    let loop_start = Instant::now();
    for (query, truth) in dataset.queries.iter().zip(dataset.truth.iter()) {
        let start = Instant::now();
        match index.search(query, K, knob) {
            Ok(results) => {
                hits += intersection_count(&results, truth, K);
            }
            Err(err) => {
                errors += 1;
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        lat.push(start.elapsed().as_secs_f64() * 1e6);
    }
    let total_s = loop_start.elapsed().as_secs_f64();

    let denominator = (dataset.queries.len() * K) as f64;
    Stats {
        recall: hits as f64 / denominator,
        p50: percentile(&mut lat.clone(), 50.0),
        p95: percentile(&mut lat.clone(), 95.0),
        p99: percentile(&mut lat.clone(), 99.0),
        qps: dataset.queries.len() as f64 / total_s.max(f64::EPSILON),
        errors,
        first_error,
    }
}

fn intersection_count(results: &[(u32, f32)], truth: &[u32], k: usize) -> usize {
    results
        .iter()
        .take(k)
        .filter(|(id, _)| truth.iter().take(k).any(|t| t == id))
        .count()
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

fn run_family(family: &Family, dataset: &Dataset) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();

    if family.rebuild {
        for &knob in family.values {
            let build_start = Instant::now();
            let mut index = (family.build)(&dataset.base, DIM, knob)
                .map_err(|e| format!("build failed at {}={knob}: {e}", family.knob))?;
            let build_s = build_start.elapsed().as_secs_f64();
            rows.push(row_for(family, dataset, knob, build_s, &mut *index));
        }
    } else {
        let build_start = Instant::now();
        let mut index = (family.build)(&dataset.base, DIM, family.values[0])
            .map_err(|e| format!("build failed: {e}"))?;
        let build_s = build_start.elapsed().as_secs_f64();
        for &knob in family.values {
            rows.push(row_for(family, dataset, knob, build_s, &mut *index));
        }
    }

    Ok(rows)
}

fn row_for(
    family: &Family,
    dataset: &Dataset,
    knob: usize,
    build_s: f64,
    index: &mut dyn Searcher,
) -> Row {
    let stats = evaluate(index, dataset, knob);
    Row {
        dataset: dataset.name,
        family: family.name,
        knob: family.knob,
        knob_value: knob,
        recall: stats.recall,
        ratio: dataset.exact_p50_us / stats.p50.max(f64::MIN_POSITIVE),
        p50: stats.p50,
        p95: stats.p95,
        p99: stats.p99,
        qps: stats.qps,
        build_s,
        errors: stats.errors,
        first_error: stats.first_error,
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn print_rows(rows: &[Row]) {
    for dataset in ["uniform", "clustered"] {
        println!("## {dataset}");
        println!();
        println!(
            "| family | knob | recall@10 | 1/ratio@10 | p50 us | p95 us | p99 us | QPS | build s |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for row in rows.iter().filter(|r| r.dataset == dataset) {
            println!(
                "| {} | {}={} | {:.3} | {:.2} | {:.1} | {:.1} | {:.1} | {:.0} | {:.2} |",
                row.family,
                row.knob,
                row.knob_value,
                row.recall,
                row.ratio,
                row.p50,
                row.p95,
                row.p99,
                row.qps,
                row.build_s
            );
            if row.errors > 0 {
                println!(
                    "| ^ {} query errors: {} | | | | | | | | |",
                    row.errors,
                    row.first_error.as_deref().unwrap_or("unknown")
                );
            }
        }
        println!();
    }
}

/// One line per family: the best operating point by recall, tie-broken by p50.
fn print_ranking(rows: &[Row]) {
    for dataset in ["uniform", "clustered"] {
        println!("## {dataset} ranking (best recall, tie-break p50)");
        println!();
        println!("| rank | family | at | recall@10 | p50 us | build s |");
        println!("| ---: | --- | --- | ---: | ---: | ---: |");
        let mut best: Vec<&Row> = Vec::new();
        for row in rows.iter().filter(|r| r.dataset == dataset) {
            match best
                .iter_mut()
                .find(|existing| existing.family == row.family)
            {
                Some(existing) => {
                    let (row_r, old_r) = (recall3(row.recall), recall3(existing.recall));
                    if row_r > old_r || (row_r == old_r && row.p50 < existing.p50) {
                        *existing = row;
                    }
                }
                None => best.push(row),
            }
        }
        best.sort_by(|a, b| {
            recall3(b.recall)
                .total_cmp(&recall3(a.recall))
                .then_with(|| a.p50.total_cmp(&b.p50))
        });
        for (i, row) in best.iter().enumerate() {
            println!(
                "| {} | {} | {}={} | {:.3} | {:.1} | {:.2} |",
                i + 1,
                row.family,
                row.knob,
                row.knob_value,
                row.recall,
                row.p50,
                row.build_s
            );
        }
        println!();
    }
}

/// Fastest operating point per family subject to `recall >= target`. Families
/// that never reach the target contribute their highest-recall row, marked `*`,
/// so the gap is visible instead of silently dropping them.
fn print_at_recall(rows: &[Row], target: f64) {
    for dataset in ["uniform", "clustered"] {
        println!(
            "## {dataset} at recall@10 >= {target:.2} (fastest p50; `*` = target not reached)"
        );
        println!();
        println!("| family | at | recall@10 | p50 us | QPS | 1/ratio@10 |");
        println!("| --- | --- | ---: | ---: | ---: | ---: |");
        let mut chosen: Vec<(bool, &Row)> = Vec::new();
        for row in rows.iter().filter(|r| r.dataset == dataset) {
            match chosen.iter_mut().find(|(_, c)| c.family == row.family) {
                Some((ok, current)) => {
                    let row_ok = recall3(row.recall) >= target;
                    if (row_ok && !*ok)
                        || (row_ok == *ok
                            && ((row_ok && row.p50 < current.p50)
                                || (!row_ok && recall3(row.recall) > recall3(current.recall))))
                    {
                        *ok = row_ok;
                        *current = row;
                    }
                }
                None => chosen.push((recall3(row.recall) >= target, row)),
            }
        }
        chosen.sort_by(|a, b| a.1.p50.total_cmp(&b.1.p50));
        for (ok, row) in chosen {
            println!(
                "| {}{} | {}={} | {:.3} | {:.1} | {:.0} | {:.2} |",
                row.family,
                if ok { "" } else { "*" },
                row.knob,
                row.knob_value,
                row.recall,
                row.p50,
                row.qps,
                row.ratio
            );
        }
        println!();
    }
}

fn print_sweeps(families: &[Family]) {
    println!("## Sweeps");
    println!();
    for family in families {
        if family.values == NO_KNOB {
            println!(
                "- `{}`: no search-time knob; one row at defaults. {}",
                family.name, family.note
            );
        } else if family.rebuild {
            println!(
                "- `{}`: `{}` swept {:?}, rebuilt per point. {}",
                family.name, family.knob, family.values, family.note
            );
        } else {
            println!(
                "- `{}`: `{}` swept {:?} on one build. {}",
                family.name, family.knob, family.values, family.note
            );
        }
    }
    println!();
}

fn print_skipped(failures: &[(String, String)]) {
    println!("## Not covered");
    println!();
    println!("| family | why |");
    println!("| --- | --- |");
    for entry in SKIPPED {
        println!("| {} | {} |", entry.family, entry.reason);
    }
    for (family, reason) in failures {
        println!("| {} | runtime: {} |", family, reason);
    }
    println!();
}

const SKIPPED: &[Skipped] = &[
    Skipped {
        family: "evoc",
        reason: "clustering wrapper: exposes fit_predict/cluster_layers, no k-NN query API",
    },
    Skipped {
        family: "lemur",
        reason: "needs externally trained LemurEncoder weights and multi-vector (token) documents; in-tree random weights are test-only",
    },
    Skipped {
        family: "sparse_mips",
        reason: "sparse-only input (SparseVector); dense 96-d vectors are fully dense, so the sparse path is not the family's workload",
    },
];

// ---------------------------------------------------------------------------
// Family builders
// ---------------------------------------------------------------------------

#[allow(clippy::vec_init_then_push)]
fn families() -> Vec<Family> {
    let mut families = Vec::new();

    #[cfg(feature = "hnsw")]
    families.push(Family {
        name: "hnsw",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "m=16, ef_construction=200, cosine; ef_search per call",
        build: build_hnsw,
    });

    #[cfg(feature = "nsw")]
    families.push(Family {
        name: "nsw",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "m=m_max=16, ef_construction=100; ef_search per call",
        build: build_nsw,
    });

    #[cfg(feature = "nsg")]
    families.push(Family {
        name: "nsg",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "R=32, L=64, K=32; search_with_ef",
        build: build_nsg,
    });

    #[cfg(feature = "finger")]
    families.push(Family {
        name: "finger",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "degree 32, ef_construction 200, alpha 1.2; search_with_ef",
        build: build_finger,
    });

    #[cfg(feature = "fresh_graph")]
    families.push(Family {
        name: "fresh_graph",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "degree 32, ef_construction 200, alpha 1.2; search_with_ef",
        build: build_fresh_graph,
    });

    #[cfg(feature = "filtered_graph")]
    families.push(Family {
        name: "filtered_graph",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "degree 32, ef_construction 200; attributes empty (unfiltered path); search_with_ef",
        build: build_filtered_graph,
    });

    #[cfg(feature = "vamana")]
    families.push(Family {
        name: "vamana",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "max_degree 64, alpha 1.3, ef_construction 200, seed 42; ef_search per call",
        build: build_vamana,
    });

    #[cfg(feature = "diskann")]
    families.push(Family {
        name: "diskann",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "in-memory DiskANNIndex: m=32, ef_construction=100, alpha 1.2, seed 42; ef_search per call",
        build: build_diskann,
    });

    #[cfg(feature = "emg")]
    families.push(Family {
        name: "emg",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "degree 32, candidate_size 64, scale_t 32, 2 iterations; search_with_ef",
        build: build_emg,
    });

    #[cfg(feature = "pipnn")]
    families.push(Family {
        name: "pipnn",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: false,
        note: "leaf 2048, degree 32, 12 hash bits, final RobustPrune; search_with_ef",
        build: build_pipnn,
    });

    #[cfg(feature = "ivf_pq")]
    families.push(Family {
        name: "ivf_pq",
        knob: "nprobe",
        values: NPROBE_VALUES,
        rebuild: false,
        note: "100 clusters (sqrt n), M=16 (16 B/vector, 24x vs 384 B raw), codebook 256; nprobe per call, search_reranked pool 100",
        build: build_ivf_pq,
    });

    #[cfg(feature = "ivf_avq")]
    families.push(Family {
        name: "ivf_avq",
        knob: "nprobe",
        values: NPROBE_VALUES,
        rebuild: false,
        note: "100 partitions (sqrt n), M=16, codebook 256, reorder 100; nprobe per call",
        build: build_ivf_avq,
    });

    #[cfg(feature = "ivf_rabitq")]
    families.push(Family {
        name: "ivf_rabitq",
        knob: "nprobe",
        values: NPROBE_VALUES,
        rebuild: false,
        note: "100 clusters (sqrt n), 4 bits/dim; nprobe per call",
        build: build_ivf_rabitq,
    });

    #[cfg(feature = "lsh")]
    families.push(Family {
        name: "lsh",
        knob: "num_probes",
        values: PROBE_VALUES,
        rebuild: true,
        note: "8 tables, seed 42; num_probes at build (multiprobe width)",
        build: build_lsh,
    });

    #[cfg(feature = "rp_quant")]
    families.push(Family {
        name: "rp_quant",
        knob: "rerank_factor",
        values: RERANK_VALUES,
        rebuild: true,
        note: "64-d projection, seed 42; rerank_factor at build",
        build: build_rp_quant,
    });

    #[cfg(feature = "sq4")]
    families.push(Family {
        name: "sq4",
        knob: "rerank_factor",
        values: RERANK_VALUES,
        rebuild: true,
        note: "4-bit codes, exact rerank; rerank_factor at build",
        build: build_sq4,
    });

    #[cfg(feature = "binary_index")]
    families.push(Family {
        name: "binary_index",
        knob: "rerank_factor",
        values: RERANK_VALUES,
        rebuild: true,
        note: "1 bit/dim after rotation, exact rerank; rerank_factor at build",
        build: build_binary_index,
    });

    #[cfg(feature = "sng")]
    families.push(Family {
        name: "sng",
        knob: "none",
        values: NO_KNOB,
        rebuild: false,
        note: "auto-optimised max_degree; index has no search-time knob",
        build: build_sng,
    });

    #[cfg(feature = "curator")]
    families.push(Family {
        name: "curator",
        knob: "ef_search",
        values: CURATOR_EF_VALUES,
        rebuild: true,
        note: "branching 16, leaf 128, beam 4; ef_search at build; unfiltered search",
        build: build_curator,
    });

    #[cfg(feature = "range_filtered")]
    families.push(Family {
        name: "range_filtered",
        knob: "ef_search",
        values: EF_VALUES,
        rebuild: true,
        note: "HNSW m=16, ef_construction=200; ef_search at build; unfiltered range",
        build: build_range_filtered,
    });

    #[cfg(feature = "kdtree")]
    families.push(Family {
        name: "kdtree",
        knob: "max_leaf_size",
        values: LEAF_VALUES,
        rebuild: true,
        note: "max_leaf_size at build; rejects dim > 50",
        build: build_kdtree,
    });

    #[cfg(feature = "balltree")]
    families.push(Family {
        name: "balltree",
        knob: "max_leaf_size",
        values: LEAF_VALUES,
        rebuild: true,
        note: "max_leaf_size at build; branch-and-bound pruning",
        build: build_balltree,
    });

    #[cfg(feature = "kmeans_tree")]
    families.push(Family {
        name: "kmeans_tree",
        knob: "branch_budget",
        values: BRANCH_VALUES,
        rebuild: false,
        note: "16 clusters, leaf 50; branch_budget per call",
        build: build_kmeans_tree,
    });

    #[cfg(feature = "rptree")]
    families.push(Family {
        name: "rptree",
        knob: "max_leaf_size",
        values: LEAF_VALUES,
        rebuild: true,
        note: "max_leaf_size at build; search visits both children (no pruning)",
        build: build_rptree,
    });

    #[cfg(feature = "rptree")]
    families.push(Family {
        name: "rp_forest",
        knob: "max_leaf_size",
        values: LEAF_VALUES,
        rebuild: true,
        note: "10 trees; max_leaf_size at build",
        build: build_rp_forest,
    });

    families
}

fn to_string_err<E: std::fmt::Display>(err: E) -> String {
    err.to_string()
}

#[cfg(feature = "hnsw")]
fn build_hnsw(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::distance::DistanceMetric;
    use vicinity::hnsw::HNSWIndex;

    let mut index = HNSWIndex::builder(dim)
        .m(16)
        .ef_construction(200)
        .metric(DistanceMetric::Cosine)
        .build()
        .map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "nsw")]
fn build_nsw(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::nsw::{NSWIndex, NSWParams};

    let mut index = NSWIndex::with_params(dim, NSWParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "nsg")]
fn build_nsg(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::nsg::{NsgIndex, NsgParams};

    let mut index = NsgIndex::new(dim, NsgParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "finger")]
fn build_finger(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::finger::{FingerIndex, FingerParams};

    let mut index = FingerIndex::new(dim, FingerParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "fresh_graph")]
fn build_fresh_graph(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};

    let mut index =
        FreshGraphIndex::new(dim, FreshGraphParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "filtered_graph")]
fn build_filtered_graph(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use std::collections::HashMap;
    use vicinity::filtered_graph::{FilteredGraphIndex, FilteredGraphParams};

    let mut index =
        FilteredGraphIndex::new(dim, FilteredGraphParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index
            .add_slice(i as u32, vector, HashMap::new())
            .map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "vamana")]
fn build_vamana(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::vamana::{VamanaIndex, VamanaParams};

    let params = VamanaParams {
        seed: Some(42),
        ..VamanaParams::default()
    };
    let mut index = VamanaIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "diskann")]
fn build_diskann(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::diskann::{DiskANNIndex, DiskANNParams};

    let params = DiskANNParams {
        seed: Some(42),
        ..DiskANNParams::default()
    };
    let mut index = DiskANNIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "emg")]
fn build_emg(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::emg::{EmgIndex, EmgParams};

    let mut index = EmgIndex::new(dim, EmgParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "pipnn")]
fn build_pipnn(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::pipnn::{PipnnIndex, PipnnParams};

    let mut index = PipnnIndex::new(dim, PipnnParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, ef| {
        index.search_with_ef(query, k, ef).map_err(to_string_err)
    })))
}

#[cfg(feature = "ivf_pq")]
fn build_ivf_pq(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

    let params = IVFPQParams {
        num_clusters: 100,
        nprobe: 4,
        num_codebooks: 16,
        codebook_size: 256,
        seed: 42,
        ..IVFPQParams::default()
    };
    let mut index = IVFPQIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    // `search_reranked` with an exact-rerank pool is the documented IVF-PQ
    // path; plain `search` returns PQ-approximated distances and ranks poorly.
    Ok(Box::new(FnIndex(move |query, k, nprobe| {
        index.set_nprobe(nprobe);
        index.search_reranked(query, k, 100).map_err(to_string_err)
    })))
}

#[cfg(feature = "ivf_avq")]
fn build_ivf_avq(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::ivf_avq::{IVFAVQIndex, IVFAVQParams};

    let params = IVFAVQParams {
        num_partitions: 100,
        nprobe: 4,
        num_reorder: 100,
        num_codebooks: 16,
        codebook_size: 256,
        seed: 42,
    };
    let mut index = IVFAVQIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, nprobe| {
        index.set_nprobe(nprobe);
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "ivf_rabitq")]
fn build_ivf_rabitq(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams};

    let params = IVFRaBitQParams {
        num_clusters: 100,
        nprobe: 4,
        total_bits: 4,
        seed: 42,
    };
    let mut index = IVFRaBitQIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, nprobe| {
        index
            .search_with_ef(query, k, nprobe)
            .map_err(to_string_err)
    })))
}

#[cfg(feature = "lsh")]
fn build_lsh(vectors: &[Vec<f32>], dim: usize, knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::lsh::{CrossPolytopeLSHIndex, LSHParams};

    let params = LSHParams {
        num_tables: 8,
        num_probes: knob,
        seed: Some(42),
    };
    let mut index = CrossPolytopeLSHIndex::new(dim, params).map_err(to_string_err)?;
    let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
    index.add_vectors(&flat).map_err(to_string_err)?;
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "rp_quant")]
fn build_rp_quant(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::rp_quant::{RpQuantIndex, RpQuantParams};

    let params = RpQuantParams {
        projected_dim: 64,
        rerank_factor: knob,
        seed: 42,
    };
    let mut index = RpQuantIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "sq4")]
fn build_sq4(vectors: &[Vec<f32>], dim: usize, knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::sq4::{SQ4Index, SQ4Params};

    let params = SQ4Params {
        rerank_factor: knob,
    };
    let mut index = SQ4Index::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "binary_index")]
fn build_binary_index(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::binary_index::{BinaryFlatIndex, BinaryFlatParams};

    let params = BinaryFlatParams {
        projected_dim: dim,
        rerank_factor: knob,
        seed: 42,
    };
    let mut index = BinaryFlatIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add_slice(i as u32, vector).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "sng")]
fn build_sng(vectors: &[Vec<f32>], dim: usize, _knob: usize) -> Result<Box<dyn Searcher>, String> {
    use vicinity::sng::{SNGIndex, SNGParams};

    let mut index = SNGIndex::new(dim, SNGParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "curator")]
fn build_curator(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::curator::{CuratorIndex, CuratorParams};

    let params = CuratorParams {
        ef_search: knob,
        ..CuratorParams::default()
    };
    let mut index = CuratorIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index
            .add(i as u32, vector.clone(), Vec::new())
            .map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    // Unfiltered search so the row is comparable with the other families; the
    // label-filtered path is a different workload.
    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "range_filtered")]
fn build_range_filtered(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::range_filtered::{RangeFilteredIndex, RangeFilteredParams};

    let params = RangeFilteredParams {
        ef_search: knob,
        ..RangeFilteredParams::default()
    };
    let mut index = RangeFilteredIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index
            .add(i as u32, vector.clone(), i as f64)
            .map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    // `search` is the unfiltered path; the attribute range is metadata only.
    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "kdtree")]
fn build_kdtree(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::kdtree::{KDTreeIndex, KDTreeParams};

    let params = KDTreeParams {
        max_leaf_size: knob,
        ..KDTreeParams::default()
    };
    let mut index = KDTreeIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "balltree")]
fn build_balltree(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::balltree::{BallTreeIndex, BallTreeParams};

    let params = BallTreeParams {
        max_leaf_size: knob,
        ..BallTreeParams::default()
    };
    let mut index = BallTreeIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "kmeans_tree")]
fn build_kmeans_tree(
    vectors: &[Vec<f32>],
    dim: usize,
    _knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::kmeans_tree::{KMeansTreeIndex, KMeansTreeParams};

    let mut index =
        KMeansTreeIndex::new(dim, KMeansTreeParams::default()).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, budget| {
        index
            .search_with_branch_budget(query, k, budget)
            .map_err(to_string_err)
    })))
}

#[cfg(feature = "rptree")]
fn build_rptree(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::random_projection::{RPTreeIndex, RPTreeParams};

    let params = RPTreeParams {
        max_leaf_size: knob,
        ..RPTreeParams::default()
    };
    let mut index = RPTreeIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}

#[cfg(feature = "rptree")]
fn build_rp_forest(
    vectors: &[Vec<f32>],
    dim: usize,
    knob: usize,
) -> Result<Box<dyn Searcher>, String> {
    use vicinity::classic::trees::rp_forest::{RPTreeParams, RpForestIndex, RpForestParams};

    let params = RpForestParams {
        num_trees: 10,
        tree_params: RPTreeParams {
            max_leaf_size: knob,
        },
    };
    let mut index = RpForestIndex::new(dim, params).map_err(to_string_err)?;
    for (i, vector) in vectors.iter().enumerate() {
        index.add(i as u32, vector.clone()).map_err(to_string_err)?;
    }
    index.build().map_err(to_string_err)?;

    Ok(Box::new(FnIndex(move |query, k, _| {
        index.search(query, k).map_err(to_string_err)
    })))
}
