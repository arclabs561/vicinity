//! Measurement harness: the random-projection tree forest (`rptree` family)
//! against Spotify's Annoy, measured on identical seeded data.
//!
//! This is a measurement, not a gate: timings are printed and never asserted.
//! Only a loose recall floor is asserted, so a regression that breaks the family
//! outright fails while ordinary variation does not.
//!
//! Run with:
//! ```text
//! cargo test --release --features annoy-bench --test annoy_parity -- --ignored --nocapture
//! ```
//!
//! Both sides see the same L2-normalized vectors, so Annoy's `angular` metric
//! and this crate's `1 - dot(q, v)` cosine are the same ordering.
//!
//! There is no `search_with_budget` on `RpForestIndex` on this branch, so the
//! forest runs its only search path: exact descent to one leaf per tree, exact
//! distance on the union. Its candidate budget is therefore structural
//! (`trees * leaf`), not a knob.

#![cfg(all(feature = "annoy-bench", feature = "rptree"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::time::{Duration, Instant};

use annoy_sys::AnnoyIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vicinity::classic::trees::rp_forest::{RPTreeParams, RpForestIndex, RpForestParams};

const N: usize = 10_000;
const DIM: usize = 96;
const K: usize = 10;
const N_QUERIES: usize = 400;
const N_CENTRES: usize = 16;
const SEED: u64 = 0x2026_1031;
/// Standard deviation of the gaussian noise around each cluster centre, before
/// re-normalization. 0.30 keeps clusters separable but overlapping.
const CLUSTER_SIGMA: f32 = 0.30;
/// Independent build+measure repetitions per configuration; timings are reduced
/// by median because a single sample on a shared machine swings several-fold.
const REPEATS: usize = 3;

// ── data ─────────────────────────────────────────────────────────────────────

struct Dataset {
    name: &'static str,
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

fn gauss(rng: &mut StdRng) -> f32 {
    // Box-Muller; clamp u1 away from 0 so ln() stays finite.
    let u1 = rng.random::<f32>().max(f32::MIN_POSITIVE);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x = (f64::from(*x) / norm) as f32;
        }
    }
}

fn random_unit(rng: &mut StdRng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| gauss(rng)).collect();
    normalize(&mut v);
    v
}

fn clustered_unit(rng: &mut StdRng, centre: &[f32]) -> Vec<f32> {
    let mut v: Vec<f32> = centre
        .iter()
        .map(|c| c + CLUSTER_SIGMA * gauss(rng))
        .collect();
    normalize(&mut v);
    v
}

fn uniform_dataset(seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    Dataset {
        name: "uniform",
        vectors: (0..N).map(|_| random_unit(&mut rng)).collect(),
        queries: (0..N_QUERIES).map(|_| random_unit(&mut rng)).collect(),
    }
}

fn clustered_dataset(seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..N_CENTRES).map(|_| random_unit(&mut rng)).collect();
    let pick = |rng: &mut StdRng| -> Vec<f32> {
        let c = centres[rng.random_range(0..N_CENTRES)].clone();
        clustered_unit(rng, &c)
    };
    Dataset {
        name: "clustered16",
        vectors: (0..N).map(|_| pick(&mut rng)).collect(),
        queries: (0..N_QUERIES).map(|_| pick(&mut rng)).collect(),
    }
}

/// Exact top-`K` by cosine similarity, accumulated in `f64` so the ground truth
/// is independent of either index's `f32` arithmetic.
fn exact_top_k(vectors: &[Vec<f32>], q: &[f32]) -> Vec<u32> {
    let mut scored: Vec<(f64, u32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f64 = v
                .iter()
                .zip(q)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            (dot, i as u32)
        })
        .collect();
    // Higher dot = closer; tie-break on id for determinism.
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(K).map(|(_, id)| id).collect()
}

// ── measurement ──────────────────────────────────────────────────────────────

struct Row {
    family: &'static str,
    trees: usize,
    setting: String,
    /// Items the search is allowed to look at: `trees * leaf` for the forest
    /// (one leaf per tree, exact distance on the union), `search_k` for Annoy.
    budget: usize,
    recall: f64,
    p50: Duration,
    build: Duration,
}

fn p50(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Median of repeated per-configuration samples.
fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn recall_and_latency<F>(queries: &[Vec<f32>], truth: &[Vec<u32>], mut search: F) -> (f64, Duration)
where
    F: FnMut(&[f32]) -> Vec<u32>,
{
    // Warm-up: page in the index, settle branch prediction and allocator state.
    for q in queries.iter().take(20) {
        let _ = search(q);
    }

    let mut hits = 0usize;
    let mut samples = Vec::with_capacity(queries.len());
    for (q, expected) in queries.iter().zip(truth) {
        let t0 = Instant::now();
        let found = search(q);
        samples.push(t0.elapsed());

        let found: HashSet<u32> = found.into_iter().collect();
        hits += expected.iter().filter(|id| found.contains(id)).count();
    }

    let recall = hits as f64 / (queries.len() * K) as f64;
    (recall, p50(samples))
}

fn measure_ours(ds: &Dataset, truth: &[Vec<u32>], trees: usize, leaf: usize) -> Row {
    let params = RpForestParams {
        num_trees: trees,
        tree_params: RPTreeParams {
            max_leaf_size: leaf,
        },
    };

    // Builds and latencies are repeated and reduced by median: a single sample
    // on a shared machine swings several-fold.
    let mut builds = Vec::with_capacity(REPEATS);
    let mut latencies = Vec::with_capacity(REPEATS);
    let mut recall = 0.0;

    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let mut index = RpForestIndex::new(DIM, params.clone()).expect("rp-forest: new");
        for (i, v) in ds.vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).expect("rp-forest: add");
        }
        index.build().expect("rp-forest: build");
        builds.push(t0.elapsed());

        let (r, p) = recall_and_latency(&ds.queries, truth, |q| {
            index
                .search(q, K)
                .expect("rp-forest: search")
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        });
        recall = r;
        latencies.push(p);
    }

    Row {
        family: "rp_forest",
        trees,
        setting: format!("leaf {leaf}"),
        budget: trees * leaf,
        recall,
        p50: median(latencies),
        build: median(builds),
    }
}

fn measure_annoy(ds: &Dataset, truth: &[Vec<u32>], trees: usize, search_k: i32) -> Row {
    let mut builds = Vec::with_capacity(REPEATS);
    let mut latencies = Vec::with_capacity(REPEATS);
    let mut recall = 0.0;
    let mut leaf_k = 0;

    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let mut index = AnnoyIndex::new(DIM);
        index.set_seed(u32::try_from(SEED & 0xFFFF_FFFF).expect("seed fits u32"));
        for (i, v) in ds.vectors.iter().enumerate() {
            assert!(index.add_item(i as u32, v), "annoy: add_item");
        }
        assert!(index.build(trees), "annoy: build");
        builds.push(t0.elapsed());

        let (r, p) = recall_and_latency(&ds.queries, truth, |q| {
            index
                .search(q, K, search_k)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        });
        recall = r;
        leaf_k = index.leaf_k();
        latencies.push(p);
    }

    let resolved = if search_k < 0 {
        K * trees
    } else {
        usize::try_from(search_k).expect("search_k fits usize")
    };
    let setting = if search_k < 0 {
        format!("search_k -1 (={resolved}), annoy K={leaf_k}")
    } else {
        format!("search_k {resolved}, annoy K={leaf_k}")
    };

    Row {
        family: "annoy",
        trees,
        setting,
        budget: resolved,
        recall,
        p50: median(latencies),
        build: median(builds),
    }
}

fn print_table(dataset: &str, rows: &[Row]) {
    println!("\n=== {dataset}: n={N} d={DIM} k={K} queries={N_QUERIES} metric=cosine ===");
    println!(
        "{:<10} {:>6} {:<38} {:>7} {:>10} {:>10} {:>10}",
        "family", "trees", "setting", "budget", "recall@10", "p50 us", "build ms"
    );
    println!("{}", "-".repeat(87));
    for r in rows {
        println!(
            "{:<10} {:>6} {:<38} {:>7} {:>10.3} {:>10.1} {:>10.0}",
            r.family,
            r.trees,
            r.setting,
            r.budget,
            r.recall,
            r.p50.as_secs_f64() * 1e6,
            r.build.as_secs_f64() * 1e3
        );
    }
}

fn print_iso_latency(rows: &[Row]) {
    println!("  recall at nearest p50 (ours vs the closest Annoy latency point):");
    for ours in rows.iter().filter(|r| r.family == "rp_forest") {
        let Some(annoy) = rows.iter().filter(|r| r.family == "annoy").min_by(|a, b| {
            let da = (a.p50.as_secs_f64() - ours.p50.as_secs_f64()).abs();
            let db = (b.p50.as_secs_f64() - ours.p50.as_secs_f64()).abs();
            da.total_cmp(&db)
        }) else {
            continue;
        };
        let ratio = annoy.p50.as_secs_f64() / ours.p50.as_secs_f64();
        println!(
            "    ours {}t/{:<10} {:>6.1} us r={:.3}  |  annoy {}t/{:<14} {:>6.1} us r={:.3}  \
             (latency x{ratio:.2}, recall {:+.3})",
            ours.trees,
            ours.setting,
            ours.p50.as_secs_f64() * 1e6,
            ours.recall,
            annoy.trees,
            annoy.setting,
            annoy.p50.as_secs_f64() * 1e6,
            annoy.recall,
            annoy.recall - ours.recall
        );
    }
}

// ── test ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "measurement harness; run explicitly with --ignored --nocapture --release"]
fn rp_forest_vs_annoy_parity() {
    let datasets = [uniform_dataset(SEED), clustered_dataset(SEED ^ 0x5EED)];
    let mut all_rows: Vec<Row> = Vec::new();

    for ds in &datasets {
        let truth: Vec<Vec<u32>> = ds
            .queries
            .iter()
            .map(|q| exact_top_k(&ds.vectors, q))
            .collect();

        // Matched-budget pairs: the forest sees one leaf per tree (exactly
        // `trees * leaf` items at most), Annoy sees up to `search_k` items
        // chosen by its best-first node queue. Extra Annoy points bracket the
        // forest's latency so recall-at-equal-latency can be read off directly.
        let rows = vec![
            measure_ours(ds, &truth, 10, 10),    // budget 100
            measure_annoy(ds, &truth, 10, -1),   // budget 100
            measure_ours(ds, &truth, 10, 20),    // budget 200
            measure_ours(ds, &truth, 50, 10),    // budget 500
            measure_annoy(ds, &truth, 50, -1),   // budget 500
            measure_annoy(ds, &truth, 10, 300),  // matches leaf 100 x 10 trees
            measure_ours(ds, &truth, 10, 100),   // budget 1000
            measure_annoy(ds, &truth, 10, 1000), // budget 1000
            measure_ours(ds, &truth, 50, 100),   // budget 5000
            measure_annoy(ds, &truth, 50, 2000), // brackets leaf 100 x 50 trees
            measure_annoy(ds, &truth, 50, 5000), // budget 5000
            measure_ours(ds, &truth, 100, 100),  // budget 10000
            measure_annoy(ds, &truth, 50, 8000), // brackets budget 10000
            measure_annoy(ds, &truth, 10, 2000),
        ];
        print_table(ds.name, &rows);
        print_iso_latency(&rows);
        all_rows.extend(rows);
    }

    // Loose recall floor only: timings are machine-dependent and never asserted.
    for r in &all_rows {
        assert!(
            r.recall >= 0.05,
            "{} with {} trees {}: recall@10 = {:.3} — below the sanity floor; \
             the family is not retrieving at all",
            r.family,
            r.trees,
            r.setting,
            r.recall
        );
    }
}
