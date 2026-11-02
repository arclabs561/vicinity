//! Measurement harness: the random-projection tree forest (`rptree` family)
//! against Spotify's Annoy, measured on identical seeded data.
//!
//! This is a measurement, not a gate: timings are printed and never asserted.
//! Only a loose recall floor is asserted, so a regression that breaks the family
//! outright fails while ordinary variation does not.
//!
//! Run from the repository root:
//! ```text
//! cargo test --release --manifest-path tools/annoy-parity/Cargo.toml -- --ignored --nocapture
//! ```
//!
//! This lives outside `vicinity`'s manifest on purpose: it links a local C++
//! shim that is `publish = false`, and a path dependency without a version in
//! the parent manifest blocks `cargo publish` outright.
//!
//! Both sides see the same L2-normalized vectors, so Annoy's `angular` metric
//! and this crate's `1 - dot(q, v)` cosine are the same ordering.
//!
//! The forest runs this branch's priority-queue traversal: both children of every
//! split are pushed and expanded in bound order, so its candidate budget is the
//! `search_k` knob of [`RpForestIndex::search_with_budget`], directly comparable
//! to Annoy's `search_k`. The build leaf size stays structural — it caps how many
//! ids one leaf contributes — so the table reports the resolved cap per row.

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

/// `leaf` is the build-time leaf cap; `0` means "derive from the dimension".
/// `budget` is the `search_k` handed to `search_with_budget`, i.e. the number of
/// distinct ids the traversal is allowed to score, matching Annoy's `search_k`.
fn measure_ours(ds: &Dataset, truth: &[Vec<u32>], trees: usize, leaf: usize, budget: usize) -> Row {
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
                .search_with_budget(q, K, budget)
                .expect("rp-forest: search_with_budget")
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        });
        recall = r;
        latencies.push(p);
    }

    let resolved = params.tree_params.leaf_size(DIM);
    Row {
        family: "rp_forest",
        trees,
        setting: format!("leaf {resolved}, search_k {budget}"),
        budget,
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

/// Annoy recall interpolated onto a target latency, from the Annoy rows of this
/// table sorted by p50. `None` when the target is faster than every Annoy point.
fn annoy_recall_at(points: &[(f64, f64)], target: f64) -> Option<f64> {
    let first = *points.first()?;
    let last = *points.last()?;
    if target < first.0 {
        return None;
    }
    if target >= last.0 {
        return Some(last.1);
    }
    for w in points.windows(2) {
        let (t0, r0) = w[0];
        let (t1, r1) = w[1];
        if target >= t0 && target <= t1 {
            if t1 <= t0 {
                return Some(r1);
            }
            let a = (target - t0) / (t1 - t0);
            return Some(r0 + a * (r1 - r0));
        }
    }
    Some(last.1)
}

fn print_iso_latency(rows: &[Row]) {
    let mut annoy: Vec<(f64, f64)> = rows
        .iter()
        .filter(|r| r.family == "annoy")
        .map(|r| (r.p50.as_secs_f64(), r.recall))
        .collect();
    annoy.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("  recall at equal p50 (Annoy interpolated onto our measured latency):");
    for ours in rows.iter().filter(|r| r.family == "rp_forest") {
        let t = ours.p50.as_secs_f64();
        match annoy_recall_at(&annoy, t) {
            Some(r) => println!(
                "    ours {}t/{:<28} {:>6.1} us r={:.3}  |  annoy at the same p50 r={:.3}  \
                 (recall {:+.3})",
                ours.trees,
                ours.setting,
                t * 1e6,
                ours.recall,
                r,
                ours.recall - r
            ),
            None => println!(
                "    ours {}t/{:<28} {:>6.1} us r={:.3}  |  faster than any Annoy point \
                 (its fastest is {:.1} us at r={:.3})",
                ours.trees,
                ours.setting,
                t * 1e6,
                ours.recall,
                annoy.first().map_or(f64::NAN, |p| p.0) * 1e6,
                annoy.first().map_or(f64::NAN, |p| p.1),
            ),
        }
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

        // Matched-budget pairs: both families are allowed to score the same
        // number of distinct ids, so a recall difference is a traversal-quality
        // difference, not a budget difference. `leaf 0` is the derived cap
        // (`dimension + 2`), i.e. what an unconfigured index builds with.
        let derived = 0;

        let rows = vec![
            measure_ours(ds, &truth, 10, derived, 100),
            measure_annoy(ds, &truth, 10, 100),
            measure_ours(ds, &truth, 10, derived, 490),
            measure_annoy(ds, &truth, 10, 490),
            measure_ours(ds, &truth, 10, derived, 980),
            measure_annoy(ds, &truth, 10, 980),
            measure_ours(ds, &truth, 50, derived, 4900),
            measure_annoy(ds, &truth, 50, 4900),
            measure_ours(ds, &truth, 10, derived, 1960),
            measure_annoy(ds, &truth, 10, 1960),
            measure_ours(ds, &truth, 10, derived, 4900),
            measure_annoy(ds, &truth, 10, 4900),
            measure_ours(ds, &truth, 10, derived, 9800),
            measure_annoy(ds, &truth, 10, 9800),
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
