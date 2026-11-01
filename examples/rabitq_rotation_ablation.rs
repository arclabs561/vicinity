#![allow(clippy::expect_used, clippy::unwrap_used)]

//! Recall@10 with and without the FHT/Kac rotation, at a fixed candidate budget.
//!
//! Builds IVF-RaBitQ over a seeded `n = 10_000, d = 96` set (isotropic,
//! clustered and anisotropic), then searches 200 held-out queries. The
//! candidate budget is held fixed: the same `nprobe` and the same k-means
//! partition in every arm, so each arm scans the same inverted lists and only
//! the rotation (and the error-bound pruning) vary.
//!
//! Three measurement modes separate the two things a rotation can change:
//!
//! * `estimate only` — compacted index, distances come from the codes alone
//!   with no bound pruning. This is the estimator's own ranking quality.
//! * `estimate + bound` — compacted index with error-bound pruning.
//! * `bound + exact rerank` — the shipped default: survivors are reranked with
//!   exact f32 distance, which can mask estimator error.
//!
//! ```sh
//! cargo run --release --features ivf_rabitq --example rabitq_rotation_ablation
//! ```
//!
//! Ground truth is exact cosine top-10 by brute force over the stored vectors.

use std::collections::HashSet;

use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams, RotationKind};

const DIM: usize = 96;
const N: usize = 10_000;
const K: usize = 10;
const NUM_QUERIES: usize = 200;
const NUM_CLUSTERS: usize = 64;
const SEED: u64 = 7;

/// SplitMix64, so the fixture is reproducible without depending on the RNG
/// crate's stream stability.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }

    fn vector(&mut self) -> Vec<f32> {
        (0..DIM).map(|_| self.uniform()).collect()
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Isotropic: every coordinate has the same distribution, so the data is
/// already invariant under rotation, as the sign model assumes.
fn uniform_set(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| normalize(rng.vector())).collect()
}

/// `NUM_CLUSTERS` centres with tight local noise, so residuals inside a cell
/// are small and quantizable, the regime RaBitQ is designed for.
fn clustered_set(n: usize, seed: u64, spread: f32) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let centers: Vec<Vec<f32>> = (0..NUM_CLUSTERS).map(|_| rng.vector()).collect();
    (0..n)
        .map(|i| {
            let center = &centers[i % NUM_CLUSTERS];
            let mut v = center.clone();
            for value in &mut v {
                *value += spread * rng.uniform();
            }
            normalize(v)
        })
        .collect()
}

/// Per-dimension variance decays geometrically across the axes, so a few
/// coordinates dominate the norm. This is the anisotropic structure real
/// embeddings have, and the case sign-only quantizers fail on without a
/// rotation that spreads the energy across dimensions.
fn anisotropic_set(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let scales: Vec<f32> = (0..DIM)
        .map(|i| {
            let t = i as f32 / (DIM - 1) as f32;
            1_000f32.powf(-t)
        })
        .collect();
    (0..n)
        .map(|_| {
            let v: Vec<f32> = scales.iter().map(|s| s * rng.uniform()).collect();
            normalize(v)
        })
        .collect()
}

fn exact_top_k(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v).map(|(a, b)| a * b).sum();
            (1.0 - dot, i as u32)
        })
        .collect();
    scored.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
    scored.truncate(k);
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().map(|(_, id)| id).collect()
}

struct Arm {
    recall: f64,
    scanned: f64,
    pruned: f64,
    reranks: f64,
}

fn measure(
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
    rotation: RotationKind,
    nprobe: usize,
    use_error_bound: bool,
    compact: bool,
) -> Arm {
    let params = IVFRaBitQParams {
        num_clusters: NUM_CLUSTERS,
        nprobe,
        total_bits: 4,
        seed: SEED,
        rotation,
        use_error_bound,
    };
    let mut index = IVFRaBitQIndex::new(DIM, params).expect("construct index");
    for (id, vector) in data.iter().enumerate() {
        index.add_slice(id as u32, vector).expect("add vector");
    }
    index.build().expect("build index");
    if compact {
        index.compact();
    }

    let mut hits = 0usize;
    let mut scanned = 0usize;
    let mut pruned = 0usize;
    let mut reranks = 0usize;
    for (query, expected) in queries.iter().zip(truth) {
        let (results, stats) = index.search_with_stats(query, K, nprobe).expect("search");
        let got: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
        hits += expected.iter().filter(|id| got.contains(id)).count();
        scanned += stats.candidates_scanned;
        pruned += stats.pruned_by_bound;
        reranks += stats.exact_reranks;
    }

    let denom = (queries.len() * K) as f64;
    let per_query = queries.len() as f64;
    Arm {
        recall: hits as f64 / denom,
        scanned: scanned as f64 / per_query,
        pruned: pruned as f64 / per_query,
        reranks: reranks as f64 / per_query,
    }
}

fn rotation_label(kind: RotationKind) -> &'static str {
    match kind {
        RotationKind::FhtKac => "FHT/Kac",
        RotationKind::Identity => "none",
    }
}

/// (label, pruning, compacted)
const MODES: [(&str, bool, bool); 3] = [
    ("estimate only", false, true),
    ("estimate + bound", true, true),
    ("bound + exact rerank", true, false),
];

fn main() {
    println!("n = {N}, d = {DIM}, k = {K}, queries = {NUM_QUERIES}, clusters = {NUM_CLUSTERS}");
    println!("codes: 4-bit RaBitQ; candidate budget fixed by nprobe (same k-means partition)\n");

    let datasets: Vec<(&str, Vec<Vec<f32>>, Vec<Vec<f32>>)> = vec![
        (
            "uniform (isotropic)",
            uniform_set(N, SEED),
            uniform_set(NUM_QUERIES, SEED ^ 0xA5A5),
        ),
        (
            "clustered",
            clustered_set(N, SEED, 0.25),
            clustered_set(NUM_QUERIES, SEED ^ 0xA5A5, 0.25),
        ),
        (
            "anisotropic",
            anisotropic_set(N, SEED),
            anisotropic_set(NUM_QUERIES, SEED ^ 0xA5A5),
        ),
    ];

    println!(
        "| dataset | rotation | nprobe | mode | recall@10 | candidates/q | pruned/q | exact reranks/q |"
    );
    println!("|---|---|---|---|---|---|---|---|");

    for (name, data, queries) in &datasets {
        let truth: Vec<Vec<u32>> = queries
            .iter()
            .map(|query| exact_top_k(data, query, K))
            .collect();
        for rotation in [RotationKind::FhtKac, RotationKind::Identity] {
            for nprobe in [4usize, NUM_CLUSTERS] {
                for (mode, use_error_bound, compact) in MODES {
                    let arm = measure(
                        data,
                        queries,
                        &truth,
                        rotation,
                        nprobe,
                        use_error_bound,
                        compact,
                    );
                    println!(
                        "| {name} | {} | {nprobe} | {mode} | {:.3} | {:.0} | {:.0} | {:.0} |",
                        rotation_label(rotation),
                        arm.recall,
                        arm.scanned,
                        arm.pruned,
                        arm.reranks
                    );
                }
            }
        }
    }
}
