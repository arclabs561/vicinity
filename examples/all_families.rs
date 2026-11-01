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
//! - `n = 10_000` base vectors, `k = 10`, `200` held-out queries. Width is a
//!   property of the domain (`96` unless stated) because real workloads differ
//!   in width as well as in structure.
//! - Seven synthetic domains, each standing in for a class of real workload and
//!   each generated deterministically from a fixed seed. All of them are
//!   L2-normalised, so cosine, angular and L2 ranking agree and every family can
//!   be fed the same vectors even though some are cosine-only (Vamana, EMG,
//!   RaBitQ) and some rank by L2 (IVF-PQ, k-d tree). See `DOMAINS` for the
//!   one-line description of each, and `measure_hardness` for the measured
//!   hardness reported beside it.
//! - Selection: `ANN_DATASETS=text,image` runs a comma-separated subset; unset
//!   runs all seven. `ANN_FVECS=<path>` replaces the synthetic domains with a
//!   `.fvecs` file (see `load_fvecs`).
//! - `ANN_JSONL=<path>` writes one JSON object per row in the same shape as
//!   `benchmarks/rows.jsonl`, so `scripts/benefits.py` reads the output.
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
//! being dropped silently. Pass a comma-separated family list as the first
//! argument to rerun just those families
//! (`cargo run --release --example all_families -- ivf_pq,hnsw`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

const EF_VALUES: &[usize] = &[10, 20, 50, 100, 128, 200];
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
    /// Resident-set growth across the build, in MiB (0 when unavailable).
    mem_mib: f64,
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
    /// One line naming the real workload this stands in for.
    description: &'static str,
    dim: usize,
    base: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    truth: Vec<Vec<u32>>,
    exact_p50_us: f64,
    /// Measured local intrinsic dimensionality (median over sampled queries).
    lid: f64,
    /// Lower and upper decile of the per-query LID, so a bimodal domain (part
    /// duplicate cloud, part uniform background) cannot hide behind the median.
    lid_low: f64,
    lid_high: f64,
    /// Measured separation: median distance to a random base point divided by
    /// the distance to the 10th neighbour. 1.0 means a query's neighbours are no
    /// closer than the rest of the set (nothing to find); large means a small
    /// cluster stands far out of the background and is easy to locate even when
    /// the inside of that cluster is high-dimensional.
    separation: f64,
}

struct Skipped {
    family: &'static str,
    reason: &'static str,
}

fn main() {
    // Optional positional filter: `all_families ivf_pq` or
    // `all_families hnsw,balltree` reruns just those families.
    let only: Vec<String> = std::env::args()
        .nth(1)
        .map(|arg| arg.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let datasets = build_datasets();
    let families: Vec<Family> = families()
        .into_iter()
        .filter(|f| only.is_empty() || only.iter().any(|name| name == f.name))
        .collect();
    let mut rows: Vec<Row> = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();

    println!("# All-family ANN benchmark");
    println!();
    println!("k={K} queries={N_QUERIES} seed=0x{SEED:016X} metric=cosine (all rows L2-normalised)");
    println!();
    println!("## Domains");
    println!();
    println!("| domain | n | dim | LID p50 [p10-p90] | separation | stands in for |");
    println!("| --- | ---: | ---: | ---: | ---: | --- |");
    for dataset in &datasets {
        println!(
            "| {} | {} | {} | {:.1} [{:.1}-{:.1}] | {:.2} | {} |",
            dataset.name,
            dataset.base.len(),
            dataset.dim,
            dataset.lid,
            dataset.lid_low,
            dataset.lid_high,
            dataset.separation,
            dataset.description
        );
    }
    println!();
    if datasets.iter().any(|dataset| dataset.name == "fvecs") {
        println!(
            "note: `fvecs` reuses the first {N_QUERIES} base rows as queries, so each query \
             matches itself and recall@10 has a {:.1} floor.",
            1.0 / K as f64
        );
        println!();
    }

    for dataset in &datasets {
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

    print_rows(&rows, &datasets);
    print_ranking(&rows, &datasets);
    print_at_recall(&rows, &datasets, 0.95);
    print_sweeps(&families);
    print_skipped(&failures);
    if let Ok(path) = std::env::var("ANN_JSONL") {
        write_jsonl(&path, &rows, &datasets);
    }
}

// ---------------------------------------------------------------------------
// Protocol: datasets, ground truth, timing
// ---------------------------------------------------------------------------

/// A synthetic domain: a name selectable through `ANN_DATASETS`, the real
/// workload it stands in for, and a deterministic generator.
struct Domain {
    name: &'static str,
    description: &'static str,
    generate: fn() -> Generated,
}

/// Generator output before ground truth, timing and hardness are attached.
struct Generated {
    dim: usize,
    base: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

/// Every domain the harness can generate. Hardness is measured after generation
/// and printed beside each name, never assumed from the generator's parameters.
const DOMAINS: &[Domain] = &[
    Domain {
        name: "uniform",
        description: "baseline: i.i.d. uniform in [-1,1]^96, no cluster structure",
        generate: generate_uniform,
    },
    Domain {
        name: "clustered16",
        description: "16 uniform centres, sigma=0.25: few strong clusters, mild LID",
        generate: generate_clustered16,
    },
    Domain {
        name: "text",
        description: "Zipf-weighted mixture of 200 topic centroids, anisotropic \
                      per-topic covariance: word/document embeddings",
        generate: generate_text,
    },
    Domain {
        name: "image",
        description: "deep hierarchy, 64 parents x 16 children at a small offset: \
                      deep-embedding locality",
        generate: generate_image,
    },
    Domain {
        name: "sparse",
        description: "d=768, ~2% nonzero, queries share terms with a document: \
                      lexical / sparse-lexical vectors",
        generate: generate_sparse,
    },
    Domain {
        name: "neardup",
        description: "90% small jitter around 64 seed documents, 10% uniform: \
                      near-duplicate-heavy corpora",
        generate: generate_neardup,
    },
    Domain {
        name: "lowdim",
        description: "d=8 points on a random 3-D subspace plus noise: \
                      intrinsically low-dimensional vectors",
        generate: generate_lowdim,
    },
];

fn build_datasets() -> Vec<Dataset> {
    if let Ok(path) = std::env::var("ANN_FVECS") {
        return vec![load_fvecs(&path)];
    }
    selected_domains()
        .into_iter()
        .map(|domain| assemble(domain, (domain.generate)()))
        .collect()
}

/// Parses `ANN_DATASETS` as a comma-separated list of domain names. Unset or
/// empty selects every domain; an unknown name is a hard error so a typo cannot
/// silently benchmark a different set.
fn selected_domains() -> Vec<&'static Domain> {
    let Ok(list) = std::env::var("ANN_DATASETS") else {
        return DOMAINS.iter().collect();
    };
    let chosen: Vec<&'static Domain> = list
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(
            |token| match DOMAINS.iter().find(|domain| domain.name == token) {
                Some(domain) => domain,
                None => {
                    let known: Vec<&str> = DOMAINS.iter().map(|domain| domain.name).collect();
                    eprintln!(
                        "unknown domain `{token}` in ANN_DATASETS; known domains: {}",
                        known.join(",")
                    );
                    std::process::exit(2);
                }
            },
        )
        .collect();
    if chosen.is_empty() {
        DOMAINS.iter().collect()
    } else {
        chosen
    }
}

/// Attaches ground truth, a like-for-like exact-scan p50 and measured hardness
/// to one generated domain.
fn assemble(domain: &'static Domain, generated: Generated) -> Dataset {
    let Generated { dim, base, queries } = generated;
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| exact_top_k(&base, q, K)).collect();
    let Hardness {
        lid,
        lid_low,
        lid_high,
        separation,
    } = measure_hardness(&base);

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
        name: domain.name,
        description: domain.description,
        dim,
        base,
        queries,
        truth,
        exact_p50_us: percentile(&mut lat.clone(), 50.0),
        lid,
        lid_low,
        lid_high,
        separation,
    }
}

// ---------------------------------------------------------------------------
// Domains
// ---------------------------------------------------------------------------

/// `uniform` and `clustered16` share one generator so the two original datasets
/// stay bit-identical to the pre-domain harness (same draws, same order).
fn generate_uniform() -> Generated {
    generate_blobs(SEED, None)
}

fn generate_clustered16() -> Generated {
    generate_blobs(SEED ^ 0xC1_0000, Some(0.25))
}

fn generate_blobs(seed: u64, spread: Option<f32>) -> Generated {
    let mut rng = StdRng::seed_from_u64(seed);
    let centres = sample_centres(&mut rng, spread.is_some());
    let base = sample_points(&mut rng, &centres, spread, N);
    let queries = sample_points(&mut rng, &centres, spread, N_QUERIES);
    Generated {
        dim: DIM,
        base,
        queries,
    }
}

/// Number of latent topics behind the `text` domain. Real embedding corpora have
/// hundreds of weak topics rather than a handful of strong ones.
const TEXT_TOPICS: usize = 200;
/// Zipf exponent over topic rank; small means the top topics hold a large share,
/// which is what makes a few topics dominate queries while the tail stays thin.
const TEXT_ZIPF: f64 = 1.1;
/// Spread inside a topic, relative to unit centroid norm.
const TEXT_SIGMA: f32 = 0.30;

/// `text`: many weak clusters with per-topic anisotropic covariance.
///
/// Real document/word embeddings show a Zipf-distributed topic mass and each
/// topic is stretched more along some directions than others. Here topics are
/// drawn from a Zipf(1.1) CDF over 200 centroids; a point is
/// `normalize(centroid + sigma * R (s_topic .* u))` where `u` is uniform noise,
/// `s_topic` is a per-topic per-axis log-normal scale vector and `R` is one
/// shared random basis, so each topic's covariance is anisotropic in its own
/// orientation.
fn generate_text() -> Generated {
    let dim = DIM;
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x7E_0000);

    let centroids: Vec<Vec<f32>> = (0..TEXT_TOPICS)
        .map(|_| unit_vector(&mut rng, dim))
        .collect();

    let mut cdf = Vec::with_capacity(TEXT_TOPICS);
    let mut acc = 0.0;
    for rank in 1..=TEXT_TOPICS {
        acc += 1.0 / (rank as f64).powf(TEXT_ZIPF);
        cdf.push(acc);
    }
    for weight in &mut cdf {
        *weight /= acc;
    }

    let scales: Vec<Vec<f32>> = (0..TEXT_TOPICS)
        .map(|_| {
            (0..dim)
                .map(|_| rng.random_range(-0.9f32..0.9f32).exp())
                .collect()
        })
        .collect();
    let basis: Vec<f32> = (0..dim * dim)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    // Keep the rotated noise at unit RMS norm so `TEXT_SIGMA` is the ratio of
    // within-topic spread to centroid separation.
    let latent_norm = (dim as f32 / 3.0).sqrt();

    let sample = |count: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| {
                let draw: f64 = rng.random_range(0.0..1.0);
                let topic = cdf.partition_point(|w| *w < draw).min(TEXT_TOPICS - 1);
                let mut latent = Vec::with_capacity(dim);
                for scale in &scales[topic] {
                    latent.push(scale * rng.random_range(-1.0f32..1.0));
                }
                let mixed: Vec<f32> = (0..dim)
                    .map(|i| {
                        let row = &basis[i * dim..(i + 1) * dim];
                        centroids[topic][i] + TEXT_SIGMA * dot(row, &latent) / latent_norm
                    })
                    .collect();
                normalize(mixed)
            })
            .collect()
    };

    let base = sample(N, &mut rng);
    let queries = sample(N_QUERIES, &mut rng);
    Generated { dim, base, queries }
}

/// Parents in the `image` hierarchy; each holds `IMAGE_CHILDREN` children.
const IMAGE_PARENTS: usize = 64;
const IMAGE_CHILDREN: usize = 16;
/// Child-centroid offset from its parent, and spread within a child.
const IMAGE_CHILD_OFFSET: f32 = 0.30;
const IMAGE_NOISE: f32 = 0.22;

/// `image`: a deep hierarchy. A query matches its child cluster first and its
/// parent cluster only after that, which is what deep-embedding locality looks
/// like — the nearest neighbours are never spread evenly over the set.
fn generate_image() -> Generated {
    let dim = DIM;
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x1A_0000);

    let parents: Vec<Vec<f32>> = (0..IMAGE_PARENTS)
        .map(|_| unit_vector(&mut rng, dim))
        .collect();
    let children: Vec<Vec<Vec<f32>>> = parents
        .iter()
        .map(|parent| {
            (0..IMAGE_CHILDREN)
                .map(|_| {
                    let offset = unit_vector(&mut rng, dim);
                    normalize(
                        parent
                            .iter()
                            .zip(offset.iter())
                            .map(|(c, o)| c + IMAGE_CHILD_OFFSET * o)
                            .collect::<Vec<f32>>(),
                    )
                })
                .collect()
        })
        .collect();

    let sample = |count: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| {
                let parent = rng.random_range(0..IMAGE_PARENTS);
                let child = rng.random_range(0..IMAGE_CHILDREN);
                let noise = unit_vector(rng, dim);
                normalize(
                    children[parent][child]
                        .iter()
                        .zip(noise.iter())
                        .map(|(c, n)| c + IMAGE_NOISE * n)
                        .collect::<Vec<f32>>(),
                )
            })
            .collect()
    };

    let base = sample(N, &mut rng);
    let queries = sample(N_QUERIES, &mut rng);
    Generated { dim, base, queries }
}

/// Width of the `sparse` domain. Lexical bag-of-words and sparse-lexical
/// embeddings are wide and almost empty.
const SPARSE_DIM: usize = 768;
/// Nonzeros per document: 15/768 is 1.95%.
const SPARSE_NNZ: usize = 15;
/// Fraction of a document's terms a query keeps, plus fresh terms it adds.
const SPARSE_QUERY_KEEP: f64 = 0.5;
const SPARSE_QUERY_EXTRA: usize = 5;

/// `sparse`: high dimension, ~2% nonzeros, cosine metric. Queries are a partial
/// bag of words taken from one document with a few unseen terms added, because
/// an independent sparse query shares no support with any document and every
/// distance is then equal.
fn generate_sparse() -> Generated {
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x5B_0000);

    let base: Vec<Vec<f32>> = (0..N)
        .map(|_| sparse_point(&mut rng, SPARSE_DIM, SPARSE_NNZ))
        .collect();

    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|_| {
            let document = &base[rng.random_range(0..base.len())];
            let mut vector = vec![0.0f32; SPARSE_DIM];
            let mut kept = 0usize;
            for (axis, value) in document.iter().enumerate() {
                if *value != 0.0 && rng.random_bool(SPARSE_QUERY_KEEP) {
                    vector[axis] = *value;
                    kept += 1;
                }
            }
            if kept == 0 {
                // Keep at least one shared term so the query is not pure noise.
                if let Some(axis) = document.iter().position(|v| *v != 0.0) {
                    vector[axis] = document[axis];
                }
            }
            let mut extra = 0usize;
            while extra < SPARSE_QUERY_EXTRA {
                let axis = rng.random_range(0..SPARSE_DIM);
                if vector[axis] == 0.0 {
                    vector[axis] = rng.random_range(0.2f32..1.0);
                    extra += 1;
                }
            }
            normalize(vector)
        })
        .collect();

    Generated {
        dim: SPARSE_DIM,
        base,
        queries,
    }
}

fn sparse_point(rng: &mut StdRng, dim: usize, nnz: usize) -> Vec<f32> {
    let mut vector = vec![0.0f32; dim];
    let mut placed = 0usize;
    while placed < nnz {
        let axis = rng.random_range(0..dim);
        if vector[axis] == 0.0 {
            // Term weights resemble tf-idf: positive, bounded, uneven.
            vector[axis] = rng.random_range(0.2f32..1.0);
            placed += 1;
        }
    }
    normalize(vector)
}

/// Seed documents the near-duplicate cloud is built around. The spec's "a base
/// point" is generalised to a small pool: with a single seed, 9000 points are
/// one tie set and recall@10 degenerates into "which of 9000 equals did you
/// return", which measures tie-breaking rather than cluster findability.
const NEARDUP_SEEDS: usize = 64;
const NEARDUP_SIGMA: f32 = 0.02;
/// Percentage of points that are jittered copies rather than fresh draws.
const NEARDUP_COPY_PCT: u32 = 90;

/// `neardup`: near-duplicate-heavy corpora. Nine in ten points are a small
/// jitter of one of 64 seed documents, so a query's neighbours are a tight
/// cluster of many almost-identical vectors and the tail is a uniform
/// background.
fn generate_neardup() -> Generated {
    let dim = DIM;
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x0D_0000);

    let seeds: Vec<Vec<f32>> = (0..NEARDUP_SEEDS)
        .map(|_| unit_vector(&mut rng, dim))
        .collect();

    let sample = |count: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| {
                if rng.random_range(0..100) < NEARDUP_COPY_PCT {
                    let seed = &seeds[rng.random_range(0..NEARDUP_SEEDS)];
                    let jitter = unit_vector(rng, dim);
                    normalize(
                        seed.iter()
                            .zip(jitter.iter())
                            .map(|(c, j)| c + NEARDUP_SIGMA * j)
                            .collect::<Vec<f32>>(),
                    )
                } else {
                    unit_vector(rng, dim)
                }
            })
            .collect()
    };

    let base = sample(N, &mut rng);
    let queries = sample(N_QUERIES, &mut rng);
    Generated { dim, base, queries }
}

/// Width and manifold rank of the `lowdim` domain.
const LOWDIM_DIM: usize = 8;
const LOWDIM_MANIFOLD: usize = 3;
const LOWDIM_NOISE: f32 = 0.02;

/// `lowdim`: intrinsically low-dimensional data, as when a downstream model's
/// outputs live near a low-rank subspace. Points are a random 3-D subspace of
/// `R^8`, sampled uniformly in its coefficients, plus small off-manifold noise.
/// Space-partitioning trees are expected to do well here and badly on `sparse`.
fn generate_lowdim() -> Generated {
    let dim = LOWDIM_DIM;
    let mut rng = StdRng::seed_from_u64(SEED ^ 0x10_0000);

    // Orthonormal basis of the 3-D subspace by Gram-Schmidt.
    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(LOWDIM_MANIFOLD);
    while basis.len() < LOWDIM_MANIFOLD {
        let mut candidate = unit_vector(&mut rng, dim);
        for previous in &basis {
            let projection = dot(&candidate, previous);
            for (value, p) in candidate.iter_mut().zip(previous.iter()) {
                *value -= projection * p;
            }
        }
        let norm = candidate.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for value in &mut candidate {
                *value /= norm;
            }
            basis.push(candidate);
        }
    }

    let sample = |count: usize, rng: &mut StdRng| -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| {
                let mut point = vec![0.0f32; dim];
                for direction in &basis {
                    let coefficient = rng.random_range(-1.0f32..1.0);
                    for (axis, value) in point.iter_mut().enumerate() {
                        *value += coefficient * direction[axis];
                    }
                }
                let noise = unit_vector(rng, dim);
                for (value, n) in point.iter_mut().zip(noise.iter()) {
                    *value += LOWDIM_NOISE * n;
                }
                normalize(point)
            })
            .collect()
    };

    let base = sample(N, &mut rng);
    let queries = sample(N_QUERIES, &mut rng);
    Generated { dim, base, queries }
}

/// `.fvecs` loading. Format assumption: each row is little-endian
/// `int32 dim` followed by `dim` little-endian `f32`, rows back to back, which
/// is what ann-benchmarks and the SIFT/GloVe/deep-image fvecs dumps use. The
/// first `N_QUERIES` rows are used as queries, so against an unmodified dump
/// those rows are also in the base set and each contributes a guaranteed self
/// hit — recall@10 therefore has a 0.1 floor on this path, which is stated in
/// the run header. Rows are L2-normalised so the inner-product ground truth and
/// every family's metric agree; feed an angular dataset (GloVe, deep-image) and
/// this is the ann-benchmarks ranking, feed a Euclidean one (SIFT) and the
/// ranking becomes cosine.
fn load_fvecs(path: &str) -> Dataset {
    let bytes = std::fs::read(path).unwrap_or_else(|err| {
        eprintln!("cannot read ANN_FVECS={path}: {err}");
        std::process::exit(2);
    });
    if bytes.len() < 4 {
        eprintln!("ANN_FVECS={path} is too short to hold a dimension");
        std::process::exit(2);
    }
    let dim = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if dim == 0 {
        eprintln!("ANN_FVECS={path} declares dim=0");
        std::process::exit(2);
    }
    let stride = 4 + 4 * dim;
    if !bytes.len().is_multiple_of(stride) {
        eprintln!(
            "ANN_FVECS={path}: {} bytes is not a whole number of dim={dim} rows (stride {stride})",
            bytes.len()
        );
        std::process::exit(2);
    }
    let rows = bytes.len() / stride;
    if rows <= N_QUERIES {
        eprintln!(
            "ANN_FVECS={path}: {rows} rows is not enough for {N_QUERIES} queries plus a base set"
        );
        std::process::exit(2);
    }

    let read_row = |row: usize| -> Vec<f32> {
        let start = row * stride + 4;
        (0..dim)
            .map(|axis| {
                let at = start + axis * 4;
                f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
            })
            .collect::<Vec<f32>>()
    };
    let base: Vec<Vec<f32>> = (0..rows).map(|row| normalize(read_row(row))).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES).map(|row| base[row].clone()).collect();
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| exact_top_k(&base, q, K)).collect();
    let Hardness {
        lid,
        lid_low,
        lid_high,
        separation,
    } = measure_hardness(&base);

    let mut lat = Vec::with_capacity(queries.len());
    for q in queries.iter().take(5) {
        let _ = exact_top_k(&base, q, K);
    }
    for q in &queries {
        let t = Instant::now();
        let _ = exact_top_k(&base, q, K);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }

    Dataset {
        name: "fvecs",
        description: "loaded from ANN_FVECS (first 200 rows also used as queries)",
        dim,
        base,
        queries,
        truth,
        exact_p50_us: percentile(&mut lat, 50.0),
        lid,
        lid_low,
        lid_high,
        separation,
    }
}

// ---------------------------------------------------------------------------
// Measured hardness
// ---------------------------------------------------------------------------

/// Measured hardness of a base set: two cheap, complementary statistics.
///
/// **LID** is the Levina-Bickel maximum-likelihood estimate of local intrinsic
/// dimensionality. For a point with neighbour distances `T_1 <= ... <= T_k`,
///
/// ```text
/// LID = -(k - 1) / sum_{i=1}^{k-1} ln(T_i / T_k)
/// ```
///
/// with `k = 10`, Euclidean distance on the L2-normalised vectors. It is
/// computed for 200 base points spread evenly across the set (the point itself
/// excluded) and reported as the median with its p10-p90 range. A local
/// neighbourhood that fills its ambient space gives a high LID; a manifold or a
/// cluster whose neighbours sit at a common radius gives a low one. Note that
/// LID is a *local* measure: the inside of a near-duplicate cloud can be as
/// high-dimensional as a uniform cloud even though the cloud itself is trivial
/// to locate. That gap is why the second statistic exists.
///
/// **Separation** is the median over the same 200 queries of
/// `median_j d(q, x_j) / T_10`: how far the 10th neighbour sits relative to a
/// typical base point. Near 1.0 means neighbours are indistinguishable from the
/// background (nothing to find); large means a query's cluster stands out of the
/// background and is easy to locate. Together, LID says how hard the *inside* of
/// a neighbourhood is and separation says how easy the neighbourhood is to
/// *reach*, which is the pair that actually predicts recall.
fn measure_hardness(base: &[Vec<f32>]) -> Hardness {
    const K_LID: usize = 10;
    const SAMPLES: usize = 200;

    let n = base.len();
    let samples = SAMPLES.min(n);
    if samples <= K_LID {
        return Hardness {
            lid: f64::NAN,
            lid_low: f64::NAN,
            lid_high: f64::NAN,
            separation: f64::NAN,
        };
    }

    let mut lids: Vec<f64> = Vec::with_capacity(samples);
    let mut separations: Vec<f64> = Vec::with_capacity(samples);
    for sample in 0..samples {
        let query_at = sample * n / samples;
        let query = &base[query_at];
        let mut scored: Vec<f32> = base
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != query_at)
            .map(|(_, vector)| squared_l2(vector, query))
            .collect();
        scored.select_nth_unstable_by(K_LID - 1, f32::total_cmp);

        let mut distances: Vec<f64> = scored[..K_LID]
            .iter()
            .map(|squared| (f64::from(*squared)).sqrt().max(f64::MIN_POSITIVE))
            .collect();
        distances.sort_by(f64::total_cmp);
        let tenth = distances[K_LID - 1];

        // Levina-Bickel MLE; the sum is negative unless every distance tied.
        let sum: f64 = distances[..K_LID - 1]
            .iter()
            .map(|distance| (distance / tenth).ln())
            .sum();
        if sum < -1e-12 {
            lids.push(-((K_LID - 1) as f64) / sum);
        }

        // Median distance to a random base point, on the same vector.
        let mid = (scored.len() - 1) / 2;
        scored.select_nth_unstable_by(mid, f32::total_cmp);
        let typical = (f64::from(scored[mid])).sqrt().max(f64::MIN_POSITIVE);
        separations.push(typical / tenth);
    }

    let mut sorted = lids.clone();
    let median = percentile(&mut sorted, 50.0);
    let mut sorted = lids.clone();
    let low = percentile(&mut sorted, 10.0);
    let mut sorted = lids.clone();
    let high = percentile(&mut sorted, 90.0);
    Hardness {
        lid: median,
        lid_low: low,
        lid_high: high,
        separation: percentile(&mut separations, 50.0),
    }
}

struct Hardness {
    lid: f64,
    lid_low: f64,
    lid_high: f64,
    separation: f64,
}

fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn sample_centres(rng: &mut StdRng, needed: bool) -> Vec<Vec<f32>> {
    if !needed {
        return Vec::new();
    }
    (0..CENTRES).map(|_| unit_vector(rng, DIM)).collect()
}

fn sample_points(
    rng: &mut StdRng,
    centres: &[Vec<f32>],
    spread: Option<f32>,
    count: usize,
) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let noise = sample_raw(rng, DIM);
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

fn sample_raw(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn unit_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    normalize(sample_raw(rng, dim))
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

/// Resident-set size of this process in MiB, measured with `ps` so it works on
/// macOS and Linux without a libc dependency.
fn rss_mib() -> Option<f64> {
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let kib: f64 = text.trim().parse().ok()?;
    Some(kib / 1024.0)
}

fn run_family(family: &Family, dataset: &Dataset) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();

    if family.rebuild {
        for &knob in family.values {
            let rss_before = rss_mib();
            let build_start = Instant::now();
            let mut index = (family.build)(&dataset.base, dataset.dim, knob)
                .map_err(|e| format!("build failed at {}={knob}: {e}", family.knob))?;
            let build_s = build_start.elapsed().as_secs_f64();
            let mem_mib = match (rss_before, rss_mib()) {
                (Some(before), Some(after)) => (after - before).max(0.0),
                _ => 0.0,
            };
            rows.push(row_for(
                family,
                dataset,
                knob,
                build_s,
                mem_mib,
                &mut *index,
            ));
        }
    } else {
        let rss_before = rss_mib();
        let build_start = Instant::now();
        let mut index = (family.build)(&dataset.base, dataset.dim, family.values[0])
            .map_err(|e| format!("build failed: {e}"))?;
        let build_s = build_start.elapsed().as_secs_f64();
        let mem_mib = match (rss_before, rss_mib()) {
            (Some(before), Some(after)) => (after - before).max(0.0),
            _ => 0.0,
        };
        for &knob in family.values {
            rows.push(row_for(
                family,
                dataset,
                knob,
                build_s,
                mem_mib,
                &mut *index,
            ));
        }
    }

    Ok(rows)
}

fn row_for(
    family: &Family,
    dataset: &Dataset,
    knob: usize,
    build_s: f64,
    mem_mib: f64,
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
        mem_mib,
        errors: stats.errors,
        first_error: stats.first_error,
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn print_rows(rows: &[Row], datasets: &[Dataset]) {
    for dataset in datasets {
        println!("## {}", dataset.name);
        println!();
        println!(
            "| family | knob | recall@10 | 1/ratio@10 | p50 us | p95 us | p99 us | QPS | build s | mem MiB |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for row in rows.iter().filter(|r| r.dataset == dataset.name) {
            println!(
                "| {} | {}={} | {:.3} | {:.2} | {:.1} | {:.1} | {:.1} | {:.0} | {:.2} | {:.1} |",
                row.family,
                row.knob,
                row.knob_value,
                row.recall,
                row.ratio,
                row.p50,
                row.p95,
                row.p99,
                row.qps,
                row.build_s,
                row.mem_mib
            );
            if row.errors > 0 {
                println!(
                    "| ^ {} query errors: {} | | | | | | | | | |",
                    row.errors,
                    row.first_error.as_deref().unwrap_or("unknown")
                );
            }
        }
        println!();
    }
}

/// One line per family: the best operating point by recall, tie-broken by p50.
fn print_ranking(rows: &[Row], datasets: &[Dataset]) {
    for dataset in datasets {
        println!("## {} ranking (best recall, tie-break p50)", dataset.name);
        println!();
        println!("| rank | family | at | recall@10 | p50 us | build s |");
        println!("| ---: | --- | --- | ---: | ---: | ---: |");
        let mut best: Vec<&Row> = Vec::new();
        for row in rows.iter().filter(|r| r.dataset == dataset.name) {
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
fn print_at_recall(rows: &[Row], datasets: &[Dataset], target: f64) {
    for dataset in datasets {
        println!(
            "## {} at recall@10 >= {target:.2} (fastest p50; `*` = target not reached)",
            dataset.name
        );
        println!();
        println!("| family | at | recall@10 | p50 us | QPS | 1/ratio@10 |");
        println!("| --- | --- | ---: | ---: | ---: | ---: |");
        let mut chosen: Vec<(bool, &Row)> = Vec::new();
        for row in rows.iter().filter(|r| r.dataset == dataset.name) {
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

/// Writes one JSON object per row, in the field names `scripts/benefits.py`
/// reads, so `ANN_JSONL=rows.jsonl` feeds the benefit profile directly. `dim`
/// and `lid` are extra context the script ignores.
fn write_jsonl(path: &str, rows: &[Row], datasets: &[Dataset]) {
    use std::fmt::Write as _;

    /// JSON has no NaN/Infinity, and `benefits.py` would fail to parse them.
    fn num(value: f64, precision: usize) -> String {
        if value.is_finite() {
            format!("{value:.precision$}")
        } else {
            "null".to_string()
        }
    }

    let mut out = String::new();
    for row in rows {
        let dataset = datasets.iter().find(|d| d.name == row.dataset);
        let dim = dataset.map_or(0, |d| d.dim);
        let lid = dataset.map_or(f64::NAN, |d| d.lid);
        let _ = writeln!(
            out,
            "{{\"family\":\"{}\",\"dataset\":\"{}\",\"knob\":\"{}\",\"knob_value\":{},\
             \"recall\":{},\"p50_us\":{},\"p95_us\":{},\"p99_us\":{},\
             \"qps\":{},\"build_s\":{},\"mem_mib\":{},\
             \"ratio\":{},\"dim\":{},\"lid\":{}}}",
            row.family,
            row.dataset,
            row.knob,
            row.knob_value,
            num(row.recall, 4),
            num(row.p50, 1),
            num(row.p95, 1),
            num(row.p99, 1),
            num(row.qps, 1),
            num(row.build_s, 4),
            num(row.mem_mib, 1),
            num(row.ratio, 4),
            dim,
            num(lid, 3),
        );
    }
    match std::fs::write(path, out) {
        Ok(()) => println!(
            "wrote {} rows to {path} (scripts/benefits.py {path})",
            rows.len()
        ),
        Err(err) => eprintln!("cannot write ANN_JSONL={path}: {err}"),
    }
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
