#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]
#![cfg(all(feature = "filtered_graph", feature = "hnsw"))]
//! Filtered-ANN selectivity sweep against exact filtered ground truth.
//!
//! A single filter selectivity is not a filtered benchmark. This test follows
//! the protocol used by ACORN (Patel et al., SIGMOD 2024) and by the
//! filtered-ANN benchmark survey (arXiv 2507.21989 / PVLDB 19(10) 2026):
//!
//! * sweep selectivity from ~20% down to 0.01%,
//! * score recall@10 against **exact filtered ground truth** (brute force over
//!   only the points that satisfy the predicate), never against unfiltered
//!   ground truth,
//! * report latency next to recall.
//!
//! Two datasets from one seeded generator: uniform unit vectors and a
//! 40-cluster variant. Attributes are assigned independently of geometry (the
//! case filters are hardest on, and the setting ACORN evaluates) with exact
//! per-value counts: `category` 100 values x 200 points, `subcategory` 2000 x
//! 10, `bucket` 10000 x 2. The six selectivities come from category membership
//! (20/5/1%), subcategory (0.5/0.1%) and bucket (0.01%), so the match counts
//! are exact — 4000/1000/200/100/20/2 — rather than approximate.
//!
//! Strategies compared over one predicate on one dataset:
//!
//! * `prefilter_exact` — evaluate the predicate, then brute-force the matching
//!   set. This is the exact answer, so recall is 1.0 by construction; it is the
//!   latency floor and the ground-truth generator.
//! * `postfilter_hnsw` — unfiltered HNSW beam search at the shared `ef`,
//!   keep the results that match, stop at `k`. Single shot, no widening retry.
//! * `acorn_hnsw` / `acorn_hnsw_hop128` — HNSW's ACORN traversal
//!   (`search_acorn`), which expands second-hop neighbors when the predicate
//!   removes first-hop edges. Predicate-aware. The second row quadruples the
//!   two-hop fan-out at the same beam width, which separates "needs a bigger
//!   expansion budget" from "structurally cannot reach the qualifying set".
//! * `filtered_graph` — the crate's `FilteredGraphIndex`. Note its internal
//!   strategy: `search_filtered` post-filters a beam above 10% selectivity and
//!   falls back to an exact scan of the inverted-list matches below 10%. It is
//!   therefore *not* predicate-aware; the test prints which branch ran.
//!
//! Run with:
//!
//! ```bash
//! cargo test --release --test filtered_recall --features filtered_graph -- --nocapture
//! ```

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rand::prelude::*;
use vicinity::distance::cosine_distance_normalized;

const N: usize = 20_000;
const DIM: usize = 96;
/// Queries per selectivity level. 40 keeps the sweep's wall time tractable:
/// `filtered_graph`'s beam costs ~1.1 s/query at ef = 1600 (see EF_SEARCH).
const QUERIES: usize = 40;
/// Queries for the unfiltered diagnostic baseline (recall + p50 only).
const BASELINE_QUERIES: usize = 16;
const K: usize = 10;
/// Search width shared by every graph strategy so the comparison is ef-matched.
///
/// 1600 is not arbitrary. At ef = 400 the crate's own `filtered_graph` scores
/// only 0.61 recall@10 at 20% selectivity on this uniform 96-d generator,
/// which cannot clear the protocol floor of 0.8; its beam needs a wider budget
/// to cover the filtered neighborhood. Measured on that dataset with this
/// generator: 0.22 / 0.35 / 0.61 / 0.87 / 0.985 at ef = 100/200/400/800/1600
/// (reproducible by changing this constant). The 20% rows below therefore use
/// the smallest budget at which the filtered index clears the floor, and every
/// strategy gets the same budget.
const EF_SEARCH: usize = 1600;

const CATEGORY_CARD: usize = 100;
const SUBCATEGORY_CARD: usize = 2_000;
const BUCKET_CARD: usize = 10_000;

/// Attribute the predicate selects on. Cardinalities are chosen so each
/// selectivity is representable with *exact* match counts at n = 20k:
/// category = 200 pts/value, subcategory = 10, bucket = 2.
#[derive(Clone, Copy, Debug)]
enum Attr {
    Category,
    Subcategory,
    Bucket,
}

#[derive(Clone, Copy, Debug)]
struct Level {
    attr: Attr,
    /// Number of attribute values the predicate admits (`In` list length).
    values: usize,
    /// Nominal selectivity, printed in the table.
    selectivity: f64,
}

const LEVELS: [Level; 6] = [
    Level {
        attr: Attr::Category,
        values: 20,
        selectivity: 0.20,
    },
    Level {
        attr: Attr::Category,
        values: 5,
        selectivity: 0.05,
    },
    Level {
        attr: Attr::Category,
        values: 1,
        selectivity: 0.01,
    },
    Level {
        attr: Attr::Subcategory,
        values: 10,
        selectivity: 0.005,
    },
    Level {
        attr: Attr::Subcategory,
        values: 2,
        selectivity: 0.001,
    },
    Level {
        attr: Attr::Bucket,
        values: 1,
        selectivity: 0.0001,
    },
];

impl Attr {
    fn name(self) -> &'static str {
        match self {
            Attr::Category => "category",
            Attr::Subcategory => "subcategory",
            Attr::Bucket => "bucket",
        }
    }
}

// ── Dataset ───────────────────────────────────────────────────────────────────

struct Dataset {
    name: &'static str,
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    category: Vec<i64>,
    subcategory: Vec<i64>,
    bucket: Vec<i64>,
}

impl Dataset {
    fn labels(&self, attr: Attr) -> &[i64] {
        match attr {
            Attr::Category => &self.category,
            Attr::Subcategory => &self.subcategory,
            Attr::Bucket => &self.bucket,
        }
    }

    /// Predicate membership, evaluated independently of every index.
    fn mask(&self, level: Level) -> Vec<bool> {
        self.labels(level.attr)
            .iter()
            .map(|&v| v < level.values as i64)
            .collect()
    }

    /// Same predicate expressed for `filtered_graph`.
    fn fg_filter(level: Level) -> vicinity::filtered_graph::Filter {
        use vicinity::filtered_graph::{AttrValue, Filter, Predicate};
        let values = (0..level.values)
            .map(|v| AttrValue::Int(v as i64))
            .collect();
        Filter::Clause(Predicate::In(level.attr.name().to_string(), values))
    }

    /// Same predicate expressed for `hnsw`'s metadata filtering.
    fn hnsw_filter(level: Level) -> vicinity::filtering::MetadataFilter {
        use vicinity::filtering::{MetadataFilter, MetadataValue};
        let clauses = (0..level.values)
            .map(|v| MetadataFilter::Equals {
                field: level.attr.name().to_string(),
                value: MetadataValue::Int(v as i64),
            })
            .collect();
        MetadataFilter::Or(clauses)
    }

    /// Structured metadata for one point (as stored in the HNSW index).
    fn metadata(&self, id: usize) -> HashMap<String, vicinity::filtering::MetadataValue> {
        use vicinity::filtering::MetadataValue;
        let mut m = HashMap::new();
        m.insert(
            "category".to_string(),
            MetadataValue::Int(self.category[id]),
        );
        m.insert(
            "subcategory".to_string(),
            MetadataValue::Int(self.subcategory[id]),
        );
        m.insert("bucket".to_string(), MetadataValue::Int(self.bucket[id]));
        m
    }
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn unit_noise(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect();
    normalize(&mut v);
    v
}

/// Balanced attribute labels: coprime strides give every value exactly
/// n/cardinality members without correlating the attributes with each other.
fn balanced_labels(n: usize, card: usize, stride: usize) -> Vec<i64> {
    (0..n).map(|i| ((i * stride) % card) as i64).collect()
}

fn build_dataset(kind: &str, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let vectors: Vec<Vec<f32>> = match kind {
        "uniform" => (0..N)
            .map(|_| {
                let mut v: Vec<f32> = (0..DIM)
                    .map(|_| rng.random_range(-1.0f32..1.0f32))
                    .collect();
                normalize(&mut v);
                v
            })
            .collect(),
        "clustered" => {
            let clusters = 40usize;
            let spread = 0.18f32;
            let centers: Vec<Vec<f32>> = (0..clusters)
                .map(|_| {
                    let mut c: Vec<f32> = (0..DIM)
                        .map(|_| rng.random_range(-1.0f32..1.0f32))
                        .collect();
                    normalize(&mut c);
                    c
                })
                .collect();
            (0..N)
                .map(|i| {
                    let center = &centers[i % clusters];
                    let mut v: Vec<f32> = (0..DIM)
                        .map(|d| center[d] + spread * rng.random_range(-1.0f32..1.0f32))
                        .collect();
                    normalize(&mut v);
                    v
                })
                .collect()
        }
        other => panic!("unknown dataset {other}"),
    };

    // Queries are drawn from the same distribution as the data.
    let queries: Vec<Vec<f32>> = match kind {
        "uniform" => (0..QUERIES).map(|_| unit_noise(&mut rng, DIM)).collect(),
        _ => {
            let clusters = 40usize;
            let spread = 0.18f32;
            // Rebuild the same centers deterministically by replaying the stream.
            let mut replay = StdRng::seed_from_u64(seed);
            let centers: Vec<Vec<f32>> = (0..clusters)
                .map(|_| {
                    let mut c: Vec<f32> = (0..DIM)
                        .map(|_| replay.random_range(-1.0f32..1.0f32))
                        .collect();
                    normalize(&mut c);
                    c
                })
                .collect();
            (0..QUERIES)
                .map(|q| {
                    let center = &centers[q % clusters];
                    let mut v: Vec<f32> = (0..DIM)
                        .map(|d| center[d] + spread * replay.random_range(-1.0f32..1.0f32))
                        .collect();
                    normalize(&mut v);
                    v
                })
                .collect()
        }
    };

    Dataset {
        name: if kind == "uniform" {
            "uniform"
        } else {
            "clustered"
        },
        vectors,
        queries,
        category: balanced_labels(N, CATEGORY_CARD, 7919),
        subcategory: balanced_labels(N, SUBCATEGORY_CARD, 104_729),
        bucket: balanced_labels(N, BUCKET_CARD, 1_299_709),
    }
}

// ── Ground truth and measurement ──────────────────────────────────────────────

/// Exact k-NN over only the points satisfying the predicate, on unit vectors.
/// Ties break by ascending id so the reference is deterministic.
fn exact_filtered_gt(query: &[f32], vectors: &[Vec<f32>], mask: &[bool], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m)
        .map(|(id, _)| (cosine_distance_normalized(query, &vectors[id]), id as u32))
        .collect();
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.truncate(k);
    scored.into_iter().map(|(_, id)| id).collect()
}

struct Row {
    recall: f64,
    p50_us: f64,
    mean_returned: f64,
}

impl Row {
    /// Throughput implied by the median latency. Deliberately not `1/mean`:
    /// with a few dozen queries a single scheduler hiccup moves the mean by an
    /// order of magnitude and says nothing about the index.
    fn qps_at_p50(&self) -> f64 {
        1e6 / self.p50_us.max(f64::MIN_POSITIVE)
    }
}

fn measure(
    queries: &[Vec<f32>],
    gts: &[Vec<u32>],
    mut search: impl FnMut(&[f32]) -> Vec<(u32, f32)>,
) -> Row {
    let mut total_recall = 0.0f64;
    let mut returned = 0usize;
    let mut latencies = Vec::with_capacity(queries.len());

    for (query, gt) in queries.iter().zip(gts) {
        let start = Instant::now();
        let results = search(query);
        latencies.push(start.elapsed().as_secs_f64() * 1e6);
        returned += results.len();

        let expected: HashSet<u32> = gt.iter().copied().collect();
        let found: HashSet<u32> = results.into_iter().map(|(id, _)| id).collect();
        // Divide by the true number of qualifying points, not by k: at 0.01%
        // selectivity there are only two matches, so recall@10 has a
        // denominator of 2.
        let denom = expected.len().max(1) as f64;
        total_recall += expected.intersection(&found).count() as f64 / denom;
    }

    latencies.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = latencies.len();
    Row {
        recall: total_recall / n as f64,
        p50_us: latencies[n / 2],
        mean_returned: returned as f64 / n as f64,
    }
}

// ── Indexes ───────────────────────────────────────────────────────────────────

fn build_filtered_graph(data: &Dataset) -> vicinity::filtered_graph::FilteredGraphIndex {
    use vicinity::filtered_graph::{AttrValue, FilteredGraphIndex, FilteredGraphParams};

    let params = FilteredGraphParams {
        max_degree: 32,
        ef_construction: 64,
        ef_search: EF_SEARCH,
        alpha: 1.2,
    };
    let mut index = FilteredGraphIndex::new(DIM, params).unwrap();
    for (id, vector) in data.vectors.iter().enumerate() {
        let mut attrs = HashMap::new();
        attrs.insert("category".to_string(), AttrValue::Int(data.category[id]));
        attrs.insert(
            "subcategory".to_string(),
            AttrValue::Int(data.subcategory[id]),
        );
        attrs.insert("bucket".to_string(), AttrValue::Int(data.bucket[id]));
        index.add_slice(id as u32, vector, attrs).unwrap();
    }
    index.build().unwrap();
    index
}

fn build_hnsw(data: &Dataset) -> vicinity::hnsw::HNSWIndex {
    let mut index = vicinity::hnsw::HNSWIndex::with_filtering(DIM, 16, 32, "category").unwrap();
    for (id, vector) in data.vectors.iter().enumerate() {
        index.add_slice(id as u32, vector).unwrap();
        index.add_metadata(id as u32, data.metadata(id)).unwrap();
    }
    index.build().unwrap();
    index
}

// ── Test ──────────────────────────────────────────────────────────────────────

/// Exact k-NN over the whole dataset (no predicate), for the unfiltered
/// baseline that separates "the graph is weak" from "the filter hurts".
fn exact_gt(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mask = vec![true; vectors.len()];
    exact_filtered_gt(query, vectors, &mask, k)
}

fn main_sweep() {
    let mut findings: Vec<String> = Vec::new();
    let mut floors: Vec<(String, f64, f64)> = Vec::new();
    // (dataset, strategy, selectivity, recall) for the collapse analysis.
    let mut curve: Vec<(&'static str, &'static str, f64, f64)> = Vec::new();
    // p50 latency at 20% selectivity, for the filtered_graph vs post-filter ratio.
    let mut p50_at_20: Vec<(&'static str, &'static str, f64)> = Vec::new();

    for (kind, seed) in [("uniform", 0x5eed_u64), ("clustered", 0xc105_u64)] {
        let data = build_dataset(kind, seed);

        let build_start = Instant::now();
        let fg = build_filtered_graph(&data);
        let fg_build_s = build_start.elapsed().as_secs_f64();
        let build_start = Instant::now();
        let hnsw = build_hnsw(&data);
        let hnsw_build_s = build_start.elapsed().as_secs_f64();

        assert_eq!(fg.len(), N);
        assert_eq!(hnsw.num_active(), N);

        // Predicate exactness. The ground truth, the post-filter and the
        // pre-filter all use `mask`; the indexes use their own predicate
        // encodings. Prove they are the same predicate before comparing.
        for level in LEVELS {
            let mask = data.mask(level);
            let expected = mask.iter().filter(|&&m| m).count();
            let fg_filter = Dataset::fg_filter(level);
            let hnsw_filter = Dataset::hnsw_filter(level);

            for (id, &expected_match) in mask.iter().enumerate() {
                if id % 97 != 0 {
                    continue; // bounded sample; structural check follows
                }
                assert_eq!(
                    hnsw_filter.matches(&data.metadata(id)),
                    expected_match,
                    "metadata predicate disagrees with mask at id {id} ({level:?})"
                );
            }

            // Below 10% selectivity `filtered_graph` answers from its
            // inverted-list matches, so k = n returns that set verbatim.
            // Equality with the mask set proves its inverted index resolves
            // the predicate exactly as the mask does.
            if level.selectivity < 0.10 {
                let probe = &data.queries[0];
                let got: HashSet<u32> = fg
                    .search_filtered(probe, N, &fg_filter)
                    .unwrap()
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                let want: HashSet<u32> = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &m)| m)
                    .map(|(id, _)| id as u32)
                    .collect();
                assert_eq!(got.len(), expected, "match count mismatch ({level:?})");
                assert_eq!(got, want, "filtered_graph predicate mismatch ({level:?})");
            }
        }

        println!(
            "\n=== filtered recall sweep: {} (n={N}, d={DIM}, queries={QUERIES}, k={K}, ef_search={EF_SEARCH}) ===",
            data.name
        );
        println!(
            "index build: filtered_graph {:.2}s, hnsw {:.2}s",
            fg_build_s, hnsw_build_s
        );

        // Unfiltered baselines: how good is each graph with no predicate at all?
        let mut unfiltered_gt: Vec<Vec<u32>> = Vec::with_capacity(BASELINE_QUERIES);
        let mut baseline_queries: Vec<Vec<f32>> = Vec::with_capacity(BASELINE_QUERIES);
        for q in data.queries.iter().take(BASELINE_QUERIES) {
            baseline_queries.push(q.clone());
            unfiltered_gt.push(exact_gt(q, &data.vectors, K));
        }
        let fg_unfiltered = measure(&baseline_queries, &unfiltered_gt, |q| {
            fg.search_with_ef(q, K, EF_SEARCH).unwrap()
        });
        let hnsw_unfiltered = measure(&baseline_queries, &unfiltered_gt, |q| {
            hnsw.search(q, K, EF_SEARCH).unwrap()
        });
        println!(
            "unfiltered baseline @ef={EF_SEARCH}: filtered_graph recall={:.3} p50={:.1}us | hnsw recall={:.3} p50={:.1}us",
            fg_unfiltered.recall, fg_unfiltered.p50_us, hnsw_unfiltered.recall, hnsw_unfiltered.p50_us
        );

        println!(
            "{:<9} {:>7} {:<19} {:>8} {:>10} {:>9} {:>8}",
            "sel", "gt_size", "strategy", "recall@10", "p50_us", "qps@p50", "returned"
        );
        println!("{}", "-".repeat(76));

        for level in LEVELS {
            let mask = data.mask(level);
            let matching = mask.iter().filter(|&&m| m).count();
            let gts: Vec<Vec<u32>> = data
                .queries
                .iter()
                .map(|q| exact_filtered_gt(q, &data.vectors, &mask, K))
                .collect();
            let gt_size = gts[0].len();

            let fg_filter = Dataset::fg_filter(level);
            let hnsw_filter = Dataset::hnsw_filter(level);

            // 1. pre-filter: evaluate the predicate, brute-force its matches.
            // Match collection is inside the timed region because any
            // pre-filtering engine pays it (filtered_graph's exact branch
            // materializes a HashSet from its inverted list there too).
            let pre = measure(&data.queries, &gts, |q| {
                let ids: Vec<u32> = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &m)| m)
                    .map(|(id, _)| id as u32)
                    .collect();
                let mut scored: Vec<(u32, f32)> = ids
                    .iter()
                    .map(|&id| {
                        (
                            id,
                            cosine_distance_normalized(q, &data.vectors[id as usize]),
                        )
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                scored.truncate(K);
                scored
            });

            // 2. post-filter: one unfiltered beam of EF_SEARCH, keep matches.
            // No widening retry, which is the naive baseline.
            let post = measure(&data.queries, &gts, |q| {
                hnsw.search(q, EF_SEARCH, EF_SEARCH)
                    .unwrap()
                    .into_iter()
                    .filter(|(id, _)| mask[*id as usize])
                    .take(K)
                    .collect::<Vec<(u32, f32)>>()
            });

            // 3. predicate-aware: ACORN second-hop traversal, two budgets.
            let acorn_config = vicinity::hnsw::AcornConfig {
                enable_two_hop: true,
                two_hop_threshold: 0.3,
                max_two_hop_neighbors: 32,
                ef_search: EF_SEARCH,
            };
            let acorn = measure(&data.queries, &gts, |q| {
                hnsw.search_acorn(q, K, &acorn_config, &hnsw_filter)
                    .unwrap()
            });
            // Same budget, 4x the two-hop fan-out: is selectivity loss a hop
            // budget problem or a structural one?
            let acorn_hop128_config = vicinity::hnsw::AcornConfig {
                enable_two_hop: true,
                two_hop_threshold: 0.3,
                max_two_hop_neighbors: 128,
                ef_search: EF_SEARCH,
            };
            let acorn_hop128 = measure(&data.queries, &gts, |q| {
                hnsw.search_acorn(q, K, &acorn_hop128_config, &hnsw_filter)
                    .unwrap()
            });

            // 4. the crate's FilteredGraphIndex (auto strategy).
            let filtered_graph = measure(&data.queries, &gts, |q| {
                fg.search_filtered(q, K, &fg_filter).unwrap()
            });

            let branch = if matching as f64 / N as f64 >= 0.10 {
                "post-filter beam"
            } else {
                "exact scan"
            };
            let sel_label = format!("{:.2}%", level.selectivity * 100.0);
            for (name, row) in [
                ("prefilter_exact", &pre),
                ("postfilter_hnsw", &post),
                ("acorn_hnsw", &acorn),
                ("acorn_hnsw_hop128", &acorn_hop128),
                ("filtered_graph", &filtered_graph),
            ] {
                println!(
                    "{:<9} {:>7} {:<19} {:>8.3} {:>10.1} {:>9.0} {:>8.1}{}",
                    sel_label,
                    gt_size,
                    name,
                    row.recall,
                    row.p50_us,
                    row.qps_at_p50(),
                    row.mean_returned,
                    if name == "filtered_graph" {
                        format!("   [{branch}]")
                    } else {
                        String::new()
                    }
                );
            }

            floors.push((
                format!("{}@{sel_label}", data.name),
                level.selectivity,
                filtered_graph.recall,
            ));
            for (name, row) in [
                ("prefilter_exact", &pre),
                ("postfilter_hnsw", &post),
                ("acorn_hnsw", &acorn),
                ("acorn_hnsw_hop128", &acorn_hop128),
                ("filtered_graph", &filtered_graph),
            ] {
                curve.push((data.name, name, level.selectivity, row.recall));
                if (level.selectivity - 0.20).abs() < 1e-12 {
                    p50_at_20.push((data.name, name, row.p50_us));
                }
            }
        }
    }

    for dataset in ["uniform", "clustered"] {
        let p50 = |strategy: &str| {
            p50_at_20
                .iter()
                .find(|(d, s, _)| *d == dataset && *s == strategy)
                .map(|(_, _, p)| *p)
                .unwrap_or(0.0)
        };
        let (fg, post) = (p50("filtered_graph"), p50("postfilter_hnsw"));
        if fg > 20.0 * post && post > 0.0 {
            findings.push(format!(
                "{dataset}: filtered_graph p50 at 20% selectivity is {:.0}x postfilter_hnsw at the same ef ({:.0}us vs {:.0}us); beam_search re-sorts its whole candidate vector on every frontier pop, so cost tracks visited x ef",
                fg / post,
                fg,
                post
            ));
        }
    }

    println!("\n=== findings (printed, not asserted) ===");
    for finding in &findings {
        println!("* {finding}");
    }

    // Where does each strategy's recall collapse? Levels are ordered from high
    // to low selectivity, so the first level under 0.5 after a level at or
    // above 0.5 is the collapse point.
    let collapse_floor = 0.5;
    println!("\nrecall@{K} collapse point per strategy (threshold {collapse_floor}):");
    for (dataset, strategy) in [
        ("uniform", "prefilter_exact"),
        ("uniform", "postfilter_hnsw"),
        ("uniform", "acorn_hnsw"),
        ("uniform", "filtered_graph"),
        ("clustered", "prefilter_exact"),
        ("clustered", "postfilter_hnsw"),
        ("clustered", "acorn_hnsw"),
        ("clustered", "filtered_graph"),
    ] {
        let mut points: Vec<(f64, f64)> = curve
            .iter()
            .filter(|(d, s, _, _)| *d == dataset && *s == strategy)
            .map(|(_, _, sel, recall)| (*sel, *recall))
            .collect();
        points.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut reported = false;
        for window in points.windows(2) {
            let ((hi_sel, hi_recall), (lo_sel, lo_recall)) = (window[0], window[1]);
            if hi_recall >= collapse_floor && lo_recall < collapse_floor {
                println!(
                    "  {dataset:<9} {strategy:<18} collapses between {:.2}% (recall {hi_recall:.3}) and {:.2}% (recall {lo_recall:.3})",
                    hi_sel * 100.0,
                    lo_sel * 100.0
                );
                reported = true;
                break;
            }
        }
        if !reported {
            let lowest = points.last().copied().unwrap_or((0.0, 0.0));
            if points.first().is_some_and(|(_, r)| *r < collapse_floor) {
                println!(
                    "  {dataset:<9} {strategy:<18} below {collapse_floor} from the top of the sweep (recall {:.3} at {:.2}%)",
                    points[0].1,
                    points[0].0 * 100.0
                );
            } else {
                println!(
                    "  {dataset:<9} {strategy:<18} no collapse: recall {:.3} at the lowest selectivity ({:.2}%)",
                    lowest.1,
                    lowest.0 * 100.0
                );
            }
        }
    }

    // Non-monotonic recall is itself a finding: a strategy that recovers at
    // lower selectivity has a traversal effect, not a coverage limit.
    for (dataset, strategy) in [
        ("uniform", "acorn_hnsw"),
        ("clustered", "acorn_hnsw"),
        ("uniform", "postfilter_hnsw"),
        ("clustered", "postfilter_hnsw"),
    ] {
        let mut points: Vec<(f64, f64)> = curve
            .iter()
            .filter(|(d, s, _, _)| *d == dataset && *s == strategy)
            .map(|(_, _, sel, recall)| (*sel, *recall))
            .collect();
        points.sort_by(|a, b| b.0.total_cmp(&a.0));
        for window in points.windows(2) {
            let (hi_recall, lo_recall) = (window[0].1, window[1].1);
            if lo_recall > hi_recall + 0.05 {
                println!(
                    "  {dataset:<9} {strategy:<18} recall rises from {hi_recall:.3} at {:.2}% to {lo_recall:.3} at {:.2}% (non-monotonic in selectivity)",
                    window[0].0 * 100.0,
                    window[1].0 * 100.0
                );
            }
        }
    }

    // Loose floors only, so a broken index fails while measurement noise does
    // not. Applied to the crate's `filtered_graph`, the index under test.
    for (label, selectivity, recall) in &floors {
        if (*selectivity - 0.20).abs() < 1e-12 {
            assert!(
                *recall >= 0.8,
                "{label}: filtered_graph recall@10 at 20% selectivity = {recall:.3}, expected >= 0.8"
            );
        }
        if (*selectivity - 0.001).abs() < 1e-12 {
            assert!(
                *recall > 0.0,
                "{label}: filtered_graph recall@10 is zero at 0.1% selectivity"
            );
        }
    }
}

#[test]
fn filtered_recall_selectivity_sweep() {
    main_sweep();
}
