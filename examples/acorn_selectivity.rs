#![allow(clippy::expect_used)]
//! ACORN filtered-search selectivity sweep.
//!
//! Builds a deterministic mutual kNN graph over synthetic normalized vectors,
//! then measures ACORN recall, QPS, latency tails, and 2-hop branch counters
//! across filter selectivity levels.
//!
//! ```bash
//! cargo run --example acorn_selectivity --release --features hnsw
//! cargo run --example acorn_selectivity --release --features hnsw -- --json --fresh
//! cargo run --example acorn_selectivity --release --features hnsw,filtered_graph,range_filtered,curator -- --json --resume
//! ```

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use vicinity::distance::cosine_distance_normalized;
use vicinity::hnsw::{acorn_search_with_stats, AcornConfig, FnFilter};

const DEFAULT_N: usize = 1_200;
const DEFAULT_DIM: usize = 32;
const DEFAULT_QUERIES: usize = 100;
const DEFAULT_K: usize = 10;
const DEFAULT_NEIGHBORS: usize = 32;
const DEFAULT_EF_SEARCH: usize = 200;
const DEFAULT_FALLBACK_SELECTIVITY_THRESHOLD: f64 = 0.02;
const SELECTIVITIES: [f64; 6] = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01];

#[derive(Clone)]
struct Config {
    n: usize,
    dim: usize,
    queries: usize,
    k: usize,
    neighbors: usize,
    ef_search: usize,
    acorn_max_two_hop_neighbors: Option<usize>,
    fallback_selectivity_threshold: f64,
    json: bool,
    results_path: PathBuf,
    fresh: bool,
    resume: bool,
}

impl Config {
    fn acorn_max_two_hop_neighbors(&self) -> usize {
        self.acorn_max_two_hop_neighbors.unwrap_or(self.neighbors)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n: DEFAULT_N,
            dim: DEFAULT_DIM,
            queries: DEFAULT_QUERIES,
            k: DEFAULT_K,
            neighbors: DEFAULT_NEIGHBORS,
            ef_search: DEFAULT_EF_SEARCH,
            acorn_max_two_hop_neighbors: None,
            fallback_selectivity_threshold: DEFAULT_FALLBACK_SELECTIVITY_THRESHOLD,
            json: false,
            results_path: PathBuf::new(),
            fresh: false,
            resume: false,
        }
    }
}

#[derive(Default)]
struct BenchResult {
    recall: f64,
    qps: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    mean_returned: f64,
    two_hop_invocations: u64,
    two_hop_nodes_examined: u64,
}

fn main() {
    let mut cfg = parse_args();
    if cfg.results_path.as_os_str().is_empty() {
        cfg.results_path = default_results_path(&cfg);
    }
    if cfg.fresh {
        std::fs::remove_file(&cfg.results_path).ok();
    }
    let completed = if cfg.resume {
        load_completed_results(&cfg.results_path)
    } else {
        Vec::new()
    };

    let vectors = make_vectors(cfg.n, cfg.dim, 0x5eed);
    let queries = make_vectors(cfg.queries, cfg.dim, 0x9e3779b97f4a7c15);

    let build_start = Instant::now();
    let graph = build_mutual_knn_graph(&vectors, cfg.neighbors);
    let build_time_s = build_start.elapsed().as_secs_f64();

    #[cfg(feature = "filtered_graph")]
    let filtered_graph = build_filtered_graph_index(&cfg, &vectors);
    #[cfg(all(feature = "range_filtered", feature = "hnsw"))]
    let range_filtered = build_range_filtered_index(&cfg, &vectors);
    #[cfg(feature = "curator")]
    let curator = build_curator_index(&cfg, &vectors);

    if cfg.json {
        if !cfg.results_path.exists() {
            emit_json_line(&cfg, &meta_line(&cfg, Some(build_time_s)));
        }
    } else {
        println!("ACORN selectivity sweep");
        println!(
            "n={}, dim={}, queries={}, k={}, neighbors={}, graph_build={:.2}s\n",
            cfg.n, cfg.dim, cfg.queries, cfg.k, cfg.neighbors, build_time_s
        );
        println!(
            "{:>14} {:>11} {:>8} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12}",
            "algorithm",
            "selectivity",
            "recall",
            "qps",
            "p50_us",
            "p95_us",
            "p99_us",
            "returned",
            "2hop_calls",
            "2hop_nodes"
        );
        println!("{}", "-".repeat(122));
    }

    for target_count in selectivity_target_counts(&cfg) {
        let result = run_selectivity(&cfg, &vectors, &queries, &graph, target_count);
        let actual_selectivity = target_count as f64 / cfg.n as f64;
        emit_result(
            &cfg,
            &completed,
            "acorn",
            actual_selectivity,
            target_count,
            &result,
        );
        let result = run_selectivity_gated(&cfg, &vectors, &queries, &graph, target_count);
        emit_result(
            &cfg,
            &completed,
            "selectivity_acorn",
            actual_selectivity,
            target_count,
            &result,
        );

        #[cfg(feature = "filtered_graph")]
        {
            let result = run_filtered_graph_selectivity(
                &cfg,
                &vectors,
                &queries,
                &filtered_graph,
                target_count,
            );
            emit_result(
                &cfg,
                &completed,
                "filtered_graph",
                actual_selectivity,
                target_count,
                &result,
            );
        }

        #[cfg(all(feature = "range_filtered", feature = "hnsw"))]
        {
            let result = run_range_filtered_selectivity(
                &cfg,
                &vectors,
                &queries,
                &range_filtered,
                target_count,
            );
            emit_result(
                &cfg,
                &completed,
                "range_filtered",
                actual_selectivity,
                target_count,
                &result,
            );
        }

        #[cfg(feature = "curator")]
        {
            let result = run_curator_selectivity(&cfg, &vectors, &queries, &curator, target_count);
            emit_result(
                &cfg,
                &completed,
                "curator",
                actual_selectivity,
                target_count,
                &result,
            );
        }
    }
}

fn emit_result(
    cfg: &Config,
    completed: &[String],
    algorithm: &str,
    actual_selectivity: f64,
    target_count: usize,
    result: &BenchResult,
) {
    if cfg.json {
        if cfg.resume && row_completed(cfg, completed, algorithm, target_count) {
            return;
        }
        let mut params = format!(
            "\"selectivity\":{:.4},\"target_count\":{},\"neighbors\":{},\"n\":{},\"dim\":{},\"queries\":{},\"ef_search\":{}",
            actual_selectivity,
            target_count,
            cfg.neighbors,
            cfg.n,
            cfg.dim,
            cfg.queries,
            cfg.ef_search
        );
        if algorithm == "acorn" || algorithm == "selectivity_acorn" {
            params.push_str(&format!(
                ",\"acorn_max_two_hop_neighbors\":{}",
                cfg.acorn_max_two_hop_neighbors()
            ));
        }
        if algorithm == "selectivity_acorn" {
            params.push_str(&format!(
                ",\"fallback_selectivity_threshold\":{:.4}",
                cfg.fallback_selectivity_threshold
            ));
        }
        let line = format!(
            "{{\"algorithm\":\"{}\",\"params\":{{{}}},\"recall_at_{}\":{:.4},\"qps\":{:.1},\"latency_us\":{:.1},\"p50_us\":{:.1},\"p95_us\":{:.1},\"p99_us\":{:.1},\"mean_returned\":{:.1},\"two_hop_invocations\":{},\"two_hop_nodes_examined\":{}}}",
            algorithm,
            params,
            cfg.k,
            result.recall,
            result.qps,
            result.mean_us,
            result.p50_us,
            result.p95_us,
            result.p99_us,
            result.mean_returned,
            result.two_hop_invocations,
            result.two_hop_nodes_examined
        );
        emit_json_line(cfg, &line);
    } else {
        println!(
            "{:>14} {:>10.1}% {:>8.3} {:>10.0} {:>10.1} {:>10.1} {:>10.1} {:>12.1} {:>12} {:>12}",
            algorithm,
            actual_selectivity * 100.0,
            result.recall,
            result.qps,
            result.p50_us,
            result.p95_us,
            result.p99_us,
            result.mean_returned,
            result.two_hop_invocations,
            result.two_hop_nodes_examined
        );
    }
}

fn meta_line(cfg: &Config, graph_build_s: Option<f64>) -> String {
    let mut line = format!(
        "{{\"_meta\":{{\"workload\":\"acorn_selectivity\",\"result_schema\":1,\"n\":{},\"dim\":{},\"queries\":{},\"k\":{},\"neighbors\":{},\"ef_search\":{},\"acorn_max_two_hop_neighbors\":{},\"fallback_selectivity_threshold\":{:.4}",
        cfg.n,
        cfg.dim,
        cfg.queries,
        cfg.k,
        cfg.neighbors,
        cfg.ef_search,
        cfg.acorn_max_two_hop_neighbors(),
        cfg.fallback_selectivity_threshold
    );
    if let Some(graph_build_s) = graph_build_s {
        line.push_str(&format!(",\"graph_build_s\":{graph_build_s:.3}"));
    }
    line.push_str("}}");
    line
}

fn default_results_path(cfg: &Config) -> PathBuf {
    let path = Path::new("data/ann-benchmarks/results");
    std::fs::create_dir_all(path).ok();
    path.join(format!(
        "acorn-selectivity-n{}-d{}-q{}-ef{}-hop{}-fallback{}.jsonl",
        cfg.n,
        cfg.dim,
        cfg.queries,
        cfg.ef_search,
        cfg.acorn_max_two_hop_neighbors(),
        threshold_slug(cfg.fallback_selectivity_threshold)
    ))
}

fn threshold_slug(value: f64) -> String {
    format!("{value:.4}").replace('.', "p")
}

fn emit_json_line(cfg: &Config, line: &str) {
    println!("{line}");
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cfg.results_path)
    {
        let _ = writeln!(file, "{line}");
    }
}

fn load_completed_results(path: &Path) -> Vec<String> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    BufReader::new(file).lines().map_while(Result::ok).collect()
}

fn row_completed(cfg: &Config, lines: &[String], algorithm: &str, target_count: usize) -> bool {
    let uses_acorn_params = matches!(algorithm, "acorn" | "selectivity_acorn");
    lines.iter().any(|line| {
        line.contains(&format!("\"algorithm\":\"{algorithm}\""))
            && line.contains(&format!("\"target_count\":{target_count}"))
            && line.contains(&format!("\"neighbors\":{}", cfg.neighbors))
            && line.contains(&format!("\"n\":{}", cfg.n))
            && line.contains(&format!("\"dim\":{}", cfg.dim))
            && line.contains(&format!("\"queries\":{}", cfg.queries))
            && line.contains(&format!("\"ef_search\":{}", cfg.ef_search))
            && (!uses_acorn_params
                || line.contains(&format!(
                    "\"acorn_max_two_hop_neighbors\":{}",
                    cfg.acorn_max_two_hop_neighbors()
                )))
            && (algorithm != "selectivity_acorn"
                || line.contains(&format!(
                    "\"fallback_selectivity_threshold\":{:.4}",
                    cfg.fallback_selectivity_threshold
                )))
    })
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                i += 1;
                if i < args.len() {
                    cfg.n = args[i].parse().unwrap_or(DEFAULT_N);
                }
            }
            "--dim" => {
                i += 1;
                if i < args.len() {
                    cfg.dim = args[i].parse().unwrap_or(DEFAULT_DIM);
                }
            }
            "--queries" => {
                i += 1;
                if i < args.len() {
                    cfg.queries = args[i].parse().unwrap_or(DEFAULT_QUERIES);
                }
            }
            "--k" => {
                i += 1;
                if i < args.len() {
                    cfg.k = args[i].parse().unwrap_or(DEFAULT_K);
                }
            }
            "--neighbors" => {
                i += 1;
                if i < args.len() {
                    cfg.neighbors = args[i].parse().unwrap_or(DEFAULT_NEIGHBORS);
                }
            }
            "--ef-search" => {
                i += 1;
                if i < args.len() {
                    cfg.ef_search = args[i].parse().unwrap_or(DEFAULT_EF_SEARCH);
                }
            }
            "--acorn-max-two-hop-neighbors" => {
                i += 1;
                if i < args.len() {
                    cfg.acorn_max_two_hop_neighbors = args[i].parse().ok();
                }
            }
            "--fallback-selectivity-threshold" => {
                i += 1;
                if i < args.len() {
                    cfg.fallback_selectivity_threshold = args[i]
                        .parse()
                        .unwrap_or(DEFAULT_FALLBACK_SELECTIVITY_THRESHOLD);
                }
            }
            "--json" => cfg.json = true,
            "--results" => {
                i += 1;
                if i < args.len() {
                    cfg.results_path = PathBuf::from(&args[i]);
                }
            }
            "--fresh" => cfg.fresh = true,
            "--resume" => cfg.resume = true,
            other => eprintln!("unknown flag: {other}"),
        }
        i += 1;
    }
    cfg
}

fn run_selectivity(
    cfg: &Config,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    graph: &[Vec<u32>],
    target_count: usize,
) -> BenchResult {
    let matching: Vec<bool> = (0..cfg.n).map(|id| id < target_count).collect();
    let filter = FnFilter(|id: u32| matching[id as usize]);
    let config = AcornConfig {
        enable_two_hop: true,
        two_hop_threshold: 0.3,
        max_two_hop_neighbors: cfg.acorn_max_two_hop_neighbors(),
        ef_search: cfg.ef_search,
    };

    let mut total_recall = 0.0;
    let mut latencies_us = Vec::with_capacity(queries.len());
    let mut two_hop_invocations = 0;
    let mut two_hop_nodes_examined = 0;
    let mut total_returned = 0usize;

    for query in queries {
        let ground_truth = filtered_ground_truth(query, vectors, &matching, cfg.k);
        let entry_point = nearest_entry_point(query, vectors);

        let start = Instant::now();
        let (results, stats) = acorn_search_with_stats(
            cfg.k,
            &config,
            &filter,
            |id| graph[id as usize].clone(),
            |id| cosine_distance_normalized(query, &vectors[id as usize]),
            entry_point,
        )
        .expect("acorn_search_with_stats failed");
        latencies_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        two_hop_invocations += stats.two_hop_invocations;
        two_hop_nodes_examined += stats.two_hop_nodes_examined;
        total_returned += results.len();

        let expected: HashSet<u32> = ground_truth.into_iter().collect();
        let found: HashSet<u32> = results.into_iter().map(|(id, _)| id).collect();
        total_recall += expected.intersection(&found).count() as f64 / cfg.k as f64;
    }

    finish_result(
        total_recall,
        latencies_us,
        total_returned,
        two_hop_invocations,
        two_hop_nodes_examined,
    )
}

fn run_selectivity_gated(
    cfg: &Config,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    graph: &[Vec<u32>],
    target_count: usize,
) -> BenchResult {
    let actual_selectivity = target_count as f64 / cfg.n as f64;
    if actual_selectivity >= cfg.fallback_selectivity_threshold {
        return run_selectivity(cfg, vectors, queries, graph, target_count);
    }

    let matching: Vec<bool> = (0..cfg.n).map(|id| id < target_count).collect();
    let matching_ids: Vec<u32> = (0..target_count as u32).collect();
    run_index_selectivity(cfg, vectors, queries, &matching, |query, k| {
        let mut distances: Vec<(u32, f32)> = matching_ids
            .iter()
            .map(|&id| (id, cosine_distance_normalized(query, &vectors[id as usize])))
            .collect();
        distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        distances.truncate(k);
        distances
    })
}

#[cfg(feature = "filtered_graph")]
fn build_filtered_graph_index(
    cfg: &Config,
    vectors: &[Vec<f32>],
) -> vicinity::filtered_graph::FilteredGraphIndex {
    use vicinity::filtered_graph::{AttrValue, FilteredGraphIndex, FilteredGraphParams};

    let params = FilteredGraphParams {
        max_degree: cfg.neighbors,
        ef_construction: cfg.neighbors * 4,
        ef_search: cfg.ef_search,
        alpha: 1.2,
    };
    let mut index = FilteredGraphIndex::new(cfg.dim, params).expect("filtered graph init failed");
    for (id, vector) in vectors.iter().enumerate() {
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("rank".to_string(), AttrValue::Int(id as i64));
        index
            .add_slice(id as u32, vector, attrs)
            .expect("filtered graph add failed");
    }
    index.build().expect("filtered graph build failed");
    index
}

#[cfg(feature = "filtered_graph")]
fn run_filtered_graph_selectivity(
    cfg: &Config,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    index: &vicinity::filtered_graph::FilteredGraphIndex,
    target_count: usize,
) -> BenchResult {
    use vicinity::filtered_graph::{AttrValue, Filter, Predicate};

    let matching: Vec<bool> = (0..cfg.n).map(|id| id < target_count).collect();
    let filter = Filter::Clause(Predicate::Lt(
        "rank".to_string(),
        AttrValue::Int(target_count as i64),
    ));
    run_index_selectivity(cfg, vectors, queries, &matching, |query, k| {
        index
            .search_filtered(query, k, &filter)
            .expect("filtered graph search failed")
    })
}

#[cfg(all(feature = "range_filtered", feature = "hnsw"))]
fn build_range_filtered_index(
    cfg: &Config,
    vectors: &[Vec<f32>],
) -> vicinity::range_filtered::RangeFilteredIndex {
    use vicinity::range_filtered::{RangeFilteredIndex, RangeFilteredParams};

    let params = RangeFilteredParams {
        hnsw_m: 16,
        hnsw_ef_construction: 100,
        ef_search: cfg.ef_search,
    };
    let mut index = RangeFilteredIndex::new(cfg.dim, params).expect("range index init failed");
    for (id, vector) in vectors.iter().enumerate() {
        index
            .add(id as u32, vector.clone(), id as f64)
            .expect("range index add failed");
    }
    index.build().expect("range index build failed");
    index
}

#[cfg(all(feature = "range_filtered", feature = "hnsw"))]
fn run_range_filtered_selectivity(
    cfg: &Config,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    index: &vicinity::range_filtered::RangeFilteredIndex,
    target_count: usize,
) -> BenchResult {
    let matching: Vec<bool> = (0..cfg.n).map(|id| id < target_count).collect();
    let hi = target_count.saturating_sub(1) as f64;
    run_index_selectivity(cfg, vectors, queries, &matching, |query, k| {
        index
            .range_search(query, k, 0.0, hi)
            .expect("range filtered search failed")
    })
}

#[cfg(feature = "curator")]
fn build_curator_index(cfg: &Config, vectors: &[Vec<f32>]) -> vicinity::curator::CuratorIndex {
    use vicinity::curator::{CuratorIndex, CuratorParams};

    let params = CuratorParams {
        branching_factor: 8,
        max_leaf_size: 64,
        ef_search: cfg.ef_search,
        beam_width: 4,
    };
    let mut index = CuratorIndex::new(cfg.dim, params).expect("curator init failed");
    let target_counts = selectivity_target_counts(cfg);
    for (id, vector) in vectors.iter().enumerate() {
        let labels = target_counts
            .iter()
            .filter(|&&target_count| id < target_count)
            .map(|&target_count| selectivity_label(target_count))
            .collect();
        index
            .add(id as u32, vector.clone(), labels)
            .expect("curator add failed");
    }
    index.build().expect("curator build failed");
    index
}

#[cfg(feature = "curator")]
fn run_curator_selectivity(
    cfg: &Config,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    index: &vicinity::curator::CuratorIndex,
    target_count: usize,
) -> BenchResult {
    let matching: Vec<bool> = (0..cfg.n).map(|id| id < target_count).collect();
    let label = selectivity_label(target_count);
    run_index_selectivity(cfg, vectors, queries, &matching, |query, k| {
        index
            .search_filtered(query, k, &label)
            .expect("curator search failed")
    })
}

fn selectivity_target_counts(cfg: &Config) -> Vec<usize> {
    let mut counts = Vec::new();
    for selectivity in SELECTIVITIES {
        let target_count = ((cfg.n as f64 * selectivity).round() as usize)
            .max(cfg.k)
            .min(cfg.n);
        if counts.last().copied() != Some(target_count) {
            counts.push(target_count);
        }
    }
    counts
}

#[cfg(feature = "curator")]
fn selectivity_label(target_count: usize) -> String {
    format!("top_{target_count}")
}

#[allow(dead_code)]
fn run_index_selectivity<F>(
    cfg: &Config,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    matching: &[bool],
    search: F,
) -> BenchResult
where
    F: Fn(&[f32], usize) -> Vec<(u32, f32)>,
{
    let mut total_recall = 0.0;
    let mut latencies_us = Vec::with_capacity(queries.len());
    let mut total_returned = 0usize;

    for query in queries {
        let ground_truth = filtered_ground_truth(query, vectors, matching, cfg.k);

        let start = Instant::now();
        let results = search(query, cfg.k);
        latencies_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        total_returned += results.len();

        let expected: HashSet<u32> = ground_truth.into_iter().collect();
        let found: HashSet<u32> = results.into_iter().map(|(id, _)| id).collect();
        total_recall += expected.intersection(&found).count() as f64 / cfg.k as f64;
    }

    finish_result(total_recall, latencies_us, total_returned, 0, 0)
}

fn finish_result(
    total_recall: f64,
    mut latencies_us: Vec<f64>,
    total_returned: usize,
    two_hop_invocations: u64,
    two_hop_nodes_examined: u64,
) -> BenchResult {
    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let total_us: f64 = latencies_us.iter().sum();
    let n = latencies_us.len();

    BenchResult {
        recall: total_recall / n as f64,
        qps: n as f64 / (total_us / 1_000_000.0),
        mean_us: total_us / n as f64,
        p50_us: latencies_us[n / 2],
        p95_us: latencies_us[((n - 1) as f64 * 0.95) as usize],
        p99_us: latencies_us[((n - 1) as f64 * 0.99) as usize],
        mean_returned: total_returned as f64 / n as f64,
        two_hop_invocations,
        two_hop_nodes_examined,
    }
}

fn build_mutual_knn_graph(vectors: &[Vec<f32>], neighbors: usize) -> Vec<Vec<u32>> {
    let n = vectors.len();
    let mut graph: Vec<HashSet<u32>> = (0..n).map(|_| HashSet::new()).collect();

    for i in 0..n {
        let mut distances: Vec<(u32, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                (
                    j as u32,
                    cosine_distance_normalized(&vectors[i], &vectors[j]),
                )
            })
            .collect();
        distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        for &(neighbor, _) in distances.iter().take(neighbors) {
            graph[i].insert(neighbor);
            graph[neighbor as usize].insert(i as u32);
        }
    }

    graph
        .into_iter()
        .map(|neighbors| neighbors.into_iter().collect())
        .collect()
}

fn filtered_ground_truth(
    query: &[f32],
    vectors: &[Vec<f32>],
    matching: &[bool],
    k: usize,
) -> Vec<u32> {
    let mut distances: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .filter(|(id, _)| matching[*id])
        .map(|(id, vector)| (id as u32, cosine_distance_normalized(query, vector)))
        .collect();
    distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    distances.into_iter().take(k).map(|(id, _)| id).collect()
}

fn nearest_entry_point(query: &[f32], vectors: &[Vec<f32>]) -> u32 {
    vectors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = cosine_distance_normalized(query, a);
            let db = cosine_distance_normalized(query, b);
            da.total_cmp(&db)
        })
        .map(|(id, _)| id as u32)
        .unwrap_or(0)
}

fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let mut vector: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
            normalize_in_place(&mut vector);
            vector
        })
        .collect()
}

fn normalize_in_place(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for value in vector {
            *value /= norm;
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}
