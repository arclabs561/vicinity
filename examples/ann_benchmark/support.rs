#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub(crate) const DEFAULT_WARMUP_QUERIES: usize = 50;
static WARMUP_QUERIES: AtomicUsize = AtomicUsize::new(DEFAULT_WARMUP_QUERIES);

pub(crate) const ALGORITHM_OPTIONS: &[&str] = &[
    "hnsw",
    "nsw",
    "ivfpq",
    "ivf_avq",
    "emg",
    "nsg",
    "dual_branch",
    "deg",
    "pipnn",
    "sng",
    "vamana",
    "diskann",
    "ivf_rabitq",
    "symphony_qg",
    "symphony_qg_vr",
    "finger",
    "fresh_graph",
    "store",
    "fresh_graph_churn",
    "inplace",
    "inplace_churn",
    "lsm_churn",
    "filtered_graph",
    "rp_quant",
    "sparse_mips",
    "curator",
    "range_filtered",
    "binary_index",
    "sq4",
    "sq4u",
    "sq8u",
    "adsampling",
    "lsh",
    "hnsw_prt",
    "kdtree",
    "balltree",
    "rptree",
    "rp_forest",
    "kmeans_tree",
    "brute",
];

pub(crate) fn algorithm_options_help() -> String {
    ALGORITHM_OPTIONS.join(", ")
}

pub(crate) fn active_features_json() -> String {
    let mut active = Vec::new();
    macro_rules! push_feature {
        ($feature:literal) => {
            if cfg!(feature = $feature) {
                active.push($feature);
            }
        };
    }

    push_feature!("balltree");
    push_feature!("benchmark");
    push_feature!("binary_index");
    push_feature!("cli");
    push_feature!("curator");
    push_feature!("diskann");
    push_feature!("emg");
    push_feature!("evoc");
    push_feature!("experimental");
    push_feature!("filtered_graph");
    push_feature!("finger");
    push_feature!("fresh_graph");
    push_feature!("hnsw");
    push_feature!("id-compression");
    push_feature!("innr");
    push_feature!("ivf_avq");
    push_feature!("ivf_pq");
    push_feature!("ivf_rabitq");
    push_feature!("kdtree");
    push_feature!("kmeans_tree");
    push_feature!("lemur");
    push_feature!("lsh");
    push_feature!("nsg");
    push_feature!("nsw");
    push_feature!("parallel");
    push_feature!("persistence");
    push_feature!("pipnn");
    push_feature!("python");
    push_feature!("qntz");
    push_feature!("quantization");
    push_feature!("rabitq");
    push_feature!("range_filtered");
    push_feature!("rmt-spectral");
    push_feature!("rp_quant");
    push_feature!("rptree");
    push_feature!("saq");
    push_feature!("serde");
    push_feature!("sng");
    push_feature!("sparse_mips");
    push_feature!("sq4");
    push_feature!("sq8");
    push_feature!("store");
    push_feature!("vamana");

    let quoted = active
        .iter()
        .map(|feature| format!("\"{feature}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{quoted}]")
}

pub(crate) fn set_warmup_queries(count: usize) {
    WARMUP_QUERIES.store(count, Ordering::Relaxed);
}

pub(crate) fn warmup_queries() -> usize {
    WARMUP_QUERIES.load(Ordering::Relaxed)
}

pub(crate) struct Config {
    pub(crate) data_dir: String,
    pub(crate) algos: Vec<String>,
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    pub(crate) ef_search_values: Vec<usize>,
    pub(crate) json: bool,
    pub(crate) results_path: PathBuf,
    pub(crate) is_euclidean: bool,
    pub(crate) pq_num_clusters: Option<usize>,
    pub(crate) pq_num_codebooks: Option<usize>,
    pub(crate) pq_codebook_size: usize,
    pub(crate) pq_training_sample_size: Option<usize>,
    pub(crate) pq_kmeans_max_iter: usize,
    pub(crate) pq_nprobe_values: Option<Vec<usize>>,
    pub(crate) pq_rerank_pools: Vec<usize>,
    pub(crate) tree_leaf_sizes: Vec<usize>,
    pub(crate) tree_depths: Vec<usize>,
    pub(crate) rp_num_trees: Vec<usize>,
    pub(crate) kmeans_clusters: Vec<usize>,
    pub(crate) kmeans_leaf_sizes: Vec<usize>,
    pub(crate) kmeans_depths: Vec<usize>,
    pub(crate) kmeans_iters: Vec<usize>,
    pub(crate) kmeans_search_branches: Vec<usize>,
    pub(crate) kmeans_leaf_budgets: Vec<usize>,
    pub(crate) batch: bool,
    pub(crate) resume: bool,
    pub(crate) snapshot_load: bool,
    pub(crate) max_train: Option<usize>,
    pub(crate) max_queries: Option<usize>,
    pub(crate) warmup_queries: usize,
    pub(crate) churn_base_size: usize,
    pub(crate) churn_cycles: usize,
    pub(crate) churn_queries: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: "data/ann-benchmarks/glove-25-angular".into(),
            algos: vec!["hnsw".into()],
            m: 16,
            ef_construction: 200,
            ef_search_values: vec![10, 20, 50, 100, 200, 400],
            json: false,
            results_path: PathBuf::new(),
            is_euclidean: false,
            pq_num_clusters: None,
            pq_num_codebooks: None,
            pq_codebook_size: 256,
            pq_training_sample_size: None,
            pq_kmeans_max_iter: 100,
            pq_nprobe_values: None,
            pq_rerank_pools: Vec::new(),
            tree_leaf_sizes: vec![10],
            tree_depths: vec![32],
            rp_num_trees: vec![10],
            kmeans_clusters: vec![16],
            kmeans_leaf_sizes: vec![50],
            kmeans_depths: vec![10],
            kmeans_iters: vec![10],
            kmeans_search_branches: vec![1],
            kmeans_leaf_budgets: Vec::new(),
            batch: false,
            resume: false,
            snapshot_load: false,
            max_train: None,
            max_queries: None,
            warmup_queries: DEFAULT_WARMUP_QUERIES,
            churn_base_size: 50_000,
            churn_cycles: 5_000,
            churn_queries: 1_000,
        }
    }
}

fn default_results_path(data_dir: &str) -> PathBuf {
    let dataset = Path::new(data_dir)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let results_dir = Path::new(data_dir)
        .parent()
        .unwrap_or(Path::new("."))
        .join("results");
    std::fs::create_dir_all(&results_dir).ok();
    results_dir.join(format!("{}-all-algos.jsonl", dataset))
}

#[derive(Default)]
pub(crate) struct CompletedResults {
    pub(crate) counts: HashMap<String, usize>,
    lines: Vec<String>,
    pub(crate) has_matching_meta: bool,
    pub(crate) has_mismatched_meta: bool,
}

fn json_string_field(line: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", field);
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn json_value_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    let needle = format!("\"{}\":", field);
    let start = line.find(&needle)? + needle.len();
    let bytes = line.as_bytes();
    match bytes.get(start)? {
        b'{' => {
            let mut depth = 0usize;
            let mut in_string = false;
            let mut escaped = false;
            for (offset, &byte) in bytes[start..].iter().enumerate() {
                if in_string {
                    if escaped {
                        escaped = false;
                    } else if byte == b'\\' {
                        escaped = true;
                    } else if byte == b'"' {
                        in_string = false;
                    }
                    continue;
                }
                match byte {
                    b'"' => in_string = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            return Some(&line[start..=start + offset]);
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        _ => {
            let end = bytes[start..]
                .iter()
                .position(|&byte| byte == b',' || byte == b'}')
                .map(|offset| start + offset)
                .unwrap_or(line.len());
            Some(&line[start..end])
        }
    }
}

fn meta_usize_field_matches(line: &str, field: &str, expected: Option<usize>) -> bool {
    match json_value_field(line, field).map(str::trim) {
        None => expected.is_none(),
        Some("null") => expected.is_none(),
        Some(raw) => raw.parse::<usize>().ok() == expected,
    }
}

fn meta_warmup_field_matches(line: &str, expected: usize) -> bool {
    match json_value_field(line, "warmup_queries").map(str::trim) {
        None => expected == DEFAULT_WARMUP_QUERIES,
        Some(raw) => raw.parse::<usize>().ok() == Some(expected),
    }
}

fn meta_has_current_result_contract(line: &str) -> bool {
    json_value_field(line, "result_schema").map(str::trim) == Some("2")
        && json_value_field(line, "index_bytes_required").map(str::trim) == Some("true")
}

#[cfg(test)]
pub(crate) fn load_completed_results(
    path: &Path,
    expected_dataset: &str,
    expected_train_limit: Option<usize>,
    expected_query_limit: Option<usize>,
) -> CompletedResults {
    load_completed_results_with_warmup(
        path,
        expected_dataset,
        expected_train_limit,
        expected_query_limit,
        DEFAULT_WARMUP_QUERIES,
    )
}

pub(crate) fn load_completed_results_with_warmup(
    path: &Path,
    expected_dataset: &str,
    expected_train_limit: Option<usize>,
    expected_query_limit: Option<usize>,
    expected_warmup_queries: usize,
) -> CompletedResults {
    let mut counts = HashMap::new();
    let mut lines = Vec::new();
    let mut seen_meta = false;
    let mut active_dataset_matches = false;
    let mut has_matching_meta = false;
    let mut has_mismatched_meta = false;
    let Ok(file) = File::open(path) else {
        return CompletedResults::default();
    };
    for line in BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        if line.contains("\"_meta\":") {
            if let Some(dataset) = json_string_field(&line, "dataset") {
                seen_meta = true;
                if dataset != expected_dataset
                    || !meta_has_current_result_contract(&line)
                    || !meta_usize_field_matches(&line, "train_limit", expected_train_limit)
                    || !meta_usize_field_matches(&line, "query_limit", expected_query_limit)
                    || !meta_warmup_field_matches(&line, expected_warmup_queries)
                {
                    has_mismatched_meta = true;
                    active_dataset_matches = false;
                } else {
                    has_matching_meta = true;
                    active_dataset_matches = true;
                }
            }
        } else if active_dataset_matches || !seen_meta {
            if let Some(algorithm) = json_string_field(&line, "algorithm") {
                *counts.entry(algorithm).or_insert(0) += 1;
            }
            lines.push(line);
        }
    }
    CompletedResults {
        counts,
        lines,
        has_matching_meta,
        has_mismatched_meta,
    }
}

enum ParamCheck {
    Any,
    Exact(String),
    Fragments(Vec<String>),
}

struct ExpectedResult {
    algorithm: String,
    params: ParamCheck,
    storage_mode: &'static str,
}

impl ExpectedResult {
    fn any_params(algorithm: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            params: ParamCheck::Any,
            storage_mode: "in_memory",
        }
    }

    fn with_params(algorithm: impl Into<String>, params_json: &str) -> Self {
        Self {
            algorithm: algorithm.into(),
            params: ParamCheck::Exact(params_json.to_string()),
            storage_mode: "in_memory",
        }
    }

    fn with_params_and_storage(
        algorithm: impl Into<String>,
        params_json: &str,
        storage_mode: &'static str,
    ) -> Self {
        Self::with_params(algorithm, params_json).with_storage(storage_mode)
    }

    fn with_param_fragments(
        algorithm: impl Into<String>,
        fragments: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            algorithm: algorithm.into(),
            params: ParamCheck::Fragments(fragments.into_iter().collect()),
            storage_mode: "in_memory",
        }
    }

    fn with_storage(mut self, storage_mode: &'static str) -> Self {
        self.storage_mode = storage_mode;
        self
    }

    fn matches(&self, line: &str) -> bool {
        if json_string_field(line, "algorithm").as_deref() != Some(self.algorithm.as_str()) {
            return false;
        }
        match &self.params {
            ParamCheck::Any => {}
            ParamCheck::Exact(expected) => {
                if json_value_field(line, "params") != Some(expected.as_str()) {
                    return false;
                }
            }
            ParamCheck::Fragments(fragments) => {
                if !fragments.iter().all(|fragment| line.contains(fragment)) {
                    return false;
                }
            }
        }
        if json_string_field(line, "storage_mode").as_deref() != Some(self.storage_mode) {
            return false;
        }
        true
    }
}

fn params_containing_check(
    algorithm: &str,
    fragments: impl IntoIterator<Item = String>,
) -> ExpectedResult {
    ExpectedResult::with_param_fragments(algorithm, fragments)
}

#[derive(Clone, Copy)]
enum StorageExpectation {
    Reload,
    ReloadAndFileOpen,
}

impl StorageExpectation {
    fn required_modes(self, cfg: &Config) -> &'static [&'static str] {
        if !cfg.snapshot_load {
            return &[];
        }

        match self {
            StorageExpectation::Reload => &["snapshot_loaded"],
            StorageExpectation::ReloadAndFileOpen => open_storage_modes(),
        }
    }
}

fn open_storage_modes() -> &'static [&'static str] {
    #[cfg(feature = "persistence")]
    {
        &["snapshot_loaded", "file", "mmap"]
    }

    #[cfg(not(feature = "persistence"))]
    {
        &["snapshot_loaded", "file"]
    }
}

fn ef_checks(
    algorithm: &str,
    cfg: &Config,
    extra_fragments: impl Fn(usize) -> Vec<String>,
) -> Vec<ExpectedResult> {
    cfg.ef_search_values
        .iter()
        .map(|&ef| {
            let mut fragments = extra_fragments(ef);
            fragments.push(format!("\"ef_search\":{}", ef));
            params_containing_check(algorithm, fragments)
        })
        .collect()
}

#[cfg(not(feature = "serde"))]
fn hnsw_quantized_checks(algorithm: &str, cfg: &Config) -> Vec<ExpectedResult> {
    ef_checks(algorithm, cfg, |ef| {
        vec![
            format!("\"m\":{}", cfg.m),
            format!("\"ef_construction\":{}", cfg.ef_construction),
            format!("\"rerank_pool\":{}", (ef * 2).max(100)),
        ]
    })
}

fn hnsw_quantized_snapshot_checks(algorithm: &str, cfg: &Config) -> Vec<ExpectedResult> {
    ef_snapshot_checks(algorithm, cfg, |ef| {
        format!(
            "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"rerank_pool\":{}}}",
            cfg.m,
            cfg.ef_construction,
            ef,
            (ef * 2).max(100)
        )
    })
}

fn storage_expectation_checks(
    algorithm: &str,
    params_json: &str,
    cfg: &Config,
    expectation: StorageExpectation,
) -> Vec<ExpectedResult> {
    std::iter::once("in_memory")
        .chain(expectation.required_modes(cfg).iter().copied())
        .map(|storage_mode| {
            ExpectedResult::with_params_and_storage(algorithm, params_json, storage_mode)
        })
        .collect()
}

fn snapshot_check(algorithm: &str, params_json: &str, cfg: &Config) -> Vec<ExpectedResult> {
    storage_expectation_checks(algorithm, params_json, cfg, StorageExpectation::Reload)
}

fn snapshot_file_checks(algorithm: &str, params_json: &str, cfg: &Config) -> Vec<ExpectedResult> {
    storage_expectation_checks(
        algorithm,
        params_json,
        cfg,
        StorageExpectation::ReloadAndFileOpen,
    )
}

fn serde_snapshot_check(algorithm: &str, params_json: &str, cfg: &Config) -> Vec<ExpectedResult> {
    #[cfg(not(feature = "serde"))]
    {
        let _ = cfg;
        vec![ExpectedResult::with_params(algorithm, params_json)]
    }

    #[cfg(feature = "serde")]
    {
        snapshot_check(algorithm, params_json, cfg)
    }
}

fn ef_snapshot_checks(
    algorithm: &str,
    cfg: &Config,
    params_json: impl Fn(usize) -> String,
) -> Vec<ExpectedResult> {
    cfg.ef_search_values
        .iter()
        .flat_map(|&ef| {
            let params_json = params_json(ef);
            snapshot_check(algorithm, &params_json, cfg)
        })
        .collect()
}

fn max_degree_ef_snapshot_checks(algorithm: &str, cfg: &Config) -> Vec<ExpectedResult> {
    ef_snapshot_checks(algorithm, cfg, |ef| {
        format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef)
    })
}

fn ef_serde_snapshot_checks(
    algorithm: &str,
    cfg: &Config,
    params_json: impl Fn(usize) -> String,
) -> Vec<ExpectedResult> {
    cfg.ef_search_values
        .iter()
        .flat_map(|&ef| {
            let params_json = params_json(ef);
            serde_snapshot_check(algorithm, &params_json, cfg)
        })
        .collect()
}

fn ef_exact_checks(
    algorithm: &str,
    cfg: &Config,
    params_json: impl Fn(usize) -> String,
) -> Vec<ExpectedResult> {
    cfg.ef_search_values
        .iter()
        .map(|&ef| {
            let params_json = params_json(ef);
            ExpectedResult::with_params(algorithm, &params_json)
        })
        .collect()
}

fn tree_snapshot_checks(enabled: bool, algorithm: &str, cfg: &Config) -> Vec<ExpectedResult> {
    if !enabled {
        return Vec::new();
    }

    cfg.tree_leaf_sizes
        .iter()
        .flat_map(|&leaf_size| {
            cfg.tree_depths.iter().flat_map(move |&depth| {
                snapshot_check(algorithm, &tree_params_json(leaf_size, depth), cfg)
            })
        })
        .collect()
}

fn rp_forest_snapshot_checks(enabled: bool, cfg: &Config) -> Vec<ExpectedResult> {
    if !enabled {
        return Vec::new();
    }

    cfg.rp_num_trees
        .iter()
        .flat_map(|&num_trees| {
            cfg.tree_leaf_sizes.iter().flat_map(move |&leaf_size| {
                snapshot_check(
                    "rp_forest",
                    &rp_forest_params_json(num_trees, leaf_size),
                    cfg,
                )
            })
        })
        .collect()
}

fn kmeans_tree_snapshot_checks(cfg: &Config) -> Vec<ExpectedResult> {
    let mut checks: Vec<_> = cfg
        .kmeans_clusters
        .iter()
        .flat_map(|&num_clusters| {
            cfg.kmeans_leaf_sizes.iter().flat_map(move |&leaf_size| {
                cfg.kmeans_depths.iter().flat_map(move |&depth| {
                    cfg.kmeans_iters.iter().flat_map(move |&max_iterations| {
                        cfg.kmeans_search_branches
                            .iter()
                            .flat_map(move |&search_branches| {
                                snapshot_check(
                                    "kmeans_tree",
                                    &kmeans_tree_params_json(
                                        num_clusters,
                                        leaf_size,
                                        depth,
                                        max_iterations,
                                        search_branches,
                                    ),
                                    cfg,
                                )
                            })
                    })
                })
            })
        })
        .collect();

    checks.extend(cfg.kmeans_clusters.iter().flat_map(|&num_clusters| {
        cfg.kmeans_leaf_sizes.iter().flat_map(move |&leaf_size| {
            cfg.kmeans_depths.iter().flat_map(move |&depth| {
                cfg.kmeans_iters.iter().flat_map(move |&max_iterations| {
                    cfg.kmeans_leaf_budgets
                        .iter()
                        .flat_map(move |&leaf_budget| {
                            snapshot_check(
                                "kmeans_tree",
                                &kmeans_tree_leaf_budget_params_json(
                                    num_clusters,
                                    leaf_size,
                                    depth,
                                    max_iterations,
                                    leaf_budget,
                                ),
                                cfg,
                            )
                        })
                })
            })
        })
    }));
    checks
}

pub(crate) fn nprobe_values(cfg: &Config, max_probe: usize) -> Vec<usize> {
    let values = cfg
        .pq_nprobe_values
        .clone()
        .unwrap_or_else(|| vec![1, 2, 5, 10, 20, 50, 100]);
    values
        .into_iter()
        .filter(|&nprobe| nprobe > 0 && nprobe <= max_probe)
        .collect()
}

pub(crate) fn ivfavq_num_reorder_values(cfg: &Config) -> Vec<usize> {
    if cfg.pq_rerank_pools.is_empty() {
        vec![100]
    } else {
        let values: Vec<_> = cfg
            .pq_rerank_pools
            .iter()
            .copied()
            .filter(|&num_reorder| num_reorder > 0)
            .collect();
        if values.is_empty() {
            vec![100]
        } else {
            values
        }
    }
}

pub(crate) fn ivfavq_params_json(
    num_partitions: usize,
    num_codebooks: usize,
    codebook_size: usize,
    nprobe: usize,
    num_reorder: usize,
) -> String {
    format!(
        "{{\"num_partitions\":{},\"num_codebooks\":{},\"codebook_size\":{},\"nprobe\":{},\"num_reorder\":{}}}",
        num_partitions, num_codebooks, codebook_size, nprobe, num_reorder
    )
}

fn diskann_checks(cfg: &Config) -> Vec<ExpectedResult> {
    const STORAGE_ROWS: [(&str, &str, &str); 3] = [
        ("diskann", "memory", "in_memory"),
        ("diskann_file", "file", "file"),
        ("diskann_mmap", "mmap", "mmap"),
    ];

    STORAGE_ROWS
        .into_iter()
        .flat_map(|(algorithm, storage, storage_mode)| {
            cfg.ef_search_values.iter().map(move |&ef| {
                ExpectedResult::with_params_and_storage(
                    algorithm,
                    &diskann_params_json(cfg, ef, storage),
                    storage_mode,
                )
            })
        })
        .collect()
}

fn hnsw_result_checks(cfg: &Config) -> Vec<ExpectedResult> {
    cfg.ef_search_values
        .iter()
        .flat_map(|&ef| {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{}}}",
                cfg.m, cfg.ef_construction, ef
            );
            #[cfg(not(feature = "parallel"))]
            {
                serde_snapshot_check("hnsw", &params_json, cfg)
            }

            #[cfg(feature = "parallel")]
            {
                let mut markers = serde_snapshot_check("hnsw", &params_json, cfg);
                if cfg.batch {
                    markers.push(params_containing_check(
                        "hnsw_parallel",
                        [
                            format!("\"m\":{}", cfg.m),
                            format!("\"ef_construction\":{}", cfg.ef_construction),
                            format!("\"ef_search\":{}", ef),
                            "\"threads\":".to_string(),
                        ],
                    ));
                }
                markers
            }
        })
        .collect()
}

fn ivfpq_result_checks(cfg: &Config, dim: usize) -> Vec<ExpectedResult> {
    let num_clusters = cfg.pq_num_clusters.unwrap_or(256);
    if num_clusters == 0 {
        return Vec::new();
    }
    let num_codebooks = cfg.pq_num_codebooks.unwrap_or_else(|| {
        (1..=8.min(dim))
            .rev()
            .find(|&c| dim.is_multiple_of(c))
            .unwrap_or(1)
    });
    if !dim.is_multiple_of(num_codebooks) {
        return Vec::new();
    }
    nprobe_values(cfg, num_clusters)
        .into_iter()
        .flat_map(|nprobe| {
            let params_json = ivfpq_params_json(
                num_clusters,
                num_codebooks,
                cfg.pq_codebook_size,
                nprobe,
                None,
                cfg.pq_training_sample_size,
                cfg.pq_kmeans_max_iter,
            );
            let mut markers = snapshot_file_checks("ivfpq", &params_json, cfg);
            markers.extend(
                cfg.pq_rerank_pools
                    .iter()
                    .copied()
                    .filter(|&rerank_pool| rerank_pool > 0)
                    .flat_map(|rerank_pool| {
                        snapshot_file_checks(
                            "ivfpq_rerank",
                            &ivfpq_params_json(
                                num_clusters,
                                num_codebooks,
                                cfg.pq_codebook_size,
                                nprobe,
                                Some(rerank_pool),
                                cfg.pq_training_sample_size,
                                cfg.pq_kmeans_max_iter,
                            ),
                            cfg,
                        )
                    }),
            );
            markers
        })
        .collect()
}

fn ivfavq_result_checks(cfg: &Config, dim: usize, train_len: usize) -> Vec<ExpectedResult> {
    let num_partitions = 256.min(train_len).max(1);
    let num_codebooks = (1..=16.min(dim))
        .rev()
        .find(|&c| dim.is_multiple_of(c))
        .unwrap_or(1);
    nprobe_values(cfg, num_partitions)
        .into_iter()
        .flat_map(|nprobe| {
            ivfavq_num_reorder_values(cfg)
                .into_iter()
                .flat_map(move |num_reorder| {
                    snapshot_file_checks(
                        "ivf_avq",
                        &ivfavq_params_json(
                            num_partitions,
                            num_codebooks,
                            256,
                            nprobe,
                            num_reorder,
                        ),
                        cfg,
                    )
                })
        })
        .collect()
}

fn ivf_rabitq_result_checks(cfg: &Config) -> Vec<ExpectedResult> {
    nprobe_values(cfg, 256)
        .into_iter()
        .flat_map(|nprobe| {
            snapshot_check(
                "ivf_rabitq",
                &format!(
                    "{{\"num_clusters\":256,\"total_bits\":4,\"nprobe\":{}}}",
                    nprobe
                ),
                cfg,
            )
        })
        .collect()
}

fn lsh_result_checks(cfg: &Config, dim: usize) -> Vec<ExpectedResult> {
    const TABLE_SWEEP: usize = 3;
    let num_tables_values = [8, 16, 32];
    let num_probes_values = [2, 4, 8, 16];
    num_tables_values
        .into_iter()
        .take(TABLE_SWEEP)
        .flat_map(|num_tables| {
            num_probes_values
                .into_iter()
                .filter(move |&num_probes| num_probes <= dim)
                .flat_map(move |num_probes| {
                    snapshot_check(
                        "lsh",
                        &format!(
                            "{{\"num_tables\":{},\"num_probes\":{}}}",
                            num_tables, num_probes
                        ),
                        cfg,
                    )
                })
        })
        .collect()
}

fn churn_shape(cfg: &Config, train_len: usize, test_len: usize) -> Option<(usize, usize, usize)> {
    let base_size = cfg
        .churn_base_size
        .min(train_len.saturating_sub(cfg.churn_cycles.max(1)));
    let cycles = cfg.churn_cycles.min(train_len.saturating_sub(base_size));
    let queries = cfg.churn_queries.min(test_len);
    if base_size == 0 || cycles == 0 || queries == 0 {
        None
    } else {
        Some((base_size, cycles, queries))
    }
}

fn required_result_checks(
    algo: &str,
    cfg: &Config,
    dim: usize,
    train_len: usize,
    test_len: usize,
) -> Vec<ExpectedResult> {
    match algo {
        "hnsw" => hnsw_result_checks(cfg),
        "nsw" => ef_snapshot_checks("nsw", cfg, |ef| {
            format!("{{\"m\":{},\"ef_search\":{}}}", cfg.m, ef)
        }),
        "ivfpq" => ivfpq_result_checks(cfg, dim),
        "ivf_avq" => ivfavq_result_checks(cfg, dim, train_len),
        "ivf_rabitq" => ivf_rabitq_result_checks(cfg),
        "rp_quant" => snapshot_check(
            "rp_quant",
            &format!("{{\"projected_dim\":{},\"rerank_factor\":10}}", 64.min(dim)),
            cfg,
        ),
        "binary_index" => snapshot_check("binary_index", "{\"rerank_factor\":10}", cfg),
        "lsh" => lsh_result_checks(cfg, dim),
        "emg" | "nsg" | "fresh_graph" => max_degree_ef_snapshot_checks(algo, cfg),
        "dual_branch" => ef_serde_snapshot_checks("dual_branch", cfg, |ef| {
            format!(
                "{{\"m\":{},\"m_high_lid\":{},\"ef_construction\":{},\"ef_search\":{}}}",
                cfg.m,
                (cfg.m + cfg.m / 2).max(cfg.m + 1),
                cfg.ef_construction,
                ef
            )
        }),
        "deg" => {
            let indexed_vectors = train_len.min(10_000);
            let capped = train_len > indexed_vectors;
            ef_serde_snapshot_checks("deg", cfg, |ef| {
                format!(
                    "{{\"base_edges\":16,\"max_edges\":32,\"min_edges\":8,\"density_k\":10,\"alpha\":1.2,\"ef_search\":{},\"indexed_vectors\":{},\"capped\":{}}}",
                    ef, indexed_vectors, capped
                )
            })
        }
        "filtered_graph" => ef_snapshot_checks("filtered_graph", cfg, |ef| {
            format!(
                "{{\"max_degree\":32,\"ef_search\":{},\"filter_mode\":\"none\"}}",
                ef
            )
        }),
        "inplace" => ef_serde_snapshot_checks("inplace", cfg, |beam_width| {
            format!(
                "{{\"max_degree\":32,\"build_beam_width\":{},\"beam_width\":{}}}",
                cfg.ef_construction, beam_width
            )
        }),
        "pipnn" => ef_snapshot_checks("pipnn", cfg, |ef| {
            format!(
                "{{\"max_degree\":32,\"max_leaf_size\":2048,\"ef_search\":{}}}",
                ef
            )
        }),
        "vamana" => ef_snapshot_checks("vamana", cfg, |ef| format!("{{\"ef_search\":{}}}", ef)),
        "diskann" => diskann_checks(cfg),
        "store" => cfg
            .ef_search_values
            .iter()
            .flat_map(|&ef| {
                let params = store_params_json(cfg, train_len, ef);
                [
                    ExpectedResult::with_params_and_storage("store", &params, "segmented_store"),
                    ExpectedResult::with_params_and_storage(
                        "store_snapshot",
                        &params,
                        "segmented_store",
                    ),
                ]
            })
            .collect(),
        "finger" => {
            let indexed_vectors = train_len.min(50_000);
            let capped = train_len > indexed_vectors;
            ef_snapshot_checks("finger", cfg, |ef| {
                format!(
                    "{{\"max_degree\":32,\"ef_search\":{},\"indexed_vectors\":{},\"capped\":{}}}",
                    ef, indexed_vectors, capped
                )
            })
        }
        "sng" => snapshot_check("sng", "{}", cfg),
        "fresh_graph_churn" => {
            let Some((base_size, cycles, queries)) = churn_shape(cfg, train_len, test_len) else {
                return Vec::new();
            };
            ef_checks("fresh_graph_churn", cfg, |_| {
                vec![
                    "\"max_degree\":32".to_string(),
                    format!("\"base_size\":{}", base_size),
                    format!("\"cycles\":{}", cycles),
                    format!("\"queries\":{}", queries),
                ]
            })
        }
        "inplace_churn" => {
            let Some((base_size, cycles, queries)) = churn_shape(cfg, train_len, test_len) else {
                return Vec::new();
            };
            ef_exact_checks("inplace_churn", cfg, |beam_width| {
                format!(
                    "{{\"max_degree\":32,\"build_beam_width\":{},\"beam_width\":{},\"base_size\":{},\"cycles\":{},\"queries\":{}}}",
                    cfg.ef_construction, beam_width, base_size, cycles, queries
                )
            })
        }
        "lsm_churn" => {
            let Some((base_size, cycles, queries)) = churn_shape(cfg, train_len, test_len) else {
                return Vec::new();
            };
            let buffer_capacity = (base_size / 10).clamp(20, 10_000);
            ef_checks("lsm_churn", cfg, |_| {
                vec![
                    format!("\"base_size\":{}", base_size),
                    format!("\"cycles\":{}", cycles),
                    format!("\"queries\":{}", queries),
                    format!("\"buffer_capacity\":{}", buffer_capacity),
                    "\"size_ratio\":4".to_string(),
                ]
            })
        }
        "adsampling" => ef_exact_checks("adsampling", cfg, |ef| {
            format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"epsilon0\":2.1}}",
                cfg.m, cfg.ef_construction, ef
            )
        }),
        "hnsw_prt" => {
            let num_projections = (dim / 4).clamp(8, 64);
            ef_exact_checks("hnsw_prt", cfg, |ef| {
                format!(
                    "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"num_projections\":{}}}",
                    cfg.m, cfg.ef_construction, ef, num_projections
                )
            })
        }
        "sq8u" | "sq4u" => hnsw_quantized_snapshot_checks(algo, cfg),
        "symphony_qg_vr" => {
            #[cfg(feature = "serde")]
            {
                hnsw_quantized_snapshot_checks(algo, cfg)
            }
            #[cfg(not(feature = "serde"))]
            {
                hnsw_quantized_checks(algo, cfg)
            }
        }
        "symphony_qg" => {
            #[cfg(feature = "serde")]
            {
                ef_snapshot_checks("symphony_qg", cfg, |ef| {
                    format!(
                        "{{\"m\":16,\"ef_search\":{},\"rerank_pool\":{}}}",
                        ef,
                        (ef * 2).max(100)
                    )
                })
            }
            #[cfg(not(feature = "serde"))]
            {
                ef_checks("symphony_qg", cfg, |_| Vec::new())
            }
        }
        "sq4" => snapshot_check("sq4", "{\"rerank_factor\":10}", cfg),
        "curator" => snapshot_check(
            "curator",
            "{\"branching_factor\":16,\"max_leaf_size\":128,\"filter_mode\":\"none\"}",
            cfg,
        ),
        "range_filtered" => snapshot_check(
            "range_filtered",
            "{\"hnsw_m\":16,\"ef_search\":100,\"filter_mode\":\"none\"}",
            cfg,
        ),
        "kdtree" => tree_snapshot_checks(!cfg.is_euclidean && dim <= 50, "kdtree", cfg),
        "balltree" => tree_snapshot_checks(!cfg.is_euclidean, "balltree", cfg),
        "rptree" => tree_snapshot_checks(!cfg.is_euclidean, "rptree", cfg),
        "rp_forest" => rp_forest_snapshot_checks(!cfg.is_euclidean, cfg),
        "kmeans_tree" => kmeans_tree_snapshot_checks(cfg),
        "brute" => vec![ExpectedResult::any_params("brute")],
        "sparse_mips" => Vec::new(),
        _ => Vec::new(),
    }
}

pub(crate) fn tree_params_json(max_leaf_size: usize, max_depth: usize) -> String {
    format!(
        "{{\"max_leaf_size\":{},\"max_depth\":{}}}",
        max_leaf_size, max_depth
    )
}

pub(crate) fn rp_forest_params_json(num_trees: usize, max_leaf_size: usize) -> String {
    format!(
        "{{\"num_trees\":{},\"max_leaf_size\":{}}}",
        num_trees, max_leaf_size
    )
}

pub(crate) fn kmeans_tree_params_json(
    num_clusters: usize,
    max_leaf_size: usize,
    max_depth: usize,
    max_iterations: usize,
    search_branches: usize,
) -> String {
    let mut params = format!(
        "{{\"num_clusters\":{},\"max_leaf_size\":{},\"max_depth\":{},\"max_iterations\":{}}}",
        num_clusters, max_leaf_size, max_depth, max_iterations
    );
    if search_branches != 1 {
        params.pop();
        params.push_str(&format!(",\"search_branches\":{search_branches}}}"));
    }
    params
}

pub(crate) fn kmeans_tree_leaf_budget_params_json(
    num_clusters: usize,
    max_leaf_size: usize,
    max_depth: usize,
    max_iterations: usize,
    leaf_budget: usize,
) -> String {
    format!(
        concat!(
            "{{\"num_clusters\":{},\"max_leaf_size\":{},\"max_depth\":{},",
            "\"max_iterations\":{},\"search_policy\":\"leaf_budget\",",
            "\"leaf_budget\":{}}}"
        ),
        num_clusters, max_leaf_size, max_depth, max_iterations, leaf_budget
    )
}

fn diskann_params_json(cfg: &Config, ef_search: usize, storage: &str) -> String {
    format!(
        "{{\"m\":{},\"ef_construction\":{},\"alpha\":1.2,\"ef_search\":{},\"storage\":\"{}\"}}",
        cfg.m, cfg.ef_construction, ef_search, storage
    )
}

pub(crate) fn store_flush_threshold(train_len: usize) -> usize {
    (train_len / 8).clamp(1, 10_000)
}

pub(crate) fn store_params_json(cfg: &Config, train_len: usize, ef_search: usize) -> String {
    format!(
        "{{\"m\":{},\"m_max\":{},\"flush_threshold\":{},\"ef_search\":{}}}",
        cfg.m,
        cfg.m * 2,
        store_flush_threshold(train_len),
        ef_search
    )
}

pub(crate) fn request_completed(
    completed: &CompletedResults,
    algo: &str,
    cfg: &Config,
    dim: usize,
    train_len: usize,
    test_len: usize,
) -> bool {
    // The dense HDF5 harness cannot construct SparseVector inputs. The runner
    // reports a skip and emits no JSONL row until a sparse dataset harness exists.
    if algo == "sparse_mips" {
        return true;
    }

    let checks = required_result_checks(algo, cfg, dim, train_len, test_len);
    !checks.is_empty()
        && checks
            .iter()
            .all(|expected| completed.lines.iter().any(|line| expected.matches(line)))
}

pub(crate) fn emit_result(results_path: &Path, line: &str) {
    println!("{}", line);
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(results_path)
    {
        let _ = writeln!(f, "{}", line);
    }
}

pub(crate) fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_string())
        .filter(|version| !version.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn parse_usize_list(value: &str, fallback: &[usize]) -> Vec<usize> {
    let parsed: Vec<usize> = value
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&n| n > 0)
        .collect();
    if parsed.is_empty() {
        fallback.to_vec()
    } else {
        parsed
    }
}

pub(crate) fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config::default();
    let mut algos_set = false;
    let mut results_override = false;
    let mut fresh = false;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--algo" => {
                i += 1;
                if !algos_set {
                    cfg.algos.clear();
                    algos_set = true;
                }
                if i < args.len() {
                    cfg.algos.push(args[i].to_lowercase());
                }
            }
            "--m" => {
                i += 1;
                if i < args.len() {
                    cfg.m = args[i].parse().unwrap_or(16);
                }
            }
            "--ef-construction" => {
                i += 1;
                if i < args.len() {
                    cfg.ef_construction = args[i].parse().unwrap_or(200);
                }
            }
            "--ef-search" => {
                i += 1;
                if i < args.len() {
                    cfg.ef_search_values = args[i]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
            "--pq-codebooks" => {
                i += 1;
                if i < args.len() {
                    cfg.pq_num_codebooks = args[i].parse().ok();
                }
            }
            "--pq-clusters" => {
                i += 1;
                if i < args.len() {
                    cfg.pq_num_clusters = args[i].parse().ok();
                }
            }
            "--pq-codebook-size" => {
                i += 1;
                if i < args.len() {
                    cfg.pq_codebook_size = args[i].parse().unwrap_or(256);
                }
            }
            "--pq-training-sample-size" => {
                i += 1;
                if i < args.len() {
                    cfg.pq_training_sample_size = args[i].parse().ok();
                }
            }
            "--pq-kmeans-max-iter" => {
                i += 1;
                if i < args.len() {
                    cfg.pq_kmeans_max_iter = args[i].parse().unwrap_or(100);
                }
            }
            "--pq-nprobes" => {
                i += 1;
                if i < args.len() {
                    cfg.pq_nprobe_values = Some(parse_usize_list(&args[i], &[]));
                }
            }
            "--pq-rerank-pools" => {
                i += 1;
                if i < args.len() {
                    cfg.pq_rerank_pools = parse_usize_list(&args[i], &[]);
                }
            }
            "--tree-leaf-sizes" => {
                i += 1;
                if i < args.len() {
                    cfg.tree_leaf_sizes = parse_usize_list(&args[i], &[10]);
                }
            }
            "--tree-depths" => {
                i += 1;
                if i < args.len() {
                    cfg.tree_depths = parse_usize_list(&args[i], &[32]);
                }
            }
            "--rp-num-trees" => {
                i += 1;
                if i < args.len() {
                    cfg.rp_num_trees = parse_usize_list(&args[i], &[10]);
                }
            }
            "--kmeans-clusters" => {
                i += 1;
                if i < args.len() {
                    cfg.kmeans_clusters = parse_usize_list(&args[i], &[16]);
                }
            }
            "--kmeans-leaf-sizes" => {
                i += 1;
                if i < args.len() {
                    cfg.kmeans_leaf_sizes = parse_usize_list(&args[i], &[50]);
                }
            }
            "--kmeans-depths" => {
                i += 1;
                if i < args.len() {
                    cfg.kmeans_depths = parse_usize_list(&args[i], &[10]);
                }
            }
            "--kmeans-iters" => {
                i += 1;
                if i < args.len() {
                    cfg.kmeans_iters = parse_usize_list(&args[i], &[10]);
                }
            }
            "--kmeans-search-branches" => {
                i += 1;
                if i < args.len() {
                    cfg.kmeans_search_branches = parse_usize_list(&args[i], &[1]);
                }
            }
            "--kmeans-leaf-budgets" => {
                i += 1;
                if i < args.len() {
                    cfg.kmeans_leaf_budgets = parse_usize_list(&args[i], &[]);
                }
            }
            "--batch" => {
                cfg.batch = true;
            }
            "--resume" => {
                cfg.resume = true;
            }
            "--snapshot-load" => {
                cfg.snapshot_load = true;
            }
            "--max-train" => {
                i += 1;
                if i < args.len() {
                    cfg.max_train = args[i].parse().ok();
                }
            }
            "--max-queries" => {
                i += 1;
                if i < args.len() {
                    cfg.max_queries = args[i].parse().ok();
                }
            }
            "--warmup-queries" => {
                i += 1;
                if i < args.len() {
                    cfg.warmup_queries = args[i].parse().unwrap_or(DEFAULT_WARMUP_QUERIES);
                }
            }
            "--churn-base-size" => {
                i += 1;
                if i < args.len() {
                    cfg.churn_base_size = args[i].parse().unwrap_or(50_000);
                }
            }
            "--churn-cycles" => {
                i += 1;
                if i < args.len() {
                    cfg.churn_cycles = args[i].parse().unwrap_or(5_000);
                }
            }
            "--churn-queries" => {
                i += 1;
                if i < args.len() {
                    cfg.churn_queries = args[i].parse().unwrap_or(1_000);
                }
            }
            "--json" => {
                cfg.json = true;
            }
            "--results" => {
                i += 1;
                if i < args.len() {
                    cfg.results_path = PathBuf::from(&args[i]);
                    results_override = true;
                }
            }
            "--fresh" => {
                fresh = true;
            }
            arg if !arg.starts_with("--") => {
                cfg.data_dir = arg.to_string();
            }
            _ => {
                eprintln!("Unknown flag: {}", args[i]);
            }
        }
        i += 1;
    }

    if !results_override {
        cfg.results_path = default_results_path(&cfg.data_dir);
    }
    if fresh {
        std::fs::remove_file(&cfg.results_path).ok();
    }

    cfg.is_euclidean = cfg.data_dir.contains("euclidean");
    cfg
}

pub(crate) fn ivfpq_params_json(
    num_clusters: usize,
    num_codebooks: usize,
    codebook_size: usize,
    nprobe: usize,
    rerank_pool: Option<usize>,
    training_sample_size: Option<usize>,
    kmeans_max_iter: usize,
) -> String {
    let training_sample = training_sample_size
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string());
    let mut params = format!(
        "{{\"num_clusters\":{},\"num_codebooks\":{},\"codebook_size\":{},\"nprobe\":{},\"training_sample_size\":{},\"kmeans_max_iter\":{}",
        num_clusters, num_codebooks, codebook_size, nprobe, training_sample, kmeans_max_iter
    );
    if let Some(rerank_pool) = rerank_pool {
        params.push_str(&format!(",\"rerank_pool\":{}", rerank_pool));
    }
    params.push('}');
    params
}

pub(crate) struct BenchResult {
    pub(crate) recall_at_k: f64,
    pub(crate) qps: f64,
    pub(crate) latency_us: f64,
    pub(crate) p50_us: f64,
    pub(crate) p95_us: f64,
    pub(crate) p99_us: f64,
}

pub(crate) struct ResultStorage<'a> {
    pub(crate) storage_mode: &'a str,
    pub(crate) cache_state: &'a str,
    pub(crate) load_time_s: Option<f64>,
    pub(crate) index_bytes: Option<u64>,
    pub(crate) diagnostics: Option<StorageDiagnostics>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StorageDiagnostics {
    pub(crate) avg_visited_nodes: f64,
    pub(crate) avg_probed_lists: f64,
    pub(crate) avg_scanned_vectors: f64,
    pub(crate) avg_partition_reads: f64,
    pub(crate) avg_partition_bytes: f64,
    pub(crate) avg_graph_reads: f64,
    pub(crate) avg_code_reads: f64,
    pub(crate) avg_vector_reads: f64,
    pub(crate) avg_graph_bytes: f64,
    pub(crate) avg_code_bytes: f64,
    pub(crate) avg_vector_bytes: f64,
    pub(crate) avg_retained_candidates: f64,
}

impl Default for ResultStorage<'_> {
    fn default() -> Self {
        Self {
            storage_mode: "in_memory",
            cache_state: "warm_after_build",
            load_time_s: None,
            index_bytes: None,
            diagnostics: None,
        }
    }
}

#[cfg(test)]
fn storage_context_from_params(params: &str) -> ResultStorage<'static> {
    if params.contains("\"storage\":\"mmap\"") {
        ResultStorage {
            storage_mode: "mmap",
            cache_state: "warm_after_open",
            ..ResultStorage::default()
        }
    } else if params.contains("\"storage\":\"file\"") {
        ResultStorage {
            storage_mode: "file",
            cache_state: "warm_after_open",
            ..ResultStorage::default()
        }
    } else {
        ResultStorage::default()
    }
}

#[cfg(test)]
pub(crate) fn json_line(
    algorithm: &str,
    params: &str,
    build_time_s: f64,
    rss_kb: Option<u64>,
    result: &BenchResult,
) -> String {
    let storage = storage_context_from_params(params);
    json_line_with_storage(algorithm, params, build_time_s, rss_kb, result, &storage)
}

#[cfg(all(test, any(feature = "fresh_graph", feature = "hnsw")))]
pub(crate) fn json_line_with_extra_fields(
    algorithm: &str,
    params: &str,
    build_time_s: f64,
    rss_kb: Option<u64>,
    result: &BenchResult,
    extra_fields: &str,
) -> String {
    let storage = storage_context_from_params(params);
    json_line_with_storage_and_extra_fields(
        algorithm,
        params,
        build_time_s,
        rss_kb,
        result,
        &storage,
        extra_fields,
    )
}

#[cfg(any(feature = "fresh_graph", feature = "hnsw"))]
pub(crate) fn json_line_with_storage_and_extra_fields(
    algorithm: &str,
    params: &str,
    build_time_s: f64,
    rss_kb: Option<u64>,
    result: &BenchResult,
    storage: &ResultStorage<'_>,
    extra_fields: &str,
) -> String {
    let mut line = json_line_with_storage(algorithm, params, build_time_s, rss_kb, result, storage);
    append_extra_fields(&mut line, extra_fields);
    line
}

pub(crate) fn json_line_with_storage(
    algorithm: &str,
    params: &str,
    build_time_s: f64,
    rss_kb: Option<u64>,
    result: &BenchResult,
    storage: &ResultStorage<'_>,
) -> String {
    let mut s = format!(
        "{{\"algorithm\":\"{}\",\"params\":{},\"storage_mode\":\"{}\",\"cache_state\":\"{}\",\"recall_at_10\":{:.4},\"qps\":{:.1},\"build_time_s\":{:.2},\"latency_us\":{:.1},\"p50_us\":{:.1},\"p95_us\":{:.1},\"p99_us\":{:.1}",
        algorithm,
        params,
        storage.storage_mode,
        storage.cache_state,
        result.recall_at_k,
        result.qps,
        build_time_s,
        result.latency_us,
        result.p50_us,
        result.p95_us,
        result.p99_us
    );
    if let Some(load_time_s) = storage.load_time_s {
        s.push_str(&format!(",\"load_time_s\":{:.4}", load_time_s));
    }
    if let Some(bytes) = storage.index_bytes {
        s.push_str(&format!(",\"index_bytes\":{}", bytes));
    }
    if let Some(diagnostics) = storage.diagnostics {
        s.push_str(&format!(
            ",\"avg_visited_nodes\":{:.2},\"avg_probed_lists\":{:.2},\"avg_scanned_vectors\":{:.2},\"avg_partition_reads\":{:.2},\"avg_partition_bytes\":{:.2},\"avg_graph_reads\":{:.2},\"avg_code_reads\":{:.2},\"avg_vector_reads\":{:.2},\"avg_graph_bytes\":{:.2},\"avg_code_bytes\":{:.2},\"avg_vector_bytes\":{:.2},\"avg_retained_candidates\":{:.2}",
            diagnostics.avg_visited_nodes,
            diagnostics.avg_probed_lists,
            diagnostics.avg_scanned_vectors,
            diagnostics.avg_partition_reads,
            diagnostics.avg_partition_bytes,
            diagnostics.avg_graph_reads,
            diagnostics.avg_code_reads,
            diagnostics.avg_vector_reads,
            diagnostics.avg_graph_bytes,
            diagnostics.avg_code_bytes,
            diagnostics.avg_vector_bytes,
            diagnostics.avg_retained_candidates
        ));
    }
    if let Some(kb) = rss_kb {
        s.push_str(&format!(",\"rss_kb\":{}", kb));
    }
    s.push('}');
    s
}

#[cfg(any(feature = "fresh_graph", feature = "hnsw"))]
fn append_extra_fields(line: &mut String, extra_fields: &str) {
    let fields = extra_fields.trim().trim_start_matches(',');
    if fields.is_empty() || !line.ends_with('}') {
        return;
    }
    line.pop();
    line.push(',');
    line.push_str(fields);
    line.push('}');
}

pub(crate) fn evaluate(
    search_fn: &dyn Fn(&[f32], usize) -> Vec<(u32, f32)>,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    k: usize,
) -> BenchResult {
    let warmup_count = warmup_queries().min(test.len());
    for query in test.iter().take(warmup_count) {
        let _ = search_fn(query, k);
    }

    let mut total_recall = 0.0;
    let mut latencies_us: Vec<f64> = Vec::with_capacity(test.len());

    for (i, query) in test.iter().enumerate() {
        let q_start = Instant::now();
        let results = search_fn(query, k);
        let q_elapsed = q_start.elapsed();
        latencies_us.push(q_elapsed.as_nanos() as f64 / 1000.0);

        let gt_set: HashSet<u32> = neighbors[i].iter().take(k).map(|&n| n as u32).collect();
        let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
        total_recall += gt_set.intersection(&found).count() as f64 / k as f64;
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = latencies_us.len();
    let total_us: f64 = latencies_us.iter().sum();

    BenchResult {
        recall_at_k: total_recall / n as f64,
        qps: n as f64 / (total_us / 1_000_000.0),
        latency_us: total_us / n as f64,
        p50_us: latencies_us[n / 2],
        p95_us: latencies_us[(n as f64 * 0.95) as usize],
        p99_us: latencies_us[(n as f64 * 0.99) as usize],
    }
}

#[cfg(feature = "parallel")]
pub(crate) fn evaluate_parallel<F>(
    search_fn: F,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    k: usize,
) -> BenchResult
where
    F: Fn(&[f32], usize) -> Vec<(u32, f32)> + Sync,
{
    use rayon::prelude::*;

    let warmup_count = warmup_queries().min(test.len());
    for query in test.iter().take(warmup_count) {
        let _ = search_fn(query, k);
    }

    let batch_start = Instant::now();
    let mut per_query: Vec<(f64, f64)> = test
        .par_iter()
        .enumerate()
        .map(|(i, query)| {
            let q_start = Instant::now();
            let results = search_fn(query, k);
            let latency_us = q_start.elapsed().as_nanos() as f64 / 1000.0;

            let gt_set: HashSet<u32> = neighbors[i].iter().take(k).map(|&n| n as u32).collect();
            let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
            let recall = gt_set.intersection(&found).count() as f64 / k as f64;
            (latency_us, recall)
        })
        .collect();
    let elapsed_s = batch_start.elapsed().as_secs_f64();

    per_query.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let n = per_query.len();
    let total_latency_us: f64 = per_query.iter().map(|(latency, _)| *latency).sum();
    let total_recall: f64 = per_query.iter().map(|(_, recall)| *recall).sum();

    BenchResult {
        recall_at_k: total_recall / n as f64,
        qps: n as f64 / elapsed_s,
        latency_us: total_latency_us / n as f64,
        p50_us: per_query[n / 2].0,
        p95_us: per_query[(n as f64 * 0.95) as usize].0,
        p99_us: per_query[(n as f64 * 0.99) as usize].0,
    }
}

pub(crate) fn current_rss_kb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&output.stdout);
        s.trim().parse::<u64>().ok()
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb_str = rest.trim().trim_end_matches(" kB").trim();
                return kb_str.parse::<u64>().ok();
            }
        }
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(any(
    feature = "balltree",
    feature = "binary_index",
    feature = "curator",
    feature = "diskann",
    feature = "emg",
    feature = "finger",
    feature = "filtered_graph",
    feature = "fresh_graph",
    feature = "ivf_avq",
    feature = "ivf_pq",
    feature = "ivf_rabitq",
    feature = "kdtree",
    feature = "kmeans_tree",
    feature = "nsg",
    feature = "nsw",
    feature = "pipnn",
    all(feature = "range_filtered", feature = "hnsw"),
    feature = "lsh",
    feature = "rp_quant",
    feature = "rptree",
    feature = "sng",
    feature = "sq4",
    feature = "sq8",
    feature = "store",
    all(feature = "hnsw", feature = "ivf_rabitq", feature = "serde"),
    feature = "vamana"
))]
pub(crate) fn dir_size_bytes(path: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += dir_size_bytes(&entry.path())?;
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}

pub(crate) fn brute_force_search(
    train: &[Vec<f32>],
    query: &[f32],
    k: usize,
    metric: vicinity::DistanceMetric,
) -> Vec<(u32, f32)> {
    let mut dists: Vec<(u32, f32)> = train
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, metric.distance(query, v)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.truncate(k);
    dists
}

pub(crate) fn brute_force_search_ids(
    train: &[Vec<f32>],
    active_ids: &[u32],
    query: &[f32],
    k: usize,
    metric: vicinity::DistanceMetric,
) -> Vec<i32> {
    let mut dists: Vec<(u32, f32)> = active_ids
        .iter()
        .map(|&id| (id, metric.distance(query, &train[id as usize])))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.into_iter().take(k).map(|(id, _)| id as i32).collect()
}

pub(crate) fn brute_force_neighbors_for_ids(
    train: &[Vec<f32>],
    active_ids: &[u32],
    test: &[Vec<f32>],
    k: usize,
    metric: vicinity::DistanceMetric,
) -> Vec<Vec<i32>> {
    test.iter()
        .map(|query| brute_force_search_ids(train, active_ids, query, k, metric))
        .collect()
}

pub(crate) fn print_header() {
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "param", "Recall@10", "QPS", "p50(us)", "p95(us)", "p99(us)"
    );
    println!("{}", "-".repeat(65));
}

pub(crate) fn print_row(param_label: &str, result: &BenchResult) {
    println!(
        "{:>10} {:>9.1}% {:>9.0} {:>9.0} {:>9.0} {:>9.0}",
        param_label,
        result.recall_at_k * 100.0,
        result.qps,
        result.p50_us,
        result.p95_us,
        result.p99_us
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diskann_line(algorithm: &str, storage: &str) -> String {
        let storage_mode = if storage == "memory" {
            "in_memory"
        } else {
            storage
        };
        format!(
            "{{\"algorithm\":\"{}\",\"params\":{{\"m\":16,\"ef_construction\":200,\"alpha\":1.2,\"ef_search\":10,\"storage\":\"{}\"}},\"storage_mode\":\"{}\",\"recall_at_10\":1.0,\"qps\":1.0}}",
            algorithm, storage, storage_mode
        )
    }

    fn legacy_diskann_line_without_storage_mode(algorithm: &str, storage: &str) -> String {
        format!(
            "{{\"algorithm\":\"{}\",\"params\":{{\"m\":16,\"ef_construction\":200,\"alpha\":1.2,\"ef_search\":10,\"storage\":\"{}\"}},\"recall_at_10\":1.0,\"qps\":1.0}}",
            algorithm, storage
        )
    }

    fn single_line(algorithm: &str, params: &str) -> String {
        format!(
            "{{\"algorithm\":\"{}\",\"params\":{},\"storage_mode\":\"in_memory\",\"recall_at_10\":1.0,\"qps\":1.0}}",
            algorithm, params
        )
    }

    fn legacy_single_line_without_storage_mode(algorithm: &str, params: &str) -> String {
        format!(
            "{{\"algorithm\":\"{}\",\"params\":{},\"recall_at_10\":1.0,\"qps\":1.0}}",
            algorithm, params
        )
    }

    fn single_line_with_storage(algorithm: &str, params: &str, storage_mode: &str) -> String {
        format!(
            "{{\"algorithm\":\"{}\",\"params\":{},\"storage_mode\":\"{}\",\"recall_at_10\":1.0,\"qps\":1.0}}",
            algorithm, params, storage_mode
        )
    }

    fn lines_for_storage_modes(
        algorithm: &str,
        params: &str,
        modes: impl IntoIterator<Item = &'static str>,
    ) -> Vec<String> {
        modes
            .into_iter()
            .map(|storage_mode| single_line_with_storage(algorithm, params, storage_mode))
            .collect()
    }

    fn snapshot_lines(algorithm: &str, params: &str) -> Vec<String> {
        lines_for_storage_modes(algorithm, params, ["in_memory", "snapshot_loaded"])
    }

    fn file_open_lines(algorithm: &str, params: &str) -> Vec<String> {
        lines_for_storage_modes(
            algorithm,
            params,
            std::iter::once("in_memory").chain(open_storage_modes().iter().copied()),
        )
    }

    #[test]
    fn algorithm_options_are_unique_and_resume_checked() {
        let cfg = Config {
            ef_search_values: vec![10],
            ..Config::default()
        };
        let mut seen = HashSet::new();

        for algorithm in ALGORITHM_OPTIONS {
            assert!(
                seen.insert(*algorithm),
                "duplicate algorithm option: {algorithm}"
            );
            let checks = required_result_checks(algorithm, &cfg, 25, 60_000, 1_000);
            if *algorithm == "sparse_mips" {
                assert!(
                    checks.is_empty(),
                    "dense harness should not expect sparse_mips rows"
                );
            } else {
                assert!(
                    !checks.is_empty(),
                    "missing resume checks for algorithm option: {algorithm}"
                );
            }
        }
    }

    #[test]
    fn active_features_json_matches_representative_cfgs() {
        let features = active_features_json();

        assert!(features.starts_with('['));
        assert!(features.ends_with(']'));
        assert_eq!(features.contains("\"hnsw\""), cfg!(feature = "hnsw"));
        assert_eq!(features.contains("\"ivf_pq\""), cfg!(feature = "ivf_pq"));
        assert_eq!(features.contains("\"diskann\""), cfg!(feature = "diskann"));
        assert_eq!(
            features.contains("\"persistence\""),
            cfg!(feature = "persistence")
        );
    }

    #[test]
    fn default_config_records_warmup_query_count() {
        assert_eq!(Config::default().warmup_queries, DEFAULT_WARMUP_QUERIES);
    }

    #[test]
    fn unknown_algorithms_are_not_resume_complete() {
        let cfg = Config::default();
        let completed = CompletedResults {
            lines: vec![single_line("unknown_algorithm", "{}")],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &completed,
            "unknown_algorithm",
            &cfg,
            25,
            2_000,
            200
        ));
    }

    fn sample_result() -> BenchResult {
        BenchResult {
            recall_at_k: 1.0,
            qps: 10.0,
            latency_us: 100.0,
            p50_us: 90.0,
            p95_us: 150.0,
            p99_us: 200.0,
        }
    }

    #[test]
    fn json_line_records_default_storage_context() {
        let line = json_line(
            "hnsw",
            "{\"m\":16,\"ef_search\":10}",
            1.0,
            Some(123),
            &sample_result(),
        );

        assert!(line.contains("\"storage_mode\":\"in_memory\""));
        assert!(line.contains("\"cache_state\":\"warm_after_build\""));
        assert!(line.contains("\"rss_kb\":123"));
    }

    #[cfg(any(feature = "fresh_graph", feature = "hnsw"))]
    #[test]
    fn json_line_with_extra_fields_appends_top_level_metrics() {
        let line = json_line_with_extra_fields(
            "inplace_churn",
            "{\"cycles\":8}",
            1.0,
            None,
            &sample_result(),
            "\"update_qps\":12.5,\"active_count\":64",
        );

        assert!(line.contains("\"params\":{\"cycles\":8}"));
        assert!(line.contains("\"update_qps\":12.5"));
        assert!(line.contains("\"active_count\":64"));
        assert!(line.ends_with('}'));
    }

    #[cfg(any(feature = "fresh_graph", feature = "hnsw"))]
    #[test]
    fn json_line_with_storage_and_extra_fields_keeps_index_bytes() {
        let storage = ResultStorage {
            index_bytes: Some(4096),
            ..ResultStorage::default()
        };
        let line = json_line_with_storage_and_extra_fields(
            "lsm_churn",
            "{\"cycles\":8}",
            1.0,
            None,
            &sample_result(),
            &storage,
            "\"update_qps\":12.5",
        );

        assert!(line.contains("\"index_bytes\":4096"));
        assert!(line.contains("\"update_qps\":12.5"));
        assert!(line.ends_with('}'));
    }

    #[test]
    fn json_line_promotes_diskann_storage_context() {
        let storage = ResultStorage {
            storage_mode: "mmap",
            cache_state: "warm_after_open",
            load_time_s: Some(0.125),
            index_bytes: Some(4096),
            diagnostics: Some(StorageDiagnostics {
                avg_visited_nodes: 12.0,
                avg_probed_lists: 3.0,
                avg_scanned_vectors: 128.0,
                avg_partition_reads: 3.0,
                avg_partition_bytes: 640.0,
                avg_graph_reads: 8.5,
                avg_code_reads: 4.0,
                avg_vector_reads: 12.0,
                avg_graph_bytes: 544.0,
                avg_code_bytes: 512.0,
                avg_vector_bytes: 1200.0,
                avg_retained_candidates: 10.0,
            }),
        };
        let line = json_line_with_storage(
            "diskann_mmap",
            "{\"storage\":\"mmap\"}",
            2.0,
            None,
            &sample_result(),
            &storage,
        );

        assert!(line.contains("\"storage_mode\":\"mmap\""));
        assert!(line.contains("\"cache_state\":\"warm_after_open\""));
        assert!(line.contains("\"load_time_s\":0.1250"));
        assert!(line.contains("\"index_bytes\":4096"));
        assert!(line.contains("\"avg_visited_nodes\":12.00"));
        assert!(line.contains("\"avg_probed_lists\":3.00"));
        assert!(line.contains("\"avg_scanned_vectors\":128.00"));
        assert!(line.contains("\"avg_partition_reads\":3.00"));
        assert!(line.contains("\"avg_partition_bytes\":640.00"));
        assert!(line.contains("\"avg_graph_reads\":8.50"));
        assert!(line.contains("\"avg_code_reads\":4.00"));
        assert!(line.contains("\"avg_code_bytes\":512.00"));
        assert!(line.contains("\"avg_vector_bytes\":1200.00"));
    }

    #[test]
    fn nprobe_values_use_default_sweep_bounded_by_partition_count() {
        let cfg = Config::default();

        assert_eq!(nprobe_values(&cfg, 12), vec![1, 2, 5, 10]);
    }

    #[test]
    fn nprobe_values_use_explicit_positive_bounded_values() {
        let cfg = Config {
            pq_nprobe_values: Some(vec![0, 3, 7, 20]),
            ..Config::default()
        };

        assert_eq!(nprobe_values(&cfg, 10), vec![3, 7]);
    }

    #[test]
    fn load_completed_results_matches_limit_metadata() {
        let file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file.as_file(),
            "{{\"_meta\":{{\"dataset\":\"data/a\",\"result_schema\":2,\"index_bytes_required\":true,\"train_limit\":1000,\"query_limit\":100}}}}"
        )
        .unwrap();
        writeln!(
            file.as_file(),
            "{{\"algorithm\":\"hnsw\",\"params\":{{\"m\":16,\"ef_construction\":200,\"ef_search\":10}},\"storage_mode\":\"in_memory\",\"recall_at_10\":1.0,\"qps\":1.0}}"
        )
        .unwrap();

        let matching = load_completed_results(file.path(), "data/a", Some(1000), Some(100));
        assert!(matching.has_matching_meta);
        assert_eq!(matching.counts.get("hnsw"), Some(&1));

        let mismatched_query = load_completed_results(file.path(), "data/a", Some(1000), Some(200));
        assert!(!mismatched_query.has_matching_meta);
        assert!(mismatched_query.has_mismatched_meta);
        assert!(mismatched_query.counts.is_empty());

        let mismatched_train = load_completed_results(file.path(), "data/a", Some(2000), Some(100));
        assert!(!mismatched_train.has_matching_meta);
        assert!(mismatched_train.has_mismatched_meta);
        assert!(mismatched_train.counts.is_empty());

        let mismatched_uncapped_train =
            load_completed_results(file.path(), "data/a", None, Some(100));
        assert!(!mismatched_uncapped_train.has_matching_meta);
        assert!(mismatched_uncapped_train.has_mismatched_meta);
        assert!(mismatched_uncapped_train.counts.is_empty());
    }

    #[test]
    fn load_completed_results_matches_warmup_metadata() {
        let file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file.as_file(),
            "{{\"_meta\":{{\"dataset\":\"data/a\",\"result_schema\":2,\"index_bytes_required\":true,\"train_limit\":1000,\"query_limit\":100,\"warmup_queries\":0}}}}"
        )
        .unwrap();
        writeln!(
            file.as_file(),
            "{{\"algorithm\":\"hnsw\",\"params\":{{\"m\":16,\"ef_construction\":200,\"ef_search\":10}},\"storage_mode\":\"in_memory\",\"recall_at_10\":1.0,\"qps\":1.0}}"
        )
        .unwrap();

        let matching =
            load_completed_results_with_warmup(file.path(), "data/a", Some(1000), Some(100), 0);
        assert!(matching.has_matching_meta);
        assert_eq!(matching.counts.get("hnsw"), Some(&1));

        let mismatched =
            load_completed_results_with_warmup(file.path(), "data/a", Some(1000), Some(100), 50);
        assert!(!mismatched.has_matching_meta);
        assert!(mismatched.has_mismatched_meta);
        assert!(mismatched.counts.is_empty());
    }

    #[test]
    fn load_completed_results_keeps_current_uncapped_metadata() {
        let file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file.as_file(),
            "{{\"_meta\":{{\"dataset\":\"data/a\",\"result_schema\":2,\"index_bytes_required\":true,\"query_limit\":100}}}}"
        )
        .unwrap();
        writeln!(
            file.as_file(),
            "{{\"algorithm\":\"hnsw\",\"params\":{{\"m\":16,\"ef_construction\":200,\"ef_search\":10}},\"storage_mode\":\"in_memory\",\"recall_at_10\":1.0,\"qps\":1.0}}"
        )
        .unwrap();

        let matching = load_completed_results(file.path(), "data/a", None, Some(100));
        assert!(matching.has_matching_meta);
        assert_eq!(matching.counts.get("hnsw"), Some(&1));

        let nondefault_warmup =
            load_completed_results_with_warmup(file.path(), "data/a", None, Some(100), 0);
        assert!(!nondefault_warmup.has_matching_meta);
        assert!(nondefault_warmup.has_mismatched_meta);
        assert!(nondefault_warmup.counts.is_empty());

        let mismatched = load_completed_results(file.path(), "data/a", Some(1000), Some(100));
        assert!(!mismatched.has_matching_meta);
        assert!(mismatched.has_mismatched_meta);
        assert!(mismatched.counts.is_empty());
    }

    #[test]
    fn load_completed_results_rejects_legacy_metadata_contract() {
        let file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file.as_file(),
            "{{\"_meta\":{{\"dataset\":\"data/a\",\"query_limit\":100}}}}"
        )
        .unwrap();
        writeln!(
            file.as_file(),
            "{{\"algorithm\":\"hnsw\",\"params\":{{\"m\":16,\"ef_construction\":200,\"ef_search\":10}},\"storage_mode\":\"in_memory\",\"recall_at_10\":1.0,\"qps\":1.0}}"
        )
        .unwrap();

        let completed = load_completed_results(file.path(), "data/a", None, Some(100));
        assert!(!completed.has_matching_meta);
        assert!(completed.has_mismatched_meta);
        assert!(completed.counts.is_empty());
    }

    #[test]
    fn load_completed_results_ignores_rows_under_mismatched_meta() {
        let file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file.as_file(),
            "{{\"_meta\":{{\"dataset\":\"data/a\",\"result_schema\":2,\"index_bytes_required\":true,\"query_limit\":100}}}}"
        )
        .unwrap();
        writeln!(
            file.as_file(),
            "{{\"algorithm\":\"hnsw\",\"params\":{{\"m\":16,\"ef_construction\":200,\"ef_search\":10}},\"storage_mode\":\"in_memory\",\"recall_at_10\":1.0,\"qps\":1.0}}"
        )
        .unwrap();
        writeln!(
            file.as_file(),
            "{{\"_meta\":{{\"dataset\":\"data/a\",\"result_schema\":2,\"index_bytes_required\":true,\"query_limit\":null}}}}"
        )
        .unwrap();
        writeln!(
            file.as_file(),
            "{{\"algorithm\":\"hnsw\",\"params\":{{\"m\":16,\"ef_construction\":200,\"ef_search\":20}},\"storage_mode\":\"in_memory\",\"recall_at_10\":1.0,\"qps\":1.0}}"
        )
        .unwrap();

        let cfg = Config {
            ef_search_values: vec![10],
            max_queries: Some(100),
            ..Config::default()
        };
        let completed =
            load_completed_results(file.path(), "data/a", cfg.max_train, cfg.max_queries);

        assert!(request_completed(&completed, "hnsw", &cfg, 25, 1_000, 100));

        let full_cfg = Config {
            ef_search_values: vec![20],
            ..Config::default()
        };
        let completed = load_completed_results(
            file.path(),
            "data/a",
            full_cfg.max_train,
            full_cfg.max_queries,
        );

        assert!(request_completed(
            &completed, "hnsw", &full_cfg, 25, 1_000, 100
        ));
        assert!(!request_completed(&completed, "hnsw", &cfg, 25, 1_000, 100));
    }

    #[test]
    fn diskann_resume_requires_memory_file_and_mmap_rows() {
        let cfg = Config {
            ef_search_values: vec![10],
            ..Config::default()
        };
        let completed = CompletedResults {
            lines: vec![
                diskann_line("diskann", "memory"),
                diskann_line("diskann_file", "file"),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &completed, "diskann", &cfg, 25, 1_000, 100
        ));
    }

    #[test]
    fn diskann_resume_accepts_all_storage_mode_rows() {
        let cfg = Config {
            ef_search_values: vec![10],
            ..Config::default()
        };
        let completed = CompletedResults {
            lines: vec![
                diskann_line("diskann", "memory"),
                diskann_line("diskann_file", "file"),
                diskann_line("diskann_mmap", "mmap"),
            ],
            ..CompletedResults::default()
        };

        assert!(request_completed(
            &completed, "diskann", &cfg, 25, 1_000, 100
        ));
    }

    #[test]
    fn diskann_resume_rejects_rows_without_storage_mode_context() {
        let cfg = Config {
            ef_search_values: vec![10],
            ..Config::default()
        };
        let completed = CompletedResults {
            lines: vec![
                legacy_diskann_line_without_storage_mode("diskann", "memory"),
                legacy_diskann_line_without_storage_mode("diskann_file", "file"),
                legacy_diskann_line_without_storage_mode("diskann_mmap", "mmap"),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &completed, "diskann", &cfg, 25, 1_000, 100
        ));
    }

    #[test]
    fn classic_tree_resume_accepts_matching_single_row() {
        let cfg = Config::default();
        let completed = CompletedResults {
            lines: vec![single_line(
                "kdtree",
                "{\"max_leaf_size\":10,\"max_depth\":32}",
            )],
            ..CompletedResults::default()
        };

        assert!(request_completed(
            &completed, "kdtree", &cfg, 25, 1_000, 100
        ));
    }

    #[test]
    fn classic_tree_resume_rejects_legacy_single_row_without_storage_mode() {
        let cfg = Config::default();
        let completed = CompletedResults {
            lines: vec![legacy_single_line_without_storage_mode(
                "kdtree",
                "{\"max_leaf_size\":10,\"max_depth\":32}",
            )],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &completed, "kdtree", &cfg, 25, 1_000, 100
        ));
    }

    #[test]
    fn classic_tree_snapshot_resume_requires_snapshot_storage_row() {
        let cfg = Config {
            snapshot_load: true,
            ..Config::default()
        };
        let params = "{\"max_leaf_size\":10,\"max_depth\":32}";
        let missing_snapshot = CompletedResults {
            lines: vec![single_line_with_storage("kdtree", params, "in_memory")],
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: vec![
                single_line_with_storage("kdtree", params, "in_memory"),
                single_line_with_storage("kdtree", params, "snapshot_loaded"),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_snapshot,
            "kdtree",
            &cfg,
            25,
            1_000,
            100
        ));
        assert!(request_completed(
            &completed, "kdtree", &cfg, 25, 1_000, 100
        ));
    }

    #[test]
    fn store_resume_requires_live_and_reopened_segmented_storage_rows() {
        let cfg = Config {
            ef_search_values: vec![10],
            ..Config::default()
        };
        let params = store_params_json(&cfg, 64, 10);
        let missing_segmented = CompletedResults {
            lines: vec![single_line_with_storage("store", &params, "in_memory")],
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: vec![single_line_with_storage(
                "store",
                &params,
                "segmented_store",
            )],
            ..CompletedResults::default()
        };
        let completed_with_snapshot = CompletedResults {
            lines: vec![
                single_line_with_storage("store", &params, "segmented_store"),
                single_line_with_storage("store_snapshot", &params, "segmented_store"),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_segmented,
            "store",
            &cfg,
            8,
            64,
            12
        ));
        assert!(!request_completed(&completed, "store", &cfg, 8, 64, 12));
        assert!(request_completed(
            &completed_with_snapshot,
            "store",
            &cfg,
            8,
            64,
            12
        ));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn inplace_snapshot_resume_requires_snapshot_storage_rows() {
        let cfg = Config {
            snapshot_load: true,
            ef_search_values: vec![10],
            ..Config::default()
        };
        let params = "{\"max_degree\":32,\"build_beam_width\":200,\"beam_width\":10}";
        let missing_snapshot = CompletedResults {
            lines: vec![single_line_with_storage("inplace", params, "in_memory")],
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: vec![
                single_line_with_storage("inplace", params, "in_memory"),
                single_line_with_storage("inplace", params, "snapshot_loaded"),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_snapshot,
            "inplace",
            &cfg,
            25,
            1_000,
            100
        ));
        assert!(request_completed(
            &completed, "inplace", &cfg, 25, 1_000, 100
        ));
    }

    #[test]
    fn ivf_snapshot_resume_requires_expected_storage_rows() {
        let cfg = Config {
            snapshot_load: true,
            pq_nprobe_values: Some(vec![1]),
            ..Config::default()
        };
        let avq_params = "{\"num_partitions\":256,\"num_codebooks\":16,\"codebook_size\":256,\"nprobe\":1,\"num_reorder\":100}";
        let rabitq_params = "{\"num_clusters\":256,\"total_bits\":4,\"nprobe\":1}";
        let missing_snapshot = CompletedResults {
            lines: vec![
                single_line_with_storage("ivf_avq", avq_params, "in_memory"),
                single_line_with_storage("ivf_rabitq", rabitq_params, "in_memory"),
            ],
            ..CompletedResults::default()
        };
        let missing_file = CompletedResults {
            lines: vec![
                single_line_with_storage("ivf_avq", avq_params, "in_memory"),
                single_line_with_storage("ivf_avq", avq_params, "snapshot_loaded"),
                single_line_with_storage("ivf_rabitq", rabitq_params, "in_memory"),
                single_line_with_storage("ivf_rabitq", rabitq_params, "snapshot_loaded"),
            ],
            ..CompletedResults::default()
        };
        let mut completed_lines = file_open_lines("ivf_avq", avq_params);
        completed_lines.extend(snapshot_lines("ivf_rabitq", rabitq_params));
        let completed = CompletedResults {
            lines: completed_lines,
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_snapshot,
            "ivf_avq",
            &cfg,
            128,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_file,
            "ivf_avq",
            &cfg,
            128,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_snapshot,
            "ivf_rabitq",
            &cfg,
            128,
            2_000,
            200
        ));
        assert!(request_completed(
            &completed, "ivf_avq", &cfg, 128, 2_000, 200
        ));
        assert!(request_completed(
            &completed,
            "ivf_rabitq",
            &cfg,
            128,
            2_000,
            200
        ));
    }

    #[test]
    fn ivf_avq_resume_requires_each_reorder_pool() {
        let cfg = Config {
            snapshot_load: true,
            pq_nprobe_values: Some(vec![1]),
            pq_rerank_pools: vec![50, 100],
            ..Config::default()
        };
        let reorder_50_params = ivfavq_params_json(256, 16, 256, 1, 50);
        let reorder_100_params = ivfavq_params_json(256, 16, 256, 1, 100);
        let missing_second_reorder = CompletedResults {
            lines: file_open_lines("ivf_avq", &reorder_50_params),
            ..CompletedResults::default()
        };
        let mut completed_lines = file_open_lines("ivf_avq", &reorder_50_params);
        completed_lines.extend(file_open_lines("ivf_avq", &reorder_100_params));
        let completed = CompletedResults {
            lines: completed_lines,
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_second_reorder,
            "ivf_avq",
            &cfg,
            128,
            2_000,
            200
        ));
        assert!(request_completed(
            &completed, "ivf_avq", &cfg, 128, 2_000, 200
        ));
    }

    #[test]
    fn ivfpq_resume_requires_file_storage_rows() {
        let cfg = Config {
            snapshot_load: true,
            pq_num_clusters: Some(4),
            pq_num_codebooks: Some(4),
            pq_codebook_size: 16,
            pq_nprobe_values: Some(vec![1]),
            pq_rerank_pools: vec![20],
            ..Config::default()
        };
        let approx_params = ivfpq_params_json(4, 4, 16, 1, None, None, 100);
        let rerank_params = ivfpq_params_json(4, 4, 16, 1, Some(20), None, 100);
        let mut missing_file_lines = snapshot_lines("ivfpq", &approx_params);
        missing_file_lines.extend(snapshot_lines("ivfpq_rerank", &rerank_params));
        let missing_file = CompletedResults {
            lines: missing_file_lines,
            ..CompletedResults::default()
        };
        let mut completed_lines = file_open_lines("ivfpq", &approx_params);
        completed_lines.extend(file_open_lines("ivfpq_rerank", &rerank_params));
        let completed = CompletedResults {
            lines: completed_lines,
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_file,
            "ivfpq",
            &cfg,
            16,
            2_000,
            200
        ));
        assert!(request_completed(&completed, "ivfpq", &cfg, 16, 2_000, 200));
    }

    #[test]
    fn flat_quant_snapshot_resume_requires_snapshot_storage_rows() {
        let cfg = Config {
            snapshot_load: true,
            ..Config::default()
        };
        let rp_params = "{\"projected_dim\":25,\"rerank_factor\":10}";
        let binary_params = "{\"rerank_factor\":10}";
        let missing_snapshot = CompletedResults {
            lines: vec![
                single_line_with_storage("rp_quant", rp_params, "in_memory"),
                single_line_with_storage("binary_index", binary_params, "in_memory"),
            ],
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: vec![
                single_line_with_storage("rp_quant", rp_params, "in_memory"),
                single_line_with_storage("rp_quant", rp_params, "snapshot_loaded"),
                single_line_with_storage("binary_index", binary_params, "in_memory"),
                single_line_with_storage("binary_index", binary_params, "snapshot_loaded"),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_snapshot,
            "rp_quant",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_snapshot,
            "binary_index",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(
            &completed, "rp_quant", &cfg, 25, 2_000, 200
        ));
        assert!(request_completed(
            &completed,
            "binary_index",
            &cfg,
            25,
            2_000,
            200
        ));
    }

    #[test]
    fn lsh_snapshot_resume_requires_snapshot_storage_rows() {
        let cfg = Config {
            snapshot_load: true,
            ..Config::default()
        };
        let params: Vec<String> = [8, 16, 32]
            .into_iter()
            .flat_map(|num_tables| {
                [2, 4, 8, 16].into_iter().map(move |num_probes| {
                    format!(
                        "{{\"num_tables\":{},\"num_probes\":{}}}",
                        num_tables, num_probes
                    )
                })
            })
            .collect();
        let missing_snapshot = CompletedResults {
            lines: params
                .iter()
                .map(|params| single_line_with_storage("lsh", params, "in_memory"))
                .collect(),
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: params
                .iter()
                .flat_map(|params| {
                    [
                        single_line_with_storage("lsh", params, "in_memory"),
                        single_line_with_storage("lsh", params, "snapshot_loaded"),
                    ]
                })
                .collect(),
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_snapshot,
            "lsh",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(&completed, "lsh", &cfg, 25, 2_000, 200));
    }

    #[test]
    fn graph_snapshot_resume_requires_snapshot_storage_rows() {
        let cfg = Config {
            snapshot_load: true,
            ef_search_values: vec![10],
            ..Config::default()
        };
        let rows = [
            ("nsw", "{\"m\":16,\"ef_search\":10}"),
            ("emg", "{\"max_degree\":32,\"ef_search\":10}"),
            ("nsg", "{\"max_degree\":32,\"ef_search\":10}"),
            (
                "pipnn",
                "{\"max_degree\":32,\"max_leaf_size\":2048,\"ef_search\":10}",
            ),
            ("vamana", "{\"ef_search\":10}"),
            (
                "finger",
                "{\"max_degree\":32,\"ef_search\":10,\"indexed_vectors\":2000,\"capped\":false}",
            ),
            ("sng", "{}"),
        ];
        let missing_snapshot = CompletedResults {
            lines: rows
                .iter()
                .map(|(algorithm, params)| single_line_with_storage(algorithm, params, "in_memory"))
                .collect(),
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: rows
                .iter()
                .flat_map(|(algorithm, params)| {
                    [
                        single_line_with_storage(algorithm, params, "in_memory"),
                        single_line_with_storage(algorithm, params, "snapshot_loaded"),
                    ]
                })
                .collect(),
            ..CompletedResults::default()
        };

        for (algorithm, _) in rows {
            assert!(!request_completed(
                &missing_snapshot,
                algorithm,
                &cfg,
                25,
                2_000,
                200
            ));
            assert!(request_completed(
                &completed, algorithm, &cfg, 25, 2_000, 200
            ));
        }
    }

    #[test]
    fn filtered_snapshot_resume_requires_snapshot_storage_rows() {
        let cfg = Config {
            snapshot_load: true,
            ef_search_values: vec![10],
            ..Config::default()
        };
        let rows = [
            ("fresh_graph", "{\"max_degree\":32,\"ef_search\":10}"),
            (
                "filtered_graph",
                "{\"max_degree\":32,\"ef_search\":10,\"filter_mode\":\"none\"}",
            ),
            (
                "curator",
                "{\"branching_factor\":16,\"max_leaf_size\":128,\"filter_mode\":\"none\"}",
            ),
            (
                "range_filtered",
                "{\"hnsw_m\":16,\"ef_search\":100,\"filter_mode\":\"none\"}",
            ),
        ];
        let missing_snapshot = CompletedResults {
            lines: rows
                .iter()
                .map(|(algorithm, params)| single_line_with_storage(algorithm, params, "in_memory"))
                .collect(),
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: rows
                .iter()
                .flat_map(|(algorithm, params)| {
                    [
                        single_line_with_storage(algorithm, params, "in_memory"),
                        single_line_with_storage(algorithm, params, "snapshot_loaded"),
                    ]
                })
                .collect(),
            ..CompletedResults::default()
        };

        for (algorithm, _) in rows {
            assert!(!request_completed(
                &missing_snapshot,
                algorithm,
                &cfg,
                25,
                2_000,
                200
            ));
            assert!(request_completed(
                &completed, algorithm, &cfg, 25, 2_000, 200
            ));
        }
    }

    #[test]
    fn sq_snapshot_resume_requires_snapshot_storage_rows() {
        let cfg = Config {
            snapshot_load: true,
            ef_search_values: vec![10],
            ..Config::default()
        };
        let sq4_params = "{\"rerank_factor\":10}";
        let sq_graph_params =
            "{\"m\":16,\"ef_construction\":200,\"ef_search\":10,\"rerank_pool\":100}";
        let missing_snapshot = CompletedResults {
            lines: vec![
                single_line_with_storage("sq4", sq4_params, "in_memory"),
                single_line_with_storage("sq4u", sq_graph_params, "in_memory"),
                single_line_with_storage("sq8u", sq_graph_params, "in_memory"),
            ],
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: vec![
                single_line_with_storage("sq4", sq4_params, "in_memory"),
                single_line_with_storage("sq4", sq4_params, "snapshot_loaded"),
                single_line_with_storage("sq4u", sq_graph_params, "in_memory"),
                single_line_with_storage("sq4u", sq_graph_params, "snapshot_loaded"),
                single_line_with_storage("sq8u", sq_graph_params, "in_memory"),
                single_line_with_storage("sq8u", sq_graph_params, "snapshot_loaded"),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_snapshot,
            "sq4",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_snapshot,
            "sq4u",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_snapshot,
            "sq8u",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(&completed, "sq4", &cfg, 25, 2_000, 200));
        assert!(request_completed(&completed, "sq4u", &cfg, 25, 2_000, 200));
        assert!(request_completed(&completed, "sq8u", &cfg, 25, 2_000, 200));
    }

    #[test]
    fn filtered_dense_resume_requires_unfiltered_mode_label() {
        let cfg = Config {
            ef_search_values: vec![50],
            ..Config::default()
        };
        let legacy_filtered_params = "{\"max_degree\":32,\"ef_search\":50}";
        let filtered_params = "{\"max_degree\":32,\"ef_search\":50,\"filter_mode\":\"none\"}";
        let curator_params =
            "{\"branching_factor\":16,\"max_leaf_size\":128,\"filter_mode\":\"none\"}";
        let range_params = "{\"hnsw_m\":16,\"ef_search\":100,\"filter_mode\":\"none\"}";
        let missing_filter_mode = CompletedResults {
            lines: vec![
                single_line("filtered_graph", legacy_filtered_params),
                single_line("curator", "{\"branching_factor\":16,\"max_leaf_size\":128}"),
                single_line("range_filtered", "{\"hnsw_m\":16,\"ef_search\":100}"),
            ],
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: vec![
                single_line("filtered_graph", filtered_params),
                single_line("curator", curator_params),
                single_line("range_filtered", range_params),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_filter_mode,
            "filtered_graph",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_filter_mode,
            "curator",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_filter_mode,
            "range_filtered",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(
            &completed,
            "filtered_graph",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(
            &completed, "curator", &cfg, 25, 2_000, 200
        ));
        assert!(request_completed(
            &completed,
            "range_filtered",
            &cfg,
            25,
            2_000,
            200
        ));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn hnsw_variant_resume_requires_snapshot_rows() {
        let cfg = Config {
            ef_search_values: vec![50],
            snapshot_load: true,
            ..Config::default()
        };
        let dual_params = "{\"m\":16,\"m_high_lid\":24,\"ef_construction\":200,\"ef_search\":50}";
        let deg_params =
            "{\"base_edges\":16,\"max_edges\":32,\"min_edges\":8,\"density_k\":10,\"alpha\":1.2,\"ef_search\":50,\"indexed_vectors\":2000,\"capped\":false}";
        let symphony_params = "{\"m\":16,\"ef_search\":50,\"rerank_pool\":100}";
        let symphony_vr_params =
            "{\"m\":16,\"ef_construction\":200,\"ef_search\":50,\"rerank_pool\":100}";
        let missing_snapshot = CompletedResults {
            lines: vec![
                single_line("dual_branch", dual_params),
                single_line("deg", deg_params),
                single_line("symphony_qg", symphony_params),
                single_line("symphony_qg_vr", symphony_vr_params),
            ],
            ..CompletedResults::default()
        };
        let completed = CompletedResults {
            lines: vec![
                single_line("dual_branch", dual_params),
                single_line_with_storage("dual_branch", dual_params, "snapshot_loaded"),
                single_line("deg", deg_params),
                single_line_with_storage("deg", deg_params, "snapshot_loaded"),
                single_line("symphony_qg", symphony_params),
                single_line_with_storage("symphony_qg", symphony_params, "snapshot_loaded"),
                single_line("symphony_qg_vr", symphony_vr_params),
                single_line_with_storage("symphony_qg_vr", symphony_vr_params, "snapshot_loaded"),
            ],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &missing_snapshot,
            "dual_branch",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_snapshot,
            "deg",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_snapshot,
            "symphony_qg",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(!request_completed(
            &missing_snapshot,
            "symphony_qg_vr",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(
            &completed,
            "dual_branch",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(&completed, "deg", &cfg, 25, 2_000, 200));
        assert!(request_completed(
            &completed,
            "symphony_qg",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(
            &completed,
            "symphony_qg_vr",
            &cfg,
            25,
            2_000,
            200
        ));
    }

    #[test]
    fn sparse_mips_resume_accepts_dense_dataset_skip() {
        let cfg = Config {
            snapshot_load: true,
            ..Config::default()
        };
        let completed = CompletedResults::default();

        assert!(request_completed(
            &completed,
            "sparse_mips",
            &cfg,
            25,
            50_000,
            1_000
        ));
    }

    #[cfg(not(feature = "serde"))]
    #[test]
    fn hnsw_variant_resume_does_not_require_disabled_snapshot_rows() {
        let cfg = Config {
            ef_search_values: vec![50],
            snapshot_load: true,
            ..Config::default()
        };
        let dual_params = "{\"m\":16,\"m_high_lid\":24,\"ef_construction\":200,\"ef_search\":50}";
        let deg_params =
            "{\"base_edges\":16,\"max_edges\":32,\"min_edges\":8,\"density_k\":10,\"alpha\":1.2,\"ef_search\":50,\"indexed_vectors\":2000,\"capped\":false}";
        let symphony_params = "{\"m\":16,\"ef_search\":50,\"rerank_pool\":100}";
        let symphony_vr_params =
            "{\"m\":16,\"ef_construction\":200,\"ef_search\":50,\"rerank_pool\":100}";
        let completed = CompletedResults {
            lines: vec![
                single_line("dual_branch", dual_params),
                single_line("deg", deg_params),
                single_line("symphony_qg", symphony_params),
                single_line("symphony_qg_vr", symphony_vr_params),
            ],
            ..CompletedResults::default()
        };

        assert!(request_completed(
            &completed,
            "dual_branch",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(&completed, "deg", &cfg, 25, 2_000, 200));
        assert!(request_completed(
            &completed,
            "symphony_qg",
            &cfg,
            25,
            2_000,
            200
        ));
        assert!(request_completed(
            &completed,
            "symphony_qg_vr",
            &cfg,
            25,
            2_000,
            200
        ));
    }

    #[test]
    fn classic_tree_resume_ignores_skipped_dataset_modes() {
        let cfg = Config {
            is_euclidean: true,
            ..Config::default()
        };
        let completed = CompletedResults {
            lines: vec![single_line(
                "kdtree",
                "{\"max_leaf_size\":10,\"max_depth\":32}",
            )],
            ..CompletedResults::default()
        };

        assert!(!request_completed(
            &completed, "kdtree", &cfg, 25, 1_000, 100
        ));
    }
}
