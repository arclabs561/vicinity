#![allow(clippy::expect_used, clippy::unwrap_used)]
//! ann-benchmarks compatible benchmark runner.
//!
//! Loads datasets converted by `scripts/download_ann_benchmarks.py` and
//! benchmarks multiple algorithms at various search parameters.
//!
//! ```bash
//! uv run scripts/download_ann_benchmarks.py glove-25-angular
//!
//! cargo run --example ann_benchmark --release --features hnsw -- data/ann-benchmarks/glove-25-angular
//!
//! cargo run --example ann_benchmark --release --features hnsw,nsw,ivf_pq,ivf_avq -- \
//!   data/ann-benchmarks/glove-25-angular --algo hnsw --algo nsw --algo ivfpq --algo ivf_avq --algo brute
//!
//! cargo run --example ann_benchmark --release --all-features -- \
//!   data/ann-benchmarks/glove-25-angular --algo hnsw --algo nsw --resume --json
//!
//! cargo run --example ann_benchmark --release --features hnsw -- \
//!   data/ann-benchmarks/glove-25-angular --algo hnsw --max-queries 1000 --json
//!
//! RAYON_NUM_THREADS=4 cargo run --example ann_benchmark --release --features hnsw,parallel -- \
//!   data/ann-benchmarks/glove-25-angular --algo hnsw --batch --json
//!
//! cargo run --example ann_benchmark --release --features ivf_pq,hnsw -- \
//!   data/ann-benchmarks/glove-25-angular --algo ivfpq \
//!   --pq-clusters 1024 --pq-codebooks 25 --pq-codebook-size 256 \
//!   --pq-training-sample-size 100000 --pq-kmeans-max-iter 20 \
//!   --pq-nprobes 16,32,64,128,256 --pq-rerank-pools 500,5000,20000
//!
//! cargo run --example ann_benchmark --release --features hnsw,fresh_graph -- \
//!   data/ann-benchmarks/glove-25-angular --algo fresh_graph_churn --json
//!
//! cargo run --example ann_benchmark --release --features hnsw -- \
//!   data/ann-benchmarks/glove-25-angular --algo inplace --algo inplace_churn --json
//!
//! cargo run --example ann_benchmark --release --features hnsw -- \
//!   data/ann-benchmarks/glove-25-angular --algo lsm_churn --json
//! ```

#[path = "common/mod.rs"]
mod common;
#[path = "ann_benchmark/external_hnsw_rs.rs"]
mod external_hnsw_rs;
#[cfg(any(
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "ivf_rabitq",
    feature = "rp_quant",
    feature = "binary_index",
    feature = "sq4",
    feature = "sq8",
    feature = "lsh"
))]
#[path = "ann_benchmark/quant.rs"]
mod quant;
#[path = "ann_benchmark/support.rs"]
mod support;
#[path = "ann_benchmark/usearch.rs"]
mod usearch_baseline;

use std::path::Path;

#[cfg(any(
    feature = "balltree",
    feature = "curator",
    feature = "diskann",
    feature = "emg",
    feature = "filtered_graph",
    feature = "finger",
    feature = "fresh_graph",
    feature = "hnsw",
    feature = "kdtree",
    feature = "kmeans_tree",
    feature = "nsg",
    feature = "nsw",
    feature = "pipnn",
    feature = "range_filtered",
    feature = "rptree",
    feature = "sng",
    feature = "store",
    feature = "vamana"
))]
use std::time::Instant;

use support::brute_force_neighbors_for_ids;
#[cfg(any(feature = "fresh_graph", feature = "hnsw"))]
use support::brute_force_search_ids;
#[cfg(any(
    feature = "balltree",
    feature = "curator",
    feature = "diskann",
    feature = "emg",
    feature = "finger",
    feature = "filtered_graph",
    feature = "fresh_graph",
    all(feature = "hnsw", feature = "serde"),
    feature = "kdtree",
    feature = "kmeans_tree",
    feature = "nsg",
    feature = "nsw",
    feature = "pipnn",
    all(feature = "range_filtered", feature = "hnsw"),
    feature = "rptree",
    feature = "sng",
    feature = "store",
    feature = "vamana"
))]
use support::dir_size_bytes;
#[cfg(feature = "parallel")]
use support::evaluate_parallel;
#[cfg(any(feature = "fresh_graph", feature = "hnsw"))]
use support::json_line_with_storage_and_extra_fields;
#[cfg(feature = "rptree")]
use support::rp_forest_params_json;
#[cfg(any(feature = "balltree", feature = "kdtree", feature = "rptree"))]
use support::tree_params_json;
use support::{
    active_features_json, algorithm_options_help, brute_force_search, cpu_model, current_rss_kb,
    emit_result, evaluate, help_requested, json_line_with_storage, load_completed_results_for_run,
    parse_args, print_header, print_row, request_completed, rustc_version, seed_fingerprint,
    set_run_identity, set_warmup_queries, should_emit_run_meta, Config, ResultStorage, HELP,
};
#[cfg(feature = "kmeans_tree")]
use support::{kmeans_tree_leaf_budget_params_json, kmeans_tree_params_json};
#[cfg(feature = "store")]
use support::{store_flush_threshold, store_params_json};
#[cfg(feature = "diskann")]
use support::{BenchResult, StorageDiagnostics};

#[cfg(any(
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "ivf_rabitq",
    feature = "rp_quant",
    feature = "binary_index",
    feature = "sq4",
    feature = "sq8",
    feature = "lsh"
))]
use quant::*;

#[cfg(any(
    feature = "balltree",
    feature = "curator",
    feature = "emg",
    feature = "finger",
    feature = "filtered_graph",
    feature = "fresh_graph",
    feature = "hnsw",
    feature = "kdtree",
    feature = "kmeans_tree",
    feature = "nsg",
    feature = "nsw",
    feature = "pipnn",
    all(feature = "range_filtered", feature = "hnsw"),
    feature = "rptree",
    feature = "sng",
    feature = "vamana"
))]
fn snapshot_storage(load_time_s: f64, index_bytes: Option<u64>) -> ResultStorage<'static> {
    ResultStorage {
        storage_mode: "snapshot_loaded",
        cache_state: "warm_after_load",
        load_time_s: Some(load_time_s),
        index_bytes,
        index_bytes_kind: Some("snapshot_bytes"),
        diagnostics: None,
    }
}

#[cfg(any(
    feature = "emg",
    feature = "finger",
    feature = "filtered_graph",
    feature = "fresh_graph",
    feature = "hnsw",
    feature = "balltree",
    feature = "curator",
    feature = "kdtree",
    feature = "kmeans_tree",
    feature = "nsg",
    feature = "nsw",
    feature = "pipnn",
    all(feature = "range_filtered", feature = "hnsw"),
    feature = "rptree",
    feature = "sng",
    feature = "vamana"
))]
fn in_memory_storage(index_bytes: Option<u64>) -> ResultStorage<'static> {
    ResultStorage {
        index_bytes,
        index_bytes_kind: Some("heap_estimate"),
        ..ResultStorage::default()
    }
}

// ─── Algorithm runners ───────────────────────────────────────────────────────

fn dataset_metric(cfg: &Config) -> vicinity::DistanceMetric {
    if cfg.is_euclidean {
        vicinity::DistanceMetric::L2
    } else {
        vicinity::DistanceMetric::Cosine
    }
}

#[cfg(any(
    feature = "filtered_graph",
    feature = "finger",
    feature = "fresh_graph",
    feature = "hnsw",
    feature = "nsg",
    feature = "sng"
))]
fn capped_neighbors_if_needed(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    indexed_len: usize,
) -> Option<Vec<Vec<i32>>> {
    if indexed_len == train.len() {
        return None;
    }
    let indexed = &train[..indexed_len];
    let active_ids: Vec<u32> = (0..indexed_len as u32).collect();
    Some(brute_force_neighbors_for_ids(
        indexed,
        &active_ids,
        test,
        10,
        dataset_metric(cfg),
    ))
}

#[cfg(feature = "hnsw")]
fn run_hnsw(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::{HNSWIndex, HNSWParams};

    let metric = if cfg.is_euclidean {
        vicinity::DistanceMetric::L2
    } else {
        vicinity::DistanceMetric::Cosine
    };
    let params = HNSWParams {
        m: cfg.m,
        m_max: cfg.m,
        ef_construction: cfg.ef_construction,
        metric,
        auto_normalize: !cfg.is_euclidean,
        seed: Some(cfg.seed),
        ..Default::default()
    };

    if !cfg.json {
        println!(
            "--- HNSW (M={}, ef_construction={}, metric={:?}) ---",
            cfg.m, cfg.ef_construction, metric
        );
    }

    let build_start = Instant::now();
    let mut index = HNSWIndex::with_params(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    #[cfg(feature = "serde")]
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for HNSW snapshot benchmark");
        let path = temp_dir.path().join("hnsw.json");
        index.save_to_file(&path).unwrap();
        let index_bytes = std::fs::metadata(&path).ok().map(|metadata| metadata.len());
        let load_start = Instant::now();
        let loaded = HNSWIndex::load_from_file(&path).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };
    #[cfg(not(feature = "serde"))]
    let snapshot_index: Option<(tempfile::TempDir, HNSWIndex, f64, Option<u64>)> = {
        if cfg.snapshot_load {
            eprintln!("hnsw --snapshot-load requested but serde feature is not enabled");
        }
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(&|q, k| index.search(q, k, ef).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{}}}",
                cfg.m, cfg.ef_construction, ef
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "hnsw",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }

        if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = &snapshot_index {
            let loaded_result = evaluate(
                &|q, k| loaded.search(q, k, ef).unwrap(),
                test,
                neighbors,
                10,
            );
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{}}}",
                cfg.m, cfg.ef_construction, ef
            );
            if cfg.json {
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "hnsw",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            } else {
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }

        #[cfg(feature = "parallel")]
        if cfg.batch {
            let result =
                evaluate_parallel(|q, k| index.search(q, k, ef).unwrap(), test, neighbors, 10);
            if cfg.json {
                let params_json = format!(
                    "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"threads\":{}}}",
                    cfg.m,
                    cfg.ef_construction,
                    ef,
                    rayon::current_num_threads()
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "hnsw_parallel",
                        &params_json,
                        build_time_s,
                        rss,
                        &result,
                        &in_memory_storage(index_bytes),
                    ),
                );
            } else {
                print_row(&format!("ef={} par", ef), &result);
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    if cfg.batch {
        eprintln!("hnsw --batch requested but parallel feature is not enabled");
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "hnsw")]
fn run_dual_branch(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::dual_branch::{DualBranchConfig, DualBranchHNSW};

    let params = DualBranchConfig {
        m: cfg.m,
        m_high_lid: (cfg.m + cfg.m / 2).max(cfg.m + 1),
        ef_construction: cfg.ef_construction,
        ef_search: 50,
        seed: Some(cfg.seed),
        ..Default::default()
    };

    if !cfg.json {
        println!(
            "--- DualBranchHNSW (m={}, m_high_lid={}, ef_c={}) ---",
            params.m, params.m_high_lid, params.ef_construction
        );
    }

    let flat: Vec<f32> = train.iter().flatten().copied().collect();
    let build_start = Instant::now();
    let mut index = DualBranchHNSW::new(dim, params.clone());
    index.add_vectors(&flat).unwrap();
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);

    #[cfg(feature = "serde")]
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir =
            tempfile::tempdir().expect("create temp dir for DualBranchHNSW snapshot benchmark");
        let path = temp_dir.path().join("dual_branch.json");
        index.save_to_file(&path).unwrap();
        let index_bytes = std::fs::metadata(&path).ok().map(|metadata| metadata.len());
        let load_start = Instant::now();
        let loaded = DualBranchHNSW::load_from_file(&path).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };
    #[cfg(not(feature = "serde"))]
    let snapshot_index: Option<(tempfile::TempDir, DualBranchHNSW, f64, Option<u64>)> = {
        if cfg.snapshot_load {
            eprintln!("dual_branch --snapshot-load requested but serde feature is not enabled");
        }
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        let params_json = format!(
            "{{\"m\":{},\"m_high_lid\":{},\"ef_construction\":{},\"ef_search\":{}}}",
            cfg.m,
            (cfg.m + cfg.m / 2).max(cfg.m + 1),
            cfg.ef_construction,
            ef
        );
        if cfg.json {
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "dual_branch",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }

        if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
            let loaded_result = evaluate(
                &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                test,
                neighbors,
                10,
            );
            if cfg.json {
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "dual_branch",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            } else {
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "hnsw")]
fn run_deg(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::deg::{DEGConfig, DEGIndex};

    const DEG_BENCH_LIMIT: usize = 10_000;
    let n = train.len().min(DEG_BENCH_LIMIT);
    let capped = train.len() > n;
    if capped {
        eprintln!(
            "DEG: capping at {DEG_BENCH_LIMIT} vectors (got {}); O(n^2) construction",
            train.len()
        );
    }
    let capped_neighbors = capped_neighbors_if_needed(cfg, train, test, n);
    let eval_neighbors = capped_neighbors.as_deref().unwrap_or(neighbors);
    let train = &train[..n];

    let params = DEGConfig::default();
    if !cfg.json {
        println!("--- DEG (base_edges={}, n={}) ---", params.base_edges, n);
    }

    let build_start = Instant::now();
    let mut index = DEGIndex::new(dim, params.clone());
    for vec in train {
        index.add(vec.clone()).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);

    #[cfg(feature = "serde")]
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for DEG snapshot benchmark");
        let path = temp_dir.path().join("deg.json");
        index.save_to_file(&path).unwrap();
        let index_bytes = std::fs::metadata(&path).ok().map(|metadata| metadata.len());
        let load_start = Instant::now();
        let loaded = DEGIndex::load_from_file(&path).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };
    #[cfg(not(feature = "serde"))]
    let snapshot_index: Option<(tempfile::TempDir, DEGIndex, f64, Option<u64>)> = {
        if cfg.snapshot_load {
            eprintln!("deg --snapshot-load requested but serde feature is not enabled");
        }
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let params_json = format!(
            "{{\"base_edges\":{},\"max_edges\":{},\"min_edges\":{},\"density_k\":{},\"alpha\":1.2,\"ef_search\":{},\"indexed_vectors\":{},\"capped\":{}}}",
            params.base_edges,
            params.max_edges,
            params.min_edges,
            params.density_k,
            ef,
            n,
            capped
        );
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            eval_neighbors,
            10,
        );
        if cfg.json {
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "deg",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "deg",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "nsw")]
fn run_nsw(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::nsw::NSWIndex;

    if !cfg.json {
        println!("--- NSW (M={}) ---", cfg.m);
    }

    let build_start = Instant::now();
    let mut index = NSWIndex::new(dim, cfg.m, cfg.m).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for NSW snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = NSWIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(&|q, k| index.search(q, k, ef).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!("{{\"m\":{},\"ef_search\":{}}}", cfg.m, ef);
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "nsw",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search(q, k, ef).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "nsw",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search(q, k, ef).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "emg")]
fn run_emg(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::emg::{EmgIndex, EmgParams};

    let params = EmgParams {
        max_degree: 32,
        candidate_size: 64,
        scale_t: 32,
        iterations: 2,
        alpha: 1.5,
        ef_search: 100,
    };

    if !cfg.json {
        println!("--- EMG (max_degree=32) ---");
    }

    let build_start = Instant::now();
    let mut index = EmgIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for EMG snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = EmgIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "emg",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "emg",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "nsg")]
fn run_nsg(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::nsg::{NsgIndex, NsgParams};

    let n = train.len().min(50_000);
    let capped = train.len() > n;
    if capped {
        eprintln!(
            "NSG: capping at 50,000 vectors (got {}); O(n^2) construction",
            train.len()
        );
    }
    let capped_neighbors = capped_neighbors_if_needed(cfg, train, test, n);
    let eval_neighbors = capped_neighbors.as_deref().unwrap_or(neighbors);
    let train = &train[..n];

    if !cfg.json {
        println!("--- NSG (max_degree=32, n={}) ---", n);
    }

    let params = NsgParams::default();

    let build_start = Instant::now();
    let mut index = NsgIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for NSG snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = NsgIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            eval_neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "nsg",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "nsg",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "pipnn")]
fn run_pipnn(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::pipnn::{PipnnIndex, PipnnParams};

    let params = PipnnParams {
        max_leaf_size: 2048,
        max_degree: 32,
        num_hash_bits: 12,
        final_prune: true,
        alpha: 1.2,
        ef_search: 100,
        ..Default::default()
    };

    if !cfg.json {
        println!("--- PiPNN (max_degree=32, max_leaf_size=2048) ---");
    }

    let build_start = Instant::now();
    let mut index = PipnnIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for PiPNN snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = PipnnIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"max_degree\":32,\"max_leaf_size\":2048,\"ef_search\":{}}}",
                ef
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "pipnn",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "pipnn",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "sng")]
fn run_sng(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::sng::SNGIndex;
    use vicinity::sng::SNGParams;

    let n = train.len().min(50_000);
    let capped = train.len() > n;
    if capped {
        eprintln!(
            "SNG: capping at 50,000 vectors (got {}); O(n^2) construction",
            train.len()
        );
    }
    let capped_neighbors = capped_neighbors_if_needed(cfg, train, test, n);
    let eval_neighbors = capped_neighbors.as_deref().unwrap_or(neighbors);
    let train = &train[..n];

    if !cfg.json {
        println!("--- SNG (n={}) ---", n);
    }

    let params = SNGParams::default();

    let build_start = Instant::now();
    let mut index = SNGIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add(i as u32, vec.clone()).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for SNG snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = SNGIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    let result = evaluate(
        &|q, k| index.search(q, k).unwrap(),
        test,
        eval_neighbors,
        10,
    );
    if cfg.json {
        let params_json = "{}";
        emit_result(
            &cfg.results_path,
            &json_line_with_storage(
                "sng",
                params_json,
                build_time_s,
                rss,
                &result,
                &in_memory_storage(index_bytes),
            ),
        );
        if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
            let loaded_result = evaluate(
                &|q, k| loaded.search(q, k).unwrap(),
                test,
                eval_neighbors,
                10,
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "sng",
                    params_json,
                    build_time_s,
                    rss,
                    &loaded_result,
                    &snapshot_storage(*load_time_s, *index_bytes),
                ),
            );
        }
    } else {
        print_row("--", &result);
        if let Some((_, loaded, _, _)) = &snapshot_index {
            let loaded_result = evaluate(
                &|q, k| loaded.search(q, k).unwrap(),
                test,
                eval_neighbors,
                10,
            );
            print_row("snapshot_loaded", &loaded_result);
        }
        println!();
    }
}

#[cfg(feature = "vamana")]
fn run_vamana(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::vamana::VamanaIndex;
    use vicinity::vamana::VamanaParams;

    if !cfg.json {
        println!("--- Vamana ---");
    }

    let params = VamanaParams {
        seed: Some(cfg.seed),
        ..VamanaParams::default()
    };

    let build_start = Instant::now();
    let mut index = VamanaIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add(i as u32, vec.clone()).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for Vamana snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = VamanaIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(&|q, k| index.search(q, k, ef).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!("{{\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "vamana",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search(q, k, ef).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "vamana",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search(q, k, ef).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "diskann")]
#[derive(Default)]
struct DiskAnnDiagnosticsTotals {
    queries: usize,
    visited_nodes: usize,
    graph_reads: usize,
    vector_reads: usize,
    page_reads: usize,
    graph_bytes: usize,
    vector_bytes: usize,
    page_bytes: usize,
    retained_candidates: usize,
}

#[cfg(feature = "diskann")]
impl DiskAnnDiagnosticsTotals {
    fn record(&mut self, diagnostics: vicinity::diskann::DiskANNSearchDiagnostics) {
        self.queries += 1;
        self.visited_nodes += diagnostics.visited_nodes;
        self.graph_reads += diagnostics.graph_reads;
        self.vector_reads += diagnostics.vector_reads;
        self.page_reads += diagnostics.page_reads;
        self.graph_bytes += diagnostics.graph_bytes;
        self.vector_bytes += diagnostics.vector_bytes;
        self.page_bytes += diagnostics.page_bytes;
        self.retained_candidates += diagnostics.retained_candidates;
    }

    fn average(&self) -> StorageDiagnostics {
        let queries = self.queries.max(1) as f64;
        StorageDiagnostics {
            avg_visited_nodes: self.visited_nodes as f64 / queries,
            avg_graph_reads: self.graph_reads as f64 / queries,
            avg_vector_reads: self.vector_reads as f64 / queries,
            avg_page_reads: self.page_reads as f64 / queries,
            avg_graph_bytes: self.graph_bytes as f64 / queries,
            avg_vector_bytes: self.vector_bytes as f64 / queries,
            avg_page_bytes: self.page_bytes as f64 / queries,
            avg_retained_candidates: self.retained_candidates as f64 / queries,
            ..StorageDiagnostics::default()
        }
    }
}

#[cfg(feature = "diskann")]
fn add_diskann_multi_recall(totals: &mut [f64; 3], truth: &[i32], results: &[(u32, f32)]) {
    use std::collections::HashSet;
    for (slot, depth) in [1, 10, 100].into_iter().enumerate() {
        let depth = depth.min(truth.len()).max(1);
        let expected: HashSet<u32> = truth.iter().take(depth).map(|&id| id as u32).collect();
        let found: HashSet<u32> = results.iter().take(depth).map(|row| row.0).collect();
        totals[slot] += expected.intersection(&found).count() as f64 / depth as f64;
    }
}

#[cfg(feature = "diskann")]
fn evaluate_diskann_searcher(
    searcher: &std::cell::RefCell<vicinity::diskann::DiskANNSearcher>,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    _k: usize,
    ef_search: usize,
) -> (BenchResult, StorageDiagnostics) {
    use std::time::Instant;

    let k = neighbors.first().map_or(1, |row| row.len().min(100));
    let warmup_count = support::warmup_queries().min(test.len());
    for query in test.iter().take(warmup_count) {
        let _ = searcher.borrow_mut().search(query, k, ef_search);
    }

    let mut recalls = [0.0; 3];
    let mut latencies_us: Vec<f64> = Vec::with_capacity(test.len());
    let mut diagnostics = DiskAnnDiagnosticsTotals::default();

    for (i, query) in test.iter().enumerate() {
        let q_start = Instant::now();
        let (results, query_diagnostics) = searcher
            .borrow_mut()
            .search_with_diagnostics(query, k, ef_search)
            .expect("DiskANN file-backed search failed");
        let q_elapsed = q_start.elapsed();
        latencies_us.push(q_elapsed.as_nanos() as f64 / 1000.0);
        diagnostics.record(query_diagnostics);

        add_diskann_multi_recall(&mut recalls, &neighbors[i], &results);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = latencies_us.len();
    let total_us: f64 = latencies_us.iter().sum();

    (
        BenchResult {
            recall_at_k: recalls[1] / n as f64,
            recall_at_1: recalls[0] / n as f64,
            recall_at_100: recalls[2] / n as f64,
            search_k: k,
            qps: n as f64 / (total_us / 1_000_000.0),
            latency_us: total_us / n as f64,
            p50_us: latencies_us[n / 2],
            p95_us: latencies_us[(n as f64 * 0.95) as usize],
            p99_us: latencies_us[(n as f64 * 0.99) as usize],
        },
        diagnostics.average(),
    )
}

#[cfg(all(feature = "diskann", feature = "benchmark"))]
fn evaluate_diskann_page_searcher(
    searcher: &std::cell::RefCell<vicinity::diskann::DiskANNPageSearcher>,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    _k: usize,
    ef_search: usize,
) -> (BenchResult, StorageDiagnostics) {
    use std::time::Instant;

    let k = neighbors.first().map_or(1, |row| row.len().min(100));
    let warmup_count = support::warmup_queries().min(test.len());
    for query in test.iter().take(warmup_count) {
        let _ = searcher.borrow_mut().search(query, k, ef_search);
    }

    let mut recalls = [0.0; 3];
    let mut latencies_us: Vec<f64> = Vec::with_capacity(test.len());
    let mut diagnostics = DiskAnnDiagnosticsTotals::default();

    for (i, query) in test.iter().enumerate() {
        let q_start = Instant::now();
        let (results, query_diagnostics) = searcher
            .borrow_mut()
            .search_with_diagnostics(query, k, ef_search)
            .expect("DiskANN page-backed search failed");
        let q_elapsed = q_start.elapsed();
        latencies_us.push(q_elapsed.as_nanos() as f64 / 1000.0);
        diagnostics.record(query_diagnostics);

        add_diskann_multi_recall(&mut recalls, &neighbors[i], &results);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = latencies_us.len();
    let total_us: f64 = latencies_us.iter().sum();

    (
        BenchResult {
            recall_at_k: recalls[1] / n as f64,
            recall_at_1: recalls[0] / n as f64,
            recall_at_100: recalls[2] / n as f64,
            search_k: k,
            qps: n as f64 / (total_us / 1_000_000.0),
            latency_us: total_us / n as f64,
            p50_us: latencies_us[n / 2],
            p95_us: latencies_us[(n as f64 * 0.95) as usize],
            p99_us: latencies_us[(n as f64 * 0.99) as usize],
        },
        diagnostics.average(),
    )
}

#[cfg(feature = "diskann")]
fn run_diskann(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use std::cell::RefCell;

    #[cfg(feature = "benchmark")]
    use vicinity::diskann::DiskANNPageSearcher;
    use vicinity::diskann::{DiskANNIndex, DiskANNParams, DiskANNSearcher};

    let params = DiskANNParams {
        m: cfg.m,
        ef_construction: cfg.ef_construction,
        alpha: 1.2,
        ef_search: 100,
        seed: Some(cfg.seed),
    };

    if !cfg.json {
        println!(
            "--- DiskANN (M={}, ef_construction={}, alpha=1.2) ---",
            cfg.m, cfg.ef_construction
        );
    }

    let build_start = Instant::now();
    let mut index = DiskANNIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let memory_index_bytes = Some(index.size_bytes() as u64);

    let temp_dir = tempfile::tempdir().expect("create temp dir for DiskANN file-backed benchmark");
    let index_dir = temp_dir.path().join("diskann");
    index.save(&index_dir).unwrap();
    let index_bytes = dir_size_bytes(&index_dir).ok();
    let file_load_start = Instant::now();
    let searcher = RefCell::new(DiskANNSearcher::load(&index_dir).unwrap());
    let file_load_time_s = file_load_start.elapsed().as_secs_f64();
    let mmap_load_start = Instant::now();
    let mmap_searcher = RefCell::new(DiskANNSearcher::load_mmap(&index_dir).unwrap());
    let mmap_load_time_s = mmap_load_start.elapsed().as_secs_f64();
    #[cfg(feature = "benchmark")]
    let page_searchers = {
        index.save_page_layout(&index_dir).unwrap();
        let page_index_bytes = std::fs::metadata(index_dir.join("nodes.page"))
            .ok()
            .map(|metadata| metadata.len());
        let page_file_load_start = Instant::now();
        let page_file_searcher = RefCell::new(DiskANNPageSearcher::load(&index_dir).unwrap());
        let page_file_load_time_s = page_file_load_start.elapsed().as_secs_f64();
        let page_mmap_load_start = Instant::now();
        let page_mmap_searcher = RefCell::new(DiskANNPageSearcher::load_mmap(&index_dir).unwrap());
        let page_mmap_load_time_s = page_mmap_load_start.elapsed().as_secs_f64();
        (
            page_file_searcher,
            page_file_load_time_s,
            page_mmap_searcher,
            page_mmap_load_time_s,
            page_index_bytes,
        )
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(&|q, k| index.search(q, k, ef).unwrap(), test, neighbors, 10);
        let (file_result, file_diagnostics) =
            evaluate_diskann_searcher(&searcher, test, neighbors, 10, ef);
        let (mmap_result, mmap_diagnostics) =
            evaluate_diskann_searcher(&mmap_searcher, test, neighbors, 10, ef);
        #[cfg(feature = "benchmark")]
        let page_results = {
            let (
                page_file_searcher,
                page_file_load_time_s,
                page_mmap_searcher,
                page_mmap_load_time_s,
                page_index_bytes,
            ) = &page_searchers;
            let (page_file_result, page_file_diagnostics) =
                evaluate_diskann_page_searcher(page_file_searcher, test, neighbors, 10, ef);
            let (page_mmap_result, page_mmap_diagnostics) =
                evaluate_diskann_page_searcher(page_mmap_searcher, test, neighbors, 10, ef);
            (
                page_file_result,
                page_file_diagnostics,
                *page_file_load_time_s,
                page_mmap_result,
                page_mmap_diagnostics,
                *page_mmap_load_time_s,
                *page_index_bytes,
            )
        };
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"alpha\":1.2,\"ef_search\":{},\"storage\":\"memory\"}}",
                cfg.m, cfg.ef_construction, ef
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "diskann",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &ResultStorage {
                        index_bytes: memory_index_bytes,
                        ..ResultStorage::default()
                    },
                ),
            );
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"alpha\":1.2,\"ef_search\":{},\"storage\":\"file\"}}",
                cfg.m, cfg.ef_construction, ef
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "diskann_file",
                    &params_json,
                    build_time_s,
                    rss,
                    &file_result,
                    &ResultStorage {
                        storage_mode: "file",
                        cache_state: "warm_after_open",
                        load_time_s: Some(file_load_time_s),
                        index_bytes,
                        index_bytes_kind: Some("storage_bytes"),
                        diagnostics: Some(file_diagnostics),
                    },
                ),
            );
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"alpha\":1.2,\"ef_search\":{},\"storage\":\"mmap\"}}",
                cfg.m, cfg.ef_construction, ef
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "diskann_mmap",
                    &params_json,
                    build_time_s,
                    rss,
                    &mmap_result,
                    &ResultStorage {
                        storage_mode: "mmap",
                        cache_state: "warm_after_open",
                        load_time_s: Some(mmap_load_time_s),
                        index_bytes,
                        index_bytes_kind: Some("storage_bytes"),
                        diagnostics: Some(mmap_diagnostics),
                    },
                ),
            );
            #[cfg(feature = "benchmark")]
            {
                let (
                    page_file_result,
                    page_file_diagnostics,
                    page_file_load_time_s,
                    page_mmap_result,
                    page_mmap_diagnostics,
                    page_mmap_load_time_s,
                    page_index_bytes,
                ) = &page_results;
                let params_json = format!(
                    "{{\"m\":{},\"ef_construction\":{},\"alpha\":1.2,\"ef_search\":{},\"storage\":\"page_file\"}}",
                    cfg.m, cfg.ef_construction, ef
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "diskann_page_file",
                        &params_json,
                        build_time_s,
                        rss,
                        page_file_result,
                        &ResultStorage {
                            storage_mode: "file",
                            cache_state: "warm_after_open",
                            load_time_s: Some(*page_file_load_time_s),
                            index_bytes: *page_index_bytes,
                            index_bytes_kind: Some("storage_bytes"),
                            diagnostics: Some(*page_file_diagnostics),
                        },
                    ),
                );
                let params_json = format!(
                    "{{\"m\":{},\"ef_construction\":{},\"alpha\":1.2,\"ef_search\":{},\"storage\":\"page_mmap\"}}",
                    cfg.m, cfg.ef_construction, ef
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "diskann_page_mmap",
                        &params_json,
                        build_time_s,
                        rss,
                        page_mmap_result,
                        &ResultStorage {
                            storage_mode: "mmap",
                            cache_state: "warm_after_open",
                            load_time_s: Some(*page_mmap_load_time_s),
                            index_bytes: *page_index_bytes,
                            index_bytes_kind: Some("storage_bytes"),
                            diagnostics: Some(*page_mmap_diagnostics),
                        },
                    ),
                );
            }
        } else {
            print_row(&format!("ef={} memory", ef), &result);
            print_row(&format!("ef={} file", ef), &file_result);
            print_row(&format!("ef={} mmap", ef), &mmap_result);
            #[cfg(feature = "benchmark")]
            {
                let (page_file_result, _, _, page_mmap_result, _, _, _) = &page_results;
                print_row(&format!("ef={} page_file", ef), page_file_result);
                print_row(&format!("ef={} page_mmap", ef), page_mmap_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "finger")]
fn run_finger(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::finger::{FingerIndex, FingerParams};

    let n = train.len().min(50_000);
    let capped = train.len() > n;
    if capped {
        eprintln!(
            "FINGER: capping at 50,000 vectors (got {}); construction is expensive",
            train.len()
        );
    }
    let capped_neighbors = capped_neighbors_if_needed(cfg, train, test, n);
    let eval_neighbors = capped_neighbors.as_deref().unwrap_or(neighbors);
    let train = &train[..n];

    let params = FingerParams {
        max_degree: 32,
        ef_construction: 200,
        ef_search: 100,
        alpha: 1.2,
    };

    if !cfg.json {
        println!("--- FINGER (max_degree=32, n={}) ---", n);
    }

    let build_start = Instant::now();
    let mut index = FingerIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for FINGER snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = FingerIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            eval_neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"max_degree\":32,\"ef_search\":{},\"indexed_vectors\":{},\"capped\":{}}}",
                ef,
                train.len(),
                capped
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "finger",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "finger",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "store")]
fn run_store(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use durability::FsDirectory;
    use vicinity::store::{SnapshotIndex, UpdatableIndex};

    if cfg.is_euclidean {
        eprintln!("store: skipping (cosine-only, dataset is euclidean)");
        return;
    }

    let flush_threshold = store_flush_threshold(train.len());
    let temp_dir = tempfile::tempdir().expect("create temp dir for store benchmark");
    let dir = FsDirectory::arc(temp_dir.path()).expect("open store benchmark directory");

    if !cfg.json {
        println!(
            "--- Store (segmented HNSW, m={}, flush={}) ---",
            cfg.m, flush_threshold
        );
    }

    let build_start = Instant::now();
    let mut index = UpdatableIndex::open(dir.clone(), flush_threshold, dim, cfg.m, cfg.m * 2)
        .expect("open store index");
    index
        .extend(
            train
                .iter()
                .enumerate()
                .map(|(id, vector)| (id as u32, vector.clone())),
        )
        .expect("ingest store benchmark vectors");
    index.checkpoint().expect("checkpoint store benchmark");
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = dir_size_bytes(temp_dir.path()).ok();
    let load_start = Instant::now();
    let snapshot =
        SnapshotIndex::open(dir, dim, cfg.m, cfg.m * 2).expect("open store snapshot index");
    if let Some(query) = test.first() {
        let warm_ef = cfg.ef_search_values.iter().copied().max().unwrap_or(10);
        snapshot
            .search(query, 10, warm_ef)
            .expect("warm store snapshot index");
    }
    let snapshot_load_time_s = load_start.elapsed().as_secs_f64();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(&|q, k| index.search(q, k, ef), test, neighbors, 10);
        if cfg.json {
            let params_json = store_params_json(cfg, train.len(), ef);
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "store",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &ResultStorage {
                        storage_mode: "segmented_store",
                        cache_state: "warm_after_checkpoint",
                        load_time_s: None,
                        index_bytes,
                        index_bytes_kind: Some("storage_bytes"),
                        diagnostics: None,
                    },
                ),
            );
            let snapshot_result = evaluate(
                &|q, k| {
                    snapshot
                        .search(q, k, ef)
                        .expect("search store snapshot index")
                },
                test,
                neighbors,
                10,
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "store_snapshot",
                    &params_json,
                    build_time_s,
                    rss,
                    &snapshot_result,
                    &ResultStorage {
                        storage_mode: "segmented_store",
                        cache_state: "warm_after_reopen",
                        load_time_s: Some(snapshot_load_time_s),
                        index_bytes,
                        index_bytes_kind: Some("storage_bytes"),
                        diagnostics: None,
                    },
                ),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
            let snapshot_result = evaluate(
                &|q, k| {
                    snapshot
                        .search(q, k, ef)
                        .expect("search store snapshot index")
                },
                test,
                neighbors,
                10,
            );
            print_row(&format!("ef={} snapshot_reopened", ef), &snapshot_result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "fresh_graph")]
fn run_fresh_graph(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};

    let n = train.len().min(50_000);
    let capped = train.len() > n;
    if capped {
        eprintln!(
            "FreshGraph: capping at 50,000 vectors (got {}); construction is expensive",
            train.len()
        );
    }
    let capped_neighbors = capped_neighbors_if_needed(cfg, train, test, n);
    let eval_neighbors = capped_neighbors.as_deref().unwrap_or(neighbors);
    let train = &train[..n];

    let params = FreshGraphParams {
        max_degree: 32,
        ef_construction: 200,
        ef_search: 100,
        alpha: 1.2,
    };

    if !cfg.json {
        println!("--- FreshGraph (max_degree=32, n={}) ---", n);
    }

    let build_start = Instant::now();
    let mut index = FreshGraphIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir =
            tempfile::tempdir().expect("create temp dir for FreshGraph snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = FreshGraphIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            eval_neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!("{{\"max_degree\":32,\"ef_search\":{}}}", ef);
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "fresh_graph",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "fresh_graph",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "fresh_graph")]
fn run_fresh_graph_churn(cfg: &Config, train: &[Vec<f32>], test: &[Vec<f32>], dim: usize) {
    use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};

    if cfg.is_euclidean {
        eprintln!("fresh_graph_churn: skipping (cosine-only, dataset is euclidean)");
        return;
    }

    let base_n = cfg
        .churn_base_size
        .min(train.len().saturating_sub(cfg.churn_cycles.max(1)));
    if base_n == 0 {
        eprintln!("fresh_graph_churn: dataset is too small for churn benchmark");
        return;
    }
    let cycles = cfg.churn_cycles.min(train.len().saturating_sub(base_n));
    let query_count = cfg.churn_queries.min(test.len());
    if cycles == 0 || query_count == 0 {
        eprintln!("fresh_graph_churn: need at least one update cycle and one query");
        return;
    }

    let params = FreshGraphParams {
        max_degree: 32,
        ef_construction: 200,
        ef_search: 100,
        alpha: 1.2,
    };

    if !cfg.json {
        println!(
            "--- FreshGraph churn (base={}, cycles={}, queries={}) ---",
            base_n, cycles, query_count
        );
    }

    let build_start = Instant::now();
    let mut index = FreshGraphIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().take(base_n).enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();

    let mut active: Vec<u32> = (0..base_n as u32).collect();
    let mut rng_state = cfg.seed ^ 0x9E37_79B9_7F4A_7C15_u64;
    let update_start = Instant::now();
    for offset in 0..cycles {
        let new_id = (base_n + offset) as u32;
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let victim_pos = ((rng_state >> 33) as usize) % active.len();
        let victim = active.swap_remove(victim_pos);
        index.delete(victim).unwrap();
        index.insert(new_id, &train[new_id as usize]).unwrap();
        active.push(new_id);
    }
    let update_time_s = update_start.elapsed().as_secs_f64();
    let update_qps = cycles as f64 / update_time_s;
    let tombstone_ratio = index.num_deleted() as f64 / index.len().max(1) as f64;
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);

    let test_subset = &test[..query_count];
    let live_neighbors: Vec<Vec<i32>> = test_subset
        .iter()
        .map(|query| {
            brute_force_search_ids(train, &active, query, 10, vicinity::DistanceMetric::Cosine)
        })
        .collect();

    if !cfg.json {
        println!(
            "Build: {:.2}s; Updates: {:.2}s ({:.0}/s); tombstones: {:.1}%\n",
            build_time_s,
            update_time_s,
            update_qps,
            tombstone_ratio * 100.0
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test_subset,
            &live_neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"max_degree\":32,\"ef_search\":{},\"base_size\":{},\"cycles\":{},\"queries\":{},\"update_time_s\":{:.4},\"update_qps\":{:.1},\"tombstone_ratio\":{:.4}}}",
                ef, base_n, cycles, query_count, update_time_s, update_qps, tombstone_ratio
            );
            let extra_json = format!(
                "\"active_count\":{},\"update_time_s\":{:.4},\"update_qps\":{:.1},\"tombstone_ratio\":{:.4}",
                active.len(),
                update_time_s,
                update_qps,
                tombstone_ratio
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage_and_extra_fields(
                    "fresh_graph_churn",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                    &extra_json,
                ),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "hnsw")]
fn run_inplace(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::{InPlaceConfig, InPlaceIndex};

    let n = train.len().min(50_000);
    let capped = train.len() > n;
    if capped {
        eprintln!(
            "InPlace: capping at 50,000 vectors (got {}); construction is expensive",
            train.len()
        );
    }
    let capped_neighbors = capped_neighbors_if_needed(cfg, train, test, n);
    let eval_neighbors = capped_neighbors.as_deref().unwrap_or(neighbors);
    let train = &train[..n];

    let params = InPlaceConfig {
        max_degree: 32,
        beam_width: cfg.ef_construction,
        alpha: 1.2,
        max_in_neighbors: 64,
        enable_back_edges: true,
    };

    if !cfg.json {
        println!(
            "--- InPlace (max_degree=32, build_beam_width={}, n={}) ---",
            cfg.ef_construction, n
        );
    }

    let build_start = Instant::now();
    let mut index = InPlaceIndex::new(dim, params);
    for vec in train {
        index.insert(vec.clone()).unwrap();
    }
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);

    #[cfg(feature = "serde")]
    let snapshot_index = if cfg.snapshot_load {
        let file = tempfile::NamedTempFile::new()
            .expect("create temp file for InPlace snapshot benchmark");
        index.save_to_file(file.path()).unwrap();
        let index_bytes = std::fs::metadata(file.path())
            .ok()
            .map(|metadata| metadata.len());
        let load_start = Instant::now();
        let loaded = InPlaceIndex::load_from_file(file.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((file, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &beam_width in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_beam(q, k, beam_width).unwrap(),
            test,
            eval_neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"max_degree\":32,\"build_beam_width\":{},\"beam_width\":{}}}",
                cfg.ef_construction, beam_width
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "inplace",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            #[cfg(feature = "serde")]
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_beam(q, k, beam_width).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "inplace",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("beam={}", beam_width), &result);
            #[cfg(feature = "serde")]
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_beam(q, k, beam_width).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                print_row(
                    &format!("beam={} snapshot_loaded", beam_width),
                    &loaded_result,
                );
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "hnsw")]
fn run_inplace_churn(cfg: &Config, train: &[Vec<f32>], test: &[Vec<f32>], dim: usize) {
    use vicinity::hnsw::{InPlaceConfig, MappedInPlaceIndex};
    use vicinity::streaming::IndexOps;

    let base_n = cfg
        .churn_base_size
        .min(train.len().saturating_sub(cfg.churn_cycles.max(1)));
    if base_n == 0 {
        eprintln!("inplace_churn: dataset is too small for churn benchmark");
        return;
    }
    let cycles = cfg.churn_cycles.min(train.len().saturating_sub(base_n));
    let query_count = cfg.churn_queries.min(test.len());
    if cycles == 0 || query_count == 0 {
        eprintln!("inplace_churn: need at least one update cycle and one query");
        return;
    }

    let params = InPlaceConfig {
        max_degree: 32,
        beam_width: cfg.ef_construction,
        alpha: 1.2,
        max_in_neighbors: 64,
        enable_back_edges: true,
    };

    if !cfg.json {
        println!(
            "--- InPlace churn (base={}, cycles={}, queries={}, build_beam_width={}) ---",
            base_n, cycles, query_count, cfg.ef_construction
        );
    }

    let build_start = Instant::now();
    let mut index = MappedInPlaceIndex::new(dim, params);
    for (i, vec) in train.iter().take(base_n).enumerate() {
        index.insert(i as u32, vec.clone()).unwrap();
    }
    let build_time_s = build_start.elapsed().as_secs_f64();

    let mut active: Vec<u32> = (0..base_n as u32).collect();
    let mut rng_state = cfg.seed ^ 0x9E37_79B9_7F4A_7C15_u64;
    let update_start = Instant::now();
    for offset in 0..cycles {
        let new_id = (base_n + offset) as u32;
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let victim_pos = ((rng_state >> 33) as usize) % active.len();
        let victim = active.swap_remove(victim_pos);
        index.delete(victim).unwrap();
        index
            .insert(new_id, train[new_id as usize].clone())
            .unwrap();
        active.push(new_id);
    }
    let update_time_s = update_start.elapsed().as_secs_f64();
    let update_qps = cycles as f64 / update_time_s;
    let stats = index.stats();
    let free_slot_ratio =
        stats.free_slots as f64 / (stats.active_nodes + stats.free_slots).max(1) as f64;
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);

    let test_subset = &test[..query_count];
    let live_neighbors: Vec<Vec<i32>> = test_subset
        .iter()
        .map(|query| brute_force_search_ids(train, &active, query, 10, dataset_metric(cfg)))
        .collect();

    if !cfg.json {
        println!(
            "Build: {:.2}s; Updates: {:.2}s ({:.0}/s); free slots: {:.1}%\n",
            build_time_s,
            update_time_s,
            update_qps,
            free_slot_ratio * 100.0
        );
        print_header();
    }

    for &beam_width in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_beam(q, k, beam_width).unwrap(),
            test_subset,
            &live_neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"max_degree\":32,\"build_beam_width\":{},\"beam_width\":{},\"base_size\":{},\"cycles\":{},\"queries\":{},\"update_time_s\":{:.4},\"update_qps\":{:.1},\"free_slot_ratio\":{:.4}}}",
                cfg.ef_construction,
                beam_width,
                base_n,
                cycles,
                query_count,
                update_time_s,
                update_qps,
                free_slot_ratio
            );
            let extra_json = format!(
                "\"active_count\":{},\"update_time_s\":{:.4},\"update_qps\":{:.1},\"free_slot_ratio\":{:.4}",
                active.len(),
                update_time_s,
                update_qps,
                free_slot_ratio
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage_and_extra_fields(
                    "inplace_churn",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                    &extra_json,
                ),
            );
        } else {
            print_row(&format!("beam={}", beam_width), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "hnsw")]
fn run_lsm_churn(cfg: &Config, train: &[Vec<f32>], test: &[Vec<f32>], dim: usize) {
    use vicinity::streaming::{LsmConfig, LsmIndex};

    let base_n = cfg
        .churn_base_size
        .min(train.len().saturating_sub(cfg.churn_cycles.max(1)));
    if base_n == 0 {
        eprintln!("lsm_churn: dataset is too small for churn benchmark");
        return;
    }
    let cycles = cfg.churn_cycles.min(train.len().saturating_sub(base_n));
    let query_count = cfg.churn_queries.min(test.len());
    if cycles == 0 || query_count == 0 {
        eprintln!("lsm_churn: need at least one update cycle and one query");
        return;
    }

    let buffer_capacity = (base_n / 10).clamp(20, 10_000);
    let lsm_cfg = LsmConfig {
        dimension: dim,
        buffer_capacity,
        size_ratio: 4,
        max_levels: 5,
        hnsw_m: cfg.m,
        hnsw_ef_construction: cfg.ef_construction,
        ef_search: cfg.ef_search_values.iter().copied().max().unwrap_or(100),
        distance_metric: dataset_metric(cfg),
    };

    if !cfg.json {
        println!(
            "--- LSM churn (base={}, cycles={}, queries={}, buffer={}, ratio=4, metric={:?}) ---",
            base_n, cycles, query_count, buffer_capacity, lsm_cfg.distance_metric
        );
    }

    let build_start = Instant::now();
    let mut index = LsmIndex::new(lsm_cfg);
    for (i, vec) in train.iter().take(base_n).enumerate() {
        index.insert_slice(i as u32, vec).unwrap();
    }
    let build_time_s = build_start.elapsed().as_secs_f64();

    let mut active: Vec<u32> = (0..base_n as u32).collect();
    let mut rng_state = cfg.seed ^ 0x517c_c1b7_2722_0a95_u64;
    let update_start = Instant::now();
    for offset in 0..cycles {
        let new_id = (base_n + offset) as u32;
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let victim_pos = ((rng_state >> 33) as usize) % active.len();
        let victim = active.swap_remove(victim_pos);
        index.delete(victim);
        index.insert_slice(new_id, &train[new_id as usize]).unwrap();
        active.push(new_id);
    }
    let update_time_s = update_start.elapsed().as_secs_f64();
    let update_qps = cycles as f64 / update_time_s;
    let stats = index.stats();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    #[cfg(feature = "serde")]
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for LSM snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = LsmIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };
    #[cfg(not(feature = "serde"))]
    if cfg.snapshot_load {
        eprintln!("lsm_churn --snapshot-load requested but serde feature is not enabled");
    }

    let test_subset = &test[..query_count];
    let live_neighbors: Vec<Vec<i32>> = test_subset
        .iter()
        .map(|query| brute_force_search_ids(train, &active, query, 10, dataset_metric(cfg)))
        .collect();

    if !cfg.json {
        println!(
            "Build: {:.2}s; Updates: {:.2}s ({:.0}/s); compactions: {}; levels: {:?}; tombstones: {}\n",
            build_time_s,
            update_time_s,
            update_qps,
            stats.total_compactions,
            stats.level_sizes,
            stats.tombstone_count
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test_subset,
            &live_neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"base_size\":{},\"cycles\":{},\"queries\":{},\"buffer_capacity\":{},\"size_ratio\":4,\"max_levels\":5,\"update_time_s\":{:.4},\"update_qps\":{:.1},\"compactions\":{},\"levels\":{},\"level_sizes\":{:?},\"tombstones\":{}}}",
                cfg.m,
                cfg.ef_construction,
                ef,
                base_n,
                cycles,
                query_count,
                buffer_capacity,
                update_time_s,
                update_qps,
                stats.total_compactions,
                stats.num_levels,
                stats.level_sizes,
                stats.tombstone_count
            );
            let extra_json = format!(
                "\"active_count\":{},\"update_time_s\":{:.4},\"update_qps\":{:.1},\"compactions\":{},\"levels\":{},\"tombstones\":{}",
                active.len(),
                update_time_s,
                update_qps,
                stats.total_compactions,
                stats.num_levels,
                stats.tombstone_count
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage_and_extra_fields(
                    "lsm_churn",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                    &extra_json,
                ),
            );
            #[cfg(feature = "serde")]
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test_subset,
                    &live_neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage_and_extra_fields(
                        "lsm_churn",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                        &extra_json,
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            #[cfg(feature = "serde")]
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test_subset,
                    &live_neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "filtered_graph")]
fn run_filtered_graph(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use std::collections::HashMap;
    use vicinity::filtered_graph::{FilteredGraphIndex, FilteredGraphParams};

    let n = train.len().min(50_000);
    let capped = train.len() > n;
    if capped {
        eprintln!(
            "FilteredGraph: capping at 50,000 vectors (got {}); construction is expensive",
            train.len()
        );
    }
    let capped_neighbors = capped_neighbors_if_needed(cfg, train, test, n);
    let eval_neighbors = capped_neighbors.as_deref().unwrap_or(neighbors);
    let train = &train[..n];

    let params = FilteredGraphParams {
        max_degree: 32,
        ef_construction: 200,
        ef_search: 100,
        alpha: 1.2,
    };

    if !cfg.json {
        println!("--- FilteredGraph (max_degree=32, unfiltered, n={}) ---", n);
    }

    let build_start = Instant::now();
    let mut index = FilteredGraphIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec, HashMap::new()).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir =
            tempfile::tempdir().expect("create temp dir for FilteredGraph snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = FilteredGraphIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| index.search_with_ef(q, k, ef).unwrap(),
            test,
            eval_neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"max_degree\":32,\"ef_search\":{},\"filter_mode\":\"none\"}}",
                ef
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "filtered_graph",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "filtered_graph",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            }
        } else {
            print_row(&format!("ef={}", ef), &result);
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_with_ef(q, k, ef).unwrap(),
                    test,
                    eval_neighbors,
                    10,
                );
                print_row(&format!("ef={} snapshot_loaded", ef), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "sparse_mips")]
fn run_sparse_mips(
    _cfg: &Config,
    _train: &[Vec<f32>],
    _test: &[Vec<f32>],
    _neighbors: &[Vec<i32>],
) {
    eprintln!(
        "sparse_mips: skipped -- index requires sparse vectors (SparseVector); \
         the dense benchmark dataset (f32 slices) is incompatible. \
         Use a sparse dataset such as SPLADE or BM25 embeddings instead."
    );
}

#[cfg(feature = "hnsw")]
fn run_adsampling(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::adsampling::{ADSamplingParams, ADSamplingState};
    use vicinity::hnsw::{HNSWIndex, HNSWParams};

    let m = cfg.m;
    let ef_construction = cfg.ef_construction;

    if !cfg.json {
        println!(
            "--- ADSampling (HNSW m={}, ef_c={}, eps0=2.1) ---",
            m, ef_construction
        );
    }

    let build_start = Instant::now();
    let metric = if cfg.is_euclidean {
        vicinity::DistanceMetric::L2
    } else {
        vicinity::DistanceMetric::Cosine
    };
    let params = HNSWParams {
        m,
        m_max: m,
        ef_construction,
        metric,
        auto_normalize: !cfg.is_euclidean,
        seed: Some(cfg.seed),
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).unwrap();
    let ids: Vec<u32> = (0..train.len() as u32).collect();
    let flat: Vec<f32> = train.iter().flatten().copied().collect();
    index.add_batch(&ids, &flat).unwrap();
    let _ = index.build();

    // Build ADSampling state from the HNSW's reordered vectors.
    // Must use from_hnsw() because build() reorders vectors for cache locality.
    let ads_params = ADSamplingParams::default();
    let state = ADSamplingState::from_hnsw(&index, ads_params);
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some((index.memory_usage().total() + state.memory_usage().total()) as u64);

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| state.search_hnsw(&index, q, k, ef).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"epsilon0\":2.1}}",
                m, ef_construction, ef
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "adsampling",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }
    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "curator")]
fn run_curator(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::curator::{CuratorIndex, CuratorParams};

    let params = CuratorParams {
        branching_factor: 16,
        max_leaf_size: 128,
        ef_search: 256,
        beam_width: 4,
    };

    if !cfg.json {
        println!("--- Curator (branching=16, leaf=128) ---");
    }

    let build_start = Instant::now();
    let mut index = CuratorIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add(i as u32, vec.clone(), vec![]).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for Curator snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = CuratorIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
    if cfg.json {
        let params_json = r#"{"branching_factor":16,"max_leaf_size":128,"filter_mode":"none"}"#;
        emit_result(
            &cfg.results_path,
            &json_line_with_storage(
                "curator",
                params_json,
                build_time_s,
                rss,
                &result,
                &in_memory_storage(index_bytes),
            ),
        );
        if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
            let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "curator",
                    params_json,
                    build_time_s,
                    rss,
                    &loaded_result,
                    &snapshot_storage(*load_time_s, *index_bytes),
                ),
            );
        }
    } else {
        print_row("--", &result);
        if let Some((_, loaded, _, _)) = &snapshot_index {
            let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
            print_row("snapshot_loaded", &loaded_result);
        }
        println!();
    }
}

#[cfg(all(feature = "range_filtered", feature = "hnsw"))]
fn run_range_filtered(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::range_filtered::{RangeFilteredIndex, RangeFilteredParams};

    let params = RangeFilteredParams {
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        ef_search: 100,
    };

    if !cfg.json {
        println!("--- RangeFiltered (m=16, ef_search=100) ---");
    }

    let build_start = Instant::now();
    let mut index = RangeFilteredIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add(i as u32, vec.clone(), 0.0).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir =
            tempfile::tempdir().expect("create temp dir for RangeFiltered snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = RangeFilteredIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        Some((temp_dir, loaded, load_time_s, index_bytes))
    } else {
        None
    };

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
    if cfg.json {
        let params_json = r#"{"hnsw_m":16,"ef_search":100,"filter_mode":"none"}"#;
        emit_result(
            &cfg.results_path,
            &json_line_with_storage(
                "range_filtered",
                params_json,
                build_time_s,
                rss,
                &result,
                &in_memory_storage(index_bytes),
            ),
        );
        if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
            let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "range_filtered",
                    params_json,
                    build_time_s,
                    rss,
                    &loaded_result,
                    &snapshot_storage(*load_time_s, *index_bytes),
                ),
            );
        }
    } else {
        print_row("--", &result);
        if let Some((_, loaded, _, _)) = &snapshot_index {
            let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
            print_row("snapshot_loaded", &loaded_result);
        }
        println!();
    }
}

#[cfg(feature = "hnsw")]
fn run_hnsw_prt(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::{HNSWIndex, HNSWParams};
    use vicinity::prt::ProbabilisticRoutingTest;
    use vicinity::DistanceMetric;

    let m = cfg.m;
    let ef_construction = cfg.ef_construction;

    if !cfg.json {
        println!("--- HNSW+PRT (M={}, ef_c={}) ---", m, ef_construction);
    }

    let metric = if cfg.is_euclidean {
        DistanceMetric::L2
    } else {
        DistanceMetric::Cosine
    };

    let build_start = Instant::now();
    let params = HNSWParams {
        m,
        m_max: m * 2,
        ef_construction,
        metric,
        auto_normalize: !cfg.is_euclidean,
        seed: Some(cfg.seed),
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).unwrap();
    let ids: Vec<u32> = (0..train.len() as u32).collect();
    let flat: Vec<f32> = train.iter().flatten().copied().collect();
    index.add_batch(&ids, &flat).unwrap();
    index.build().unwrap();

    // Build PRT state: project all database vectors.
    let num_proj = (dim / 4).clamp(8, 64); // heuristic: d/4, clamped [8, 64]
    let mut prt = ProbabilisticRoutingTest::new(dim, num_proj, Some(cfg.seed));
    prt.project_database(index.raw_vectors());

    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let projection_bytes = (prt.num_projections() * dim
        + prt.num_vectors() * prt.num_projections())
        * std::mem::size_of::<f32>();
    let index_bytes = Some((index.memory_usage().total() + projection_bytes) as u64);

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec), PRT projections: {}\n",
            build_time_s,
            train.len() as f64 / build_time_s,
            num_proj
        );
        print_header();
    }

    for &ef in &cfg.ef_search_values {
        let result = evaluate(
            &|q, k| {
                index
                    .search_prt(q, k, ef, &prt, 1.5, 0.95)
                    .map(|(results, _)| results)
                    .unwrap_or_default()
            },
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"num_projections\":{}}}",
                m, ef_construction, ef, num_proj
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "hnsw_prt",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &in_memory_storage(index_bytes),
                ),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

fn run_brute(cfg: &Config, train: &[Vec<f32>], test: &[Vec<f32>], neighbors: &[Vec<i32>]) {
    if !cfg.json {
        println!("--- Brute Force (linear scan) ---");
    }

    let build_time_s = 0.0; // no build step
    let rss = current_rss_kb();

    if !cfg.json {
        println!("Build: N/A (no index)\n");
        print_header();
    }

    let metric = if cfg.is_euclidean {
        vicinity::DistanceMetric::L2
    } else {
        vicinity::DistanceMetric::Cosine
    };
    let result = evaluate(
        &|q, k| brute_force_search(train, q, k, metric),
        test,
        neighbors,
        10,
    );

    if cfg.json {
        let params_json = "{}";
        let storage = ResultStorage {
            index_bytes: Some(0),
            index_bytes_kind: Some("none"),
            ..ResultStorage::default()
        };
        emit_result(
            &cfg.results_path,
            &json_line_with_storage("brute", params_json, build_time_s, rss, &result, &storage),
        );
    } else {
        print_row("--", &result);
        println!();
    }
}

#[cfg(feature = "kdtree")]
fn run_kdtree(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::classic::trees::kdtree::{KDTreeIndex, KDTreeParams};

    if dim > 50 {
        eprintln!("kdtree: skipping (implementation rejects dimensions > 50)");
        return;
    }
    if cfg.is_euclidean {
        eprintln!("kdtree: skipping (cosine-only search, dataset is euclidean)");
        return;
    }

    if !cfg.json {
        println!("--- KD-Tree ---");
        print_header();
    }

    for &max_leaf_size in &cfg.tree_leaf_sizes {
        for &max_depth in &cfg.tree_depths {
            let params = KDTreeParams {
                max_leaf_size,
                max_depth,
            };
            let build_start = Instant::now();
            let mut index = KDTreeIndex::new(dim, params.clone()).unwrap();
            for (i, vec) in train.iter().enumerate() {
                index.add(i as u32, vec.clone()).unwrap();
            }
            index.build().unwrap();
            let build_time_s = build_start.elapsed().as_secs_f64();
            let rss = current_rss_kb();
            let index_bytes = Some(index.memory_usage().total() as u64);

            let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
            let params_json = tree_params_json(params.max_leaf_size, params.max_depth);
            let label = format!("leaf={} depth={}", params.max_leaf_size, params.max_depth);
            if cfg.json {
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "kdtree",
                        &params_json,
                        build_time_s,
                        rss,
                        &result,
                        &in_memory_storage(index_bytes),
                    ),
                );
            } else {
                print_row(&label, &result);
            }

            if cfg.snapshot_load {
                let temp_dir =
                    tempfile::tempdir().expect("create temp dir for KD-tree snapshot benchmark");
                index.save_to_dir(temp_dir.path()).unwrap();
                let index_bytes = dir_size_bytes(temp_dir.path()).ok();
                let load_start = Instant::now();
                let loaded = KDTreeIndex::load_from_dir(temp_dir.path()).unwrap();
                let load_time_s = load_start.elapsed().as_secs_f64();
                let loaded_result =
                    evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
                if cfg.json {
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "kdtree",
                            &params_json,
                            build_time_s,
                            rss,
                            &loaded_result,
                            &snapshot_storage(load_time_s, index_bytes),
                        ),
                    );
                } else {
                    print_row(&format!("{label} snapshot_loaded"), &loaded_result);
                }
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "balltree")]
fn run_balltree(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::classic::trees::balltree::{BallTreeIndex, BallTreeParams};

    if cfg.is_euclidean {
        eprintln!("balltree: skipping (search uses cosine leaf distances, dataset is euclidean)");
        return;
    }

    if !cfg.json {
        println!("--- Ball Tree ---");
        print_header();
    }

    for &max_leaf_size in &cfg.tree_leaf_sizes {
        for &max_depth in &cfg.tree_depths {
            let params = BallTreeParams {
                max_leaf_size,
                max_depth,
            };
            let build_start = Instant::now();
            let mut index = BallTreeIndex::new(dim, params.clone()).unwrap();
            for (i, vec) in train.iter().enumerate() {
                index.add(i as u32, vec.clone()).unwrap();
            }
            index.build().unwrap();
            let build_time_s = build_start.elapsed().as_secs_f64();
            let rss = current_rss_kb();
            let index_bytes = Some(index.memory_usage().total() as u64);

            let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
            let params_json = tree_params_json(params.max_leaf_size, params.max_depth);
            let label = format!("leaf={} depth={}", params.max_leaf_size, params.max_depth);
            if cfg.json {
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "balltree",
                        &params_json,
                        build_time_s,
                        rss,
                        &result,
                        &in_memory_storage(index_bytes),
                    ),
                );
            } else {
                print_row(&label, &result);
            }

            if cfg.snapshot_load {
                let temp_dir =
                    tempfile::tempdir().expect("create temp dir for Ball tree snapshot benchmark");
                index.save_to_dir(temp_dir.path()).unwrap();
                let index_bytes = dir_size_bytes(temp_dir.path()).ok();
                let load_start = Instant::now();
                let loaded = BallTreeIndex::load_from_dir(temp_dir.path()).unwrap();
                let load_time_s = load_start.elapsed().as_secs_f64();
                let loaded_result =
                    evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
                if cfg.json {
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "balltree",
                            &params_json,
                            build_time_s,
                            rss,
                            &loaded_result,
                            &snapshot_storage(load_time_s, index_bytes),
                        ),
                    );
                } else {
                    print_row(&format!("{label} snapshot_loaded"), &loaded_result);
                }
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "rptree")]
fn run_rptree(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::classic::trees::random_projection::{RPTreeIndex, RPTreeParams};

    if cfg.is_euclidean {
        eprintln!("rptree: skipping (cosine-only search, dataset is euclidean)");
        return;
    }

    if !cfg.json {
        println!("--- Random Projection Tree ---");
    }

    if !cfg.json {
        print_header();
    }

    for &max_leaf_size in &cfg.tree_leaf_sizes {
        for &max_depth in &cfg.tree_depths {
            let params = RPTreeParams {
                max_leaf_size,
                max_depth,
            };
            let build_start = Instant::now();
            let mut index = RPTreeIndex::new(dim, params.clone()).unwrap();
            for (i, vec) in train.iter().enumerate() {
                index.add(i as u32, vec.clone()).unwrap();
            }
            index.build().unwrap();
            let build_time_s = build_start.elapsed().as_secs_f64();
            let rss = current_rss_kb();
            let index_bytes = Some(index.memory_usage().total() as u64);

            let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
            let params_json = tree_params_json(params.max_leaf_size, params.max_depth);
            let label = format!("leaf={} depth={}", params.max_leaf_size, params.max_depth);
            if cfg.json {
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "rptree",
                        &params_json,
                        build_time_s,
                        rss,
                        &result,
                        &in_memory_storage(index_bytes),
                    ),
                );
            } else {
                print_row(&label, &result);
            }

            if cfg.snapshot_load {
                let temp_dir =
                    tempfile::tempdir().expect("create temp dir for RP-tree snapshot benchmark");
                index.save_to_dir(temp_dir.path()).unwrap();
                let index_bytes = dir_size_bytes(temp_dir.path()).ok();
                let load_start = Instant::now();
                let loaded = RPTreeIndex::load_from_dir(temp_dir.path()).unwrap();
                let load_time_s = load_start.elapsed().as_secs_f64();
                let loaded_result =
                    evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
                if cfg.json {
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "rptree",
                            &params_json,
                            build_time_s,
                            rss,
                            &loaded_result,
                            &snapshot_storage(load_time_s, index_bytes),
                        ),
                    );
                } else {
                    print_row(&format!("{label} snapshot_loaded"), &loaded_result);
                }
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "rptree")]
fn run_rp_forest(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::classic::trees::rp_forest::{RPTreeParams, RpForestIndex, RpForestParams};

    if cfg.is_euclidean {
        eprintln!("rp_forest: skipping (cosine-only search, dataset is euclidean)");
        return;
    }

    if !cfg.json {
        println!("--- Random Projection Forest ---");
    }

    if !cfg.json {
        print_header();
    }

    for &num_trees in &cfg.rp_num_trees {
        for &max_leaf_size in &cfg.tree_leaf_sizes {
            let params = RpForestParams {
                num_trees,
                tree_params: RPTreeParams { max_leaf_size },
            };
            let build_start = Instant::now();
            let mut index = RpForestIndex::new(dim, params.clone()).unwrap();
            for (i, vec) in train.iter().enumerate() {
                index.add(i as u32, vec.clone()).unwrap();
            }
            index.build().unwrap();
            let build_time_s = build_start.elapsed().as_secs_f64();
            let rss = current_rss_kb();
            let index_bytes = Some(index.memory_usage().total() as u64);

            let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
            let params_json =
                rp_forest_params_json(params.num_trees, params.tree_params.max_leaf_size);
            let label = format!(
                "trees={} leaf={}",
                params.num_trees, params.tree_params.max_leaf_size
            );
            if cfg.json {
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "rp_forest",
                        &params_json,
                        build_time_s,
                        rss,
                        &result,
                        &in_memory_storage(index_bytes),
                    ),
                );
            } else {
                print_row(&label, &result);
            }

            if cfg.snapshot_load {
                let temp_dir =
                    tempfile::tempdir().expect("create temp dir for RP-forest snapshot benchmark");
                index.save_to_dir(temp_dir.path()).unwrap();
                let index_bytes = dir_size_bytes(temp_dir.path()).ok();
                let load_start = Instant::now();
                let loaded = RpForestIndex::load_from_dir(temp_dir.path()).unwrap();
                let load_time_s = load_start.elapsed().as_secs_f64();
                let loaded_result =
                    evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
                if cfg.json {
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "rp_forest",
                            &params_json,
                            build_time_s,
                            rss,
                            &loaded_result,
                            &snapshot_storage(load_time_s, index_bytes),
                        ),
                    );
                } else {
                    print_row(&format!("{label} snapshot_loaded"), &loaded_result);
                }
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "kmeans_tree")]
fn run_kmeans_tree(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::classic::trees::kmeans_tree::{KMeansTreeIndex, KMeansTreeParams};

    if !cfg.json {
        println!("--- K-Means Tree ---");
    }

    if !cfg.json {
        print_header();
    }

    for &num_clusters in &cfg.kmeans_clusters {
        for &max_leaf_size in &cfg.kmeans_leaf_sizes {
            for &max_depth in &cfg.kmeans_depths {
                for &max_iterations in &cfg.kmeans_iters {
                    let params = KMeansTreeParams {
                        num_clusters,
                        max_leaf_size,
                        max_depth,
                        max_iterations,
                    };
                    let build_start = Instant::now();
                    let mut index = KMeansTreeIndex::new(dim, params.clone()).unwrap();
                    for (i, vec) in train.iter().enumerate() {
                        index.add(i as u32, vec.clone()).unwrap();
                    }
                    index.build().unwrap();
                    let build_time_s = build_start.elapsed().as_secs_f64();
                    let rss = current_rss_kb();
                    let index_bytes = Some(index.memory_usage().total() as u64);

                    for &search_branches in &cfg.kmeans_search_branches {
                        let result = evaluate(
                            &|q, k| {
                                index
                                    .search_with_branch_budget(q, k, search_branches)
                                    .unwrap()
                            },
                            test,
                            neighbors,
                            10,
                        );
                        let params_json = kmeans_tree_params_json(
                            params.num_clusters,
                            params.max_leaf_size,
                            params.max_depth,
                            params.max_iterations,
                            search_branches,
                        );
                        let label = format!(
                            "clusters={} leaf={} depth={} iters={} branches={}",
                            params.num_clusters,
                            params.max_leaf_size,
                            params.max_depth,
                            params.max_iterations,
                            search_branches
                        );
                        if cfg.json {
                            emit_result(
                                &cfg.results_path,
                                &json_line_with_storage(
                                    "kmeans_tree",
                                    &params_json,
                                    build_time_s,
                                    rss,
                                    &result,
                                    &in_memory_storage(index_bytes),
                                ),
                            );
                        } else {
                            print_row(&label, &result);
                        }

                        if cfg.snapshot_load {
                            let temp_dir = tempfile::tempdir()
                                .expect("create temp dir for K-means tree snapshot benchmark");
                            index.save_to_dir(temp_dir.path()).unwrap();
                            let index_bytes = dir_size_bytes(temp_dir.path()).ok();
                            let load_start = Instant::now();
                            let loaded = KMeansTreeIndex::load_from_dir(temp_dir.path()).unwrap();
                            let load_time_s = load_start.elapsed().as_secs_f64();
                            let loaded_result = evaluate(
                                &|q, k| {
                                    loaded
                                        .search_with_branch_budget(q, k, search_branches)
                                        .unwrap()
                                },
                                test,
                                neighbors,
                                10,
                            );
                            if cfg.json {
                                emit_result(
                                    &cfg.results_path,
                                    &json_line_with_storage(
                                        "kmeans_tree",
                                        &params_json,
                                        build_time_s,
                                        rss,
                                        &loaded_result,
                                        &snapshot_storage(load_time_s, index_bytes),
                                    ),
                                );
                            } else {
                                print_row(&format!("{label} snapshot_loaded"), &loaded_result);
                            }
                        }
                    }

                    for &leaf_budget in &cfg.kmeans_leaf_budgets {
                        let result = evaluate(
                            &|q, k| index.search_with_leaf_budget(q, k, leaf_budget).unwrap(),
                            test,
                            neighbors,
                            10,
                        );
                        let params_json = kmeans_tree_leaf_budget_params_json(
                            params.num_clusters,
                            params.max_leaf_size,
                            params.max_depth,
                            params.max_iterations,
                            leaf_budget,
                        );
                        let label = format!(
                            "clusters={} leaf={} depth={} iters={} leaf_budget={}",
                            params.num_clusters,
                            params.max_leaf_size,
                            params.max_depth,
                            params.max_iterations,
                            leaf_budget
                        );
                        if cfg.json {
                            emit_result(
                                &cfg.results_path,
                                &json_line_with_storage(
                                    "kmeans_tree",
                                    &params_json,
                                    build_time_s,
                                    rss,
                                    &result,
                                    &in_memory_storage(index_bytes),
                                ),
                            );
                        } else {
                            print_row(&label, &result);
                        }

                        if cfg.snapshot_load {
                            let temp_dir = tempfile::tempdir()
                                .expect("create temp dir for K-means tree snapshot benchmark");
                            index.save_to_dir(temp_dir.path()).unwrap();
                            let index_bytes = dir_size_bytes(temp_dir.path()).ok();
                            let load_start = Instant::now();
                            let loaded = KMeansTreeIndex::load_from_dir(temp_dir.path()).unwrap();
                            let load_time_s = load_start.elapsed().as_secs_f64();
                            let loaded_result = evaluate(
                                &|q, k| loaded.search_with_leaf_budget(q, k, leaf_budget).unwrap(),
                                test,
                                neighbors,
                                10,
                            );
                            if cfg.json {
                                emit_result(
                                    &cfg.results_path,
                                    &json_line_with_storage(
                                        "kmeans_tree",
                                        &params_json,
                                        build_time_s,
                                        rss,
                                        &loaded_result,
                                        &snapshot_storage(load_time_s, index_bytes),
                                    ),
                                );
                            } else {
                                print_row(&format!("{label} snapshot_loaded"), &loaded_result);
                            }
                        }
                    }
                }
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if help_requested(std::env::args().skip(1)) {
        print!("{HELP}");
        return Ok(());
    }

    let cfg = parse_args();
    set_warmup_queries(cfg.warmup_queries);
    set_run_identity(cfg.seed, cfg.repeat);

    if !Path::new(&cfg.data_dir).join("train.bin").exists() {
        eprintln!("Dataset not found at: {}/train.bin", cfg.data_dir);
        eprintln!("Run: uv run scripts/download_ann_benchmarks.py <dataset>");
        std::process::exit(1);
    }

    if !cfg.json {
        println!("ANN Benchmark");
        println!("=============");
        println!("Data: {}\n", cfg.data_dir);
    }

    let (mut train, dim) = common::load_vectors(&format!("{}/train.bin", cfg.data_dir))?;
    let (mut test, _) = common::load_vectors(&format!("{}/test.bin", cfg.data_dir))?;
    let (mut neighbors, k_gt) = common::load_neighbors(&format!("{}/neighbors.bin", cfg.data_dir))?;

    if let Some(max_queries) = cfg.max_queries {
        if max_queries == 0 {
            return Err("--max-queries must be greater than zero".into());
        }
        let capped_len = max_queries.min(test.len()).min(neighbors.len());
        test.truncate(capped_len);
        neighbors.truncate(capped_len);
    }

    if let Some(max_train) = cfg.max_train {
        if max_train == 0 {
            return Err("--max-train must be greater than zero".into());
        }
        if max_train < train.len() {
            train.truncate(max_train);
            let active_ids: Vec<u32> = (0..train.len() as u32).collect();
            neighbors = brute_force_neighbors_for_ids(
                &train,
                &active_ids,
                &test,
                k_gt,
                dataset_metric(&cfg),
            );
        }
    }

    let meta = || {
        format!(
            "{{\"_meta\":{{\"dataset\":\"{}\",\"metric\":\"{}\",\"result_schema\":3,\"index_bytes_required\":true,\"seed\":{},\"repeat\":{},\"run_id\":\"seed-{}-repeat-{}\",\"seed_fingerprint\":\"{}\",\"cpu\":\"{}\",\"architecture\":\"{}\",\"threads\":{},\"train_full\":{},\"query_full\":{},\"rustc\":\"{}\",\"rust_msrv\":\"{}\",\"vicinity\":\"{}\",\"features\":{},\"train_limit\":{},\"indexed_vectors\":{},\"query_limit\":{},\"queries\":{},\"warmup_queries\":{}}}}}",
            cfg.data_dir,
            if cfg.is_euclidean { "l2" } else { "cosine" },
            cfg.seed,
            cfg.repeat,
            cfg.seed,
            cfg.repeat,
            seed_fingerprint(cfg.seed),
            cpu_model(),
            std::env::consts::ARCH,
            std::thread::available_parallelism().map_or(1, usize::from),
            cfg.max_train.is_none(),
            cfg.max_queries.is_none(),
            rustc_version(),
            env!("CARGO_PKG_RUST_VERSION"),
            env!("CARGO_PKG_VERSION"),
            active_features_json(),
            cfg.max_train
                .map(|limit| limit.to_string())
                .unwrap_or_else(|| "null".to_string()),
            train.len(),
            cfg.max_queries
                .map(|limit| limit.to_string())
                .unwrap_or_else(|| "null".to_string()),
            test.len(),
            cfg.warmup_queries,
        )
    };

    // Every invocation starts a fresh metadata scope before appending raw rows.
    if should_emit_run_meta(cfg.json, cfg.resume, cfg.results_path.exists()) {
        emit_result(&cfg.results_path, &meta());
    }

    if !cfg.json {
        println!("Train: {} vectors x {} dims", train.len(), dim);
        println!("Test:  {} queries", test.len());
        println!("Ground truth: {} neighbors per query\n", k_gt);
    }

    let completed = if cfg.resume {
        load_completed_results_for_run(
            &cfg.results_path,
            &cfg.data_dir,
            cfg.max_train,
            cfg.max_queries,
            cfg.warmup_queries,
            cfg.seed,
            cfg.repeat,
        )
    } else {
        Default::default()
    };
    if cfg.json && cfg.resume && completed.has_mismatched_meta && !completed.has_matching_meta {
        let message = format!(
            "Ignoring existing resume rows in {}: no _meta entry matches dataset {}",
            cfg.results_path.display(),
            cfg.data_dir
        );
        eprintln!("{}", message);
        emit_result(&cfg.results_path, &meta());
    }
    if !completed.counts.is_empty() {
        eprintln!(
            "Resuming: found result rows for {} algorithm(s): {}",
            completed.counts.len(),
            completed
                .counts
                .iter()
                .map(|(algo, count)| format!("{algo}={count}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        eprintln!("Results file: {}\n", cfg.results_path.display());
    } else {
        eprintln!("Results file: {}\n", cfg.results_path.display());
    }

    for algo in &cfg.algos {
        if request_completed(&completed, algo, &cfg, dim, train.len(), test.len()) {
            continue;
        }
        match algo.as_str() {
            "external_hnsw_rs" => {
                external_hnsw_rs::run_hnsw_rs(&cfg, &train, &test, &neighbors);
            }

            "external_usearch" => {
                usearch_baseline::run_usearch(&cfg, &train, &test, &neighbors, dim);
            }

            #[cfg(feature = "hnsw")]
            "hnsw" => run_hnsw(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "hnsw"))]
            "hnsw" => {
                eprintln!("HNSW not available (compile with --features hnsw)");
            }

            #[cfg(feature = "nsw")]
            "nsw" if !cfg.is_euclidean => run_nsw(&cfg, &train, &test, &neighbors, dim),

            #[cfg(feature = "nsw")]
            "nsw" => {
                eprintln!("nsw: skipping (cosine-only, dataset is euclidean)");
            }

            #[cfg(not(feature = "nsw"))]
            "nsw" => {
                eprintln!("NSW not available (compile with --features nsw)");
            }

            #[cfg(feature = "ivf_pq")]
            "ivfpq" => run_ivfpq(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "ivf_pq"))]
            "ivfpq" => {
                eprintln!("IVF-PQ not available (compile with --features ivf_pq)");
            }

            #[cfg(feature = "ivf_avq")]
            "ivf_avq" if !cfg.is_euclidean => run_ivf_avq(&cfg, &train, &test, &neighbors, dim),

            #[cfg(feature = "ivf_avq")]
            "ivf_avq" => {
                eprintln!("ivf_avq: skipping (MIPS/angular-oriented, dataset is euclidean)");
            }

            #[cfg(not(feature = "ivf_avq"))]
            "ivf_avq" => {
                eprintln!("IVF-AVQ not available (compile with --features ivf_avq)");
            }

            #[cfg(feature = "emg")]
            "emg" if !cfg.is_euclidean => run_emg(&cfg, &train, &test, &neighbors, dim),
            #[cfg(feature = "emg")]
            "emg" => eprintln!("emg: skipping (cosine-only, dataset is euclidean)"),
            #[cfg(not(feature = "emg"))]
            "emg" => eprintln!("EMG not available (compile with --features emg)"),

            #[cfg(feature = "nsg")]
            "nsg" if !cfg.is_euclidean => run_nsg(&cfg, &train, &test, &neighbors, dim),
            #[cfg(feature = "nsg")]
            "nsg" => eprintln!("nsg: skipping (cosine-only, dataset is euclidean)"),
            #[cfg(not(feature = "nsg"))]
            "nsg" => eprintln!("NSG not available (compile with --features nsg)"),

            #[cfg(feature = "hnsw")]
            "dual_branch" => run_dual_branch(&cfg, &train, &test, &neighbors, dim),
            #[cfg(not(feature = "hnsw"))]
            "dual_branch" => {
                eprintln!("DualBranchHNSW not available (compile with --features hnsw)");
            }

            #[cfg(feature = "hnsw")]
            "deg" => run_deg(&cfg, &train, &test, &neighbors, dim),
            #[cfg(not(feature = "hnsw"))]
            "deg" => eprintln!("DEG not available (compile with --features hnsw)"),

            #[cfg(feature = "pipnn")]
            "pipnn" if !cfg.is_euclidean => run_pipnn(&cfg, &train, &test, &neighbors, dim),
            #[cfg(feature = "pipnn")]
            "pipnn" => eprintln!("pipnn: skipping (cosine-only, dataset is euclidean)"),
            #[cfg(not(feature = "pipnn"))]
            "pipnn" => eprintln!("PiPNN not available (compile with --features pipnn)"),

            #[cfg(feature = "sng")]
            "sng" if !cfg.is_euclidean => run_sng(&cfg, &train, &test, &neighbors, dim),
            #[cfg(feature = "sng")]
            "sng" => eprintln!("sng: skipping (cosine-only, dataset is euclidean)"),
            #[cfg(not(feature = "sng"))]
            "sng" => eprintln!("SNG not available (compile with --features sng)"),

            #[cfg(feature = "vamana")]
            "vamana" => run_vamana(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "vamana"))]
            "vamana" => {
                eprintln!("Vamana not available (compile with --features vamana)");
            }

            #[cfg(feature = "diskann")]
            "diskann" => run_diskann(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "diskann"))]
            "diskann" => {
                eprintln!("DiskANN not available (compile with --features diskann)");
            }

            #[cfg(feature = "ivf_rabitq")]
            "ivf_rabitq" => run_ivf_rabitq(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "ivf_rabitq"))]
            "ivf_rabitq" => {
                eprintln!("IVF-RaBitQ not available (compile with --features ivf_rabitq)");
            }

            #[cfg(feature = "finger")]
            "finger" if !cfg.is_euclidean => run_finger(&cfg, &train, &test, &neighbors, dim),
            #[cfg(feature = "finger")]
            "finger" => eprintln!("finger: skipping (cosine-only, dataset is euclidean)"),
            #[cfg(not(feature = "finger"))]
            "finger" => eprintln!("FINGER not available (compile with --features finger)"),

            #[cfg(feature = "fresh_graph")]
            "fresh_graph" => run_fresh_graph(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "fresh_graph"))]
            "fresh_graph" => {
                eprintln!("FreshGraph not available (compile with --features fresh_graph)");
            }

            #[cfg(feature = "store")]
            "store" => run_store(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "store"))]
            "store" => {
                eprintln!("Store not available (compile with --features store)");
            }

            #[cfg(feature = "fresh_graph")]
            "fresh_graph_churn" => run_fresh_graph_churn(&cfg, &train, &test, dim),

            #[cfg(not(feature = "fresh_graph"))]
            "fresh_graph_churn" => {
                eprintln!("FreshGraph churn not available (compile with --features fresh_graph)");
            }

            #[cfg(feature = "hnsw")]
            "inplace" => run_inplace(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "hnsw"))]
            "inplace" => {
                eprintln!("InPlace not available (compile with --features hnsw)");
            }

            #[cfg(feature = "hnsw")]
            "inplace_churn" => run_inplace_churn(&cfg, &train, &test, dim),

            #[cfg(not(feature = "hnsw"))]
            "inplace_churn" => {
                eprintln!("InPlace churn not available (compile with --features hnsw)");
            }

            #[cfg(feature = "hnsw")]
            "lsm_churn" => run_lsm_churn(&cfg, &train, &test, dim),

            #[cfg(not(feature = "hnsw"))]
            "lsm_churn" => {
                eprintln!("LSM churn not available (compile with --features hnsw)");
            }

            #[cfg(feature = "filtered_graph")]
            "filtered_graph" => run_filtered_graph(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "filtered_graph"))]
            "filtered_graph" => {
                eprintln!("FilteredGraph not available (compile with --features filtered_graph)");
            }

            #[cfg(feature = "rp_quant")]
            "rp_quant" => run_rp_quant(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "rp_quant"))]
            "rp_quant" => {
                eprintln!("RpQuant not available (compile with --features rp_quant)");
            }

            #[cfg(feature = "sparse_mips")]
            "sparse_mips" => run_sparse_mips(&cfg, &train, &test, &neighbors),

            #[cfg(not(feature = "sparse_mips"))]
            "sparse_mips" => {
                eprintln!("sparse_mips not available (compile with --features sparse_mips)");
            }

            #[cfg(all(feature = "hnsw", feature = "ivf_rabitq"))]
            "symphony_qg" if !cfg.is_euclidean => {
                run_symphony_qg(&cfg, &train, &test, &neighbors, dim);
            }

            #[cfg(all(feature = "hnsw", feature = "ivf_rabitq"))]
            "symphony_qg" => {
                eprintln!("symphony_qg: skipping (cosine-only, dataset is euclidean; use symphony_qg_vr for L2)");
            }

            #[cfg(not(all(feature = "hnsw", feature = "ivf_rabitq")))]
            "symphony_qg" => {
                eprintln!("SymphonyQG not available (compile with --features hnsw,ivf_rabitq)");
            }

            #[cfg(feature = "curator")]
            "curator" => run_curator(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "curator"))]
            "curator" => {
                eprintln!("Curator not available (compile with --features curator)");
            }

            #[cfg(all(feature = "range_filtered", feature = "hnsw"))]
            "range_filtered" => run_range_filtered(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(all(feature = "range_filtered", feature = "hnsw")))]
            "range_filtered" => {
                eprintln!(
                    "range_filtered not available (compile with --features range_filtered,hnsw)"
                );
            }

            #[cfg(feature = "binary_index")]
            "binary_index" => run_binary_index(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "binary_index"))]
            "binary_index" => {
                eprintln!("BinaryIndex not available (compile with --features binary_index)");
            }

            #[cfg(feature = "sq4")]
            "sq4" => run_sq4(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "sq4"))]
            "sq4" => {
                eprintln!("SQ4 not available (compile with --features sq4)");
            }

            #[cfg(all(feature = "hnsw", feature = "sq4"))]
            "sq4u" => run_sq4u(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(all(feature = "hnsw", feature = "sq4")))]
            "sq4u" => {
                eprintln!("SQ4U not available (compile with --features hnsw,sq4)");
            }

            #[cfg(all(feature = "hnsw", feature = "sq8"))]
            "sq8u" => run_sq8u(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(all(feature = "hnsw", feature = "sq8")))]
            "sq8u" => {
                eprintln!("SQ8U not available (compile with --features hnsw,sq8)");
            }

            #[cfg(all(feature = "hnsw", feature = "ivf_rabitq"))]
            "symphony_qg_vr" => run_symphony_qg_vr(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(all(feature = "hnsw", feature = "ivf_rabitq")))]
            "symphony_qg_vr" => {
                eprintln!("SymphonyQG-VR not available (compile with --features hnsw,ivf_rabitq)");
            }

            #[cfg(feature = "hnsw")]
            "adsampling" => run_adsampling(&cfg, &train, &test, &neighbors, dim),

            #[cfg(feature = "lsh")]
            "lsh" => run_lsh(&cfg, &train, &test, &neighbors, dim),

            #[cfg(not(feature = "lsh"))]
            "lsh" => {
                eprintln!("LSH not available (compile with --features lsh)");
            }

            #[cfg(feature = "hnsw")]
            "hnsw_prt" => run_hnsw_prt(&cfg, &train, &test, &neighbors, dim),

            "brute" => run_brute(&cfg, &train, &test, &neighbors),

            #[cfg(feature = "kdtree")]
            "kdtree" => run_kdtree(&cfg, &train, &test, &neighbors, dim),
            #[cfg(not(feature = "kdtree"))]
            "kdtree" => eprintln!("KD-Tree not available (compile with --features kdtree)"),

            #[cfg(feature = "balltree")]
            "balltree" => run_balltree(&cfg, &train, &test, &neighbors, dim),
            #[cfg(not(feature = "balltree"))]
            "balltree" => eprintln!("Ball Tree not available (compile with --features balltree)"),

            #[cfg(feature = "rptree")]
            "rptree" => run_rptree(&cfg, &train, &test, &neighbors, dim),
            #[cfg(not(feature = "rptree"))]
            "rptree" => eprintln!("RP-Tree not available (compile with --features rptree)"),

            #[cfg(feature = "rptree")]
            "rp_forest" => run_rp_forest(&cfg, &train, &test, &neighbors, dim),
            #[cfg(not(feature = "rptree"))]
            "rp_forest" => eprintln!("RP-Forest not available (compile with --features rptree)"),

            #[cfg(feature = "kmeans_tree")]
            "kmeans_tree" => run_kmeans_tree(&cfg, &train, &test, &neighbors, dim),
            #[cfg(not(feature = "kmeans_tree"))]
            "kmeans_tree" => {
                eprintln!("K-Means Tree not available (compile with --features kmeans_tree)");
            }

            other => {
                eprintln!(
                    "Unknown algorithm: {}. Options: {}",
                    other,
                    algorithm_options_help()
                );
            }
        }
    }

    Ok(())
}

// File loading now in examples/common/mod.rs (shared across benchmark examples).
