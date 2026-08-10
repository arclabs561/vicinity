#![allow(clippy::expect_used, clippy::unwrap_used)]

#[cfg(any(feature = "ivf_pq", feature = "ivf_avq"))]
use std::cell::RefCell;
use std::time::Instant;

use crate::support::{current_rss_kb, emit_result, evaluate, print_header, print_row, Config};

#[cfg(feature = "ivf_pq")]
use crate::support::ivfpq_params_json;
#[cfg(any(feature = "ivf_pq", feature = "ivf_avq", feature = "ivf_rabitq"))]
use crate::support::nprobe_values;
#[cfg(any(feature = "ivf_pq", feature = "ivf_avq"))]
use crate::support::warmup_queries;
#[cfg(any(feature = "ivf_pq", feature = "ivf_avq"))]
use crate::support::BenchResult;
#[cfg(any(feature = "ivf_pq", feature = "ivf_avq"))]
use crate::support::StorageDiagnostics;
#[cfg(any(
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "ivf_rabitq",
    feature = "rp_quant",
    feature = "binary_index",
    feature = "lsh",
    feature = "sq4",
    feature = "sq8",
    all(feature = "hnsw", feature = "ivf_rabitq", feature = "serde")
))]
use crate::support::{dir_size_bytes, json_line_with_storage, ResultStorage};
#[cfg(feature = "ivf_avq")]
use crate::support::{ivfavq_num_reorder_values, ivfavq_params_json};

#[cfg(any(
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "ivf_rabitq",
    feature = "rp_quant",
    feature = "binary_index",
    feature = "lsh",
    feature = "sq4",
    feature = "sq8",
    all(feature = "hnsw", feature = "ivf_rabitq", feature = "serde")
))]
fn snapshot_storage(load_time_s: f64, index_bytes: Option<u64>) -> ResultStorage<'static> {
    ResultStorage {
        storage_mode: "snapshot_loaded",
        cache_state: "warm_after_load",
        load_time_s: Some(load_time_s),
        index_bytes,
        index_bytes_kind: None,
        diagnostics: None,
    }
}

#[cfg(any(feature = "ivf_pq", feature = "ivf_avq"))]
fn opened_storage_with_diagnostics(
    storage_mode: &'static str,
    load_time_s: f64,
    index_bytes: Option<u64>,
    diagnostics: StorageDiagnostics,
) -> ResultStorage<'static> {
    ResultStorage {
        storage_mode,
        cache_state: "warm_after_open",
        load_time_s: Some(load_time_s),
        index_bytes,
        index_bytes_kind: None,
        diagnostics: Some(diagnostics),
    }
}

#[cfg(any(feature = "ivf_pq", feature = "ivf_avq"))]
fn add_multi_recall(totals: &mut [f64; 3], truth: &[i32], results: &[(u32, f32)]) {
    use std::collections::HashSet;
    for (slot, depth) in [1, 10, 100].into_iter().enumerate() {
        let depth = depth.min(truth.len()).max(1);
        let expected: HashSet<u32> = truth.iter().take(depth).map(|&id| id as u32).collect();
        let found: HashSet<u32> = results.iter().take(depth).map(|row| row.0).collect();
        totals[slot] += expected.intersection(&found).count() as f64 / depth as f64;
    }
}

#[cfg(feature = "ivf_pq")]
fn evaluate_ivfpq_file_reranked(
    searcher: &RefCell<vicinity::ivf_pq::IVFPQFileSearcher>,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    _k: usize,
    candidate_pool: usize,
) -> (BenchResult, StorageDiagnostics) {
    let k = neighbors.first().map_or(1, |row| row.len().min(100));
    let warmup_count = warmup_queries().min(test.len());
    for query in test.iter().take(warmup_count) {
        let _ = searcher
            .borrow_mut()
            .search_reranked(query, k, candidate_pool);
    }

    let mut recalls = [0.0; 3];
    let mut latencies_us: Vec<f64> = Vec::with_capacity(test.len());
    let mut raw_vector_reads = 0usize;
    let mut raw_vector_bytes = 0usize;
    let mut reranked_candidates = 0usize;

    for (i, query) in test.iter().enumerate() {
        let q_start = Instant::now();
        let (results, diagnostics) = searcher
            .borrow_mut()
            .search_reranked_with_diagnostics(query, k, candidate_pool)
            .expect("IVF-PQ file-backed rerank failed");
        let q_elapsed = q_start.elapsed();
        latencies_us.push(q_elapsed.as_nanos() as f64 / 1000.0);

        raw_vector_reads += diagnostics.raw_vector_reads;
        raw_vector_bytes += diagnostics.raw_vector_bytes;
        reranked_candidates += diagnostics.reranked_candidates;

        add_multi_recall(&mut recalls, &neighbors[i], &results);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = latencies_us.len();
    let total_us: f64 = latencies_us.iter().sum();
    let queries = n.max(1) as f64;

    (
        BenchResult {
            recall_at_k: recalls[1] / queries,
            recall_at_1: recalls[0] / queries,
            recall_at_100: recalls[2] / queries,
            search_k: k,
            qps: queries / (total_us / 1_000_000.0),
            latency_us: total_us / queries,
            p50_us: latencies_us[n / 2],
            p95_us: latencies_us[(n as f64 * 0.95) as usize],
            p99_us: latencies_us[(n as f64 * 0.99) as usize],
        },
        StorageDiagnostics {
            avg_vector_reads: raw_vector_reads as f64 / queries,
            avg_vector_bytes: raw_vector_bytes as f64 / queries,
            avg_retained_candidates: reranked_candidates as f64 / queries,
            ..StorageDiagnostics::default()
        },
    )
}

#[cfg(feature = "ivf_avq")]
fn evaluate_ivfavq_file(
    searcher: &RefCell<vicinity::ivf_avq::IVFAVQFileSearcher>,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    _k: usize,
) -> (BenchResult, StorageDiagnostics) {
    let k = neighbors.first().map_or(1, |row| row.len().min(100));
    let warmup_count = warmup_queries().min(test.len());
    for query in test.iter().take(warmup_count) {
        let _ = searcher.borrow_mut().search(query, k);
    }

    let mut recalls = [0.0; 3];
    let mut latencies_us: Vec<f64> = Vec::with_capacity(test.len());
    let mut probed_lists = 0usize;
    let mut scanned_vectors = 0usize;
    let mut partition_reads = 0usize;
    let mut partition_bytes = 0usize;
    let mut code_reads = 0usize;
    let mut code_bytes = 0usize;
    let mut vector_reads = 0usize;
    let mut vector_bytes = 0usize;
    let mut retained_candidates = 0usize;

    for (i, query) in test.iter().enumerate() {
        let q_start = Instant::now();
        let (results, diagnostics) = searcher
            .borrow_mut()
            .search_with_diagnostics(query, k)
            .expect("IVF-AVQ file-backed search failed");
        let q_elapsed = q_start.elapsed();
        latencies_us.push(q_elapsed.as_nanos() as f64 / 1000.0);

        probed_lists += diagnostics.probed_lists;
        scanned_vectors += diagnostics.scanned_vectors;
        partition_reads += diagnostics.partition_reads;
        partition_bytes += diagnostics.partition_bytes;
        code_reads += diagnostics.code_reads;
        code_bytes += diagnostics.code_bytes;
        vector_reads += diagnostics.raw_vector_reads;
        vector_bytes += diagnostics.raw_vector_bytes;
        retained_candidates += diagnostics.retained_candidates;

        add_multi_recall(&mut recalls, &neighbors[i], &results);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = latencies_us.len();
    let total_us: f64 = latencies_us.iter().sum();
    let queries = n.max(1) as f64;

    (
        BenchResult {
            recall_at_k: recalls[1] / queries,
            recall_at_1: recalls[0] / queries,
            recall_at_100: recalls[2] / queries,
            search_k: k,
            qps: queries / (total_us / 1_000_000.0),
            latency_us: total_us / queries,
            p50_us: latencies_us[n / 2],
            p95_us: latencies_us[(n as f64 * 0.95) as usize],
            p99_us: latencies_us[(n as f64 * 0.99) as usize],
        },
        StorageDiagnostics {
            avg_probed_lists: probed_lists as f64 / queries,
            avg_scanned_vectors: scanned_vectors as f64 / queries,
            avg_partition_reads: partition_reads as f64 / queries,
            avg_partition_bytes: partition_bytes as f64 / queries,
            avg_code_reads: code_reads as f64 / queries,
            avg_code_bytes: code_bytes as f64 / queries,
            avg_vector_reads: vector_reads as f64 / queries,
            avg_vector_bytes: vector_bytes as f64 / queries,
            avg_retained_candidates: retained_candidates as f64 / queries,
            ..StorageDiagnostics::default()
        },
    )
}

#[cfg(feature = "ivf_pq")]
fn evaluate_ivfpq_file_approx(
    searcher: &RefCell<vicinity::ivf_pq::IVFPQFileSearcher>,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    _k: usize,
) -> (BenchResult, StorageDiagnostics) {
    let k = neighbors.first().map_or(1, |row| row.len().min(100));
    let warmup_count = warmup_queries().min(test.len());
    for query in test.iter().take(warmup_count) {
        let _ = searcher.borrow_mut().search(query, k);
    }

    let mut recalls = [0.0; 3];
    let mut latencies_us: Vec<f64> = Vec::with_capacity(test.len());
    let mut probed_lists = 0usize;
    let mut scanned_vectors = 0usize;
    let mut code_reads = 0usize;
    let mut code_bytes = 0usize;
    let mut retained_candidates = 0usize;

    for (i, query) in test.iter().enumerate() {
        let q_start = Instant::now();
        let (results, diagnostics) = searcher
            .borrow_mut()
            .search_with_diagnostics(query, k)
            .expect("IVF-PQ file-backed approximate search failed");
        let q_elapsed = q_start.elapsed();
        latencies_us.push(q_elapsed.as_nanos() as f64 / 1000.0);

        probed_lists += diagnostics.probed_lists;
        scanned_vectors += diagnostics.scanned_vectors;
        code_reads += diagnostics.code_reads;
        code_bytes += diagnostics.code_bytes;
        retained_candidates += diagnostics.retained_candidates;

        add_multi_recall(&mut recalls, &neighbors[i], &results);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = latencies_us.len();
    let total_us: f64 = latencies_us.iter().sum();
    let queries = n.max(1) as f64;

    (
        BenchResult {
            recall_at_k: recalls[1] / queries,
            recall_at_1: recalls[0] / queries,
            recall_at_100: recalls[2] / queries,
            search_k: k,
            qps: queries / (total_us / 1_000_000.0),
            latency_us: total_us / queries,
            p50_us: latencies_us[n / 2],
            p95_us: latencies_us[(n as f64 * 0.95) as usize],
            p99_us: latencies_us[(n as f64 * 0.99) as usize],
        },
        StorageDiagnostics {
            avg_probed_lists: probed_lists as f64 / queries,
            avg_scanned_vectors: scanned_vectors as f64 / queries,
            avg_code_reads: code_reads as f64 / queries,
            avg_code_bytes: code_bytes as f64 / queries,
            avg_retained_candidates: retained_candidates as f64 / queries,
            ..StorageDiagnostics::default()
        },
    )
}

#[cfg(feature = "ivf_pq")]
pub(crate) fn run_ivfpq(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::ivf_pq::{IVFPQFileSearcher, IVFPQIndex, IVFPQParams};

    struct SnapshotIndexes {
        _temp_dir: tempfile::TempDir,
        loaded: IVFPQIndex,
        load_time_s: f64,
        index_bytes: Option<u64>,
        file_searcher: RefCell<IVFPQFileSearcher>,
        file_load_time_s: f64,
        #[cfg(feature = "persistence")]
        mmap_searcher: RefCell<IVFPQFileSearcher>,
        #[cfg(feature = "persistence")]
        mmap_load_time_s: f64,
    }

    let num_clusters = cfg.pq_num_clusters.unwrap_or(256);
    if num_clusters == 0 {
        eprintln!("IVF-PQ: skipping invalid --pq-clusters=0");
        return;
    }
    // num_codebooks must divide dim evenly. Pick the largest divisor <= 8 unless
    // the caller explicitly requests a PQ shape for benchmarking.
    let num_codebooks = cfg.pq_num_codebooks.unwrap_or_else(|| {
        (1..=8.min(dim))
            .rev()
            .find(|&c| dim.is_multiple_of(c))
            .unwrap_or(1)
    });
    if !dim.is_multiple_of(num_codebooks) {
        eprintln!(
            "IVF-PQ: skipping invalid --pq-codebooks={} for dim={} (must divide dimension)",
            num_codebooks, dim
        );
        return;
    }
    let codebook_size = cfg.pq_codebook_size;
    if codebook_size == 0 || codebook_size > 256 {
        eprintln!(
            "IVF-PQ: skipping invalid --pq-codebook-size={} (expected 1..=256)",
            codebook_size
        );
        return;
    }

    if !cfg.json {
        println!(
            "--- IVF-PQ (clusters={}, codebooks={}, codebook_size={}) ---",
            num_clusters, num_codebooks, codebook_size
        );
    }

    let params = IVFPQParams {
        num_clusters,
        num_codebooks,
        codebook_size,
        nprobe: 1, // will be swept
        seed: cfg.seed,
        ..Default::default()
    };

    let build_start = Instant::now();
    let mut index = IVFPQIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index
        .build_with_training_options(cfg.pq_training_sample_size, cfg.pq_kmeans_max_iter)
        .unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let mut snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for IVF-PQ snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = IVFPQIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        let file_load_start = Instant::now();
        let file_searcher = RefCell::new(IVFPQFileSearcher::load(temp_dir.path()).unwrap());
        let file_load_time_s = file_load_start.elapsed().as_secs_f64();
        #[cfg(feature = "persistence")]
        let mmap_load_start = Instant::now();
        #[cfg(feature = "persistence")]
        let mmap_searcher = RefCell::new(IVFPQFileSearcher::load_mmap(temp_dir.path()).unwrap());
        #[cfg(feature = "persistence")]
        let mmap_load_time_s = mmap_load_start.elapsed().as_secs_f64();
        Some(SnapshotIndexes {
            _temp_dir: temp_dir,
            loaded,
            load_time_s,
            index_bytes,
            file_searcher,
            file_load_time_s,
            #[cfg(feature = "persistence")]
            mmap_searcher,
            #[cfg(feature = "persistence")]
            mmap_load_time_s,
        })
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

    // Sweep nprobe values (analogous to ef_search for graph methods)
    for nprobe in nprobe_values(cfg, num_clusters) {
        index.set_nprobe(nprobe);
        let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = ivfpq_params_json(
                num_clusters,
                num_codebooks,
                codebook_size,
                nprobe,
                None,
                cfg.pq_training_sample_size,
                cfg.pq_kmeans_max_iter,
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "ivfpq",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &ResultStorage {
                        index_bytes,
                        ..ResultStorage::default()
                    },
                ),
            );
        } else {
            print_row(&format!("np={}", nprobe), &result);
        }

        if let Some(snapshot) = snapshot_index.as_mut() {
            snapshot.loaded.set_nprobe(nprobe);
            snapshot.file_searcher.borrow_mut().set_nprobe(nprobe);
            #[cfg(feature = "persistence")]
            snapshot.mmap_searcher.borrow_mut().set_nprobe(nprobe);
            let loaded_result = evaluate(
                &|q, k| snapshot.loaded.search(q, k).unwrap(),
                test,
                neighbors,
                10,
            );
            let (file_result, file_diagnostics) =
                evaluate_ivfpq_file_approx(&snapshot.file_searcher, test, neighbors, 10);
            #[cfg(feature = "persistence")]
            let (mmap_result, mmap_diagnostics) =
                evaluate_ivfpq_file_approx(&snapshot.mmap_searcher, test, neighbors, 10);
            let params_json = ivfpq_params_json(
                num_clusters,
                num_codebooks,
                codebook_size,
                nprobe,
                None,
                cfg.pq_training_sample_size,
                cfg.pq_kmeans_max_iter,
            );
            if cfg.json {
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "ivfpq",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(snapshot.load_time_s, snapshot.index_bytes),
                    ),
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "ivfpq",
                        &params_json,
                        build_time_s,
                        rss,
                        &file_result,
                        &opened_storage_with_diagnostics(
                            "file",
                            snapshot.file_load_time_s,
                            snapshot.index_bytes,
                            file_diagnostics,
                        ),
                    ),
                );
                #[cfg(feature = "persistence")]
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "ivfpq",
                        &params_json,
                        build_time_s,
                        rss,
                        &mmap_result,
                        &opened_storage_with_diagnostics(
                            "mmap",
                            snapshot.mmap_load_time_s,
                            snapshot.index_bytes,
                            mmap_diagnostics,
                        ),
                    ),
                );
            } else {
                print_row(&format!("np={} snapshot_loaded", nprobe), &loaded_result);
                print_row(&format!("np={} file", nprobe), &file_result);
                #[cfg(feature = "persistence")]
                print_row(&format!("np={} mmap", nprobe), &mmap_result);
            }
        }

        for &rerank_pool in &cfg.pq_rerank_pools {
            if rerank_pool == 0 {
                continue;
            }
            let result = evaluate(
                &|q, k| index.search_reranked(q, k, rerank_pool).unwrap(),
                test,
                neighbors,
                10,
            );
            if cfg.json {
                let params_json = ivfpq_params_json(
                    num_clusters,
                    num_codebooks,
                    codebook_size,
                    nprobe,
                    Some(rerank_pool),
                    cfg.pq_training_sample_size,
                    cfg.pq_kmeans_max_iter,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "ivfpq_rerank",
                        &params_json,
                        build_time_s,
                        rss,
                        &result,
                        &ResultStorage {
                            index_bytes,
                            ..ResultStorage::default()
                        },
                    ),
                );
            } else {
                print_row(&format!("np={} rr={}", nprobe, rerank_pool), &result);
            }

            if let Some(snapshot) = snapshot_index.as_mut() {
                snapshot.loaded.set_nprobe(nprobe);
                snapshot.file_searcher.borrow_mut().set_nprobe(nprobe);
                #[cfg(feature = "persistence")]
                snapshot.mmap_searcher.borrow_mut().set_nprobe(nprobe);
                let loaded_result = evaluate(
                    &|q, k| snapshot.loaded.search_reranked(q, k, rerank_pool).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                let (file_result, file_diagnostics) = evaluate_ivfpq_file_reranked(
                    &snapshot.file_searcher,
                    test,
                    neighbors,
                    10,
                    rerank_pool,
                );
                #[cfg(feature = "persistence")]
                let (mmap_result, mmap_diagnostics) = evaluate_ivfpq_file_reranked(
                    &snapshot.mmap_searcher,
                    test,
                    neighbors,
                    10,
                    rerank_pool,
                );
                let params_json = ivfpq_params_json(
                    num_clusters,
                    num_codebooks,
                    codebook_size,
                    nprobe,
                    Some(rerank_pool),
                    cfg.pq_training_sample_size,
                    cfg.pq_kmeans_max_iter,
                );
                if cfg.json {
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "ivfpq_rerank",
                            &params_json,
                            build_time_s,
                            rss,
                            &loaded_result,
                            &snapshot_storage(snapshot.load_time_s, snapshot.index_bytes),
                        ),
                    );
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "ivfpq_rerank",
                            &params_json,
                            build_time_s,
                            rss,
                            &file_result,
                            &opened_storage_with_diagnostics(
                                "file",
                                snapshot.file_load_time_s,
                                snapshot.index_bytes,
                                file_diagnostics,
                            ),
                        ),
                    );
                    #[cfg(feature = "persistence")]
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "ivfpq_rerank",
                            &params_json,
                            build_time_s,
                            rss,
                            &mmap_result,
                            &opened_storage_with_diagnostics(
                                "mmap",
                                snapshot.mmap_load_time_s,
                                snapshot.index_bytes,
                                mmap_diagnostics,
                            ),
                        ),
                    );
                } else {
                    print_row(
                        &format!("np={} rr={} snapshot_loaded", nprobe, rerank_pool),
                        &loaded_result,
                    );
                    print_row(
                        &format!("np={} rr={} file", nprobe, rerank_pool),
                        &file_result,
                    );
                    #[cfg(feature = "persistence")]
                    print_row(
                        &format!("np={} rr={} mmap", nprobe, rerank_pool),
                        &mmap_result,
                    );
                }
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "ivf_avq")]
pub(crate) fn run_ivf_avq(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::ivf_avq::{IVFAVQFileSearcher, IVFAVQIndex, IVFAVQParams};

    struct SnapshotIndexes {
        _temp_dir: tempfile::TempDir,
        loaded: IVFAVQIndex,
        load_time_s: f64,
        file_searcher: RefCell<IVFAVQFileSearcher>,
        file_load_time_s: f64,
        #[cfg(feature = "persistence")]
        mmap_searcher: RefCell<IVFAVQFileSearcher>,
        #[cfg(feature = "persistence")]
        mmap_load_time_s: f64,
        index_bytes: Option<u64>,
    }

    let num_partitions = 256.min(train.len()).max(1);
    let num_codebooks = (1..=16.min(dim))
        .rev()
        .find(|&c| dim.is_multiple_of(c))
        .unwrap_or(1);
    let codebook_size = 256;
    let num_reorder_values = ivfavq_num_reorder_values(cfg);
    let initial_num_reorder = num_reorder_values.first().copied().unwrap_or(100);

    if !cfg.json {
        println!(
            "--- IVF-AVQ (partitions={}, codebooks={}, reorder={:?}) ---",
            num_partitions, num_codebooks, num_reorder_values
        );
    }

    let params = IVFAVQParams {
        num_partitions,
        nprobe: 1,
        num_reorder: initial_num_reorder,
        num_codebooks,
        codebook_size,
        seed: cfg.seed,
    };

    let build_start = Instant::now();
    let mut index = IVFAVQIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let mut snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for IVF-AVQ snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = IVFAVQIndex::load_from_dir(temp_dir.path()).unwrap();
        let load_time_s = load_start.elapsed().as_secs_f64();
        let file_load_start = Instant::now();
        let file_searcher = RefCell::new(IVFAVQFileSearcher::open(temp_dir.path()).unwrap());
        let file_load_time_s = file_load_start.elapsed().as_secs_f64();
        #[cfg(feature = "persistence")]
        let mmap_load_start = Instant::now();
        #[cfg(feature = "persistence")]
        let mmap_searcher = RefCell::new(IVFAVQFileSearcher::open_mmap(temp_dir.path()).unwrap());
        #[cfg(feature = "persistence")]
        let mmap_load_time_s = mmap_load_start.elapsed().as_secs_f64();
        Some(SnapshotIndexes {
            _temp_dir: temp_dir,
            loaded,
            load_time_s,
            file_searcher,
            file_load_time_s,
            #[cfg(feature = "persistence")]
            mmap_searcher,
            #[cfg(feature = "persistence")]
            mmap_load_time_s,
            index_bytes,
        })
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

    for nprobe in nprobe_values(cfg, num_partitions) {
        for &num_reorder in &num_reorder_values {
            index.set_nprobe(nprobe);
            index.set_num_reorder(num_reorder);
            let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
            if cfg.json {
                let params_json = ivfavq_params_json(
                    num_partitions,
                    num_codebooks,
                    codebook_size,
                    nprobe,
                    num_reorder,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "ivf_avq",
                        &params_json,
                        build_time_s,
                        rss,
                        &result,
                        &ResultStorage {
                            index_bytes,
                            ..ResultStorage::default()
                        },
                    ),
                );
            } else {
                print_row(&format!("np={} reorder={}", nprobe, num_reorder), &result);
            }

            if let Some(snapshot) = snapshot_index.as_mut() {
                snapshot.loaded.set_nprobe(nprobe);
                snapshot.loaded.set_num_reorder(num_reorder);
                let loaded_result = evaluate(
                    &|q, k| snapshot.loaded.search(q, k).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                snapshot.file_searcher.borrow_mut().set_nprobe(nprobe);
                snapshot
                    .file_searcher
                    .borrow_mut()
                    .set_num_reorder(num_reorder);
                #[cfg(feature = "persistence")]
                snapshot.mmap_searcher.borrow_mut().set_nprobe(nprobe);
                #[cfg(feature = "persistence")]
                snapshot
                    .mmap_searcher
                    .borrow_mut()
                    .set_num_reorder(num_reorder);
                let (file_result, file_diagnostics) =
                    evaluate_ivfavq_file(&snapshot.file_searcher, test, neighbors, 10);
                #[cfg(feature = "persistence")]
                let (mmap_result, mmap_diagnostics) =
                    evaluate_ivfavq_file(&snapshot.mmap_searcher, test, neighbors, 10);
                let params_json = ivfavq_params_json(
                    num_partitions,
                    num_codebooks,
                    codebook_size,
                    nprobe,
                    num_reorder,
                );
                if cfg.json {
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "ivf_avq",
                            &params_json,
                            build_time_s,
                            rss,
                            &loaded_result,
                            &snapshot_storage(snapshot.load_time_s, snapshot.index_bytes),
                        ),
                    );
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "ivf_avq",
                            &params_json,
                            build_time_s,
                            rss,
                            &file_result,
                            &opened_storage_with_diagnostics(
                                "file",
                                snapshot.file_load_time_s,
                                snapshot.index_bytes,
                                file_diagnostics,
                            ),
                        ),
                    );
                    #[cfg(feature = "persistence")]
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "ivf_avq",
                            &params_json,
                            build_time_s,
                            rss,
                            &mmap_result,
                            &opened_storage_with_diagnostics(
                                "mmap",
                                snapshot.mmap_load_time_s,
                                snapshot.index_bytes,
                                mmap_diagnostics,
                            ),
                        ),
                    );
                } else {
                    print_row(
                        &format!("np={} reorder={} snapshot_loaded", nprobe, num_reorder),
                        &loaded_result,
                    );
                    print_row(
                        &format!("np={} reorder={} file", nprobe, num_reorder),
                        &file_result,
                    );
                    #[cfg(feature = "persistence")]
                    print_row(
                        &format!("np={} reorder={} mmap", nprobe, num_reorder),
                        &mmap_result,
                    );
                }
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "ivf_rabitq")]
pub(crate) fn run_ivf_rabitq(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams};

    let num_clusters = 256;

    if !cfg.json {
        println!(
            "--- IVF-RaBitQ (clusters={}, total_bits=4) ---",
            num_clusters
        );
    }

    let params = IVFRaBitQParams {
        num_clusters,
        nprobe: 10,
        total_bits: 4,
        seed: cfg.seed,
    };

    let build_start = Instant::now();
    let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let mut snapshot_index = if cfg.snapshot_load {
        let temp_dir =
            tempfile::tempdir().expect("create temp dir for IVF-RaBitQ snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = IVFRaBitQIndex::load_from_dir(temp_dir.path()).unwrap();
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

    for nprobe in nprobe_values(cfg, num_clusters) {
        index.set_nprobe(nprobe);
        let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!(
                "{{\"num_clusters\":{},\"total_bits\":4,\"nprobe\":{}}}",
                num_clusters, nprobe
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "ivf_rabitq",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &ResultStorage {
                        index_bytes,
                        ..ResultStorage::default()
                    },
                ),
            );
        } else {
            print_row(&format!("np={}", nprobe), &result);
        }

        if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index.as_mut() {
            loaded.set_nprobe(nprobe);
            let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
            let params_json = format!(
                "{{\"num_clusters\":{},\"total_bits\":4,\"nprobe\":{}}}",
                num_clusters, nprobe
            );
            if cfg.json {
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "ivf_rabitq",
                        &params_json,
                        build_time_s,
                        rss,
                        &loaded_result,
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            } else {
                print_row(&format!("np={} snapshot_loaded", nprobe), &loaded_result);
            }
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(feature = "rp_quant")]
pub(crate) fn run_rp_quant(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::rp_quant::{RpQuantIndex, RpQuantParams};

    let projected_dim = 64.min(dim);

    if !cfg.json {
        println!(
            "--- RpQuant (projected_dim={}, rerank=10) ---",
            projected_dim
        );
    }

    let params = RpQuantParams {
        projected_dim,
        rerank_factor: 10,
        seed: cfg.seed,
    };

    let build_start = Instant::now();
    let mut index = RpQuantIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for RpQuant snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = RpQuantIndex::load_from_dir(temp_dir.path()).unwrap();
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
    let params_json = format!(
        "{{\"projected_dim\":{},\"rerank_factor\":10}}",
        projected_dim
    );
    if cfg.json {
        emit_result(
            &cfg.results_path,
            &json_line_with_storage(
                "rp_quant",
                &params_json,
                build_time_s,
                rss,
                &result,
                &ResultStorage {
                    index_bytes,
                    ..ResultStorage::default()
                },
            ),
        );
    } else {
        print_row("--", &result);
        println!();
    }

    if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index {
        let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
        if cfg.json {
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "rp_quant",
                    &params_json,
                    build_time_s,
                    rss,
                    &loaded_result,
                    &snapshot_storage(load_time_s, index_bytes),
                ),
            );
        } else {
            print_row("snapshot_loaded", &loaded_result);
            println!();
        }
    }
}

#[cfg(feature = "sq4")]
pub(crate) fn run_sq4(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::sq4::{SQ4Index, SQ4Params};

    if !cfg.json {
        println!("--- SQ4 (4-bit scalar quantization, rerank=10) ---");
    }

    let params = SQ4Params { rerank_factor: 10 };

    let build_start = Instant::now();
    let mut index = SQ4Index::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for SQ4 snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = SQ4Index::load_from_dir(temp_dir.path()).unwrap();
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
    let params_json = "{\"rerank_factor\":10}";
    if cfg.json {
        emit_result(
            &cfg.results_path,
            &json_line_with_storage(
                "sq4",
                params_json,
                build_time_s,
                rss,
                &result,
                &ResultStorage {
                    index_bytes,
                    ..ResultStorage::default()
                },
            ),
        );
    } else {
        print_row("--", &result);
        println!();
    }

    if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index {
        let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
        if cfg.json {
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "sq4",
                    params_json,
                    build_time_s,
                    rss,
                    &loaded_result,
                    &snapshot_storage(load_time_s, index_bytes),
                ),
            );
        } else {
            print_row("snapshot_loaded", &loaded_result);
            println!();
        }
    }
}

#[cfg(all(feature = "hnsw", feature = "ivf_rabitq"))]
pub(crate) fn run_symphony_qg(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::SymphonyQGIndex;

    if !cfg.json {
        println!("--- SymphonyQG (m=16, 4-bit RaBitQ) ---");
    }

    let build_start = Instant::now();
    let mut index = SymphonyQGIndex::new(dim, 16, 16).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    #[cfg(feature = "serde")]
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir =
            tempfile::tempdir().expect("create temp dir for SymphonyQG snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = SymphonyQGIndex::load_from_dir(temp_dir.path()).unwrap();
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
        let rerank_pool = (ef * 2).max(100);
        let result = evaluate(
            &|q, k| index.search_reranked(q, k, ef, rerank_pool).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"m\":16,\"ef_search\":{},\"rerank_pool\":{}}}",
                ef, rerank_pool
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "symphony_qg",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &ResultStorage {
                        index_bytes,
                        ..ResultStorage::default()
                    },
                ),
            );
            #[cfg(feature = "serde")]
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_reranked(q, k, ef, rerank_pool).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "symphony_qg",
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
            #[cfg(feature = "serde")]
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_reranked(q, k, ef, rerank_pool).unwrap(),
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

#[cfg(feature = "binary_index")]
pub(crate) fn run_binary_index(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::binary_index::{BinaryFlatIndex, BinaryFlatParams};

    let params = BinaryFlatParams {
        rerank_factor: 10,
        seed: cfg.seed,
        ..Default::default()
    };

    if !cfg.json {
        println!("--- BinaryFlat (rerank=10) ---");
    }

    let build_start = Instant::now();
    let mut index = BinaryFlatIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir =
            tempfile::tempdir().expect("create temp dir for BinaryFlat snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = BinaryFlatIndex::load_from_dir(temp_dir.path()).unwrap();
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
    let params_json = r#"{"rerank_factor":10}"#;
    if cfg.json {
        emit_result(
            &cfg.results_path,
            &json_line_with_storage(
                "binary_index",
                params_json,
                build_time_s,
                rss,
                &result,
                &ResultStorage {
                    index_bytes,
                    ..ResultStorage::default()
                },
            ),
        );
    } else {
        print_row("--", &result);
        println!();
    }

    if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index {
        let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
        if cfg.json {
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "binary_index",
                    params_json,
                    build_time_s,
                    rss,
                    &loaded_result,
                    &snapshot_storage(load_time_s, index_bytes),
                ),
            );
        } else {
            print_row("snapshot_loaded", &loaded_result);
            println!();
        }
    }
}

#[cfg(feature = "lsh")]
pub(crate) fn run_lsh(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::lsh::{CrossPolytopeLSHIndex, LSHParams};

    // Sweep over (num_tables, num_probes) combinations.
    let table_counts = [8, 16, 32];
    let probe_counts = [2, 4, 8, 16];

    for &num_tables in &table_counts {
        let params = LSHParams {
            num_tables,
            num_probes: 1, // build once per table count
            seed: Some(cfg.seed),
        };

        if !cfg.json {
            println!("--- LSH (tables={}) ---", num_tables);
        }

        let build_start = Instant::now();
        let mut index = CrossPolytopeLSHIndex::new(dim, params).unwrap();

        // Flatten train vectors for add_vectors.
        let flat: Vec<f32> = train.iter().flat_map(|v| v.iter().copied()).collect();
        index.add_vectors(&flat).unwrap();
        index.build().unwrap();
        let build_time_s = build_start.elapsed().as_secs_f64();
        let rss = current_rss_kb();

        if !cfg.json {
            println!(
                "Build: {:.2}s ({:.0} vectors/sec)\n",
                build_time_s,
                train.len() as f64 / build_time_s
            );
            print_header();
        }

        for &num_probes in &probe_counts {
            if num_probes > dim {
                continue; // probes > dimension is wasteful
            }

            // Rebuild with different probe count (same rotation matrices via same seed).
            let search_params = LSHParams {
                num_tables,
                num_probes,
                seed: Some(cfg.seed),
            };
            let mut search_index = CrossPolytopeLSHIndex::new(dim, search_params).unwrap();
            search_index.add_vectors(&flat).unwrap();
            search_index.build().unwrap();
            let index_bytes = Some(search_index.memory_usage().total() as u64);

            let result = evaluate(
                &|q, k| search_index.search(q, k).unwrap_or_default(),
                test,
                neighbors,
                10,
            );

            if cfg.json {
                let params_json = format!(
                    "{{\"num_tables\":{},\"num_probes\":{}}}",
                    num_tables, num_probes
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "lsh",
                        &params_json,
                        build_time_s,
                        rss,
                        &result,
                        &ResultStorage {
                            index_bytes,
                            ..ResultStorage::default()
                        },
                    ),
                );
            } else {
                print_row(&format!("probes={}", num_probes), &result);
            }

            if cfg.snapshot_load {
                let temp_dir = tempfile::tempdir().expect("create temp dir for LSH snapshot");
                search_index.save_to_dir(temp_dir.path()).unwrap();
                let index_bytes = dir_size_bytes(temp_dir.path()).ok();
                let load_start = Instant::now();
                let loaded = CrossPolytopeLSHIndex::load_from_dir(temp_dir.path()).unwrap();
                let load_time_s = load_start.elapsed().as_secs_f64();
                let loaded_result = evaluate(
                    &|q, k| loaded.search(q, k).unwrap_or_default(),
                    test,
                    neighbors,
                    10,
                );
                if cfg.json {
                    let params_json = format!(
                        "{{\"num_tables\":{},\"num_probes\":{}}}",
                        num_tables, num_probes
                    );
                    emit_result(
                        &cfg.results_path,
                        &json_line_with_storage(
                            "lsh",
                            &params_json,
                            build_time_s,
                            rss,
                            &loaded_result,
                            &snapshot_storage(load_time_s, index_bytes),
                        ),
                    );
                } else {
                    print_row(
                        &format!("probes={} snapshot_loaded", num_probes),
                        &loaded_result,
                    );
                }
            }
        }

        if !cfg.json {
            println!();
        }
    }
}

#[cfg(all(feature = "hnsw", feature = "sq8"))]
pub(crate) fn run_sq8u(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::{HNSWParams, HNSWSq8Index};
    use vicinity::DistanceMetric;

    let m = cfg.m;
    let ef_construction = cfg.ef_construction;

    if !cfg.json {
        println!("--- SQ8U (M={}, ef_c={}) ---", m, ef_construction);
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
    let mut index = HNSWSq8Index::with_params(dim, params).unwrap();
    let ids: Vec<u32> = (0..train.len() as u32).collect();
    let flat: Vec<f32> = train.iter().flatten().copied().collect();
    index.add_batch(&ids, &flat).unwrap();
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some((index.inner().memory_usage().total() + index.code_memory()) as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for SQ8U snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = HNSWSq8Index::load_from_dir(temp_dir.path()).unwrap();
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
        let rerank_pool = (ef * 2).max(100);
        let result = evaluate(
            &|q, k| index.search_reranked(q, k, ef, rerank_pool).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"rerank_pool\":{}}}",
                m, ef_construction, ef, rerank_pool
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "sq8u",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &ResultStorage {
                        index_bytes,
                        ..ResultStorage::default()
                    },
                ),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }

        if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index.as_ref() {
            let loaded_result = evaluate(
                &|q, k| loaded.search_reranked(q, k, ef, rerank_pool).unwrap(),
                test,
                neighbors,
                10,
            );
            if cfg.json {
                let params_json = format!(
                    "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"rerank_pool\":{}}}",
                    m, ef_construction, ef, rerank_pool
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "sq8u",
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

#[cfg(all(feature = "hnsw", feature = "sq4"))]
pub(crate) fn run_sq4u(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::hnsw::{HNSWParams, HNSWSq4Index};
    use vicinity::DistanceMetric;

    let m = cfg.m;
    let ef_construction = cfg.ef_construction;

    if !cfg.json {
        println!("--- SQ4U (M={}, ef_c={}) ---", m, ef_construction);
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
    let mut index = HNSWSq4Index::with_params(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage().total() as u64);
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for SQ4U snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = HNSWSq4Index::load_from_dir(temp_dir.path()).unwrap();
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
        let rerank_pool = (ef * 2).max(100);
        let result = evaluate(
            &|q, k| index.search_reranked(q, k, ef, rerank_pool).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"rerank_pool\":{}}}",
                m, ef_construction, ef, rerank_pool
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "sq4u",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &ResultStorage {
                        index_bytes,
                        ..ResultStorage::default()
                    },
                ),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }

        if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index.as_ref() {
            let loaded_result = evaluate(
                &|q, k| loaded.search_reranked(q, k, ef, rerank_pool).unwrap(),
                test,
                neighbors,
                10,
            );
            if cfg.json {
                let params_json = format!(
                    "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"rerank_pool\":{}}}",
                    m, ef_construction, ef, rerank_pool
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "sq4u",
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

#[cfg(all(feature = "hnsw", feature = "ivf_rabitq"))]
pub(crate) fn run_symphony_qg_vr(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use qntz::rabitq::RaBitQConfig;
    use vicinity::hnsw::{HNSWParams, SymphonyQGVRIndex};
    use vicinity::DistanceMetric;

    let m = cfg.m;
    let ef_construction = cfg.ef_construction;

    if !cfg.json {
        println!(
            "--- SymphonyQG-VR (M={}, ef_c={}, L2-capable) ---",
            m, ef_construction
        );
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
    let mut index = SymphonyQGVRIndex::new(dim, params, RaBitQConfig::bits4(), cfg.seed).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let index_bytes = Some(index.memory_usage_bytes().total() as u64);
    #[cfg(feature = "serde")]
    let snapshot_index = if cfg.snapshot_load {
        let temp_dir =
            tempfile::tempdir().expect("create temp dir for SymphonyQG-VR snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = SymphonyQGVRIndex::load_from_dir(temp_dir.path()).unwrap();
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
        let rerank_pool = (ef * 2).max(100);
        let result = evaluate(
            &|q, k| index.search_reranked(q, k, ef, rerank_pool).unwrap(),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            let params_json = format!(
                "{{\"m\":{},\"ef_construction\":{},\"ef_search\":{},\"rerank_pool\":{}}}",
                m, ef_construction, ef, rerank_pool
            );
            emit_result(
                &cfg.results_path,
                &json_line_with_storage(
                    "symphony_qg_vr",
                    &params_json,
                    build_time_s,
                    rss,
                    &result,
                    &ResultStorage {
                        index_bytes,
                        ..ResultStorage::default()
                    },
                ),
            );
            #[cfg(feature = "serde")]
            if let Some((_, loaded, load_time_s, index_bytes)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_reranked(q, k, ef, rerank_pool).unwrap(),
                    test,
                    neighbors,
                    10,
                );
                emit_result(
                    &cfg.results_path,
                    &json_line_with_storage(
                        "symphony_qg_vr",
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
            #[cfg(feature = "serde")]
            if let Some((_, loaded, _, _)) = &snapshot_index {
                let loaded_result = evaluate(
                    &|q, k| loaded.search_reranked(q, k, ef, rerank_pool).unwrap(),
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
