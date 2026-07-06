#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::time::Instant;

use crate::support::{
    current_rss_kb, emit_result, evaluate, json_line, print_header, print_row, Config,
};

#[cfg(feature = "ivf_pq")]
use crate::support::ivfpq_params_json;
#[cfg(any(feature = "ivf_pq", feature = "ivf_avq", feature = "ivf_rabitq"))]
use crate::support::nprobe_values;
#[cfg(any(
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "ivf_rabitq",
    feature = "rp_quant",
    feature = "binary_index",
    feature = "sq4",
    feature = "sq8"
))]
use crate::support::{dir_size_bytes, json_line_with_storage, ResultStorage};

#[cfg(any(
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "ivf_rabitq",
    feature = "rp_quant",
    feature = "binary_index",
    feature = "sq4",
    feature = "sq8"
))]
fn snapshot_storage(load_time_s: f64, index_bytes: Option<u64>) -> ResultStorage<'static> {
    ResultStorage {
        storage_mode: "snapshot_loaded",
        cache_state: "warm_after_load",
        load_time_s: Some(load_time_s),
        index_bytes,
    }
}

#[cfg(feature = "ivf_pq")]
pub(crate) fn run_ivfpq(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

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
        seed: 42,
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
    let mut snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for IVF-PQ snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = IVFPQIndex::load_from_dir(temp_dir.path()).unwrap();
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
                &json_line("ivfpq", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("np={}", nprobe), &result);
        }

        if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index.as_mut() {
            loaded.set_nprobe(nprobe);
            let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
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
                        &snapshot_storage(*load_time_s, *index_bytes),
                    ),
                );
            } else {
                print_row(&format!("np={} snapshot_loaded", nprobe), &loaded_result);
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
                    &json_line("ivfpq_rerank", &params_json, build_time_s, rss, &result),
                );
            } else {
                print_row(&format!("np={} rr={}", nprobe, rerank_pool), &result);
            }

            if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index.as_mut() {
                loaded.set_nprobe(nprobe);
                let loaded_result = evaluate(
                    &|q, k| loaded.search_reranked(q, k, rerank_pool).unwrap(),
                    test,
                    neighbors,
                    10,
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
                            &snapshot_storage(*load_time_s, *index_bytes),
                        ),
                    );
                } else {
                    print_row(
                        &format!("np={} rr={} snapshot_loaded", nprobe, rerank_pool),
                        &loaded_result,
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
    use vicinity::ivf_avq::{IVFAVQIndex, IVFAVQParams};

    let num_partitions = 256.min(train.len()).max(1);
    let num_codebooks = (1..=16.min(dim))
        .rev()
        .find(|&c| dim.is_multiple_of(c))
        .unwrap_or(1);
    let codebook_size = 256;
    let num_reorder = 100;

    if !cfg.json {
        println!(
            "--- IVF-AVQ (partitions={}, codebooks={}, reorder={}) ---",
            num_partitions, num_codebooks, num_reorder
        );
    }

    let params = IVFAVQParams {
        num_partitions,
        nprobe: 1,
        num_reorder,
        num_codebooks,
        codebook_size,
        seed: 42,
    };

    let build_start = Instant::now();
    let mut index = IVFAVQIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
    let mut snapshot_index = if cfg.snapshot_load {
        let temp_dir = tempfile::tempdir().expect("create temp dir for IVF-AVQ snapshot benchmark");
        index.save_to_dir(temp_dir.path()).unwrap();
        let index_bytes = dir_size_bytes(temp_dir.path()).ok();
        let load_start = Instant::now();
        let loaded = IVFAVQIndex::load_from_dir(temp_dir.path()).unwrap();
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

    for nprobe in nprobe_values(cfg, num_partitions) {
        index.set_nprobe(nprobe);
        let result = evaluate(&|q, k| index.search(q, k).unwrap(), test, neighbors, 10);
        if cfg.json {
            let params_json = format!(
                "{{\"num_partitions\":{},\"num_codebooks\":{},\"codebook_size\":{},\"nprobe\":{},\"num_reorder\":{}}}",
                num_partitions, num_codebooks, codebook_size, nprobe, num_reorder
            );
            emit_result(
                &cfg.results_path,
                &json_line("ivf_avq", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("np={}", nprobe), &result);
        }

        if let Some((_temp_dir, loaded, load_time_s, index_bytes)) = snapshot_index.as_mut() {
            loaded.set_nprobe(nprobe);
            let loaded_result = evaluate(&|q, k| loaded.search(q, k).unwrap(), test, neighbors, 10);
            let params_json = format!(
                "{{\"num_partitions\":{},\"num_codebooks\":{},\"codebook_size\":{},\"nprobe\":{},\"num_reorder\":{}}}",
                num_partitions, num_codebooks, codebook_size, nprobe, num_reorder
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
        seed: 42,
    };

    let build_start = Instant::now();
    let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
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
                &json_line("ivf_rabitq", &params_json, build_time_s, rss, &result),
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
        seed: 42,
    };

    let build_start = Instant::now();
    let mut index = RpQuantIndex::new(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
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
            &json_line("rp_quant", &params_json, build_time_s, rss, &result),
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
            &json_line("sq4", params_json, build_time_s, rss, &result),
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
                &json_line("symphony_qg", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
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
        seed: 42,
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
            &json_line("binary_index", params_json, build_time_s, rss, &result),
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
            seed: Some(42),
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
                seed: Some(42),
            };
            let mut search_index = CrossPolytopeLSHIndex::new(dim, search_params).unwrap();
            search_index.add_vectors(&flat).unwrap();
            search_index.build().unwrap();

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
                    &json_line("lsh", &params_json, build_time_s, rss, &result),
                );
            } else {
                print_row(&format!("probes={}", num_probes), &result);
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
        seed: Some(42),
        ..Default::default()
    };
    let mut index = HNSWSq8Index::with_params(dim, params).unwrap();
    let ids: Vec<u32> = (0..train.len() as u32).collect();
    let flat: Vec<f32> = train.iter().flatten().copied().collect();
    index.add_batch(&ids, &flat).unwrap();
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
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
                &json_line("sq8u", &params_json, build_time_s, rss, &result),
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

#[cfg(feature = "sq4")]
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
        seed: Some(42),
        ..Default::default()
    };
    let mut index = HNSWSq4Index::with_params(dim, params).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
    index.build().unwrap();
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss = current_rss_kb();
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
                &json_line("sq4u", &params_json, build_time_s, rss, &result),
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
        seed: Some(42),
        ..Default::default()
    };
    let mut index = SymphonyQGVRIndex::new(dim, params, RaBitQConfig::bits4(), 42).unwrap();
    for (i, vec) in train.iter().enumerate() {
        index.add_slice(i as u32, vec).unwrap();
    }
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
                &json_line("symphony_qg_vr", &params_json, build_time_s, rss, &result),
            );
        } else {
            print_row(&format!("ef={}", ef), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}
