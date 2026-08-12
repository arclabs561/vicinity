//! Benchmark-only adapter for `hnsw_rs`.
//!
//! The upstream builder samples graph levels from an OS-seeded RNG and does
//! not expose a caller seed. Sequential insertion makes scheduling controlled,
//! but repeated builds are still intentionally reported as non-deterministic.

use std::time::Instant;

use hnsw_rs::prelude::{AnnT, DistCosine, DistL2, Hnsw, Neighbour};

use crate::support::{
    current_rss_kb, effective_search_values, emit_result, evaluate, evaluation_search_k,
    json_line_with_storage_and_extra_fields, print_header, print_row, Config, ResultStorage,
};

const CRATE_VERSION: &str = "0.3.4";
const MAX_LAYER: usize = 16;

fn params_json(cfg: &Config, ef_search: usize) -> String {
    format!(
        concat!(
            "{{\"crate_version\":\"{}\",\"m\":{},\"ef_construction\":{},",
            "\"ef_search\":{},\"max_layer\":{},\"build_mode\":\"sequential\",",
            "\"seed_control\":\"unavailable\",\"rng_source\":\"os\"}}"
        ),
        CRATE_VERSION, cfg.m, cfg.ef_construction, ef_search, MAX_LAYER
    )
}

fn results(neighbors: Vec<Neighbour>) -> Vec<(u32, f32)> {
    neighbors
        .into_iter()
        .map(|neighbor| {
            (
                u32::try_from(neighbor.d_id).expect("hnsw_rs external ID exceeds u32"),
                neighbor.distance,
            )
        })
        .collect()
}

fn serialized_size<D>(index: &Hnsw<'_, f32, D>) -> u64
where
    D: hnsw_rs::prelude::Distance<f32> + Send + Sync,
{
    let directory = tempfile::tempdir().expect("create hnsw_rs size directory");
    let basename = index
        .file_dump(directory.path(), "external_hnsw_rs")
        .expect("serialize hnsw_rs index");
    ["hnsw.graph", "hnsw.data"]
        .iter()
        .map(|suffix| {
            std::fs::metadata(directory.path().join(format!("{basename}.{suffix}")))
                .expect("read hnsw_rs serialized component metadata")
                .len()
        })
        .sum()
}

fn storage(index_bytes: u64) -> ResultStorage<'static> {
    ResultStorage {
        index_bytes: Some(index_bytes),
        index_bytes_kind: Some("serialized"),
        ..ResultStorage::default()
    }
}

fn emit_search_results<D>(
    cfg: &Config,
    index: &Hnsw<'_, f32, D>,
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    build_time_s: f64,
    rss_kb: Option<u64>,
    index_bytes: u64,
) where
    D: hnsw_rs::prelude::Distance<f32> + Send + Sync,
{
    let search_k = evaluation_search_k(neighbors);
    for ef_search in effective_search_values(&cfg.ef_search_values, search_k) {
        let result = evaluate(
            &|query, k| results(index.search(query, k, ef_search)),
            test,
            neighbors,
            10,
        );
        if cfg.json {
            emit_result(
                &cfg.results_path,
                &json_line_with_storage_and_extra_fields(
                    "external_hnsw_rs",
                    &params_json(cfg, ef_search),
                    build_time_s,
                    rss_kb,
                    &result,
                    &storage(index_bytes),
                    "\"construction_seed\":null,\"construction_seed_control\":\"unavailable\"",
                ),
            );
        } else {
            print_row(&format!("ef={ef_search}"), &result);
        }
    }
}

fn build_and_run<D>(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    distance: D,
) where
    D: hnsw_rs::prelude::Distance<f32> + Send + Sync,
{
    let build_start = Instant::now();
    let mut index =
        Hnsw::<f32, D>::new(cfg.m, train.len(), MAX_LAYER, cfg.ef_construction, distance);
    for (external_id, vector) in train.iter().enumerate() {
        index.insert((vector, external_id));
    }
    index.set_searching_mode(true);
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss_kb = current_rss_kb();
    let index_bytes = serialized_size(&index);

    if !cfg.json {
        println!(
            concat!(
                "--- external hnsw_rs {} (M={}, ef_construction={}, ",
                "sequential build, caller seed unavailable) ---"
            ),
            CRATE_VERSION, cfg.m, cfg.ef_construction
        );
        println!(
            "Build: {:.2}s ({:.0} vectors/sec)\n",
            build_time_s,
            train.len() as f64 / build_time_s
        );
        print_header();
    }

    emit_search_results(
        cfg,
        &index,
        test,
        neighbors,
        build_time_s,
        rss_kb,
        index_bytes,
    );

    if !cfg.json {
        println!();
    }
}

pub(crate) fn run_hnsw_rs(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
) {
    if cfg.is_euclidean {
        build_and_run(cfg, train, test, neighbors, DistL2 {});
    } else {
        build_and_run(cfg, train, test, neighbors, DistCosine {});
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_disclose_uncontrolled_rng_and_version() {
        let cfg = Config::default();
        let params = params_json(&cfg, 80);

        assert!(params.contains("\"crate_version\":\"0.3.4\""));
        assert!(params.contains("\"build_mode\":\"sequential\""));
        assert!(params.contains("\"seed_control\":\"unavailable\""));
        assert!(params.contains("\"rng_source\":\"os\""));
        assert!(params.contains("\"ef_search\":80"));
    }

    #[test]
    fn search_results_preserve_external_ids() {
        let train = [vec![0.0, 0.0], vec![10.0, 10.0], vec![1.0, 1.0]];
        let mut index = Hnsw::<f32, DistL2>::new(4, train.len(), 4, 20, DistL2 {});
        for (external_id, vector) in train.iter().enumerate() {
            index.insert((vector, external_id));
        }
        index.set_searching_mode(true);

        let found = results(index.search(&[0.9, 0.9], 1, 10));
        assert_eq!(found[0].0, 2);
    }
}
