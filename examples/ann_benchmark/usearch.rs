#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::time::Instant;

use crate::support::{
    current_rss_kb, emit_result, evaluate, json_line_with_storage, print_header, print_row, Config,
    ResultStorage,
};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

fn available_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

fn search(index: &Index, query: &[f32], count: usize) -> Vec<(u32, f32)> {
    let matches = index.search(query, count).expect("USearch search failed");
    matches
        .keys
        .into_iter()
        .zip(matches.distances)
        .map(|(key, distance)| {
            (
                u32::try_from(key)
                    .expect("USearch returned an ID outside the benchmark's u32 range"),
                distance,
            )
        })
        .collect()
}

fn append_serialized_bytes(mut line: String, serialized_bytes: usize) -> String {
    assert!(
        line.ends_with('}'),
        "benchmark JSON must end with an object"
    );
    line.pop();
    line.push_str(&format!(",\"serialized_bytes\":{serialized_bytes}}}"));
    line
}

/// Runs the published USearch implementation as a benchmark-only baseline.
///
/// USearch 2.x does not expose a construction-seed control in its Rust API, so
/// the emitted parameters explicitly record that the requested benchmark seed
/// cannot govern construction.
pub(crate) fn run_usearch(
    cfg: &Config,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    dim: usize,
) {
    let metric = if cfg.is_euclidean {
        MetricKind::L2sq
    } else {
        MetricKind::Cos
    };
    let options = IndexOptions {
        dimensions: dim,
        metric,
        quantization: ScalarKind::F32,
        connectivity: cfg.m,
        expansion_add: cfg.ef_construction,
        ..Default::default()
    };
    let index = Index::new(&options).expect("create USearch index");
    let threads = available_threads();

    if !cfg.json {
        println!(
            "--- USearch (M={}, expansion_add={}, metric={:?}, scalar=F32, reserved_threads={}) ---",
            cfg.m, cfg.ef_construction, metric, threads
        );
    }

    let build_start = Instant::now();
    index
        .reserve_capacity_and_threads(train.len(), threads)
        .expect("reserve USearch capacity and thread contexts");
    for (id, vector) in train.iter().enumerate() {
        index
            .add(id as u64, vector)
            .expect("add vector to USearch index");
    }
    let build_time_s = build_start.elapsed().as_secs_f64();
    let rss_kb = current_rss_kb();
    let memory_bytes = index.memory_usage();
    let serialized_bytes = index.serialized_length();

    if !cfg.json {
        println!(
            "Build: {:.2}s ({:.0} vectors/sec), memory: {} bytes, serialized: {} bytes\n",
            build_time_s,
            train.len() as f64 / build_time_s,
            memory_bytes,
            serialized_bytes
        );
        print_header();
    }

    for &expansion_search in &cfg.ef_search_values {
        index.change_expansion_search(expansion_search);
        let result = evaluate(
            &|query, count| search(&index, query, count),
            test,
            neighbors,
            10,
        );

        if cfg.json {
            let params = format!(
                "{{\"crate_version\":\"2.26.0\",\"connectivity\":{},\"expansion_add\":{},\"expansion_search\":{},\"metric\":\"{}\",\"scalar\":\"f32\",\"reserved_threads\":{},\"build_mode\":\"sequential_add\",\"query_mode\":\"sequential\",\"construction_seed\":null,\"construction_seed_control\":\"unavailable\"}}",
                cfg.m,
                cfg.ef_construction,
                expansion_search,
                if cfg.is_euclidean { "l2sq" } else { "cos" },
                threads,
            );
            let storage = ResultStorage {
                index_bytes: Some(memory_bytes as u64),
                index_bytes_kind: Some("heap_estimate"),
                ..ResultStorage::default()
            };
            let line = json_line_with_storage(
                "external_usearch",
                &params,
                build_time_s,
                rss_kb,
                &result,
                &storage,
            );
            emit_result(
                &cfg.results_path,
                &append_serialized_bytes(line, serialized_bytes),
            );
        } else {
            print_row(&format!("expansion={expansion_search}"), &result);
        }
    }

    if !cfg.json {
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_external_ids_with_f32_vectors() {
        let index = Index::new(&IndexOptions {
            dimensions: 2,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            ..Default::default()
        })
        .unwrap();
        index.reserve_capacity_and_threads(2, 1).unwrap();
        index.add(7, &[0.0_f32, 0.0]).unwrap();
        index.add(42, &[10.0_f32, 10.0]).unwrap();

        let result = search(&index, &[0.1, 0.1], 2);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 7);
        assert_eq!(result[1].0, 42);
        assert!(index.memory_usage() > 0);
        assert!(index.serialized_length() > 0);
    }

    #[test]
    fn serialized_bytes_is_a_top_level_result_field() {
        let line = append_serialized_bytes("{\"algorithm\":\"external_usearch\"}".to_owned(), 1234);
        assert_eq!(
            line,
            "{\"algorithm\":\"external_usearch\",\"serialized_bytes\":1234}"
        );
    }
}
