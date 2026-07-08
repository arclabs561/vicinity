#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Search-only ACORN filtered HNSW benchmark.
//!
//! This target isolates the ACORN traversal loop from the selectivity example's
//! graph construction and exact-oracle work. It intentionally exercises the
//! public `get_neighbors -> Vec<u32>` callback path so clone/copy cost remains
//! visible in profiles.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use std::collections::HashSet;
use vicinity::distance::{cosine_distance_normalized, normalize};
use vicinity::hnsw::{acorn_search_with_stats, AcornConfig, FnFilter};

const N_VECTORS: usize = 3_000;
const DIM: usize = 32;
const N_QUERIES: usize = 128;
const K: usize = 10;
const NEIGHBORS: usize = 32;
const EF_SEARCH: usize = 800;
const TWO_HOP_NEIGHBORS: usize = 128;

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            normalize(&v)
        })
        .collect()
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

fn run_acorn_batch(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    entry_points: &[u32],
    graph: &[Vec<u32>],
    matching: &[bool],
    config: &AcornConfig,
) -> (usize, u64, u64) {
    let filter = FnFilter(|id: u32| matching[id as usize]);
    let mut result_count = 0usize;
    let mut two_hop_invocations = 0u64;
    let mut two_hop_nodes_examined = 0u64;

    for (query, &entry_point) in queries.iter().zip(entry_points.iter()) {
        let (results, stats) = acorn_search_with_stats(
            K,
            config,
            &filter,
            |id| graph[id as usize].as_slice(),
            |id| cosine_distance_normalized(query, &vectors[id as usize]),
            entry_point,
        )
        .expect("acorn search failed");
        result_count += results.len();
        two_hop_invocations += stats.two_hop_invocations;
        two_hop_nodes_examined += stats.two_hop_nodes_examined;
    }

    (result_count, two_hop_invocations, two_hop_nodes_examined)
}

fn print_acorn_summary(
    label: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    entry_points: &[u32],
    graph: &[Vec<u32>],
    matching: &[bool],
    config: &AcornConfig,
) {
    let (result_count, two_hop_invocations, two_hop_nodes_examined) =
        run_acorn_batch(vectors, queries, entry_points, graph, matching, config);
    let queries_len = queries.len().max(1) as f64;
    eprintln!(
        "acorn profile {label}: 2hop_calls={:.1}/query 2hop_nodes={:.1}/query results={result_count}",
        two_hop_invocations as f64 / queries_len,
        two_hop_nodes_examined as f64 / queries_len,
    );
    black_box(result_count);
}

fn bench_acorn_search_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("acorn_search_only");

    let vectors = random_vectors(N_VECTORS, DIM, 42);
    let queries = random_vectors(N_QUERIES, DIM, 123);
    let entry_points: Vec<u32> = queries
        .iter()
        .map(|query| nearest_entry_point(query, &vectors))
        .collect();
    let graph = build_mutual_knn_graph(&vectors, NEIGHBORS);
    let config = AcornConfig {
        enable_two_hop: true,
        two_hop_threshold: 0.3,
        max_two_hop_neighbors: TWO_HOP_NEIGHBORS,
        ef_search: EF_SEARCH,
    };

    group.throughput(Throughput::Elements(N_QUERIES as u64));
    for selectivity in [0.50, 0.10, 0.02] {
        let target_count = ((N_VECTORS as f64 * selectivity).round() as usize).clamp(K, N_VECTORS);
        let matching: Vec<bool> = (0..N_VECTORS).map(|id| id < target_count).collect();
        let label = format!("selectivity_{selectivity:.2}");
        print_acorn_summary(
            &label,
            &vectors,
            &queries,
            &entry_points,
            &graph,
            &matching,
            &config,
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &matching,
            |bench, matching| {
                bench.iter(|| {
                    let (result_count, two_hop_invocations, two_hop_nodes_examined) =
                        run_acorn_batch(
                            black_box(&vectors),
                            black_box(&queries),
                            black_box(&entry_points),
                            black_box(&graph),
                            black_box(matching),
                            black_box(&config),
                        );
                    black_box((result_count, two_hop_invocations, two_hop_nodes_examined));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_acorn_search_only);
criterion_main!(benches);
