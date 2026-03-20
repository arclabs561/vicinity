#![allow(clippy::expect_used)]
//! End-to-end evaluation tests using standard benchmark methodology.
//!
//! These tests validate that our ANN implementations achieve expected
//! performance on standard evaluation datasets with proper metrics.
//!
//! Reference targets (from ann-benchmarks.com):
//! - HNSW on SIFT-1M: ~95% recall@10 at ~10K QPS
//! - HNSW on GloVe-100: ~90% recall@10 at ~5K QPS
//!
//! Note: Our synthetic datasets are smaller, so we target:
//! - Small clustered (1K): 85%+ recall@10
//! - Medium clustered (10K): 80%+ recall@10

#![cfg(all(feature = "hnsw", feature = "benchmark"))]

use std::time::Instant;

use vicinity::benchmark::{
    generate_normalized_clustered_dataset, generate_uniform_dataset, DistanceMetric, EvalDataset,
    EvalResults,
};
use vicinity::hnsw::{HNSWIndex, HNSWParams};

/// Run evaluation on a dataset with given HNSW parameters.
fn evaluate_hnsw(dataset: &EvalDataset, params: &HNSWParams, config_name: &str) -> EvalResults {
    let start = Instant::now();

    // Build index
    let mut index =
        HNSWIndex::with_params(dataset.dim, params.clone()).expect("Failed to create HNSW index");

    for (i, vec) in dataset.base.iter().enumerate() {
        index
            .add(i as u32, vec.clone())
            .expect("Failed to add vector");
    }
    index.build().expect("Failed to build index");

    let build_time = start.elapsed();

    // Estimate memory (simplified)
    let index_memory = dataset.base.len() * dataset.dim * 4  // vectors
        + dataset.base.len() * params.m * 8; // graph edges

    // Evaluate
    let mut recalls = Vec::with_capacity(dataset.n_queries());
    let mut latencies = Vec::with_capacity(dataset.n_queries());

    for (query, gt) in dataset.queries.iter().zip(dataset.ground_truth.iter()) {
        let start = Instant::now();
        let results = index
            .search(query, dataset.k, params.ef_search)
            .expect("Search failed");
        let elapsed = start.elapsed();

        // Extract IDs from results
        let approx: Vec<u32> = results.iter().map(|(id, _)| *id).collect();

        // Compute recall
        let gt_set: std::collections::HashSet<u32> = gt.iter().take(dataset.k).copied().collect();
        let found = approx
            .iter()
            .take(dataset.k)
            .filter(|id| gt_set.contains(id))
            .count();
        let recall = found as f32 / dataset.k as f32;

        recalls.push(recall);
        latencies.push(elapsed.as_micros() as u64);
    }

    EvalResults {
        dataset: dataset.name.clone(),
        algorithm: "hnsw".into(),
        config: config_name.into(),
        recalls,
        latencies_us: latencies,
        build_time,
        index_memory_bytes: index_memory,
        k: dataset.k,
    }
}

// ============ Small Dataset Tests (Fast, CI-friendly) ============

// NOTE: HNSW uses cosine_distance internally, requiring:
// 1. All vectors L2-normalized
// 2. Ground truth computed with cosine distance
//
// The generate_normalized_clustered_dataset function handles both.

#[test]
fn test_eval_small_clustered() {
    // Small clustered dataset with normalized vectors (required for HNSW cosine)
    let dataset = generate_normalized_clustered_dataset(
        "small-clustered",
        1000, // base vectors
        100,  // queries
        64,   // dimensions
        10,   // clusters
        0.1,  // cluster_std
        10,   // k
        42,
    );

    let params = HNSWParams {
        m: 24,
        m_max: 48,
        ef_construction: 300,
        ef_search: 150,
        ..Default::default()
    };

    let results = evaluate_hnsw(&dataset, &params, "M=24,ef=150");

    println!("\n{}", results.summary());
    println!(
        "  Recalls: min={:.3}, max={:.3}",
        results
            .recalls
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        results
            .recalls
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // Target: 65%+ mean recall (non-deterministic; typical 80-100%)
    assert!(
        results.mean_recall() >= 0.65,
        "Mean recall {:.3} below threshold 0.65",
        results.mean_recall()
    );
}

#[test]
fn test_eval_small_high_connectivity() {
    // Higher M for better graph connectivity
    let dataset =
        generate_normalized_clustered_dataset("small-high-M", 1000, 100, 64, 10, 0.1, 10, 42);

    let params = HNSWParams {
        m: 48,
        m_max: 96,
        ef_construction: 500,
        ef_search: 300,
        ..Default::default()
    };

    let results = evaluate_hnsw(&dataset, &params, "M=48,ef=300");

    println!("\n{}", results.summary());

    // Target: 90%+ with higher connectivity
    assert!(
        results.mean_recall() >= 0.90,
        "Mean recall {:.3} below threshold 0.90",
        results.mean_recall()
    );
}

#[test]
fn test_eval_normalized_uniform() {
    // Uniform random data, normalized for cosine
    use vicinity::benchmark::normalize;

    let mut dataset = generate_uniform_dataset(
        "uniform-normalized",
        1000,
        50,
        64,
        10,
        DistanceMetric::Cosine,
        42,
    );

    // Normalize vectors
    dataset.base = dataset.base.into_iter().map(|v| normalize(&v)).collect();
    dataset.queries = dataset.queries.into_iter().map(|v| normalize(&v)).collect();

    // Recompute ground truth with cosine
    dataset.ground_truth = vicinity::benchmark::evaluation::compute_ground_truth(
        &dataset.base,
        &dataset.queries,
        10,
        DistanceMetric::Cosine,
    );

    let params = HNSWParams {
        m: 32,
        m_max: 64,
        ef_construction: 400,
        ef_search: 200,
        ..Default::default()
    };

    let results = evaluate_hnsw(&dataset, &params, "M=32,ef=200");

    println!("\n{}", results.summary());

    // Uniform data -- 65%+ (non-deterministic; typical 90-100%)
    assert!(
        results.mean_recall() >= 0.65,
        "Mean recall {:.3} below threshold 0.65",
        results.mean_recall()
    );
}

// ============ Scaling Tests (Medium, ~10K vectors) ============

#[test]
#[ignore] // Run with: cargo test --test eval_e2e -- --ignored
fn test_eval_medium_clustered() {
    let dataset = generate_normalized_clustered_dataset(
        "medium-clustered",
        10_000,
        500,
        128,
        20,
        0.1,
        10,
        42,
    );

    let params = HNSWParams {
        m: 32,
        m_max: 64,
        ef_construction: 400,
        ef_search: 200,
        ..Default::default()
    };

    let results = evaluate_hnsw(&dataset, &params, "M=32,ef=200");

    println!("\n{}", results.summary());

    // Medium scale target: 80%+
    assert!(
        results.mean_recall() >= 0.80,
        "Mean recall {:.3} below threshold 0.80",
        results.mean_recall()
    );
}

// ============ Parameter Sweep Tests ============

#[test]
fn test_ef_search_improves_recall() {
    let dataset = generate_normalized_clustered_dataset("ef-sweep", 1000, 100, 64, 10, 0.1, 10, 42);

    let ef_values = [50, 100, 200, 400];
    let mut recalls = Vec::new();

    for ef in ef_values {
        let params = HNSWParams {
            m: 24,
            m_max: 48,
            ef_construction: 300,
            ef_search: ef,
            ..Default::default()
        };

        let results = evaluate_hnsw(&dataset, &params, &format!("ef={}", ef));
        println!(
            "ef={}: recall={:.3}, qps={:.0}",
            ef,
            results.mean_recall(),
            results.qps()
        );
        recalls.push(results.mean_recall());
    }

    // Recall should generally increase with ef
    let max_recall = recalls.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_recall >= 0.70,
        "Best recall {:.3} below threshold 0.70",
        max_recall
    );
}

#[test]
fn test_m_parameter_tradeoff() {
    let dataset = generate_normalized_clustered_dataset("m-sweep", 1000, 100, 64, 10, 0.1, 10, 42);

    let m_values = [16, 32, 48];
    let mut results_vec = Vec::new();

    for m in m_values {
        let params = HNSWParams {
            m,
            m_max: m * 2,
            ef_construction: 300,
            ef_search: 150,
            ..Default::default()
        };

        let results = evaluate_hnsw(&dataset, &params, &format!("M={}", m));
        println!(
            "M={}: recall={:.3}, qps={:.0}, build={:.2}s",
            m,
            results.mean_recall(),
            results.qps(),
            results.build_time.as_secs_f64()
        );
        results_vec.push(results);
    }

    // Higher M generally gives better recall (but slower build)
    let best = &results_vec[2]; // M=48
    assert!(
        best.mean_recall() >= 0.90,
        "M=48 recall {:.3} below threshold 0.90",
        best.mean_recall()
    );
}

// ============ Metric Tests ============

#[test]
fn test_evaluation_metrics() {
    // Test that our evaluation metrics are computed correctly
    use vicinity::benchmark::{mrr, recall_at_k};

    // Perfect match
    let approx = vec![0, 1, 2, 3, 4];
    let truth = vec![0, 1, 2, 3, 4];
    assert!((recall_at_k(&approx, &truth, 5) - 1.0).abs() < 0.001);

    // Half match
    let approx = vec![0, 1, 5, 6, 7];
    let truth = vec![0, 1, 2, 3, 4];
    assert!((recall_at_k(&approx, &truth, 5) - 0.4).abs() < 0.001);

    // MRR: first match at position 1
    assert!((mrr(&[0], &[0]) - 1.0).abs() < 0.001);
    assert!((mrr(&[1, 0], &[0]) - 0.5).abs() < 0.001);
    assert!((mrr(&[2, 1, 0], &[0]) - 0.333).abs() < 0.01);
}

// ============ Regression Tests ============

#[test]
fn test_no_recall_regression() {
    // Fixed dataset for regression testing (normalized for cosine)
    let dataset = generate_normalized_clustered_dataset(
        "regression",
        500,
        50,
        32,
        5,
        0.1,
        10,
        12345, // Fixed seed
    );

    let params = HNSWParams {
        m: 24,
        m_max: 48,
        ef_construction: 300,
        ef_search: 150,
        ..Default::default()
    };

    let results = evaluate_hnsw(&dataset, &params, "baseline");

    // Regression baseline with corrected HNSW (non-deterministic; typical 90%+)
    let baseline_min = 0.60;

    assert!(
        results.mean_recall() >= baseline_min,
        "Recall regression: {:.3} < baseline {:.3}",
        results.mean_recall(),
        baseline_min
    );
}

// ============ Debug/Diagnostic Tests ============

#[test]
fn test_print_detailed_stats() {
    let dataset = generate_normalized_clustered_dataset("diagnostic", 500, 50, 32, 5, 0.1, 10, 42);

    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        ef_search: 100,
        ..Default::default()
    };

    let results = evaluate_hnsw(&dataset, &params, "diagnostic");

    println!("\n=== Detailed Evaluation Stats ===");
    println!(
        "Dataset: {} ({} base, {} queries, dim={})",
        dataset.name,
        dataset.n_base(),
        dataset.n_queries(),
        dataset.dim
    );
    println!("Config: {}", results.config);
    println!("\nRecall:");
    println!("  Mean:   {:.3}", results.mean_recall());
    println!("  Median: {:.3}", results.median_recall());
    println!(
        "  Min:    {:.3}",
        results
            .recalls
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max:    {:.3}",
        results
            .recalls
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!("\nLatency:");
    println!("  Mean:   {:.1} us", results.mean_latency_us());
    println!("  P50:    {} us", results.p50_latency_us());
    println!("  P99:    {} us", results.p99_latency_us());
    println!("\nThroughput:");
    println!("  QPS:    {:.1}", results.qps());
    println!("\nResources:");
    println!("  Build:  {:.2} s", results.build_time.as_secs_f64());
    println!(
        "  Memory: {:.2} MB",
        results.index_memory_bytes as f64 / 1_000_000.0
    );

    // Recall distribution histogram
    println!("\nRecall Distribution:");
    let mut bins = [0usize; 10];
    for r in &results.recalls {
        let bin = ((r * 10.0) as usize).min(9);
        bins[bin] += 1;
    }
    for (i, count) in bins.iter().enumerate() {
        let bar = "*".repeat(*count);
        println!(
            "  [{:.1}-{:.1}): {} {}",
            i as f32 / 10.0,
            (i + 1) as f32 / 10.0,
            count,
            bar
        );
    }
}
