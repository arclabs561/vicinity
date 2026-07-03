#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Dual-Branch HNSW Demo
//!
//! Measures LID on clustered data with outliers, then compares a
//! LID-aware HNSW variant with standard HNSW on the same queries.
//!
//! Based on "Dual-Branch HNSW with Skip Bridges" (arXiv 2501.13992, Jan 2025).
//!
//! ```bash
//! cargo run --example dual_branch_demo --release
//! ```

use std::collections::HashSet;
use std::time::Instant;
use vicinity::hnsw::dual_branch::{DualBranchConfig, DualBranchHNSW};
use vicinity::hnsw::HNSWIndex;
use vicinity::lid::{estimate_lid, LidConfig};
use vicinity::DistanceMetric;

fn main() -> vicinity::Result<()> {
    println!("Dual-Branch HNSW: LID diagnostics");
    println!("==================================\n");

    demo_outlier_problem()?;
    demo_lid_analysis();
    demo_comparison()?;

    println!("Done!");
    Ok(())
}

/// Demonstrate how outliers degrade standard HNSW recall.
fn demo_outlier_problem() -> vicinity::Result<()> {
    println!("1. Recall Split by Point Type");
    println!("   ---------------------------\n");

    let dim = 64;
    let n_clusters = 5;
    let points_per_cluster = 100;
    let n_outliers = 20;

    // Generate clustered data + outliers
    let (data, labels) =
        generate_clustered_with_outliers(dim, n_clusters, points_per_cluster, n_outliers);
    let n_total = data.len() / dim;

    println!(
        "   Dataset: {} vectors ({} clustered + {} outliers) in {}-D\n",
        n_total,
        n_clusters * points_per_cluster,
        n_outliers,
        dim
    );

    let k = 10;
    let ef = 20;

    // Show standard HNSW under both metrics, each scored against its own
    // metric's brute-force ground truth (mixing an index metric with a
    // different ground-truth metric would misreport recall). The outlier
    // deficit that motivates the LID-aware variant is metric-dependent, so
    // reporting one metric alone is misleading.
    println!("   Standard HNSW recall@{k}, by query type and metric:\n");
    println!(
        "   {:>8}  {:>11}  {:>11}  {:>7}",
        "Metric", "Clustered", "Outlier", "Gap"
    );
    println!("   {:->8}  {:->11}  {:->11}  {:->7}", "", "", "", "");
    for metric in [DistanceMetric::Cosine, DistanceMetric::L2] {
        let (avg_clustered, avg_outlier) =
            standard_recall_split(&data, &labels, dim, metric, k, ef)?;
        println!(
            "   {:>8}  {:>10.1}%  {:>10.1}%  {:>6.1}%",
            metric_name(metric),
            avg_clustered * 100.0,
            avg_outlier * 100.0,
            (avg_clustered - avg_outlier) * 100.0
        );
    }
    println!();

    println!("   This split is diagnostic, not a guaranteed failure case. In this");
    println!("   deterministic synthetic run, standard HNSW does not show a large");
    println!("   outlier recall deficit once each metric is scored against its own");
    println!("   brute-force ground truth. The LID-aware variant below should be read");
    println!("   as a mechanism check, not as a recall improvement claim.\n");

    Ok(())
}

/// Build a standard HNSW under `metric`, then return
/// `(mean clustered recall, mean outlier recall)` scored against that same
/// metric's brute-force ground truth.
fn standard_recall_split(
    data: &[f32],
    labels: &[usize],
    dim: usize,
    metric: DistanceMetric,
    k: usize,
    ef: usize,
) -> vicinity::Result<(f32, f32)> {
    let n_total = data.len() / dim;
    // Cosine/angular HNSW requires L2-normalized vectors; auto_normalize does it
    // at add and query time. Cosine ground truth is scale-invariant, so it stays
    // consistent with the raw data.
    let normalize = matches!(metric, DistanceMetric::Cosine | DistanceMetric::Angular);
    let mut index = HNSWIndex::builder(dim)
        .m(16)
        .m_max(32)
        .metric(metric)
        .auto_normalize(normalize)
        .build()?;
    for i in 0..n_total {
        index.add(i as u32, data[i * dim..(i + 1) * dim].to_vec())?;
    }
    index.build()?;

    let mut clustered = Vec::new();
    let mut outlier = Vec::new();
    for i in 0..n_total {
        let query = &data[i * dim..(i + 1) * dim];
        let result_ids: HashSet<u32> = index
            .search(query, k, ef)?
            .iter()
            .map(|(id, _)| *id)
            .collect();
        let gt_ids: HashSet<u32> = brute_force_knn_metric(data, dim, query, k, metric)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        let recall = result_ids.intersection(&gt_ids).count() as f32 / k as f32;
        if labels[i] == 999 {
            outlier.push(recall);
        } else {
            clustered.push(recall);
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    Ok((mean(&clustered), mean(&outlier)))
}

fn metric_name(metric: DistanceMetric) -> &'static str {
    match metric {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::L2 => "L2",
        DistanceMetric::Angular => "angular",
        _ => "other",
    }
}

/// Demonstrate LID analysis of the dataset.
fn demo_lid_analysis() {
    println!("2. Local Intrinsic Dimensionality Analysis");
    println!("   ----------------------------------------\n");

    let dim = 64;
    let n_clusters = 5;
    let points_per_cluster = 100;
    let n_outliers = 20;

    let (data, labels) =
        generate_clustered_with_outliers(dim, n_clusters, points_per_cluster, n_outliers);
    let n_total = data.len() / dim;

    // Compute LID for each point
    let config = LidConfig {
        k: 20,
        ..Default::default()
    };
    let mut lid_estimates = Vec::new();

    for i in 0..n_total {
        let query = &data[i * dim..(i + 1) * dim];
        let dists = compute_distances_from(query, &data, dim, i);
        let estimate = estimate_lid(&dists, &config);
        lid_estimates.push((i, estimate.lid, labels[i]));
    }

    // Separate by type
    let clustered_lids: Vec<f32> = lid_estimates
        .iter()
        .filter(|(_, _, label)| *label != 999)
        .map(|(_, lid, _)| *lid)
        .collect();

    let outlier_lids: Vec<f32> = lid_estimates
        .iter()
        .filter(|(_, _, label)| *label == 999)
        .map(|(_, lid, _)| *lid)
        .collect();

    let avg_clustered_lid = clustered_lids.iter().sum::<f32>() / clustered_lids.len() as f32;
    let avg_outlier_lid = outlier_lids.iter().sum::<f32>() / outlier_lids.len() as f32;

    println!("   LID Statistics by Point Type:\n");
    println!("                   Mean LID    Min       Max");
    println!("   ------------------------------------------------");
    println!(
        "   Clustered:      {:>6.2}      {:>6.2}    {:>6.2}",
        avg_clustered_lid,
        clustered_lids.iter().cloned().fold(f32::INFINITY, f32::min),
        clustered_lids
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "   Outliers:       {:>6.2}      {:>6.2}    {:>6.2}",
        avg_outlier_lid,
        outlier_lids.iter().cloned().fold(f32::INFINITY, f32::min),
        outlier_lids
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!();

    println!("   LID interpretation for this synthetic dataset:");
    println!("   - Outliers sit in sparse regions where distance growth is irregular");
    println!("   - Fewer equidistant neighbors = higher MLE estimate");
    println!("   - LID separates most outliers in this dataset\n");

    // Show top high-LID points
    let mut sorted = lid_estimates.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("   Top 10 highest-LID points:");
    println!("   {:>6}  {:>8}  {:>10}", "Index", "LID", "Type");
    println!("   {:->6}  {:->8}  {:->10}", "", "", "");
    for (idx, lid, label) in sorted.iter().take(10) {
        let ptype = if *label == 999 {
            "OUTLIER"
        } else {
            "clustered"
        };
        println!("   {:>6}  {:>8.2}  {:>10}", idx, lid, ptype);
    }
    println!();
}

/// Compare Dual-Branch HNSW vs Standard HNSW.
fn demo_comparison() -> vicinity::Result<()> {
    println!("3. Dual-Branch HNSW vs Standard HNSW");
    println!("   ----------------------------------\n");

    let dim = 64;
    let n_clusters = 5;
    let points_per_cluster = 100;
    let n_outliers = 20;

    let (data, labels) =
        generate_clustered_with_outliers(dim, n_clusters, points_per_cluster, n_outliers);
    let n_total = data.len() / dim;

    // Build Dual-Branch HNSW
    let config = DualBranchConfig {
        m: 16,
        m_high_lid: 24, // 1.5x connections for high-LID points
        ef_construction: 200,
        ef_search: 50,
        lid_k: 20,
        lid_threshold_sigma: 1.5,
        skip_bridge_probability: 0.1,
        max_skip_length: 3,
        seed: Some(42),
    };

    let build_start = Instant::now();
    let mut dual_index = DualBranchHNSW::new(dim, config);
    dual_index.add_vectors(&data)?;
    dual_index.build()?;
    let dual_build_time = build_start.elapsed();

    // Build Standard HNSW
    let build_start = Instant::now();
    let mut std_index = HNSWIndex::builder(dim)
        .m(16)
        .m_max(32)
        .metric(DistanceMetric::L2)
        .build()?;
    for i in 0..n_total {
        let vec = data[i * dim..(i + 1) * dim].to_vec();
        std_index.add(i as u32, vec)?;
    }
    std_index.build()?;
    let std_build_time = build_start.elapsed();

    // Get Dual-Branch statistics
    let stats = dual_index.stats();

    println!("   Build Statistics:");
    println!("   {:>20}  {:>12}  {:>12}", "", "Standard", "Dual-Branch");
    println!("   {:->20}  {:->12}  {:->12}", "", "", "");
    println!(
        "   {:>20}  {:>12?}  {:>12?}",
        "Build time:", std_build_time, dual_build_time
    );
    println!(
        "   {:>20}  {:>12}  {:>12}",
        "Skip bridges:", "-", stats.num_skip_bridges
    );
    println!(
        "   {:>20}  {:>12}  {:>12}",
        "High-LID nodes:", "-", stats.high_lid_nodes
    );
    println!();

    // Compare recall on different point types
    let k = 10;
    let ef = 20;

    let mut std_clustered = Vec::new();
    let mut std_outlier = Vec::new();
    let mut dual_clustered = Vec::new();
    let mut dual_outlier = Vec::new();

    for i in 0..n_total {
        let query = &data[i * dim..(i + 1) * dim];
        let gt = brute_force_knn(&data, dim, query, k);
        let gt_ids: HashSet<u32> = gt.iter().map(|(id, _)| *id).collect();

        // Standard HNSW
        let std_results = std_index.search(query, k, ef)?;
        let std_ids: HashSet<u32> = std_results.iter().map(|(id, _)| *id).collect();
        let std_recall = std_ids.intersection(&gt_ids).count() as f32 / k as f32;

        // Dual-Branch HNSW
        let dual_results = dual_index.search(query, k)?;
        let dual_ids: HashSet<u32> = dual_results.iter().map(|(id, _)| *id).collect();
        let dual_recall = dual_ids.intersection(&gt_ids).count() as f32 / k as f32;

        if labels[i] == 999 {
            std_outlier.push(std_recall);
            dual_outlier.push(dual_recall);
        } else {
            std_clustered.push(std_recall);
            dual_clustered.push(dual_recall);
        }
    }

    let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32 * 100.0;

    println!("   Recall@{} Comparison:", k);
    println!(
        "   {:>20}  {:>12}  {:>12}  {:>10}",
        "Query Type", "Standard", "Dual-Branch", "Delta"
    );
    println!("   {:->20}  {:->12}  {:->12}  {:->10}", "", "", "", "");
    println!(
        "   {:>20}  {:>11.1}%  {:>11.1}%  {:>+9.1}%",
        "Clustered",
        avg(&std_clustered),
        avg(&dual_clustered),
        avg(&dual_clustered) - avg(&std_clustered)
    );
    println!(
        "   {:>20}  {:>11.1}%  {:>11.1}%  {:>+9.1}%",
        "Outliers",
        avg(&std_outlier),
        avg(&dual_outlier),
        avg(&dual_outlier) - avg(&std_outlier)
    );
    println!();

    let std_overall = (std_clustered.iter().sum::<f32>() + std_outlier.iter().sum::<f32>())
        / (std_clustered.len() + std_outlier.len()) as f32
        * 100.0;
    let dual_overall = (dual_clustered.iter().sum::<f32>() + dual_outlier.iter().sum::<f32>())
        / (dual_clustered.len() + dual_outlier.len()) as f32
        * 100.0;

    println!(
        "   Overall Recall: Standard={:.1}%, Dual-Branch={:.1}%\n",
        std_overall, dual_overall
    );

    println!("   Dual-Branch mechanisms under test:");
    println!("   - High-LID points receive more connections (M=24 vs M=16)");
    println!("   - Skip bridges add long-range edges from selected high-LID nodes");
    println!("   - Search can explore local neighbors and skip paths");
    println!();

    Ok(())
}

// --- Data Generation ---

fn generate_clustered_with_outliers(
    dim: usize,
    n_clusters: usize,
    points_per_cluster: usize,
    n_outliers: usize,
) -> (Vec<f32>, Vec<usize>) {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    // Generate clusters
    for c in 0..n_clusters {
        // Cluster center
        let center: Vec<f32> = (0..dim)
            .map(|d| {
                let seed = (c * dim + d) as f32;
                (seed * 0.618_034).fract() * 10.0 - 5.0
            })
            .collect();

        // Points around center
        for p in 0..points_per_cluster {
            for (d, &c_val) in center.iter().enumerate().take(dim) {
                let noise = ((c * points_per_cluster * dim + p * dim + d) as f32 * 0.1).sin() * 0.5;
                data.push(c_val + noise);
            }
            labels.push(c);
        }
    }

    // Generate outliers (far from clusters)
    for o in 0..n_outliers {
        for d in 0..dim {
            let seed = (1000000 + o * dim + d) as f32;
            let val = (seed * 0.618_034).fract() * 40.0 - 20.0; // Wider range
            data.push(val);
        }
        labels.push(999); // Special label for outliers
    }

    (data, labels)
}

fn brute_force_knn(data: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    brute_force_knn_metric(data, dim, query, k, DistanceMetric::L2)
}

/// Brute-force k-NN under a specific metric, so recall is scored against the
/// same metric the index uses.
fn brute_force_knn_metric(
    data: &[f32],
    dim: usize,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(u32, f32)> {
    let n = data.len() / dim;
    let mut distances: Vec<(u32, f32)> = (0..n)
        .map(|i| {
            let vec = &data[i * dim..(i + 1) * dim];
            (i as u32, metric_distance(query, vec, metric))
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).collect()
}

fn metric_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine | DistanceMetric::Angular => cosine_distance(a, b),
        _ => l2_distance(a, b),
    }
}

/// Cosine distance `1 - cos(a, b)`; zero-norm vectors are maximally distant.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

fn compute_distances_from(query: &[f32], data: &[f32], dim: usize, skip_idx: usize) -> Vec<f32> {
    let n = data.len() / dim;
    let mut dists: Vec<f32> = (0..n)
        .filter(|&i| i != skip_idx)
        .map(|i| {
            let vec = &data[i * dim..(i + 1) * dim];
            l2_distance(query, vec)
        })
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dists
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
