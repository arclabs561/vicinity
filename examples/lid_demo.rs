#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Local Intrinsic Dimensionality (LID) Demonstration
//!
//! Shows LID estimation working on synthetic data with known ground truth.
//!
//! ```bash
//! cargo run --example lid_demo --release
//! ```
//!
//! Key demonstrations:
//! 1. LID correctly estimates dimension for d-dimensional spheres embedded in D dimensions
//! 2. TwoNN estimator gives similar results to MLE
//! 3. LID identifies outliers (points with unusually high LID)
//! 4. Confidence intervals provide uncertainty quantification

use vicinity::lid::{
    aggregate_lid, estimate_lid_batch, estimate_twonn, estimate_twonn_with_ci, LidAggregation,
    LidCategory, LidConfig, LidStats,
};

fn main() {
    println!("Local Intrinsic Dimensionality (LID) Demonstration");
    println!("===================================================\n");

    demo_synthetic_sphere();
    demo_twonn_vs_mle();
    demo_outlier_detection();
    demo_confidence_intervals();
}

/// Demo 1: Estimate ID of d-dimensional sphere embedded in D dimensions
fn demo_synthetic_sphere() {
    println!("1. Estimating intrinsic dimension of embedded manifolds");
    println!("   -----------------------------------------------\n");

    // Generate points in a 2D subspace embedded in 10D
    // True intrinsic dimension = 2
    let n = 1000; // More points for better estimation
    let true_id = 2;
    let ambient_dim = 10;

    println!(
        "   Generating {} points on {}-dim manifold in {}-dim space...",
        n, true_id, ambient_dim
    );

    let vectors = generate_sphere_points(n, true_id, ambient_dim, 0.001);
    let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

    // Debug: check distance distribution
    let sample_idx = 0;
    let query = &flat_vectors[sample_idx * ambient_dim..(sample_idx + 1) * ambient_dim];
    let mut dists: Vec<f32> = (1..n.min(50))
        .map(|j| {
            let other = &flat_vectors[j * ambient_dim..(j + 1) * ambient_dim];
            query
                .iter()
                .zip(other.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .collect();
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("\n   Sample distances from point 0 (first 10):");
    for (i, d) in dists.iter().take(10).enumerate() {
        println!("     d[{}] = {:.4}", i + 1, d);
    }

    // Estimate using MLE with appropriate k
    let config = LidConfig {
        k: 30, // Use more neighbors for stability
        epsilon: 1e-10,
    };
    let estimates = estimate_lid_batch(&flat_vectors, ambient_dim, &config);

    // Aggregate using different methods
    let mean_lid = aggregate_lid(&estimates, LidAggregation::Mean);
    let median_lid = aggregate_lid(&estimates, LidAggregation::Median);
    let harmonic_lid = aggregate_lid(&estimates, LidAggregation::HarmonicMean);

    println!("\n   MLE estimates (k={}):", config.k);
    println!("     Mean LID:     {:.2} (true: {})", mean_lid, true_id);
    println!("     Median LID:   {:.2} (true: {})", median_lid, true_id);
    println!("     Harmonic LID: {:.2} (true: {})", harmonic_lid, true_id);

    // Also try TwoNN
    let mu_ratios = compute_twonn_ratios(&flat_vectors, ambient_dim);
    let twonn_lid = estimate_twonn(&mu_ratios, 0.1);
    println!("\n   TwoNN estimate: {:.2} (true: {})", twonn_lid, true_id);

    // Check if estimate is reasonable (within 50% of true value)
    let error = ((median_lid - true_id as f32) / true_id as f32).abs();
    if error < 0.5 {
        println!("\n   [PASS] Median MLE estimate within 50% of true dimension");
    } else {
        println!(
            "\n   [NOTE] MLE estimate differs from true dimension by {:.0}%",
            error * 100.0
        );
    }

    let twonn_error = ((twonn_lid - true_id as f32) / true_id as f32).abs();
    if twonn_error < 0.5 {
        println!("   [PASS] TwoNN estimate within 50% of true dimension");
    } else {
        println!(
            "   [NOTE] TwoNN estimate differs from true dimension by {:.0}%",
            twonn_error * 100.0
        );
    }
    println!();
}

/// Demo 2: Compare TwoNN estimator with MLE
fn demo_twonn_vs_mle() {
    println!("2. Comparing TwoNN vs MLE estimators");
    println!("   ----------------------------------\n");

    let test_cases = [
        ("2D data in 10D", 2, 10, 300),
        ("5D data in 50D", 5, 50, 500),
        ("10D data in 100D", 10, 100, 1000),
    ];

    println!(
        "   {:25} {:>8} {:>10} {:>10}",
        "Dataset", "True ID", "MLE", "TwoNN"
    );
    println!("   {}", "-".repeat(55));

    for (name, true_id, ambient_dim, n) in test_cases {
        let vectors = generate_sphere_points(n, true_id, ambient_dim, 0.01);
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        // MLE estimate
        let config = LidConfig {
            k: 20,
            epsilon: 1e-10,
        };
        let estimates = estimate_lid_batch(&flat_vectors, ambient_dim, &config);
        let mle_lid = aggregate_lid(&estimates, LidAggregation::Median);

        // TwoNN estimate (need to compute mu ratios from 2-NN distances)
        let mu_ratios = compute_twonn_ratios(&flat_vectors, ambient_dim);
        let twonn_lid = estimate_twonn(&mu_ratios, 0.1);

        println!(
            "   {:25} {:>8} {:>10.2} {:>10.2}",
            name, true_id, mle_lid, twonn_lid
        );
    }
    println!();
}

/// Demo 3: Use LID for outlier detection
fn demo_outlier_detection() {
    println!("3. Outlier detection using LID");
    println!("   ---------------------------\n");

    // Generate 2D cluster with some outliers
    let n_normal = 100;
    let n_outliers = 5;
    let dim = 10;

    println!(
        "   Generating {} normal points + {} outliers in {}D...",
        n_normal, n_outliers, dim
    );

    let mut vectors = generate_sphere_points(n_normal, 2, dim, 0.01);

    // Add outliers (random high-dimensional points)
    for i in 0..n_outliers {
        let outlier: Vec<f32> = (0..dim)
            .map(|j| ((i * dim + j) as f32 * 0.7).sin() * 5.0)
            .collect();
        vectors.push(outlier);
    }

    let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

    // Estimate LID for all points
    let config = LidConfig {
        k: 15,
        epsilon: 1e-10,
    };
    let estimates = estimate_lid_batch(&flat_vectors, dim, &config);

    // Get statistics
    let stats = LidStats::from_estimates(&estimates);

    println!("\n   LID statistics:");
    println!("     Mean:   {:.2}", stats.mean);
    println!("     Median: {:.2}", stats.median);
    println!("     Std:    {:.2}", stats.std_dev);
    println!("     Min:    {:.2}", stats.min);
    println!("     Max:    {:.2}", stats.max);
    println!("     High threshold: {:.2}", stats.high_lid_threshold());

    // Count how many outliers are detected
    let mut detected_outliers = 0;
    for (i, estimate) in estimates.iter().enumerate() {
        let is_actual_outlier = i >= n_normal;
        let is_detected_high = stats.categorize(estimate.lid) == LidCategory::High;

        if is_actual_outlier && is_detected_high {
            detected_outliers += 1;
        }
    }

    println!(
        "\n   Outliers detected via high LID: {}/{}",
        detected_outliers, n_outliers
    );
    assert!(
        detected_outliers > 0,
        "LID failed to identify any of the {n_outliers} planted outliers via high LID"
    );
    println!("   [PASS] LID identified {detected_outliers}/{n_outliers} outliers via high LID");
    println!();
}

/// Demo 4: TwoNN with confidence intervals
fn demo_confidence_intervals() {
    println!("4. TwoNN with confidence intervals");
    println!("   --------------------------------\n");

    let test_cases = [
        ("Small sample (n=50)", 50),
        ("Medium sample (n=200)", 200),
        ("Large sample (n=1000)", 1000),
    ];

    let true_id = 3;
    let ambient_dim = 20;

    println!("   True ID = {}, ambient dim = {}\n", true_id, ambient_dim);
    println!(
        "   {:25} {:>10} {:>8} {:>18}",
        "Sample size", "Estimate", "SE", "95% CI"
    );
    println!("   {}", "-".repeat(65));

    for (name, n) in test_cases {
        let vectors = generate_sphere_points(n, true_id, ambient_dim, 0.01);
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        let mu_ratios = compute_twonn_ratios(&flat_vectors, ambient_dim);
        let result = estimate_twonn_with_ci(&mu_ratios, 0.1);

        println!(
            "   {:25} {:>10.2} {:>8.3} [{:>6.2}, {:>6.2}]",
            name, result.dimension, result.std_error, result.ci_lower, result.ci_upper
        );

        // Check if true value is within CI
        if result.ci_lower <= true_id as f32 && true_id as f32 <= result.ci_upper {
            println!("   {:25} (true value {} is within CI)", "", true_id);
        }
    }
    println!();
}

/// Generate points uniformly distributed in a d-dimensional subspace embedded in D dimensions.
///
/// This creates a proper d-dimensional manifold (a flat d-dim hyperplane) which should
/// have intrinsic dimension exactly d.
fn generate_sphere_points(
    n: usize,
    intrinsic_dim: usize,
    ambient_dim: usize,
    noise: f32,
) -> Vec<Vec<f32>> {
    let mut points = Vec::with_capacity(n);

    // Use golden ratio based quasi-random sequence for better uniformity
    let phi = 1.618_034_f32;

    for i in 0..n {
        let mut point = vec![0.0f32; ambient_dim];

        // Generate uniform coordinates in the first `intrinsic_dim` dimensions
        // Using quasi-random (low-discrepancy) sequence for better coverage
        for (j, p) in point.iter_mut().enumerate().take(intrinsic_dim) {
            // Quasi-random value in [0, 1] using generalized golden ratio
            let alpha = (1.0 / (j as f32 + 1.0 + phi)).fract();
            let val = ((i as f32 + 0.5) * alpha).fract();
            // Scale to [-1, 1]
            *p = (val * 2.0 - 1.0) * 1.0;
        }

        // Add small isotropic noise in ALL dimensions
        // This simulates measurement noise but shouldn't change intrinsic dimension
        for (j, p) in point.iter_mut().enumerate().take(ambient_dim) {
            let noise_val = noise
                * (((i * ambient_dim + j) as f32 * 0.414213).sin()
                    + ((i * ambient_dim + j) as f32 * 0.732051).cos())
                * 0.5;
            *p += noise_val;
        }

        points.push(point);
    }

    points
}

/// Compute TwoNN mu = r2/r1 ratios for all points
fn compute_twonn_ratios(flat_vectors: &[f32], dim: usize) -> Vec<f32> {
    let n = flat_vectors.len() / dim;
    let mut ratios = Vec::with_capacity(n);

    for i in 0..n {
        let query = &flat_vectors[i * dim..(i + 1) * dim];

        // Find 2 nearest neighbors (excluding self)
        let mut distances: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let other = &flat_vectors[j * dim..(j + 1) * dim];
                let dist_sq: f32 = query
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (j, dist_sq.sqrt())
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if distances.len() >= 2 {
            let r1 = distances[0].1;
            let r2 = distances[1].1;
            if r1 > 1e-10 {
                ratios.push(r2 / r1);
            }
        }
    }

    ratios
}
