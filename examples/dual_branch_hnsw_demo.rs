//! LID-Aware Graph Construction Demo
//!
//! Estimates Local Intrinsic Dimensionality (LID) on clustered data with
//! outliers and reports how the estimates separate point types.
//!
//! ```bash
//! cargo run --example dual_branch_hnsw_demo --release
//! ```

use vicinity::lid::{estimate_lid_batch, LidCategory, LidConfig, LidStats};

fn main() {
    println!("Local Intrinsic Dimensionality for ANN Index Quality");
    println!("=====================================================\n");

    println!("This example reports LID estimates for clustered points and outliers.\n");

    demo_lid_detection();
    demo_lid_for_index_quality();
    demo_research_directions();
}

fn demo_lid_detection() {
    println!("1. LID Identifies Outliers");
    println!("   -----------------------\n");

    let dim = 32;
    let n_clusters = 5;
    let points_per_cluster = 100;
    let n_outliers = 10;

    // Generate clustered data
    let mut data: Vec<f32> = Vec::new();
    for c in 0..n_clusters {
        let center: Vec<f32> = (0..dim)
            .map(|j| ((c * dim + j) as f32 * 0.618).sin() * 10.0)
            .collect();

        for i in 0..points_per_cluster {
            let point: Vec<f32> = center
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let noise = ((c * points_per_cluster + i) * dim + j) as f32 * 0.01;
                    v + noise.sin() * 0.5
                })
                .collect();
            data.extend(point);
        }
    }

    let n_normal = n_clusters * points_per_cluster;

    // Add outliers (far from clusters)
    for i in 0..n_outliers {
        let outlier: Vec<f32> = (0..dim)
            .map(|j| ((i * dim + j) as f32 * 0.7).sin() * 50.0)
            .collect();
        data.extend(outlier);
    }

    // Estimate LID
    let config = LidConfig {
        k: 15,
        epsilon: 1e-10,
    };
    let estimates = estimate_lid_batch(&data, dim, &config);
    let stats = LidStats::from_estimates(&estimates);

    println!(
        "   Generated {} cluster points + {} outliers in {}D\n",
        n_normal, n_outliers, dim
    );
    println!("   LID distribution:");
    println!("     Mean:      {:.2}", stats.mean);
    println!("     Median:    {:.2}", stats.median);
    println!("     Std Dev:   {:.2}", stats.std_dev);
    println!("     Range:     [{:.2}, {:.2}]", stats.min, stats.max);
    println!(
        "     Threshold: {:.2} (median + 1*std)",
        stats.high_lid_threshold()
    );

    // Count detection accuracy
    let threshold = stats.high_lid_threshold();
    let mut true_positives = 0;
    let mut false_positives = 0;

    for (i, est) in estimates.iter().enumerate() {
        let is_outlier = i >= n_normal;
        let detected_high = est.lid > threshold;

        if is_outlier && detected_high {
            true_positives += 1;
        } else if !is_outlier && detected_high {
            false_positives += 1;
        }
    }

    println!("\n   Detection accuracy:");
    println!(
        "     True outliers found: {}/{}",
        true_positives, n_outliers
    );
    println!("     False positives:     {}/{}", false_positives, n_normal);
    println!(
        "     Precision: {:.0}%",
        true_positives as f64 / (true_positives + false_positives).max(1) as f64 * 100.0
    );
    println!();
}

fn demo_lid_for_index_quality() {
    println!("2. LID Categories for ANN Diagnostics");
    println!("   -----------------------------------\n");

    println!("   High-LID points are associated with ANN failure modes:");
    println!("     - They have few true near neighbors");
    println!("     - Greedy search can miss them entirely");
    println!("     - Standard graph construction under-connects them\n");

    println!("   LID categorization:");
    println!("     Low LID    (<median-std): Dense region, easy to find");
    println!("     Normal LID (within 1std): Typical point");
    println!("     High LID   (>median+std): Sparse region, hard to find\n");

    // Generate example and show LID categories
    let dim = 64;
    let n = 500;
    let data: Vec<f32> = (0..n * dim)
        .map(|i| (i as f32 * 0.618).fract() * 2.0 - 1.0)
        .collect();

    let config = LidConfig {
        k: 20,
        epsilon: 1e-10,
    };
    let estimates = estimate_lid_batch(&data, dim, &config);
    let stats = LidStats::from_estimates(&estimates);

    let mut counts = [0usize; 3]; // low, medium, high
    for est in &estimates {
        match stats.categorize(est.lid) {
            LidCategory::Low => counts[0] += 1,
            LidCategory::Normal => counts[1] += 1,
            LidCategory::High => counts[2] += 1,
        }
    }

    println!("   Example ({} points, {}D):", n, dim);
    println!(
        "     Low LID:    {:>4} points ({:.0}%)",
        counts[0],
        counts[0] as f64 / n as f64 * 100.0
    );
    println!(
        "     Normal LID: {:>4} points ({:.0}%)",
        counts[1],
        counts[1] as f64 / n as f64 * 100.0
    );
    println!(
        "     High LID:   {:>4} points ({:.0}%)",
        counts[2],
        counts[2] as f64 / n as f64 * 100.0
    );
    println!();
}

fn demo_research_directions() {
    println!("3. Research Applications");
    println!("   ----------------------\n");

    println!("   Using LID to improve ANN indices:");
    println!();
    println!("   a) Adaptive neighbor count:");
    println!("      - Low-LID points:  m neighbors (standard)");
    println!("      - High-LID points: 1.5-2x m neighbors (more connectivity)");
    println!();
    println!("   b) Skip bridges (Dual-Branch HNSW):");
    println!("      - Connect high-LID points with long-range edges");
    println!("      - Helps search escape local minima");
    println!();
    println!("   c) Query-time adaptation:");
    println!("      - Estimate query LID on-the-fly");
    println!("      - Use higher ef_search for high-LID queries");
    println!();
    println!("   d) Index quality diagnostics:");
    println!("      - Track recall by LID category");
    println!("      - Identify systematic failure modes");
    println!();
    println!("   References:");
    println!("     - Amsaleg et al. (2015): LID estimation methods");
    println!("     - Houle (2017): Dimensionality, discriminability, and ANN");
    println!("     - Dual-Branch HNSW (2025): LID-aware graph construction");
    println!();
}
