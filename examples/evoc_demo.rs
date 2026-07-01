//! EVōC: Hierarchical Clustering for Embeddings
//!
//! EVōC (Embedding Vector Oriented Clustering) combines:
//!   - Dimensionality reduction (PCA-like) to ~15D
//!   - MST-based hierarchical clustering (HDBSCAN-style)
//!   - Multi-granularity cluster extraction
//!
//! # Ecosystem Context
//!
//! | Crate     | Algorithm        | Use                           |
//! |-----------|------------------|-------------------------------|
//! | vicinity  | EVōC             | Embedding exploration, outliers |
//! | stratify  | K-means, GMM     | Known k, repeated runs        |
//! | stratify  | Leiden           | Graph community detection     |
//! | stratify  | Hierarchical     | Dendrograms, any-k extraction |
//! | hop       | RAPTOR           | RAG tree summarization        |
//!
//! **Decision Flow:**
//! 1. Don't know k? -> EVōC (explores structure) or stratify::HierarchicalClustering
//! 2. Know k, need speed? -> stratify::Kmeans or stratify::KmeansElkan
//! 3. Need soft assignments? -> stratify::Gmm
//! 4. Graph/community structure? -> stratify::Leiden
//! 5. RAG summaries? -> hop (uses stratify internally)
//!
//! ```bash
//! cargo run --example evoc_demo --release
//! ```

use vicinity::evoc::{EVoC, EVoCParams};

fn main() -> vicinity::Result<()> {
    println!("EVōC: Hierarchical Embedding Clustering");
    println!("========================================\n");

    println!("EVōC discovers cluster structure without specifying k.");
    println!("It builds a hierarchy you can cut at any granularity.\n");

    demo_basic_clustering()?;
    demo_hierarchy_exploration()?;
    demo_when_to_use()?;

    println!("Done!");
    Ok(())
}

fn demo_basic_clustering() -> vicinity::Result<()> {
    println!("1. Basic Clustering: Discover Natural Groups");
    println!("   ------------------------------------------\n");

    // Generate embeddings with clear cluster structure
    let dim = 128;
    let n_clusters = 5;
    let points_per_cluster = 40;
    let n_outliers = 10;

    let (data, labels) = generate_clustered_embeddings(dim, n_clusters, points_per_cluster);
    let n_normal = n_clusters * points_per_cluster;

    // Add outliers (isolated points)
    let mut full_data = data.clone();
    for i in 0..n_outliers {
        let outlier: Vec<f32> = (0..dim)
            .map(|j| {
                let seed = (n_normal + i) * dim + j;
                deterministic_gauss(seed as u64) * 3.0 // Far from clusters
            })
            .collect();
        let outlier = normalize(&outlier);
        full_data.push(outlier);
    }

    let n_total = n_normal + n_outliers;
    let flat_data: Vec<f32> = full_data.into_iter().flatten().collect();

    println!(
        "   Generated: {} cluster points + {} outliers in {}D\n",
        n_normal, n_outliers, dim
    );

    // Fit EVōC
    let params = EVoCParams {
        intermediate_dim: 15, // Project to 15D for clustering
        min_cluster_size: 5,  // Ignore tiny clusters
        noise_level: 0.0,
        min_number_clusters: None, // Let it discover
    };

    let mut evoc = EVoC::new(dim, params)?;
    let assignments = evoc.fit_predict(&flat_data, n_total)?;

    // Count clusters found
    let assigned: Vec<_> = assignments.iter().filter(|x| x.is_some()).collect();
    let noise: Vec<_> = assignments.iter().filter(|x| x.is_none()).collect();

    println!("   EVōC found:");
    println!("     Points assigned to clusters: {}", assigned.len());
    println!(
        "     Points marked as noise:     {} (expected: ~{})",
        noise.len(),
        n_outliers
    );

    // Check if outliers were correctly identified as noise
    let mut outlier_noise_count = 0;
    for assignment in assignments.iter().take(n_total).skip(n_normal) {
        if assignment.is_none() {
            outlier_noise_count += 1;
        }
    }
    println!(
        "     Outliers detected as noise:  {}/{}",
        outlier_noise_count, n_outliers
    );

    // Compare with ground truth
    let nmi = normalized_mutual_info(&assignments[..n_normal], &labels);
    println!("\n   Quality metrics:");
    println!("     NMI with ground truth: {:.3} (1.0 = perfect)", nmi);

    println!();
    Ok(())
}

fn demo_hierarchy_exploration() -> vicinity::Result<()> {
    println!("2. Hierarchy Exploration: Multiple Granularities");
    println!("   ----------------------------------------------\n");

    println!("   EVōC builds a dendrogram. Cut at different heights for different k.\n");

    // Generate data with nested structure
    let dim = 64;
    let (data, _) = generate_nested_clusters(dim);
    let n = data.len() / dim;
    let flat_data: Vec<f32> = data;

    let params = EVoCParams {
        intermediate_dim: 12,
        min_cluster_size: 3,
        noise_level: 0.0,
        min_number_clusters: None,
    };

    let mut evoc = EVoC::new(dim, params)?;
    let _ = evoc.fit_predict(&flat_data, n)?;

    // Extract layers at different granularities
    println!("   Cluster counts at different granularities:");
    println!(
        "   {:>12} {:>15} {:>15}",
        "Threshold", "Clusters", "Noise points"
    );
    println!("   {}", "-".repeat(45));

    // EVōC stores cluster layers internally
    let layers = evoc.cluster_layers();

    if layers.is_empty() {
        // If no layers, show what the hierarchy found
        println!("   (Hierarchy built, {} points processed)", n);
    } else {
        for layer in layers {
            println!(
                "   {:>12} {:>15} {:>15}",
                "-",
                layer.num_clusters,
                layer.assignments.iter().filter(|x| x.is_none()).count()
            );
        }
    }

    println!("\n   EVōC exposes several granularities from one hierarchy.");
    println!();
    Ok(())
}

fn demo_when_to_use() -> vicinity::Result<()> {
    println!("3. Clustering Decision Guide");
    println!("   -------------------------\n");

    println!(
        "   | Criterion           | EVōC (vicinity)  | K-means (stratify) | GMM (stratify)   |"
    );
    println!(
        "   |---------------------|------------------|--------------------|--------------------|"
    );
    println!(
        "   | Known cluster count | No               | Yes                | Yes                |"
    );
    println!(
        "   | Cluster shape       | Arbitrary        | Spherical          | Ellipsoidal        |"
    );
    println!(
        "   | Outlier handling    | Built-in         | None               | Soft (low prob)    |"
    );
    println!(
        "   | Hierarchy           | Yes              | No                 | No                 |"
    );
    println!(
        "   | Speed               | O(n^2 log n)     | O(n*k*iter)        | O(n*k^2*iter)      |"
    );
    println!(
        "   | Use                 | Exploration      | Repeated runs      | Soft assignments   |"
    );
    println!();

    println!("   Pipeline sketch:");
    println!("     1. EVōC (vicinity) to estimate a natural cluster count");
    println!("     2. stratify::Kmeans with discovered k for repeated runs");
    println!("     3. stratify::Gmm if you need probability per cluster");
    println!("     4. stratify::Leiden for graph community structure");
    println!();

    println!("   Cross-crate integration example:");
    println!("     ```rust,ignore");
    println!("     // Explore with EVōC");
    println!("     let evoc_labels = evoc.fit_predict(&embeddings, n)?;");
    println!("     let k = evoc_labels.iter().filter_map(|&x| x).max().unwrap_or(0) + 1;");
    println!("     ");
    println!("     // Reuse discovered k with stratify");
    println!("     use stratify::{{Kmeans, Clustering}};");
    println!("     let kmeans = Kmeans::new(k);");
    println!("     let labels = kmeans.fit_predict(&data)?;");
    println!("     ```");
    println!();

    println!("   See also:");
    println!("     - stratify/clustering_demo.rs: K-means, GMM, hierarchical comparison");
    println!("     - stratify/community_detection.rs: Leiden algorithm for graphs");
    println!("     - hop: RAPTOR tree building (uses stratify clustering internally)");

    Ok(())
}

// =============================================================================
// Data Generation
// =============================================================================

/// Generate embeddings with clear cluster structure.
fn generate_clustered_embeddings(
    dim: usize,
    n_clusters: usize,
    points_per_cluster: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for c in 0..n_clusters {
        // Cluster center
        let center: Vec<f32> = (0..dim)
            .map(|j| deterministic_gauss((c * dim + j) as u64))
            .collect();
        let center = normalize(&center);

        // Points around center
        for i in 0..points_per_cluster {
            let mut point: Vec<f32> = center
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let seed = (c * points_per_cluster + i) * dim + j;
                    v + deterministic_gauss(seed as u64) * 0.1 // Tight clusters
                })
                .collect();
            point = normalize(&point);
            data.push(point);
            labels.push(c);
        }
    }

    (data, labels)
}

/// Generate data with nested cluster structure (2 super-clusters, each with 3 sub-clusters).
fn generate_nested_clusters(dim: usize) -> (Vec<f32>, Vec<usize>) {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    let n_super = 2;
    let n_sub = 3;
    let points_per = 15;

    for super_c in 0..n_super {
        // Super-cluster center
        let super_center: Vec<f32> = (0..dim)
            .map(|j| deterministic_gauss((super_c * 1000 + j) as u64) * 2.0)
            .collect();

        for sub_c in 0..n_sub {
            // Sub-cluster offset
            let sub_offset: Vec<f32> = (0..dim)
                .map(|j| deterministic_gauss((super_c * 100 + sub_c * 10 + j) as u64) * 0.5)
                .collect();

            let center: Vec<f32> = super_center
                .iter()
                .zip(&sub_offset)
                .map(|(s, o)| s + o)
                .collect();
            let center = normalize(&center);

            for i in 0..points_per {
                let mut point: Vec<f32> = center
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        let seed =
                            (super_c * n_sub * points_per + sub_c * points_per + i) * dim + j;
                        v + deterministic_gauss(seed as u64) * 0.05
                    })
                    .collect();
                point = normalize(&point);
                data.extend(point);
                labels.push(super_c * n_sub + sub_c);
            }
        }
    }

    (data, labels)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn deterministic_gauss(seed: u64) -> f32 {
    // LCG for reproducibility
    let a: u64 = 6364136223846793005;
    let c: u64 = 1442695040888963407;
    let s1 = seed.wrapping_mul(a).wrapping_add(c);
    let s2 = s1.wrapping_mul(a).wrapping_add(c);

    let u1 = (s1 as f64 / u64::MAX as f64).max(1e-10);
    let u2 = s2 as f64 / u64::MAX as f64;

    let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    g as f32
}

/// Simplified NMI calculation.
fn normalized_mutual_info(pred: &[Option<usize>], truth: &[usize]) -> f64 {
    if pred.len() != truth.len() || pred.is_empty() {
        return 0.0;
    }

    // Convert predictions to usize (noise = separate cluster)
    let max_truth = *truth.iter().max().unwrap_or(&0);
    let pred_labels: Vec<usize> = pred.iter().map(|&p| p.unwrap_or(max_truth + 1)).collect();

    // Count contingency table
    use std::collections::HashMap;
    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut pred_counts: HashMap<usize, usize> = HashMap::new();
    let mut truth_counts: HashMap<usize, usize> = HashMap::new();

    for (&p, &t) in pred_labels.iter().zip(truth.iter()) {
        *contingency.entry((p, t)).or_insert(0) += 1;
        *pred_counts.entry(p).or_insert(0) += 1;
        *truth_counts.entry(t).or_insert(0) += 1;
    }

    let n = pred.len() as f64;

    // Calculate MI
    let mut mi = 0.0;
    for (&(p, t), &count) in &contingency {
        if count > 0 {
            let pxy = count as f64 / n;
            let px = pred_counts[&p] as f64 / n;
            let py = truth_counts[&t] as f64 / n;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }

    // Calculate entropies
    let h_pred: f64 = pred_counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    let h_truth: f64 = truth_counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    // NMI
    let denom = (h_pred + h_truth) / 2.0;
    if denom > 0.0 {
        mi / denom
    } else {
        0.0
    }
}
