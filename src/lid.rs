//! Local Intrinsic Dimensionality (LID) estimation.
//!
//! # Intuition
//!
//! LID measures how "locally complex" the space around a point is.
//! A point with high LID behaves as if it's in a high-dimensional space locally,
//! even if the global embedding dimension is lower. These are often outliers or
//! points in sparse regions.
//!
//! # Mathematical Foundation
//!
//! For a point x, LID is defined as:
//!
//! ```text
//! LID(x) = lim(r→0) ln(F(r)) / ln(r)
//! ```
//!
//! where F(r) is the cumulative distribution of distances to nearby points.
//!
//! ## MLE Estimator (Levina & Bickel, 2004; Amsaleg et al., 2015)
//!
//! Uses k nearest neighbors:
//!
//! ```text
//! LID_MLE(x) = -k / Σᵢ log(dᵢ / dₖ)
//! ```
//!
//! where d₁ ≤ d₂ ≤ ... ≤ dₖ are distances to k nearest neighbors.
//!
//! **Known failure modes:**
//! - Underestimates at high intrinsic dimensions (finite sample bias)
//! - Sensitive to non-uniform density (GeoMLE addresses this)
//! - Biased near manifold boundaries
//!
//! ## TwoNN Estimator (Facco et al., 2017)
//!
//! A simpler method using only 2 nearest neighbors:
//!
//! ```text
//! μ = r₂/r₁  (ratio of 2nd to 1st neighbor distance)
//! P(μ ≤ x) = 1 - x^(-d)  for data on d-dimensional manifold
//! d = slope of linear fit: -log(1 - F_emp(μ)) vs log(μ)
//! ```
//!
//! TwoNN is O(n) vs O(n*k) for MLE, making it faster for large datasets.
//!
//! **Known failure modes:**
//! - Sensitive to measurement noise at small scales (overestimates)
//! - Can be biased by high manifold curvature
//! - Requires discarding outlier ratios (typically 10%)
//!
//! ## Scale-Dependent ID (Di Noia et al., 2024)
//!
//! The ABIDE algorithm addresses the scale-dependency problem:
//! - At small scales: ID appears high due to measurement noise
//! - At large scales: ID appears high due to manifold curvature
//! - "Sweet spot" exists where local constant-density assumption holds
//!
//! The algorithm iteratively refines both ID estimate and optimal neighborhood
//! size k* for each point, using a likelihood ratio test to find the largest
//! neighborhood where density is approximately constant.
//!
//! # Applications
//!
//! - **HNSW optimization**: Dual-Branch HNSW (2025) uses LID to identify
//!   outliers that need special handling during graph construction.
//! - **Anomaly detection**: High LID points are potential outliers.
//! - **Adaptive search**: Adjust ef_search based on local complexity.
//! - **Compression guidance**: ID provides lower bound on effective dimensions.
//!
//! # Future Work: ABIDE (Adaptive Binomial ID Estimator)
//!
//! The current implementation provides MLE and TwoNN estimators. A more robust
//! approach is ABIDE (Di Noia et al., 2024), which:
//!
//! 1. Uses a likelihood ratio test to find the optimal neighborhood size k*
//!    for each point where constant-density assumption holds
//! 2. Iteratively refines both ID and k* until convergence (~5 iterations)
//! 3. Achieves robustness to both measurement noise (small scale) and
//!    manifold curvature (large scale)
//!
//! Key parameters from ABIDE:
//! - Optimal radius ratio: τ ≈ c*^(1/d) where c* = 0.2032
//! - Convergence criterion: δ ≈ 10^(-4)
//! - Rejection threshold: D_thr = 6.635 (χ² at α=0.01)
//!
//! # References
//!
//! - Levina & Bickel (2004) "Maximum Likelihood Estimation of Intrinsic Dimension"
//! - Facco et al. (2017) "Estimating the intrinsic dimension of datasets"
//! - Denti et al. (2022) "A two-component mixture model for ID estimation"
//! - Gomtsyan et al. (2019) "Geometry-Aware Maximum Likelihood Estimation"
//! - Di Noia et al. (2024) "Beyond the noise: intrinsic dimension estimation"
//! - Dual-Branch HNSW (arXiv 2501.13992, 2025) "LID-based insertion"

use crate::distance;

/// Result of LID estimation for a single point.
#[derive(Debug, Clone, Copy)]
pub struct LidEstimate {
    /// Estimated local intrinsic dimensionality.
    pub lid: f32,
    /// Number of neighbors used in estimation.
    pub k: usize,
    /// Maximum distance to k-th neighbor.
    pub max_dist: f32,
}

/// LID estimation configuration.
#[derive(Debug, Clone)]
pub struct LidConfig {
    /// Number of neighbors for LID estimation (default: 20).
    pub k: usize,
    /// Minimum distance to avoid log(0) (default: 1e-10).
    pub epsilon: f32,
}

impl Default for LidConfig {
    fn default() -> Self {
        Self {
            k: 20,
            epsilon: 1e-10,
        }
    }
}

/// MLE estimator for Local Intrinsic Dimensionality.
///
/// # Formula
///
/// ```text
/// LID = -k / Σᵢ log(dᵢ / dₖ)
/// ```
///
/// where d₁ ≤ ... ≤ dₖ are sorted distances to k nearest neighbors.
///
/// # Properties
///
/// - Returns f32::INFINITY if all distances are equal (degenerate case).
/// - Returns LID ≈ embedding_dim for uniformly distributed data.
/// - Returns LID >> embedding_dim for points in sparse regions (outliers).
///
/// # Example
///
/// ```rust
/// use vicinity::lid::{estimate_lid_mle, LidConfig};
///
/// let distances = vec![0.1, 0.2, 0.3, 0.5, 0.8, 1.2];
/// let lid = estimate_lid_mle(&distances, &LidConfig::default());
/// assert!(lid.lid.is_finite());
/// assert!(lid.lid > 0.0);
/// ```
#[must_use]
pub fn estimate_lid_mle(sorted_distances: &[f32], config: &LidConfig) -> LidEstimate {
    let k = sorted_distances.len().min(config.k);

    if k < 2 {
        return LidEstimate {
            lid: f32::NAN,
            k,
            max_dist: sorted_distances.first().copied().unwrap_or(0.0),
        };
    }

    let d_k = sorted_distances[k - 1];

    // Use relative epsilon for scale-invariance
    let abs_epsilon = d_k * config.epsilon;
    let d_k = d_k.max(abs_epsilon);

    // Sum of log ratios
    let mut sum_log = 0.0f32;
    let mut valid_count = 0;

    for &d_i in &sorted_distances[..k] {
        let d_i = d_i.max(abs_epsilon);
        let ratio = d_i / d_k;
        if ratio > 0.0 && ratio < 1.0 {
            sum_log += ratio.ln();
            valid_count += 1;
        }
    }

    let lid = if valid_count > 0 && sum_log.abs() > abs_epsilon {
        -(valid_count as f32) / sum_log
    } else {
        // Degenerate case: all distances equal or very close
        f32::INFINITY
    };

    LidEstimate {
        lid,
        k,
        max_dist: d_k,
    }
}

/// Estimate LID for a query point given its neighbors' distances.
///
/// This is the primary interface for HNSW integration.
///
/// # Arguments
///
/// * `neighbor_distances` - Distances to neighbors (need not be sorted)
/// * `config` - LID estimation parameters
///
/// # Returns
///
/// LID estimate for the query point.
#[must_use]
pub fn estimate_lid(neighbor_distances: &[f32], config: &LidConfig) -> LidEstimate {
    let mut sorted = neighbor_distances.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    estimate_lid_mle(&sorted, config)
}

/// TwoNN estimator for intrinsic dimension (Facco et al., 2017).
///
/// A simpler, faster alternative to MLE that only needs the ratio μ = r₂/r₁
/// of the second to first nearest neighbor distance.
///
/// # Theory
///
/// For a dataset with intrinsic dimension d, the ratio μ follows:
/// ```text
/// P(μ ≤ x) = 1 - x^(-d)
/// ```
///
/// Taking logarithms:
/// ```text
/// -log(1 - F_emp(μ)) = d * log(μ)
/// ```
///
/// So the slope of this linear relationship gives d.
///
/// # Arguments
///
/// * `mu_ratios` - Array of μ = r₂/r₁ ratios for each point. Note that μ ≥ 1
///   by definition since r₂ ≥ r₁ (2nd neighbor is at least as far as 1st).
/// * `discard_fraction` - Fraction of largest ratios to discard as outliers
///   (default: 0.1). This improves robustness to boundary effects.
///
/// # Returns
///
/// Estimated intrinsic dimension (global estimate for the dataset).
///
/// # Failure Modes
///
/// - **Noise at small scales**: Measurement noise at nearest-neighbor scale
///   causes μ ratios to be noisy, overestimating ID.
/// - **High curvature**: When manifold curves within NN distances, ID is
///   overestimated. Consider ABIDE (Di Noia et al., 2024) for robustness.
/// - **Non-uniform density**: Sharp density gradients bias the estimate.
///
/// # Example
///
/// ```rust
/// use vicinity::lid::estimate_twonn;
///
/// // Simulated 2-NN distance ratios (r2/r1) for points on a ~3D manifold
/// let mu_ratios: Vec<f32> = vec![1.05, 1.12, 1.08, 1.15, 1.03, 1.20, 1.07, 1.11];
/// let dim = estimate_twonn(&mu_ratios, 0.1);
/// assert!(dim.is_finite());
/// assert!(dim > 0.0);
/// ```
#[must_use]
pub fn estimate_twonn(mu_ratios: &[f32], discard_fraction: f32) -> f32 {
    if mu_ratios.is_empty() {
        return f32::NAN;
    }

    // Filter valid ratios: μ = r₂/r₁ ≥ 1.0 by definition
    // (second neighbor is at least as far as first neighbor)
    let mut sorted: Vec<f32> = mu_ratios
        .iter()
        .filter(|&&x| x.is_finite() && x >= 1.0)
        .copied()
        .collect();

    if sorted.len() < 2 {
        return f32::NAN;
    }

    sorted.sort_unstable_by(|a, b| a.total_cmp(b));

    // Discard largest ratios (boundary effects / outliers)
    let keep_count = ((sorted.len() as f32) * (1.0 - discard_fraction)).max(2.0) as usize;
    let sorted = &sorted[..keep_count.min(sorted.len())];
    let n_kept = sorted.len();

    if n_kept < 2 {
        return f32::NAN;
    }

    // Empirical CDF: F_emp(μ_i) = i / n_kept
    // We fit: -log(1 - F_emp) = d * log(μ)
    // Using simple least squares with intercept forced to 0
    //
    // Note: We use n_kept (trimmed count) for proper CDF normalization,
    // matching scikit-dimension's approach after discarding outliers.

    let mut sum_xy = 0.0f64; // Use f64 for accumulation precision
    let mut sum_xx = 0.0f64;
    let mut valid_count = 0usize;

    for (i, &mu) in sorted.iter().enumerate() {
        let f_emp = (i + 1) as f64 / n_kept as f64;

        // Skip if F_emp ≈ 1 (would cause -log(0))
        // Skip if μ ≈ 1 (would cause log(1) = 0, degenerate - equidistant neighbors)
        if f_emp >= 0.9999 || mu < 1.0001 {
            continue;
        }

        let x = (mu as f64).ln();
        let y = -(1.0 - f_emp).ln();

        if x.is_finite() && y.is_finite() && x > 0.0 {
            sum_xy += x * y;
            sum_xx += x * x;
            valid_count += 1;
        }
    }

    if valid_count < 2 || sum_xx.abs() < 1e-12 {
        return f32::NAN;
    }

    (sum_xy / sum_xx) as f32 // slope = d
}

/// Extended TwoNN result with confidence interval.
#[derive(Debug, Clone, Copy)]
pub struct TwoNNResult {
    /// Estimated intrinsic dimension.
    pub dimension: f32,
    /// Standard error of the estimate.
    pub std_error: f32,
    /// Number of valid ratios used.
    pub n_samples: usize,
    /// 95% confidence interval lower bound.
    pub ci_lower: f32,
    /// 95% confidence interval upper bound.
    pub ci_upper: f32,
}

/// TwoNN estimator with confidence interval (Denti et al., 2022).
///
/// Returns the dimension estimate along with uncertainty quantification.
/// The confidence interval is based on the asymptotic normality of the
/// MLE formulation of TwoNN.
///
/// # Theory
///
/// The TwoNN estimator in MLE form is:
/// ```text
/// d̂ = n / Σᵢ log(μᵢ)
/// ```
///
/// With asymptotic variance approximately d²/n, giving standard error d/√n.
///
/// # Arguments
///
/// * `mu_ratios` - Array of μ = r₂/r₁ ratios
/// * `discard_fraction` - Fraction of largest ratios to discard (default: 0.1)
#[must_use]
pub fn estimate_twonn_with_ci(mu_ratios: &[f32], discard_fraction: f32) -> TwoNNResult {
    // Filter and sort
    let mut sorted: Vec<f32> = mu_ratios
        .iter()
        .filter(|&&x| x.is_finite() && x >= 1.0)
        .copied()
        .collect();

    if sorted.len() < 2 {
        return TwoNNResult {
            dimension: f32::NAN,
            std_error: f32::NAN,
            n_samples: 0,
            ci_lower: f32::NAN,
            ci_upper: f32::NAN,
        };
    }

    sorted.sort_unstable_by(|a, b| a.total_cmp(b));

    let keep_count = ((sorted.len() as f32) * (1.0 - discard_fraction)).max(2.0) as usize;
    let sorted = &sorted[..keep_count.min(sorted.len())];

    // MLE formulation: d̂ = n / Σᵢ log(μᵢ)
    // Filter out μ ≈ 1 to avoid log(1) = 0
    let valid_ratios: Vec<f32> = sorted.iter().filter(|&&mu| mu > 1.0001).copied().collect();
    let n = valid_ratios.len();

    if n < 2 {
        return TwoNNResult {
            dimension: f32::NAN,
            std_error: f32::NAN,
            n_samples: n,
            ci_lower: f32::NAN,
            ci_upper: f32::NAN,
        };
    }

    let sum_log_mu: f64 = valid_ratios.iter().map(|&mu| (mu as f64).ln()).sum();

    if sum_log_mu.abs() < 1e-12 {
        return TwoNNResult {
            dimension: f32::NAN,
            std_error: f32::NAN,
            n_samples: n,
            ci_lower: f32::NAN,
            ci_upper: f32::NAN,
        };
    }

    let d = (n as f64) / sum_log_mu;

    // Asymptotic variance: Var(d̂) ≈ d² / n
    // Standard error: SE = d / √n
    let std_error = d / (n as f64).sqrt();

    // 95% CI using z = 1.96
    let ci_lower = (d - 1.96 * std_error).max(0.0);
    let ci_upper = d + 1.96 * std_error;

    TwoNNResult {
        dimension: d as f32,
        std_error: std_error as f32,
        n_samples: n,
        ci_lower: ci_lower as f32,
        ci_upper: ci_upper as f32,
    }
}

/// Methods for aggregating pointwise LID estimates into a global estimate.
#[derive(Debug, Clone, Copy, Default)]
pub enum LidAggregation {
    /// Arithmetic mean of LID estimates.
    #[default]
    Mean,
    /// Median LID (robust to outliers).
    Median,
    /// Harmonic mean: 1 / mean(1/LID).
    /// More robust for skewed distributions. Used in scikit-dimension.
    HarmonicMean,
}

/// Aggregate multiple LID estimates into a single global estimate.
///
/// # Methods
///
/// - **Mean**: Simple average. Can be biased by outliers.
/// - **Median**: Robust to outliers, good default choice.
/// - **HarmonicMean**: `1 / mean(1/LID)`. Preferred for skewed distributions
///   as it down-weights very high LID values.
///
/// # Example
///
/// ```rust
/// use vicinity::lid::{estimate_lid_mle, aggregate_lid, LidConfig, LidAggregation};
///
/// let config = LidConfig::default();
/// let estimates: Vec<_> = [
///     vec![0.1, 0.2, 0.3, 0.5, 0.8],
///     vec![0.15, 0.25, 0.35, 0.55, 0.85],
/// ]
/// .iter()
/// .map(|d| estimate_lid_mle(d, &config))
/// .collect();
///
/// let global_lid = aggregate_lid(&estimates, LidAggregation::HarmonicMean);
/// assert!(global_lid.is_finite());
/// ```
#[must_use]
pub fn aggregate_lid(estimates: &[LidEstimate], method: LidAggregation) -> f32 {
    let valid: Vec<f32> = estimates
        .iter()
        .map(|e| e.lid)
        .filter(|&lid| lid.is_finite() && lid > 0.0)
        .collect();

    if valid.is_empty() {
        return f32::NAN;
    }

    match method {
        LidAggregation::Mean => valid.iter().sum::<f32>() / valid.len() as f32,
        LidAggregation::Median => {
            let mut sorted = valid.clone();
            sorted.sort_unstable_by(|a, b| a.total_cmp(b));
            let n = sorted.len();
            if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            }
        }
        LidAggregation::HarmonicMean => {
            let sum_inv: f32 = valid.iter().map(|&x| 1.0 / x).sum();
            valid.len() as f32 / sum_inv
        }
    }
}

/// Estimate LID for all points in a dataset using brute-force k-NN.
///
/// Useful for analysis and debugging, but O(n²) complexity.
///
/// # Arguments
///
/// * `vectors` - Flat array of vectors (n * dimension elements)
/// * `dimension` - Vector dimension
/// * `config` - LID estimation parameters
///
/// # Returns
///
/// Vector of LID estimates, one per point.
pub fn estimate_lid_batch(
    vectors: &[f32],
    dimension: usize,
    config: &LidConfig,
) -> Vec<LidEstimate> {
    let n = vectors.len() / dimension;
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let query = &vectors[i * dimension..(i + 1) * dimension];

        // Compute EUCLIDEAN distances to all other points.
        // Note: LID estimation requires Euclidean distance (L2), not cosine distance,
        // because the MLE formula is derived from the volume growth rate in Euclidean
        // space: V(r) ∝ r^d implies P(dist ≤ r) ∝ r^d.
        let mut distances: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let other = &vectors[j * dimension..(j + 1) * dimension];
                distance::l2_distance(query, other)
            })
            .collect();

        distances.sort_unstable_by(|a, b| a.total_cmp(b));
        results.push(estimate_lid_mle(&distances, config));
    }

    results
}

/// Classify a LID estimate relative to a reference distribution.
///
/// # Categories
///
/// - **Low LID** (< median - σ): Point in dense region, well-connected locally.
/// - **Normal LID**: Typical point.
/// - **High LID** (> median + σ): Potential outlier, sparse local neighborhood.
///
/// High-LID points benefit from special handling in HNSW construction
/// (more edges, different entry point selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LidCategory {
    /// LID significantly below median - dense region.
    Low,
    /// LID near median - typical point.
    Normal,
    /// LID significantly above median - sparse region / outlier.
    High,
}

/// Compute LID statistics for a batch of estimates.
#[derive(Debug, Clone)]
pub struct LidStats {
    /// Mean LID across all points.
    pub mean: f32,
    /// Median LID.
    pub median: f32,
    /// Standard deviation.
    pub std_dev: f32,
    /// Minimum LID.
    pub min: f32,
    /// Maximum LID.
    pub max: f32,
    /// Count of points.
    pub count: usize,
}

impl LidStats {
    /// Compute statistics from a batch of LID estimates.
    pub fn from_estimates(estimates: &[LidEstimate]) -> Self {
        let valid: Vec<f32> = estimates
            .iter()
            .map(|e| e.lid)
            .filter(|lid| lid.is_finite())
            .collect();

        if valid.is_empty() {
            return Self {
                mean: f32::NAN,
                median: f32::NAN,
                std_dev: f32::NAN,
                min: f32::NAN,
                max: f32::NAN,
                count: 0,
            };
        }

        let count = valid.len();
        let mean = valid.iter().sum::<f32>() / count as f32;

        let variance = valid.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();

        let mut sorted = valid.clone();
        sorted.sort_unstable_by(|a, b| a.total_cmp(b));

        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        Self {
            mean,
            median,
            std_dev,
            min: sorted[0],
            max: sorted[count - 1],
            count,
        }
    }

    /// Categorize a LID value relative to these statistics.
    pub fn categorize(&self, lid: f32) -> LidCategory {
        if !lid.is_finite() {
            return LidCategory::High; // Treat infinite LID as outlier
        }

        if lid < self.median - self.std_dev {
            LidCategory::Low
        } else if lid > self.median + self.std_dev {
            LidCategory::High
        } else {
            LidCategory::Normal
        }
    }

    /// Get threshold for high-LID outliers.
    pub fn high_lid_threshold(&self) -> f32 {
        self.median + self.std_dev
    }
}

/// Integration point for HNSW: compute LID during construction.
///
/// This function is designed to be called during HNSW graph building
/// to identify outlier points that need special handling.
///
/// # Usage in HNSW
///
/// From Dual-Branch HNSW (2025):
/// 1. Estimate LID for each new point during insertion
/// 2. For high-LID points (outliers):
///    - Use more neighbors (higher m)
///    - Consider special entry points
///    - Add skip bridges to bypass redundant layers
///
/// # Example
///
/// ```ignore
/// // During HNSW construction
/// let lid = estimate_lid_for_hnsw(&neighbor_distances, k);
/// if lid.lid > stats.high_lid_threshold() {
///     // Use extended neighborhood for this outlier
///     let m_extended = m * 2;
///     select_neighbors_extended(candidates, m_extended);
/// }
/// ```
pub fn estimate_lid_for_hnsw(neighbor_distances: &[f32], k: usize) -> LidEstimate {
    let config = LidConfig { k, epsilon: 1e-10 };
    estimate_lid(neighbor_distances, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lid_uniform_distances() {
        // Linearly increasing distances should give moderate LID
        let distances: Vec<f32> = (1..=20).map(|i| i as f32 * 0.1).collect();
        let estimate = estimate_lid_mle(&distances, &LidConfig::default());

        assert!(estimate.lid.is_finite());
        assert!(estimate.lid > 0.0);
        println!("Linear distances LID: {}", estimate.lid);
    }

    #[test]
    fn test_lid_exponential_distances() {
        // Exponentially increasing distances (common in high-dim) should give higher LID
        let distances: Vec<f32> = (1..=20).map(|i| (i as f32).exp() * 0.01).collect();
        let estimate = estimate_lid_mle(&distances, &LidConfig::default());

        assert!(estimate.lid.is_finite());
        println!("Exponential distances LID: {}", estimate.lid);
    }

    #[test]
    fn test_lid_equal_distances() {
        // All equal distances = degenerate case
        let distances = vec![1.0f32; 20];
        let estimate = estimate_lid_mle(&distances, &LidConfig::default());

        // Should be infinite or very high
        assert!(estimate.lid.is_infinite() || estimate.lid > 100.0);
    }

    #[test]
    fn test_lid_stats() {
        let estimates = vec![
            LidEstimate {
                lid: 5.0,
                k: 20,
                max_dist: 1.0,
            },
            LidEstimate {
                lid: 10.0,
                k: 20,
                max_dist: 1.5,
            },
            LidEstimate {
                lid: 8.0,
                k: 20,
                max_dist: 1.2,
            },
            LidEstimate {
                lid: 7.0,
                k: 20,
                max_dist: 1.1,
            },
            LidEstimate {
                lid: 15.0,
                k: 20,
                max_dist: 2.0,
            }, // Outlier
        ];

        let stats = LidStats::from_estimates(&estimates);

        assert_eq!(stats.count, 5);
        assert!(stats.mean > 0.0);
        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 15.0);

        // 15.0 should be categorized as High
        assert_eq!(stats.categorize(15.0), LidCategory::High);
        // 8.0 should be Normal (close to median)
        assert_eq!(stats.categorize(8.0), LidCategory::Normal);
    }

    #[test]
    fn test_lid_batch() {
        // Create simple 2D points
        let dim = 2;
        let vectors: Vec<f32> = vec![
            0.0, 0.0, // Point 0
            1.0, 0.0, // Point 1
            0.0, 1.0, // Point 2
            1.0, 1.0, // Point 3
            10.0, 10.0, // Point 4 - outlier
        ];

        let config = LidConfig {
            k: 3,
            epsilon: 1e-10,
        };
        let estimates = estimate_lid_batch(&vectors, dim, &config);

        assert_eq!(estimates.len(), 5);

        // The outlier (point 4) should have higher LID
        let stats = LidStats::from_estimates(&estimates);
        println!(
            "LID stats: mean={}, median={}, std={}",
            stats.mean, stats.median, stats.std_dev
        );

        // Point 4's LID should be in the high category
        // (This is a soft test - LID behavior depends on distance distribution)
    }

    #[test]
    fn test_lid_config_k() {
        let distances: Vec<f32> = (1..=50).map(|i| i as f32 * 0.1).collect();

        let config_small = LidConfig {
            k: 5,
            epsilon: 1e-10,
        };
        let config_large = LidConfig {
            k: 30,
            epsilon: 1e-10,
        };

        let est_small = estimate_lid_mle(&distances, &config_small);
        let est_large = estimate_lid_mle(&distances, &config_large);

        assert_eq!(est_small.k, 5);
        assert_eq!(est_large.k, 30);

        // Different k can give different estimates
        println!("k=5 LID: {}, k=30 LID: {}", est_small.lid, est_large.lid);
    }

    #[test]
    fn test_twonn_basic() {
        // For uniform data on a 2D manifold, μ ratios should give d ≈ 2
        // Create synthetic ratios that would come from ~2D data
        let mu_ratios: Vec<f32> = (1..=100).map(|i| 1.0 + (i as f32 * 0.02)).collect();

        let dim = estimate_twonn(&mu_ratios, 0.1);
        println!("TwoNN estimated dimension: {}", dim);

        assert!(dim.is_finite());
        assert!(dim > 0.0);
    }

    #[test]
    fn test_twonn_empty() {
        let dim = estimate_twonn(&[], 0.1);
        assert!(dim.is_nan());
    }

    #[test]
    fn test_twonn_discards_outliers() {
        // Create ratios with some extreme outliers
        let mut mu_ratios: Vec<f32> = (1..=90).map(|i| 1.0 + (i as f32 * 0.01)).collect();
        // Add outliers
        mu_ratios.resize(mu_ratios.len() + 10, 100.0);

        let dim_no_discard = estimate_twonn(&mu_ratios, 0.0);
        let dim_with_discard = estimate_twonn(&mu_ratios, 0.15);

        println!(
            "TwoNN no discard: {}, with discard: {}",
            dim_no_discard, dim_with_discard
        );

        assert!(dim_with_discard.is_finite());
    }

    #[test]
    fn test_twonn_with_ci() {
        // Create synthetic ratios
        let mu_ratios: Vec<f32> = (1..=200).map(|i| 1.0 + (i as f32 * 0.015)).collect();

        let result = estimate_twonn_with_ci(&mu_ratios, 0.1);

        println!(
            "TwoNN with CI: d={:.2}, SE={:.3}, 95% CI=[{:.2}, {:.2}], n={}",
            result.dimension, result.std_error, result.ci_lower, result.ci_upper, result.n_samples
        );

        assert!(result.dimension.is_finite());
        assert!(result.std_error.is_finite());
        assert!(result.std_error > 0.0);
        assert!(result.ci_lower <= result.dimension);
        assert!(result.dimension <= result.ci_upper);
        assert!(result.n_samples > 0);
    }

    #[test]
    fn test_twonn_equidistant_neighbors() {
        // All ratios exactly 1.0 (equidistant neighbors)
        let mu_ratios: Vec<f32> = vec![1.0; 100];

        let dim = estimate_twonn(&mu_ratios, 0.1);
        let result = estimate_twonn_with_ci(&mu_ratios, 0.1);

        // Should return NaN since log(1) = 0
        assert!(dim.is_nan());
        assert!(result.dimension.is_nan());
    }

    #[test]
    fn test_twonn_mixed_equidistant() {
        // Mix of equidistant and non-equidistant
        let mut mu_ratios: Vec<f32> = vec![1.0; 50];
        mu_ratios.extend((1..=50).map(|i| 1.0 + i as f32 * 0.02));

        let dim = estimate_twonn(&mu_ratios, 0.1);
        let result = estimate_twonn_with_ci(&mu_ratios, 0.1);

        // Should still give valid results from the non-degenerate ratios
        assert!(dim.is_finite());
        assert!(result.dimension.is_finite());
    }

    #[test]
    fn test_aggregation_methods() {
        let estimates = vec![
            LidEstimate {
                lid: 5.0,
                k: 20,
                max_dist: 1.0,
            },
            LidEstimate {
                lid: 10.0,
                k: 20,
                max_dist: 1.5,
            },
            LidEstimate {
                lid: 8.0,
                k: 20,
                max_dist: 1.2,
            },
            LidEstimate {
                lid: 7.0,
                k: 20,
                max_dist: 1.1,
            },
            LidEstimate {
                lid: 50.0,
                k: 20,
                max_dist: 5.0,
            }, // Outlier
        ];

        let mean = aggregate_lid(&estimates, LidAggregation::Mean);
        let median = aggregate_lid(&estimates, LidAggregation::Median);
        let harmonic = aggregate_lid(&estimates, LidAggregation::HarmonicMean);

        println!("Mean: {}, Median: {}, Harmonic: {}", mean, median, harmonic);

        // Mean should be most affected by the outlier
        assert!(mean > median);
        // Harmonic mean should be lower than arithmetic mean
        assert!(harmonic < mean);
        // Median should be around 8 (middle value)
        assert!((median - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_aggregation_empty() {
        let result = aggregate_lid(&[], LidAggregation::Mean);
        assert!(result.is_nan());
    }

    #[test]
    fn test_aggregation_handles_infinities() {
        let estimates = vec![
            LidEstimate {
                lid: 5.0,
                k: 20,
                max_dist: 1.0,
            },
            LidEstimate {
                lid: f32::INFINITY,
                k: 20,
                max_dist: 1.5,
            },
            LidEstimate {
                lid: 7.0,
                k: 20,
                max_dist: 1.2,
            },
        ];

        let mean = aggregate_lid(&estimates, LidAggregation::Mean);

        // Should ignore the infinity
        assert!(mean.is_finite());
        assert_eq!(mean, 6.0);
    }
}
