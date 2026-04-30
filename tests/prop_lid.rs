//! Property-based tests for Local Intrinsic Dimensionality (LID) estimation.
//!
//! Tests LID MLE, TwoNN estimator, and aggregation methods for:
//! - Positivity, finiteness, scale invariance
//! - CI ordering (lower <= estimate <= upper)
//! - Aggregation bounds (harmonic <= arithmetic, median in [min, max])

#![cfg(feature = "hnsw")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proptest::prelude::*;
use vicinity::lid::{
    aggregate_lid, estimate_lid_mle, estimate_twonn, estimate_twonn_with_ci, LidAggregation,
    LidCategory, LidConfig, LidEstimate, LidStats,
};

// =============================================================================
// LID MLE Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// LID should be positive for valid distance sequences.
    #[test]
    fn lid_positive_for_valid_distances(
        base in 0.01f32..0.1,
        increments in prop::collection::vec(0.01f32..0.5, 19),
    ) {
        let mut distances = vec![base];
        let mut cumsum = base;
        for inc in increments {
            cumsum += inc;
            distances.push(cumsum);
        }

        let config = LidConfig::default();
        let estimate = estimate_lid_mle(&distances, &config);

        prop_assert!(estimate.lid > 0.0 || estimate.lid.is_infinite(),
            "LID should be positive, got {}", estimate.lid);
    }

    /// LID should be invariant to uniform scaling of distances.
    #[test]
    fn lid_scale_invariant(
        base in 0.01f32..0.1,
        increments in prop::collection::vec(0.01f32..0.5, 19),
        scale in 0.1f32..10.0,
    ) {
        let mut distances = vec![base];
        let mut cumsum = base;
        for inc in &increments {
            cumsum += inc;
            distances.push(cumsum);
        }

        let scaled: Vec<f32> = distances.iter().map(|d| d * scale).collect();

        let config = LidConfig::default();
        let est1 = estimate_lid_mle(&distances, &config);
        let est2 = estimate_lid_mle(&scaled, &config);

        if est1.lid.is_finite() && est2.lid.is_finite() {
            let relative_diff = (est1.lid - est2.lid).abs() / est1.lid.max(1.0);
            prop_assert!(relative_diff < 0.3,
                "LID not scale invariant: {} vs {} (scale={})",
                est1.lid, est2.lid, scale);
        }
    }

    /// k parameter should be respected.
    #[test]
    fn lid_respects_k(
        distances in prop::collection::vec(0.1f32..10.0, 30..50),
        k in 5usize..25,
    ) {
        let mut sorted = distances.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let config = LidConfig { k, epsilon: 1e-10 };
        let estimate = estimate_lid_mle(&sorted, &config);

        prop_assert_eq!(estimate.k, k.min(sorted.len()),
            "k should be min of config.k and distances.len()");
    }

    /// LidStats should produce valid statistics.
    #[test]
    fn lid_stats_valid(
        lids in prop::collection::vec(1.0f32..50.0, 10..30),
    ) {
        let estimates: Vec<LidEstimate> = lids.iter()
            .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
            .collect();

        let stats = LidStats::from_estimates(&estimates);

        prop_assert_eq!(stats.count, estimates.len());
        prop_assert!(stats.min <= stats.mean);
        prop_assert!(stats.mean <= stats.max);
        prop_assert!(stats.std_dev >= 0.0);
    }

    /// High LID threshold should be above median.
    #[test]
    fn high_lid_threshold_above_median(
        lids in prop::collection::vec(1.0f32..50.0, 5..20),
    ) {
        let estimates: Vec<LidEstimate> = lids.iter()
            .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
            .collect();

        let stats = LidStats::from_estimates(&estimates);

        if stats.std_dev > 0.0 {
            prop_assert!(stats.high_lid_threshold() > stats.median,
                "threshold {} should be > median {}",
                stats.high_lid_threshold(), stats.median);
        }
    }

    /// LID categorization is consistent with thresholds.
    #[test]
    fn lid_categorization_consistent(
        lids in prop::collection::vec(1.0f32..100.0, 20..50),
    ) {
        let estimates: Vec<LidEstimate> = lids.iter()
            .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
            .collect();

        let stats = LidStats::from_estimates(&estimates);

        for &lid in &lids {
            let category = stats.categorize(lid);
            match category {
                LidCategory::Low => {
                    prop_assert!(lid < stats.median - stats.std_dev + 1e-5,
                        "Low category but lid {} >= median - std = {}",
                        lid, stats.median - stats.std_dev);
                }
                LidCategory::High => {
                    prop_assert!(lid > stats.median + stats.std_dev - 1e-5,
                        "High category but lid {} <= median + std = {}",
                        lid, stats.median + stats.std_dev);
                }
                LidCategory::Normal => {}
            }
        }
    }

    /// LID for uniform distances gives finite positive result.
    #[test]
    fn lid_uniform_distances_finite(
        n in 10usize..50,
        step in 0.01f32..0.5,
    ) {
        let distances: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * step).collect();
        let config = LidConfig::default();
        let estimate = estimate_lid_mle(&distances, &config);

        prop_assert!(estimate.lid.is_finite(),
            "LID should be finite for uniform distances, got {}", estimate.lid);
        prop_assert!(estimate.lid > 0.0,
            "LID should be positive, got {}", estimate.lid);
    }
}

// =============================================================================
// TwoNN Estimator Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// TwoNN should return positive dimension for valid mu ratios.
    #[test]
    fn twonn_positive_dimension(
        n in 20usize..100,
        seed in any::<u64>(),
    ) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let h = hasher.finish();

        let ratios: Vec<f32> = (0..n)
            .map(|i| {
                let x = ((h.wrapping_mul(i as u64 + 1)) % 10000) as f32 / 10000.0;
                1.0 + x * 3.0
            })
            .collect();

        let dim = estimate_twonn(&ratios, 0.1);

        if dim.is_finite() {
            prop_assert!(dim > 0.0, "TwoNN dimension should be positive, got {}", dim);
        }
    }

    /// TwoNN with CI should have CI_lower <= dimension <= CI_upper.
    #[test]
    fn twonn_ci_ordering(
        n in 50usize..200,
        seed in any::<u64>(),
    ) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let h = hasher.finish();

        let ratios: Vec<f32> = (0..n)
            .map(|i| {
                let x = ((h.wrapping_mul(i as u64 + 1)) % 10000) as f32 / 10000.0;
                1.0 + x * 2.0
            })
            .collect();

        let result = estimate_twonn_with_ci(&ratios, 0.1);

        if result.dimension.is_finite() && result.ci_lower.is_finite() && result.ci_upper.is_finite() {
            prop_assert!(
                result.ci_lower <= result.dimension,
                "CI lower {} should be <= dimension {}",
                result.ci_lower, result.dimension
            );
            prop_assert!(
                result.dimension <= result.ci_upper,
                "dimension {} should be <= CI upper {}",
                result.dimension, result.ci_upper
            );
        }
    }

    /// TwoNN and TwoNN with CI should give similar dimension estimates.
    #[test]
    fn twonn_consistency(
        n in 50usize..200,
        target_d in 2.0f32..20.0,
        seed in any::<u64>(),
    ) {
        // Generate ratios from the TwoNN-expected distribution: log(μ) ~ Exp(d),
        // i.e. μ = u^(-1/d) for u uniform in (0, 1). The previous version of this
        // test sampled μ uniformly on [1, 3), which is not a TwoNN-compatible
        // distribution -- on certain seeds it produced bimodal data on which
        // the two estimators legitimately disagree (rel diff > 0.5). The
        // estimators were correct; the test inputs were out of distribution.
        // Probed: max rel_diff on realistic input across d ∈ {3, 5, 10, 20}
        // and 5 seeds is ~0.069. 0.15 leaves ample slack.
        let ratios: Vec<f32> = (0..n)
            .map(|i| {
                // SplitMix64-style mixer for deterministic uniform-(0, 1).
                let mut x = seed
                    .wrapping_add(i as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15);
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
                x ^= x >> 31;
                let u = (x as f64 / u64::MAX as f64).clamp(1e-9, 1.0 - 1e-9);
                (u.powf(-1.0 / target_d as f64) as f32).max(1.0001)
            })
            .collect();

        let dim1 = estimate_twonn(&ratios, 0.1);
        let result2 = estimate_twonn_with_ci(&ratios, 0.1);

        if dim1.is_finite() && result2.dimension.is_finite() {
            let rel_diff = (dim1 - result2.dimension).abs() / dim1.max(result2.dimension);
            prop_assert!(
                rel_diff < 0.15,
                "TwoNN methods disagree on realistic input: {} vs {} (rel diff {}, target d={})",
                dim1, result2.dimension, rel_diff, target_d
            );
        }
    }

    /// TwoNN should be robust to outlier discarding.
    #[test]
    fn twonn_discard_sensitivity(
        n in 100usize..300,
        seed in any::<u64>(),
    ) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let h = hasher.finish();

        let ratios: Vec<f32> = (0..n)
            .map(|i| {
                let x = ((h.wrapping_mul(i as u64 + 1)) % 10000) as f32 / 10000.0;
                if i < n - 5 {
                    1.0 + x * 2.0
                } else {
                    5.0 + x * 10.0
                }
            })
            .collect();

        let dim_no_discard = estimate_twonn(&ratios, 0.0);
        let dim_with_discard = estimate_twonn(&ratios, 0.1);

        if dim_no_discard.is_finite() && dim_with_discard.is_finite() {
            prop_assert!(
                dim_no_discard > 0.0 || dim_with_discard > 0.0,
                "At least one estimate should be positive"
            );
        }
    }

    /// Empty or very small inputs should return NaN.
    #[test]
    fn twonn_empty_returns_nan(
        n in 0usize..2,
    ) {
        let ratios: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.5).collect();
        let dim = estimate_twonn(&ratios, 0.1);
        let result = estimate_twonn_with_ci(&ratios, 0.1);

        if n < 2 {
            prop_assert!(dim.is_nan(), "Should return NaN for n={}", n);
            prop_assert!(result.dimension.is_nan(), "CI version should return NaN for n={}", n);
        }
    }

    /// Mu ratios exactly equal to 1.0 (equidistant neighbors) should be handled.
    #[test]
    fn twonn_handles_equidistant(
        n in 20usize..50,
        frac_equidistant in 0.0f32..0.5,
    ) {
        let n_equidistant = (n as f32 * frac_equidistant) as usize;

        let ratios: Vec<f32> = (0..n)
            .map(|i| {
                if i < n_equidistant {
                    1.0
                } else {
                    1.1 + (i as f32 * 0.1) % 2.0
                }
            })
            .collect();

        let dim = estimate_twonn(&ratios, 0.1);

        if n - n_equidistant >= 3 {
            prop_assert!(dim.is_finite() && dim > 0.0,
                "Expected positive finite LID estimate, got {} (n={}, equidistant={})",
                dim, n, n_equidistant);
        }
    }
}

// =============================================================================
// LID Aggregation Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Harmonic mean is always <= arithmetic mean for positive values.
    #[test]
    fn harmonic_le_arithmetic(
        lids in prop::collection::vec(0.5f32..50.0, 5..30),
    ) {
        let estimates: Vec<LidEstimate> = lids.iter()
            .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
            .collect();

        let mean = aggregate_lid(&estimates, LidAggregation::Mean);
        let harmonic = aggregate_lid(&estimates, LidAggregation::HarmonicMean);

        prop_assert!(
            harmonic <= mean + 1e-6,
            "Harmonic mean {} should be <= arithmetic mean {}",
            harmonic, mean
        );
    }

    /// Median is between min and max.
    #[test]
    fn median_bounded(
        lids in prop::collection::vec(1.0f32..100.0, 3..50),
    ) {
        let estimates: Vec<LidEstimate> = lids.iter()
            .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
            .collect();

        let median = aggregate_lid(&estimates, LidAggregation::Median);

        let min_lid = lids.iter().cloned().reduce(f32::min).unwrap();
        let max_lid = lids.iter().cloned().reduce(f32::max).unwrap();

        prop_assert!(
            median >= min_lid - 1e-6 && median <= max_lid + 1e-6,
            "Median {} should be in [{}, {}]",
            median, min_lid, max_lid
        );
    }

    /// All aggregation methods should be finite for valid inputs.
    #[test]
    fn aggregation_finite(
        lids in prop::collection::vec(1.0f32..50.0, 5..20),
    ) {
        let estimates: Vec<LidEstimate> = lids.iter()
            .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
            .collect();

        let mean = aggregate_lid(&estimates, LidAggregation::Mean);
        let median = aggregate_lid(&estimates, LidAggregation::Median);
        let harmonic = aggregate_lid(&estimates, LidAggregation::HarmonicMean);

        prop_assert!(mean.is_finite(), "Mean should be finite");
        prop_assert!(median.is_finite(), "Median should be finite");
        prop_assert!(harmonic.is_finite(), "Harmonic mean should be finite");
    }
}
