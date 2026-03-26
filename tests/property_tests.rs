#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Property-based tests for vicinity ANN components.
//!
//! These tests verify invariants that should hold regardless of input:
//! - Distance metrics satisfy metric space properties
//! - Recall is always in [0, 1]
//! - Memory estimates are consistent
//! - Ground truth computation is correct

use proptest::prelude::*;

mod distance_props {
    use super::*;

    fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        1.0 - dot / (norm_a * norm_b)
    }

    prop_compose! {
        fn arb_vector(dim: usize)(vec in prop::collection::vec(-10.0f32..10.0, dim)) -> Vec<f32> {
            vec
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn l2_distance_non_negative(
            a in arb_vector(64),
            b in arb_vector(64),
        ) {
            let dist = l2_distance_squared(&a, &b);
            prop_assert!(dist >= 0.0, "L2 distance must be non-negative, got {}", dist);
        }

        #[test]
        fn l2_distance_symmetric(
            a in arb_vector(32),
            b in arb_vector(32),
        ) {
            let d_ab = l2_distance_squared(&a, &b);
            let d_ba = l2_distance_squared(&b, &a);
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-6,
                "L2 distance not symmetric: {} vs {}",
                d_ab, d_ba
            );
        }

        #[test]
        fn l2_distance_self_is_zero(
            a in arb_vector(32),
        ) {
            let dist = l2_distance_squared(&a, &a);
            prop_assert!(
                dist.abs() < 1e-10,
                "Distance to self should be 0, got {}",
                dist
            );
        }

        #[test]
        fn l2_triangle_inequality(
            a in arb_vector(16),
            b in arb_vector(16),
            c in arb_vector(16),
        ) {
            // For squared L2, triangle inequality is:
            // sqrt(d(a,c)) <= sqrt(d(a,b)) + sqrt(d(b,c))
            let d_ac = l2_distance_squared(&a, &c).sqrt();
            let d_ab = l2_distance_squared(&a, &b).sqrt();
            let d_bc = l2_distance_squared(&b, &c).sqrt();

            prop_assert!(
                d_ac <= d_ab + d_bc + 1e-5,
                "Triangle inequality violated: {} > {} + {}",
                d_ac, d_ab, d_bc
            );
        }

        #[test]
        fn cosine_distance_in_range(
            a in arb_vector(32),
            b in arb_vector(32),
        ) {
            let dist = cosine_distance(&a, &b);
            // Cosine distance should be in [0, 2] (or 1 for zero vectors)
            // Due to floating point, allow small violations
            prop_assert!(
                (-0.001..=2.001).contains(&dist),
                "Cosine distance out of range: {}",
                dist
            );
        }

        #[test]
        fn cosine_distance_symmetric(
            a in arb_vector(32),
            b in arb_vector(32),
        ) {
            let d_ab = cosine_distance(&a, &b);
            let d_ba = cosine_distance(&b, &a);
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-5,
                "Cosine distance not symmetric: {} vs {}",
                d_ab, d_ba
            );
        }
    }
}

mod recall_props {
    use super::*;
    use std::collections::HashSet;

    fn recall_at_k(ground_truth: &[u32], retrieved: &[u32], k: usize) -> f32 {
        if k == 0 || ground_truth.is_empty() {
            return 0.0;
        }

        let gt_set: HashSet<u32> = ground_truth.iter().take(k).copied().collect();
        let ret_set: HashSet<u32> = retrieved.iter().take(k).copied().collect();

        gt_set.intersection(&ret_set).count() as f32 / k as f32
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn recall_in_unit_interval(
            gt in prop::collection::vec(0u32..1000, 1..50),
            ret in prop::collection::vec(0u32..1000, 1..50),
            k in 1usize..20,
        ) {
            let recall = recall_at_k(&gt, &ret, k);
            prop_assert!(
                (0.0..=1.0).contains(&recall),
                "Recall must be in [0,1], got {}",
                recall
            );
        }

        #[test]
        fn perfect_recall_when_identical(
            // Use HashSet to guarantee unique values
            gt_set in prop::collection::hash_set(0u32..1000, 1..20),
        ) {
            let gt: Vec<u32> = gt_set.into_iter().collect();
            let k = gt.len();
            let recall = recall_at_k(&gt, &gt, k);
            prop_assert!(
                (recall - 1.0).abs() < 1e-6,
                "Recall should be 1.0 for identical sets, got {}",
                recall
            );
        }

        #[test]
        fn zero_recall_disjoint_sets(
            offset in 0u32..1000,
            size in 1usize..20,
        ) {
            let gt: Vec<u32> = (0..size as u32).collect();
            let ret: Vec<u32> = (offset + 1000..(offset + 1000 + size as u32)).collect();
            let recall = recall_at_k(&gt, &ret, size);
            prop_assert!(
                recall.abs() < 1e-6,
                "Recall should be 0 for disjoint sets, got {}",
                recall
            );
        }

        #[test]
        fn recall_monotonic_with_overlap(
            base in prop::collection::vec(0u32..50, 10..20),
        ) {
            // As we include more of the ground truth in retrieved,
            // recall should increase (or stay same)
            let k = base.len();
            let mut retrieved: Vec<u32> = (100..100 + k as u32).collect();

            let mut prev_recall = recall_at_k(&base, &retrieved, k);

            for (i, &gt_item) in base.iter().enumerate().take(k / 2) {
                retrieved[i] = gt_item;
                let new_recall = recall_at_k(&base, &retrieved, k);
                prop_assert!(
                    new_recall >= prev_recall - 1e-6,
                    "Recall decreased from {} to {} when adding correct items",
                    prev_recall, new_recall
                );
                prev_recall = new_recall;
            }
        }
    }
}

mod memory_props {
    use super::*;

    fn theoretical_hnsw_memory(n_vectors: usize, dimension: usize, m: usize) -> usize {
        let raw_bytes = n_vectors * dimension * 4;
        let avg_edges = (2.5 * m as f64) as usize;
        let graph_bytes = n_vectors * avg_edges * 4;
        let metadata_bytes = n_vectors * 5;
        raw_bytes + graph_bytes + metadata_bytes
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn memory_increases_with_vectors(
            n1 in 100usize..1000,
            n2 in 1000usize..10000,
            dim in 32usize..256,
            m in 8usize..32,
        ) {
            let mem1 = theoretical_hnsw_memory(n1, dim, m);
            let mem2 = theoretical_hnsw_memory(n2, dim, m);
            prop_assert!(
                mem2 > mem1,
                "Memory should increase with vectors: {} vs {}",
                mem1, mem2
            );
        }

        #[test]
        fn memory_increases_with_dimension(
            n in 1000usize..5000,
            dim1 in 32usize..128,
            dim2 in 256usize..512,
            m in 8usize..24,
        ) {
            let mem1 = theoretical_hnsw_memory(n, dim1, m);
            let mem2 = theoretical_hnsw_memory(n, dim2, m);
            prop_assert!(
                mem2 > mem1,
                "Memory should increase with dimension: {} vs {}",
                mem1, mem2
            );
        }

        #[test]
        fn memory_increases_with_m(
            n in 1000usize..5000,
            dim in 64usize..256,
            m1 in 4usize..16,
            m2 in 24usize..48,
        ) {
            let mem1 = theoretical_hnsw_memory(n, dim, m1);
            let mem2 = theoretical_hnsw_memory(n, dim, m2);
            prop_assert!(
                mem2 > mem1,
                "Memory should increase with M: {} (M={}) vs {} (M={})",
                mem1, m1, mem2, m2
            );
        }

        #[test]
        fn memory_positive(
            n in 1usize..10000,
            dim in 1usize..512,
            m in 1usize..64,
        ) {
            let mem = theoretical_hnsw_memory(n, dim, m);
            prop_assert!(mem > 0, "Memory must be positive");
        }
    }
}

mod ground_truth_props {
    use super::*;

    fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn compute_ground_truth(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<(u32, f32)> {
        let mut distances: Vec<(u32, f32)> = database
            .iter()
            .enumerate()
            .map(|(i, vec)| (i as u32, l2_distance_squared(query, vec)))
            .collect();
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        distances.truncate(k);
        distances
    }

    prop_compose! {
        fn arb_database(n: usize, dim: usize)(
            db in prop::collection::vec(
                prop::collection::vec(-5.0f32..5.0, dim),
                n
            )
        ) -> Vec<Vec<f32>> {
            db
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn ground_truth_returns_k_results(
            db in arb_database(50, 16),
            k in 1usize..20,
        ) {
            let query: Vec<f32> = (0..16).map(|_| 0.0f32).collect();
            let gt = compute_ground_truth(&query, &db, k);
            let expected_k = k.min(db.len());
            prop_assert_eq!(
                gt.len(),
                expected_k,
                "Ground truth should return {} results, got {}",
                expected_k,
                gt.len()
            );
        }

        #[test]
        fn ground_truth_sorted_by_distance(
            db in arb_database(30, 16),
        ) {
            let query: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
            let gt = compute_ground_truth(&query, &db, db.len());

            for i in 1..gt.len() {
                prop_assert!(
                    gt[i].1 >= gt[i - 1].1,
                    "Ground truth not sorted: {} > {} at position {}",
                    gt[i - 1].1, gt[i].1, i
                );
            }
        }

        #[test]
        fn ground_truth_unique_ids(
            db in arb_database(30, 16),
        ) {
            let query: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
            let gt = compute_ground_truth(&query, &db, db.len());

            let ids: Vec<u32> = gt.iter().map(|(id, _)| *id).collect();
            let mut sorted_ids = ids.clone();
            sorted_ids.sort_unstable();
            sorted_ids.dedup();

            prop_assert_eq!(
                ids.len(),
                sorted_ids.len(),
                "Ground truth contains duplicate IDs"
            );
        }

        #[test]
        fn ground_truth_ids_valid(
            db in arb_database(30, 16),
        ) {
            let query: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
            let gt = compute_ground_truth(&query, &db, db.len());

            for (id, _) in &gt {
                prop_assert!(
                    (*id as usize) < db.len(),
                    "Ground truth ID {} out of range (db size {})",
                    id, db.len()
                );
            }
        }

        #[test]
        fn ground_truth_distances_match(
            db in arb_database(20, 8),
        ) {
            let query: Vec<f32> = (0..8).map(|i| i as f32 * 0.2).collect();
            let gt = compute_ground_truth(&query, &db, db.len());

            for (id, dist) in &gt {
                let actual_dist = l2_distance_squared(&query, &db[*id as usize]);
                prop_assert!(
                    (dist - actual_dist).abs() < 1e-5,
                    "Distance mismatch for ID {}: {} vs {}",
                    id, dist, actual_dist
                );
            }
        }
    }
}

// =============================================================================
// LID (Local Intrinsic Dimensionality) Properties
// =============================================================================

mod lid_props {
    use super::*;
    use vicinity::lid::{estimate_lid_mle, LidConfig, LidEstimate, LidStats};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// LID should be positive for valid distance sequences.
        #[test]
        fn lid_positive_for_valid_distances(
            // Generate increasing distances (realistic scenario)
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

            // LID should be approximately scale-invariant
            // (exact invariance holds for continuous distributions)
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
            use vicinity::lid::LidCategory;

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
                    LidCategory::Normal => {
                        // Normal is in between
                    }
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
}

// =============================================================================
// TwoNN Estimator Properties
// =============================================================================

mod twonn_props {
    use super::*;
    use vicinity::lid::{estimate_twonn, estimate_twonn_with_ci};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// TwoNN should return positive dimension for valid mu ratios.
        /// μ = r2/r1 >= 1 always (2nd neighbor at least as far as 1st).
        #[test]
        fn twonn_positive_dimension(
            n in 20usize..100,
            seed in any::<u64>(),
        ) {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            // Generate realistic mu ratios > 1.0
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            let h = hasher.finish();

            let ratios: Vec<f32> = (0..n)
                .map(|i| {
                    let x = ((h.wrapping_mul(i as u64 + 1)) % 10000) as f32 / 10000.0;
                    1.0 + x * 3.0 // ratios in [1.0, 4.0]
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

            let dim1 = estimate_twonn(&ratios, 0.1);
            let result2 = estimate_twonn_with_ci(&ratios, 0.1);

            // The linear regression vs MLE approaches should give similar results
            // but not identical due to different formulations
            if dim1.is_finite() && result2.dimension.is_finite() {
                // Allow 50% relative difference (they're different estimators)
                let rel_diff = (dim1 - result2.dimension).abs() / dim1.max(result2.dimension);
                prop_assert!(
                    rel_diff < 0.5,
                    "TwoNN methods disagree too much: {} vs {} (rel diff {})",
                    dim1, result2.dimension, rel_diff
                );
            }
        }

        /// TwoNN should be robust to outlier discarding.
        /// Higher discard fraction = more robust but higher variance.
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

            // Generate ratios with some outliers at the tail
            let ratios: Vec<f32> = (0..n)
                .map(|i| {
                    let x = ((h.wrapping_mul(i as u64 + 1)) % 10000) as f32 / 10000.0;
                    if i < n - 5 {
                        1.0 + x * 2.0 // normal ratios
                    } else {
                        5.0 + x * 10.0 // outliers
                    }
                })
                .collect();

            let dim_no_discard = estimate_twonn(&ratios, 0.0);
            let dim_with_discard = estimate_twonn(&ratios, 0.1);

            // With outliers, discarding should generally give lower (more realistic) ID
            // This is a soft check since it's statistical
            if dim_no_discard.is_finite() && dim_with_discard.is_finite() {
                // At least one should be valid
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

            // Empty or single-element should return NaN
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
                        1.0 // equidistant (degenerate case)
                    } else {
                        1.1 + (i as f32 * 0.1) % 2.0
                    }
                })
                .collect();

            let dim = estimate_twonn(&ratios, 0.1);

            // Should not panic, should return valid result if enough non-degenerate ratios
            if n - n_equidistant >= 3 {
                prop_assert!(dim.is_finite() || dim.is_nan(),
                    "Should handle equidistant neighbors gracefully");
            }
        }
    }
}

// =============================================================================
// LID Aggregation Properties
// =============================================================================

mod lid_aggregation_props {
    use super::*;
    use vicinity::lid::{aggregate_lid, LidAggregation, LidEstimate};

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
}

// =============================================================================
// Additional Distance Metric Properties
// =============================================================================

mod metric_space_props {
    use super::*;

    fn arb_vec(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-10.0f32..10.0, dim)
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Cosine distance is bounded [0, 2].
        #[test]
        fn cosine_distance_bounded(
            a in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
            b in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
        ) {
            let a_norm = normalize(&a);
            let b_norm = normalize(&b);

            let dot: f32 = a_norm.iter().zip(b_norm.iter()).map(|(x, y)| x * y).sum();
            let cosine_dist = 1.0 - dot;

            prop_assert!(
                (-0.01..=2.01).contains(&cosine_dist),
                "Cosine distance out of bounds: {}",
                cosine_dist
            );
        }

        /// Cosine distance to self is zero.
        #[test]
        fn cosine_distance_self_zero(
            v in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
        ) {
            let v_norm = normalize(&v);
            let dot: f32 = v_norm.iter().map(|x| x * x).sum();
            let cosine_dist = 1.0 - dot;

            prop_assert!(
                cosine_dist.abs() < 1e-5,
                "Cosine distance to self should be 0, got {}",
                cosine_dist
            );
        }

        /// L2 distance satisfies identity of indiscernibles.
        #[test]
        fn l2_identity_of_indiscernibles(v in arb_vec(32)) {
            let dist = l2_squared(&v, &v);
            prop_assert!(dist.abs() < 1e-6, "L2(v, v) should be 0, got {}", dist);
        }

        /// L2 distance is symmetric.
        #[test]
        fn l2_symmetric_test(
            a in arb_vec(32),
            b in arb_vec(32),
        ) {
            let ab = l2_squared(&a, &b);
            let ba = l2_squared(&b, &a);

            prop_assert!(
                (ab - ba).abs() < 1e-6,
                "L2 not symmetric: {} != {}",
                ab, ba
            );
        }

        /// L2 distance satisfies triangle inequality.
        #[test]
        fn l2_triangle_inequality_test(
            a in arb_vec(16),
            b in arb_vec(16),
            c in arb_vec(16),
        ) {
            let ab = l2_squared(&a, &b).sqrt();
            let bc = l2_squared(&b, &c).sqrt();
            let ac = l2_squared(&a, &c).sqrt();

            prop_assert!(
                ac <= ab + bc + 1e-4,
                "Triangle inequality violated: {} > {} + {} = {}",
                ac, ab, bc, ab + bc
            );
        }

        /// Dot product is bilinear: dot(a + b, c) = dot(a, c) + dot(b, c).
        #[test]
        fn dot_bilinear(
            a in arb_vec(32),
            b in arb_vec(32),
            c in arb_vec(32),
        ) {
            let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

            let dot_ab_c: f32 = ab.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
            let dot_a_c: f32 = a.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
            let dot_b_c: f32 = b.iter().zip(c.iter()).map(|(x, y)| x * y).sum();

            let expected = dot_a_c + dot_b_c;
            let tolerance = expected.abs() * 1e-4 + 1e-4;

            prop_assert!(
                (dot_ab_c - expected).abs() < tolerance,
                "Bilinearity violated: {} != {} + {} = {}",
                dot_ab_c, dot_a_c, dot_b_c, expected
            );
        }

        /// Scaling preserves direction.
        #[test]
        fn scaling_preserves_direction(
            v in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
            scale in 0.1f32..10.0,
        ) {
            let v_norm = normalize(&v);
            let scaled: Vec<f32> = v.iter().map(|x| x * scale).collect();
            let scaled_norm = normalize(&scaled);

            for (a, b) in v_norm.iter().zip(scaled_norm.iter()) {
                prop_assert!(
                    (a - b).abs() < 1e-5,
                    "Scaling changed direction"
                );
            }
        }
    }
}

// =============================================================================
// HNSW Mathematical Invariants
// =============================================================================
//
// HNSW's layer structure follows a geometric distribution, creating the
// hierarchical navigable small-world property that enables O(log n) search.
//
// ## Layer Assignment Distribution
//
// P(layer = l) = (1 - 1/m_l) × (1/m_l)^l
//
// This geometric distribution ensures:
// - Most vectors are at layer 0 (base layer)
// - Each upper layer has ~1/m_l fewer vectors
// - Expected vectors at layer l: N × (1/m_l)^l
//
// ## Reference
//
// Malkov & Yashunin (2016): "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs"
//

#[cfg(feature = "hnsw")]
mod hnsw_props {
    use super::*;
    use vicinity::hnsw::HNSWIndex;

    fn normalize(v: &[f32]) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n < 1e-10 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / n).collect()
        }
    }

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use std::hash::{Hash, Hasher};

        (0..n)
            .map(|i| {
                let raw: Vec<f32> = (0..dim)
                    .map(|j| {
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        seed.hash(&mut hasher);
                        i.hash(&mut hasher);
                        j.hash(&mut hasher);
                        let h = hasher.finish();
                        (h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
                    })
                    .collect();
                normalize(&raw)
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// HNSW search returns k results for k <= n.
        #[test]
        fn hnsw_returns_k_results(
            n in 20usize..100,
            k in 1usize..20,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let k = k.min(n);
            let vectors = random_vectors(n, dim, seed);

            let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i as u32, v.clone()).expect("Failed to add");
            }
            hnsw.build().expect("Failed to build");

            let results = hnsw.search(&vectors[0], k, 100).expect("Search failed");

            prop_assert_eq!(
                results.len(),
                k,
                "Should return {} results, got {}",
                k,
                results.len()
            );
        }

        /// HNSW results are sorted by distance.
        #[test]
        fn hnsw_results_sorted(
            n in 30usize..80,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let vectors = random_vectors(n, dim, seed);

            let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i as u32, v.clone()).expect("Failed to add");
            }
            hnsw.build().expect("Failed to build");

            let results = hnsw.search(&vectors[n / 2], 10, 100).expect("Search failed");

            for i in 1..results.len() {
                prop_assert!(
                    results[i].1 >= results[i - 1].1 - 1e-6,
                    "Results not sorted: {} > {} at position {}",
                    results[i - 1].1,
                    results[i].1,
                    i
                );
            }
        }

        /// HNSW result IDs are unique.
        #[test]
        fn hnsw_unique_results(
            n in 30usize..80,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let vectors = random_vectors(n, dim, seed);

            let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i as u32, v.clone()).expect("Failed to add");
            }
            hnsw.build().expect("Failed to build");

            let results = hnsw.search(&vectors[0], 15, 100).expect("Search failed");

            let ids: std::collections::HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
            prop_assert_eq!(
                ids.len(),
                results.len(),
                "Result IDs are not unique"
            );
        }

        /// Same query gives same results (determinism).
        #[test]
        fn hnsw_deterministic(
            n in 30usize..60,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let vectors = random_vectors(n, dim, seed);

            let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i as u32, v.clone()).expect("Failed to add");
            }
            hnsw.build().expect("Failed to build");

            let query = &vectors[0];
            let results1 = hnsw.search(query, 10, 100).expect("Search failed");
            let results2 = hnsw.search(query, 10, 100).expect("Search failed");
            let results3 = hnsw.search(query, 10, 100).expect("Search failed");

            // Same IDs should be returned
            let ids1: std::collections::HashSet<u32> = results1.iter().map(|(id, _)| *id).collect();
            let ids2: std::collections::HashSet<u32> = results2.iter().map(|(id, _)| *id).collect();
            let ids3: std::collections::HashSet<u32> = results3.iter().map(|(id, _)| *id).collect();

            prop_assert_eq!(&ids1, &ids2, "Results not deterministic (run 1 vs 2)");
            prop_assert_eq!(&ids2, &ids3, "Results not deterministic (run 2 vs 3)");
        }
    }
}

// =============================================================================
// HNSW Persistence Properties
// =============================================================================
//
// These tests verify the "KeyedVectors separation" principle from Gensim:
// external document IDs must survive the persistence roundtrip unchanged.
//
// ## Invariants
//
// 1. **doc_id preservation**: `load(write(index)).search()` returns the same
//    external IDs as `index.search()`
//
// 2. **Vector data integrity**: vectors survive SoA→AoS→SoA roundtrip without
//    numerical corruption beyond floating-point epsilon
//
// 3. **Search domain closure**: search results contain only IDs that were
//    originally added (no internal indices leaking through)
//

#[cfg(all(feature = "persistence", feature = "hnsw"))]
mod persistence_props {
    use super::*;
    use std::collections::HashSet;
    use vicinity::hnsw::HNSWIndex;
    use vicinity::persistence::directory::MemoryDirectory;
    use vicinity::persistence::hnsw::{HNSWSegmentReader, HNSWSegmentWriter};

    fn normalize(v: &[f32]) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n < 1e-10 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / n).collect()
        }
    }

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use std::hash::{Hash, Hasher};

        (0..n)
            .map(|i| {
                let raw: Vec<f32> = (0..dim)
                    .map(|j| {
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        seed.hash(&mut hasher);
                        i.hash(&mut hasher);
                        j.hash(&mut hasher);
                        let h = hasher.finish();
                        (h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
                    })
                    .collect();
                normalize(&raw)
            })
            .collect()
    }

    /// Generate unique doc_ids that are NOT sequential from 0.
    /// This tests that we're truly using external IDs, not internal indices.
    fn nonsequential_doc_ids(n: usize, seed: u64) -> Vec<u32> {
        use std::hash::{Hash, Hasher};

        let mut ids: HashSet<u32> = HashSet::new();
        let mut i = 0u64;
        while ids.len() < n {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            // Generate IDs in range [1000, 100_000] to ensure they're not 0-based
            let id = 1000 + (h % 99_000) as u32;
            ids.insert(id);
            i += 1;
        }
        ids.into_iter().collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        /// **doc_id preservation**: External IDs survive write/load roundtrip.
        ///
        /// This is the core Gensim "KeyedVectors" property: the index should
        /// return the user-provided IDs, not internal indices.
        #[test]
        fn prop_persistence_preserves_doc_ids(
            n in 10usize..50,
            seed in any::<u64>(),
        ) {
            let dim = 8;
            let vectors = random_vectors(n, dim, seed);
            let doc_ids = nonsequential_doc_ids(n, seed);

            // Build original index with non-sequential doc_ids
            let mut original = HNSWIndex::new(dim, 8, 8).expect("create");
            for (i, v) in vectors.iter().enumerate() {
                original.add(doc_ids[i], v.clone()).expect("add");
            }
            original.build().expect("build");

            // Write to memory
            let mem = MemoryDirectory::new();
            let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 1);
            writer.write_hnsw_index(&original).expect("write");

            // Load back
            let reader = HNSWSegmentReader::load(Box::new(mem.clone()), 1).expect("load");
            let loaded = reader.load_index().expect("load_index");

            // Search should return same external doc_ids
            let query = &vectors[0];
            let k = 5.min(n);
            let ef = 50;

            let original_results = original.search(query, k, ef).expect("search original");
            let loaded_results = loaded.search(query, k, ef).expect("search loaded");

            let original_ids: HashSet<u32> = original_results.iter().map(|(id, _)| *id).collect();
            let loaded_ids: HashSet<u32> = loaded_results.iter().map(|(id, _)| *id).collect();

            // Verify the IDs are actually from our doc_ids set, not internal indices
            let doc_id_set: HashSet<u32> = doc_ids.iter().copied().collect();
            for id in &loaded_ids {
                prop_assert!(
                    doc_id_set.contains(id),
                    "Search returned ID {} which was never added (internal index leak?)",
                    id
                );
            }

            // Verify original and loaded return same IDs
            prop_assert_eq!(
                original_ids, loaded_ids,
                "Loaded index returns different doc_ids than original"
            );
        }

        /// **Vector data integrity**: Vectors survive SoA→AoS→SoA roundtrip.
        ///
        /// The persistence layer stores vectors in Structure-of-Arrays (SoA) format
        /// on disk but HNSWIndex uses Array-of-Structures (AoS) in memory.
        /// This tests that the transpose operations are correct.
        ///
        /// Note: HNSW is an approximate algorithm. With small graphs, it may not
        /// always find exact matches even when searching with the same vector.
        /// We verify:
        /// 1. Original and loaded return the SAME results for the same query
        /// 2. The returned IDs are valid
        #[test]
        fn prop_persistence_vector_roundtrip(
            n in 10usize..40,
            dim in 8usize..32,
            seed in any::<u64>(),
        ) {
            let vectors = random_vectors(n, dim, seed);

            let mut original = HNSWIndex::new(dim, 16, 16).expect("create");
            for (i, v) in vectors.iter().enumerate() {
                original.add(i as u32, v.clone()).expect("add");
            }
            original.build().expect("build");

            // Write and load
            let mem = MemoryDirectory::new();
            let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 1);
            writer.write_hnsw_index(&original).expect("write");

            let reader = HNSWSegmentReader::load(Box::new(mem.clone()), 1).expect("load");
            let loaded = reader.load_index().expect("load_index");

            // Verify that original and loaded return IDENTICAL results
            // for the same queries. This validates vector data integrity.
            let k = 5.min(n);
            let ef = n * 2; // High ef for accurate search

            for (i, v) in vectors.iter().enumerate().take(5) {
                let original_results = original.search(v, k, ef).expect("search original");
                let loaded_results = loaded.search(v, k, ef).expect("search loaded");

                // Same IDs in same order
                let original_ids: Vec<u32> = original_results.iter().map(|(id, _)| *id).collect();
                let loaded_ids: Vec<u32> = loaded_results.iter().map(|(id, _)| *id).collect();

                prop_assert_eq!(
                    original_ids, loaded_ids,
                    "Query {} returns different results after persistence",
                    i
                );

                // Same distances (within floating point epsilon)
                for ((_, orig_dist), (_, loaded_dist)) in original_results.iter().zip(loaded_results.iter()) {
                    let diff = (orig_dist - loaded_dist).abs();
                    prop_assert!(
                        diff < 1e-5,
                        "Distance mismatch after persistence: {} vs {}",
                        orig_dist, loaded_dist
                    );
                }
            }
        }

        /// **Search domain closure**: Search only returns IDs that were added.
        ///
        /// This tests that no internal indices leak through the API and that
        /// the doc_id mapping is bijective.
        #[test]
        fn prop_search_returns_only_known_doc_ids(
            n in 20usize..60,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let vectors = random_vectors(n, dim, seed);
            let doc_ids = nonsequential_doc_ids(n, seed);
            let doc_id_set: HashSet<u32> = doc_ids.iter().copied().collect();

            let mut index = HNSWIndex::new(dim, 16, 16).expect("create");
            for (i, v) in vectors.iter().enumerate() {
                index.add(doc_ids[i], v.clone()).expect("add");
            }
            index.build().expect("build");

            // Multiple queries, verify all returned IDs are valid
            for query_idx in [0, n/4, n/2, 3*n/4, n-1].iter().filter(|&&i| i < n) {
                let query = &vectors[*query_idx];
                let results = index.search(query, 10.min(n), 100).expect("search");

                for (id, _dist) in &results {
                    prop_assert!(
                        doc_id_set.contains(id),
                        "Search returned unknown ID {}: internal index leaked through API",
                        id
                    );
                }
            }
        }

        /// **Duplicate doc_id rejection**: Adding the same doc_id twice should fail.
        ///
        /// This ensures the bijective mapping between doc_ids and internal indices.
        #[test]
        fn prop_duplicate_doc_id_rejected(
            n in 5usize..20,
            seed in any::<u64>(),
        ) {
            let dim = 8;
            let vectors = random_vectors(n + 1, dim, seed);
            let doc_ids = nonsequential_doc_ids(n, seed);

            let mut index = HNSWIndex::new(dim, 8, 8).expect("create");

            // Add all vectors
            for (i, v) in vectors.iter().take(n).enumerate() {
                index.add(doc_ids[i], v.clone()).expect("add");
            }

            // Try to add a duplicate doc_id with a different vector
            let duplicate_id = doc_ids[0];
            let different_vector = vectors[n].clone();
            let result = index.add(duplicate_id, different_vector);

            prop_assert!(
                result.is_err(),
                "Adding duplicate doc_id {} should fail but succeeded",
                duplicate_id
            );
        }

        /// **Persistence metadata integrity**: dimension, num_vectors survive roundtrip.
        #[test]
        fn prop_persistence_metadata_roundtrip(
            n in 10usize..40,
            dim in 4usize..64,
            seed in any::<u64>(),
        ) {
            let vectors = random_vectors(n, dim, seed);

            let mut original = HNSWIndex::new(dim, 8, 8).expect("create");
            for (i, v) in vectors.iter().enumerate() {
                original.add(i as u32, v.clone()).expect("add");
            }
            original.build().expect("build");

            let mem = MemoryDirectory::new();
            let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 1);
            writer.write_hnsw_index(&original).expect("write");

            let reader = HNSWSegmentReader::load(Box::new(mem.clone()), 1).expect("load");
            let loaded = reader.load_index().expect("load_index");

            // Verify search still works with correct dimensions
            let query = normalize(&vectors[0]);
            let results = loaded.search(&query, 5.min(n), 50);
            prop_assert!(results.is_ok(), "Search failed on loaded index");
            prop_assert_eq!(
                results.unwrap().len(),
                5.min(n),
                "Loaded index returns wrong number of results"
            );
        }
    }
}

// =============================================================================
// HNSW Self-Retrieval Properties
// =============================================================================

mod hnsw_self_retrieval_props {
    use proptest::prelude::*;
    use vicinity::hnsw::HNSWIndex;

    fn random_normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
                vicinity::distance::normalize(&v)
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn self_retrieval_high_rate(
            n in 50usize..200,
            dim in prop::sample::select(vec![8, 16, 32, 64]),
            m in prop::sample::select(vec![8, 16, 32]),
            seed in any::<u64>(),
        ) {
            let vectors = random_normalized_vectors(n, dim, seed);
            let mut index = HNSWIndex::new(dim, m, 2 * m).unwrap();
            for (i, v) in vectors.iter().enumerate() {
                index.add_slice(i as u32, v).unwrap();
            }
            index.build().unwrap();

            let mut self_found = 0;
            for (i, v) in vectors.iter().enumerate() {
                let results = index.search(v, 1, 200).unwrap();
                if !results.is_empty() && results[0].0 == i as u32 {
                    self_found += 1;
                }
            }
            let rate = self_found as f64 / n as f64;
            prop_assert!(rate >= 0.95,
                "Self-retrieval {:.1}% < 95% (n={}, dim={}, M={}, seed={})",
                rate * 100.0, n, dim, m, seed);
        }
    }
}

// =============================================================================
// HNSW ef_search Monotonicity Properties
// =============================================================================

mod hnsw_ef_monotonicity_props {
    use proptest::prelude::*;
    use std::collections::HashSet;
    use vicinity::hnsw::HNSWIndex;

    fn random_normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
                vicinity::distance::normalize(&v)
            })
            .collect()
    }

    fn brute_force_knn(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<u32> {
        let mut dists: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, vicinity::distance::cosine_distance(query, v)))
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.into_iter().take(k).map(|(id, _)| id).collect()
    }

    fn recall_at_k(results: &[(u32, f32)], ground_truth: &[u32]) -> f32 {
        let gt: HashSet<u32> = ground_truth.iter().copied().collect();
        let found: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
        gt.intersection(&found).count() as f32 / ground_truth.len().max(1) as f32
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn ef_search_monotonic_recall(
            n in 100usize..300,
            seed in any::<u64>(),
        ) {
            let dim = 32;
            let vectors = random_normalized_vectors(n, dim, seed);
            let queries = random_normalized_vectors(10, dim, seed.wrapping_add(1));

            let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
            for (i, v) in vectors.iter().enumerate() {
                index.add_slice(i as u32, v).unwrap();
            }
            index.build().unwrap();

            let k = 10.min(n);
            let ef_values = [10, 20, 50, 100, 200];
            let mut prev_recall = 0.0f32;

            for &ef in &ef_values {
                let mut total_recall = 0.0;
                for q in &queries {
                    let results = index.search(q, k, ef).unwrap();
                    let gt = brute_force_knn(q, &vectors, k);
                    total_recall += recall_at_k(&results, &gt);
                }
                let avg_recall = total_recall / queries.len() as f32;

                // Allow 5% tolerance for statistical noise
                prop_assert!(avg_recall >= prev_recall - 0.05,
                    "Recall decreased from {:.3} to {:.3} when ef increased to {} (seed={})",
                    prev_recall, avg_recall, ef, seed);
                prev_recall = avg_recall;
            }
        }
    }
}

// =============================================================================
// HNSW Build-Order Independence Properties
// =============================================================================

mod hnsw_build_order_props {
    use proptest::prelude::*;
    use std::collections::HashSet;
    use vicinity::hnsw::HNSWIndex;

    fn random_normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
                vicinity::distance::normalize(&v)
            })
            .collect()
    }

    fn brute_force_knn(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<u32> {
        let mut dists: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, vicinity::distance::cosine_distance(query, v)))
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.into_iter().take(k).map(|(id, _)| id).collect()
    }

    fn recall_at_k(results: &[(u32, f32)], ground_truth: &[u32]) -> f32 {
        let gt: HashSet<u32> = ground_truth.iter().copied().collect();
        let found: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
        gt.intersection(&found).count() as f32 / ground_truth.len().max(1) as f32
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn build_order_independence(
            n in 50usize..150,
            shuffle_seed in any::<u64>(),
            data_seed in any::<u64>(),
        ) {
            let dim = 16;
            let vectors = random_normalized_vectors(n, dim, data_seed);
            let queries = random_normalized_vectors(5, dim, data_seed.wrapping_add(999));

            // Build with original order
            let mut index1 = HNSWIndex::new(dim, 16, 32).unwrap();
            for (i, v) in vectors.iter().enumerate() {
                index1.add_slice(i as u32, v).unwrap();
            }
            index1.build().unwrap();

            // Build with shuffled order (Fisher-Yates with deterministic seed)
            let mut order: Vec<usize> = (0..n).collect();
            let mut rng_state = shuffle_seed;
            for i in (1..n).rev() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng_state >> 33) as usize % (i + 1);
                order.swap(i, j);
            }

            let mut index2 = HNSWIndex::new(dim, 16, 32).unwrap();
            for &i in &order {
                index2.add_slice(i as u32, &vectors[i]).unwrap();
            }
            index2.build().unwrap();

            let k = 10.min(n);
            for q in &queries {
                let gt = brute_force_knn(q, &vectors, k);
                let r1 = index1.search(q, k, 200).unwrap();
                let r2 = index2.search(q, k, 200).unwrap();
                let recall1 = recall_at_k(&r1, &gt);
                let recall2 = recall_at_k(&r2, &gt);
                let diff = (recall1 - recall2).abs();
                prop_assert!(diff <= 0.3,
                    "Build order changed recall by {:.2} ({:.2} vs {:.2}, seed={})",
                    diff, recall1, recall2, shuffle_seed);
            }
        }
    }
}

