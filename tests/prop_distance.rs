//! Property-based tests for distance metrics and recall computation.
//!
//! Verifies metric space axioms (non-negativity, symmetry, identity, triangle
//! inequality) and recall utility correctness.

#[path = "common/mod.rs"]
mod common;

use proptest::prelude::*;

// =============================================================================
// Distance Metric Properties
// =============================================================================

mod distance_props {
    use super::*;
    use vicinity::distance::{cosine_distance_normalized, normalize};

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

        /// cosine_distance_normalized is symmetric on unit vectors.
        #[test]
        fn cosine_distance_normalized_symmetric(
            a in arb_vector(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-4)),
            b in arb_vector(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-4)),
        ) {
            let a_n = normalize(&a);
            let b_n = normalize(&b);
            let d_ab = cosine_distance_normalized(&a_n, &b_n);
            let d_ba = cosine_distance_normalized(&b_n, &a_n);
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-5,
                "cosine_distance_normalized not symmetric: {} vs {}",
                d_ab, d_ba
            );
        }

        /// cosine_distance_normalized is zero for identical unit vectors.
        #[test]
        fn cosine_distance_normalized_self_zero(
            a in arb_vector(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-4)),
        ) {
            let a_n = normalize(&a);
            let d = cosine_distance_normalized(&a_n, &a_n);
            prop_assert!(d.abs() < 1e-5, "cosine_distance_normalized(a, a) = {}", d);
        }

        /// cosine_distance_normalized is in [0, 2] for unit vectors.
        #[test]
        fn cosine_distance_normalized_bounded(
            a in arb_vector(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-4)),
            b in arb_vector(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-4)),
        ) {
            let a_n = normalize(&a);
            let b_n = normalize(&b);
            let d = cosine_distance_normalized(&a_n, &b_n);
            prop_assert!(
                (-0.01..=2.01).contains(&d),
                "cosine_distance_normalized out of [0,2]: {}",
                d
            );
        }
    }
}

// =============================================================================
// Recall Computation Properties
// =============================================================================

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

// =============================================================================
// Additional Metric Space Properties
// =============================================================================

mod metric_space_props {
    use super::common::normalize;
    use super::*;

    fn arb_vec(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-10.0f32..10.0, dim)
    }

    fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

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

        #[test]
        fn l2_identity_of_indiscernibles(v in arb_vec(32)) {
            let dist = l2_squared(&v, &v);
            prop_assert!(dist.abs() < 1e-6, "L2(v, v) should be 0, got {}", dist);
        }

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
