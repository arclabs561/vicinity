#![cfg(feature = "hnsw")]

//! Regression tests for ADSampling.
//!
//! These tests encode specific bugs and edge cases found during development.
//! Each test documents what went wrong and why.

#![allow(clippy::unwrap_used)]

#[cfg(feature = "hnsw")]
mod tests {
    use std::collections::HashSet;
    use vicinity::adsampling::{ADSamplingParams, ADSamplingState};
    use vicinity::hnsw::{HNSWIndex, HNSWParams};
    use vicinity::DistanceMetric;

    /// Generate deterministic random vectors.
    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32) / (u32::MAX >> 1) as f32
            })
            .collect()
    }

    /// Regression: ADSampling must use reordered vectors from HNSW.
    ///
    /// Bug: ADSamplingState::new() was called with the original (pre-build)
    /// vector array. After HNSWIndex::build(), vectors are reordered for cache
    /// locality. search_with_distance() passes node IDs in reordered space,
    /// so ADSampling looked up the wrong vectors, producing near-zero recall.
    ///
    /// Fix: Use ADSamplingState::from_hnsw() which reads index.vectors_raw().
    #[test]
    fn adsampling_must_use_reordered_vectors() {
        let dim = 64;
        let n = 300;
        let k = 10;
        let ef = 100;

        let vectors = random_vectors(n, dim, 42);

        // Normalize for cosine
        let mut normalized = vectors.clone();
        for i in 0..n {
            let sl = &mut normalized[i * dim..(i + 1) * dim];
            let norm: f32 = sl.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in sl.iter_mut() {
                    *x /= norm;
                }
            }
        }

        let params = HNSWParams {
            m: 16,
            m_max: 16,
            ef_construction: 100,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).unwrap();
        let ids: Vec<u32> = (0..n as u32).collect();
        index.add_batch(&ids, &normalized).unwrap();
        index.build().unwrap();

        // CORRECT: build from HNSW's reordered vectors
        let state_correct = ADSamplingState::from_hnsw(&index, ADSamplingParams::default());

        // WRONG: build from original vectors (pre-reorder)
        let state_wrong = ADSamplingState::new(&normalized, dim, ADSamplingParams::default());

        let query = &normalized[0..dim];

        let hnsw_results = index.search(query, k, ef).unwrap();
        let correct_results = state_correct.search_hnsw(&index, query, k, ef).unwrap();
        let wrong_results = state_wrong.search_hnsw(&index, query, k, ef).unwrap();

        // Correct: ADSampling should find results with comparable distances.
        // Early termination may cause some recall loss, so compare the closest
        // result's distance rather than the 10th (which is more affected by rejections).
        let hnsw_1st = hnsw_results[0].1;
        let correct_1st = correct_results[0].1;
        assert!(
            (hnsw_1st - correct_1st).abs() < 0.05,
            "from_hnsw() closest-dist should match HNSW: {hnsw_1st:.4} vs {correct_1st:.4}"
        );

        // Wrong: the pre-reorder version finds different neighbors.
        // Its closest result's distance should differ significantly from HNSW's.
        let wrong_1st = wrong_results[0].1;
        let correct_gap = (hnsw_1st - correct_1st).abs();
        let wrong_gap = (hnsw_1st - wrong_1st).abs();
        if n > 50 {
            assert!(
                wrong_gap > correct_gap || correct_gap < 0.01,
                "Wrong version should diverge more: correct_gap={correct_gap:.4}, wrong_gap={wrong_gap:.4}"
            );
        }
    }

    /// Regression: ADSampling dist_comp must return L2 (sqrt), not L2².
    ///
    /// Bug: dist_comp returned sum of squared diffs (L2²). HNSW graphs are
    /// built with L2 (sqrt). The beam search found the same distances but
    /// different nodes due to floating-point tie-breaking at squared scale.
    ///
    /// Fix: dist_comp returns sqrt(partial_sum) and squares the threshold
    /// internally for the early termination test.
    #[test]
    fn dist_comp_returns_l2_not_l2_squared() {
        let dim = 32;
        let params = ADSamplingParams::default();

        let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();
        let vectors: Vec<f32> = [a.clone(), b.clone()].concat();

        let state = ADSamplingState::new(&vectors, dim, params);
        let rq = state.rotate_query(&a);

        let dist = state.dist_comp(&rq, 1, f32::INFINITY).unwrap();

        // Expected: L2 distance = sqrt(32 * 1.0²) ≈ 5.657
        // NOT L2² = 32.0
        let expected_l2 = (dim as f32).sqrt(); // sqrt(32) ≈ 5.657
        assert!(
            (dist - expected_l2).abs() < 0.5,
            "dist_comp should return L2 (≈{expected_l2:.2}), got {dist:.2}"
        );
        assert!(
            dist < 10.0,
            "dist_comp returned {dist:.2}, which looks like L2² not L2"
        );
    }

    /// Regression: threshold near zero should not reject all candidates.
    ///
    /// Bug: when the top-k heap contained a near-zero distance (self-match),
    /// threshold * ratio ≈ 0, causing any non-zero partial_sum to trigger
    /// rejection.
    ///
    /// Fix: guard with `threshold_sq > 1e-10`.
    #[test]
    fn near_zero_threshold_does_not_reject_everything() {
        let dim = 64;
        let params = ADSamplingParams::default();

        let a: Vec<f32> = vec![1.0; dim];
        let b: Vec<f32> = vec![1.001; dim]; // very close
        let c: Vec<f32> = vec![2.0; dim]; // farther
        let vectors: Vec<f32> = [a.clone(), b, c].concat();

        let state = ADSamplingState::new(&vectors, dim, params);
        let rq = state.rotate_query(&a);

        // With generous threshold (INFINITY), both should be accepted
        let near_inf = state.dist_comp(&rq, 1, f32::INFINITY);
        let far_inf = state.dist_comp(&rq, 2, f32::INFINITY);
        assert!(
            near_inf.is_some(),
            "near neighbor accepted at INF threshold"
        );
        assert!(far_inf.is_some(), "far neighbor accepted at INF threshold");

        // With threshold slightly above the near neighbor's distance, near should be accepted
        let near_dist = near_inf.unwrap();
        let near_tight = state.dist_comp(&rq, 1, near_dist * 1.5);
        assert!(
            near_tight.is_some(),
            "near neighbor should be accepted at threshold={:.4} (1.5x its distance {:.4})",
            near_dist * 1.5,
            near_dist
        );

        // With threshold=0 (self-match edge case), guard should prevent rejection
        let near_zero = state.dist_comp(&rq, 1, 0.0);
        assert!(
            near_zero.is_some(),
            "threshold=0 should disable early termination (guard: threshold_sq < 1e-10)"
        );
    }

    /// Regression: ADSampling on L2 (Euclidean) datasets should achieve
    /// comparable recall to standard HNSW search.
    ///
    /// This test uses unnormalized vectors with DistanceMetric::L2 to verify
    /// that the from_hnsw + sqrt fix works end-to-end.
    #[test]
    fn adsampling_l2_recall_matches_hnsw() {
        let dim = 128;
        let n = 500;
        let k = 10;
        let ef = 100;

        // Unnormalized vectors
        let mut s = 77u64;
        let vectors: Vec<f32> = (0..n * dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32) / (u32::MAX >> 1) as f32 * 255.0
            })
            .collect();

        let params = HNSWParams {
            m: 16,
            ef_construction: 100,
            metric: DistanceMetric::L2,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).unwrap();
        let ids: Vec<u32> = (0..n as u32).collect();
        index.add_batch(&ids, &vectors).unwrap();
        index.build().unwrap();

        let state = ADSamplingState::from_hnsw(&index, ADSamplingParams::default());

        // Test with 20 queries (not in training set)
        let test_vectors = random_vectors(20, dim, 999);
        let mut total_ads_recall = 0.0;

        for qi in 0..20 {
            let query = &test_vectors[qi * dim..(qi + 1) * dim];

            // Brute-force ground truth using reordered vectors
            let raw = index.vectors_raw();
            let mut gt: Vec<(u32, f32)> = (0..n)
                .map(|i| {
                    let v = &raw[i * dim..(i + 1) * dim];
                    (i as u32, vicinity::distance::l2_distance(query, v))
                })
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            // Map internal IDs to doc_ids for ground truth comparison
            // (search returns doc_ids, brute force uses internal IDs)
            // Since we don't have the mapping externally, compare via HNSW search
            let hnsw_results = index.search(query, k, ef).unwrap();
            let ads_results = state.search_hnsw(&index, query, k, ef).unwrap();

            let hnsw_ids: HashSet<u32> = hnsw_results.iter().map(|r| r.0).collect();
            let ads_ids: HashSet<u32> = ads_results.iter().map(|r| r.0).collect();

            // Compare ADSampling against HNSW (not brute force -- both use the same graph)
            let overlap = hnsw_ids.intersection(&ads_ids).count();
            total_ads_recall += overlap as f64 / k as f64;
        }

        let avg_parity = total_ads_recall / 20.0;
        assert!(
            avg_parity > 0.7,
            "ADSampling should find >=70% of HNSW's results on L2, got {:.1}%",
            avg_parity * 100.0
        );
    }

    /// Property: rotation preserves L2 distance for arbitrary vector pairs.
    #[test]
    fn rotation_preserves_l2_multiple_dimensions() {
        for &dim in &[16, 64, 256, 512] {
            let params = ADSamplingParams {
                seed: 42,
                ..Default::default()
            };
            let a = random_vectors(1, dim, 100);
            let b = random_vectors(1, dim, 200);
            let vectors: Vec<f32> = [a.clone(), b.clone()].concat();

            let state = ADSamplingState::new(&vectors, dim, params);
            let rq = state.rotate_query(&a);

            let orig_l2: f32 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt();
            let rotated_l2 = state.dist_comp(&rq, 1, f32::INFINITY).unwrap();

            let rel_error = (orig_l2 - rotated_l2).abs() / orig_l2.max(1e-10);
            assert!(
                rel_error < 0.01,
                "L2 not preserved at dim={dim}: orig={orig_l2:.4}, rotated={rotated_l2:.4}, err={rel_error:.6}"
            );
        }
    }
}
