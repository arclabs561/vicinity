//! Property-based tests for HNSW structural invariants.
//!
//! Tests that HNSW satisfies universal ANN properties (results sorted, unique,
//! bounded) plus HNSW-specific properties (layer distribution, ef monotonicity,
//! degree bounds, build-order independence, auto-normalize equivalence).

#![cfg(feature = "hnsw")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/mod.rs"]
mod common;
use common::*;

use proptest::prelude::*;
use vicinity::hnsw::HNSWIndex;

// =============================================================================
// Universal ANN Properties (for HNSW)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Search returns exactly k results for k <= n.
    #[test]
    fn returns_k_results(
        n in 20usize..100,
        k in 1usize..20,
        seed in any::<u64>(),
    ) {
        let dim = 16;
        let k = k.min(n);
        let vectors = random_normalized_vectors(n, dim, seed);

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

    /// Results are sorted by ascending distance.
    #[test]
    fn results_sorted(
        n in 30usize..80,
        seed in any::<u64>(),
    ) {
        let dim = 16;
        let vectors = random_normalized_vectors(n, dim, seed);

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

    /// Result IDs are unique.
    #[test]
    fn unique_results(
        n in 30usize..80,
        seed in any::<u64>(),
    ) {
        let dim = 16;
        let vectors = random_normalized_vectors(n, dim, seed);

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
    fn deterministic(
        n in 30usize..60,
        seed in any::<u64>(),
    ) {
        let dim = 16;
        let vectors = random_normalized_vectors(n, dim, seed);

        let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
        for (i, v) in vectors.iter().enumerate() {
            hnsw.add(i as u32, v.clone()).expect("Failed to add");
        }
        hnsw.build().expect("Failed to build");

        let query = &vectors[0];
        let results1 = hnsw.search(query, 10, 100).expect("Search failed");
        let results2 = hnsw.search(query, 10, 100).expect("Search failed");
        let results3 = hnsw.search(query, 10, 100).expect("Search failed");

        let ids1: std::collections::HashSet<u32> = results1.iter().map(|(id, _)| *id).collect();
        let ids2: std::collections::HashSet<u32> = results2.iter().map(|(id, _)| *id).collect();
        let ids3: std::collections::HashSet<u32> = results3.iter().map(|(id, _)| *id).collect();

        prop_assert_eq!(&ids1, &ids2, "Results not deterministic (run 1 vs 2)");
        prop_assert_eq!(&ids2, &ids3, "Results not deterministic (run 2 vs 3)");
    }
}

// =============================================================================
// Self-Retrieval Properties
// =============================================================================

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

// =============================================================================
// ef_search Monotonicity
// =============================================================================

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

            prop_assert!(avg_recall >= prev_recall - 0.05,
                "Recall decreased from {:.3} to {:.3} when ef increased to {} (seed={})",
                prev_recall, avg_recall, ef, seed);
            prev_recall = avg_recall;
        }
    }
}

// =============================================================================
// Build-Order Independence
// =============================================================================

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

        let mut index1 = HNSWIndex::new(dim, 16, 32).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index1.add_slice(i as u32, v).unwrap();
        }
        index1.build().unwrap();

        // Fisher-Yates with deterministic seed
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

// =============================================================================
// Degree Bounds
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn neighbor_degree_within_bound(
        n in 50usize..200,
        m in prop::sample::select(vec![4, 8, 16]),
        seed in any::<u64>(),
    ) {
        let m_max = 2 * m;
        let vectors = random_normalized_vectors(n, m_max.max(8), seed);

        let mut index = HNSWIndex::new(vectors[0].len(), m, m_max).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let (layer, node, degree) = index.max_node_degree();
        let bound = if layer == 0 { m_max } else { m };
        prop_assert!(degree <= bound,
            "Node {} on layer {} has degree {} > bound {} (m={}, m_max={})",
            node, layer, degree, bound, m, m_max);
    }
}

// =============================================================================
// Top-k Subset Property
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn top_k_subset_property(
        n in 50usize..200,
        seed in any::<u64>(),
    ) {
        let dim = 16;
        let vectors = random_normalized_vectors(n, dim, seed);

        let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let query = &vectors[0];
        let ef = n;
        let k_small = 5.min(n);
        let k_large = 15.min(n);

        let results_small = index.search(query, k_small, ef).unwrap();
        let results_large = index.search(query, k_large, ef).unwrap();

        let ids_small: std::collections::HashSet<u32> = results_small.iter().map(|(id, _)| *id).collect();
        let ids_large: std::collections::HashSet<u32> = results_large.iter().map(|(id, _)| *id).collect();

        let missing: Vec<u32> = ids_small.difference(&ids_large).copied().collect();
        prop_assert!(missing.is_empty(),
            "top-{} result(s) {:?} not in top-{} (seed={})",
            k_small, missing, k_large, seed);
    }
}

// =============================================================================
// Auto-Normalize Equivalence
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn auto_normalize_equivalent_to_manual_normalize(
        n in 30usize..80,
        seed in any::<u64>(),
    ) {
        let dim = 16;
        let vectors = random_vectors(n, dim, seed);

        let m = 8;
        let ef_c = 64;

        let mut hnsw_auto = HNSWIndex::builder(dim)
            .m(m)
            .ef_construction(ef_c)
            .auto_normalize(true)
            .build()
            .expect("builder failed");

        let mut hnsw_manual = HNSWIndex::builder(dim)
            .m(m)
            .ef_construction(ef_c)
            .build()
            .expect("builder failed");

        for (i, v) in vectors.iter().enumerate() {
            let unnorm: Vec<f32> = v.iter().map(|x| x * 2.0 + 0.5).collect();
            hnsw_auto.add_slice(i as u32, &unnorm).expect("add failed");
            let norm = normalize(&unnorm);
            hnsw_manual.add_slice(i as u32, &norm).expect("add failed");
        }
        hnsw_auto.build().expect("build failed");
        hnsw_manual.build().expect("build failed");

        let raw_q: Vec<f32> = vectors[0].iter().map(|x| x * 3.0 + 1.0).collect();
        let norm_q = normalize(&raw_q);

        let results_auto = hnsw_auto.search(&norm_q, 5, 100).expect("auto search failed");
        let results_manual = hnsw_manual.search(&norm_q, 5, 100).expect("manual search failed");

        let ids_auto: std::collections::HashSet<u32> =
            results_auto.iter().map(|(id, _)| *id).collect();
        let ids_manual: std::collections::HashSet<u32> =
            results_manual.iter().map(|(id, _)| *id).collect();

        if let (Some((id_auto, _)), Some((id_manual, _))) =
            (results_auto.first(), results_manual.first())
        {
            prop_assert_eq!(
                id_auto, id_manual,
                "auto_normalize top-1={} but manual top-1={} (seed={})",
                id_auto, id_manual, seed
            );
        }

        let overlap = ids_auto.intersection(&ids_manual).count();
        prop_assert!(
            overlap >= 3,
            "auto_normalize and manual normalize disagree: overlap={}/5 (seed={})",
            overlap, seed
        );
    }
}
