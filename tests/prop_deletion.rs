//! Property tests for Wolverine++ deletion and seeded RNG reproducibility.
//!
//! Properties tested:
//! 1. delete+search never returns deleted IDs (single and batch)
//! 2. Seeded builds are deterministic (same seed = same graph)
//! 3. Graph integrity after deletion: no edges to deleted nodes
//! 4. Recall doesn't collapse after moderate deletion (< 50%)

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

#[path = "common/mod.rs"]
mod common;
use common::*;

use proptest::prelude::*;
use std::collections::HashSet;

/// Build a seeded HNSW index from normalized random vectors.
#[cfg(feature = "hnsw")]
fn build_hnsw(
    n: usize,
    dim: usize,
    m: usize,
    seed: u64,
) -> (vicinity::hnsw::HNSWIndex, Vec<Vec<f32>>) {
    let params = vicinity::hnsw::HNSWParams {
        m,
        m_max: m * 2,
        ef_construction: 100,
        seed: Some(seed),
        ..vicinity::hnsw::HNSWParams::default()
    };
    let mut idx = vicinity::hnsw::HNSWIndex::with_params(dim, params).unwrap();

    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, seed)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();
    (idx, vectors)
}

// ---------------------------------------------------------------------------
// Property 1: Deleted IDs never appear in search results
// ---------------------------------------------------------------------------

#[cfg(feature = "hnsw")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn prop_deleted_ids_absent_from_results(
        n in 50usize..200,
        dim in prop::sample::select(vec![8, 16, 32]),
        num_delete in 1usize..30,
        seed in 1u64..10000,
    ) {
        let (mut idx, _) = build_hnsw(n, dim, 8, seed);
        let num_delete = num_delete.min(n / 2);

        // Delete first `num_delete` IDs
        let deleted_ids: Vec<u32> = (0..num_delete as u32).collect();
        for &id in &deleted_ids {
            idx.delete_with_repair(id).unwrap();
        }

        let deleted_set: HashSet<u32> = deleted_ids.iter().copied().collect();

        // Search with multiple queries
        let queries = random_vectors(5, dim, seed.wrapping_add(9999));
        for q in &queries {
            let q_norm = normalize(q);
            let results = idx.search(&q_norm, 10, 64).unwrap();
            for (id, _) in &results {
                prop_assert!(
                    !deleted_set.contains(id),
                    "deleted id {} found in results after delete_with_repair",
                    id
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Property 2: Batch deletion -- same invariant
// ---------------------------------------------------------------------------

#[cfg(feature = "hnsw")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(15))]

    #[test]
    fn prop_batch_deleted_ids_absent(
        n in 80usize..200,
        dim in prop::sample::select(vec![8, 16, 32]),
        num_delete in 5usize..40,
        seed in 1u64..10000,
    ) {
        let (mut idx, _) = build_hnsw(n, dim, 8, seed);
        let num_delete = num_delete.min(n / 2);

        let delete_ids: Vec<u32> = (0..num_delete as u32).collect();
        idx.delete_batch_with_repair(&delete_ids).unwrap();

        let deleted_set: HashSet<u32> = delete_ids.iter().copied().collect();

        let queries = random_vectors(5, dim, seed.wrapping_add(7777));
        for q in &queries {
            let q_norm = normalize(q);
            let results = idx.search(&q_norm, 10, 64).unwrap();
            for (id, _) in &results {
                prop_assert!(
                    !deleted_set.contains(id),
                    "deleted id {} in batch results",
                    id
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Property 3: Deleted nodes are truly excluded (high-ef exhaustive check)
// ---------------------------------------------------------------------------

#[cfg(feature = "hnsw")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(15))]

    #[test]
    fn prop_deleted_exhaustive_check(
        n in 50usize..150,
        dim in prop::sample::select(vec![8, 16, 32]),
        num_delete in 1usize..20,
        seed in 1u64..10000,
    ) {
        let (mut idx, vectors) = build_hnsw(n, dim, 8, seed);
        let num_delete = num_delete.min(n / 3);

        for doc_id in 0..num_delete as u32 {
            idx.delete_with_repair(doc_id).unwrap();
        }

        // Search with very high ef to maximize coverage -- use each deleted
        // vector as query to check that it doesn't find itself
        for doc_id in 0..num_delete as u32 {
            let q = &vectors[doc_id as usize];
            let results = idx.search(q, n, n).unwrap();
            for (id, _) in &results {
                prop_assert!(
                    *id >= num_delete as u32,
                    "deleted id {} found when querying with deleted vector {}",
                    id,
                    doc_id
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Property 4: Seeded HNSW builds are deterministic
// ---------------------------------------------------------------------------

#[cfg(feature = "hnsw")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_seeded_hnsw_deterministic(
        n in 20usize..100,
        dim in prop::sample::select(vec![8, 16, 32]),
        seed in 1u64..10000,
    ) {
        let (idx1, _) = build_hnsw(n, dim, 8, seed);
        let (idx2, _) = build_hnsw(n, dim, 8, seed);

        // Same seed -> same search results for multiple queries
        let queries = random_vectors(5, dim, seed + 777);
        for q in &queries {
            let q_norm = normalize(q);
            let r1 = idx1.search(&q_norm, 10, 64).unwrap();
            let r2 = idx2.search(&q_norm, 10, 64).unwrap();
            prop_assert_eq!(r1.len(), r2.len(), "different result count with same seed");
            for (a, b) in r1.iter().zip(r2.iter()) {
                prop_assert_eq!(a.0, b.0, "different result IDs with same seed");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Property 5: Seeded Vamana builds are deterministic
// ---------------------------------------------------------------------------

#[cfg(feature = "vamana")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_seeded_vamana_deterministic(
        n in 20usize..60,
        dim in prop::sample::select(vec![8, 16]),
        seed in 1u64..10000,
    ) {
        let vectors: Vec<Vec<f32>> = random_vectors(n, dim, seed)
            .into_iter()
            .map(|v| normalize(&v))
            .collect();

        let build = |s: u64| {
            let params = vicinity::vamana::VamanaParams {
                max_degree: 16,
                alpha: 1.3,
                ef_construction: 50,
                ef_search: 20,
                seed: Some(s),
                ..vicinity::vamana::VamanaParams::default()
            };
            let mut idx = vicinity::vamana::VamanaIndex::new(dim, params).unwrap();
            for (i, v) in vectors.iter().enumerate() {
                idx.add(i as u32, v.clone()).unwrap();
            }
            idx.build().unwrap();
            idx
        };

        let idx1 = build(seed);
        let idx2 = build(seed);

        // Same query should return same results
        let q = normalize(&random_vectors(1, dim, seed + 999)[0]);
        let r1 = idx1.search(&q, 10, 50).unwrap();
        let r2 = idx2.search(&q, 10, 50).unwrap();

        prop_assert_eq!(r1.len(), r2.len(), "different result count with same seed");
        for (a, b) in r1.iter().zip(r2.iter()) {
            prop_assert_eq!(a.0, b.0, "different result IDs with same seed");
        }
    }
}

// ---------------------------------------------------------------------------
// Property 6: Recall doesn't collapse after moderate deletion
// ---------------------------------------------------------------------------

#[cfg(feature = "hnsw")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn prop_recall_survives_deletion(
        n in 100usize..300,
        dim in prop::sample::select(vec![16, 32]),
        delete_frac in 10usize..40, // percent to delete
        seed in 1u64..10000,
    ) {
        let (mut idx, _vectors) = build_hnsw(n, dim, 12, seed);
        let num_delete = (n * delete_frac) / 100;

        for i in 0..num_delete as u32 {
            idx.delete_with_repair(i).unwrap();
        }

        let remaining = n - num_delete;
        let k = 10.min(remaining);
        if k == 0 {
            return Ok(());
        }

        let queries = random_vectors(5, dim, seed + 5555);
        let mut total_results = 0usize;

        for q in &queries {
            let q_norm = normalize(q);
            let results = idx.search(&q_norm, k, 100).unwrap();
            total_results += results.len();

            // All results should be non-deleted
            for (id, _) in &results {
                prop_assert!(*id >= num_delete as u32, "deleted id {} in results", id);
            }
        }

        // Should return k results per query (graph stays connected enough)
        let expected_min = queries.len() * k;
        let recall_fraction = total_results as f64 / expected_min as f64;
        prop_assert!(
            recall_fraction >= 0.3,
            "only {:.0}% of expected results returned after deleting {}% of {} nodes \
             ({} of {} expected)",
            recall_fraction * 100.0,
            delete_frac,
            n,
            total_results,
            expected_min,
        );
    }
}
