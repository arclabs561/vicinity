//! Property-based tests for IVF-PQ.
//!
//! Verifies domain closure (search returns only inserted doc_ids) and
//! result count bounds.

#![cfg(feature = "ivf_pq")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_update)]

#[path = "common/mod.rs"]
mod common;
use common::normalize;

use proptest::prelude::*;
use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

fn default_params() -> IVFPQParams {
    IVFPQParams {
        num_clusters: 4,
        nprobe: 4,
        num_codebooks: 2,
        codebook_size: 16,
        use_opq: false,
        ..IVFPQParams::default()
    }
}

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(20))]

    /// Search results contain only doc_ids that were added.
    #[test]
    fn search_results_contain_only_inserted_doc_ids(
        base_id in 100u32..200u32,
        n in 20usize..40usize,
        seed in 0u64..1000u64,
    ) {
        let dim = 8usize;
        let params = default_params();
        let mut index = IVFPQIndex::new(dim, params).unwrap();

        let mut rng = seed;
        let mut lcg = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let inserted_ids: std::collections::HashSet<u32> =
            (base_id..base_id + n as u32).collect();

        for doc_id in base_id..base_id + n as u32 {
            let v: Vec<f32> = (0..dim).map(|_| lcg()).collect();
            let v = normalize(&v);
            index.add(doc_id, v).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| lcg()).collect();
        let query = normalize(&query);

        let results = index.search(&query, 5);
        prop_assert!(results.is_ok(), "search failed: {:?}", results.err());
        let results = results.unwrap();
        for (returned_id, _dist) in &results {
            prop_assert!(
                inserted_ids.contains(returned_id),
                "search returned id={} which was not inserted (range {}..{})",
                returned_id, base_id, base_id + n as u32,
            );
        }
    }

    /// Result count is bounded by k and by n.
    #[test]
    fn search_result_count_bounded(
        n in 20usize..40usize,
        k in 1usize..10usize,
        seed in 0u64..1000u64,
    ) {
        let dim = 8usize;
        let params = default_params();
        let mut index = IVFPQIndex::new(dim, params).unwrap();

        let mut rng = seed;
        let mut lcg = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        for i in 0u32..n as u32 {
            let v: Vec<f32> = (0..dim).map(|_| lcg()).collect();
            let v = normalize(&v);
            index.add(i, v).unwrap();
        }
        index.build().unwrap();

        let query: Vec<f32> = (0..dim).map(|_| lcg()).collect();
        let query = normalize(&query);

        let results = index.search(&query, k);
        prop_assert!(results.is_ok(), "search failed: {:?}", results.err());
        let results = results.unwrap();
        prop_assert!(
            results.len() <= k,
            "returned {} results but k={}",
            results.len(), k
        );
    }
}
