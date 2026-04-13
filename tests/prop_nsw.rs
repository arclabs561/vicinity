//! Property-based tests for NSW.
//!
//! Verifies domain closure, sorted results, and result count bounds.

#![cfg(feature = "nsw")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/mod.rs"]
mod common;
use common::normalize;

use proptest::prelude::*;
use vicinity::nsw::{NSWIndex, NSWParams};

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(30))]

    /// Results contain only IDs that were inserted (domain closure).
    #[test]
    fn search_domain_closure(
        base_id in 50u32..100u32,
        n in 20usize..60usize,
        seed in 0u64..500u64,
    ) {
        let dim = 8usize;
        let params = NSWParams { m: 8, m_max: 16, ef_search: 50, ef_construction: 20 };
        let mut index = NSWIndex::with_params(dim, params).unwrap();

        let mut rng = seed;
        let mut lcg = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let inserted: std::collections::HashSet<u32> =
            (base_id..base_id + n as u32).collect();

        for doc_id in base_id..base_id + n as u32 {
            let v: Vec<f32> = (0..dim).map(|_| lcg()).collect();
            index.add(doc_id, normalize(&v)).unwrap();
        }
        index.build().unwrap();

        let q: Vec<f32> = (0..dim).map(|_| lcg()).collect();
        let results = index.search(&normalize(&q), 5, 50).unwrap();

        for (returned_id, _dist) in &results {
            prop_assert!(
                inserted.contains(returned_id),
                "returned id {} not in inserted range {}..{}",
                returned_id, base_id, base_id + n as u32
            );
        }
    }

    /// Results are sorted by ascending distance.
    #[test]
    fn results_sorted_ascending(
        n in 20usize..60usize,
        seed in 0u64..500u64,
    ) {
        let dim = 8usize;
        let params = NSWParams { m: 8, m_max: 16, ef_search: 50, ef_construction: 20 };
        let mut index = NSWIndex::with_params(dim, params).unwrap();

        let mut rng = seed;
        let mut lcg = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        for i in 0u32..n as u32 {
            let v: Vec<f32> = (0..dim).map(|_| lcg()).collect();
            index.add(i, normalize(&v)).unwrap();
        }
        index.build().unwrap();

        let q: Vec<f32> = (0..dim).map(|_| lcg()).collect();
        let results = index.search(&normalize(&q), 10, 50).unwrap();

        for window in results.windows(2) {
            prop_assert!(
                window[0].1 <= window[1].1 + 1e-6,
                "results not sorted: {} > {}",
                window[0].1, window[1].1
            );
        }
    }

    /// Result count is at most k.
    #[test]
    fn result_count_at_most_k(
        n in 20usize..60usize,
        k in 1usize..15usize,
        seed in 0u64..500u64,
    ) {
        let dim = 8usize;
        let params = NSWParams { m: 8, m_max: 16, ef_search: 50, ef_construction: 20 };
        let mut index = NSWIndex::with_params(dim, params).unwrap();

        let mut rng = seed;
        let mut lcg = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        for i in 0u32..n as u32 {
            let v: Vec<f32> = (0..dim).map(|_| lcg()).collect();
            index.add(i, normalize(&v)).unwrap();
        }
        index.build().unwrap();

        let q: Vec<f32> = (0..dim).map(|_| lcg()).collect();
        let results = index.search(&normalize(&q), k, 50).unwrap();

        prop_assert!(
            results.len() <= k,
            "returned {} results but k={}",
            results.len(), k
        );
    }
}
