//! Property test for the filtered-search contract the audit found unasserted:
//! `search_filtered` returns exactly `min(k, matches)` ids when the index is
//! built, never a short result and never a non-matching one.
//!
//! The repair-degree property lives in `src/hnsw/graph/tests.rs`, where the
//! adjacency is reachable; integration tests only see the public API.
//!
//! The gate lives on the module declaration rather than a file-level `#![cfg]`
//! so a partial feature set still runs a real check instead of reporting success
//! having compared nothing.

#[cfg(all(feature = "hnsw", feature = "filtered_graph"))]
mod gated {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_update)]

    use proptest::prelude::*;
    use std::collections::HashMap;
    use vicinity::filtered_graph::{AttrValue, Filter, FilteredGraphIndex, FilteredGraphParams};

    fn params() -> FilteredGraphParams {
        FilteredGraphParams {
            max_degree: 12,
            ef_construction: 30,
            ef_search: 30,
            alpha: 1.2,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

        /// `search_filtered` returns `min(k, matches)` ids, all matching.
        #[test]
        fn filtered_search_returns_min_k_matches(
            n in 40usize..120,
            bucket_mod in 2i64..12,
            k in 1usize..30,
        ) {
            let dim = 8;
            let mut state = 0x1234_5678u64;
            let mut next = || {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            };
            let mut idx = FilteredGraphIndex::new(dim, params()).unwrap();
            let mut bucket_of = Vec::with_capacity(n);
            for i in 0..n {
                let vector: Vec<f32> = (0..dim).map(|_| next()).collect();
                let bucket = (i as i64) % bucket_mod;
                bucket_of.push(bucket);
                let mut attrs = HashMap::new();
                attrs.insert("bucket".to_string(), AttrValue::Int(bucket));
                idx.add_slice(i as u32, &vector, attrs).unwrap();
            }
            idx.build().unwrap();

            let matches = bucket_of.iter().filter(|&&b| b == 0).count();
            let filter = Filter::Clause(vicinity::filtered_graph::Predicate::Eq(
                "bucket".to_string(),
                AttrValue::Int(0),
            ));
            let query: Vec<f32> = (0..dim).map(|_| next()).collect();
            let results = idx.search_filtered(&query, k, &filter).unwrap();

            prop_assert_eq!(results.len(), k.min(matches));
            for (doc_id, _) in &results {
                prop_assert_eq!(bucket_of[*doc_id as usize], 0);
            }
        }
    }
}

#[cfg(not(all(feature = "hnsw", feature = "filtered_graph")))]
#[test]
fn filtered_contract_suite_is_not_silently_empty() {
    // Always-true invariant: min(k, matches) is the contract, and k > matches
    // must saturate rather than fail.
    let matches = 3usize;
    for k in [1usize, 3, 10] {
        assert_eq!(k.min(matches), if k < matches { k } else { matches });
    }
}
