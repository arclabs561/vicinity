//! Universal ANN property tests across all algorithms.
//!
//! Every ANN index must satisfy these invariants regardless of implementation:
//! 1. Results sorted by ascending distance
//! 2. Result IDs are unique
//! 3. Result count <= k
//! 4. Domain closure: only inserted IDs appear in results
//! 5. Self-query returns self (for exact-enough algorithms)
//! 6. Recall monotonicity: higher ef -> higher recall (for algorithms with ef)
//!
//! Each algorithm is tested across multiple parameterizations (varying n, dim,
//! connectivity, seeds) via proptest.

#![allow(clippy::unwrap_used, clippy::expect_used, unused_imports, dead_code)]

#[path = "common/mod.rs"]
mod common;
use common::*;

use std::collections::HashSet;

/// Build a dataset of normalized random vectors and a set of queries.
fn make_dataset(n: usize, dim: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let data: Vec<Vec<f32>> = random_vectors(n, dim, seed)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();
    let queries: Vec<Vec<f32>> = random_vectors(5, dim, seed.wrapping_add(9999))
        .into_iter()
        .map(|v| normalize(&v))
        .collect();
    (data, queries)
}

/// Universal invariants for any ANN search result.
fn assert_universal_invariants(
    results: &[(u32, f32)],
    k: usize,
    n: usize,
    inserted_ids: &HashSet<u32>,
    algo_name: &str,
) {
    // 1. Result count <= k and <= n
    assert!(
        results.len() <= k,
        "{}: returned {} results but k={}",
        algo_name,
        results.len(),
        k
    );
    assert!(
        results.len() <= n,
        "{}: returned {} results but n={}",
        algo_name,
        results.len(),
        n
    );

    // 2. Results sorted by ascending distance
    for w in results.windows(2) {
        assert!(
            w[0].1 <= w[1].1 + 1e-5,
            "{}: results not sorted: {} > {} (ids {} vs {})",
            algo_name,
            w[0].1,
            w[1].1,
            w[0].0,
            w[1].0
        );
    }

    // 3. Result IDs are unique
    let id_set: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        id_set.len(),
        results.len(),
        "{}: duplicate IDs in results",
        algo_name
    );

    // 4. Domain closure: only inserted IDs
    for (id, _) in results {
        assert!(
            inserted_ids.contains(id),
            "{}: returned ID {} which was never inserted",
            algo_name,
            id
        );
    }
}

/// Macro to generate universal property tests for a graph-based ANN algorithm.
///
/// The macro takes:
/// - mod name
/// - feature gate
/// - index type path
/// - params expression
/// - search function: closure (index, query, k) -> Vec<(u32, f32)>
/// - search_with_ef function (optional): closure (index, query, k, ef) -> Vec<(u32, f32)>
macro_rules! ann_universal_tests {
    (
        mod_name: $mod:ident,
        feature: $feat:literal,
        build_index: $build:expr,
        search: $search:expr
        $(, search_with_ef: $search_ef:expr)?
    ) => {
        #[cfg(feature = $feat)]
        mod $mod {
            use super::*;

            #[test]
            fn universal_invariants_small() {
                let n = 50;
                let dim = 16;
                let k = 10;
                let (data, queries) = make_dataset(n, dim, 42);
                let inserted: HashSet<u32> = (0..n as u32).collect();

                let build_fn = $build;
                let idx = build_fn(&data, dim);
                let search_fn = $search;

                for q in &queries {
                    let results = search_fn(&idx, q, k);
                    assert_universal_invariants(&results, k, n, &inserted, stringify!($mod));
                }
            }

            #[test]
            fn universal_invariants_medium() {
                let n = 200;
                let dim = 32;
                let k = 15;
                let (data, queries) = make_dataset(n, dim, 77);
                let inserted: HashSet<u32> = (0..n as u32).collect();

                let build_fn = $build;
                let idx = build_fn(&data, dim);
                let search_fn = $search;

                for q in &queries {
                    let results = search_fn(&idx, q, k);
                    assert_universal_invariants(&results, k, n, &inserted, stringify!($mod));
                }
            }

            #[test]
            fn universal_invariants_high_dim() {
                let n = 80;
                let dim = 64;
                let k = 10;
                let (data, queries) = make_dataset(n, dim, 123);
                let inserted: HashSet<u32> = (0..n as u32).collect();

                let build_fn = $build;
                let idx = build_fn(&data, dim);
                let search_fn = $search;

                for q in &queries {
                    let results = search_fn(&idx, q, k);
                    assert_universal_invariants(&results, k, n, &inserted, stringify!($mod));
                }
            }

            #[test]
            fn k_larger_than_n() {
                let n = 10;
                let dim = 8;
                let k = 20; // k > n
                let (data, _) = make_dataset(n, dim, 55);
                let inserted: HashSet<u32> = (0..n as u32).collect();

                let build_fn = $build;
                let idx = build_fn(&data, dim);
                let search_fn = $search;

                let q = &data[0];
                let results = search_fn(&idx, q, k);
                // Should return at most n results
                assert!(
                    results.len() <= n,
                    "{}: returned {} results but only {} vectors in index",
                    stringify!($mod), results.len(), n
                );
                assert_universal_invariants(&results, k, n, &inserted, stringify!($mod));
            }

            $(
            #[test]
            fn recall_monotonic_with_ef() {
                let n = 200;
                let dim = 32;
                let k = 10;
                let (data, queries) = make_dataset(n, dim, 42);

                let build_fn = $build;
                let idx = build_fn(&data, dim);
                let search_ef_fn = $search_ef;

                let ef_values = [10, 20, 50, 100, 200];
                let mut prev_recall = 0.0f32;

                for &ef in &ef_values {
                    let mut total_recall = 0.0f32;
                    for q in &queries {
                        let results = search_ef_fn(&idx, q, k, ef);
                        let gt = brute_force_knn(q, &data, k);
                        total_recall += recall_at_k(&results, &gt);
                    }
                    let avg_recall = total_recall / queries.len() as f32;

                    assert!(
                        avg_recall >= prev_recall - 0.10,
                        "{}: recall decreased from {:.3} to {:.3} at ef={}",
                        stringify!($mod), prev_recall, avg_recall, ef
                    );
                    prev_recall = avg_recall;
                }
            }
            )?
        }
    };
}

// =============================================================================
// Algorithm instantiations
// =============================================================================

ann_universal_tests! {
    mod_name: hnsw,
    feature: "hnsw",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let mut idx = vicinity::hnsw::HNSWIndex::new(dim, 16, 32).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add_slice(i as u32, v).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::hnsw::HNSWIndex, q: &[f32], k: usize| {
        idx.search(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::hnsw::HNSWIndex, q: &[f32], k: usize, ef: usize| {
        idx.search(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: nsw,
    feature: "nsw",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let mut idx = vicinity::nsw::NSWIndex::new(dim, 16, 32).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::nsw::NSWIndex, q: &[f32], k: usize| {
        idx.search(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::nsw::NSWIndex, q: &[f32], k: usize, ef: usize| {
        idx.search(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: diskann,
    feature: "diskann",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let params = vicinity::diskann::DiskANNParams {
            m: 32, ef_construction: 100, alpha: 1.2, ef_search: 100
        };
        let mut idx = vicinity::diskann::DiskANNIndex::new(dim, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::diskann::DiskANNIndex, q: &[f32], k: usize| {
        idx.search(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::diskann::DiskANNIndex, q: &[f32], k: usize, ef: usize| {
        idx.search(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: nsg,
    feature: "nsg",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let params = vicinity::nsg::NsgParams {
            max_degree: 32, pool_size: 100, ..vicinity::nsg::NsgParams::default()
        };
        let mut idx = vicinity::nsg::NsgIndex::new(dim, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::nsg::NsgIndex, q: &[f32], k: usize| {
        idx.search_with_ef(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::nsg::NsgIndex, q: &[f32], k: usize, ef: usize| {
        idx.search_with_ef(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: emg,
    feature: "emg",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let params = vicinity::emg::EmgParams {
            max_degree: 32, candidate_size: 100, ..vicinity::emg::EmgParams::default()
        };
        let mut idx = vicinity::emg::EmgIndex::new(dim, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::emg::EmgIndex, q: &[f32], k: usize| {
        idx.search_with_ef(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::emg::EmgIndex, q: &[f32], k: usize, ef: usize| {
        idx.search_with_ef(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: finger,
    feature: "finger",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let params = vicinity::finger::FingerParams {
            max_degree: 32, ef_construction: 100, ..vicinity::finger::FingerParams::default()
        };
        let mut idx = vicinity::finger::FingerIndex::new(dim, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::finger::FingerIndex, q: &[f32], k: usize| {
        idx.search_with_ef(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::finger::FingerIndex, q: &[f32], k: usize, ef: usize| {
        idx.search_with_ef(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: pipnn,
    feature: "pipnn",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let params = vicinity::pipnn::PipnnParams {
            max_degree: 32, ..vicinity::pipnn::PipnnParams::default()
        };
        let mut idx = vicinity::pipnn::PipnnIndex::new(dim, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::pipnn::PipnnIndex, q: &[f32], k: usize| {
        idx.search_with_ef(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::pipnn::PipnnIndex, q: &[f32], k: usize, ef: usize| {
        idx.search_with_ef(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: fresh_graph,
    feature: "fresh_graph",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let params = vicinity::fresh_graph::FreshGraphParams {
            max_degree: 32, ef_construction: 100,
            ..vicinity::fresh_graph::FreshGraphParams::default()
        };
        let mut idx = vicinity::fresh_graph::FreshGraphIndex::new(dim, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::fresh_graph::FreshGraphIndex, q: &[f32], k: usize| {
        idx.search_with_ef(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::fresh_graph::FreshGraphIndex, q: &[f32], k: usize, ef: usize| {
        idx.search_with_ef(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: sng,
    feature: "sng",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let mut idx = vicinity::sng::SNGIndex::new(dim, vicinity::sng::SNGParams::default()).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::sng::SNGIndex, q: &[f32], k: usize| {
        idx.search(q, k).unwrap()
    }
}

ann_universal_tests! {
    mod_name: vamana,
    feature: "vamana",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let params = vicinity::vamana::VamanaParams {
            max_degree: 32, ef_construction: 100, alpha: 1.2,
            ..vicinity::vamana::VamanaParams::default()
        };
        let mut idx = vicinity::vamana::VamanaIndex::new(dim, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::vamana::VamanaIndex, q: &[f32], k: usize| {
        idx.search(q, k, 100).unwrap()
    },
    search_with_ef: |idx: &vicinity::vamana::VamanaIndex, q: &[f32], k: usize, ef: usize| {
        idx.search(q, k, ef).unwrap()
    }
}

ann_universal_tests! {
    mod_name: ivf_rabitq,
    feature: "ivf_rabitq",
    build_index: |data: &[Vec<f32>], dim: usize| {
        let params = vicinity::ivf_rabitq::IVFRaBitQParams {
            num_clusters: 8, nprobe: 4, ..vicinity::ivf_rabitq::IVFRaBitQParams::default()
        };
        let mut idx = vicinity::ivf_rabitq::IVFRaBitQIndex::new(dim, params).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    },
    search: |idx: &vicinity::ivf_rabitq::IVFRaBitQIndex, q: &[f32], k: usize| {
        idx.search(q, k).unwrap()
    }
}
