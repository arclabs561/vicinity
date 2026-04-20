#![cfg(feature = "hnsw")]

//! Test: does search_with_distance() give the same results as search()?
//!
//! This test is ADSampling-independent. It verifies that the two HNSW
//! search paths (greedy_search_layer vs greedy_search_layer_custom) produce
//! identical results when given the same distance function.
//!
//! If this test FAILS, the bug is in search_with_distance itself.
//! If this test PASSES, the bug is in how ADSampling's closure interacts.

#![allow(clippy::unwrap_used)]

#[cfg(feature = "hnsw")]
mod test {
    use std::collections::HashSet;
    use vicinity::distance::l2_distance;
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
                ((s >> 33) as f32) / (u32::MAX >> 1) as f32 * 255.0
            })
            .collect()
    }

    /// Build an HNSW index with L2 metric.
    fn build_l2_index(vectors: &[f32], dim: usize) -> HNSWIndex {
        let n = vectors.len() / dim;
        let params = HNSWParams {
            m: 16,
            m_max: 16,
            ef_construction: 100,
            metric: DistanceMetric::L2,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).unwrap();
        let ids: Vec<u32> = (0..n as u32).collect();
        index.add_batch(&ids, vectors).unwrap();
        index.build().unwrap();
        index
    }

    /// Compute brute-force top-k by L2 distance.
    fn brute_force_l2(train: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<(u32, f32)> {
        let n = train.len() / dim;
        let mut dists: Vec<(u32, f32)> = (0..n)
            .map(|i| {
                let v = &train[i * dim..(i + 1) * dim];
                (i as u32, l2_distance(query, v))
            })
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.truncate(k);
        dists
    }

    #[test]
    fn search_vs_search_with_distance_l2() {
        let dim = 128;
        let n_train = 5000;
        let n_test = 50;
        let k = 10;
        let ef = 100;

        let train = random_vectors(n_train, dim, 42);
        let test = random_vectors(n_test, dim, 99);
        let index = build_l2_index(&train, dim);

        let mut search_recall = 0.0f64;
        let mut swd_recall = 0.0f64;

        for qi in 0..n_test {
            let query = &test[qi * dim..(qi + 1) * dim];

            // Ground truth -- use index's reordered vectors + doc_id mapping
            // search() returns doc_ids, so ground truth should also use doc_ids
            let gt = brute_force_l2(&train, query, dim, k); // original order OK for GT
            let gt_ids: HashSet<u32> = gt.iter().map(|r| r.0).collect();

            // search()
            let r1 = index.search(query, k, ef).unwrap();
            let r1_ids: HashSet<u32> = r1.iter().map(|r| r.0).collect();
            search_recall += gt_ids.intersection(&r1_ids).count() as f64 / k as f64;

            // search_with_distance() using same L2 -- MUST use index's reordered vectors
            let r2 = index
                .search_with_distance(query, k, ef, &|q: &[f32], nid: u32| {
                    let v = &index.vectors_raw()[nid as usize * dim..(nid as usize + 1) * dim];
                    l2_distance(q, v)
                })
                .unwrap();
            let r2_ids: HashSet<u32> = r2.iter().map(|r| r.0).collect();
            swd_recall += gt_ids.intersection(&r2_ids).count() as f64 / k as f64;
        }

        let avg_search = search_recall / n_test as f64;
        let avg_swd = swd_recall / n_test as f64;

        eprintln!("search() recall:              {:.1}%", avg_search * 100.0);
        eprintln!("search_with_distance() recall: {:.1}%", avg_swd * 100.0);

        // Both should achieve reasonable recall
        assert!(
            avg_search > 0.5,
            "search() recall too low: {:.1}%",
            avg_search * 100.0
        );
        assert!(
            avg_swd > 0.5,
            "search_with_distance() recall too low: {:.1}%",
            avg_swd * 100.0
        );

        // The two methods should be within 5% of each other
        assert!(
            (avg_search - avg_swd).abs() < 0.05,
            "search() and search_with_distance() recall diverge: {:.1}% vs {:.1}%",
            avg_search * 100.0,
            avg_swd * 100.0
        );
    }

    /// Same test but using the HNSW's internal vectors (to eliminate
    /// any possibility that the separate `train` slice differs from
    /// the stored vectors).
    #[test]
    fn swd_reads_correct_vectors() {
        let dim = 64;
        let n = 200;
        let k = 10;
        let ef = 50;

        let vectors = random_vectors(n, dim, 77);
        let index = build_l2_index(&vectors, dim);

        // Query with a training vector (self-match expected)
        let query = &vectors[0..dim];
        let r1 = index.search(query, k, ef).unwrap();
        let first_nid = std::cell::Cell::new(u32::MAX);
        let r2 = index
            .search_with_distance(query, k, ef, &|q: &[f32], nid: u32| {
                if first_nid.get() == u32::MAX {
                    first_nid.set(nid);
                }
                let v = &index.vectors_raw()[nid as usize * dim..(nid as usize + 1) * dim];
                l2_distance(q, v)
            })
            .unwrap();
        eprintln!("First nid in dist_fn: {}", first_nid.get());

        // Debug: entry point and results
        let (ep, el) = index.entry_point().unwrap();
        eprintln!("Entry point: node {ep}, layer {el}");
        eprintln!("search() top-3: {:?}", &r1[..3]);
        eprintln!("swd()    top-3: {:?}", &r2[..3]);

        // Check: what distance does the closure compute for nid=0?
        let d_node0 = l2_distance(query, &vectors[0..dim]);
        let d_node9 = l2_distance(query, &vectors[9 * dim..10 * dim]);
        eprintln!("L2(query, vectors[0]) = {d_node0}");
        eprintln!("L2(query, vectors[9]) = {d_node9}");

        // Self-match: both should return doc_id 0 at distance ~0
        assert_eq!(r1[0].0, 0, "search() should find self-match");
        assert_eq!(
            r2[0].0, 0,
            "search_with_distance() should find self-match at dist {}, got doc_id {} at dist {}",
            d_node0, r2[0].0, r2[0].1
        );
        assert!(r1[0].1 < 0.01);
        assert!(r2[0].1 < 0.01);

        // Non-self query
        let query2 = &vectors[dim..2 * dim]; // vector 1
        let r3 = index.search(query2, k, ef).unwrap();
        let r4 = index
            .search_with_distance(query2, k, ef, &|q: &[f32], nid: u32| {
                let v = &index.vectors_raw()[nid as usize * dim..(nid as usize + 1) * dim];
                l2_distance(q, v)
            })
            .unwrap();

        let r3_ids: HashSet<u32> = r3.iter().map(|r| r.0).collect();
        let r4_ids: HashSet<u32> = r4.iter().map(|r| r.0).collect();
        let overlap = r3_ids.intersection(&r4_ids).count();

        assert!(
            overlap >= 8,
            "Non-self query: search() vs swd() should agree, got {overlap}/10.\n\
             search(): {:?}\nswd():    {:?}",
            &r3[..5],
            &r4[..5]
        );
    }
}
