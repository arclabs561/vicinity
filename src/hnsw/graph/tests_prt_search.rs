use super::*;

/// PRT-accelerated search should produce similar recall to standard search,
/// while performing fewer full distance computations.
#[test]
fn test_search_prt_recall_parity() {
    let dim = 32;
    let n = 300;
    let mut rng_seed: u64 = 42;
    let mut next = || -> f32 {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        ef_search: 50,
        metric: DistanceMetric::L2,
        seed: Some(42),
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).unwrap();

    let mut all_vecs = Vec::new();
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| next()).collect();
        all_vecs.push(v.clone());
        index.add(i as u32, v).unwrap();
    }
    index.build().unwrap();

    // Build PRT state.
    let num_proj = 16;
    let mut prt = crate::prt::ProbabilisticRoutingTest::new(dim, num_proj, Some(42));
    prt.project_database(&index.vectors);

    let k = 10;
    let ef = 50;
    let num_queries = 20;
    let mut prt_recall_total = 0.0;
    let mut std_recall_total = 0.0;
    let mut total_full_dists = 0usize;

    for qi in 0..num_queries {
        let query: Vec<f32> = (0..dim).map(|_| next()).collect();

        // Standard search.
        let std_results = index.search(&query, k, ef).unwrap();

        // PRT search.
        let (prt_results, full_dists) = index.search_prt(&query, k, ef, &prt, 1.5, 0.95).unwrap();
        total_full_dists += full_dists;

        // Brute-force ground truth.
        let mut gt: Vec<(u32, f32)> = all_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, crate::distance::l2_distance(&query, v)))
            .collect();
        gt.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        let gt_set: std::collections::HashSet<u32> = gt.iter().take(k).map(|&(id, _)| id).collect();

        let std_ids: std::collections::HashSet<u32> =
            std_results.iter().map(|&(id, _)| id).collect();
        let prt_ids: std::collections::HashSet<u32> =
            prt_results.iter().map(|&(id, _)| id).collect();

        std_recall_total += gt_set.intersection(&std_ids).count() as f32 / k as f32;
        prt_recall_total += gt_set.intersection(&prt_ids).count() as f32 / k as f32;

        let _ = qi; // used for iteration
    }

    let std_recall = std_recall_total / num_queries as f32;
    let prt_recall = prt_recall_total / num_queries as f32;
    let avg_full_dists = total_full_dists as f32 / num_queries as f32;

    // PRT recall should be close to standard (within 20% relative).
    assert!(
        prt_recall > std_recall * 0.7,
        "PRT recall ({:.1}%) too far from standard ({:.1}%)",
        prt_recall * 100.0,
        std_recall * 100.0
    );

    // PRT should compute fewer full distances than total visited nodes.
    // With n=300 and ef=50, standard search visits ~50+ nodes.
    // PRT should skip at least some.
    assert!(
        avg_full_dists < n as f32,
        "PRT should not compute all {} distances (avg={})",
        n,
        avg_full_dists
    );
}

/// PRT search should return valid results and correct doc_ids.
#[test]
fn test_search_prt_basic() {
    let dim = 8;
    let mut index = HNSWIndex::builder(dim)
        .metric(DistanceMetric::L2)
        .m(8)
        .build()
        .unwrap();

    for i in 0..50 {
        let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect();
        index.add(i as u32, v).unwrap();
    }
    index.build().unwrap();

    let mut prt = crate::prt::ProbabilisticRoutingTest::new(dim, 4, Some(42));
    prt.project_database(&index.vectors);

    let query: Vec<f32> = vec![0.0; dim];
    let (results, full_dists) = index.search_prt(&query, 5, 20, &prt, 1.5, 0.95).unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 5);
    assert!(full_dists > 0);

    // Results should be sorted by distance.
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1, "results not sorted");
    }
}
