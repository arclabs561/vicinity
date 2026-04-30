use super::*;
use std::collections::HashSet;

/// Verify the capacity-law guard: when k > 2*ef, the guard auto-adjusts ef
/// to prevent catastrophic recall collapse. Without the guard, requesting
/// k=50 with ef=5 would return geometrically meaningless results.
#[test]
fn test_capacity_law_guard_prevents_recall_collapse() {
    let dim = 16;
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

    let query: Vec<f32> = (0..dim).map(|_| next()).collect();
    let k = 20;

    // With capacity-law guard, ef=5 gets auto-adjusted to k=20.
    let results = index.search(&query, k, 5).unwrap();

    // Brute-force ground truth using stored vectors and L2 distance.
    let mut gt: Vec<(u32, f32)> = all_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, crate::distance::l2_distance(&query, v)))
        .collect();
    gt.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let gt_set: HashSet<u32> = gt.iter().take(k).map(|&(id, _)| id).collect();
    let result_ids: HashSet<u32> = results.iter().map(|&(doc_id, _)| doc_id).collect();

    let recall = gt_set.intersection(&result_ids).count() as f32 / k as f32;
    // The guard auto-adjusts ef from 5 to 20, producing reasonable recall.
    assert!(
        recall > 0.3,
        "Capacity-law guard should prevent collapse. recall={:.1}%",
        recall * 100.0
    );
}

#[cfg(feature = "parallel")]
#[test]
fn test_build_parallel_recall() {
    let dim = 32;
    let n = 500;
    let k = 10;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let params = HNSWParams {
        m: 16,
        m_max: 32,
        ef_construction: 200,
        metric: DistanceMetric::L2,
        seed: Some(42),
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).unwrap();

    let mut all_vecs = Vec::new();
    for i in 0..n {
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                use rand::Rng;
                rng.random::<f32>() * 2.0 - 1.0
            })
            .collect();
        all_vecs.push(v.clone());
        index.add(i as u32, v).unwrap();
    }
    index.build_parallel(64).unwrap();

    let query = &all_vecs[0];
    let results = index.search(query, k, 100).unwrap();

    // Brute-force ground truth.
    let mut gt: Vec<(u32, f32)> = all_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, crate::distance::l2_distance(query, v)))
        .collect();
    gt.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let gt_ids: HashSet<u32> = gt.iter().take(k).map(|(id, _)| *id).collect();
    let result_ids: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();

    let recall = gt_ids.intersection(&result_ids).count() as f32 / k as f32;
    assert!(
        recall >= 0.5,
        "Parallel build recall@{k} = {:.1}%, expected >= 50%",
        recall * 100.0
    );
}
