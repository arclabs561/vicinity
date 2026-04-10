//! Shared test helpers for vicinity integration tests.
#![allow(dead_code)]

use std::collections::HashSet;

/// Normalize a vector to unit length. Returns the original vector if near-zero.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Generate deterministic random vectors using DefaultHasher (no `rand` dep needed).
pub fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::hash::{Hash, Hasher};

    (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    seed.hash(&mut hasher);
                    i.hash(&mut hasher);
                    j.hash(&mut hasher);
                    let h = hasher.finish();
                    (h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
                })
                .collect()
        })
        .collect()
}

/// Generate deterministic normalized random vectors using `StdRng`.
pub fn random_normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
            vicinity::distance::normalize(&v)
        })
        .collect()
}

/// Compute exact k-NN using cosine distance (brute force). Returns doc IDs.
pub fn brute_force_knn(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, vicinity::distance::cosine_distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Recall@k: fraction of ground truth IDs found in search results.
///
/// `results` is the ANN search output (doc_id, distance pairs).
/// `ground_truth` is the list of exact nearest neighbor doc IDs.
pub fn recall_at_k(results: &[(u32, f32)], ground_truth: &[u32]) -> f32 {
    let gt_set: HashSet<u32> = ground_truth.iter().copied().collect();
    let hits = results.iter().filter(|(id, _)| gt_set.contains(id)).count();
    hits as f32 / ground_truth.len().max(1) as f32
}

/// Recall@k comparing two sets of (doc_id, distance) results.
pub fn recall_at_k_sets(exact: &[(u32, f32)], approx: &[(u32, f32)], k: usize) -> f32 {
    let exact_set: HashSet<u32> = exact.iter().take(k).map(|(i, _)| *i).collect();
    let approx_set: HashSet<u32> = approx.iter().take(k).map(|(i, _)| *i).collect();
    exact_set.intersection(&approx_set).count() as f32 / k as f32
}
