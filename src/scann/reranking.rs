//! Re-ranking stage for SCANN.

use crate::distance::cosine_distance_normalized;

/// Re-rank candidates using exact distance computation.
///
/// Takes approximate results from quantization stage and re-computes
/// exact distances for top candidates to improve accuracy.
pub fn rerank(
    query: &[f32],
    candidates: &[(u32, f32)],
    vectors: &[f32],
    dimension: usize,
    k: usize,
) -> Vec<(u32, f32)> {
    // Re-compute exact distances
    let mut reranked: Vec<(u32, f32)> = candidates
        .iter()
        .map(|(id, _approx_dist)| {
            let vec = get_vector(vectors, dimension, *id as usize);
            let exact_dist = cosine_distance_normalized(query, vec);
            (*id, exact_dist)
        })
        .collect();

    // Sort by exact distance
    reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1)); // Unstable for better performance

    // Return top k
    reranked.into_iter().take(k).collect()
}

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}
