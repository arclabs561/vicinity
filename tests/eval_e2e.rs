//! End-to-end evaluation tests using standard benchmark methodology.
//!
//! The suite needs `hnsw` and `benchmark`. The gate lives on the module
//! declaration below instead of a file-level `#![cfg(...)]`, so a partial
//! feature set still compiles and runs a real check rather than reporting
//! success having compared nothing.

#[cfg(all(feature = "hnsw", feature = "benchmark"))]
#[path = "eval_e2e/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "hnsw", feature = "benchmark")))]
#[test]
fn eval_e2e_suite_is_not_silently_empty() {
    // Always-true invariant: the ranking the evaluation measures against is
    // reproducible by hand — exact recall@1 picks the closest vector.
    let query = [0.0f32, 0.0];
    let corpus = [[1.0f32, 0.0], [4.0f32, 0.0], [0.5f32, 0.0]];
    let nearest = corpus
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            vicinity::distance::l2_distance(&query, *a)
                .total_cmp(&vicinity::distance::l2_distance(&query, *b))
        })
        .map(|(id, _)| id);
    assert_eq!(nearest, Some(2), "exact recall@1 must find the closest id");

    eprintln!(
        "tests/eval_e2e.rs: enable features hnsw,benchmark to run the gated evaluation suite"
    );
}
