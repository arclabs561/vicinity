//! Filtered-ANN selectivity sweep against exact filtered ground truth.
//!
//! The suite needs `filtered_graph` and `hnsw`. The gate lives on the module
//! declaration below instead of a file-level `#![cfg(...)]`, so a partial
//! feature set still compiles and runs a real check rather than reporting
//! success having compared nothing.

#[cfg(all(feature = "filtered_graph", feature = "hnsw"))]
#[path = "filtered_recall/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "filtered_graph", feature = "hnsw")))]
#[test]
fn filtered_recall_suite_is_not_silently_empty() {
    // Always-true invariant: prefilter-then-brute-force over the matching subset
    // is the exact answer the sweep scores against, so it must return the
    // closest *matching* id and never an excluded one.
    let query = [0.0f32, 0.0];
    let corpus = [[1.0f32, 0.0], [0.25f32, 0.0], [3.0f32, 0.0]];
    let matches = |id: usize| id % 2 == 1;
    let best = corpus
        .iter()
        .enumerate()
        .filter(|(id, _)| matches(*id))
        .min_by(|(_, a), (_, b)| {
            vicinity::distance::l2_distance(&query, *a)
                .total_cmp(&vicinity::distance::l2_distance(&query, *b))
        })
        .map(|(id, _)| id);
    assert_eq!(
        best,
        Some(1),
        "exact filtered top-1 must skip non-matching ids"
    );

    eprintln!(
        "tests/filtered_recall.rs: enable features filtered_graph,hnsw to run the gated sweep"
    );
}
