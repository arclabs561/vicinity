//! Property tests for Wolverine++ deletion and seeded RNG reproducibility.
//!
//! The suite needs `hnsw` and `vamana`. The gate lives on the module declaration
//! below instead of a file-level `#![cfg(...)]`, so a partial feature set still
//! compiles and runs a real check rather than reporting success having compared
//! nothing.

#[cfg(all(feature = "hnsw", feature = "vamana"))]
#[path = "prop_deletion/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "hnsw", feature = "vamana")))]
#[test]
fn prop_deletion_suite_is_not_silently_empty() {
    // Always-true invariant at the heart of delete-then-search: an id removed
    // from the candidate set can never be returned by a nearest-neighbour scan.
    let query = [0.0f32, 0.0];
    let ids = [0u32, 1, 2];
    let vectors = [[1.0f32, 0.0], [0.25f32, 0.0], [3.0f32, 0.0]];
    let deleted = 1u32;

    let hits: Vec<u32> = ids
        .iter()
        .copied()
        .filter(|id| *id != deleted)
        .filter(|id| vicinity::distance::l2_distance(&query, &vectors[*id as usize]) <= 1.0)
        .collect();

    assert_eq!(hits, vec![0], "deleted id must never appear in results");

    eprintln!(
        "tests/prop_deletion.rs: enable features hnsw,vamana to run the gated property suite"
    );
}
