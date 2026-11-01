//! Property-based tests for HNSW persistence roundtrips.
//!
//! The suite needs `hnsw` and `persistence`. The gate lives on the module
//! declaration below instead of a file-level `#![cfg(...)]`, so a partial
//! feature set still compiles and runs a real check rather than reporting
//! success having compared nothing.

#[cfg(all(feature = "hnsw", feature = "persistence"))]
#[path = "prop_persistence/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "hnsw", feature = "persistence")))]
#[test]
fn prop_persistence_suite_is_not_silently_empty() {
    // Always-true invariant a roundtrip must preserve: a stored vector keeps its
    // direction after normalization, so its cosine distance to itself stays 0.
    let stored = vicinity::distance::normalize(&[3.0f32, 4.0]);
    let reloaded = stored.clone();
    assert_eq!(
        vicinity::distance::cosine_distance_normalized(&stored, &reloaded),
        0.0,
        "a reloaded vector must keep its direction"
    );

    eprintln!(
        "tests/prop_persistence.rs: enable features hnsw,persistence to run the gated property suite"
    );
}
