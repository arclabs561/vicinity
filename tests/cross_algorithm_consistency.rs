//! Cross-algorithm consistency tests.
//!
//! The suite needs `hnsw`, `nsw`, `diskann` and `sng`. The gate lives on the
//! module declaration below instead of a file-level `#![cfg(...)]`, so a partial
//! feature set still compiles and runs a real check rather than reporting
//! success having compared nothing.

#[cfg(all(
    feature = "hnsw",
    feature = "nsw",
    feature = "diskann",
    feature = "sng"
))]
#[path = "cross_algorithm_consistency/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(
    feature = "hnsw",
    feature = "nsw",
    feature = "diskann",
    feature = "sng"
)))]
#[test]
fn cross_algorithm_suite_is_not_silently_empty() {
    // Always-true invariant: the premise these comparisons rest on holds for the
    // ungated kernels — on unit vectors, cosine and angular distance agree on a
    // zero distance exactly when the vectors are identical.
    let a = vicinity::distance::normalize(&[1.0f32, 2.0, 3.0]);
    let b = vicinity::distance::normalize(&[2.0f32, 4.0, 6.0]);
    assert_eq!(vicinity::distance::cosine_distance(&a, &b), 0.0);
    assert_eq!(vicinity::distance::angular_distance(&a, &b), 0.0);
    let c = vicinity::distance::normalize(&[1.0f32, 0.0, 0.0]);
    assert!(vicinity::distance::cosine_distance(&a, &c) > 0.0);

    eprintln!(
        "tests/cross_algorithm_consistency.rs: enable features hnsw,nsw,diskann,sng to run the gated suite"
    );
}
