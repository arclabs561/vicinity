//! Correctness regression tests for fixed algorithm bugs.
//!
//! The suite needs `hnsw`, `ivf_pq`, `ivf_avq` and `nsw`. The gate lives on the
//! module declaration below instead of a file-level `#![cfg(...)]`, so a partial
//! feature set still compiles and runs a real check rather than reporting
//! success having compared nothing.

#[cfg(all(
    feature = "hnsw",
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "nsw"
))]
#[path = "correctness_regression/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(
    feature = "hnsw",
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "nsw"
)))]
#[test]
fn correctness_regression_suite_is_not_silently_empty() {
    // Always-true invariant: the ungated distance kernels the fixed bugs were
    // measured against still agree with hand-computed values.
    let a = [0.0f32, 0.0];
    let b = [3.0f32, 4.0];
    assert_eq!(vicinity::distance::l2_distance(&a, &b), 5.0);
    assert_eq!(
        vicinity::distance::l2_distance(&a, &b),
        vicinity::distance::l2_distance(&b, &a),
        "L2 distance must be symmetric"
    );

    eprintln!(
        "tests/correctness_regression.rs: enable features hnsw,ivf_pq,ivf_avq,nsw to run the gated suite"
    );
}
