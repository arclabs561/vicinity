//! Persistence robustness tests for the HNSW serde path.
//!
//! The suite needs `hnsw` and `serde`. The gate lives on the module declaration
//! below instead of a file-level `#![cfg(...)]`, so a partial feature set still
//! compiles and runs a real check rather than reporting success having compared
//! nothing.

#[cfg(all(feature = "hnsw", feature = "serde"))]
#[path = "persistence_robustness/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "hnsw", feature = "serde")))]
#[test]
fn persistence_robustness_suite_is_not_silently_empty() {
    // Always-true invariant: the normalization a roundtrip must preserve is
    // idempotent and leaves unit vectors at unit norm.
    let v = vicinity::distance::normalize(&[3.0f32, 4.0]);
    assert_eq!(v, vec![0.6, 0.8]);
    assert_eq!(vicinity::distance::normalize(&v), v);

    eprintln!("tests/persistence_robustness.rs: enable features hnsw,serde to run the gated suite");
}
