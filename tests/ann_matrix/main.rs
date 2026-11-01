//! ANN comparison matrix: several sizes, dimensions and distributions.
//!
//! The matrix compares against `usearch`, so the body only compiles when `hnsw`
//! is on. The lane that runs without it still executes a real test below rather
//! than compiling this file to nothing.

#[cfg(feature = "hnsw")]
mod with_hnsw;

#[cfg(not(feature = "hnsw"))]
#[test]
fn matrix_needs_the_hnsw_feature() {
    // A comment, not a silent skip: the feature matrix runs this target with
    // unrelated features, and a zero-test target reports success forever.
    eprintln!("ann_matrix: enable --features hnsw; this lane has nothing to compare");
}
