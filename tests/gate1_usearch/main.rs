//! Gate 1 of the breadth kill test: HNSW at matched recall against `usearch`.
//!
//! Compares against `usearch`, so the body needs `hnsw`; other lanes still run a
//! real test instead of compiling this target to nothing.

#[cfg(feature = "hnsw")]
mod with_hnsw;

#[cfg(not(feature = "hnsw"))]
#[test]
fn gate1_needs_the_hnsw_feature() {
    eprintln!("gate1_usearch: enable --features hnsw; this lane has nothing to compare");
}
