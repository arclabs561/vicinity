//! Partitioning/clustering for ANN methods.
//!
//! Provides k-means clustering used by IVF-PQ and ScaNN.

#[cfg(any(feature = "scann", feature = "ivf_pq"))]
pub mod kmeans;
