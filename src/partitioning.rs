//! Partitioning/clustering for ANN methods.
//!
//! Provides k-means clustering used by IVF-PQ, IVF-AVQ, and IVF-RaBitQ.

#[cfg(any(feature = "ivf_avq", feature = "ivf_pq", feature = "ivf_rabitq"))]
pub mod kmeans;
