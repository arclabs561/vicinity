//! Approximate Nearest Neighbor (ANN) search algorithms.
//!
//! Pure Rust implementations of ANN algorithms:
//! - **HNSW**: Hierarchical Navigable Small World (graph-based) - see [`crate::hnsw`]
//! - **NSW**: Flat Navigable Small World (single-layer graph) - see `crate::nsw`
//! - **AnisotropicVQ-kmeans**: Anisotropic Vector Quantization with k-means Partitioning
//!   (vendor name: SCANN/ScaNN) - see `crate::scann`
//! - **IVF-PQ**: Inverted File Index with Product Quantization - see `crate::ivf_pq`
//! - **DiskANN**: Disk-based ANN for very large datasets - see `crate::diskann`
//!
//! All algorithms are optimized with SIMD acceleration and minimal dependencies.
//! Use concrete index types directly (e.g., `hnsw::HNSWIndex`, `diskann::DiskANNIndex`).
