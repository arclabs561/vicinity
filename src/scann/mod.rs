//! ScaNN: Google's algorithm for Maximum Inner Product Search (MIPS).
//!
//! Optimized for **recommendations and retrieval** where you need inner products.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.1", features = ["scann"] }
//! ```
//!
//! # Status: Experimental
//!
//! Google Research's ScaNN algorithm. Under active development.
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::scann::{SCANNIndex, SCANNParams};
//!
//! let params = SCANNParams {
//!     num_partitions: 256,
//!     num_reorder: 100,
//!     num_codebooks: 16,
//!     codebook_size: 256,
//!     seed: 42,
//! };
//!
//! let mut index = SCANNIndex::new(128, params);
//! index.train(&training_vectors)?;
//! index.add_batch(&database)?;
//! index.build()?;
//!
//! // Returns (index, inner_product) pairs
//! let results = index.search_mips(&query, 10)?;
//! ```
//!
//! # The Problem: PQ is Wrong for Inner Products
//!
//! Standard PQ minimizes reconstruction error:
//!
//! ```text
//! PQ objective:  min ||x - Q(x)||²
//! ```
//!
//! But for MIPS, we care about **inner product preservation**, not reconstruction.
//! PQ can systematically bias results toward high-norm vectors.
//!
//! # The Solution: Anisotropic Quantization
//!
//! AVQ weights quantization error by **dimension importance**:
//!
//! ```text
//! AVQ objective:  min ||x - Q(x)||²_W   (weighted by query distribution)
//! ```
//!
//! If queries have large values in dimension i, errors in i matter more.
//!
//! # Three-Stage Pipeline
//!
//! ```text
//! Query → [1. Coarse] → [2. Fine] → [3. Rerank]
//!          k-means       AVQ scan    Exact MIPS
//!          ~256 cells    Fast approx Top 100 only
//! ```
//!
//! 1. **Coarse search**: Find nearest cluster centroids
//! 2. **Fine search**: AVQ codes for fast approximate inner products
//! 3. **Rerank**: Exact computation on top candidates
//!
//! # ScaNN vs IVF-PQ
//!
//! | | IVF-PQ | ScaNN |
//! |-|--------|-------|
//! | **Optimized for** | L2 distance | Inner product |
//! | **Quantization** | Standard PQ | Anisotropic (AVQ) |
//! | **Use case** | Similarity search | Recommendations |
//! | **Reranking** | Optional | Required (part of design) |
//!
//! # Parameter Recommendations
//!
//! | Dataset | num_partitions | num_reorder | num_codebooks |
//! |---------|----------------|-------------|---------------|
//! | 1M | 256 | 100 | 16 |
//! | 10M | 1024 | 200 | 16-32 |
//! | 100M | 4096 | 500 | 32 |
//!
//! **Rule of thumb**: `num_reorder` ≈ 10-50x k (number of results)
//!
//! # When to Use
//!
//! - **Maximum Inner Product Search** (MIPS)
//! - Recommendation systems (user dot item)
//! - Two-tower retrieval models
//! - Dense retrieval where inner product matters
//!
//! # When NOT to Use
//!
//! - L2/Euclidean distance → use IVF-PQ
//! - Cosine similarity → normalize vectors, use any index
//! - Small datasets (< 100K) → brute force is fine
//! - Need real-time updates → ScaNN requires batch rebuild
//!
//! # References
//!
//! - Guo et al. (2020). "Accelerating Large-Scale Inference with Anisotropic Vector Quantization." `https://arxiv.org/abs/1908.10396`
//! - (ScaNN overview page) `https://github.com/google-research/google-research/tree/master/scann`
//! - Sun et al. (2023). "SOAR: Improved Indexing for Approximate Nearest Neighbor Search."

/// K-means partitioning for ScaNN coarse quantization.
pub mod partitioning;
mod quantization;
mod reranking;
pub mod search;

pub use search::{SCANNIndex, SCANNParams};
