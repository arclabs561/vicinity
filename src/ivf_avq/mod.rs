//! IVF-AVQ: k-means partitioning + Anisotropic Vector Quantization + reranking.
//!
//! Optimized for **inner product search** (MIPS): recommendations, two-tower retrieval.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.10.5", features = ["ivf_avq"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::ivf_avq::{IVFAVQIndex, IVFAVQParams};
//!
//! let params = IVFAVQParams {
//!     num_partitions: 256,
//!     nprobe: 20,
//!     num_reorder: 100,
//!     num_codebooks: 16,
//!     codebook_size: 256,
//!     seed: 42,
//! };
//!
//! let mut index = IVFAVQIndex::new(128, params);
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
//! Built indexes can be saved with `IVFAVQIndex::save_to_dir()` and restored
//! into memory with `IVFAVQIndex::load_from_dir()`. `IVFAVQFileSearcher` opens
//! the same saved format for direct file-backed search; mmap search is tracked
//! separately in `docs/persistence.md`.
//!
//! # IVF-AVQ vs IVF-PQ
//!
//! | | IVF-PQ | IVF-AVQ |
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
//! - Need real-time updates → IVF-AVQ requires batch rebuild
//!
//! # References
//!
//! - Guo et al. (2020). "Accelerating Large-Scale Inference with Anisotropic Vector Quantization." `https://arxiv.org/abs/1908.10396`
//! - Guo et al. (2020). ScaNN paper with AVQ algorithm details: `https://arxiv.org/abs/1908.10396`
//! - Sun et al. (2023). "SOAR: Improved Indexing for Approximate Nearest Neighbor Search."

/// K-means partitioning for IVF-AVQ coarse quantization.
pub mod partitioning;
mod quantization;
mod reranking;
pub mod search;

pub use search::{IVFAVQFileSearcher, IVFAVQIndex, IVFAVQParams};
