//! IVF-PQ: Inverted File with Product Quantization.
//!
//! The workhorse of billion-scale similarity search. IVF + PQ can provide large memory savings,
//! with the recall/latency trade-off controlled by parameters like `nprobe`, `num_clusters`, and PQ
//! codebook configuration. Exact numbers are workload-dependent; see the references below for
//! baseline results and methodology.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.3", features = ["ivf_pq"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};
//!
//! let params = IVFPQParams {
//!     num_clusters: 1024,   // sqrt(n) rule of thumb
//!     num_codebooks: 8,     // M subvectors
//!     codebook_size: 256,   // 8-bit codes
//!     nprobe: 10,           // cells to search
//! };
//!
//! let mut index = IVFPQIndex::new(128, params)?;
//! index.train(&training_vectors)?;  // Need ~10k-100k samples
//! index.add_batch(&database)?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Memory Calculation
//!
//! ```text
//! Compressed: n × M bytes  (M = num_codebooks)
//! Codebooks:  M × 256 × (d/M) × 4 bytes
//! Centroids:  k × d × 4 bytes
//!
//! Example: 1B vectors, d=128, M=8, k=4096
//!   Vectors:   1B × 8 = 8 GB     (vs 512 GB uncompressed!)
//!   Codebooks: 8 × 256 × 16 × 4 = 128 KB
//!   Centroids: 4096 × 128 × 4 = 2 MB
//!   Total:     ~8 GB
//! ```
//!
//! # Two Key Ideas
//!
//! ## 1. IVF: Partition and Prune
//!
//! Cluster vectors into k cells. At search time, only scan `nprobe` nearest cells:
//!
//! ```text
//! Brute force: O(n)      →   IVF: O(nprobe × n/k)
//!
//!           Query
//!             |
//!     +-------+-------+
//!     |               |
//!   Cell A          Cell B      (probe 2 cells)
//!   [vectors]       [vectors]   (skip other 1022 cells)
//! ```
//!
//! ## 2. PQ: Compress Vectors
//!
//! Split vector into M subvectors. Quantize each to 256 codewords (8 bits):
//!
//! ```text
//! Original:  [v₁ v₂ ... v₁₂₈]  (512 bytes)
//!             └─┬─┘ └─┬─┘
//!               ↓     ↓
//!            [c₁]   [c₂] ...   (8 bytes for M=8)
//! ```
//!
//! Distance computed via table lookup (precompute query-to-codebook distances).
//!
//! # Parameter Recommendations
//!
//! | Dataset Size | num_clusters | num_codebooks | nprobe |
//! |--------------|--------------|---------------|--------|
//! | 100K | 256 | 8 | 4 |
//! | 1M | 1024 | 8-16 | 8 |
//! | 10M | 4096 | 16 | 16 |
//! | 100M | 16384 | 16-32 | 32 |
//! | 1B | 65536 | 32-64 | 64 |
//!
//! **Rules of thumb**:
//! - `num_clusters` ≈ 4√n (slightly more aggressive than √n)
//! - `nprobe` ≈ 1-5% of clusters for 90%+ recall
//! - `num_codebooks` = d/16 is often good (d/M ≈ 8-16 dims per subvector)
//!
//! # Trade-offs
//!
//! | ↑ Parameter | Better | Worse |
//! |-------------|--------|-------|
//! | nprobe | Recall | Search latency |
//! | num_clusters | Search speed | Training time, accuracy at edges |
//! | num_codebooks | Accuracy | Memory, training time |
//!
//! # When to Use
//!
//! - Dataset **doesn't fit in RAM** (> 10M vectors on typical hardware)
//! - Can tolerate **85-95% recall** (vs 99%+ with HNSW)
//! - Need **sub-second search at billion scale**
//!
//! # When NOT to Use
//!
//! - Dataset fits in RAM → use HNSW (better recall)
//! - Need > 95% recall → use HNSW or exact search
//! - Can't provide training data → PQ codebooks need ~10k samples
//!
//! # OPQ: Optimized Product Quantization
//!
//! PQ assumes subspaces are independent. Real data has correlations.
//! OPQ learns a rotation matrix that decorrelates dimensions first,
//! which can improve recall at the same code size (see OPQ reference).
//!
//! # References
//!
//! - Jégou, Douze, Schmid (2011). "Product Quantization for Nearest Neighbor Search." `https://ieeexplore.ieee.org/document/5432202`
//! - Ge et al. (2014). "Optimized Product Quantization." `https://arxiv.org/abs/1311.4055`

// IVF-PQ core implementation (always available when ivf_pq feature is enabled)
pub mod opq;
pub mod pq;
pub mod search;
pub use search::{IVFPQIndex, IVFPQParams};

// NOTE: Advanced IVF-PQ components (OPQ, pure IVF) are stubs awaiting implementation.
// The core IVFPQIndex in search.rs provides the main functionality.
// TODO: Implement full OPQ (rotation matrix learning) and separate IVF index
