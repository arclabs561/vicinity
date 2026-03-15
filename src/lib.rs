// Crate-level lint configuration
#![warn(missing_docs)]

//! vicinity: Approximate Nearest Neighbor Search primitives.
//!
//! Provides standalone implementations of state-of-the-art ANN algorithms:
//!
//! - **Graph-based**: [`hnsw`], `nsw`, `sng`, `vamana`
//! - **Partition-based**: `ivf_pq`, `scann`
//! - **Quantization**: `quantization` (PQ, RaBitQ)
//!
//! # Which Index Should I Use?
//!
//! | Situation | Recommendation | Feature |
//! |-----------|----------------|---------|
//! | **General Purpose** (Best Recall/Speed) | [`hnsw::HNSWIndex`] | `hnsw` (default) |
//! | **Billion-Scale** (Memory Constrained) | `ivf_pq::IVFPQIndex` | `ivf_pq` |
//! | **Flat Graph** (Simpler graph, often worth trying for modern embeddings) | `nsw::NSWIndex` | `nsw` |
//! | **Attribute Filtering** | [`hnsw::filtered`] | `hnsw` |
//! | **Out-of-Core** (SSD-based) | `diskann` | `diskann` (experimental) |
//!
//! **Default features**: `hnsw`, `innr` (SIMD).
//!
//! ## Recommendation Logic
//!
//! 1. **Start with HNSW**. It's the industry standard for a reason. It offers the best
//!    trade-off between search speed and recall for datasets that fit in RAM.
//!
//! 2. **Use IVF-PQ** if your dataset is too large for RAM (e.g., > 10M vectors on a laptop).
//!    It compresses vectors (32x-64x) but has lower recall than HNSW.
//!
//! 3. **Try NSW (Flat)** if you want a simpler graph, or you are benchmarking on
//!    modern embeddings (hundreds/thousands of dimensions). Recent empirical work suggests the
//!    hierarchy may provide less incremental value in that regime (see arXiv:2412.01940).
//!    *Note: HNSW is the more common default in production systems, so it’s still a safe first choice.*
//!
//! 4. **Use DiskANN** (experimental) if you have an NVMe SSD and 1B+ vectors.
//!
//! ```toml
//! # Minimal (HNSW + SIMD)
//! vicinity = "0.1"
//!
//! # With quantization support
//! vicinity = { version = "0.1", features = ["ivf_pq"] }
//! ```
//!
//! # Notes (evidence-backed)
//!
//! - **Flat vs hierarchical graphs**: Munyampirwa et al. (2024) empirically argue that, on
//!   high-dimensional datasets, a flat small-world graph can match HNSW’s recall/latency
//!   benefits because “hub” nodes provide routing power without explicit hierarchy
//!   (arXiv:2412.01940). This doesn’t make HNSW “wrong” — it just means NSW is often a
//!   worthwhile baseline to benchmark.
//!
//! - **Memory**: for modern embeddings, the raw vector store (n × d × 4 bytes) can dominate.
//!   The extra hierarchy layers and graph edges still matter, but you should measure on your
//!   actual (n, d, M, ef) and memory layout.
//!
//! - **Quantization**: IVF-PQ and related techniques trade recall for memory. `vicinity` exposes
//!   IVF-PQ under the `ivf_pq` feature, but you should treat parameter selection as workload-
//!   dependent (benchmark recall@k vs latency vs memory).
//!
//! ## Background (kept short; pointers to sources)
//!
//! - **Distance concentration**: in high dimensions, nearest-neighbor distances can become
//!   less discriminative; see Beyer et al. (1999), “When Is Nearest Neighbor Meaningful?”
//!   (DOI: 10.1007/s007780050006).
//!
//! - **Hubness**: some points appear as nearest neighbors for many queries (“hubs”); see
//!   Radovanović et al. (2010), “Hubs in Space”.
//!
//! - **Benchmarking**: for real comparisons, report recall@k vs latency/QPS curves and include
//!   memory and build time. When in doubt, use the `ann-benchmarks` datasets and methodology:
//!   `http://ann-benchmarks.com/`.
//!
//! For a curated bibliography covering HNSW/NSW/NSG/DiskANN/PQ/OPQ/ScaNN and related phenomena,
//! see `doc/references.md` in the repo.

pub mod ann;
pub mod classic;

#[cfg(feature = "diskann")]
pub mod diskann;

// Shared helpers for clump-backed modules (evoc, kmeans partitioning).
#[cfg(any(feature = "evoc", feature = "scann", feature = "ivf_pq"))]
pub(crate) mod clump_compat;

#[cfg(feature = "evoc")]
pub mod evoc;

#[cfg(feature = "hnsw")]
pub mod hnsw;

#[cfg(feature = "ivf_pq")]
pub mod ivf_pq;

#[cfg(feature = "nsw")]
pub mod nsw;

#[cfg(feature = "quantization")]
pub mod quantization;

#[cfg(feature = "scann")]
pub mod scann;

#[cfg(feature = "sng")]
pub mod sng;

#[cfg(feature = "vamana")]
pub mod vamana;

pub(crate) mod adaptive;
pub(crate) mod matryoshka;
pub mod partitioning;
pub(crate) mod pq_simd;

pub mod distance;
pub mod filtering;
pub mod lid;
pub mod simd;

// Spectral sanity helpers (feature-gated).
#[cfg(feature = "rmt-spectral")]
pub mod spectral;

// Re-exports
pub use distance::DistanceMetric;
pub use error::{Result, RetrieveError};

pub mod benchmark;
pub mod compression;
pub mod error;
pub mod persistence;
pub mod streaming;
