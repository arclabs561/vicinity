// Crate-level lint configuration
#![warn(missing_docs)]

//! vicinity: Approximate Nearest Neighbor Search primitives.
//!
//! Provides implementations of ANN algorithms:
//!
//! - **Graph-based**: [`hnsw`], `nsw`, `sng`, `vamana`, `nsg`, `emg`, `finger`, `pipnn`, `fresh_graph`
//! - **Graph + quantization**: `hnsw::symphony_qg` (RaBitQ inside HNSW beam search)
//! - **Partition-based**: `ivf_pq`, `ivf_avq`, `ivf_rabitq`, `curator`
//! - **Quantization**: `quantization` (RaBitQ, SAQ), `rp_quant`, `binary_index` (1-bit + rerank)
//! - **Filtered**: `filtered_graph` (predicate filters), `esg` (range filters), `curator` (label filters)
//! - **Sparse vectors**: `sparse_mips` (inner-product graph for SPLADE/BM25)
//! - **Streaming**: `streaming::lsm` (LSM-tree tiered HNSW)
//! - **Clustering**: `evoc` (EVoC hierarchical clustering)
//!
//! # Which Index Should I Use?
//!
//! | Situation | Recommendation | Feature |
//! |-----------|----------------|---------|
//! | **General Purpose** (Best Recall/Speed) | [`hnsw::HNSWIndex`] | `hnsw` (default) |
//! | **Billion-Scale** (Memory Constrained) | `ivf_pq::IVFPQIndex` | `ivf_pq` |
//! | **Flat Graph** (Simpler, competitive on high-d) | `nsw::NSWIndex` | `nsw` |
//! | **Label Filtering** (Low selectivity) | `curator::CuratorIndex` | `curator` |
//! | **Complex Predicates** (AND/OR filters) | `filtered_graph::FilteredGraphIndex` | `filtered_graph` |
//! | **Range Filtering** (Numeric attributes) | `esg::EsgIndex` | `esg` |
//! | **Dynamic Insert/Delete** | `fresh_graph::FreshGraphIndex` | `fresh_graph` |
//! | **Sparse Vectors** (SPLADE/BM25) | `sparse_mips::SparseMipsIndex` | `sparse_mips` |
//! | **High-d Compression** (768d+) | `rp_quant::RpQuantIndex` | `rp_quant` |
//! | **Quantized Graph** (HNSW + RaBitQ) | `hnsw::SymphonyQGIndex` | `hnsw` + `ivf_rabitq` |
//! | **Binary Quantization** (1-bit + rerank) | `binary_index::BinaryFlatIndex` | `binary_index` |
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
//! vicinity = "0.3"
//!
//! # With quantization support
//! vicinity = { version = "0.3", features = ["ivf_pq"] }
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
//! see `docs/references.md` in the repo.

pub mod classic;

#[cfg(feature = "fresh_graph")]
pub mod fresh_graph;

#[cfg(feature = "curator")]
pub mod curator;

#[cfg(feature = "diskann")]
pub mod diskann;

#[cfg(feature = "emg")]
pub mod emg;

#[cfg(feature = "esg")]
pub mod esg;

// Shared helpers for clump-backed modules (evoc, kmeans partitioning).
#[cfg(any(
    feature = "evoc",
    feature = "ivf_pq",
    feature = "ivf_avq",
    feature = "ivf_rabitq"
))]
pub(crate) mod clump_compat;

#[cfg(feature = "evoc")]
pub mod evoc;

#[cfg(feature = "hnsw")]
pub mod hnsw;

#[cfg(feature = "ivf_pq")]
pub mod ivf_pq;

#[cfg(feature = "nsg")]
pub mod nsg;

#[cfg(feature = "finger")]
pub mod finger;

#[cfg(feature = "filtered_graph")]
pub mod filtered_graph;

#[cfg(feature = "nsw")]
pub mod nsw;

#[cfg(feature = "quantization")]
pub mod quantization;

#[cfg(feature = "ivf_avq")]
pub mod ivf_avq;

#[cfg(feature = "pipnn")]
pub mod pipnn;

#[cfg(feature = "ivf_rabitq")]
pub mod ivf_rabitq;

#[cfg(feature = "rp_quant")]
pub mod rp_quant;

#[cfg(feature = "binary_index")]
pub mod binary_index;

#[cfg(feature = "sparse_mips")]
pub mod sparse_mips;

#[cfg(feature = "sng")]
pub mod sng;

#[cfg(feature = "vamana")]
pub mod vamana;

#[cfg(feature = "hnsw")]
pub(crate) mod adaptive;
pub mod partitioning;
#[cfg(feature = "ivf_pq")]
pub(crate) mod pq_simd;

pub mod distance;
pub mod filtering;
#[cfg(any(
    feature = "finger",
    feature = "filtered_graph",
    feature = "nsg",
    feature = "sparse_mips",
    feature = "fresh_graph",
    feature = "emg",
    feature = "pipnn"
))]
pub(crate) mod graph_utils;
#[cfg(feature = "hnsw")]
pub mod lid;
pub(crate) mod simd;

// Spectral sanity helpers (feature-gated).
#[cfg(feature = "rmt-spectral")]
pub mod spectral;

// Re-exports
pub use distance::DistanceMetric;
pub use error::{Result, RetrieveError};

#[cfg(feature = "benchmark")]
pub mod benchmark;
pub mod compression;
pub mod error;
pub mod persistence;
#[cfg(feature = "hnsw")]
pub mod streaming;
