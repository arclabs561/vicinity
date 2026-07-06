// Crate-level lint configuration
#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::needless_update,
        clippy::needless_range_loop,
        clippy::useless_vec
    )
)]

//! vicinity: Approximate Nearest Neighbor Search primitives.
//!
//! Provides implementations of ANN algorithms:
//!
//! - **Graph-based**: [`hnsw`], `nsw`, `sng`, `vamana`, `nsg`, `emg`, `finger`, `pipnn`, `fresh_graph`
//! - **Graph + quantization**: `hnsw::symphony_qg` (RaBitQ inside HNSW beam search)
//! - **Partition-based**: `ivf_pq`, `ivf_avq`, `ivf_rabitq`, `curator`
//! - **Quantization**: `quantization` (RaBitQ, SAQ), `rp_quant`, `binary_index` (1-bit + rerank)
//! - **Filtered**: `filtered_graph` (predicate filters), `range_filtered` (numeric range filters), `curator` (label filters)
//! - **Sparse vectors**: `sparse_mips` (inner-product graph for SPLADE/BM25)
//! - **Streaming**: `streaming::lsm` (LSM-tree tiered HNSW)
//! - **Clustering**: `evoc` (EVoC hierarchical clustering)
//!
//! # Which Index Should I Use?
//!
//! | Situation | Recommendation | Feature | Persistence |
//! |-----------|----------------|---------|-------------|
//! | **General Purpose** (Best Recall/Speed) | [`hnsw::HNSWIndex`] | `hnsw` (default) | Yes: JSON via `serde` (`save_to_file`); binary via `persistence` (segment writer) |
//! | **Memory Constrained** (compressed in-memory search) | `ivf_pq::IVFPQIndex` | `ivf_pq` | Yes: snapshot reloads into memory |
//! | **Flat Graph** (Simpler, competitive on high-d) | `nsw::NSWIndex` | `nsw` | Yes: snapshot reloads into memory |
//! | **Label Filtering** (Low selectivity) | `curator::CuratorIndex` | `curator` | Yes: snapshot reloads and rebuilds tree |
//! | **Complex Predicates** (AND/OR filters) | `filtered_graph::FilteredGraphIndex` | `filtered_graph` | Yes: snapshot reloads into memory |
//! | **Range Filtering** (Numeric attributes) | `range_filtered::RangeFilteredIndex` | `range_filtered` | Yes: snapshot reloads and rebuilds HNSW |
//! | **Dynamic Insert/Delete** (per-op latency) | `fresh_graph::FreshGraphIndex` | `fresh_graph` | Yes: snapshot reloads into memory |
//! | **Streaming Bulk Writes** (write throughput) | `streaming::lsm::LsmIndex` | `hnsw` (LSM is built-in) | No |
//! | **Sparse Vectors** (SPLADE/BM25) | `sparse_mips::SparseMipsIndex` | `sparse_mips` | No |
//! | **High-d Compression** (768d+) | `rp_quant::RpQuantIndex` | `rp_quant` | No |
//! | **Quantized Graph** (HNSW + RaBitQ, cosine) | `hnsw::SymphonyQGIndex` | `hnsw` + `ivf_rabitq` | No |
//! | **Quantized Graph** (HNSW + RaBitQ, cosine + L2) | `hnsw::SymphonyQGVRIndex` | `hnsw` + `ivf_rabitq` | No |
//! | **Binary Quantization** (1-bit + rerank) | `binary_index::BinaryFlatIndex` | `binary_index` | No |
//! | **File-Backed Search** (SSD-based, experimental) | `diskann` | `diskann` (experimental) | Yes: file and mmap searcher |
//! | **4-bit Scalar Quant** (8x compression) | `sq4::SQ4Index` | `sq4` | No |
//! | **8-bit Scalar Quant** (4x compression) | `hnsw::HNSWSq8Index` | `hnsw` + `sq8` | No |
//!
//! **Default features**: `hnsw`, `innr` (SIMD).
//!
//! ## Recommendation Logic
//!
//! 1. **Start with HNSW**. It's the industry standard for a reason. It offers the best
//!    trade-off between search speed and recall for datasets that fit in RAM.
//!
//! 2. **Use IVF-PQ** when raw vectors dominate memory. It compresses the vector
//!    payload and can drop raw vectors with `compact()`, but it is still an
//!    in-memory ANN index rather than a disk-resident search path.
//!
//! 3. **Try NSW (Flat)** if you want a simpler graph, or you are benchmarking on
//!    modern embeddings (hundreds/thousands of dimensions). Recent empirical work suggests the
//!    hierarchy may provide less incremental value in that regime (see arXiv:2412.01940).
//!    *Note: HNSW is the more common default in production systems, so it’s still a safe first choice.*
//!
//! 4. **Use DiskANN** (experimental) when you need a file-backed searcher.
//!    Treat mmap/page-layout behavior as a separate benchmark target.
//!
//! ```toml
//! # Minimal (HNSW + SIMD)
//! vicinity = "0.10.5"
//!
//! # With quantization support
//! vicinity = { version = "0.10.5", features = ["ivf_pq"] }
//! ```
//!
//! # Notes (evidence-backed)
//!
//! - **Flat vs hierarchical graphs**: Munyampirwa et al. (2024) empirically argue that, on
//!   high-dimensional datasets, a flat small-world graph can match HNSW’s recall/latency
//!   benefits because “hub” nodes provide routing power without explicit hierarchy
//!   (arXiv:2412.01940). This doesn’t make HNSW “wrong”; it just means NSW is often a
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

/// Updatable, durable multi-segment ANN index via segstore (the `store` feature).
#[cfg(feature = "store")]
pub mod store;

#[cfg(feature = "curator")]
pub mod curator;

#[cfg(feature = "diskann")]
pub mod diskann;

#[cfg(feature = "emg")]
pub mod emg;

/// Range-filtered ANN search (HNSW + attribute-range post-filter).
///
/// Renamed from `esg` in 0.8.0. The 0.7.x `esg` name implied fidelity
/// to the partition-aware structure of arXiv:2504.04018; the shipped
/// implementation is the paper's strawman baseline (HNSW + range
/// post-filter), so the module was renamed to reflect what it actually
/// does. A paper-fidelity ESG variant is planned as a separate module.
#[cfg(feature = "range_filtered")]
pub mod range_filtered;

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

#[cfg(feature = "lsh")]
pub mod lsh;

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

#[cfg(feature = "sq4")]
pub mod sq4;

#[cfg(feature = "lemur")]
pub mod lemur;

#[cfg(feature = "sng")]
pub mod sng;

#[cfg(feature = "vamana")]
pub mod vamana;

#[cfg(feature = "hnsw")]
pub(crate) mod adaptive;
#[cfg(feature = "hnsw")]
pub mod adsampling;
pub mod partitioning;
#[cfg(feature = "ivf_pq")]
// pq_simd at crate root (not under `quantization::`) because the
// `quantization` mod is feature-gated by `quantization` while these
// kernels are needed by `ivf_pq` independently. Both are pub(crate).
pub(crate) mod pq_simd;
#[cfg(feature = "hnsw")]
pub mod prt;

pub mod distance;
pub mod filtering;
#[cfg(any(
    feature = "nsw",
    feature = "sng",
    feature = "vamana",
    feature = "nsg",
    feature = "finger",
    feature = "pipnn",
    feature = "emg"
))]
pub(crate) mod graph_snapshot;
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
pub mod memory;
pub(crate) mod simd;

// Spectral sanity helpers (feature-gated). Folded into the only
// consumer (`adsampling`) in 0.8.0 to drop the unused public path.
#[cfg(feature = "rmt-spectral")]
pub(crate) mod spectral;

// Re-exports
pub use distance::DistanceMetric;
pub use error::{Result, RetrieveError};
pub use memory::MemoryReport;

#[cfg(feature = "benchmark")]
pub mod benchmark;
pub mod compression;
pub mod error;
pub mod persistence;
#[cfg(feature = "python")]
pub mod python;
#[cfg(feature = "hnsw")]
pub mod streaming;
