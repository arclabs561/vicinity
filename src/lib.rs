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

//! Approximate nearest-neighbor search.
//!
//! `vicinity` provides Rust indexes and Python bindings for vector search. The
//! default feature set enables HNSW and SIMD distance kernels. Other indexes are
//! available behind feature flags.
//!
//! # Install
//!
//! ```toml
//! vicinity = { version = "0.10.5", features = ["hnsw"] }
//! ```
//!
//! # Start With HNSW
//!
//! HNSW is the default in-memory index for dense vectors. Cosine distance
//! expects unit-norm vectors unless `auto_normalize(true)` is set.
//!
//! ```no_run
//! # fn main() -> Result<(), vicinity::RetrieveError> {
//! use vicinity::hnsw::HNSWIndex;
//!
//! let mut index = HNSWIndex::builder(128)
//!     .m(16)
//!     .ef_search(50)
//!     .auto_normalize(true)
//!     .build()?;
//!
//! index.add_slice(0, &[0.1; 128])?;
//! index.add_slice(1, &[0.2; 128])?;
//! index.build()?;
//!
//! let results = index.search(&[0.1; 128], 5, 50)?;
//! # let _ = results;
//! # Ok(())
//! # }
//! ```
//!
//! # Other Indexes
//!
//! | Workload | Start with | Feature |
//! | --- | --- | --- |
//! | Dense vectors that fit in memory | `hnsw::HNSWIndex` | `hnsw` |
//! | Raw vectors dominate RAM | `ivf_pq::IVFPQIndex` | `ivf_pq` |
//! | Frequent writes/deletes | `store::UpdatableIndex` or FreshGraph | `store`, `fresh_graph` |
//! | Metadata filters | HNSW post-filtering, ACORN, Curator, or FilteredGraph | `hnsw`, `curator`, `filtered_graph` |
//! | Sparse learned retrieval | SparseMIPS | `sparse_mips` |
//! | File-backed search | DiskANN | `diskann` |
//!
//! The repository README has current benchmark commands. The full feature flag
//! table is in `docs/algorithms.md`.

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
#[cfg(any(feature = "diskann", feature = "ivf_pq"))]
pub(crate) mod file_io;
pub mod filtering;
#[cfg(any(
    feature = "nsw",
    feature = "sng",
    feature = "vamana",
    feature = "nsg",
    feature = "finger",
    feature = "pipnn",
    feature = "emg",
    feature = "binary_index",
    feature = "rp_quant",
    feature = "sparse_mips",
    feature = "lsh",
    feature = "sq4",
    all(feature = "hnsw", feature = "sq4"),
    all(feature = "hnsw", feature = "sq8")
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
