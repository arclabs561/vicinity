//! DiskANN/Vamana-style graph with file and mmap search paths.
//!
//! This module implements an experimental Vamana-family graph index plus
//! searchers that can read the saved graph and vector files through positional
//! file I/O or read-only memory maps. It is related to Microsoft DiskANN, but it
//! is not a full reproduction of the billion-scale DiskANN system described in
//! the paper.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.10.5", features = ["diskann"] }
//! ```
//!
//! # Current Scope
//!
//! Construction stores vectors and graph edges in memory, then serializes them.
//! `DiskANNSearcher` can search the saved graph and vector files directly, and
//! `search_with_diagnostics` reports logical graph and vector reads for
//! page-layout work.
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::diskann::{DiskANNIndex, DiskANNParams};
//!
//! let params = DiskANNParams {
//!     m: 32,
//!     alpha: 1.2,
//!     ef_construction: 100,
//!     ef_search: 100,
//! };
//!
//! let mut index = DiskANNIndex::new(128, params);
//! index.add(0, vec![0.1; 128]);
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Why File-Backed Search Matters
//!
//! ```text
//! 1B vectors × 768 dims × 4 bytes = 3 TB
//! ```
//!
//! Large dense-vector collections do not always fit comfortably in memory.
//! This implementation separates in-memory construction from file and mmap
//! search rows so benchmarks can show the storage cost explicitly.
//!
//! # Current Storage Path
//!
//! Current code searches separate fixed-record graph and vector files with
//! positional file reads or mmap-backed slices.
//!
//! # Page-Layout Target
//!
//! Co-locating each node's vector and neighbors in one page is the design
//! target for the next DiskANN performance pass, not the current layout.
//!
//! ```text
//! Memory:  [Cache: ~1% hot nodes] + [Beam search state]
//!              ↓ cache miss
//! NVMe:    [Node 0: vector + edges][Node 1: vector + edges]...
//!              └─ 4KB aligned for efficient I/O
//! ```
//!
//! 1. **Vamana graph**: Single-layer (no hierarchy = sequential I/O)
//! 2. **Target co-located storage**: Vector + neighbor list in same disk block
//! 3. **Target prefetch**: Hide SSD latency with batched or async I/O
//!
//! # Reference Paper Performance
//!
//! These are DiskANN paper results, not vicinity benchmark results. Use
//! `docs/benchmark-results.md` for measurements from this implementation.
//!
//! | Scale | Recall@10 | Latency | Memory | Storage |
//! |-------|-----------|---------|--------|---------|
//! | 100M | 95% | ~1ms | 8 GB | 50 GB |
//! | 1B | 95% | ~5ms | 64 GB | 500 GB |
//!
//! **Throughput**: see the paper for reported throughput; it depends strongly on
//! hardware and parameter choices.
//!
//! # Paper-Scale Parameters
//!
//! These values are starting points from the DiskANN/Vamana family, not
//! validated defaults for every dataset.
//!
//! | Dataset | m | alpha | ef_construction |
//! |---------|---|-------|-----------------|
//! | 100M | 32 | 1.2 | 100 |
//! | 1B | 64 | 1.2 | 200 |
//! | > 1B | 96 | 1.4 | 400 |
//!
//! # Current Fit
//!
//! Use this module to evaluate Vamana-style graph construction and file or mmap
//! search behavior. Do not read its rustdoc as a production-latency claim.
//!
//! # Why Single-Layer (Vamana) Instead of HNSW?
//!
//! - Hierarchy = random I/O (jump between layers on disk = slow)
//! - Flat graph = sequential reads possible
//! - Alpha-pruning provides long-range connections without layers
//!
//! # References
//!
//! - Jayaram Subramanya et al. (2019). "DiskANN: Fast Accurate Billion-point
//!   Nearest Neighbor Search on a Single Node."
//! - See also: `vamana` module for the graph construction algorithm
//! - NeurIPS 2019 landing page: `https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html`

pub mod cache;
pub mod disk_io;
pub mod graph;
#[cfg(any(test, feature = "benchmark"))]
pub(crate) mod page_io;

pub use graph::DiskANNIndex;
pub use graph::DiskANNParams;
pub use graph::DiskANNSearchDiagnostics;
pub use graph::DiskANNSearcher;
#[cfg(feature = "benchmark")]
#[doc(hidden)]
pub use page_io::DiskANNPageSearcher;
