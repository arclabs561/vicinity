//! Hierarchical Navigable Small World (HNSW) approximate nearest neighbor search.
//!
//! The industry-standard graph-based ANN algorithm. Pure Rust with SIMD acceleration.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use vicinity::hnsw::HNSWIndex;
//!
//! fn main() -> Result<(), vicinity::RetrieveError> {
//!     // dimension=128, M=16, m_max=16
//!     let mut index = HNSWIndex::new(128, 16, 16)?;
//!
//!     // Vectors must be L2-normalized for cosine distance
//!     let v0 = vicinity::distance::normalize(&vec![0.1; 128]);
//!     let v1 = vicinity::distance::normalize(&vec![0.2; 128]);
//!     index.add_slice(0, &v0)?;
//!     index.add_slice(1, &v1)?;
//!     index.build()?;
//!
//!     // k=10, ef_search=50
//!     let q = vec![0.15; 128];
//!     let results = index.search(&q, 10, 50)?;
//!     Ok(())
//! }
//! ```
//!
//! # Parameter Recommendations
//!
//! | Dataset Size | M | ef_construction | ef_search | Memory/vector |
//! |--------------|---|-----------------|-----------|---------------|
//! | < 100K | 16 | 100 | 50 | ~1.2 KB |
//! | 100K - 1M | 16 | 200 | 100 | ~1.2 KB |
//! | 1M - 10M | 32 | 200 | 100-200 | ~2.4 KB |
//! | > 10M | 48 | 400 | 200+ | ~3.6 KB |
//!
//! **Memory formula**: `n × (d × 4 + M × 8 + overhead)` bytes
//! - For 1M vectors at d=768, M=16: ~3.5 GB
//!
//! # The Small-World Insight
//!
//! HNSW exploits the **small-world network property**: in well-constructed graphs,
//! any two nodes can be reached in O(log n) hops via greedy routing.
//!
//! ```text
//! Layer 3: [sparse]     o───────────────o  (long jumps, few nodes)
//! Layer 2:          o───o───o───o───o───o  (medium connections)
//! Layer 1:       o─o─o─o─o─o─o─o─o─o─o─o─o  (local connections)
//! Layer 0:     ooooooooooooooooooooooooooo  (all nodes, dense)
//! ```
//!
//! Search starts at sparse top layer, descends using each layer as a better starting point.
//!
//! # Parameter Effects
//!
//! | Parameter | Effect when increased |
//! |-----------|----------------------|
//! | **M** | Better recall, more memory, slower build |
//! | **ef_construction** | Better graph quality, slower build |
//! | **ef_search** | Better recall, slower search |
//!
//! # When NOT to Use HNSW
//!
//! - **< 10K vectors**: Brute force is faster (no graph overhead)
//! - **> 99.9% recall required**: Graph methods have a recall ceiling
//! - **Memory constrained + > 10M vectors**: Use IVF-PQ instead
//!
//! # Variants
//!
//! | Variant | Type | Use case |
//! |---------|------|----------|
//! | [`HNSWIndex`] | Standard | General-purpose ANN; batch build then search |
//! | [`inplace::InPlaceIndex`] | Streaming | Per-operation insert/delete without batch rebuild (IP-DiskANN style) |
//! | [`dual_branch::DualBranchHNSW`] | LID-aware | Datasets with outliers or varying density; uses skip bridges for sparse regions |
//!
//! **Standard** ([`HNSWIndex`]): The default. Build the full graph, then query. Best when
//! the dataset is static or changes infrequently.
//!
//! **Streaming** ([`inplace::InPlaceIndex`]): Supports efficient per-operation insertions
//! and deletions via in-neighbor tracking, without batch consolidation. Use when the index
//! must stay current under continuous writes.
//!
//! **LID-aware** ([`dual_branch::DualBranchHNSW`]): Assigns high-LID (outlier) points to
//! higher layers and adds skip bridges for long-range navigation. Maintains two search
//! fronts (standard greedy + skip-bridge exploration). Use when recall degrades on datasets
//! with non-uniform density.
//!
//! # Hierarchy in High Dimensions
//!
//! Recent empirical work suggests the *hierarchical* aspect of HNSW can provide
//! **less incremental benefit** on modern, high-dimensional embedding datasets,
//! where “hub” nodes emerge and are sufficient for fast routing.
//!
//! Concretely, Munyampirwa et al. (2024) benchmark HNSW against a **flat** navigable
//! small-world graph and report that the flat graph can retain the key latency/recall
//! benefits of HNSW on high-dimensional datasets.
//!
//! **Practical advice**:
//! - HNSW remains a safe default (widely used; robust).
//! - If you are indexing modern embeddings (hundreds/thousands of dims) and want to
//!   simplify or reduce overhead, consider trying flat `crate::nsw` (or other flat
//!   graph variants like Vamana / DiskANN-style graphs) and compare recall/latency on
//!   your workload.
//!
//! # Advanced Features
//!
//! - [`filtered`]: ACORN-style attribute filtering
//! - [`dual_branch`]: LID-based insertion with skip bridges (see arXiv:2501.13992)
//!
//! # Historical Lineage
//!
//! HNSW builds on a rich history of small-world network theory:
//!
//! | Year | Work | Contribution |
//! |------|------|--------------|
//! | 1967 | Milgram | "Six degrees of separation" experiment |
//! | 2000 | Kleinberg | Proved O(log²n) greedy routing possible on augmented grids |
//! | 2011 | Malkov et al. | NSW: flat navigable small-world graph for ANN |
//! | 2014 | Malkov et al. | Improved NSW with better neighbor selection |
//! | 2016 | Malkov & Yashunin | HNSW: hierarchical NSW with skip-list-like layers |
//!
//! **Kleinberg's insight** (2000): Random long-range edges enable efficient routing
//! if their probability decays as `P(distance) ∝ d^(-r)` with `r = dim` (dimension).
//! This explains why HNSW works: the hierarchical structure implicitly creates
//! this distance-dependent edge distribution.
//!
//! # References
//!
//! - Milgram (1967). "The Small World Problem." Psychology Today.
//! - Kleinberg (2000). "The Small-World Phenomenon: An Algorithmic Perspective."
//! - Malkov et al. (2014). "Approximate nearest neighbor algorithm based on
//!   navigable small world graphs." Information Systems.
//! - Malkov & Yashunin (2016). "Efficient and robust approximate nearest neighbor
//!   search using Hierarchical Navigable Small World graphs." IEEE TPAMI.
//! - Munyampirwa et al. (2024). "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'." (arXiv:2412.01940)

#[cfg(feature = "hnsw")]
pub(crate) mod construction;
#[cfg(feature = "hnsw")]
pub(crate) mod graph;
#[cfg(feature = "hnsw")]
mod search;

#[cfg(feature = "hnsw")]
pub use graph::{
    HNSWBuilder, HNSWIndex, HNSWParams, NeighborhoodDiversification, SeedSelectionStrategy,
};

// Filtered search (ACORN-style)
#[cfg(feature = "hnsw")]
pub mod filtered;
#[cfg(feature = "hnsw")]
pub use filtered::{
    acorn_search, AcornConfig, FilterPredicate, FilterStrategy, FnFilter, MetadataFilterAdapter,
    NoFilter,
};

// Graph repair (MN-RU algorithm for deletions)
#[cfg(feature = "hnsw")]
pub mod repair;
#[cfg(feature = "hnsw")]
pub use repair::{
    compute_repair_operations, validate_connectivity, GraphRepairer, RepairConfig, RepairStats,
};

// In-place updates (IP-DiskANN style)
#[cfg(feature = "hnsw")]
pub mod inplace;
#[cfg(feature = "hnsw")]
pub use inplace::{InPlaceConfig, InPlaceIndex, InPlaceStats, MappedInPlaceIndex};

// Dual-Branch HNSW with LID-based insertion and skip bridges (arXiv 2501.13992)
#[cfg(feature = "hnsw")]
pub mod dual_branch;
#[cfg(feature = "hnsw")]
pub use dual_branch::{DualBranchConfig, DualBranchHNSW, DualBranchStats, SkipBridge};

// Scalar quantization (SQ8) wrapper
#[cfg(all(feature = "hnsw", feature = "innr"))]
pub mod scalar_quantized;

// ─── Experimental modules ────────────────────────────────────────────────────
// These are research implementations. Public for experimentation but not part
// of the stable API. Types are accessible via submodule paths
// (e.g., `vicinity::hnsw::fused::FusedIndex`) but not re-exported at
// `vicinity::hnsw::*`.

/// Tombstone-based deletions for streaming updates.
#[cfg(feature = "hnsw")]
#[doc(hidden)]
pub mod tombstones;

// ─── Experimental modules ────────────────────────────────────────────────────
// Research implementations gated behind `hnsw` only. Not re-exported. Access
// via submodule paths (e.g., `vicinity::hnsw::fused::FusedIndex`).

/// FusedANN: Attribute-vector fusion for filtered search.
#[cfg(feature = "hnsw")]
#[doc(hidden)]
pub mod fused;

/// Dynamic Edge Navigation Graph (DEG) for bimodal data.
#[cfg(feature = "hnsw")]
#[doc(hidden)]
pub mod deg;

/// HNSW index merging algorithms (NGM, IGTM, CGTM).
#[cfg(feature = "hnsw")]
#[doc(hidden)]
pub mod merge;

/// Random walk-based graph repair (alternative to MN-RU).
#[cfg(feature = "hnsw")]
#[doc(hidden)]
pub mod random_walk_repair;

/// Incremental learning patterns (edge refinement, temporal locality).
#[cfg(feature = "hnsw")]
#[doc(hidden)]
pub mod incremental;

/// Probabilistic edge routing (PEOs) for QPS improvement.
#[cfg(feature = "hnsw")]
#[doc(hidden)]
pub mod probabilistic_routing;

/// Vamana graph construction (DiskANN-style alpha-pruning).
/// Prefer `crate::vamana` for the integrated implementation.
#[cfg(feature = "hnsw")]
#[doc(hidden)]
pub mod vamana;
