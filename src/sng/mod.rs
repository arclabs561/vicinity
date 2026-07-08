//! Experimental OPT-SNG-style sparse neighborhood graph.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.10.5", features = ["sng"] }
//! ```
//!
//! # Status: Experimental
//!
//! This module implements an OPT-SNG-inspired graph with automatic truncation
//! parameter selection and snapshot persistence. The current construction path
//! still performs pairwise distance work and rejects builds above 50,000 vectors
//! to keep accidental full-corpus runs bounded. Treat it as a research baseline
//! until benchmark rows show a recall, QPS, or build-time win over HNSW on a
//! documented workload.
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::sng::{SNGIndex, SNGParams};
//!
//! // No parameter tuning needed!
//! let mut index = SNGIndex::new(128, SNGParams::default());
//!
//! index.add(0, vec![0.1; 128]);
//! index.build()?; // Parameters auto-optimized
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # How: Martingale-Based Pruning
//!
//! The construction model uses candidate-set evolution and an adaptive
//! truncation radius during graph construction:
//!
//! ```text
//! Traditional:              OPT-SNG:
//! 1. Compute ALL distances  1. Compute distances incrementally
//! 2. Prune afterward        2. Stop when E[improvement] < threshold
//!                           3. Truncation radius R adapts per-node
//! ```
//!
//! # Automatic Optimization
//!
//! | Parameter | HNSW | OPT-SNG |
//! |-----------|------|---------|
//! | Max degree (M) | Manual | Auto: O(n^{2/3+ε}) |
//! | Truncation (R) | N/A | Auto per-node |
//! | ef_construction | Manual | Implicit in martingale |
//!
//! # When to Use
//!
//! - Research comparisons against sparse-neighborhood graph construction.
//! - Small or capped benchmark runs where O(n²) construction is acceptable.
//! - Snapshot-memory persistence checks for graph-family coverage.
//!
//! # When NOT to Use
//!
//! - Production defaults. Start with HNSW, Vamana, NSG, or IVF-PQ.
//! - Full 1M-vector benchmark builds with the current implementation.
//! - Claims about external SNG paper numbers without a local benchmark row.
//!
//! # Theoretical Guarantees
//!
//! These are paper-level targets, not measured guarantees for this
//! implementation:
//!
//! | Metric | Bound |
//! |--------|-------|
//! | Search path length | O(log n) expected |
//! | Max out-degree | O(n^{2/3+ε}) |
//! | Construction time | O(n^{5/3+ε}) vs O(n²) naive |
//!
//! # References
//!
//! - Ma et al. (2025). "Sparse Neighborhood Graph-Based Approximate Nearest
//!   Neighbor Search Revisited: Theoretical Analysis and Optimization."
//!   arXiv:2509.15531

mod graph;
mod martingale;
mod optimization;
mod search;

pub use graph::{SNGIndex, SNGParams};
