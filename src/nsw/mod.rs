//! Flat Navigable Small World (NSW) graph.
//!
//! Single-layer variant of HNSW. Same search quality, ~25% less memory.
//!
//! # Feature Flag
//!
//! Requires the `nsw` feature:
//! ```toml
//! vicinity = { version = "0.1", features = ["nsw"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::nsw::{NSWIndex, NSWParams};
//!
//! let params = NSWParams { m: 16, ..Default::default() };
//! let mut index = NSWIndex::new(768, params);
//!
//! index.add(0, vec![0.1; 768])?;
//! index.build()?;
//!
//! let results = index.search(&query, 10, 50)?;  // k=10, ef=50
//! ```
//!
//! # Why Flat?
//!
//! HNSW's hierarchy (multiple layers) was designed to provide "express lanes"
//! for long-range navigation. Recent empirical work suggests that on modern,
//! high-dimensional embedding workloads the hierarchy can matter less than the
//! presence of “hub” nodes and the quality of neighbor selection.
//!
//! One concrete reference is Munyampirwa et al. (2024), who benchmark HNSW against a
//! flat NSW-like graph and report that the flat graph can retain HNSW’s latency/recall
//! benefits on high-dimensional datasets.
//!
//! Note: removing hierarchy eliminates upper-layer graph storage and can reduce overhead,
//! but for high-dimensional embeddings the **vector storage often dominates** total memory.
//! Treat any “% savings” claims as workload-dependent until you measure.
//!
//! # When to Use
//!
//! | Situation | Recommendation |
//! |-----------|----------------|
//! | High-dimensional embeddings, want a simpler graph | **NSW** |
//! | Want the most battle-tested default | HNSW |
//! | Want simpler code to understand | **NSW** |
//! | Production with many edge cases | HNSW (more battle-tested) |
//!
//! # Memory Comparison
//!
//! For 1M vectors at d=768, M=16:
//! - **HNSW**: ~3.5 GB (multi-layer storage)
//! - **NSW**: ~2.7 GB (single layer)
//! - **Savings**: ~800 MB
//!
//! # The Small World Property
//!
//! Even without hierarchy, a well-constructed graph has "six degrees of separation."
//! Greedy routing finds near-optimal paths in O(log n) hops:
//!
//! ```text
//! Query → Hub node → Intermediate → Target (3-4 hops typical)
//! ```
//!
//! Hubs emerge naturally from degree distribution and serve as express lanes.
//!
//! # References
//!
//! - Munyampirwa et al. (2024). "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'." (arXiv:2412.01940)
//! - See [`hnsw`](crate::hnsw) for the hierarchical variant

pub mod construction;
pub mod graph;
pub mod search;

pub use graph::{NSWIndex, NSWParams};
