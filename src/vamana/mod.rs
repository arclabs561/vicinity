//! Vamana approximate nearest neighbor search.
//!
//! # Feature Flag
//!
//! Requires the `vamana` feature:
//! ```toml
//! vicinity = { version = "0.8", features = ["vamana"] }
//! ```
//!
//! Vamana is a graph-based ANN algorithm that uses two-pass construction with
//! RRND (Relaxed Relative Neighborhood Diversification) and RND strategies.
//!
//! Vamana-style flat graphs are commonly used as the basis for SSD-oriented ANN systems
//! (e.g. DiskANN), where a single-layer graph is friendlier to disk access patterns than a
//! multi-layer hierarchy.
//!
//! # Algorithm
//!
//! Vamana constructs a proximity graph using:
//! 1. Random graph initialization with node degree >= log(n)
//! 2. First pass: Refine using RND (alpha=1.0, strict diversity)
//! 3. Second pass: Further refine using RRND (alpha>1.0, relaxed)
//! 4. Two-pass construction ensures better graph quality
//!
//! # Performance
//!
//! - Competitive with HNSW on large datasets (100GB-1B vectors)
//! - Better for SSD-based serving (higher points/node ratio)
//! - Two-pass construction: Higher indexing time but better graph quality
//!
//! # Usage
//!
//! Requires `features = ["vamana"]`:
//!
//! ```ignore
//! use vicinity::vamana::{VamanaIndex, VamanaParams};
//!
//! let params = VamanaParams {
//!     max_degree: 64,
//!     alpha: 1.3,
//!     ..Default::default()
//! };
//! let mut index = VamanaIndex::new(128, params)?;
//!
//! index.add(0, vec![0.1; 128])?;
//! index.build()?;
//!
//! let results = index.search(&vec![0.15; 128], 10, 50)?;
//! ```
//!
//! # References
//!
//! - Subramanya et al. (2019): "DiskANN: Fast accurate billion-point nearest neighbor search"

#[cfg(feature = "vamana")]
mod construction;
#[cfg(feature = "vamana")]
mod graph;
#[cfg(feature = "vamana")]
mod search;

#[cfg(feature = "vamana")]
pub use graph::{VamanaIndex, VamanaParams};
