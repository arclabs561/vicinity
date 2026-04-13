//! Disk persistence for `vicinity` indexes.
//!
//! Two persistence paths are available:
//!
//! - **Simple** (`serde` feature): `save_to_writer` / `load_from_reader` on `HNSWIndex`.
//!   JSON-based, one function call each way. Best for checkpoint/restore workflows.
//!
//! - **Segment** (`persistence` feature): Binary SoA format via `HNSWSegmentWriter` /
//!   `HNSWSegmentReader`. Smaller on disk, cache-friendly layout.
//!
//! The `persistence` feature also brings WAL support via `durability`.

pub mod directory;
pub mod error;

#[cfg(feature = "persistence")]
pub mod format;

#[cfg(feature = "persistence")]
pub mod wal;

#[cfg(all(feature = "persistence", feature = "hnsw"))]
pub mod hnsw;

pub use error::PersistenceError;
