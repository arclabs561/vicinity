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
//!
//! # Format compatibility
//!
//! Segment metadata written by vicinity 0.7+ starts with an 8-byte magic
//! (`format::HNSW_SEGMENT_MAGIC`, `VCNHNSW\x01`) followed by a version number
//! (`format::FORMAT_VERSION`). The loader's contract:
//!
//! - Segments written by 0.6.x (no magic) load transparently via a legacy
//!   v0 decode path.
//! - A version newer than the running crate supports is rejected with
//!   [`PersistenceError::Format`]; old crates never silently misread new
//!   files.
//! - Corrupt or truncated input returns [`PersistenceError`] (`Format` or
//!   `Io`); load never panics and never constructs an index from garbage.
//!   Size guards reject files claiming unreasonable dimensions, vector
//!   counts, or neighbor counts.
//!
//! The simple JSON path makes the same error-not-panic promise: corrupt or
//! truncated JSON fails `load_from_reader` with a typed error, and structural
//! invariants are validated before the index is usable.

pub mod directory;
pub mod error;

#[cfg(feature = "persistence")]
pub mod format;

#[cfg(feature = "persistence")]
pub mod wal;

#[cfg(all(feature = "persistence", feature = "hnsw"))]
pub mod hnsw;

pub use error::PersistenceError;
