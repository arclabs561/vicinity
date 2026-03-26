//! Disk persistence for `vicinity` indexes.
//!
//! This module provides crash-safe, concurrent persistence for dense vector indexes
//! (HNSW, IVF-PQ, DiskANN).
//!
//! # Design Philosophy
//!
//! The persistence layer prioritizes:
//! - **Correctness**: Crash-safe, ACID guarantees, data integrity
//! - **Concurrency**: Multiple readers, single writer with snapshot isolation
//! - **Performance**: Memory mapping, SIMD-accelerated compression, efficient formats
//! - **Flexibility**: Support for all retrieval methods, configurable trade-offs
//!

pub mod directory;
pub mod error;

#[cfg(feature = "persistence")]
pub mod format;

#[cfg(feature = "persistence")]
pub mod wal;

#[cfg(feature = "persistence")]
pub mod checkpoint;

#[cfg(feature = "persistence")]
pub mod recovery;

#[cfg(all(feature = "persistence", feature = "hnsw"))]
pub mod hnsw;

#[cfg(feature = "persistence")]
pub mod locking;

pub use error::PersistenceError;
