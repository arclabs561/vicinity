//! Lossless compression for vector IDs in ANN indexes.
//!
//! This module provides compression algorithms that exploit ordering invariance
//! in vector ID collections (IVF clusters, HNSW neighbor lists) to achieve
//! significant compression ratios (5-7x for large sets).
//!
//! Based on "Lossless Compression of Vector IDs for Approximate Nearest Neighbor Search"
//! (Severo et al., 2025).
//!
//! # Implementation
//!
//! When the `id-compression` feature is enabled, compression primitives are
//! provided by the `cnk` crate, which implements:
//! - **ROC (Random Order Coding)**: Compress sets using bits-back coding with ANS
//! - **Delta encoding**: Simple baseline with varint
//!
//! # Usage
//!
//! ```rust,ignore
//! use vicinity::compression::{RocCompressor, IdSetCompressor};
//!
//! let compressor = RocCompressor::new();
//! let ids: Vec<u32> = vec![1, 5, 10, 20, 50];
//! let universe_size = 1000;
//!
//! // Compress
//! let compressed = compressor.compress_set(&ids, universe_size)?;
//!
//! // Decompress
//! let decompressed = compressor.decompress_set(&compressed, universe_size)?;
//! ```

// When id-compression feature is enabled, re-export everything from cnk
#[cfg(feature = "id-compression")]
pub use cnk::{
    choose_method, compress_set_auto, compress_set_enveloped, decompress_set_auto,
    decompress_set_enveloped, AutoConfig, ChooseConfig, CodecChoice, CompressionError,
    IdCompressionMethod, IdListStats, IdSetCompressor, RocCompressor,
};

// Fallback types when id-compression is not enabled
#[cfg(not(feature = "id-compression"))]
mod error;
#[cfg(not(feature = "id-compression"))]
mod traits;

#[cfg(not(feature = "id-compression"))]
pub use error::CompressionError;
#[cfg(not(feature = "id-compression"))]
pub use traits::IdSetCompressor;

/// Compression method selection.
#[cfg(not(feature = "id-compression"))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum IdCompressionMethod {
    /// No compression (uncompressed storage).
    #[default]
    None,
    /// Elias-Fano encoding (baseline, sorted sequences).
    EliasFano,
    /// Partitioned Elias–Fano (cluster-aware monotone sequences).
    PartitionedEliasFano,
    /// Random Order Coding (optimal for sets, uses bits-back with ANS).
    Roc,
    /// Wavelet tree (full random access, future).
    WaveletTree,
}

#[cfg(all(test, feature = "id-compression"))]
mod tests {
    use super::*;

    #[test]
    fn test_cnk_integration() {
        let compressor = RocCompressor::new();
        let ids: Vec<u32> = vec![1, 5, 10, 20, 50, 100];
        let universe_size = 1000;

        let compressed = compressor.compress_set(&ids, universe_size).unwrap();
        let decompressed = compressor
            .decompress_set(&compressed, universe_size)
            .unwrap();

        assert_eq!(ids, decompressed);
    }

    #[test]
    fn test_compression_ratio() {
        let compressor = RocCompressor::new();
        let ids: Vec<u32> = (0..1000).collect();
        let universe_size = 100_000;

        let compressed = compressor.compress_set(&ids, universe_size).unwrap();
        let ratio = (ids.len() * 4) as f64 / compressed.len() as f64;

        // Should achieve compression
        assert!(
            ratio > 1.5,
            "Expected compression ratio > 1.5, got {}",
            ratio
        );
    }
}
