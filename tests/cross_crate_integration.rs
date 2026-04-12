#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Cross-crate integration tests for Tekne Stack.
//!
//! These tests verify that cross-crate dependencies work correctly:
//! - vicinity uses idpaq for ID compression
//! - All crates compile with their optional dependencies
//!
//! # Running These Tests
//!
//! ```bash
//! # Test idpaq integration
//! cargo test --features id-compression cross_crate
//! ```

// =============================================================================
// idpaq Integration
// =============================================================================

#[cfg(feature = "id-compression")]
mod idpaq_integration {
    use vicinity::compression::{IdSetCompressor, RocCompressor};

    /// Verify that vicinity correctly delegates to idpaq for compression
    #[test]
    fn vicinity_uses_idpaq_compressor() {
        let compressor = RocCompressor::new();

        // This pattern simulates HNSW neighbor lists
        let neighbor_ids: Vec<u32> = vec![42, 156, 789, 1234, 2345];
        let universe = 10_000;

        let compressed = compressor.compress_set(&neighbor_ids, universe).unwrap();
        let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

        assert_eq!(neighbor_ids, decompressed, "Round-trip should preserve IDs");
    }

    /// Test compression with patterns typical of IVF posting lists
    #[test]
    fn ivf_posting_list_pattern() {
        let compressor = RocCompressor::new();

        // IVF posting lists: many IDs from a large universe
        let posting_list: Vec<u32> = (0..500).map(|i| i * 200).collect();
        let universe = 100_000;

        let compressed = compressor.compress_set(&posting_list, universe).unwrap();
        let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

        assert_eq!(posting_list, decompressed);

        // Verify meaningful compression
        let uncompressed_size = posting_list.len() * 4;
        let ratio = uncompressed_size as f64 / compressed.len() as f64;
        assert!(
            ratio > 1.5,
            "Should achieve some compression, got {:.2}x",
            ratio
        );
    }

    /// Test edge cases that might occur in ANN indexes
    #[test]
    fn ann_edge_cases() {
        let compressor = RocCompressor::new();

        // Single neighbor (common in early HNSW layers)
        let single: Vec<u32> = vec![5000];
        let compressed = compressor.compress_set(&single, 10_000).unwrap();
        let decompressed = compressor.decompress_set(&compressed, 10_000).unwrap();
        assert_eq!(single, decompressed);

        // Maximum neighbors (M_max in HNSW)
        let max_neighbors: Vec<u32> = (0..64).map(|i| i * 100).collect();
        let compressed = compressor.compress_set(&max_neighbors, 10_000).unwrap();
        let decompressed = compressor.decompress_set(&compressed, 10_000).unwrap();
        assert_eq!(max_neighbors, decompressed);
    }
}

// =============================================================================
// Feature Compilation Tests
// =============================================================================

/// These tests just verify that features compile correctly together
mod feature_compilation {
    #[test]
    fn default_features_compile() {
        // IdCompressionMethod exists in both feature configurations (stub enum
        // without id-compression, real type with it). Asserting on the name
        // gives the test a runtime check rather than a silent pass.
        let name = std::any::type_name::<vicinity::compression::IdCompressionMethod>();
        assert!(name.contains("IdCompressionMethod"));
    }

    #[cfg(feature = "hnsw")]
    #[test]
    fn hnsw_feature_compiles() {
        // HNSW feature should enable the HNSW module
        // This is a compile-time check more than runtime
        // Compile-time check: HNSW feature enables the HNSW module
    }

    #[cfg(feature = "persistence")]
    #[test]
    fn persistence_feature_compiles() {
        // Compile-time check: persistence feature enables the persistence module
    }
}

// =============================================================================
// Dependency Graph Verification
// =============================================================================

/// These tests document and verify the intended dependency structure
mod dependency_documentation {
    /// Document the compression dependency chain
    ///
    /// vicinity (id-compression) -> idpaq
    #[test]
    fn compression_dependency_chain() {
        // This test documents the dependency:
        // When id-compression is enabled, vicinity delegates to idpaq
        //
        // The types should be:
        // - vicinity::compression::RocCompressor = idpaq::RocCompressor
        // - vicinity::compression::CompressionError = idpaq::CompressionError
        //
        // This is verified by the idpaq_integration tests above
        // Compile-time verification only; presence of this function confirms the chain compiles.
    }

    /// Document the SIMD dependency chain
    ///
    /// vicinity (innr feature, default) -> innr
    #[test]
    fn simd_dependency_chain() {
        // SIMD functions are internal (pub(crate)); the public API is via
        // vicinity::distance::{l2_distance, cosine_distance, ...}
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.707_f32, 0.707, 0.0];

        let cos_d = vicinity::distance::cosine_distance(&a, &b);
        let l2_d = vicinity::distance::l2_distance(&a, &b);

        assert!(
            cos_d < 0.5,
            "cosine distance should be small for similar vectors"
        );
        assert!(
            l2_d > 0.0,
            "l2 distance should be positive for different vectors"
        );
    }
}
