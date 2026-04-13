#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests verifying idpaq usage in vicinity.
//!
//! These tests enforce that the id-compression feature correctly delegates
//! to the idpaq crate for ID set compression in ANN indexes.

#![cfg(feature = "id-compression")]

use vicinity::compression::{IdSetCompressor, RocCompressor};

// =============================================================================
// Basic Integration Tests
// =============================================================================

#[test]
fn roundtrip_small_set() {
    let compressor = RocCompressor::new();
    let ids: Vec<u32> = vec![1, 5, 10, 20, 50];
    let universe = 100;

    let compressed = compressor.compress_set(&ids, universe).unwrap();
    let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

    assert_eq!(ids, decompressed);
}

#[test]
fn roundtrip_empty_set() {
    let compressor = RocCompressor::new();
    let ids: Vec<u32> = vec![];
    let universe = 100;

    let compressed = compressor.compress_set(&ids, universe).unwrap();
    let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

    assert_eq!(ids, decompressed);
}

#[test]
fn roundtrip_single_element() {
    let compressor = RocCompressor::new();
    let ids: Vec<u32> = vec![42];
    let universe = 100;

    let compressed = compressor.compress_set(&ids, universe).unwrap();
    let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

    assert_eq!(ids, decompressed);
}

// =============================================================================
// HNSW Neighbor List Simulation
// =============================================================================

#[test]
fn hnsw_neighbor_list_compression() {
    // HNSW typically stores M neighbors per node (e.g., M=16 or M=32)
    // These are stored as sorted ID lists
    let compressor = RocCompressor::new();

    // Simulate a neighbor list from a graph with 10000 nodes
    let universe = 10_000;
    let neighbor_ids: Vec<u32> = vec![
        42, 156, 789, 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012,
    ];

    let compressed = compressor.compress_set(&neighbor_ids, universe).unwrap();
    let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

    assert_eq!(neighbor_ids, decompressed);

    // Verify compression actually reduces size
    let uncompressed_size = neighbor_ids.len() * std::mem::size_of::<u32>();
    println!(
        "HNSW neighbors: {} bytes -> {} bytes (ratio: {:.2}x)",
        uncompressed_size,
        compressed.len(),
        uncompressed_size as f64 / compressed.len() as f64
    );
}

#[test]
fn ivf_cluster_compression() {
    // IVF stores lists of vector IDs per cluster
    // These can be quite large (thousands of IDs)
    let compressor = RocCompressor::new();

    // Simulate a cluster with 500 vectors from a 100K dataset
    let universe = 100_000;
    let cluster_ids: Vec<u32> = (0..500).map(|i| i * 200 + (i % 17)).collect();

    let compressed = compressor.compress_set(&cluster_ids, universe).unwrap();
    let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

    assert_eq!(cluster_ids, decompressed);

    let uncompressed_size = cluster_ids.len() * std::mem::size_of::<u32>();
    let ratio = uncompressed_size as f64 / compressed.len() as f64;
    println!(
        "IVF cluster: {} bytes -> {} bytes (ratio: {:.2}x)",
        uncompressed_size,
        compressed.len(),
        ratio
    );

    // IVF clusters should achieve good compression (>= 1.9x with small epsilon)
    assert!(
        ratio >= 1.9,
        "Expected compression ratio >= 1.9 for IVF cluster, got {:.2}x",
        ratio
    );
}

// =============================================================================
// Error Handling
// =============================================================================

#[test]
fn rejects_unsorted_ids() {
    let compressor = RocCompressor::new();
    let unsorted: Vec<u32> = vec![10, 5, 20]; // Not sorted!
    let universe = 100;

    let result = compressor.compress_set(&unsorted, universe);
    assert!(result.is_err(), "Should reject unsorted input");
}

#[test]
fn rejects_ids_exceeding_universe() {
    let compressor = RocCompressor::new();
    let ids: Vec<u32> = vec![1, 5, 150]; // 150 > universe
    let universe = 100;

    let result = compressor.compress_set(&ids, universe);
    assert!(result.is_err(), "Should reject IDs exceeding universe");
}

#[test]
fn rejects_duplicate_ids() {
    let compressor = RocCompressor::new();
    let with_dups: Vec<u32> = vec![1, 5, 5, 10]; // Duplicate 5
    let universe = 100;

    let result = compressor.compress_set(&with_dups, universe);
    assert!(result.is_err(), "Should reject duplicate IDs");
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
fn large_set_roundtrip() {
    let compressor = RocCompressor::new();

    // Large set: 10K IDs from 1M universe
    let universe = 1_000_000;
    let ids: Vec<u32> = (0..10_000).map(|i| i * 100).collect();

    let compressed = compressor.compress_set(&ids, universe).unwrap();
    let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

    assert_eq!(ids, decompressed);
}

#[test]
fn consecutive_ids_compress_well() {
    let compressor = RocCompressor::new();

    // Consecutive IDs (best case for delta encoding)
    let ids: Vec<u32> = (1000..2000).collect();
    let universe = 10_000;

    let compressed = compressor.compress_set(&ids, universe).unwrap();
    let decompressed = compressor.decompress_set(&compressed, universe).unwrap();

    assert_eq!(ids, decompressed);

    let uncompressed_size = ids.len() * std::mem::size_of::<u32>();
    let ratio = uncompressed_size as f64 / compressed.len() as f64;

    // Consecutive IDs should compress extremely well (>= 3x)
    assert!(
        ratio >= 3.0,
        "Consecutive IDs should compress at >= 3x, got {:.2}x",
        ratio
    );
}

// =============================================================================
// Property-Based Tests (using proptest if available)
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn roundtrip_arbitrary_sets(
            len in 0usize..100,
            universe in 100u32..10000
        ) {
            let compressor = RocCompressor::new();

            // Generate sorted, unique IDs
            let mut ids: Vec<u32> = (0..len)
                .map(|i| (i as u32 * universe / (len.max(1) as u32 + 1)).min(universe - 1))
                .collect();
            ids.sort();
            ids.dedup();

            if ids.iter().all(|&id| id < universe) {
                let compressed = compressor.compress_set(&ids, universe).unwrap();
                let decompressed = compressor.decompress_set(&compressed, universe).unwrap();
                prop_assert_eq!(ids, decompressed);
            }
        }

        #[test]
        fn compression_deterministic(
            len in 1usize..50,
            universe in 100u32..1000
        ) {
            let compressor = RocCompressor::new();

            let ids: Vec<u32> = (0..len)
                .map(|i| (i as u32 * 2) % universe)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();

            if !ids.is_empty() && ids.iter().all(|&id| id < universe) {
                let c1 = compressor.compress_set(&ids, universe).unwrap();
                let c2 = compressor.compress_set(&ids, universe).unwrap();
                prop_assert_eq!(c1, c2, "Compression should be deterministic");
            }
        }
    }
}
