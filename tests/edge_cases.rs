#![cfg(feature = "hnsw")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Edge case tests for vicinity.
//!
//! Tests unusual inputs and boundary conditions that could cause failures.

#[path = "common/mod.rs"]
mod common;
use common::*;

use vicinity::hnsw::HNSWIndex;

// =============================================================================
// Dimension edge cases
// =============================================================================

#[test]
fn very_small_dimension() {
    let dim = 2; // Minimum practical dimension
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let angle = (i as f32) * 0.1;
            normalize(&[angle.cos(), angle.sin()])
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&vectors[0], 5, 50).expect("Search failed");
    assert_eq!(results.len(), 5);
    assert_eq!(results[0].0, 0); // Should find itself
}

#[test]
fn high_dimension() {
    let dim = 1024; // Higher than typical BERT (768)
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let vectors: Vec<Vec<f32>> = (0..20)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i + 1) as f32 * (d + 1) as f32 * 0.1).sin())
                .collect();
            normalize(&v)
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&vectors[10], 5, 50).expect("Search failed");
    assert!(!results.is_empty());
}

// =============================================================================
// Vector count edge cases
// =============================================================================

#[test]
fn small_index() {
    let dim = 32;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    // Only 3 vectors (less than M)
    let vectors: Vec<Vec<f32>> = (0..3)
        .map(|i| normalize(&vec![(i + 1) as f32; dim]))
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&vectors[0], 10, 50).expect("Search failed");
    assert_eq!(results.len(), 3, "Should return all 3 vectors");
}

#[test]
fn index_with_m_vectors() {
    // Exactly M vectors - boundary case for neighbor lists
    let dim = 32;
    let m = 16;
    let mut hnsw = HNSWIndex::new(dim, m, m).expect("Failed to create");

    let vectors: Vec<Vec<f32>> = (0..m)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|d| ((i + d) as f32 * 0.1).sin()).collect();
            normalize(&v)
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&vectors[0], m, 50).expect("Search failed");
    assert_eq!(results.len(), m);
}

// =============================================================================
// Special vector patterns
// =============================================================================

#[test]
fn identical_vectors() {
    let dim = 32;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    // All vectors are identical
    let base = normalize(&vec![1.0; dim]);
    for i in 0..10 {
        hnsw.add(i as u32, base.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&base, 5, 50).expect("Search failed");

    // All results should have distance ~0
    for (_, dist) in &results {
        assert!(*dist < 0.01, "Identical vectors should have ~0 distance");
    }
}

#[test]
fn nearly_identical_vectors() {
    let dim = 64;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let base: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();

    // Add slightly perturbed versions
    for i in 0..50 {
        let mut v = base.clone();
        v[i % dim] += 1e-5 * (i as f32);
        hnsw.add(i as u32, normalize(&v)).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let results = hnsw
        .search(&normalize(&base), 10, 100)
        .expect("Search failed");
    assert_eq!(results.len(), 10);
}

#[test]
fn well_clustered_vectors() {
    // Create two distinct clusters
    let dim = 32;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    // Cluster 1: centered around [1, 0, 0, ...]
    for i in 0..25 {
        let mut v = vec![0.0; dim];
        v[0] = 1.0;
        v[(i % (dim - 1)) + 1] = 0.1;
        hnsw.add(i as u32, normalize(&v)).expect("Failed to add");
    }

    // Cluster 2: centered around [-1, 0, 0, ...]
    for i in 25..50 {
        let mut v = vec![0.0; dim];
        v[0] = -1.0;
        v[(i % (dim - 1)) + 1] = 0.1;
        hnsw.add(i as u32, normalize(&v)).expect("Failed to add");
    }

    hnsw.build().expect("Failed to build");

    // Query from cluster 1
    let mut query = vec![0.0; dim];
    query[0] = 1.0;
    let results = hnsw
        .search(&normalize(&query), 10, 100)
        .expect("Search failed");

    // Should mostly find cluster 1 vectors (indices 0-24)
    let cluster1_count = results.iter().filter(|(i, _)| *i < 25).count();
    assert!(
        cluster1_count >= 8,
        "Should find mostly cluster 1 vectors, got {}/10",
        cluster1_count
    );
}

// =============================================================================
// Query edge cases
// =============================================================================

#[test]
fn query_not_in_index() {
    let dim = 32;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    // Add only positive vectors
    for i in 0..30 {
        let v: Vec<f32> = (0..dim).map(|d| ((i + d) as f32).abs()).collect();
        hnsw.add(i as u32, normalize(&v)).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    // Query with a negative vector (opposite direction)
    let query: Vec<f32> = (0..dim).map(|d| -(d as f32 + 1.0)).collect();
    let results = hnsw
        .search(&normalize(&query), 5, 50)
        .expect("Search failed");

    assert_eq!(results.len(), 5);
    // Distances should be high (opposite direction)
    assert!(
        results[0].1 > 0.5,
        "Query in opposite direction should have high distance"
    );
}

#[test]
fn multiple_queries_returns_results() {
    let dim = 32;
    let mut hnsw = HNSWIndex::new(dim, 32, 32).expect("Failed to create");

    // Use normalized distinct vectors
    let vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let angle = i as f32 * 0.2;
            let mut v = vec![0.0; dim];
            v[0] = angle.cos();
            v[1] = angle.sin();
            // Add small variation in other dimensions
            for (d, val) in v.iter_mut().enumerate().skip(2) {
                *val = (d as f32 * 0.01) * (i as f32 * 0.1).sin();
            }
            normalize(&v)
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    // Run queries and verify we get results
    for (i, query_vec) in vectors.iter().enumerate().take(10) {
        let results = hnsw.search(query_vec, 5, 100).expect("Search failed");
        assert_eq!(results.len(), 5, "Should return 5 results for query {}", i);

        // Results should be sorted by distance
        for j in 1..results.len() {
            assert!(
                results[j].1 >= results[j - 1].1 - 1e-5,
                "Results not sorted at query {}: {} > {}",
                i,
                results[j - 1].1,
                results[j].1
            );
        }
    }
}

// =============================================================================
// Parameter edge cases
// =============================================================================

#[test]
fn small_ef_search() {
    let dim = 32;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let vectors: Vec<Vec<f32>> = (0..50)
        .map(|i| normalize(&vec![(i + 1) as f32 * 0.1; dim]))
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    // Very small ef_search
    let results = hnsw.search(&vectors[25], 5, 5).expect("Search failed");

    // Should still return 5 results
    assert_eq!(results.len(), 5);
}

#[test]
fn large_ef_search() {
    let dim = 32;
    let n = 100;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| normalize(&vec![(i + 1) as f32 * 0.1; dim]))
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    // ef_search larger than index size
    let results = hnsw.search(&vectors[50], 10, 500).expect("Search failed");
    assert_eq!(results.len(), 10);
}

#[test]
fn k_equals_n() {
    let dim = 32;
    let n = 50;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    // Use distinct vectors
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut v = vec![0.0; dim];
            v[i % dim] = 1.0;
            v[(i + 7) % dim] = 0.5;
            normalize(&v)
        })
        .collect();

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    // Request all vectors with high ef
    let results = hnsw.search(&vectors[0], n, 200).expect("Search failed");

    // Should return at least most vectors (HNSW may miss some with low connectivity)
    assert!(
        results.len() >= n - 5,
        "Should return most vectors, got {}/{}",
        results.len(),
        n
    );
}

// =============================================================================
// External-label-id round-trip edge cases
//
// vicinity returns the user-provided `doc_id` directly via
// `doc_id_to_internal: HashMap<u32, usize>`. A regression that returned the
// internal slot index instead of the external `doc_id` would be silently
// wrong. These tests pin the external-id contract at the high end of the u32
// range, where a wraparound or off-by-one is most likely to surface.
// =============================================================================

#[test]
fn search_returns_external_doc_ids_at_high_offset() {
    let dim = 16;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let offset: u32 = u32::MAX - 200;
    let n: u32 = 100;

    // Deterministic vectors via a simple LCG so the test does not depend on
    // a thread-local RNG.
    let mut state: u64 = 0xCAFE_F00D_DEAD_BEEF;
    let mut next_f32 = || -> f32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let v = normalize(&v);
        hnsw.add(offset.wrapping_add(i), v)
            .expect("add at high offset should succeed");
    }
    hnsw.build().expect("Failed to build");

    let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
    let query = normalize(&query);
    let results = hnsw.search(&query, 5, 50).expect("Search failed");

    assert!(!results.is_empty(), "should return some results");
    for (id, _dist) in &results {
        assert!(
            *id >= offset,
            "got id {} below offset {}; likely returning internal slot index instead of external doc_id",
            id,
            offset
        );
        // Also ensure the id is one we actually inserted.
        assert!(
            id.wrapping_sub(offset) < n,
            "id {} is past the inserted range [{}, {}]",
            id,
            offset,
            offset.wrapping_add(n)
        );
    }
}

// =============================================================================
// Degenerate query edge cases
//
// `src/hnsw/graph.rs:1406` documents that zero vectors are accepted under
// `auto_normalize` ("degenerate but valid"). Without `auto_normalize`, the
// search path normalizes the query in cosine mode and divides by the norm.
// A zero query is a 0/0 path; the contract is "either reject explicitly or
// return finite distances", not panic.
// =============================================================================

#[test]
fn zero_query_does_not_panic_with_auto_normalize() {
    let dim = 16;
    let mut hnsw = HNSWIndex::builder(dim)
        .m(16)
        .ef_search(50)
        .auto_normalize(true)
        .build()
        .expect("Failed to build builder");

    // Populate with normal data.
    let mut state: u64 = 1;
    let mut next_f32 = || -> f32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
    };
    for i in 0..50u32 {
        let v: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        hnsw.add_slice(i, &v).expect("add should succeed");
    }
    hnsw.build().expect("Failed to build");

    // The all-zero query.
    let zero = vec![0.0_f32; dim];
    let result = hnsw.search(&zero, 5, 50);

    match result {
        Ok(r) => {
            // If the search succeeded, every distance must be finite; the
            // contract bans NaN/Inf escaping back to the caller even on a
            // 0/0 normalization path.
            for (id, dist) in &r {
                assert!(
                    dist.is_finite(),
                    "doc_id {} returned non-finite distance {}",
                    id,
                    dist
                );
            }
        }
        Err(_) => {
            // Explicit rejection is also a valid contract; documenting the
            // path is acceptable.
        }
    }
}
