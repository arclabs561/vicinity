//! Property-based tests for HNSW persistence roundtrips.
//!
//! Verifies the "KeyedVectors separation" principle: external doc_ids survive
//! persistence unchanged, vectors survive SoA/AoS roundtrip, and search
//! results only contain IDs that were originally added.

#![cfg(all(feature = "persistence", feature = "hnsw"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/mod.rs"]
mod common;
use common::{normalize, random_vectors};

use proptest::prelude::*;
use std::collections::HashSet;
use vicinity::hnsw::HNSWIndex;
use vicinity::persistence::directory::MemoryDirectory;
use vicinity::persistence::hnsw::{HNSWSegmentReader, HNSWSegmentWriter};

/// Generate unique doc_ids that are NOT sequential from 0.
/// Tests that we're truly using external IDs, not internal indices.
fn nonsequential_doc_ids(n: usize, seed: u64) -> Vec<u32> {
    use std::hash::{Hash, Hasher};

    let mut ids: HashSet<u32> = HashSet::new();
    let mut i = 0u64;
    while ids.len() < n {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        i.hash(&mut hasher);
        let h = hasher.finish();
        let id = 1000 + (h % 99_000) as u32;
        ids.insert(id);
        i += 1;
    }
    ids.into_iter().collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// External IDs survive write/load roundtrip.
    #[test]
    fn persistence_preserves_doc_ids(
        n in 10usize..50,
        seed in any::<u64>(),
    ) {
        let dim = 8;
        let vectors: Vec<Vec<f32>> = random_vectors(n, dim, seed).into_iter().map(|v| normalize(&v)).collect();
        let doc_ids = nonsequential_doc_ids(n, seed);

        let mut original = HNSWIndex::new(dim, 8, 8).expect("create");
        for (i, v) in vectors.iter().enumerate() {
            original.add(doc_ids[i], v.clone()).expect("add");
        }
        original.build().expect("build");

        let mem = MemoryDirectory::new();
        let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 1);
        writer.write_hnsw_index(&original).expect("write");

        let reader = HNSWSegmentReader::load(Box::new(mem.clone()), 1).expect("load");
        let loaded = reader.load_index().expect("load_index");

        let query = &vectors[0];
        let k = 5.min(n);
        let ef = 50;

        let original_results = original.search(query, k, ef).expect("search original");
        let loaded_results = loaded.search(query, k, ef).expect("search loaded");

        let original_ids: HashSet<u32> = original_results.iter().map(|(id, _)| *id).collect();
        let loaded_ids: HashSet<u32> = loaded_results.iter().map(|(id, _)| *id).collect();

        let doc_id_set: HashSet<u32> = doc_ids.iter().copied().collect();
        for id in &loaded_ids {
            prop_assert!(
                doc_id_set.contains(id),
                "Search returned ID {} which was never added (internal index leak?)",
                id
            );
        }

        prop_assert_eq!(
            original_ids, loaded_ids,
            "Loaded index returns different doc_ids than original"
        );
    }

    /// Vectors survive SoA/AoS roundtrip without numerical corruption.
    #[test]
    fn persistence_vector_roundtrip(
        n in 10usize..40,
        dim in 8usize..32,
        seed in any::<u64>(),
    ) {
        let vectors: Vec<Vec<f32>> = random_vectors(n, dim, seed).into_iter().map(|v| normalize(&v)).collect();

        let mut original = HNSWIndex::new(dim, 16, 16).expect("create");
        for (i, v) in vectors.iter().enumerate() {
            original.add(i as u32, v.clone()).expect("add");
        }
        original.build().expect("build");

        let mem = MemoryDirectory::new();
        let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 1);
        writer.write_hnsw_index(&original).expect("write");

        let reader = HNSWSegmentReader::load(Box::new(mem.clone()), 1).expect("load");
        let loaded = reader.load_index().expect("load_index");

        let k = 5.min(n);
        let ef = n * 2;

        for (i, v) in vectors.iter().enumerate().take(5) {
            let original_results = original.search(v, k, ef).expect("search original");
            let loaded_results = loaded.search(v, k, ef).expect("search loaded");

            let original_ids: Vec<u32> = original_results.iter().map(|(id, _)| *id).collect();
            let loaded_ids: Vec<u32> = loaded_results.iter().map(|(id, _)| *id).collect();

            prop_assert_eq!(
                original_ids, loaded_ids,
                "Query {} returns different results after persistence",
                i
            );

            for ((_, orig_dist), (_, loaded_dist)) in original_results.iter().zip(loaded_results.iter()) {
                let diff = (orig_dist - loaded_dist).abs();
                prop_assert!(
                    diff < 1e-5,
                    "Distance mismatch after persistence: {} vs {}",
                    orig_dist, loaded_dist
                );
            }
        }
    }

    /// Search only returns IDs that were added (domain closure).
    #[test]
    fn search_returns_only_known_doc_ids(
        n in 20usize..60,
        seed in any::<u64>(),
    ) {
        let dim = 16;
        let vectors: Vec<Vec<f32>> = random_vectors(n, dim, seed).into_iter().map(|v| normalize(&v)).collect();
        let doc_ids = nonsequential_doc_ids(n, seed);
        let doc_id_set: HashSet<u32> = doc_ids.iter().copied().collect();

        let mut index = HNSWIndex::new(dim, 16, 16).expect("create");
        for (i, v) in vectors.iter().enumerate() {
            index.add(doc_ids[i], v.clone()).expect("add");
        }
        index.build().expect("build");

        for query_idx in [0, n/4, n/2, 3*n/4, n-1].iter().filter(|&&i| i < n) {
            let query = &vectors[*query_idx];
            let results = index.search(query, 10.min(n), 100).expect("search");

            for (id, _dist) in &results {
                prop_assert!(
                    doc_id_set.contains(id),
                    "Search returned unknown ID {}: internal index leaked through API",
                    id
                );
            }
        }
    }

    /// Adding the same doc_id twice should fail.
    #[test]
    fn duplicate_doc_id_rejected(
        n in 5usize..20,
        seed in any::<u64>(),
    ) {
        let dim = 8;
        let vectors: Vec<Vec<f32>> = random_vectors(n + 1, dim, seed).into_iter().map(|v| normalize(&v)).collect();
        let doc_ids = nonsequential_doc_ids(n, seed);

        let mut index = HNSWIndex::new(dim, 8, 8).expect("create");

        for (i, v) in vectors.iter().take(n).enumerate() {
            index.add(doc_ids[i], v.clone()).expect("add");
        }

        let duplicate_id = doc_ids[0];
        let different_vector = vectors[n].clone();
        let result = index.add(duplicate_id, different_vector);

        prop_assert!(
            result.is_err(),
            "Adding duplicate doc_id {} should fail but succeeded",
            duplicate_id
        );
    }

    /// Metadata (dimension, num_vectors) survives roundtrip.
    #[test]
    fn persistence_metadata_roundtrip(
        n in 10usize..40,
        dim in 4usize..64,
        seed in any::<u64>(),
    ) {
        let vectors: Vec<Vec<f32>> = random_vectors(n, dim, seed).into_iter().map(|v| normalize(&v)).collect();

        let mut original = HNSWIndex::new(dim, 8, 8).expect("create");
        for (i, v) in vectors.iter().enumerate() {
            original.add(i as u32, v.clone()).expect("add");
        }
        original.build().expect("build");

        let mem = MemoryDirectory::new();
        let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 1);
        writer.write_hnsw_index(&original).expect("write");

        let reader = HNSWSegmentReader::load(Box::new(mem.clone()), 1).expect("load");
        let loaded = reader.load_index().expect("load_index");

        let query = normalize(&vectors[0]);
        let results = loaded.search(&query, 5.min(n), 50);
        prop_assert!(results.is_ok(), "Search failed on loaded index");
        prop_assert_eq!(
            results.unwrap().len(),
            5.min(n),
            "Loaded index returns wrong number of results"
        );
    }
}
