use super::*;

#[test]
fn test_create_index() {
    let index = HNSWIndex::new(128, 16, 16).unwrap();
    assert_eq!(index.dimension, 128);
    assert_eq!(index.num_vectors, 0);
}

#[test]
fn test_add_vectors() {
    let mut index = HNSWIndex::new(3, 16, 16).unwrap();

    index.add(0, vec![1.0, 0.0, 0.0]).unwrap();
    index.add(1, vec![0.0, 1.0, 0.0]).unwrap();

    assert_eq!(index.num_vectors, 2);
}

#[test]
fn test_dimension_mismatch() {
    let mut index = HNSWIndex::new(3, 16, 16).unwrap();

    let result = index.add(0, vec![1.0, 0.0]); // Wrong dimension
    assert!(result.is_err());
}

/// Build a small index for adaptive search tests.
/// Returns (index, query_vector).
fn build_test_index() -> (HNSWIndex, Vec<f32>) {
    let dim = 32;
    let n = 200;
    let mut index = HNSWIndex::new(dim, 16, 32).unwrap();

    // Deterministic pseudo-random vectors using LCG
    let mut seed: u64 = 42;
    let mut next = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    for i in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
        // L2-normalize for cosine distance
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        index.add(i as u32, v).unwrap();
    }
    index.build().unwrap();

    let mut q: Vec<f32> = (0..dim).map(|_| next()).collect();
    let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        q.iter_mut().for_each(|x| *x /= norm);
    }

    (index, q)
}

#[test]
fn test_search_adaptive_conservative_matches_search() {
    let (index, q) = build_test_index();
    let k = 5;
    let ef = 64;

    let baseline = index.search(&q, k, ef).unwrap();
    let config = crate::adaptive::AdaptiveConfig::conservative();
    let (adaptive, _evaluated) = index.search_adaptive(&q, k, ef, &config).unwrap();

    // Conservative config should return the same top-k (or very close)
    assert_eq!(adaptive.len(), baseline.len());
    // At minimum the nearest neighbor should match
    assert_eq!(adaptive[0].0, baseline[0].0);
}

#[test]
fn test_search_adaptive_aggressive_fewer_evaluations() {
    let (index, q) = build_test_index();
    let k = 5;
    let ef = 64;

    let conservative = crate::adaptive::AdaptiveConfig::conservative();
    let aggressive = crate::adaptive::AdaptiveConfig::aggressive();

    let (_results_c, evaluated_c) = index.search_adaptive(&q, k, ef, &conservative).unwrap();
    let (_results_a, evaluated_a) = index.search_adaptive(&q, k, ef, &aggressive).unwrap();

    // Aggressive config should evaluate fewer (or equal) candidates
    assert!(
        evaluated_a <= evaluated_c,
        "aggressive ({}) should evaluate <= conservative ({})",
        evaluated_a,
        evaluated_c,
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_hnsw_save_load_roundtrip() {
    let (index, q) = build_test_index();
    let k = 5;
    let ef = 64;

    // Search the original index.
    let original_results = index.search(&q, k, ef).unwrap();

    // Serialize to an in-memory buffer.
    let mut buf = Vec::new();
    index.save_to_writer(&mut buf).unwrap();
    assert!(!buf.is_empty(), "serialized output should not be empty");

    // Deserialize back.
    let loaded = HNSWIndex::load_from_reader(buf.as_slice()).unwrap();

    // Basic structural checks.
    assert_eq!(loaded.dimension, index.dimension);
    assert_eq!(loaded.num_vectors, index.num_vectors);
    assert!(loaded.is_built());
    assert_eq!(loaded.doc_ids, index.doc_ids);
    // Reverse map should have been rebuilt.
    assert_eq!(
        loaded.doc_id_to_internal.len(),
        index.doc_id_to_internal.len()
    );

    // Search the loaded index and compare results.
    let loaded_results = loaded.search(&q, k, ef).unwrap();
    assert_eq!(
        loaded_results, original_results,
        "search results should be identical after save/load roundtrip"
    );
}

/// Round-trip via the path-based API (covers `save_to_file` /
/// `load_from_file`, which the in-memory roundtrip above does not).
/// Also asserts that `save_to_file` leaves no `.tmp` sibling on
/// success -- if it does, the temp+rename atomicity got broken
/// (e.g., the rename was skipped or the temp name doesn't match).
#[cfg(feature = "serde")]
#[test]
fn test_hnsw_save_to_file_roundtrip_and_no_temp_leftover() {
    let (index, q) = build_test_index();
    let k = 5;
    let ef = 64;
    let original_results = index.search(&q, k, ef).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.json");
    index.save_to_file(&path).unwrap();

    // Final file must exist; the sibling .tmp must not.
    assert!(path.exists(), "saved file should exist at target path");
    let tmp_path = dir.path().join("index.json.tmp");
    assert!(
        !tmp_path.exists(),
        "save_to_file left a .tmp sibling: {}",
        tmp_path.display()
    );

    let loaded = HNSWIndex::load_from_file(&path).unwrap();
    let loaded_results = loaded.search(&q, k, ef).unwrap();
    assert_eq!(loaded_results, original_results);
}

/// Atomicity guarantee: when `save_to_file` overwrites an existing
/// file, the prior contents are replaced wholesale rather than
/// partially truncated. The check is indirect (the bytes match the
/// new serialization, not the old) but the failure shape if the
/// rename were skipped would be a file containing the OLD payload
/// or a truncated mix.
#[cfg(feature = "serde")]
#[test]
fn test_hnsw_save_to_file_overwrites_existing_atomically() {
    let (index, _q) = build_test_index();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.json");

    // Stage a sentinel "previous" file that is clearly not a valid index.
    std::fs::write(&path, b"{\"sentinel\": true}").unwrap();
    let pre_size = std::fs::metadata(&path).unwrap().len();

    index.save_to_file(&path).unwrap();

    // The replacement must be a valid index payload, not the sentinel.
    let post_size = std::fs::metadata(&path).unwrap().len();
    assert!(
        post_size > pre_size,
        "post-save size {} should exceed sentinel size {}",
        post_size,
        pre_size
    );
    let _ = HNSWIndex::load_from_file(&path).expect("post-overwrite file must load");
    // No temp leftover.
    assert!(!dir.path().join("index.json.tmp").exists());
}

/// Negative-path cleanup guarantee: when `save_to_file` fails at the
/// rename step, it must not leak the `<path>.tmp` sibling. We force the
/// rename to fail by pre-creating a directory at the target path; POSIX
/// rename refuses to replace a non-empty directory with a file (EISDIR /
/// ENOTEMPTY), and Windows MoveFileEx refuses similarly. The exact errno
/// is platform-dependent so we don't assert on the error variant; we
/// only assert the cleanup happened.
///
/// Without the `remove_file(&tmp_path)` line in the rename-error branch,
/// callers retrying the save would race against a stale temp from the
/// previous attempt or accumulate junk files in the target directory.
#[cfg(feature = "serde")]
#[test]
fn test_hnsw_save_to_file_cleans_up_tmp_on_rename_failure() {
    let (index, _q) = build_test_index();
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("save.json");

    // Pre-create a directory at the target path so the rename of
    // `<target>.tmp -> <target>` fails. The directory must be non-empty
    // on some platforms to guarantee EISDIR/ENOTEMPTY; an inner sentinel
    // file ensures the rename can't succeed by accident.
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("blocker"), b"x").unwrap();

    let result = index.save_to_file(&target);
    assert!(
        result.is_err(),
        "rename into a non-empty directory should fail"
    );

    let tmp_path = dir.path().join("save.json.tmp");
    assert!(
        !tmp_path.exists(),
        "save_to_file failed but left a temp file at {} \
             (rename-failure cleanup branch did not fire)",
        tmp_path.display(),
    );
}

/// One-shot measurement of save_to_file overhead vs a non-atomic
/// File::create+write baseline. The atomic path adds: temp file
/// creation, sync_all on the temp, rename, parent-dir sync_all.
/// On SSD, sync_all is the dominant cost (single-digit ms typical).
/// For a JSON-serialized HNSW index in the 100KB-few-MB range, the
/// total is dominated by serialization + write throughput, not the
/// extra fsyncs. This probe prints both numbers so future work can
/// compare. Marked `#[ignore]` because it's a measurement, not a
/// regression guard.
///
/// Run with:
///   cargo test --release --features hnsw,serde
///       hnsw::graph::tests::save_to_file_overhead_probe
///       -- --ignored --nocapture
#[cfg(feature = "serde")]
#[test]
#[ignore = "measurement only; run with --release --ignored --nocapture"]
fn save_to_file_overhead_probe() {
    use std::time::Instant;

    let (index, _q) = build_test_index();
    let dir = tempfile::tempdir().unwrap();

    // Atomic path (current code): temp + sync + rename + dir sync.
    let iters = 30;
    let path = dir.path().join("atomic.json");
    let t = Instant::now();
    for _ in 0..iters {
        index.save_to_file(&path).unwrap();
    }
    let atomic_ms_per = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Non-atomic baseline: File::create + write + drop. No syncs.
    let path2 = dir.path().join("naive.json");
    let t = Instant::now();
    for _ in 0..iters {
        let f = std::fs::File::create(&path2).unwrap();
        index.save_to_writer(std::io::BufWriter::new(f)).unwrap();
    }
    let naive_ms_per = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let file_kb = std::fs::metadata(&path).unwrap().len() / 1024;
    println!("# save_to_file_overhead_probe");
    println!(
        "# build_test_index ({} vectors, {} dim, ~{} KB JSON)",
        index.num_vectors, index.dimension, file_kb
    );
    println!("# atomic   {:.2} ms/op", atomic_ms_per);
    println!("# naive    {:.2} ms/op", naive_ms_per);
    println!(
        "# overhead {:.2} ms/op ({:+.0}%)",
        atomic_ms_per - naive_ms_per,
        100.0 * (atomic_ms_per - naive_ms_per) / naive_ms_per
    );
}

/// End-to-end build → save_to_file → drop → load_from_file → search
/// equality. The in-memory roundtrip
/// (`test_hnsw_save_load_roundtrip`) covers the
/// `save_to_writer`/`load_from_reader` half; the v0 fixture decode
/// tests pin a static byte sequence; this test plugs the gap by
/// proving the writer and reader are mutually consistent through the
/// path-based API on a freshly built live index.
///
/// qdrant ships an analogous round-trip test for its `GraphLayers`
/// persistence; this is the same shape adapted to vicinity's HNSW.
#[cfg(feature = "serde")]
#[test]
fn test_hnsw_build_save_reload_search_equality() {
    let (index, q) = build_test_index();
    let k = 10;
    let ef = 64;
    let original_results = index.search(&q, k, ef).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.json");
    index.save_to_file(&path).unwrap();
    // Drop the in-memory copy explicitly so the load is from disk,
    // not aliasing.
    drop(index);

    let loaded = HNSWIndex::load_from_file(&path).expect("load_from_file roundtrip");
    let loaded_results = loaded.search(&q, k, ef).unwrap();

    assert_eq!(
        loaded_results, original_results,
        "search results diverged across path-based save/load roundtrip. \
             likely cause: writer and reader formats drifted (one side \
             changed without the other), or a transient field was \
             dropped during serialization that affects search behavior."
    );
}

/// Gate-fired check that `save_to_file` actually uses temp+rename and
/// not in-place truncation. Truncate-overwrite preserves the inode;
/// rename-overwrite replaces it with the temp's inode. This distinction
/// is invisible to the happy-path tests above but is the entire point
/// of the atomicity refactor: a crash mid-truncate corrupts the file,
/// while a crash mid-temp-write leaves the original intact.
#[cfg(all(feature = "serde", unix))]
#[test]
fn test_hnsw_save_to_file_uses_rename_not_truncate() {
    use std::os::unix::fs::MetadataExt;

    let (index, _q) = build_test_index();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.json");

    // First save: gives us a baseline file with an inode.
    index.save_to_file(&path).unwrap();
    let ino_before = std::fs::metadata(&path).unwrap().ino();

    // Second save over the same path. With temp+rename, the inode
    // changes (the new file is the temp file under its new name);
    // with in-place truncation, the inode stays the same.
    index.save_to_file(&path).unwrap();
    let ino_after = std::fs::metadata(&path).unwrap().ino();

    assert_ne!(
        ino_before, ino_after,
        "save_to_file must replace via rename (inode changes), not truncate \
             in place (inode preserved). truncate-overwrite would leave the file \
             in a partially-written state if a crash interrupts the write -- this \
             is exactly the partial-write corruption mode the atomic-rename refactor \
             is meant to prevent."
    );
}

#[test]
fn test_delete_excludes_from_results() {
    let (mut index, q) = build_test_index();
    let k = 10;
    let ef = 64;

    let before = index.search(&q, k, ef).unwrap();
    assert!(!before.is_empty());

    // Delete the nearest neighbor
    let nearest_id = before[0].0;
    index.delete(nearest_id).unwrap();

    let after = index.search(&q, k, ef).unwrap();
    let after_ids: Vec<u32> = after.iter().map(|(id, _)| *id).collect();
    assert!(
        !after_ids.contains(&nearest_id),
        "deleted doc_id {} should not appear in results",
        nearest_id
    );
}

#[test]
fn test_delete_all_returns_empty() {
    let dim = 4;
    let mut index = HNSWIndex::new(dim, 4, 4).unwrap();

    // Insert 5 normalized vectors
    for i in 0..5u32 {
        let mut v = vec![0.0f32; dim];
        v[i as usize % dim] = 1.0;
        index.add(i, v).unwrap();
    }
    index.build().unwrap();

    // Delete all
    for i in 0..5u32 {
        index.delete(i).unwrap();
    }

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5, 50).unwrap();
    assert!(
        results.is_empty(),
        "all vectors deleted, results should be empty"
    );
}

#[test]
fn test_delete_nonexistent_returns_error() {
    let mut index = HNSWIndex::new(4, 4, 4).unwrap();
    index.add(0, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    index.build().unwrap();

    let result = index.delete(999);
    assert!(result.is_err(), "deleting nonexistent doc_id should error");
}

#[test]
fn test_delete_idempotent() {
    let (mut index, _q) = build_test_index();

    // Delete same ID twice -- second should succeed silently
    index.delete(0).unwrap();
    index.delete(0).unwrap();
}

#[test]
fn test_is_deleted() {
    let (mut index, _q) = build_test_index();
    assert!(!index.is_deleted(0));
    index.delete(0).unwrap();
    assert!(index.is_deleted(0));
    assert!(!index.is_deleted(1));
}

#[test]
fn test_num_active() {
    let (mut index, _q) = build_test_index();
    let total = index.num_vectors;
    assert_eq!(index.num_active(), total);

    index.delete(0).unwrap();
    index.delete(1).unwrap();
    assert_eq!(index.num_active(), total - 2);
}

#[test]
fn test_builder_produces_working_index() {
    let mut index = HNSWIndex::builder(4)
        .m(8)
        .ef_search(32)
        .auto_normalize(true)
        .build()
        .unwrap();

    // auto_normalize is on, so raw (un-normalized) vectors should be accepted
    // for both add and search.
    index.add_slice(0, &[3.0, 4.0, 0.0, 0.0]).unwrap();
    index.add_slice(1, &[0.0, 0.0, 3.0, 4.0]).unwrap();
    index.build().unwrap();

    let results = index.search(&[3.0, 4.0, 0.0, 0.0], 1, 32).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 0, "nearest neighbor should be doc 0");
    assert!(
        results[0].1 < 0.01,
        "self-distance after normalization should be ~0, got {}",
        results[0].1
    );
}

#[test]
fn test_auto_normalize_symmetric_for_angular() {
    // Angular metric: parallel to the Python binding's
    // test_auto_normalize_symmetric_for_angular. Mirrors the cosine case
    // above for the second metric the flag covers.
    let mut index = HNSWIndex::builder(4)
        .m(8)
        .ef_search(32)
        .metric(DistanceMetric::Angular)
        .auto_normalize(true)
        .build()
        .unwrap();

    index.add_slice(0, &[3.0, 4.0, 0.0, 0.0]).unwrap();
    index.add_slice(1, &[0.0, 0.0, 3.0, 4.0]).unwrap();
    index.build().unwrap();

    // Un-normalized query (norm = 5); auto_normalize must apply on
    // search too or angular distance reads as ≫ 0 against itself.
    let results = index.search(&[3.0, 4.0, 0.0, 0.0], 1, 32).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 0, "nearest neighbor should be doc 0");
    assert!(
        results[0].1 < 0.05,
        "angular self-distance after normalization should be ~0, got {}",
        results[0].1
    );
}

#[cfg(feature = "parallel")]
#[test]
fn test_search_batch_matches_sequential() {
    let (index, q) = build_test_index();
    let k = 5;
    let ef = 64;

    // Create several query vectors by rotating the base query
    let dim = index.dimension;
    let mut queries_flat: Vec<f32> = Vec::new();
    let num_queries = 8;

    let mut seed: u64 = 99;
    let mut next = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    for _ in 0..num_queries {
        let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        queries_flat.extend_from_slice(&v);
    }

    // Sequential results
    let sequential: Vec<Vec<(u32, f32)>> = (0..num_queries)
        .map(|i| {
            let query = &queries_flat[i * dim..(i + 1) * dim];
            index.search(query, k, ef).unwrap()
        })
        .collect();

    // Batch (parallel) results -- slice-of-slices variant
    let query_slices: Vec<&[f32]> = (0..num_queries)
        .map(|i| &queries_flat[i * dim..(i + 1) * dim])
        .collect();
    let batch = index.search_batch(&query_slices, k, ef).unwrap();

    // Batch (parallel) results -- flat-buffer variant
    let batch_flat = index
        .search_batch_flat(&queries_flat, num_queries, k, ef)
        .unwrap();

    // Also include the original query to make sure it still works
    let _ = index.search(&q, k, ef).unwrap();

    for i in 0..num_queries {
        assert_eq!(
            sequential[i], batch[i],
            "search_batch result {} differs from sequential",
            i
        );
        assert_eq!(
            sequential[i], batch_flat[i],
            "search_batch_flat result {} differs from sequential",
            i
        );
    }
}

/// Build a small index for structural invariant tests.
fn build_structural_test_index(n: usize, dim: usize, m: usize) -> HNSWIndex {
    let params = HNSWParams {
        m,
        m_max: 2 * m,
        seed: Some(42),
        ..Default::default()
    };
    let mut index = HNSWIndex::with_params(dim, params).unwrap();
    for i in 0..n {
        let v: Vec<f32> = (0..dim)
            .map(|j| ((i * 7 + j * 3) % 100) as f32 / 100.0)
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normed: Vec<f32> = v.iter().map(|x| x / norm).collect();
        index.add(i as u32, normed).unwrap();
    }
    index.build().unwrap();
    index
}

#[test]
fn test_layer_assignment_distribution() {
    // With 500 vectors, we should see layer 0 most populated and
    // exponentially fewer vectors in higher layers.
    let index = build_structural_test_index(500, 32, 16);

    let max_layer = *index.layer_assignments.iter().max().unwrap_or(&0) as usize;

    let mut layer_counts = vec![0usize; max_layer + 1];
    for &l in &index.layer_assignments {
        layer_counts[l as usize] += 1;
    }

    // With m_l ≈ 0.36 (default for M=16), ~94% of vectors land on layer 0.
    // Use a generous threshold to avoid flaking on unlucky seeds.
    let layer0_frac = layer_counts[0] as f64 / 500.0;
    assert!(
        layer0_frac > 0.5,
        "Layer 0 fraction {:.2} should be > 0.5 (got {} of 500)",
        layer0_frac,
        layer_counts[0]
    );

    // Layer 0 must have strictly the most vectors
    for l in 1..layer_counts.len() {
        assert!(
            layer_counts[l] < layer_counts[0],
            "Layer {} ({}) should have fewer vectors than layer 0 ({})",
            l,
            layer_counts[l],
            layer_counts[0]
        );
    }
}

#[test]
fn test_m_max_enforced() {
    // Verify no node in layer 0 exceeds m_max neighbors,
    // and no node in upper layers exceeds m.
    let m = 8;
    let m_max = 2 * m;
    let index = build_structural_test_index(200, 16, m);

    for (layer_idx, layer) in index.layers.iter().enumerate() {
        let limit = if layer_idx == 0 { m_max } else { m };
        if let Some(neighbors) = layer.get_all_neighbors() {
            for (node_id, nbrs) in neighbors.iter().enumerate() {
                assert!(
                    nbrs.len() <= limit,
                    "Node {} at layer {} has {} neighbors, limit is {}",
                    node_id,
                    layer_idx,
                    nbrs.len(),
                    limit
                );
            }
        }
    }
}

#[test]
fn test_neighbor_ids_in_bounds() {
    // All neighbor IDs must reference valid nodes.
    let index = build_structural_test_index(100, 16, 8);

    for (layer_idx, layer) in index.layers.iter().enumerate() {
        if let Some(neighbors) = layer.get_all_neighbors() {
            for (node_id, nbrs) in neighbors.iter().enumerate() {
                for &nbr in nbrs.iter() {
                    assert!(
                        (nbr as usize) < index.num_vectors,
                        "Node {} at layer {} has out-of-bounds neighbor {}",
                        node_id,
                        layer_idx,
                        nbr
                    );
                }
            }
        }
    }
}

#[test]
fn test_layer_assignment_matches_layers() {
    // Every node assigned to layer L should exist in layers 0..=L.
    let index = build_structural_test_index(100, 16, 8);

    for (node_id, &assigned_layer) in index.layer_assignments.iter().enumerate() {
        // Node should be present in layers 0 through assigned_layer
        for l in 0..=assigned_layer as usize {
            assert!(
                l < index.layers.len(),
                "Node {} assigned to layer {} but only {} layers exist",
                node_id,
                assigned_layer,
                index.layers.len()
            );
        }
    }
}

/// `batch_search_mqo` must return results in input-query order and find at
/// least as good a nearest neighbour as individual `search` calls.
#[test]
fn test_batch_search_mqo_order_and_recall() {
    let (index, _) = build_test_index();
    let k = 5;
    let ef = 64;

    // Build a small set of varied queries using the same deterministic LCG as
    // `build_test_index` but with a different seed so queries differ from the
    // indexed vectors.
    let dim = index.dimension;
    let mut seed: u64 = 99;
    let mut next = || -> f32 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    let num_queries = 10;
    let queries_owned: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        })
        .collect();
    let queries: Vec<&[f32]> = queries_owned.iter().map(|v| v.as_slice()).collect();

    // Individual searches (baseline).
    let individual: Vec<Vec<(u32, f32)>> = queries
        .iter()
        .map(|q| index.search(q, k, ef).unwrap())
        .collect();

    // Batch MQO search.
    let batch = index.batch_search_mqo(&queries, k, ef).unwrap();

    // Results must be in input order and have the right length.
    assert_eq!(batch.len(), num_queries);

    for (i, (ind, mqo)) in individual.iter().zip(batch.iter()).enumerate() {
        // MQO must return a result for every query.
        assert!(
            !mqo.is_empty(),
            "query {}: batch_search_mqo returned no results",
            i
        );
        // The nearest neighbour found by MQO must be at least as close as the
        // one from individual search (same ef, so recall can only stay equal or
        // improve with an additional warm-start entry point).
        let ind_best_dist = ind.first().map(|(_, d)| *d).unwrap_or(f32::INFINITY);
        let mqo_best_dist = mqo.first().map(|(_, d)| *d).unwrap_or(f32::INFINITY);
        assert!(
            mqo_best_dist <= ind_best_dist + 1e-5,
            "query {}: MQO nearest dist {} > individual nearest dist {} (regression)",
            i,
            mqo_best_dist,
            ind_best_dist,
        );
    }
}

/// Single-query batch_search_mqo must return the same result as search.
#[test]
fn test_batch_search_mqo_single_query() {
    let (index, q) = build_test_index();
    let k = 5;
    let ef = 64;

    let single = index.search(&q, k, ef).unwrap();
    let batch = index.batch_search_mqo(&[q.as_slice()], k, ef).unwrap();

    assert_eq!(batch.len(), 1);
    assert_eq!(batch[0], single, "single-query MQO should match search");
}

#[test]
fn test_search_with_distance_matches_standard() {
    let (index, q) = build_test_index();
    let k = 5;
    let ef = 64;

    let standard = index.search(&q, k, ef).unwrap();

    // Custom distance that does the same thing as the standard path
    let vectors = &index.vectors;
    let dim = index.dimension;
    let dist_fn = |query: &[f32], internal_id: u32| -> f32 {
        let start = internal_id as usize * dim;
        let vec = &vectors[start..start + dim];
        crate::distance::cosine_distance_normalized(query, vec)
    };
    let custom = index.search_with_distance(&q, k, ef, &dist_fn).unwrap();

    assert_eq!(standard.len(), custom.len());
    for (s, c) in standard.iter().zip(custom.iter()) {
        assert_eq!(s.0, c.0, "doc_ids should match");
        assert!((s.1 - c.1).abs() < 1e-6, "distances should match");
    }
}

#[test]
fn test_search_with_distance_custom_metric() {
    // Build index with cosine, search with L2 -- should reorder results
    let (index, q) = build_test_index();
    let k = 5;
    let ef = 64;

    let vectors = &index.vectors;
    let dim = index.dimension;
    let l2_dist = |query: &[f32], internal_id: u32| -> f32 {
        let start = internal_id as usize * dim;
        let vec = &vectors[start..start + dim];
        crate::distance::l2_distance(query, vec)
    };
    let results = index.search_with_distance(&q, k, ef, &l2_dist).unwrap();

    // Results should be sorted by L2 distance
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1, "results not sorted by custom distance");
    }
}

// -----------------------------------------------------------------------
// Wolverine++ delete_with_repair tests
// -----------------------------------------------------------------------

#[test]
fn test_delete_with_repair_excludes_from_results() {
    let (mut index, q) = build_test_index();
    let k = 10;
    let ef = 64;

    let before = index.search(&q, k, ef).unwrap();
    let nearest_id = before[0].0;

    let repairs = index.delete_with_repair(nearest_id).unwrap();

    let after = index.search(&q, k, ef).unwrap();
    let after_ids: Vec<u32> = after.iter().map(|(id, _)| *id).collect();
    assert!(
        !after_ids.contains(&nearest_id),
        "deleted doc_id {} should not appear in results (repairs={})",
        nearest_id,
        repairs,
    );
}

#[test]
fn test_delete_with_repair_maintains_recall() {
    let (mut index, q) = build_test_index();
    let k = 10;
    let ef = 100;

    let before = index.search(&q, k, ef).unwrap();
    assert_eq!(before.len(), k);

    // Delete 20% of nodes (IDs 0..40 out of 200)
    for i in 0..40u32 {
        index.delete_with_repair(i).unwrap();
    }

    let after = index.search(&q, k, ef).unwrap();
    // Should still return k results (160 nodes remain)
    assert_eq!(
        after.len(),
        k,
        "should still return k={} results after deleting 20%",
        k
    );

    // No deleted IDs in results
    for (id, _) in &after {
        assert!(*id >= 40, "deleted id {} appeared in results", id);
    }
}

/// Companion to `test_delete_with_repair_maintains_recall`.
///
/// The original test asserts the right *count* of results comes back and
/// that deleted IDs are absent. It does NOT verify recall against ground
/// truth on the surviving IDs. This test does:
///
/// 1. Generate the same 200 deterministic vectors used by `build_test_index`.
/// 2. Compute the brute-force top-`k` cosine neighbors of the query among
///    the surviving IDs `40..200`.
/// 3. Run the HNSW search after deletion.
/// 4. Assert the overlap (recall) between the brute-force ground truth and
///    the HNSW results is at least 0.7. The threshold is conservative for
///    a 200-node graph; the relevant signal is "no recall cliff", not "near 1.0".
#[test]
fn test_delete_with_repair_recall_floor_against_ground_truth() {
    let dim = 32usize;
    let n = 200u32;
    let deleted_count = 40u32;
    let k = 10usize;
    let ef = 100usize;
    let recall_floor = 0.7f32;

    // Reproduce the exact vector generation used by build_test_index().
    let mut seed: u64 = 42;
    let mut next = || -> f32 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        vectors.push(v);
    }
    // Same query construction as build_test_index().
    let mut q: Vec<f32> = (0..dim).map(|_| next()).collect();
    let qnorm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    if qnorm > 0.0 {
        q.iter_mut().for_each(|x| *x /= qnorm);
    }

    // Brute-force ground truth on the surviving IDs only.
    let mut ground: Vec<(u32, f32)> = (deleted_count..n)
        .map(|i| {
            let v = &vectors[i as usize];
            // 1 - dot product = cosine distance for unit vectors.
            let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            (i, 1.0 - dot)
        })
        .collect();
    ground.sort_by(|a, b| a.1.total_cmp(&b.1));
    let truth_ids: std::collections::HashSet<u32> =
        ground.iter().take(k).map(|(id, _)| *id).collect();

    // Build the index and apply the same deletions as the count test above.
    let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        index.add(i as u32, v.clone()).unwrap();
    }
    index.build().unwrap();
    for i in 0..deleted_count {
        index.delete_with_repair(i).unwrap();
    }

    // Recall = |hnsw_topk ∩ truth_topk| / k.
    let results = index.search(&q, k, ef).unwrap();
    assert_eq!(results.len(), k, "search should return k results");
    let hits = results
        .iter()
        .filter(|(id, _)| truth_ids.contains(id))
        .count();
    let recall = hits as f32 / k as f32;
    assert!(
        recall >= recall_floor,
        "post-delete recall@{}={:.2} fell below floor {:.2}; deleted_count={}, n={}",
        k,
        recall,
        recall_floor,
        deleted_count,
        n
    );
}

/// Self-search reachability under aggressive deletion.
///
/// Existing tests for `delete_with_repair` cover small deletion ratios
/// (20-30%) and assert recall against a ground-truth top-k or that the
/// result count survives. None probe the failure mode where many
/// repairs leave individual live nodes orphaned -- the analog of
/// FreshGraph's delete-reinsert reachability bug (arxiv:2407.07871).
///
/// This test deletes 60% of nodes and asserts every surviving doc_id
/// is still self-search-reachable. If a live node becomes unreachable
/// after repair (e.g., its only in-edges were through nodes that were
/// later deleted, and the crescent-locus replacement failed to add
/// new in-edges), the assertion lists the orphaned ids.
///
/// HNSW does not currently support post-build inserts, so this is a
/// long-delete test rather than a delete-reinsert cycle. The bug class
/// is the same: cumulative repair operations eroding reachability for
/// the surviving subgraph.
#[test]
fn test_delete_with_repair_preserves_self_search_reachability() {
    let dim = 32usize;
    let n = 300u32;
    let delete_ratio = 0.6f32;
    let deleted_count = (n as f32 * delete_ratio) as u32;
    let k = 5usize;
    let ef = 200usize;

    // Same deterministic LCG as build_test_index().
    let mut seed: u64 = 42;
    let mut next = || -> f32 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        vectors.push(v);
    }

    let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        index.add(i as u32, v.clone()).unwrap();
    }
    index.build().unwrap();

    // Delete the first `deleted_count` ids. Pseudo-random selection
    // would be more thorough but the IDs are already random with
    // respect to graph layout (assign_layer is the only ID-correlated
    // structure, and it's an exponential RV per node).
    for i in 0..deleted_count {
        index.delete_with_repair(i).unwrap();
    }

    // For each surviving id, self-search and check reachability.
    let mut unreachable: Vec<u32> = Vec::new();
    for id in deleted_count..n {
        let results = index.search(&vectors[id as usize], k, ef).unwrap();
        if !results.iter().any(|(rid, _)| *rid == id) {
            unreachable.push(id);
        }
    }

    assert!(
        unreachable.is_empty(),
        "{} of {} live ids became unreachable after deleting {} (ratio {:.0}%): {:?}. \
             likely cause: cumulative delete_with_repair calls left live nodes orphaned -- \
             crescent-locus replacement may have failed to add in-edges, or the entry-point \
             repromotion in find_entry_point_excluding picked an anchor whose neighborhood \
             has drifted away from these ids.",
        unreachable.len(),
        n - deleted_count,
        deleted_count,
        delete_ratio * 100.0,
        &unreachable[..unreachable.len().min(10)]
    );
}

#[test]
fn test_delete_with_repair_entry_point() {
    let (mut index, q) = build_test_index();
    let ep = index.cached_entry_point.unwrap();
    let ep_doc_id = index.doc_ids[ep as usize];

    // Delete the entry point
    index.delete_with_repair(ep_doc_id).unwrap();

    // Entry point should have changed
    assert_ne!(
        index.cached_entry_point,
        Some(ep),
        "entry point should change after deleting it"
    );
    assert!(
        index.cached_entry_point.is_some(),
        "should find a new entry point"
    );

    // Search should still work
    let results = index.search(&q, 5, 64).unwrap();
    assert!(!results.is_empty(), "search should work after EP deletion");
}

#[test]
fn test_delete_with_repair_graph_edges_cleaned() {
    let (mut index, _q) = build_test_index();
    let target_doc_id = 50u32;
    let internal_id = index.doc_id_to_internal[&target_doc_id];

    index.delete_with_repair(target_doc_id).unwrap();

    // No node in layer 0 should point to the deleted internal ID
    let layer = &index.layers[0];
    for node_id in 0..layer.len() as u32 {
        let neighbors = layer.get_neighbors(node_id);
        assert!(
            !neighbors.contains(&internal_id),
            "node {} still points to deleted node {} after repair",
            node_id,
            internal_id
        );
    }

    // Deleted node's own neighbor list should be empty
    assert!(
        layer.get_neighbors(internal_id).is_empty(),
        "deleted node should have empty neighbor list"
    );
}

#[test]
fn test_delete_batch_with_repair() {
    let (mut index, q) = build_test_index();
    let k = 10;
    let ef = 100;

    // Batch delete 30 nodes
    let ids_to_delete: Vec<u32> = (10..40).collect();
    let repairs = index.delete_batch_with_repair(&ids_to_delete).unwrap();

    let after = index.search(&q, k, ef).unwrap();
    assert_eq!(after.len(), k);

    // No deleted IDs in results
    let deleted_set: std::collections::HashSet<u32> = ids_to_delete.iter().copied().collect();
    for (id, _) in &after {
        assert!(
            !deleted_set.contains(id),
            "deleted id {} in results (repairs={})",
            id,
            repairs,
        );
    }
}

#[test]
fn test_delete_with_repair_nonexistent_errors() {
    let (mut index, _q) = build_test_index();
    assert!(index.delete_with_repair(9999).is_err());
}

#[test]
fn test_delete_with_repair_recall_vs_tombstone() {
    // Compare recall after repair-delete vs tombstone-delete.
    // Repair should maintain better recall because the graph stays connected.
    let dim = 32;
    let n = 300;

    let mut seed: u64 = 77;
    let mut next = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        vectors.push(v);
    }

    let mut query: Vec<f32> = (0..dim).map(|_| next()).collect();
    let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        query.iter_mut().for_each(|x| *x /= norm);
    }

    let build = |seed_val: u64| -> HNSWIndex {
        let params = HNSWParams {
            seed: Some(seed_val),
            ..HNSWParams::default()
        };
        let mut idx = HNSWIndex::with_params(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            idx.add(i as u32, v.clone()).unwrap();
        }
        idx.build().unwrap();
        idx
    };

    let mut repair_idx = build(42);
    let mut tombstone_idx = build(42);

    // Delete 40% of nodes
    let delete_ids: Vec<u32> = (0..120).collect();
    for &id in &delete_ids {
        repair_idx.delete_with_repair(id).unwrap();
        tombstone_idx.delete(id).unwrap();
    }

    let k = 10;
    let ef = 100;

    let repair_results = repair_idx.search(&query, k, ef).unwrap();
    let tombstone_results = tombstone_idx.search(&query, k, ef).unwrap();

    assert!(
        !repair_results.is_empty(),
        "repair search should return results"
    );
    assert!(
        !tombstone_results.is_empty(),
        "tombstone search should return results"
    );

    // Repair should return at least as many results (graph stays connected)
    assert!(
        repair_results.len() >= tombstone_results.len(),
        "repair ({}) should return >= tombstone ({}) results",
        repair_results.len(),
        tombstone_results.len(),
    );
}
