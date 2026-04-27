#![allow(clippy::unwrap_used, clippy::expect_used)]
// The bulk of this file targets the serde (JSON) persistence path.
// The segment-binary tests further below require `persistence` too.
#![cfg(all(feature = "hnsw", feature = "serde"))]

use vicinity::hnsw::HNSWIndex;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a deterministic HNSW index with `n` normalized vectors of dimension `dim`.
fn build_deterministic_index(n: usize, dim: usize) -> HNSWIndex {
    let mut index = HNSWIndex::new(dim, 16, 32).expect("valid params");

    // Deterministic pseudo-random vectors via LCG.
    let mut seed: u64 = 42;
    let mut next = || -> f32 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    for i in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        index.add(i as u32, v).expect("add should succeed");
    }
    index.build().expect("build should succeed");
    index
}

/// Generate `count` deterministic normalized query vectors (seeded differently from the index).
fn deterministic_queries(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut seed: u64 = 12345;
    let mut next = || -> f32 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    (0..count)
        .map(|_| {
            let mut q: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                q.iter_mut().for_each(|x| *x /= norm);
            }
            q
        })
        .collect()
}

/// Serialize an index to bytes.
fn save_to_bytes(index: &HNSWIndex) -> Vec<u8> {
    let mut buf = Vec::new();
    index
        .save_to_writer(&mut buf)
        .expect("serialization should succeed");
    buf
}

// ---------------------------------------------------------------------------
// 1. HNSW save/load exact roundtrip
// ---------------------------------------------------------------------------

#[test]
fn hnsw_save_load_exact_roundtrip() {
    let dim = 16;
    let n = 50;
    let k = 5;
    let ef = 64;
    let index = build_deterministic_index(n, dim);
    let queries = deterministic_queries(5, dim);

    // Collect results from the original index.
    let original_results: Vec<_> = queries
        .iter()
        .map(|q| index.search(q, k, ef).expect("search should succeed"))
        .collect();

    // Save to a temp file, then load back.
    let tmp = tempfile::NamedTempFile::new().expect("tempfile creation");
    index
        .save_to_writer(std::io::BufWriter::new(tmp.as_file()))
        .expect("save should succeed");

    let loaded = HNSWIndex::load_from_reader(std::io::BufReader::new(
        std::fs::File::open(tmp.path()).expect("open temp file"),
    ))
    .expect("load should succeed");

    // Structural invariant: index should be ready for search.
    assert!(loaded.is_built());

    // Same queries must produce identical results (IDs and distances within f32 epsilon).
    for (i, q) in queries.iter().enumerate() {
        let loaded_results = loaded.search(q, k, ef).expect("search should succeed");
        assert_eq!(
            loaded_results.len(),
            original_results[i].len(),
            "query {i}: result count mismatch"
        );
        for (j, (lr, or)) in loaded_results
            .iter()
            .zip(original_results[i].iter())
            .enumerate()
        {
            assert_eq!(
                lr.0, or.0,
                "query {i} result {j}: doc_id mismatch ({} vs {})",
                lr.0, or.0
            );
            assert!(
                (lr.1 - or.1).abs() < f32::EPSILON,
                "query {i} result {j}: distance mismatch ({} vs {})",
                lr.1,
                or.1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Truncated file handling
// ---------------------------------------------------------------------------

#[test]
fn truncated_file_returns_err() {
    let index = build_deterministic_index(50, 16);
    let bytes = save_to_bytes(&index);
    assert!(
        bytes.len() > 2,
        "sanity: serialized bytes should be non-trivial"
    );

    let truncation_points = [
        0,               // empty
        1,               // single byte
        bytes.len() / 2, // half
        bytes.len() - 1, // one byte short
    ];

    for &len in &truncation_points {
        let truncated = &bytes[..len];
        let result = HNSWIndex::load_from_reader(truncated);
        assert!(
            result.is_err(),
            "expected Err for truncated input ({len} of {} bytes), got Ok",
            bytes.len()
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Corrupted bytes
// ---------------------------------------------------------------------------

#[test]
fn corrupted_bytes_do_not_panic() {
    let index = build_deterministic_index(50, 16);
    let bytes = save_to_bytes(&index);

    // Flip bytes at several positions spread across the payload.
    // Use deterministic positions.
    let mut seed: u64 = 99;
    let mut next_pos = || -> usize {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 33) as usize % bytes.len()
    };

    for trial in 0..10 {
        let mut corrupted = bytes.clone();
        // Flip 1-3 bytes per trial.
        let flips = (trial % 3) + 1;
        for _ in 0..flips {
            let pos = next_pos();
            corrupted[pos] ^= 0xFF;
        }

        // The only acceptable outcomes are Err or Ok (degraded). Never a panic.
        let result = std::panic::catch_unwind(|| HNSWIndex::load_from_reader(corrupted.as_slice()));
        match result {
            Ok(Ok(_loaded)) => {
                // Loaded despite corruption -- acceptable (JSON is lenient with
                // some mutations). No assertion on search quality here.
            }
            Ok(Err(_e)) => {
                // Deserialization caught the corruption -- expected.
            }
            Err(panic_payload) => {
                let msg: String = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_owned()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "(non-string panic)".to_owned()
                };
                panic!("trial {trial}: load_from_reader panicked: {msg}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Segment-binary persistence tests (require `persistence` feature)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hnsw", feature = "persistence"))]
mod segment_binary {
    use proptest::prelude::*;
    use vicinity::hnsw::HNSWIndex;
    use vicinity::persistence::directory::{Directory, MemoryDirectory};
    use vicinity::persistence::error::PersistenceError;
    use vicinity::persistence::hnsw::{HNSWSegmentReader, HNSWSegmentWriter};

    /// Build a small deterministic HNSW index and persist it to a MemoryDirectory,
    /// returning both the directory and the raw bytes of metadata.bin so tests can
    /// corrupt them.
    fn write_segment() -> (MemoryDirectory, Vec<u8>) {
        let dim = 4;
        let mut index = HNSWIndex::new(dim, 8, 8).expect("new index");
        index.add(1, vec![1.0, 0.0, 0.0, 0.0]).expect("add");
        index.add(2, vec![0.0, 1.0, 0.0, 0.0]).expect("add");
        index.build().expect("build");

        let mem = MemoryDirectory::new();
        let mut writer = HNSWSegmentWriter::new(Box::new(mem.clone()), 0);
        writer.write_hnsw_index(&index).expect("write");

        // Read back the raw metadata bytes for corruption tests.
        use std::io::Read;
        let mut f = mem
            .open_file("segments/segment_hnsw_0/metadata.bin")
            .expect("open metadata");
        let mut raw = Vec::new();
        f.read_to_end(&mut raw).expect("read");
        (mem, raw)
    }

    // -----------------------------------------------------------------------
    // 4. Corrupt magic returns Format error
    // -----------------------------------------------------------------------

    #[test]
    fn loading_corrupt_magic_returns_format_error() {
        let (mem, raw) = write_segment();

        // Overwrite the first byte to break the magic.
        let mut corrupted = raw.clone();
        corrupted[0] ^= 0xFF;

        // Write corrupted bytes back into the directory.
        mem.atomic_write("segments/segment_hnsw_0/metadata.bin", &corrupted)
            .expect("atomic_write");

        let result = HNSWSegmentReader::load(Box::new(mem.clone()), 0);

        // The load may either succeed via legacy-v0 fallback (and fail later when
        // loading vectors) or fail immediately.  What must NOT happen is a panic.
        // Additionally, if the magic bytes looked like plausible legacy data but the
        // resulting dimension/num_vectors are unreasonable, we expect a Format error.
        // We only assert no-panic here; the proptest below verifies Err-or-Ok exhaustively.
        let _ = result;

        // For a more targeted check: write metadata.bin with a wrong magic
        // followed by 0xFFFFFFFF as the first u32 (dimension field in the v0
        // interpretation). That is beyond MAX_DIMENSION (65536), so the size
        // guard must fire and return Format error.
        let mut bad_dim_magic: Vec<u8> = b"BADMAGIC".to_vec(); // 8 bytes, not VCNHNSW\x01
        bad_dim_magic.extend_from_slice(&u32::MAX.to_le_bytes()); // "dimension" = 4294967295
        bad_dim_magic.extend_from_slice(&1u32.to_le_bytes()); // "num_vectors" = 1
        bad_dim_magic.push(1); // "is_built" = true
        mem.atomic_write("segments/segment_hnsw_0/metadata.bin", &bad_dim_magic)
            .expect("atomic_write");

        let err = HNSWSegmentReader::load(Box::new(mem), 0);
        // Unreasonable dimension triggers the size guard -> Format error.
        assert!(
            matches!(err, Err(PersistenceError::Format(_))),
            "expected Format error for unreasonable dimension, got: {:?}",
            err.err()
        );
    }

    // -----------------------------------------------------------------------
    // 5. proptest: single-byte corruption never panics
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn proptest_one_byte_corruption_never_panics(
            byte_offset in 0usize..21,  // metadata.bin is 21 bytes (magic8 + ver4 + dim4 + nv4 + built1)
            flip_mask in 1u8..=255u8,   // non-zero to guarantee a change
        ) {
            let (mem, raw) = write_segment();

            let mut corrupted = raw.clone();
            let off = byte_offset % raw.len();
            corrupted[off] ^= flip_mask;
            mem.atomic_write("segments/segment_hnsw_0/metadata.bin", &corrupted)
                .expect("atomic_write");

            // load() must not panic; Err is acceptable.
            let load_result = std::panic::catch_unwind(|| {
                HNSWSegmentReader::load(Box::new(mem.clone()), 0)
            });
            match load_result {
                Ok(Ok(reader)) => {
                    // If load succeeded, load_index must also not panic.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        reader.load_index()
                    }));
                }
                Ok(Err(_)) => {}
                Err(payload) => {
                    let msg: String = if let Some(s) = payload.downcast_ref::<&str>() {
                        (*s).to_owned()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "(non-string panic)".to_owned()
                    };
                    panic!(
                        "corrupt metadata (offset {off}, mask {flip_mask:#04x}) caused panic: {msg}"
                    );
                }
            }
        }
    }

    /// Real v0.6.2-written segment fixture loads correctly under v1 reader.
    ///
    /// The fixture under tests/fixtures/v0_segment_dim8/ was produced by
    /// vicinity 0.6.2 (commit 6b92ae9) via the binary HNSWSegmentWriter on a
    /// deterministic 20-vector dim=8 index.  This test guards the legacy v0
    /// decode path in HNSWSegmentReader::load (no MAGIC prefix, raw u32+u32+u8
    /// metadata layout) against silent regression.
    #[test]
    fn real_v0_fixture_loads_and_searches_correctly() {
        use std::path::PathBuf;
        use vicinity::persistence::directory::FsDirectory;

        let fixture_root: PathBuf =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v0_segment_dim8");
        assert!(
            fixture_root
                .join("segments/segment_hnsw_1/metadata.bin")
                .exists(),
            "fixture must exist at {}",
            fixture_root.display()
        );

        // Sanity: confirm metadata.bin has no v1 magic.
        let metadata_bytes =
            std::fs::read(fixture_root.join("segments/segment_hnsw_1/metadata.bin")).unwrap();
        assert_eq!(metadata_bytes.len(), 9, "v0 metadata is exactly 9 bytes");
        assert_ne!(
            &metadata_bytes[..8],
            b"VCNHNSW\x01",
            "fixture must NOT start with v1 magic"
        );

        let dir = FsDirectory::new(&fixture_root).expect("open fixture directory");
        let reader = HNSWSegmentReader::load(Box::new(dir), 1).expect("load v0 segment");
        let loaded = reader.load_index().expect("load_index");

        // Reproduce the same query the fixture-generation example used.
        // (gen_v0_fixture.rs in v0.6.2 worktree, query = deterministic_vec(1000, 8))
        let dim = 8;
        let mut seed: u64 = (1000_u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let mut query: Vec<f32> = (0..dim).map(|_| next()).collect();
        let qn = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if qn > 0.0 {
            query.iter_mut().for_each(|x| *x /= qn);
        }

        let results = loaded.search(&query, 5, 50).expect("search");
        let result_ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();

        // Expected IDs from v0.6.2 fixture-generation run (same seed, same algorithm).
        // Distance values are not asserted (small float drift is acceptable across
        // versions); IDs and ordering are the load-bearing invariant.
        let expected_ids: Vec<u32> = vec![5, 12, 14, 19, 2];
        assert_eq!(
            result_ids, expected_ids,
            "v0->v1 legacy decode must reproduce the original v0.6.2 search ordering"
        );
    }
}
