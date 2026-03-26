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
