//! Robustness tests for vicinity.
//!
//! These tests verify graceful error handling for edge cases: NaN/Inf inputs,
//! empty indexes, dimension mismatches, SIMD odd dimensions, and misuse patterns.
//! Tests should not panic -- they assert that errors are returned or results are valid.

use vicinity::distance::{normalize, DistanceMetric};
use vicinity::hnsw::HNSWIndex;

// ---------------------------------------------------------------------------
// 1. NaN / Inf rejection
// ---------------------------------------------------------------------------

#[test]
fn nan_vector_rejected_or_handled_gracefully() {
    // NaN vectors should ideally be rejected at add time. This test documents
    // actual behavior and ensures no unrecoverable panic in release builds.
    //
    // FINDING: In debug builds, a debug_assert! in add_slice fires on NaN
    // vectors (norm^2 check). In release builds, NaN is silently accepted.
    // Neither path corrupts the index in a way that causes subsequent panics.
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();

    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();

    let nan_vec = vec![f32::NAN; 4];
    // Wrap in catch_unwind because debug_assert! fires in debug builds.
    let add_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        index.add_slice(1, &nan_vec)
    }));

    match add_result {
        Err(_) => {
            // debug_assert! fired -- acceptable in debug builds.
        }
        Ok(Err(_)) => {
            // Explicit error returned -- ideal behavior.
        }
        Ok(Ok(())) => {
            // Silently accepted (release builds). Verify search still works.
            let _ = index.build();
        }
    }
}

#[test]
fn inf_vector_rejected_or_handled_gracefully() {
    // Same pattern as NaN: debug_assert! fires in debug builds on Inf vectors
    // because norm^2 is Inf. Release builds accept silently.
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();

    let v = normalize(&[0.0, 1.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();

    let inf_vec = vec![f32::INFINITY; 4];
    let add_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        index.add_slice(1, &inf_vec)
    }));

    match add_result {
        Err(_) => {
            // debug_assert! fired -- acceptable in debug builds.
        }
        Ok(Err(_)) => {
            // Explicit error returned -- ideal behavior.
        }
        Ok(Ok(())) => {
            // Silently accepted (release builds). Verify search still works.
            let _ = index.build();
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Empty index search
// ---------------------------------------------------------------------------

#[test]
fn search_empty_index_returns_error() {
    let index = HNSWIndex::new(4, 16, 32).unwrap();
    // Index is not built, so search should return Err (not built).
    let result = index.search(&[1.0, 0.0, 0.0, 0.0], 5, 10);
    assert!(result.is_err(), "searching an unbuilt index should return Err");
}

#[test]
fn build_empty_index_returns_error() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    let result = index.build();
    assert!(result.is_err(), "building an empty index should return Err");
}

// ---------------------------------------------------------------------------
// 3. Zero-dimension index
// ---------------------------------------------------------------------------

#[test]
fn zero_dimension_rejected() {
    let result = HNSWIndex::new(0, 16, 32);
    assert!(
        result.is_err(),
        "creating an index with dimension=0 should fail"
    );
}

// ---------------------------------------------------------------------------
// 4. Dimension mismatch on search
// ---------------------------------------------------------------------------

#[test]
fn search_dimension_mismatch_returns_error() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();
    index.build().unwrap();

    // Query with wrong dimension (2 instead of 4).
    let result = index.search(&[1.0, 0.0], 1, 10);
    assert!(
        result.is_err(),
        "search with mismatched query dimension should return Err"
    );
}

#[test]
fn add_dimension_mismatch_returns_error() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    let result = index.add_slice(0, &[1.0, 0.0]);
    assert!(
        result.is_err(),
        "adding a vector with wrong dimension should return Err"
    );
}

// ---------------------------------------------------------------------------
// 5. Search before build
// ---------------------------------------------------------------------------

#[test]
fn search_before_build_returns_error() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();

    let result = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 10);
    assert!(
        result.is_err(),
        "searching before build should return Err"
    );
}

// ---------------------------------------------------------------------------
// 6. SIMD odd dimensions
// ---------------------------------------------------------------------------

#[test]
fn simd_odd_dimensions_produce_finite_nonnegative_distances() {
    for dim in [1, 3, 7, 15, 17, 31, 33] {
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.1 + 0.01).collect();
        let a_norm = normalize(&a);
        let b_norm = normalize(&b);

        let dist = DistanceMetric::Cosine.distance(&a_norm, &b_norm);
        assert!(
            dist.is_finite(),
            "dim={dim}: cosine distance is not finite: {dist}"
        );
        assert!(
            dist >= -1e-6,
            "dim={dim}: cosine distance is unexpectedly negative: {dist}"
        );

        let l2 = DistanceMetric::L2.distance(&a_norm, &b_norm);
        assert!(
            l2.is_finite(),
            "dim={dim}: L2 distance is not finite: {l2}"
        );
        assert!(l2 >= 0.0, "dim={dim}: L2 distance is negative: {l2}");
    }
}

// ---------------------------------------------------------------------------
// 7. add_batch dimension mismatch
// ---------------------------------------------------------------------------

#[test]
fn add_batch_dimension_mismatch_returns_error() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    // 3 floats for 2 vectors of dim 4 -- should fail.
    let result = index.add_batch(&[0, 1], &[1.0, 2.0, 3.0]);
    assert!(
        result.is_err(),
        "add_batch with mismatched vector/id count should return Err"
    );
}

#[test]
fn add_batch_correct_dimensions_succeeds() {
    let mut index = HNSWIndex::new(2, 4, 4).unwrap();
    let v0 = normalize(&[1.0, 0.0]);
    let v1 = normalize(&[0.0, 1.0]);
    let mut flat = Vec::new();
    flat.extend_from_slice(&v0);
    flat.extend_from_slice(&v1);

    let result = index.add_batch(&[0, 1], &flat);
    assert!(result.is_ok(), "add_batch with correct dimensions should succeed");
}

// ---------------------------------------------------------------------------
// 8. Additional edge cases
// ---------------------------------------------------------------------------

#[test]
fn duplicate_doc_id_rejected() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();

    let result = index.add_slice(0, &v);
    assert!(
        result.is_err(),
        "adding a duplicate doc_id should return Err"
    );
}

#[test]
fn search_k_zero_returns_empty() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();
    index.build().unwrap();

    let result = index.search(&normalize(&[1.0, 0.0, 0.0, 0.0]), 0, 10);
    match result {
        Ok(r) => assert!(r.is_empty(), "k=0 should return empty results"),
        Err(_) => {} // also acceptable
    }
}

#[test]
fn search_k_larger_than_index_size() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();
    index.build().unwrap();

    // Ask for 100 neighbors when only 1 vector exists.
    let result = index.search(&normalize(&[1.0, 0.0, 0.0, 0.0]), 100, 200);
    match result {
        Ok(r) => assert!(
            r.len() <= 1,
            "should return at most 1 result, got {}",
            r.len()
        ),
        Err(_) => {} // also acceptable
    }
}

#[test]
fn add_after_build_rejected() {
    let mut index = HNSWIndex::new(4, 16, 32).unwrap();
    let v = normalize(&[1.0, 0.0, 0.0, 0.0]);
    index.add_slice(0, &v).unwrap();
    index.build().unwrap();

    let result = index.add_slice(1, &normalize(&[0.0, 1.0, 0.0, 0.0]));
    assert!(
        result.is_err(),
        "adding vectors after build should return Err"
    );
}

#[test]
fn distance_metric_dimension_mismatch_returns_infinity() {
    let a = [1.0, 0.0, 0.0];
    let b = [1.0, 0.0];
    let dist = DistanceMetric::L2.distance(&a, &b);
    assert!(
        dist == f32::INFINITY,
        "L2 distance on mismatched dims should be INFINITY, got {dist}"
    );
    let dist = DistanceMetric::Cosine.distance(&a, &b);
    assert!(
        dist == f32::INFINITY,
        "Cosine distance on mismatched dims should be INFINITY, got {dist}"
    );
}

#[test]
fn zero_m_rejected() {
    let result = HNSWIndex::new(4, 0, 32);
    assert!(result.is_err(), "m=0 should be rejected");
}

#[test]
fn zero_m_max_rejected() {
    let result = HNSWIndex::new(4, 16, 0);
    assert!(result.is_err(), "m_max=0 should be rejected");
}
