#![cfg(all(
    feature = "hnsw",
    feature = "nsw",
    feature = "diskann",
    feature = "sng"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Cross-algorithm consistency tests.
//!
//! Verifies that all graph-based ANN algorithms (HNSW, NSW, DiskANN, SNG)
//! produce results consistent with brute-force ground truth on the same
//! dataset. Algorithms use different distance metrics internally, but on
//! normalized vectors the nearest-neighbor rankings are identical.

use std::collections::HashSet;
use vicinity::diskann::{DiskANNIndex, DiskANNParams};
use vicinity::hnsw::HNSWIndex;
use vicinity::nsw::NSWIndex;
use vicinity::sng::{SNGIndex, SNGParams};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random vectors via std hash.
fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::hash::{Hash, Hasher};

    (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    seed.hash(&mut hasher);
                    i.hash(&mut hasher);
                    j.hash(&mut hasher);
                    let h = hasher.finish();
                    (h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
                })
                .collect()
        })
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Cosine distance: 1 - cosine_similarity.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 1.0;
    }
    1.0 - dot / (norm_a * norm_b)
}

/// Brute-force exact k-NN using cosine distance.
///
/// For normalized vectors the ranking is the same regardless of whether the
/// index uses cosine, dot-product, or L2 internally.
fn exact_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut dists: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine_distance(v, query)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

/// Recall@k between exact and approximate result sets (by doc_id).
fn recall_at_k(exact: &[(u32, f32)], approx: &[(u32, f32)], k: usize) -> f32 {
    let exact_set: HashSet<u32> = exact.iter().take(k).map(|(id, _)| *id).collect();
    let approx_set: HashSet<u32> = approx.iter().take(k).map(|(id, _)| *id).collect();
    exact_set.intersection(&approx_set).count() as f32 / k as f32
}

// ---------------------------------------------------------------------------
// Shared dataset
// ---------------------------------------------------------------------------

const N: usize = 200;
const DIM: usize = 32;
const N_QUERIES: usize = 20;
const K: usize = 10;
const SEED_DATA: u64 = 42;
const SEED_QUERIES: u64 = 999;

fn dataset() -> Vec<Vec<f32>> {
    random_vectors(N, DIM, SEED_DATA)
        .into_iter()
        .map(|v| normalize(&v))
        .collect()
}

fn queries() -> Vec<Vec<f32>> {
    random_vectors(N_QUERIES, DIM, SEED_QUERIES)
        .into_iter()
        .map(|v| normalize(&v))
        .collect()
}

// ---------------------------------------------------------------------------
// Index builders
// ---------------------------------------------------------------------------

fn build_hnsw(vectors: &[Vec<f32>]) -> HNSWIndex {
    let mut idx = HNSWIndex::new(DIM, 16, 100).expect("hnsw: create");
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).expect("hnsw: add");
    }
    idx.build().expect("hnsw: build");
    idx
}

fn build_nsw(vectors: &[Vec<f32>]) -> NSWIndex {
    let mut idx = NSWIndex::new(DIM, 16, 16).expect("nsw: create");
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).expect("nsw: add");
    }
    idx.build().expect("nsw: build");
    idx
}

fn build_diskann(vectors: &[Vec<f32>]) -> DiskANNIndex {
    let params = DiskANNParams {
        m: 32,
        ef_construction: 100,
        alpha: 1.2,
        ef_search: 100,
    };
    let mut idx = DiskANNIndex::new(DIM, params).expect("diskann: create");
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).expect("diskann: add");
    }
    idx.build().expect("diskann: build");
    idx
}

fn build_sng(vectors: &[Vec<f32>]) -> SNGIndex {
    let mut idx = SNGIndex::new(DIM, SNGParams::default()).expect("sng: create");
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).expect("sng: add");
    }
    idx.build().expect("sng: build");
    idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// All four algorithms achieve at least 50% recall@10 vs brute-force ground truth.
#[test]
fn cross_algorithm_recall_vs_brute_force() {
    let vectors = dataset();
    let qs = queries();

    let hnsw = build_hnsw(&vectors);
    let nsw = build_nsw(&vectors);
    let diskann = build_diskann(&vectors);
    let sng = build_sng(&vectors);

    let ef = 100;

    let mut recall_hnsw_total = 0.0_f32;
    let mut recall_nsw_total = 0.0_f32;
    let mut recall_diskann_total = 0.0_f32;
    let mut recall_sng_total = 0.0_f32;

    for q in &qs {
        let gt = exact_knn(&vectors, q, K);

        let r_hnsw = hnsw.search(q, K, ef).expect("hnsw search");
        let r_nsw = nsw.search(q, K, ef).expect("nsw search");
        let r_diskann = diskann.search(q, K, ef).expect("diskann search");
        let r_sng = sng.search(q, K).expect("sng search");

        recall_hnsw_total += recall_at_k(&gt, &r_hnsw, K);
        recall_nsw_total += recall_at_k(&gt, &r_nsw, K);
        recall_diskann_total += recall_at_k(&gt, &r_diskann, K);
        recall_sng_total += recall_at_k(&gt, &r_sng, K);
    }

    let n_q = N_QUERIES as f32;
    let avg_hnsw = recall_hnsw_total / n_q;
    let avg_nsw = recall_nsw_total / n_q;
    let avg_diskann = recall_diskann_total / n_q;
    let avg_sng = recall_sng_total / n_q;

    let threshold = 0.5;

    assert!(
        avg_hnsw >= threshold,
        "HNSW avg recall@{K} = {avg_hnsw:.3}, expected >= {threshold}"
    );
    assert!(
        avg_nsw >= threshold,
        "NSW avg recall@{K} = {avg_nsw:.3}, expected >= {threshold}"
    );
    assert!(
        avg_diskann >= threshold,
        "DiskANN avg recall@{K} = {avg_diskann:.3}, expected >= {threshold}"
    );
    assert!(
        avg_sng >= threshold,
        "SNG avg recall@{K} = {avg_sng:.3}, expected >= {threshold}"
    );
}

/// Stable algorithms (HNSW, NSW, DiskANN) return themselves as the nearest
/// neighbor when queried with a dataset vector. SNG (experimental) is checked
/// with a softer containment assertion (top-10).
#[test]
fn self_query_returns_self() {
    let vectors = dataset();

    let hnsw = build_hnsw(&vectors);
    let nsw = build_nsw(&vectors);
    let diskann = build_diskann(&vectors);
    let sng = build_sng(&vectors);

    let ef = 100;

    for i in 0..N {
        let q = &vectors[i];
        let doc_id = i as u32;

        // HNSW -- strict top-1
        let r = hnsw.search(q, 1, ef).expect("hnsw self-query");
        assert!(!r.is_empty());
        assert_eq!(
            r[0].0, doc_id,
            "HNSW: self-query doc_id={doc_id} returned {} (dist={:.6})",
            r[0].0, r[0].1
        );

        // NSW -- strict top-1
        let r = nsw.search(q, 1, ef).expect("nsw self-query");
        assert!(!r.is_empty());
        assert_eq!(
            r[0].0, doc_id,
            "NSW: self-query doc_id={doc_id} returned {} (dist={:.6})",
            r[0].0, r[0].1
        );

        // DiskANN -- strict top-1
        let r = diskann.search(q, 1, ef).expect("diskann self-query");
        assert!(!r.is_empty());
        assert_eq!(
            r[0].0, doc_id,
            "DiskANN: self-query doc_id={doc_id} returned {} (dist={:.6})",
            r[0].0, r[0].1
        );
    }

    // SNG (experimental) -- softer: self must appear in top-10
    let mut sng_found = 0;
    for i in 0..N {
        let q = &vectors[i];
        let doc_id = i as u32;
        let r = sng.search(q, 10).expect("sng self-query");
        if r.iter().any(|(id, _)| *id == doc_id) {
            sng_found += 1;
        }
    }
    let sng_rate = sng_found as f32 / N as f32;
    assert!(
        sng_rate >= 0.8,
        "SNG: only {:.1}% of self-queries found in top-10 (expected >= 80%)",
        sng_rate * 100.0
    );
}
