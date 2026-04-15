#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]
//! Brute-force recall tests for algorithms that previously lacked ground-truth coverage.
//!
//! Each test builds an index from normalized random vectors, searches with held-out
//! queries, and compares recall@10 against exhaustive cosine k-NN. Thresholds are
//! intentionally moderate (>= 40-60%) to catch broken construction/search logic
//! without being flaky on small datasets.
//!
//! Algorithms already covered by cross_algorithm_consistency.rs (HNSW, NSW, DiskANN, SNG)
//! and correctness_regression.rs (IVF-AVQ) are not duplicated here.

#[path = "common/mod.rs"]
mod common;
use common::*;

const N: usize = 300;
const DIM: usize = 32;
const N_QUERIES: usize = 20;
const K: usize = 10;
const SEED_DATA: u64 = 77;
const SEED_QUERIES: u64 = 1337;

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

fn exact_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut dists: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, vicinity::distance::cosine_distance(v, query)))
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists.truncate(k);
    dists
}

fn avg_recall(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    search_fn: impl Fn(&[f32]) -> Vec<(u32, f32)>,
) -> f32 {
    let mut total = 0.0f32;
    for q in queries {
        let gt = exact_knn(vectors, q, K);
        let results = search_fn(q);
        total += recall_at_k_sets(&gt, &results, K);
    }
    total / queries.len() as f32
}

// ---------------------------------------------------------------------------
// NSG
// ---------------------------------------------------------------------------

#[cfg(feature = "nsg")]
#[test]
fn nsg_recall_vs_brute_force() {
    use vicinity::nsg::{NsgIndex, NsgParams};

    let vectors = dataset();
    let qs = queries();

    let params = NsgParams {
        max_degree: 32,
        pool_size: 100,
        ..NsgParams::default()
    };
    let mut idx = NsgIndex::new(DIM, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();

    let recall = avg_recall(&vectors, &qs, |q| idx.search_with_ef(q, K, 100).unwrap());
    assert!(
        recall >= 0.40,
        "NSG avg recall@{K} = {recall:.3}, expected >= 0.40"
    );
}

// ---------------------------------------------------------------------------
// EMG
// ---------------------------------------------------------------------------

#[cfg(feature = "emg")]
#[test]
fn emg_recall_vs_brute_force() {
    use vicinity::emg::{EmgIndex, EmgParams};

    let vectors = dataset();
    let qs = queries();

    let params = EmgParams {
        max_degree: 32,
        candidate_size: 100,
        ..EmgParams::default()
    };
    let mut idx = EmgIndex::new(DIM, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();

    let recall = avg_recall(&vectors, &qs, |q| idx.search_with_ef(q, K, 100).unwrap());
    assert!(
        recall >= 0.50,
        "EMG avg recall@{K} = {recall:.3}, expected >= 0.50"
    );
}

// ---------------------------------------------------------------------------
// FINGER
// ---------------------------------------------------------------------------

#[cfg(feature = "finger")]
#[test]
fn finger_recall_vs_brute_force() {
    use vicinity::finger::{FingerIndex, FingerParams};

    let vectors = dataset();
    let qs = queries();

    let params = FingerParams {
        max_degree: 32,
        ef_construction: 100,
        ..FingerParams::default()
    };
    let mut idx = FingerIndex::new(DIM, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();

    let recall = avg_recall(&vectors, &qs, |q| idx.search_with_ef(q, K, 100).unwrap());
    assert!(
        recall >= 0.50,
        "FINGER avg recall@{K} = {recall:.3}, expected >= 0.50"
    );
}

// ---------------------------------------------------------------------------
// PiPNN
// ---------------------------------------------------------------------------

#[cfg(feature = "pipnn")]
#[test]
fn pipnn_recall_vs_brute_force() {
    use vicinity::pipnn::{PipnnIndex, PipnnParams};

    let vectors = dataset();
    let qs = queries();

    let params = PipnnParams {
        max_degree: 32,
        ..PipnnParams::default()
    };
    let mut idx = PipnnIndex::new(DIM, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();

    let recall = avg_recall(&vectors, &qs, |q| idx.search_with_ef(q, K, 100).unwrap());
    assert!(
        recall >= 0.50,
        "PiPNN avg recall@{K} = {recall:.3}, expected >= 0.50"
    );
}

// ---------------------------------------------------------------------------
// FreshGraph
// ---------------------------------------------------------------------------

#[cfg(feature = "fresh_graph")]
#[test]
fn fresh_graph_recall_vs_brute_force() {
    use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};

    let vectors = dataset();
    let qs = queries();

    let params = FreshGraphParams {
        max_degree: 32,
        ef_construction: 100,
        ..FreshGraphParams::default()
    };
    let mut idx = FreshGraphIndex::new(DIM, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();

    let recall = avg_recall(&vectors, &qs, |q| idx.search_with_ef(q, K, 100).unwrap());
    assert!(
        recall >= 0.50,
        "FreshGraph avg recall@{K} = {recall:.3}, expected >= 0.50"
    );
}

// ---------------------------------------------------------------------------
// IVF-RaBitQ
// ---------------------------------------------------------------------------

#[cfg(feature = "ivf_rabitq")]
#[test]
fn ivf_rabitq_recall_vs_brute_force() {
    use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams};

    let vectors = dataset();
    let qs = queries();

    let params = IVFRaBitQParams {
        num_clusters: 8,
        nprobe: 4,
        ..IVFRaBitQParams::default()
    };
    let mut idx = IVFRaBitQIndex::new(DIM, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();

    let recall = avg_recall(&vectors, &qs, |q| idx.search(q, K).unwrap());
    assert!(
        recall >= 0.40,
        "IVF-RaBitQ avg recall@{K} = {recall:.3}, expected >= 0.40"
    );
}

// ---------------------------------------------------------------------------
// SymphonyQG
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hnsw", feature = "ivf_rabitq"))]
#[test]
fn symphonyqg_recall_vs_brute_force() {
    use vicinity::hnsw::symphony_qg::SymphonyQGIndex;

    let vectors = dataset();
    let qs = queries();

    let mut idx = SymphonyQGIndex::new(DIM, 16, 100).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add_slice(i as u32, v).unwrap();
    }
    idx.build().unwrap();

    // Non-reranked quantized search -- expected to have lower recall due to
    // RaBitQ approximation error, especially at low dimensions.
    let recall_raw = avg_recall(&vectors, &qs, |q| idx.search(q, K, 100).unwrap());

    // Reranked search with exact cosine rescoring -- the recommended search path.
    let recall_reranked = avg_recall(&vectors, &qs, |q| {
        idx.search_reranked(q, K, 100, 50).unwrap()
    });

    assert!(
        recall_reranked >= 0.40,
        "SymphonyQG reranked avg recall@{K} = {recall_reranked:.3}, expected >= 0.40 \
         (raw quantized recall = {recall_raw:.3})"
    );
}

// Regression test: 4-bit and 8-bit RaBitQ must achieve comparable recall to
// 1-bit.  Before the fix, correction factors were computed from binary codes
// only, causing a ~16x/128x scale mismatch with the multi-bit inner product
// at query time, collapsing 4-bit recall to ~6% and 8-bit to ~5%.
// ---------------------------------------------------------------------------
// Vamana
// ---------------------------------------------------------------------------

#[test]
fn vamana_recall_vs_brute_force() {
    use vicinity::vamana::{VamanaIndex, VamanaParams};

    let vectors = dataset();
    let qs = queries();

    let params = VamanaParams {
        max_degree: 32,
        ef_construction: 100,
        alpha: 1.2,
        ef_search: 100,
        ..VamanaParams::default()
    };
    let mut idx = VamanaIndex::new(DIM, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();

    let recall = avg_recall(&vectors, &qs, |q| idx.search(q, K, 100).unwrap());
    assert!(
        recall >= 0.40,
        "Vamana avg recall@{K} = {recall:.3}, expected >= 0.40"
    );
}

// ---------------------------------------------------------------------------
// IVF-PQ
// ---------------------------------------------------------------------------

#[cfg(feature = "ivf_pq")]
#[test]
fn ivf_pq_recall_vs_brute_force() {
    use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

    let vectors = dataset();
    let qs = queries();

    let params = IVFPQParams {
        num_clusters: 4,
        nprobe: 4,
        num_codebooks: 4,
        codebook_size: 16,
        ..IVFPQParams::default()
    };
    let mut idx = IVFPQIndex::new(DIM, params).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.add(i as u32, v.clone()).unwrap();
    }
    idx.build().unwrap();

    let recall = avg_recall(&vectors, &qs, |q| idx.search(q, K).unwrap());
    // IVF-PQ at low dimensions with few clusters has limited recall
    assert!(
        recall >= 0.20,
        "IVF-PQ avg recall@{K} = {recall:.3}, expected >= 0.20"
    );
}

// ---------------------------------------------------------------------------
// ADSampling + HNSW
// ---------------------------------------------------------------------------

#[cfg(feature = "hnsw")]
#[test]
fn adsampling_recall_vs_brute_force() {
    use vicinity::adsampling::{ADSamplingParams, ADSamplingState};
    use vicinity::hnsw::HNSWIndex;

    let vectors = dataset();
    let qs = queries();

    // Build HNSW
    let mut hnsw = HNSWIndex::new(DIM, 16, 32).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add_slice(i as u32, v).unwrap();
    }
    hnsw.build().unwrap();

    // Build ADSampling from reordered HNSW vectors
    let state = ADSamplingState::from_hnsw(&hnsw, ADSamplingParams::default());

    let recall = avg_recall(&vectors, &qs, |q| {
        state.search_hnsw(&hnsw, q, K, 64).unwrap()
    });
    assert!(
        recall >= 0.40,
        "ADSampling+HNSW avg recall@{K} = {recall:.3}, expected >= 0.40"
    );
}

// ---------------------------------------------------------------------------
// SymphonyQG multi-bit regression
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hnsw", feature = "ivf_rabitq"))]
#[test]
fn symphonyqg_multibit_recall_regression() {
    use qntz::rabitq::RaBitQConfig;
    use vicinity::hnsw::symphony_qg::SymphonyQGIndex;

    let vectors = dataset();
    let qs = queries();

    for (label, config) in [
        ("4-bit", RaBitQConfig::bits4()),
        ("8-bit", RaBitQConfig::bits8()),
    ] {
        let mut idx = SymphonyQGIndex::with_config(DIM, 16, 100, config, 42).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            idx.add_slice(i as u32, v).unwrap();
        }
        idx.build().unwrap();

        let recall = avg_recall(&vectors, &qs, |q| {
            idx.search_reranked(q, K, 100, 50).unwrap()
        });

        assert!(
            recall >= 0.30,
            "SymphonyQG {label} reranked recall@{K} = {recall:.3}, expected >= 0.30 \
             (regression: pre-fix multi-bit recall was 6-12%)"
        );
    }
}
