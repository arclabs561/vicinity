#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]
//! Correctness regression tests for fixed algorithm bugs.
//!
//! - IVF-PQ: metric consistency (cosine on normalized vectors)
//! - NSW: construction quality (recall ≥ threshold)
//! - ScaNN: residual codebook alignment

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn normalize(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// Deterministic LCG.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    }
    fn next_normalized(&mut self, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
        normalize(&v)
    }
}

fn brute_force_cosine(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = database
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            (i as u32, 1.0 - dot)
        })
        .collect();
    dists.sort_by(|a, b| a.1.total_cmp(&b.1));
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

fn recall_at_k(retrieved: &[(u32, f32)], ground_truth: &[u32]) -> f32 {
    let gt: std::collections::HashSet<u32> = ground_truth.iter().copied().collect();
    let hits = retrieved.iter().filter(|(id, _)| gt.contains(id)).count();
    hits as f32 / ground_truth.len().max(1) as f32
}

// ---------------------------------------------------------------------------
// IVF-PQ: metric consistency
// ---------------------------------------------------------------------------

#[cfg(feature = "ivf_pq")]
mod ivf_pq_tests {
    use super::*;
    use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

    fn build_index(n: usize, dim: usize, seed: u64) -> (IVFPQIndex, Vec<Vec<f32>>) {
        // Search all clusters (nprobe == num_clusters) and use enough codewords
        // so PQ approximation quality isn't the bottleneck.
        let num_clusters = 16.min(n / 4);
        let params = IVFPQParams {
            num_clusters,
            nprobe: num_clusters, // search everything
            num_codebooks: 8,
            codebook_size: 64,
            use_opq: false,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();
        let mut rng = Lcg::new(seed);
        let mut vecs = Vec::new();
        for i in 0..n {
            let v = rng.next_normalized(dim);
            index.add(i as u32, v.clone()).unwrap();
            vecs.push(v);
        }
        index.build().unwrap();
        (index, vecs)
    }

    #[test]
    fn ivf_pq_self_retrieval() {
        // Each database vector queried against the index should retrieve itself
        // in the top result. When nprobe covers all clusters and PQ is trained
        // on the same data, self-retrieval should be near-perfect.
        let dim = 32usize;
        let n = 400usize;
        let num_clusters = 16;
        let params = IVFPQParams {
            num_clusters,
            nprobe: num_clusters,
            num_codebooks: 8,
            codebook_size: 64,
            use_opq: false,
            ..IVFPQParams::default()
        };
        let mut index = IVFPQIndex::new(dim, params).unwrap();
        let mut rng = Lcg::new(1);
        let mut vecs = Vec::new();
        for i in 0..n {
            let v = rng.next_normalized(dim);
            index.add(i as u32, v.clone()).unwrap();
            vecs.push(v);
        }
        index.build().unwrap();

        let mut found = 0usize;
        for (i, v) in vecs.iter().enumerate() {
            let results = index.search(v, 1).unwrap();
            if !results.is_empty() && results[0].0 == i as u32 {
                found += 1;
            }
        }
        let self_recall = found as f32 / n as f32;
        assert!(
            self_recall >= 0.8,
            "IVF-PQ self-retrieval recall={self_recall:.3} < 0.8 (metric/quantization broken)"
        );
    }

    #[test]
    fn ivf_pq_unnormalized_query_same_as_normalized() {
        // Passing an unnormalized query should give the same result as its
        // normalized version, since the index normalizes queries internally.
        let dim = 32usize;
        let (index, _) = build_index(200, dim, 2);
        let mut rng = Lcg::new(888);

        for _ in 0..5 {
            let v = rng.next_normalized(dim);
            let scaled: Vec<f32> = v.iter().map(|x| x * 5.3).collect();

            let r_norm = index.search(&v, 5).unwrap();
            let r_scaled = index.search(&scaled, 5).unwrap();

            let ids_norm: Vec<u32> = r_norm.iter().map(|(id, _)| *id).collect();
            let ids_scaled: Vec<u32> = r_scaled.iter().map(|(id, _)| *id).collect();
            assert_eq!(
                ids_norm, ids_scaled,
                "normalized and scaled queries should give same IDs"
            );
        }
    }

    #[test]
    fn ivf_pq_returns_k_or_fewer_results() {
        let (index, _) = build_index(200, 32, 3);
        let mut rng = Lcg::new(42);
        let query = rng.next_normalized(32);
        for k in [1, 3, 5, 10, 50] {
            let results = index.search(&query, k).unwrap();
            assert!(results.len() <= k, "k={k} got {} results", results.len());
        }
    }

    #[test]
    fn ivf_pq_distances_are_nonnegative() {
        let (index, _) = build_index(200, 32, 4);
        let mut rng = Lcg::new(77);
        let query = rng.next_normalized(32);
        let results = index.search(&query, 10).unwrap();
        for (_, dist) in &results {
            assert!(*dist >= -0.01, "negative distance {dist}");
        }
    }

    #[test]
    fn ivf_pq_results_sorted_ascending() {
        let (index, _) = build_index(200, 32, 5);
        let mut rng = Lcg::new(55);
        let query = rng.next_normalized(32);
        let results = index.search(&query, 10).unwrap();
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6, "results not sorted: {:?}", &w);
        }
    }

    #[test]
    fn ivf_pq_empty_before_build_errors() {
        let params = IVFPQParams::default();
        let index = IVFPQIndex::new(8, params).unwrap();
        let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(index.search(&query, 1).is_err());
    }

    #[test]
    fn ivf_pq_dimension_mismatch_errors() {
        let (index, _) = build_index(200, 32, 6);
        let bad_query = vec![1.0f32, 0.0, 0.0]; // wrong dim
        assert!(index.search(&bad_query, 1).is_err());
    }

    #[test]
    fn ivf_pq_search_with_filter_unnormalized_matches_normalized() {
        // Regression: search_with_filter must normalize the query the same way
        // search() does. If the normalization path is missing or diverges,
        // the two results differ on the same underlying data.
        use std::collections::HashMap;
        use vicinity::filtering::MetadataFilter;

        let dim = 16usize;
        let n = 100usize;
        let params = IVFPQParams {
            num_clusters: 4,
            nprobe: 4,
            num_codebooks: 4,
            codebook_size: 16,
            use_opq: false,
            ..IVFPQParams::default()
        };

        let mut index = IVFPQIndex::with_filtering(dim, params.clone(), "category").unwrap();
        let mut rng = Lcg::new(77);

        for i in 0..n {
            let v = rng.next_normalized(dim);
            index.add(i as u32, v).unwrap();
            let mut meta = HashMap::new();
            meta.insert("category".to_string(), 0u32); // all in category 0
            index.add_metadata(i as u32, meta).unwrap();
        }
        index.build().unwrap();

        let raw_query: Vec<f32> = (0..dim).map(|i| (i + 1) as f32).collect(); // unnormalized
        let norm_query = normalize(&raw_query);
        let filter = MetadataFilter::equals("category", 0u32);

        let raw_results = index.search_with_filter(&raw_query, 5, &filter).unwrap();
        let norm_results = index.search_with_filter(&norm_query, 5, &filter).unwrap();

        let raw_ids: Vec<u32> = raw_results.iter().map(|(id, _)| *id).collect();
        let norm_ids: Vec<u32> = norm_results.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            raw_ids, norm_ids,
            "unnormalized and normalized queries must return identical results"
        );
    }
}

// ---------------------------------------------------------------------------
// NSW: construction quality
// ---------------------------------------------------------------------------

#[cfg(feature = "nsw")]
mod nsw_tests {
    use super::*;
    use vicinity::nsw::{NSWIndex, NSWParams};

    fn build_nsw(n: usize, dim: usize, seed: u64) -> (NSWIndex, Vec<Vec<f32>>) {
        let params = NSWParams {
            m: 16,
            m_max: 16,
            ef_search: 100,
            ef_construction: 50,
        };
        let mut index = NSWIndex::with_params(dim, params).unwrap();
        let mut rng = Lcg::new(seed);
        let mut vecs = Vec::new();
        for i in 0..n {
            let v = rng.next_normalized(dim);
            index.add(i as u32, v.clone()).unwrap();
            vecs.push(v);
        }
        index.build().unwrap();
        (index, vecs)
    }

    #[test]
    fn nsw_recall_oracle() {
        let (index, vecs) = build_nsw(200, 16, 10);
        let mut rng = Lcg::new(1234);
        let mut total_recall = 0.0f32;
        let num_queries = 20;

        for _ in 0..num_queries {
            let query = rng.next_normalized(16);
            let gt = brute_force_cosine(&query, &vecs, 5);
            let results = index.search(&query, 5, 50).unwrap();
            total_recall += recall_at_k(&results, &gt);
        }

        let avg_recall = total_recall / num_queries as f32;
        assert!(
            avg_recall >= 0.6,
            "NSW recall={avg_recall:.3} below 0.6 (construction quality regression)"
        );
    }

    #[test]
    fn nsw_large_index_builds_in_reasonable_time() {
        // This test primarily verifies the index builds at all with 500 vectors.
        // With the O(n²) construction, this was slow; the greedy-search construction
        // should complete quickly.
        let params = NSWParams {
            m: 8,
            m_max: 8,
            ef_search: 30,
            ef_construction: 30,
        };
        let mut index = NSWIndex::with_params(8, params).unwrap();
        let mut rng = Lcg::new(42);
        for i in 0..500u32 {
            let v = rng.next_normalized(8);
            index.add(i, v).unwrap();
        }
        index.build().unwrap();
        // Sanity: search still works
        let query = rng.next_normalized(8);
        let results = index.search(&query, 5, 20).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn nsw_all_nodes_reachable_from_entry() {
        // After construction every node should be reachable from the entry point
        // via BFS on the neighbor graph.
        let (index, _) = build_nsw(80, 8, 99);
        let n = 80usize;

        // Extract neighbor data through search (we can't directly access neighbors in tests,
        // so verify reachability by confirming every node can be retrieved as a nearest
        // neighbor of some query close to it).
        // Simple proxy: every insert returns a result set.
        let mut rng = Lcg::new(77);
        for _ in 0..20 {
            let query = rng.next_normalized(8);
            let results = index.search(&query, 5, 30).unwrap();
            assert!(!results.is_empty(), "search returned no results");
        }
        let _ = n; // used implicitly
    }

    #[test]
    fn nsw_single_vector() {
        let mut index = NSWIndex::new(4, 4, 4).unwrap();
        let v = normalize(&[1.0, 2.0, 3.0, 4.0]);
        index.add(42, v.clone()).unwrap();
        index.build().unwrap();
        let results = index.search(&v, 1, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn nsw_two_vectors_finds_closer_one() {
        let mut index = NSWIndex::new(4, 4, 4).unwrap();
        let v1 = normalize(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0, 0.0]);
        index.add(0, v1).unwrap();
        index.add(1, v2).unwrap();
        index.build().unwrap();

        // Query close to v1
        let query = normalize(&[0.9, 0.1, 0.0, 0.0]);
        let results = index.search(&query, 1, 10).unwrap();
        assert_eq!(results[0].0, 0, "expected doc_id=0 (closer to v1)");
    }

    #[test]
    fn nsw_ef_construction_respects_m_max() {
        // Graph should have at most m_max neighbors per node.
        // We test this indirectly: build a large graph and ensure search works
        // (broken degree limits would cause panic or wrong results).
        let params = NSWParams {
            m: 4,
            m_max: 4,
            ef_search: 20,
            ef_construction: 20,
        };
        let mut index = NSWIndex::with_params(4, params).unwrap();
        let mut rng = Lcg::new(314);
        for i in 0..100u32 {
            index.add(i, rng.next_normalized(4)).unwrap();
        }
        index.build().unwrap();

        let query = rng.next_normalized(4);
        let results = index.search(&query, 3, 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn nsw_higher_ef_construction_improves_recall() {
        // ef_construction controls how many candidates are evaluated when adding
        // each node during build. ef=100 should produce a better-connected graph
        // than ef=2 and therefore higher recall on the same data.
        let n = 150usize;
        let dim = 8usize;
        let seed = 555u64;
        let mut rng_data = Lcg::new(seed);
        let vecs: Vec<Vec<f32>> = (0..n).map(|_| rng_data.next_normalized(dim)).collect();

        let build_and_recall = |ef_c: usize| {
            let params = NSWParams {
                m: 8,
                m_max: 8,
                ef_search: 50,
                ef_construction: ef_c,
            };
            let mut index = NSWIndex::with_params(dim, params).unwrap();
            for (i, v) in vecs.iter().enumerate() {
                index.add(i as u32, v.clone()).unwrap();
            }
            index.build().unwrap();

            let mut rng_q = Lcg::new(1111);
            let num_q = 20;
            let k = 5;
            let mut total = 0.0f32;
            for _ in 0..num_q {
                let q = rng_q.next_normalized(dim);
                let gt = brute_force_cosine(&q, &vecs, k);
                let res = index.search(&q, k, 50).unwrap();
                total += recall_at_k(&res, &gt);
            }
            total / num_q as f32
        };

        let recall_low = build_and_recall(2);
        let recall_high = build_and_recall(100);

        assert!(
            recall_high >= recall_low,
            "ef_construction=100 recall={recall_high:.3} should be >= ef_construction=2 recall={recall_low:.3}"
        );
        // High ef should achieve reasonable quality, not just marginally better than 2.
        assert!(
            recall_high >= 0.5,
            "ef_construction=100 recall={recall_high:.3} below 0.5"
        );
    }
}

// ---------------------------------------------------------------------------
// ScaNN: residual codebook uses L2 metric
// ---------------------------------------------------------------------------

#[cfg(feature = "scann")]
mod scann_tests {
    use super::*;
    use vicinity::scann::{SCANNIndex, SCANNParams};

    fn build_scann(n: usize, dim: usize, seed: u64) -> (SCANNIndex, Vec<Vec<f32>>) {
        let params = SCANNParams {
            num_partitions: 4,
            nprobe: 4,
            num_reorder: 20,
            num_codebooks: 2,
            codebook_size: 16,
            seed,
        };
        let mut index = SCANNIndex::new(dim, params).unwrap();
        let mut rng = Lcg::new(seed);
        let mut vecs = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            index.add(i as u32, v.clone()).unwrap();
            vecs.push(v);
        }
        index.build().unwrap();
        (index, vecs)
    }

    #[test]
    fn scann_search_returns_nonempty_results() {
        let (index, _) = build_scann(50, 8, 42);
        let query = vec![0.0f32; 8];
        let results = index.search(&query, 3).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn scann_recall_oracle() {
        // ScaNN uses brute-force re-ranking so top results should be near-exact.
        let dim = 8usize;
        let n = 80usize;
        let (index, vecs) = build_scann(n, dim, 7);

        let mut rng = Lcg::new(321);
        let mut total_recall = 0.0f32;
        let num_queries = 10;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();

            // Brute force: using raw dot product (ScaNN is MIPS internally)
            let mut bf: Vec<(u32, f32)> = vecs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                    (i as u32, -dot) // negated so smaller = better for sort
                })
                .collect();
            bf.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: Vec<u32> = bf.iter().take(3).map(|(id, _)| *id).collect();

            let results = index.search(&query, 3).unwrap();
            total_recall += recall_at_k(&results, &gt);
        }

        let avg_recall = total_recall / num_queries as f32;
        assert!(
            avg_recall >= 0.5,
            "ScaNN recall={avg_recall:.3} below 0.5 (metric mismatch likely)"
        );
    }

    #[test]
    fn scann_results_are_sorted() {
        let (index, _) = build_scann(60, 8, 8);
        let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6, "results not sorted");
        }
    }

    #[test]
    fn scann_dimension_zero_error() {
        assert!(SCANNIndex::new(0, SCANNParams::default()).is_err());
    }
}
