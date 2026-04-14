#![allow(clippy::unwrap_used, clippy::expect_used)]
//! SymphonyQG-style quantized HNSW search
//!
//! Demonstrates using `search_with_distance` to drive HNSW beam search with
//! RaBitQ approximate distances instead of full f32 cosine distance.
//!
//! Two-phase pattern (matches the SymphonyQG paper):
//!
//! 1. Build HNSW on full f32 vectors (quality graph construction)
//! 2. Quantize vectors with RaBitQ (fit + quantize from reordered storage)
//! 3. Search: rotate query once (O(d^2)), then score each neighbor with
//!    `approximate_l2_sqr_prerotated` (O(d) over quantized codes)
//! 4. Rerank the beam-search candidate set with exact cosine distance
//!
//! ```bash
//! cargo run --example symphonyqg_demo --release --features "hnsw,rabitq,quantization"
//! ```

use std::time::Instant;
use vicinity::hnsw::HNSWIndex;
use vicinity::quantization::rabitq::{RaBitQConfig, RaBitQQuantizer};

fn main() -> vicinity::Result<()> {
    println!("SymphonyQG-style quantized HNSW search");
    println!("=======================================\n");

    let dim = 64;
    let n = 10_000;
    let k = 10;
    let ef = 128;
    let n_queries = 200;
    let seed = 42u64;

    // --- Generate data -------------------------------------------------------

    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized(i, dim)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| {
            // Perturb a database vector slightly so ground truth is deterministic.
            let base = &vectors[(i * 37 + 13) % n];
            let perturbed: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(j, &v)| v + ((i * dim + j) as f32 * 0.0003).sin() * 0.05)
                .collect();
            normalize(&perturbed)
        })
        .collect();

    // --- Build HNSW ----------------------------------------------------------

    println!("Building HNSW ({n} vectors, dim={dim})...");
    let build_start = Instant::now();
    let mut index = HNSWIndex::new(dim, 16, 32)?;
    for (i, v) in vectors.iter().enumerate() {
        index.add_slice(i as u32, v)?;
    }
    index.build()?;
    println!("  built in {:.2?}", build_start.elapsed());

    // --- Compute exact ground truth ------------------------------------------

    let gt: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| brute_force_k_nearest(&vectors, q, k))
        .collect();

    // --- Exact HNSW search (baseline) ----------------------------------------

    let exact_start = Instant::now();
    let exact_results: Vec<Vec<(u32, f32)>> = queries
        .iter()
        .map(|q| index.search(q, k, ef).unwrap())
        .collect();
    let exact_elapsed = exact_start.elapsed();

    let exact_recall = recall_at_k(&exact_results, &gt, k);
    let exact_qps = n_queries as f64 / exact_elapsed.as_secs_f64();

    println!(
        "\nExact HNSW (ef={ef}):       recall@{k} = {:.1}%  ({:.0} QPS)",
        exact_recall * 100.0,
        exact_qps
    );

    // --- Quantized HNSW search -----------------------------------------------
    //
    // After build(), vectors are BFS-reordered for cache locality.
    // internal_id passed to the dist_fn closure is the position in raw_vectors().
    // We MUST quantize from raw_vectors() (post-reorder), not from the original
    // insertion-order array -- otherwise codes[internal_id] would be mismatched.

    let raw = index.raw_vectors();
    let n_stored = raw.len() / dim; // may equal n

    let config = RaBitQConfig::bits4();
    let mut quantizer = RaBitQQuantizer::with_config(dim, seed, config)?;

    // fit() computes the centroid used for pre-rotation.
    quantizer.fit(raw, n_stored)?;

    // One QuantizedVector per internal node, indexed by internal_id.
    let codes: Vec<_> = (0..n_stored)
        .map(|i| {
            let start = i * dim;
            quantizer.quantize(&raw[start..start + dim]).unwrap()
        })
        .collect();

    // Closure: rotate query once outside, capture rotated form.
    // This is the key SymphonyQG trick -- O(d^2) rotation amortized across
    // all neighbor evaluations in the beam.
    let quant_start = Instant::now();
    let mut quant_results: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n_queries);

    for q in &queries {
        let rotated = quantizer.rotate_query(q)?;

        // Phase 1: beam search driven by approximate distance.
        let approx_candidates = index.search_with_distance(q, k * 4, ef, &|_q, id| {
            RaBitQQuantizer::approximate_l2_sqr_prerotated(&rotated, &codes[id as usize])
        })?;

        // Phase 2: rerank the expanded candidate set with exact cosine distance.
        // approx candidates already filtered to non-deleted + doc_id-mapped by
        // search_with_distance; rerank on the fly.
        let mut reranked: Vec<(u32, f32)> = approx_candidates
            .into_iter()
            .map(|(doc_id, _approx)| {
                let exact = cosine_dist_normalized(q, &vectors[doc_id as usize]);
                (doc_id, exact)
            })
            .collect();
        reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        reranked.truncate(k);
        quant_results.push(reranked);
    }

    let quant_elapsed = quant_start.elapsed();
    let quant_recall = recall_at_k(&quant_results, &gt, k);
    let quant_qps = n_queries as f64 / quant_elapsed.as_secs_f64();

    println!(
        "Quantized HNSW (ef={ef}):   recall@{k} = {:.1}%  ({:.0} QPS)",
        quant_recall * 100.0,
        quant_qps,
    );

    // --- Quantized beam only (no rerank) -------------------------------------

    let norank_start = Instant::now();
    let mut norank_results: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n_queries);

    for q in &queries {
        let rotated = quantizer.rotate_query(q)?;
        let results = index.search_with_distance(q, k, ef, &|_q, id| {
            RaBitQQuantizer::approximate_l2_sqr_prerotated(&rotated, &codes[id as usize])
        })?;
        norank_results.push(results);
    }

    let norank_elapsed = norank_start.elapsed();
    let norank_recall = recall_at_k(&norank_results, &gt, k);
    let norank_qps = n_queries as f64 / norank_elapsed.as_secs_f64();

    println!(
        "Quantized beam only:       recall@{k} = {:.1}%  ({:.0} QPS)",
        norank_recall * 100.0,
        norank_qps,
    );

    // --- Summary -------------------------------------------------------------

    println!("\nSummary");
    println!("-------");
    println!(
        "  Reranked QPS speedup over exact: {:.2}x",
        quant_qps / exact_qps
    );
    println!(
        "  Beam-only QPS speedup over exact: {:.2}x",
        norank_qps / exact_qps
    );
    println!(
        "  Recall: exact={:.1}%  reranked={:.1}%  beam-only={:.1}%",
        exact_recall * 100.0,
        quant_recall * 100.0,
        norank_recall * 100.0,
    );
    println!("\nNote: At dim=64 RaBitQ approximation is coarser than at high dimensions.");
    println!("      Recall improves at dim>=256 (error bound is O(1/sqrt(d))).");

    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_normalized(seed: usize, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim)
        .map(|j| ((seed * dim + j) as f32 * 0.618_034).fract() * 2.0 - 1.0)
        .collect();
    normalize(&v)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn cosine_dist_normalized(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    (1.0 - dot).max(0.0)
}

/// Brute-force ground truth: returns sorted doc_ids of the k nearest vectors.
fn brute_force_k_nearest(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine_dist_normalized(query, v)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.truncate(k);
    dists.into_iter().map(|(id, _)| id).collect()
}

/// Mean recall@k: fraction of ground-truth neighbors found on average.
fn recall_at_k(results: &[Vec<(u32, f32)>], gt: &[Vec<u32>], k: usize) -> f64 {
    let total: usize = results
        .iter()
        .zip(gt)
        .map(|(res, truth)| {
            let found = res
                .iter()
                .take(k)
                .filter(|(id, _)| truth.contains(id))
                .count();
            found
        })
        .sum();
    total as f64 / (results.len() * k) as f64
}
