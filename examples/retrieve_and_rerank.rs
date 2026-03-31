//! ANN retrieval followed by diversity reranking.
//!
//! Demonstrates the two-stage pattern:
//! 1. HNSW index returns top-k candidates ranked by distance.
//! 2. `rankops::mmr_embeddings` reranks those candidates for diversity via MMR.
//!
//! The data contains 4 "topics" with 5 near-duplicate vectors each. The query
//! overlaps all topics, so raw retrieval returns results sorted by distance
//! alone -- often clustering same-topic items together. MMR reshuffles the
//! list to ensure cross-topic coverage early in the ranking.
//!
//! ```bash
//! cargo run --example retrieve_and_rerank --release
//! ```

use std::collections::HashMap;

use rankops::{mmr_embeddings, MmrConfig};
use vicinity::hnsw::HNSWIndex;

const DIM: usize = 16;
const N_TOPICS: usize = 4;
const PER_TOPIC: usize = 5;

fn main() -> vicinity::Result<()> {
    // -- Build 20 vectors across 4 topics --
    // Each topic has a distinct direction. Within a topic, vectors are
    // near-identical (small perturbation). This simulates redundant retrieval
    // results: many docs about the same subtopic.
    let topic_dirs: [[f32; DIM]; N_TOPICS] = [
        // topic 0: signal in first 4 dims
        [
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        // topic 1: signal in dims 4..8
        [
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        // topic 2: signal in dims 8..12
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ],
        // topic 3: signal in dims 12..16
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ],
    ];

    let mut vectors: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut index = HNSWIndex::new(DIM, 8, 16)?;

    for (t, dir) in topic_dirs.iter().enumerate() {
        for dup in 0..PER_TOPIC {
            let id = (t * PER_TOPIC + dup) as u32;
            let v: Vec<f32> = dir
                .iter()
                .enumerate()
                .map(|(d, &val)| val + 0.02 * ((dup * 7 + d * 5) as f32 * 1.13).sin())
                .collect();
            let v = normalize(&v);
            index.add_slice(id, &v)?;
            vectors.insert(id, v);
        }
    }

    index.build()?;

    // -- Stage 1: ANN retrieval --
    // Query: uniform across all dims -> equidistant from all 4 topics.
    // Retrieve all 20 candidates (simulating an over-fetch from a larger index).
    let query = normalize(&[1.0; DIM]);
    let raw = index.search(&query, 20, 50)?;

    println!("--- Stage 1: HNSW retrieval (top 12 of 20) ---");
    for (id, dist) in raw.iter().take(12) {
        println!("  id={:2}  topic={}  distance={:.4}", id, id / 5, dist);
    }

    // -- Stage 2: MMR reranking --
    // Convert HNSW distances to relevance scores (higher = better).
    let max_dist = raw.iter().map(|(_, d)| *d).fold(0.0_f32, f32::max);

    let candidates: Vec<(u32, f32, Vec<f32>)> = raw
        .iter()
        .map(|&(id, dist)| {
            let relevance = (max_dist - dist) + 0.01;
            (id, relevance, vectors[&id].clone())
        })
        .collect();

    let config = MmrConfig::new(0.5).with_top_k(8);
    let reranked = mmr_embeddings(&candidates, config);

    println!("\n--- Stage 2: MMR reranked (top 8, lambda=0.5) ---");
    for (id, score) in &reranked {
        println!("  id={:2}  topic={}  mmr_score={:.4}", id, id / 5, score);
    }

    // -- Compare first-N topic coverage --
    let n = N_TOPICS;
    let raw_topics: Vec<u32> = raw.iter().take(n).map(|(id, _)| id / 5).collect();
    let mmr_topics: Vec<u32> = reranked.iter().take(n).map(|(id, _)| id / 5).collect();

    let raw_uniq = unique_count(&raw_topics);
    let mmr_uniq = unique_count(&mmr_topics);

    println!("\n--- Topic coverage (first {n} picks) ---");
    println!("  HNSW : {raw_uniq} unique topics {raw_topics:?}");
    println!("  MMR  : {mmr_uniq} unique topics {mmr_topics:?}");

    if mmr_uniq > raw_uniq {
        println!("  -> MMR spread results across more topics.");
    } else {
        println!("  -> Equal coverage (topics were already interleaved by distance).");
    }

    Ok(())
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn unique_count(items: &[u32]) -> usize {
    let mut seen = items.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}
