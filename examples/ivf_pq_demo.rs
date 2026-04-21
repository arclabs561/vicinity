#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_update)]
//! IVF-PQ: Inverted File with Product Quantization
//!
//! The workhorse of billion-scale search. Combines two ideas:
//! 1. **IVF** (Inverted File): Partition space into cells, only search nearby cells
//! 2. **PQ** (Product Quantization): Compress vectors to ~32 bytes
//!
//! # How It Works
//!
//! ```text
//! Index time:
//!   vectors -> k-means clusters (IVF) -> compress residuals (PQ)
//!
//! Query time:
//!   query -> find nprobe nearest clusters -> scan compressed vectors
//!          -> rerank top candidates with exact distance
//! ```
//!
//! # Trade-offs
//!
//! | Parameter    | Higher Value                | Lower Value             |
//! |--------------|-----------------------------|-----------------------  |
//! | n_lists      | More clusters, faster scan  | Better recall           |
//! | n_subvectors | More compression            | Higher accuracy         |
//! | nprobe       | Better recall               | Faster queries          |
//!
//! # When to Use
//!
//! - 1M+ vectors where HNSW memory is prohibitive
//! - GPU-accelerated search (IVF is parallelizable)
//! - When you have training data to learn codebooks
//!
//! ```bash
//! cargo run --example ivf_pq_demo --release --features ivf_pq
//! ```

use std::collections::HashSet;
use std::time::Instant;
use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};

fn main() -> vicinity::Result<()> {
    println!("IVF-PQ: Billion-Scale Vector Search");
    println!("====================================\n");

    demo_basic_search()?;
    demo_parameter_tuning()?;
    demo_memory_analysis()?;
    demo_when_to_use()?;

    println!("Done!");
    Ok(())
}

fn demo_basic_search() -> vicinity::Result<()> {
    println!("1. Basic IVF-PQ Search");
    println!("   --------------------\n");

    let dim = 128;
    let n = 5000;
    let k = 10;

    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| generate_embedding(dim, i as u64)).collect();

    println!("   Building index: {} vectors, dim={}", n, dim);

    // Create index
    let params = IVFPQParams {
        num_clusters: 100,  // 100 clusters
        num_codebooks: 16,  // Split into 16 sub-vectors (codebooks)
        codebook_size: 256, // 256 centroids per codebook
        nprobe: 10,         // Search 10 clusters
        ..Default::default()
    };

    let mut index = IVFPQIndex::new(dim, params)?;

    // Add vectors
    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        index.add(i as u32, vec.clone())?;
    }
    let add_time = start.elapsed();

    // Build (train PQ codebooks, cluster)
    let start = Instant::now();
    index.build()?;
    let build_time = start.elapsed();

    println!("   Add time: {:?}", add_time);
    println!("   Build time: {:?}", build_time);

    // Search
    let query = &vectors[0];
    let start = Instant::now();
    let results = index.search(query, k)?;
    let search_time = start.elapsed();

    println!("\n   Query time: {:?}", search_time);
    println!("   Top {} results:", k);
    for (i, (id, dist)) in results.iter().enumerate() {
        println!("     {}: id={}, dist={:.4}", i + 1, id, dist);
    }

    // Measure recall
    let gt = brute_force_knn(query, &vectors, k);
    let gt_set: HashSet<u32> = gt.into_iter().collect();
    let result_set: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
    let recall = gt_set.intersection(&result_set).count() as f32 / k as f32;

    println!("\n   Recall@{}: {:.1}%", k, recall * 100.0);
    println!();

    Ok(())
}

fn demo_parameter_tuning() -> vicinity::Result<()> {
    println!("2. Parameter Impact on Recall/Speed");
    println!("   ---------------------------------\n");

    let dim = 64;
    let n = 2000;
    let k = 10;

    let vectors: Vec<Vec<f32>> = (0..n).map(|i| generate_embedding(dim, i as u64)).collect();

    println!("   Testing nprobe impact (n_lists=50):\n");
    println!("   {:>8} {:>10} {:>10}", "nprobe", "Recall@10", "Time");
    println!("   {}", "-".repeat(30));

    for nprobe in [1, 5, 10, 25, 50] {
        let params = IVFPQParams {
            num_clusters: 50,
            num_codebooks: 8,
            codebook_size: 256,
            nprobe,
            ..Default::default()
        };

        let mut index = IVFPQIndex::new(dim, params)?;
        for (i, vec) in vectors.iter().enumerate() {
            index.add(i as u32, vec.clone())?;
        }
        index.build()?;

        // Measure recall and time over multiple queries
        let mut total_recall = 0.0f32;
        let n_queries = 20;
        let start = Instant::now();

        for q in 0..n_queries {
            let query = &vectors[(q * 17) % n];
            let results = index.search(query, k)?;

            let gt = brute_force_knn(query, &vectors, k);
            let gt_set: HashSet<u32> = gt.into_iter().collect();
            let result_set: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
            total_recall += gt_set.intersection(&result_set).count() as f32 / k as f32;
        }

        let avg_time = start.elapsed() / n_queries as u32;
        let avg_recall = total_recall / n_queries as f32;

        println!(
            "   {:>8} {:>9.1}% {:>10?}",
            nprobe,
            avg_recall * 100.0,
            avg_time
        );
    }

    println!("\n   Observation: nprobe controls recall/speed trade-off at query time.");
    println!("   Higher nprobe = better recall but slower queries.");
    println!();

    Ok(())
}

fn demo_memory_analysis() -> vicinity::Result<()> {
    println!("3. Memory Analysis");
    println!("   ----------------\n");

    let dim: u64 = 768; // BERT-base
    let n: u64 = 1_000_000;

    // Calculate memory for different configurations
    println!("   For {} vectors, dim={}:\n", n, dim);
    println!(
        "   {:20} {:>15} {:>12}",
        "Configuration", "Memory", "Compression"
    );
    println!("   {}", "-".repeat(50));

    // Full f32
    let full_bytes = n * dim * 4;
    println!(
        "   {:20} {:>12.1} GB {:>11.1}x",
        "Full f32",
        full_bytes as f64 / 1e9,
        1.0
    );

    // IVF-PQ configurations
    // m codebooks, each stores 1 byte code (256 centroids)
    for (m, name) in [
        (64u64, "IVF-PQ (m=64)"),
        (32u64, "IVF-PQ (m=32)"),
        (16u64, "IVF-PQ (m=16)"),
    ] {
        // m bytes per vector (1 byte per codebook)
        let pq_bytes = n * m;
        let compression = full_bytes as f64 / pq_bytes as f64;
        println!(
            "   {:20} {:>12.2} GB {:>11.1}x",
            name,
            pq_bytes as f64 / 1e9,
            compression
        );
    }

    // RaBitQ for comparison
    let rabitq_4bit = n * dim / 2; // 4 bits per dim = 0.5 bytes per dim
    println!(
        "   {:20} {:>12.2} GB {:>11.1}x",
        "RaBitQ (4-bit)",
        rabitq_4bit as f64 / 1e9,
        full_bytes as f64 / rabitq_4bit as f64
    );

    println!("\n   Key insight: PQ achieves 48-192x compression by learning");
    println!("   codebooks that capture the data distribution.");
    println!();

    Ok(())
}

fn demo_when_to_use() -> vicinity::Result<()> {
    println!("4. IVF-PQ vs HNSW Decision Guide");
    println!("   ------------------------------\n");

    println!("   | Criterion       | IVF-PQ           | HNSW             |");
    println!("   |-----------------|------------------|------------------|");
    println!("   | Scale           | 1B+ vectors      | 1-100M vectors   |");
    println!("   | Memory          | ~32 bytes/vec    | ~300 bytes/vec   |");
    println!("   | Query latency   | ~1-10ms          | ~0.1-1ms         |");
    println!("   | Recall@10       | 80-95%           | 95-99%           |");
    println!("   | GPU support     | Excellent        | Limited          |");
    println!("   | Training        | Required         | None             |");
    println!();

    println!("   Use IVF-PQ when:");
    println!("     - Billions of vectors (memory matters)");
    println!("     - GPU acceleration available (FAISS, etc.)");
    println!("     - Batch queries (amortize overhead)");
    println!("     - Have training data for codebook learning");
    println!();

    println!("   Use HNSW when:");
    println!("     - <100M vectors and memory is OK");
    println!("     - Single-query latency is critical");
    println!("     - Highest recall required");
    println!("     - No training data available");
    println!();

    println!("   Hybrid approach (HNSW + PQ):");
    println!("     - HNSW for graph navigation");
    println!("     - PQ codes stored per node for memory reduction");
    println!("     - Rerank top candidates with exact vectors");
    println!();

    println!("   See also:");
    println!("     - `rabitq_demo.rs`: Training-free quantization");
    println!("     - `hnsw_benchmark.rs`: HNSW performance analysis");
    println!("     - `02_measure_recall.rs`: Recall measurement methodology");

    Ok(())
}

// =============================================================================
// Helpers
// =============================================================================

fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let embedding: Vec<f32> = (0..dim)
        .map(|i| {
            let u1 = lcg_random(seed.wrapping_add(i as u64 * 2));
            let u2 = lcg_random(seed.wrapping_add(i as u64 * 2 + 1));
            (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    normalize(&embedding)
}

fn lcg_random(seed: u64) -> f32 {
    let a: u64 = 6364136223846793005;
    let c: u64 = 1442695040888963407;
    let next = seed.wrapping_mul(a).wrapping_add(c);
    (next as f64 / u64::MAX as f64) as f32
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn brute_force_knn(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist: f32 = query.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
            (i as u32, dist)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.into_iter().take(k).map(|(id, _)| id).collect()
}
