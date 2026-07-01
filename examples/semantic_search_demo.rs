#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Semantic Search Demo
//!
//! Synthetic document-search workload:
//! - Synthetic embeddings mimicking real-world distributions
//! - HNSW index construction and tuning
//! - Query latency and recall analysis
//!
//! ```bash
//! cargo run --example semantic_search_demo --release
//! ```

use std::collections::HashSet;
use std::time::Instant;
use vicinity::hnsw::HNSWIndex;

fn main() -> vicinity::Result<()> {
    println!("Semantic Search Demo");
    println!("====================\n");

    println!("Simulating a document search system with synthetic embeddings.\n");

    // 1. Generate synthetic document embeddings
    let (corpus, metadata) = generate_corpus();
    println!("Corpus Statistics:");
    println!("  Documents: {}", metadata.len());
    println!("  Embedding dimension: 768");
    println!("  Categories: technology, science, business, health, sports\n");

    // 2. Build HNSW index
    demo_index_construction(&corpus)?;

    // 3. Show recall vs latency tradeoff
    demo_recall_latency_tradeoff(&corpus)?;

    // 4. Interactive search simulation
    demo_search(&corpus, &metadata)?;

    println!("Done!");
    Ok(())
}

struct DocMetadata {
    #[allow(dead_code)]
    id: usize,
    category: &'static str,
    title: String,
}

fn generate_corpus() -> (Vec<Vec<f32>>, Vec<DocMetadata>) {
    let dim = 768;
    let categories = [
        ("technology", 2000),
        ("science", 1500),
        ("business", 1800),
        ("health", 1200),
        ("sports", 1500),
    ];

    let mut embeddings = Vec::new();
    let mut metadata = Vec::new();

    // Each category gets a distinct "cluster" in embedding space
    for (cat_idx, (category, count)) in categories.iter().enumerate() {
        // Category centroid (different regions of the embedding space)
        let centroid: Vec<f32> = (0..dim)
            .map(|d| {
                let phase = (cat_idx * 137 + d * 17) as f32 * 0.01;
                (phase.sin() + phase.cos()) * (cat_idx as f32 + 1.0) * 0.3
            })
            .collect();

        for doc_idx in 0..*count {
            // Document embedding: centroid + semantic variation + noise
            let embedding: Vec<f32> = (0..dim)
                .map(|d| {
                    let semantic_variation = ((doc_idx * dim + d) as f32 * 0.001).sin() * 0.5;
                    let noise = ((doc_idx * dim + d + 7919) as f32 * 0.0001).cos() * 0.1;
                    centroid[d] + semantic_variation + noise
                })
                .collect();

            // Normalize to unit length (typical for sentence embeddings)
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

            embeddings.push(normalized);
            metadata.push(DocMetadata {
                id: embeddings.len() - 1,
                category,
                title: format!("{} Article #{}", category, doc_idx + 1),
            });
        }
    }

    (embeddings, metadata)
}

fn demo_index_construction(corpus: &[Vec<f32>]) -> vicinity::Result<()> {
    println!("Index Construction");
    println!("------------------\n");

    let dim = corpus[0].len();

    // Compare different M values
    let configs = [
        ("Low connectivity (M=8)", 8, 16),
        ("Medium connectivity (M=16)", 16, 32),
        ("High connectivity (M=32)", 32, 64),
    ];

    println!(
        "  {:>30}  {:>12}  {:>15}",
        "Configuration", "Build Time", "Memory Est."
    );
    println!("  {:->30}  {:->12}  {:->15}", "", "", "");

    for (name, m, m_max) in &configs {
        let start = Instant::now();
        let mut index = HNSWIndex::new(dim, *m, *m_max)?;
        for (i, vec) in corpus.iter().enumerate() {
            index.add(i as u32, vec.clone())?;
        }
        index.build()?;
        let build_time = start.elapsed();

        // Estimate memory (vectors + graph edges)
        let vector_mem = corpus.len() * dim * 4; // f32
        let edge_mem = corpus.len() * (*m_max) * 8; // u32 edges + overhead
        let total_mem = (vector_mem + edge_mem) as f64 / 1_000_000.0;

        println!(
            "  {:>30}  {:>10.2}s  {:>13.1}MB",
            name,
            build_time.as_secs_f64(),
            total_mem
        );
    }
    println!();

    Ok(())
}

fn demo_recall_latency_tradeoff(corpus: &[Vec<f32>]) -> vicinity::Result<()> {
    println!("Recall vs Latency Tradeoff");
    println!("--------------------------\n");

    let dim = corpus[0].len();
    let n_queries = 100;
    let k = 10;

    // Build index
    let mut index = HNSWIndex::new(dim, 16, 32)?;
    for (i, vec) in corpus.iter().enumerate() {
        index.add(i as u32, vec.clone())?;
    }
    index.build()?;

    // Generate queries (perturbed versions of random docs)
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| {
            let base = &corpus[(i * 79) % corpus.len()];
            let perturbed: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let noise = ((i * dim + j) as f32 * 0.001).sin() * 0.05;
                    v + noise
                })
                .collect();
            normalize(&perturbed)
        })
        .collect();

    // Compute ground truth (brute force)
    let ground_truth: Vec<HashSet<u32>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(u32, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, doc)| (i as u32, cosine_distance(q, doc)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(id, _)| *id).collect()
        })
        .collect();

    // Test different ef values
    println!(
        "  {:>8}  {:>12}  {:>12}  {:>10}",
        "ef", "Recall@10", "Latency", "QPS"
    );
    println!("  {:->8}  {:->12}  {:->12}  {:->10}", "", "", "", "");

    for ef in [10, 20, 50, 100, 200, 500] {
        let start = Instant::now();
        let mut total_recall = 0.0;

        for (i, query) in queries.iter().enumerate() {
            let results = index.search(query, k, ef)?;
            let result_ids: HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
            let recall = result_ids.intersection(&ground_truth[i]).count() as f32 / k as f32;
            total_recall += recall;
        }

        let elapsed = start.elapsed();
        let avg_recall = total_recall / n_queries as f32;
        let qps = n_queries as f64 / elapsed.as_secs_f64();
        let latency_us = elapsed.as_micros() as f64 / n_queries as f64;

        println!(
            "  {:>8}  {:>11.1}%  {:>10.0}us  {:>9.0}",
            ef,
            avg_recall * 100.0,
            latency_us,
            qps
        );
    }
    println!();

    println!("  In this synthetic corpus, higher ef trades QPS for recall.");
    println!("  Choose ef from the measured table above.\n");

    Ok(())
}

fn demo_search(corpus: &[Vec<f32>], metadata: &[DocMetadata]) -> vicinity::Result<()> {
    println!("Search Demonstration");
    println!("--------------------\n");

    let dim = corpus[0].len();

    // Build index
    let mut index = HNSWIndex::new(dim, 16, 32)?;
    for (i, vec) in corpus.iter().enumerate() {
        index.add(i as u32, vec.clone())?;
    }
    index.build()?;

    // Simulate different query types
    let scenarios = [
        ("Technology query", 0, "technology"), // Query from tech cluster
        ("Cross-domain query", 2500, "science"), // Query from science, might find related tech
        ("Rare topic query", 7500, "sports"),  // Query from sports
    ];

    for (name, base_idx, expected_cat) in &scenarios {
        println!("  Scenario: {}", name);
        println!("  Expected category: {}\n", expected_cat);

        // Create query by perturbing a document
        let query = perturb(&corpus[*base_idx], 0.1);

        let start = Instant::now();
        let results = index.search(&query, 5, 50)?;
        let latency = start.elapsed();

        println!(
            "  {:>4}  {:>35}  {:>12}  {:>8}",
            "Rank", "Title", "Category", "Score"
        );
        println!("  {:->4}  {:->35}  {:->12}  {:->8}", "", "", "", "");

        for (rank, (id, dist)) in results.iter().enumerate() {
            let doc = &metadata[*id as usize];
            let sim = 1.0 - dist; // Convert distance to similarity
            println!(
                "  {:>4}  {:>35}  {:>12}  {:>7.3}",
                rank + 1,
                &doc.title[..doc.title.len().min(35)],
                doc.category,
                sim
            );
        }
        println!("  Latency: {:?}\n", latency);
    }

    // Category distribution analysis
    println!("  Category Distribution in Top-5 Results:");
    println!("  ----------------------------------------");

    let mut category_counts: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    let n_sample_queries = 500;

    for i in 0..n_sample_queries {
        let query = perturb(&corpus[(i * 17) % corpus.len()], 0.1);
        let results = index.search(&query, 5, 50)?;

        for (id, _) in results {
            let cat = metadata[id as usize].category;
            *category_counts.entry(cat).or_insert(0) += 1;
        }
    }

    for (cat, count) in &category_counts {
        let pct = *count as f64 / (n_sample_queries * 5) as f64 * 100.0;
        println!("  {:>12}: {:>5} ({:>5.1}%)", cat, count, pct);
    }
    println!();

    Ok(())
}

// --- Helpers ---

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn perturb(v: &[f32], magnitude: f32) -> Vec<f32> {
    let perturbed: Vec<f32> = v
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let noise = ((i as f32 * 0.01).sin() + (i as f32 * 0.017).cos()) * magnitude;
            x + noise
        })
        .collect();
    normalize(&perturbed)
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (norm_a * norm_b + f32::EPSILON)
}
