#![allow(clippy::unwrap_used)]
//! Embedding inference + vector search pipeline.
//!
//! Demonstrates the pattern for using vicinity as the retrieval layer in a
//! RAG (Retrieval-Augmented Generation) pipeline:
//!
//! 1. Embed documents using a model (mocked here; use candle/burn in production)
//! 2. Build a vicinity HNSW index from the embeddings
//! 3. At query time: embed the query, search the index, return documents
//!
//! ## With candle (production)
//!
//! ```rust,ignore
//! use candle_core::{Device, Tensor};
//! use candle_transformers::models::bert::{BertModel, Config};
//!
//! let model = BertModel::load(weights, &config, device)?;
//! let embeddings = model.forward(&token_ids)?;
//! let embedding_vec: Vec<f32> = embeddings.to_vec1()?;
//! index.add_slice(doc_id, &embedding_vec)?;
//! ```
//!
//! ```sh
//! cargo run --example embedding_pipeline --features hnsw
//! ```

use vicinity::hnsw::HNSWIndex;

const DIM: usize = 64;

/// Mock embedding function.
///
/// Hashes each word and scatters contributions across dimensions, then
/// normalizes. Texts sharing words produce similar vectors -- enough to
/// demonstrate the retrieval pattern without a real model.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    for word in text.split_whitespace() {
        // Simple hash: sum of (byte * position) per character.
        let h: u64 = word
            .bytes()
            .enumerate()
            .map(|(i, b)| (i as u64 + 1) * b as u64)
            .sum();
        // Spread across multiple dimensions using the hash.
        for d in 0..dim {
            let angle = (h.wrapping_mul(d as u64 + 1)) as f32 * 0.01;
            v[d] += angle.sin();
        }
    }
    normalize(&v)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn main() -> vicinity::Result<()> {
    // -- 1. Document corpus --
    let documents = [
        "Rust is a systems programming language focused on safety and performance",
        "Python is popular for data science and machine learning workflows",
        "JavaScript powers interactive web applications and server-side code",
        "The Rust borrow checker prevents data races at compile time",
        "Neural networks learn representations from large training datasets",
        "PostgreSQL is a relational database with strong ACID guarantees",
        "Transformers revolutionized natural language processing tasks",
        "WebAssembly enables near-native performance in the browser",
        "Gradient descent optimizes model parameters during training",
        "Redis provides in-memory key-value storage for caching",
        "HNSW indexes enable fast approximate nearest neighbor search",
        "Kubernetes orchestrates containerized application deployments",
        "Attention mechanisms let models focus on relevant input tokens",
        "Rust async/await enables efficient concurrent IO without threads",
        "Vector databases store embeddings for semantic retrieval",
    ];

    // -- 2. Embed all documents --
    let embeddings: Vec<Vec<f32>> = documents.iter().map(|doc| embed(doc, DIM)).collect();

    // -- 3. Build HNSW index --
    let mut index = HNSWIndex::new(DIM, 16, 32)?;
    for (i, emb) in embeddings.iter().enumerate() {
        index.add_slice(i as u32, emb)?;
    }
    index.build()?;

    println!("Indexed {} documents (dim={})\n", documents.len(), DIM);

    // -- 4. Query --
    let queries = [
        "safe systems programming with compile-time checks",
        "machine learning model training",
        "fast similarity search over vectors",
    ];

    for query in &queries {
        let query_emb = embed(query, DIM);
        let results = index.search(&query_emb, 5, 50)?;

        println!("Query: \"{query}\"");
        for (rank, (doc_id, dist)) in results.iter().enumerate() {
            let sim = 1.0 - dist;
            println!(
                "  {}. [{:.3}] {}",
                rank + 1,
                sim,
                documents[*doc_id as usize]
            );
        }
        println!();
    }

    Ok(())
}
