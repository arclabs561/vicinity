//! Hybrid search: combining text (BM25) and vector similarity via
//! Reciprocal Rank Fusion (RRF).
//!
//! This example shows how to combine vicinity's HNSW vector search with
//! a simple BM25 text scorer. In production, replace the BM25 implementation
//! with tantivy or another full-text search library.
//!
//! ```bash
//! cargo run --example hybrid_search --features hnsw
//! ```

use std::collections::HashMap;

use vicinity::hnsw::HNSWIndex;

// ---------------------------------------------------------------------------
// Document corpus
// ---------------------------------------------------------------------------

struct Document {
    title: &'static str,
    text: &'static str,
    embedding: Vec<f32>,
}

fn build_corpus() -> Vec<Document> {
    // 16-dim embeddings. Each document gets a hand-crafted direction so that
    // semantically related docs cluster together in vector space.
    let docs: Vec<(&str, &str, [f32; 16])> = vec![
        (
            "Rust basics",
            "Rust is a systems programming language focused on safety and performance",
            [
                1.0, 0.8, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            "C++ overview",
            "C++ is a fast systems language used for operating systems and game engines",
            [
                0.9, 0.7, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            "Python intro",
            "Python is a high-level language popular for scripting and data science",
            [
                0.1, 0.1, 0.0, 0.0, 0.9, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            "Go concurrency",
            "Go provides goroutines and channels for concurrent systems programming",
            [
                0.7, 0.5, 0.4, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            "JavaScript web",
            "JavaScript powers interactive web applications and runs in the browser",
            [
                0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.0, 0.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            "Rust async",
            "Async Rust uses futures and tokio for high-performance network services",
            [
                0.9, 0.7, 0.5, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            "Database systems",
            "Relational databases use SQL for querying structured data with ACID guarantees",
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.3, 0.0, 0.0, 0.0,
            ],
        ),
        (
            "Machine learning",
            "Neural networks learn patterns from data using gradient descent optimization",
            [
                0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.2, 0.0,
            ],
        ),
        (
            "Operating systems",
            "An OS kernel manages hardware resources and provides fast system call interfaces",
            [
                0.6, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8,
            ],
        ),
        (
            "Rust memory",
            "Rust ownership and borrowing prevent memory bugs without garbage collection",
            [
                0.9, 0.9, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
            ],
        ),
        (
            "WebAssembly",
            "WebAssembly compiles systems languages to run fast portable code in browsers",
            [
                0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            "Garbage collection",
            "Tracing garbage collectors reclaim memory automatically but add latency pauses",
            [
                0.2, 0.2, 0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6,
            ],
        ),
    ];

    docs.into_iter()
        .map(|(title, text, emb)| Document {
            title,
            text,
            embedding: normalize(&emb),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Minimal BM25 scorer
// ---------------------------------------------------------------------------

/// Compute BM25 scores for a query against a set of documents.
/// Returns (doc_index, score) pairs sorted by descending score.
fn bm25_rank(docs: &[Document], query: &str) -> Vec<(usize, f64)> {
    let k1: f64 = 1.2;
    let b: f64 = 0.75;
    let n = docs.len() as f64;

    let query_terms: Vec<String> = tokenize(query);

    // Precompute document lengths and average length.
    let doc_tokens: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d.text)).collect();
    let avg_dl: f64 = doc_tokens.iter().map(|t| t.len() as f64).sum::<f64>() / n;

    // Document frequency per query term.
    let mut df: HashMap<&str, usize> = HashMap::new();
    for qt in &query_terms {
        let count = doc_tokens
            .iter()
            .filter(|tokens| tokens.iter().any(|t| t == qt))
            .count();
        df.insert(qt.as_str(), count);
    }

    let mut scores: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let dl = doc_tokens[i].len() as f64;
            let mut score = 0.0_f64;
            for qt in &query_terms {
                let tf = doc_tokens[i].iter().filter(|t| *t == qt).count() as f64;
                let doc_freq = *df.get(qt.as_str()).unwrap_or(&0) as f64;
                if doc_freq == 0.0 {
                    continue;
                }
                // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let idf = ((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
                // TF saturation
                let tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avg_dl));
                score += idf * tf_norm;
            }
            (i, score)
        })
        .filter(|(_, s)| *s > 0.0)
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

// ---------------------------------------------------------------------------
// Reciprocal Rank Fusion
// ---------------------------------------------------------------------------

/// Fuse multiple ranked lists using RRF.
/// Each input is a list of (doc_index, score) in descending-score order.
/// Returns fused (doc_index, rrf_score) sorted by descending RRF score.
fn reciprocal_rank_fusion(lists: &[&[(usize, f64)]], k: f64) -> Vec<(usize, f64)> {
    let mut fused: HashMap<usize, f64> = HashMap::new();
    for list in lists {
        for (rank, &(doc_id, _score)) in list.iter().enumerate() {
            *fused.entry(doc_id).or_default() += 1.0 / (k + rank as f64 + 1.0);
        }
    }
    let mut result: Vec<(usize, f64)> = fused.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> vicinity::Result<()> {
    let corpus = build_corpus();
    let dim = 16;

    // -- Build HNSW index on embeddings --
    let mut index = HNSWIndex::new(dim, 8, 16)?;
    for (i, doc) in corpus.iter().enumerate() {
        index.add_slice(i as u32, &doc.embedding)?;
    }
    index.build()?;

    // -- Queries --
    // Text query targets BM25; vector query targets HNSW.
    // A real system would embed the text query; here we use a hand-crafted
    // vector that points toward "systems programming" in our embedding space.
    let text_query = "fast systems programming language";
    let vector_query = normalize(&[
        0.9, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1,
    ]);

    // -- BM25 ranking --
    let bm25_results = bm25_rank(&corpus, text_query);

    println!("=== BM25 results for \"{text_query}\" ===");
    for (doc_id, score) in &bm25_results {
        println!("  [{:2}] {:.4}  {}", doc_id, score, corpus[*doc_id].title);
    }

    // -- HNSW vector ranking --
    let k_neighbors = corpus.len();
    let hnsw_raw = index.search(&vector_query, k_neighbors, 50)?;

    // Convert to (usize, f64) with similarity score (1 - cosine distance).
    let hnsw_results: Vec<(usize, f64)> = hnsw_raw
        .iter()
        .map(|&(id, dist)| (id as usize, (1.0 - dist) as f64))
        .collect();

    println!("\n=== HNSW vector results ===");
    for (doc_id, sim) in &hnsw_results {
        println!("  [{:2}] {:.4}  {}", doc_id, sim, corpus[*doc_id].title);
    }

    // -- Reciprocal Rank Fusion (k=60, the standard constant) --
    let bm25_slice: Vec<(usize, f64)> = bm25_results.clone();
    let hnsw_slice: Vec<(usize, f64)> = hnsw_results.clone();
    let fused = reciprocal_rank_fusion(&[&bm25_slice, &hnsw_slice], 60.0);

    println!("\n=== Hybrid results (RRF, k=60) ===");
    for (doc_id, rrf_score) in &fused {
        // Show which lists contributed to each result.
        let bm25_rank = bm25_results
            .iter()
            .position(|(id, _)| id == doc_id)
            .map(|r| format!("#{}", r + 1));
        let hnsw_rank = hnsw_results
            .iter()
            .position(|(id, _)| id == doc_id)
            .map(|r| format!("#{}", r + 1));

        println!(
            "  [{:2}] {:.6}  {:20}  bm25={}  vec={}",
            doc_id,
            rrf_score,
            corpus[*doc_id].title,
            bm25_rank.unwrap_or_else(|| "-".into()),
            hnsw_rank.unwrap_or_else(|| "-".into()),
        );
    }

    println!("\nRRF promotes documents that rank well in BOTH text and vector search.");
    println!(
        "Top result: \"{}\" -- strong match on both lexical and semantic axes.",
        corpus[fused[0].0].title
    );

    Ok(())
}
