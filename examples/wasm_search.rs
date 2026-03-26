//! WASM vector search with vicinity.
//!
//! vicinity compiles to `wasm32-unknown-unknown` with default features.
//! This example runs natively but the same code works in the browser.
//!
//! ## Build for WASM
//!
//! ```sh
//! # Install wasm-pack (if not already)
//! cargo install wasm-pack
//!
//! # Check that it compiles for WASM
//! cargo check --target wasm32-unknown-unknown --features hnsw
//!
//! # Build a WASM library (requires wasm-bindgen in dependencies)
//! wasm-pack build --target web --features hnsw
//! ```
//!
//! ## Use from JavaScript
//!
//! Wrap the index in a `wasm-bindgen` struct to expose it to JS:
//!
//! ```js
//! import init, { VectorIndex } from './pkg/vicinity.js';
//!
//! await init();
//!
//! const index = new VectorIndex(128, 16, 32);
//! index.add(0, new Float32Array([0.1, 0.2, ...]));
//! index.build();
//! const results = index.search(query, 5, 50);
//! ```
//!
//! ## What works in WASM
//!
//! - HNSW index construction and search
//! - All distance functions (cosine, euclidean, dot product)
//! - `innr` crate's portable distance kernels
//!
//! ## What does not work in WASM
//!
//! - `persistence` feature (file I/O, mmap)
//! - `simsimd` feature (native SIMD intrinsics via C)
//! - `parallel` feature (rayon thread pool)
//!
//! ## Performance notes
//!
//! - For 100K vectors at dim=128, expect sub-10ms search times in WASM
//! - Memory: ~50MB for 100K f32 vectors at dim=128
//! - WASM SIMD (128-bit) is available in all major browsers since 2021
//!
//! ## Run natively
//!
//! ```sh
//! cargo run --example wasm_search --features hnsw
//! ```

use vicinity::hnsw::HNSWIndex;

fn main() -> vicinity::Result<()> {
    let dim = 32;
    let n = 200;
    let k = 5;
    let ef = 50;

    // Generate sample vectors (deterministic, no file I/O).
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| normalize(&pseudo_random_vec(dim, i))).collect();

    // Build an HNSW index.
    let mut index = HNSWIndex::new(dim, 16, 32)?;
    for (id, vec) in vectors.iter().enumerate() {
        index.add(id as u32, vec.clone())?;
    }
    index.build()?;

    // Search for the 5 nearest neighbors of vector 0.
    let query = &vectors[0];
    let results = index.search(query, k, ef)?;

    println!("HNSW search (n={n}, dim={dim}, k={k}, ef={ef})");
    println!("query: vector 0");
    println!();
    for (id, distance) in &results {
        println!("  id={id:3}  distance={distance:.6}");
    }

    // Sanity: the query itself should be the nearest neighbor.
    assert_eq!(results[0].0, 0, "expected vector 0 as its own nearest neighbor");
    println!("\npassed: vector 0 is its own nearest neighbor");

    Ok(())
}

/// Deterministic pseudo-random vector (no RNG crate needed).
fn pseudo_random_vec(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 31 + i * 17) as f32 * 0.001).sin())
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}
