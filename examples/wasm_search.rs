//! WASM vector search with vicinity.
//!
//! Runs the same HNSW build/search path used by a browser wrapper.
//!
//! Check the target build with:
//! `cargo check --target wasm32-unknown-unknown --features hnsw`
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
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| normalize(&pseudo_random_vec(dim, i)))
        .collect();

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
    assert_eq!(
        results[0].0, 0,
        "expected vector 0 as its own nearest neighbor"
    );
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
