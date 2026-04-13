//! Save an HNSW index to disk and load it back.
//!
//! ```sh
//! cargo run --example 06_save_and_load --features hnsw,serde
//! ```

use vicinity::hnsw::HNSWIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim = 32;
    let mut index = HNSWIndex::builder(dim)
        .m(16)
        .ef_search(50)
        .auto_normalize(true)
        .build()?;

    // Add some vectors.
    for i in 0..100u32 {
        let v: Vec<f32> = (0..dim)
            .map(|d| ((i * 7 + d as u32) as f32).sin())
            .collect();
        index.add_slice(i, &v)?;
    }
    index.build()?;

    // Save to a temporary file.
    let path = std::env::temp_dir().join("vicinity_demo_index.json");
    index.save_to_file(&path)?;
    println!("Saved index to {}", path.display());

    // Load it back.
    let loaded = HNSWIndex::load_from_file(&path)?;

    // Verify search still works.
    let query: Vec<f32> = (0..dim).map(|d| (d as f32).sin()).collect();
    let results = loaded.search(&query, 5, 50)?;
    println!("Top 5 results: {:?}", results);

    // Clean up.
    std::fs::remove_file(&path)?;

    Ok(())
}
