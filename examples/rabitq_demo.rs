//! RaBitQ: Randomized Binary Quantization
//!
//! RaBitQ achieves state-of-the-art compression without training data.
//! The key insight: random orthogonal rotation makes quantization error predictable.
//!
//! # Why Random Rotation Works
//!
//! In high dimensions, random orthogonal rotations have a remarkable property:
//! they distribute energy uniformly across dimensions. After rotation:
//! - Coordinates become nearly independent
//! - Each follows approximately Gaussian distribution
//! - Sign bit captures ~63% of variance (proved by concentration inequalities)
//!
//! This means we can quantize without seeing the data first!
//!
//! # Comparison with Other Methods
//!
//! | Method       | Bits/dim | Training | Accuracy | Use Case           |
//! |--------------|----------|----------|----------|--------------------|
//! | RaBitQ       | 1-8      | None     | Good     | Streaming, no data |
//! | Product Quant| 4-8      | Required | Better   | Known distribution |
//! | Ternary      | 1.58     | None     | Lower    | Extreme compress   |
//! | SQ (scalar)  | 8        | None     | High     | Simple baseline    |
//!
//! ```bash
//! cargo run --example rabitq_demo --release --features rabitq
//! ```

use vicinity::quantization::rabitq::{RaBitQConfig, RaBitQQuantizer};

fn main() -> vicinity::Result<()> {
    println!("RaBitQ: Randomized Binary Quantization");
    println!("=======================================\n");

    demo_basic_quantization()?;
    demo_compression_accuracy_tradeoff()?;
    demo_distance_estimation()?;
    demo_when_to_use()?;

    println!("Done!");
    Ok(())
}

fn demo_basic_quantization() -> vicinity::Result<()> {
    println!("1. Basic Quantization: Random Rotation + Binary Codes");
    println!("   ---------------------------------------------------\n");

    let dim = 128;
    let seed = 42;

    // Create quantizer with 4-bit codes
    let quantizer = RaBitQQuantizer::with_config(dim, seed, RaBitQConfig::bits4())?;

    // Generate a sample embedding
    let embedding: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.618).sin() * 0.5).collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let embedding: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

    println!(
        "   Original embedding: {} dimensions, ||x|| = {:.4}",
        dim, 1.0
    );

    // Quantize
    let quantized = quantizer.quantize(&embedding)?;

    println!("   Quantized representation:");
    println!(
        "     - Binary codes: {} bytes",
        quantized.binary_codes.len()
    );
    println!(
        "     - Extended codes: {} bytes",
        quantized.extended_codes.len()
    );
    println!(
        "     - Total: {} bytes ({:.1}x compression vs f32)",
        quantized.binary_codes.len() + quantized.extended_codes.len(),
        (dim * 4) as f32 / (quantized.binary_codes.len() + quantized.extended_codes.len()) as f32
    );
    println!(
        "     - Corrective factors: f_add={:.4}, f_rescale={:.4}",
        quantized.f_add, quantized.f_rescale
    );
    println!();

    Ok(())
}

fn demo_compression_accuracy_tradeoff() -> vicinity::Result<()> {
    println!("2. Compression vs Accuracy Trade-off");
    println!("   ----------------------------------\n");

    let dim = 768; // BERT-base dimension
    let n_vectors = 100;
    let seed = 42;

    // Generate test embeddings
    let embeddings: Vec<Vec<f32>> = (0..n_vectors)
        .map(|i| generate_embedding(dim, i as u64))
        .collect();

    println!("   Testing on {} vectors of dimension {}\n", n_vectors, dim);
    println!(
        "   {:>6} {:>12} {:>12} {:>12} {:>12}",
        "Bits", "Bytes/vec", "Compression", "Avg Error", "Max Error"
    );
    println!("   {}", "-".repeat(60));

    for bits in [1, 2, 4, 8] {
        let config = RaBitQConfig {
            total_bits: bits,
            t_const: None,
        };
        let quantizer = RaBitQQuantizer::with_config(dim, seed, config)?;

        let mut total_error = 0.0f32;
        let mut max_error = 0.0f32;
        let mut total_bytes = 0usize;

        for emb in &embeddings {
            let quantized = quantizer.quantize(emb)?;
            total_bytes += quantized.binary_codes.len() + quantized.extended_codes.len();

            // Estimate reconstruction error from corrective factors
            let error = quantized.f_error;
            total_error += error;
            max_error = max_error.max(error);
        }

        let avg_bytes = total_bytes / n_vectors;
        let compression = (dim * 4) as f32 / avg_bytes as f32;
        let avg_error = total_error / n_vectors as f32;

        println!(
            "   {:>6} {:>12} {:>11.1}x {:>12.4} {:>12.4}",
            bits, avg_bytes, compression, avg_error, max_error
        );
    }

    println!("\n   Key insight: 4-bit (8x compression) is the sweet spot for most applications.");
    println!();

    Ok(())
}

fn demo_distance_estimation() -> vicinity::Result<()> {
    println!("3. Distance Estimation with Corrective Factors");
    println!("   --------------------------------------------\n");

    let dim = 256;
    let seed = 42;
    let n_pairs = 50;

    let quantizer = RaBitQQuantizer::with_config(dim, seed, RaBitQConfig::bits4())?;

    // Generate similar and dissimilar pairs
    let mut similar_errors = Vec::new();
    let mut dissimilar_errors = Vec::new();

    for i in 0..n_pairs {
        // Similar pair: small perturbation
        let base = generate_embedding(dim, i as u64);
        let similar: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(j, &x)| x + 0.05 * ((j as f32 * 0.3).sin()))
            .collect();
        let similar = normalize(&similar);

        // Dissimilar pair: different seed
        let dissimilar = generate_embedding(dim, (i + 1000) as u64);

        // Exact distances (squared L2)
        let exact_similar = l2_distance_sqr(&base, &similar);
        let exact_dissimilar = l2_distance_sqr(&base, &dissimilar);

        // Quantized distances using approximate_l2_sqr
        let q_similar = quantizer.quantize(&similar)?;
        let q_dissimilar = quantizer.quantize(&dissimilar)?;

        let approx_similar = quantizer.approximate_l2_sqr(&base, &q_similar)?;
        let approx_dissimilar = quantizer.approximate_l2_sqr(&base, &q_dissimilar)?;

        similar_errors.push((exact_similar - approx_similar).abs() / exact_similar.max(0.001));
        dissimilar_errors
            .push((exact_dissimilar - approx_dissimilar).abs() / exact_dissimilar.max(0.001));
    }

    let avg_similar_err: f32 = similar_errors.iter().sum::<f32>() / n_pairs as f32;
    let avg_dissimilar_err: f32 = dissimilar_errors.iter().sum::<f32>() / n_pairs as f32;

    println!("   Distance estimation accuracy (relative error on L2^2):");
    println!("     Similar pairs:    {:.1}%", avg_similar_err * 100.0);
    println!("     Dissimilar pairs: {:.1}%", avg_dissimilar_err * 100.0);
    println!();
    println!("   The corrective factors (f_add, f_rescale) compensate for");
    println!("   systematic quantization bias, making distance estimates accurate.");
    println!();

    Ok(())
}

fn demo_when_to_use() -> vicinity::Result<()> {
    println!("4. When to Use RaBitQ");
    println!("   -------------------\n");

    println!("   Use RaBitQ when:");
    println!("     - No training data available (streaming, cold start)");
    println!("     - Memory is the bottleneck (edge deployment, mobile)");
    println!("     - L2 distance metric is acceptable");
    println!("     - 4-8x compression is sufficient");
    println!();

    println!("   Consider alternatives when:");
    println!("     - Inner product metric required -> use ScaNN or OPQ");
    println!("     - Have representative training data -> PQ may be better");
    println!("     - Need >8x compression -> use ternary (innr) or LSH");
    println!("     - Very low dimensions (<32) -> random rotation benefits vanish");
    println!();

    println!("   Integration with HNSW:");
    println!("     ```rust,ignore");
    println!("     // Build HNSW on full vectors for graph structure");
    println!("     let mut index = HNSWIndex::new(dim, 32, 64)?;");
    println!("     for (id, vec) in vectors.iter().enumerate() {{");
    println!("         index.add(id as u32, vec.clone())?;");
    println!("     }}");
    println!("     index.build()?;");
    println!("     ");
    println!("     // Store quantized vectors for memory efficiency");
    println!("     let quantizer = RaBitQQuantizer::new(dim, 42)?;");
    println!("     let quantized: Vec<_> = vectors.iter()");
    println!("         .map(|v| quantizer.quantize(v))");
    println!("         .collect::<Result<_, _>>()?;");
    println!("     ```");
    println!();

    println!("   See also:");
    println!("     - `innr/ternary_demo.rs`: 1.58-bit extreme compression");
    println!("     - `02_measure_recall.rs`: HNSW recall benchmarking");
    println!("     - `05_normalization_matters.rs`: L2-normalization contract");

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

fn l2_distance_sqr(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>()
}
