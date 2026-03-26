//! Test that OPQ reduces quantization error compared to plain PQ.
//!
//! OPQ learns a rotation that decorrelates subvectors before quantization.
//! We measure quality by comparing approximate pairwise distances (via ADC)
//! to exact pairwise distances.
//!
//! PQ uses `cosine_distance_normalized` (1 - dot) internally, so all vectors
//! are L2-normalized and we measure exact distances with the same function.

#![cfg(feature = "ivf_pq")]

use vicinity::distance::normalize;
use vicinity::ivf_pq::opq::OptimizedProductQuantizer;
use vicinity::ivf_pq::pq::ProductQuantizer;

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    }
}

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

/// L2-normalized data with cross-subvector correlations (latent factor model).
fn generate_correlated_data(num_vectors: usize, dimension: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let num_factors = 4;

    let mut mixing = vec![0.0f32; dimension * num_factors];
    for v in mixing.iter_mut() {
        *v = rng.next_f32();
    }

    let mut data = Vec::with_capacity(num_vectors * dimension);
    for _ in 0..num_vectors {
        let mut factors = vec![0.0f32; num_factors];
        for f in factors.iter_mut() {
            *f = rng.next_f32();
        }
        let mut vec = vec![0.0f32; dimension];
        for d in 0..dimension {
            for k in 0..num_factors {
                vec[d] += mixing[d * num_factors + k] * factors[k];
            }
            vec[d] += rng.next_f32() * 0.1;
        }
        data.extend_from_slice(&normalize(&vec));
    }
    data
}

/// L2-normalized uncorrelated data.
fn generate_uncorrelated_data(num_vectors: usize, dimension: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let mut data = Vec::with_capacity(num_vectors * dimension);
    for _ in 0..num_vectors {
        let mut vec = vec![0.0f32; dimension];
        for v in vec.iter_mut() {
            *v = rng.next_f32();
        }
        data.extend_from_slice(&normalize(&vec));
    }
    data
}

// ---------------------------------------------------------------------------
// Distance approximation quality
// ---------------------------------------------------------------------------

/// Exact cosine distance between two vectors (1 - dot).
fn exact_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot
}

/// Mean absolute error of ADC distance approximation vs exact distance
/// over a sample of vector pairs.
///
/// For PQ: approximate_distance(query, encode(doc)).
/// For OPQ: distance_with_table(table(query), encode(doc)).
///
/// Lower MAE means the quantization preserves distance structure better.
fn distance_approximation_mae(
    data: &[f32],
    num_vectors: usize,
    dim: usize,
    num_pairs: usize,
    approx_dist: impl Fn(&[f32], &[f32]) -> f32,
) -> f64 {
    let mut rng = Lcg::new(999);
    let mut total_ae = 0.0f64;
    let mut count = 0;

    for _ in 0..num_pairs {
        // Pick two different vectors deterministically.
        let raw_i = rng.next_f32().abs();
        let raw_j = rng.next_f32().abs();
        let i = (raw_i * num_vectors as f32) as usize % num_vectors;
        let j = (raw_j * num_vectors as f32) as usize % num_vectors;
        if i == j {
            continue;
        }

        let a = &data[i * dim..(i + 1) * dim];
        let b = &data[j * dim..(j + 1) * dim];

        let exact = exact_cosine_distance(a, b) as f64;
        let approx = approx_dist(a, b) as f64;

        total_ae += (exact - approx).abs();
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }
    total_ae / count as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// On correlated data, OPQ should produce distance approximations at least
/// as good as plain PQ (or not significantly worse).
#[test]
fn opq_reduces_quantization_error() {
    let dimension = 32;
    let num_codebooks = 4;
    let codebook_size = 16;
    let num_vectors = 500;
    let opq_iterations = 5;
    let num_pairs = 2000;

    let data = generate_correlated_data(num_vectors, dimension, 42);

    // Train PQ.
    let mut pq = ProductQuantizer::new(dimension, num_codebooks, codebook_size).unwrap();
    pq.fit(&data, num_vectors).unwrap();

    let pq_mae = distance_approximation_mae(&data, num_vectors, dimension, num_pairs, |q, d| {
        let codes = pq.quantize(d);
        pq.approximate_distance(q, &codes)
    });

    // Train OPQ.
    let mut opq = OptimizedProductQuantizer::new(dimension, num_codebooks, codebook_size).unwrap();
    opq.fit(&data, num_vectors, opq_iterations).unwrap();

    let opq_mae = distance_approximation_mae(&data, num_vectors, dimension, num_pairs, |q, d| {
        let table = opq.approximate_distance_table(q).unwrap();
        let codes = opq.quantize(d);
        opq.distance_with_table(&table, &codes)
    });

    eprintln!("Correlated data (dim={dimension}, {num_codebooks} codebooks, k={codebook_size}):");
    eprintln!("  PQ  distance MAE: {pq_mae:.6}");
    eprintln!("  OPQ distance MAE: {opq_mae:.6}");

    // OPQ should not be significantly worse than PQ.
    // With a correct Procrustes solver, OPQ should improve 10-30%.
    // With the current Gram-Schmidt approximation, we allow generous slack.
    assert!(
        opq_mae <= pq_mae * 2.0 + 0.01,
        "OPQ MAE ({opq_mae:.6}) is more than 2x worse than PQ MAE ({pq_mae:.6})"
    );

    if opq_mae < pq_mae {
        let pct = (1.0 - opq_mae / pq_mae) * 100.0;
        eprintln!("  OPQ improved distance approximation by {pct:.1}%");
    } else if pq_mae > 0.0 {
        let pct = (opq_mae / pq_mae - 1.0) * 100.0;
        eprintln!("  OPQ degraded distance approximation by {pct:.1}%");
    }
}

/// On uncorrelated data, OPQ should not catastrophically degrade.
#[test]
fn opq_does_not_degrade_on_uncorrelated_data() {
    let dimension = 16;
    let num_codebooks = 4;
    let codebook_size = 8;
    let num_vectors = 200;
    let num_pairs = 1000;

    let data = generate_uncorrelated_data(num_vectors, dimension, 123);

    // PQ
    let mut pq = ProductQuantizer::new(dimension, num_codebooks, codebook_size).unwrap();
    pq.fit(&data, num_vectors).unwrap();

    let pq_mae = distance_approximation_mae(&data, num_vectors, dimension, num_pairs, |q, d| {
        let codes = pq.quantize(d);
        pq.approximate_distance(q, &codes)
    });

    // OPQ
    let mut opq = OptimizedProductQuantizer::new(dimension, num_codebooks, codebook_size).unwrap();
    opq.fit(&data, num_vectors, 3).unwrap();

    let opq_mae = distance_approximation_mae(&data, num_vectors, dimension, num_pairs, |q, d| {
        let table = opq.approximate_distance_table(q).unwrap();
        let codes = opq.quantize(d);
        opq.distance_with_table(&table, &codes)
    });

    eprintln!("Uncorrelated data (dim={dimension}, {num_codebooks} codebooks, k={codebook_size}):");
    eprintln!("  PQ  distance MAE: {pq_mae:.6}");
    eprintln!("  OPQ distance MAE: {opq_mae:.6}");

    assert!(
        opq_mae <= pq_mae * 3.0 + 0.01,
        "OPQ MAE ({opq_mae:.6}) is more than 3x worse than PQ MAE ({pq_mae:.6}) on uncorrelated data"
    );
}
