//! Vector quantization: compress vectors while preserving distance.
//!
//! # Which Method Should I Use?
//!
//! | Situation | Method | Compression | Feature |
//! |-----------|--------|-------------|---------|
//! | **Best accuracy** | RaBitQ 4-bit | 8x | `rabitq` |
//! | **Best compression** | Ternary | 20x | `saq` |
//! | **No training data** | Binary (sign) | 32x | `saq` |
//! | **Multi-dimensional** | Product Quantization | 32x | See `ivf_pq` |
//!
//! # Feature Flags
//!
//! ```toml
//! vicinity = { version = "0.1", features = ["rabitq"] }  # RaBitQ
//! vicinity = { version = "0.1", features = ["saq"] }     # Ternary/Binary
//! ```
//!
//! # The Problem: Memory at Scale
//!
//! ```text
//! 1B vectors × 768 dims × 4 bytes = 3 TB
//! ```
//!
//! Quantization compresses vectors while preserving distance accuracy:
//!
//! | Method | Bits/dim | Compression | Recall@10 |
//! |--------|----------|-------------|-----------|
//! | float32 | 32 | 1x | 100% |
//! | **RaBitQ 4-bit** | 4 | 8x | 95%+ |
//! | **Ternary** | 1.58 | 20x | 85%+ |
//! | Binary | 1 | 32x | 75%+ |
//!
//! ## Scalar vs Vector Quantization
//!
//! **Scalar quantization** (this module): Quantize each dimension independently.
//! Simple but loses correlations between dimensions.
//!
//! **Vector quantization** (see `ivf_pq`): Learn codebooks that capture
//! multi-dimensional structure. Better accuracy, more complex.
//!
//! ## RaBitQ: The Modern Approach
//!
//! [Gao et al. 2024](https://arxiv.org/abs/2409.09913) introduces randomized
//! binary quantization with corrective factors.
//!
//! **Key insight**: Random rotation before quantization spreads information
//! evenly across dimensions. Then:
//!
//! 1. **Sign bit**: Direction of each rotated dimension
//! 2. **Extended bits**: Magnitude refinement (optional)
//! 3. **Corrective factors**: Learned f_add, f_scale for distance estimation
//!
//! ```text
//! Original: [0.2, -0.7, 0.1, 0.9]
//!    ↓ random rotation
//! Rotated: [0.5, 0.3, -0.6, 0.4]
//!    ↓ quantize
//! Codes:   [+, +, -, +]  (1-bit: signs only)
//!    or    [+2, +1, -2, +1]  (4-bit: with magnitude)
//! ```
//!
//! **Why random rotation?** It makes the quantization error independent
//! across dimensions, which allows accurate distance estimation via
//! expectation formulas.
//!
//! ## Ternary Quantization
//!
//! Ultra-aggressive: map each dimension to {-1, 0, +1}.
//!
//! - **1.58 bits/dim** (log₂(3))
//! - **Hamming-like distance** with popcount operations
//! - Best for high-dimensional embeddings where redundancy is high
//!
//! ## Distance Computation
//!
//! The magic of good quantization: distance in quantized space approximates
//! distance in original space.
//!
//! **Asymmetric**: Query is exact, database is quantized
//! ```text
//! d(q, x) ≈ d(q, Q(x)) + correction
//! ```
//!
//! **Symmetric**: Both quantized (faster but less accurate)
//! ```text
//! d(q, x) ≈ d(Q(q), Q(x))
//! ```
//!
//! ## Usage
//!
//! Requires `features = ["rabitq"]`:
//!
//! ```ignore
//! use vicinity::quantization::rabitq::{RaBitQConfig, RaBitQQuantizer};
//!
//! let config = RaBitQConfig::bits4();  // 4-bit quantization
//! let mut quantizer = RaBitQQuantizer::with_config(768, 42, config)?;
//!
//! // Train on sample vectors
//! quantizer.fit(&sample_vectors)?;
//!
//! // Quantize database
//! let codes: Vec<_> = vectors.iter()
//!     .map(|v| quantizer.quantize(v))
//!     .collect();
//!
//! // Distance estimation
//! let dist = quantizer.approximate_distance(&query, &codes[0])?;
//! ```
//!
//! ## References
//!
//! - Gao et al. (2024). "RaBitQ: Quantizing High-Dimensional Vectors with
//!   Randomized Binary Quantization."
//! - See also: Product Quantization (`ivf_pq`), ScaNN (`scann`).

#![allow(dead_code)]

#[cfg(feature = "rabitq")]
pub use qntz::rabitq;

#[cfg(feature = "rabitq")]
pub mod simd_ops;

#[cfg(feature = "saq")]
pub mod saq;

#[cfg(feature = "saq")]
pub use qntz::ternary;

// Re-exports for convenience
#[cfg(feature = "saq")]
pub use ternary::{
    asymmetric_cosine_distance, asymmetric_inner_product, ternary_cosine_similarity,
    ternary_hamming, ternary_inner_product, TernaryConfig, TernaryQuantizer, TernaryVector,
};

#[cfg(test)]
mod tests {
    #[cfg(feature = "rabitq")]
    mod simd_tests {
        use crate::quantization::simd_ops::{
            hamming_distance, pack_binary_fast, unpack_binary_fast,
        };

        #[test]
        fn hamming_identical_packed() {
            let a = vec![0xAB_u8; 16];
            assert_eq!(hamming_distance(&a, &a), 0);
        }

        #[test]
        fn hamming_all_different() {
            let a = vec![0x00_u8; 4];
            let b = vec![0xFF_u8; 4];
            assert_eq!(hamming_distance(&a, &b), 32);
        }

        #[test]
        fn hamming_single_bit_diff() {
            let a = vec![0b0000_0001_u8];
            let b = vec![0b0000_0000_u8];
            assert_eq!(hamming_distance(&a, &b), 1);
        }

        #[test]
        fn hamming_empty_inputs() {
            assert_eq!(hamming_distance(&[], &[]), 0);
        }

        #[test]
        fn hamming_mismatched_lengths_uses_min() {
            let a = vec![0xFF_u8; 2]; // 16 bits
            let b = vec![0x00_u8; 1]; // 8 bits
                                      // Only compares first byte
            assert_eq!(hamming_distance(&a, &b), 8);
        }

        /// Verify pack/unpack roundtrip matches a scalar reference.
        #[test]
        fn pack_unpack_matches_scalar_reference() {
            let codes: Vec<u8> = vec![1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1];
            let packed_len = codes.len().div_ceil(8);
            let mut packed = vec![0u8; packed_len];
            pack_binary_fast(&codes, &mut packed).unwrap();

            // Scalar reference: manually compute expected packed bytes
            // Byte 0: bits 0-7 = [1,0,0,1,1,1,0,1] = 0b10111001 = 0xB9
            assert_eq!(packed[0], 0b10111001);
            // Byte 1: bits 8-10 = [0,1,1] + padding = 0b00000110 = 0x06
            assert_eq!(packed[1], 0b00000110);

            let mut unpacked = vec![0u8; codes.len()];
            unpack_binary_fast(&packed, &mut unpacked, codes.len()).unwrap();
            assert_eq!(unpacked, codes);
        }

        /// Hamming distance computed via packed codes matches scalar popcount.
        #[test]
        fn hamming_matches_scalar_reference() {
            let a_codes: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1];
            let b_codes: Vec<u8> = vec![0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1];

            // Scalar reference: count positions where a != b
            let scalar_hamming: u32 = a_codes
                .iter()
                .zip(b_codes.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();

            let mut a_packed = vec![0u8; 2];
            let mut b_packed = vec![0u8; 2];
            pack_binary_fast(&a_codes, &mut a_packed).unwrap();
            pack_binary_fast(&b_codes, &mut b_packed).unwrap();

            let simd_hamming = hamming_distance(&a_packed, &b_packed);
            assert_eq!(
                simd_hamming, scalar_hamming,
                "packed hamming {} != scalar reference {}",
                simd_hamming, scalar_hamming
            );
        }
    }
}
