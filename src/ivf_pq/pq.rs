//! Product Quantization (PQ) implementation.

use crate::partitioning::kmeans::KMeansEuclidean;
use crate::simd::l2_distance_squared;
use crate::RetrieveError;

use serde::{Deserialize, Serialize};

/// Product Quantizer.
///
/// Decomposes vectors into subvectors and quantizes each subvector independently.
/// Codebooks are stored in a flat contiguous buffer for cache-friendly access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    dimension: usize,
    num_codebooks: usize,
    codebook_size: usize,
    subvector_dim: usize,
    /// Flat codebook storage: `[cb0_cw0_d0..d_sub, cb0_cw1_d0..d_sub, ..., cb1_cw0_d0..d_sub, ...]`
    /// Total length: `num_codebooks * codebook_size * subvector_dim`.
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Create new product quantizer.
    pub fn new(
        dimension: usize,
        num_codebooks: usize,
        codebook_size: usize,
    ) -> Result<Self, RetrieveError> {
        if dimension == 0 || num_codebooks == 0 || codebook_size == 0 {
            return Err(RetrieveError::InvalidConfig {
                field: "dimension, num_codebooks, codebook_size",
                expected: "must all be greater than 0",
            });
        }

        if codebook_size > 256 {
            return Err(RetrieveError::InvalidConfig {
                field: "codebook_size",
                expected: "must be at most 256 because codes are stored as u8",
            });
        }

        if !dimension.is_multiple_of(num_codebooks) {
            return Err(RetrieveError::InvalidConfig {
                field: "dimension",
                expected: "must be divisible by num_codebooks",
            });
        }

        Ok(Self {
            dimension,
            num_codebooks,
            codebook_size,
            subvector_dim: dimension / num_codebooks,
            codebooks: Vec::new(),
        })
    }

    /// Train quantizer on vectors.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        self.fit_with_seed(vectors, num_vectors, None)
    }

    pub(crate) fn fit_with_seed(
        &mut self,
        vectors: &[f32],
        num_vectors: usize,
        seed: Option<u64>,
    ) -> Result<(), RetrieveError> {
        self.fit_with_seed_and_max_iter(vectors, num_vectors, seed, 100)
    }

    pub(crate) fn fit_with_seed_and_max_iter(
        &mut self,
        vectors: &[f32],
        num_vectors: usize,
        seed: Option<u64>,
        max_iter: usize,
    ) -> Result<(), RetrieveError> {
        let mut flat_codebooks =
            Vec::with_capacity(self.num_codebooks * self.codebook_size * self.subvector_dim);
        let mut actual_codebook_size = self.codebook_size;

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;

            // Extract subvectors into flat buffer for k-means
            let mut flat = Vec::with_capacity(num_vectors * self.subvector_dim);
            for i in 0..num_vectors {
                let vec = get_vector(vectors, self.dimension, i);
                flat.extend_from_slice(&vec[start_dim..end_dim]);
            }

            // Train k-means on subvectors using L2 distance.
            // PQ codebooks live in residual space where magnitude matters;
            // cosine k-means would normalize codewords to unit vectors and
            // break the L2 ADC approximation.
            let mut kmeans = KMeansEuclidean::new(self.subvector_dim, self.codebook_size)?
                .with_max_iter(max_iter);
            if let Some(seed) = seed {
                kmeans = kmeans.with_seed(seed.wrapping_add(codebook_idx as u64));
            }
            kmeans.fit(&flat, num_vectors)?;

            let centroids = kmeans.centroids();
            if codebook_idx == 0 {
                // k-means may produce fewer centroids than requested
                actual_codebook_size = centroids.len();
            }

            // Flatten centroids into the contiguous buffer
            for codeword in centroids {
                flat_codebooks.extend_from_slice(codeword);
            }
        }

        self.codebook_size = actual_codebook_size;
        self.codebooks = flat_codebooks;

        Ok(())
    }

    /// Get a codeword slice from the flat codebook storage.
    #[inline]
    fn get_codeword(&self, codebook_idx: usize, code: usize) -> &[f32] {
        let offset = (codebook_idx * self.codebook_size + code) * self.subvector_dim;
        &self.codebooks[offset..offset + self.subvector_dim]
    }

    /// Quantize a vector.
    ///
    /// Returns codebook indices for each subvector.
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_codebooks);

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];

            // Find closest codeword
            let mut best_code = 0u8;
            let mut best_dist = f32::INFINITY;

            for code in 0..self.codebook_size {
                let codeword = self.get_codeword(codebook_idx, code);
                let dist = l2_distance_squared(subvector, codeword);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = code.min(255) as u8;
                }
            }

            codes.push(best_code);
        }

        codes
    }

    /// Compute approximate distance using quantized codes.
    ///
    /// Uses lookup tables for fast computation.
    pub fn approximate_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut total_dist = 0.0;

        for (codebook_idx, &code) in codes.iter().enumerate() {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let query_subvector = &query[start_dim..end_dim];
            let codeword = self.get_codeword(codebook_idx, code as usize);

            total_dist += l2_distance_squared(query_subvector, codeword);
        }

        total_dist
    }

    /// Compute ADC (Asymmetric Distance Computation) lookup table.
    ///
    /// Precomputes distances from query subvectors to all codebook centroids.
    /// Returns a flat table of size `num_codebooks * codebook_size`.
    ///
    /// Table layout: `[codebook_0_codeword_0, codebook_0_codeword_1, ..., codebook_1_codeword_0, ...]`
    pub fn compute_adc_table(&self, query: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        let mut table = Vec::with_capacity(self.num_codebooks * self.codebook_size);
        self.compute_adc_table_into(query, &mut table)?;
        Ok(table)
    }

    /// Compute ADC table into a caller-provided buffer (avoids allocation per cluster).
    ///
    /// The buffer is cleared and filled with `num_codebooks * codebook_size` entries.
    pub fn compute_adc_table_into(
        &self,
        query: &[f32],
        table: &mut Vec<f32>,
    ) -> Result<(), RetrieveError> {
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let table_len = self.num_codebooks * self.codebook_size;
        table.clear();

        if self.subvector_dim == 1 {
            table.resize(table_len, 0.0);
            for (codebook_idx, (&q, table_chunk)) in query
                .iter()
                .take(self.num_codebooks)
                .zip(table.chunks_exact_mut(self.codebook_size))
                .enumerate()
            {
                let codebook_offset = codebook_idx * self.codebook_size;
                let codebook =
                    &self.codebooks[codebook_offset..codebook_offset + self.codebook_size];
                for (&codeword, out) in codebook.iter().zip(table_chunk) {
                    let diff = q - codeword;
                    *out = diff * diff;
                }
            }
            return Ok(());
        }

        table.reserve(table_len);

        if self.subvector_dim <= 8 {
            for codebook_idx in 0..self.num_codebooks {
                let start_dim = codebook_idx * self.subvector_dim;
                let end_dim = start_dim + self.subvector_dim;
                let query_subvector = &query[start_dim..end_dim];
                let codebook_offset = codebook_idx * self.codebook_size * self.subvector_dim;
                let codebook_end = codebook_offset + self.codebook_size * self.subvector_dim;
                let codebook = &self.codebooks[codebook_offset..codebook_end];

                for codeword in codebook.chunks_exact(self.subvector_dim) {
                    let mut dist = 0.0f32;
                    for dim in 0..self.subvector_dim {
                        let diff = query_subvector[dim] - codeword[dim];
                        dist += diff * diff;
                    }
                    table.push(dist);
                }
            }
            return Ok(());
        }

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let query_subvector = &query[start_dim..end_dim];

            for code in 0..self.codebook_size {
                let codeword = self.get_codeword(codebook_idx, code);
                let dist = l2_distance_squared(query_subvector, codeword);
                table.push(dist);
            }
        }

        Ok(())
    }

    /// Compute distance using ADC table.
    ///
    /// Very fast: only table lookups and additions.
    #[inline(always)]
    pub fn distance_with_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        let mut total_dist = 0.0;
        for (codebook_idx, &code) in codes.iter().enumerate() {
            let idx = codebook_idx * self.codebook_size + code as usize;
            total_dist += table[idx];
        }
        total_dist
    }

    /// Reconstruct a vector from PQ codes.
    ///
    /// Concatenates the codewords for each subvector.
    pub fn reconstruct(&self, codes: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dimension);
        for (m, &code) in codes.iter().enumerate() {
            result.extend_from_slice(self.get_codeword(m, code as usize));
        }
        result
    }

    /// Number of codebooks.
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Subvector dimension.
    pub fn subvector_dim(&self) -> usize {
        self.subvector_dim
    }

    /// Codebook size (number of codewords per codebook).
    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    pub(crate) fn owned_bytes(&self) -> usize {
        self.codebooks.capacity() * std::mem::size_of::<f32>()
    }
}

/// Get vector from SoA storage.
#[inline]
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn rejects_codebook_size_that_does_not_fit_u8_codes() {
        // Codes are stored as u8, so a codebook has at most 256 entries.
        assert!(ProductQuantizer::new(8, 2, 256).is_ok());
        assert!(matches!(
            ProductQuantizer::new(8, 2, 257),
            Err(RetrieveError::InvalidParameter(_))
        ));
    }

    #[test]
    fn adc_table_specializes_one_dimensional_subvectors() {
        let pq = ProductQuantizer {
            dimension: 2,
            num_codebooks: 2,
            codebook_size: 3,
            subvector_dim: 1,
            codebooks: vec![0.0, 1.0, 3.0, -1.0, 2.0, 4.0],
        };
        let mut table = Vec::new();

        pq.compute_adc_table_into(&[2.0, 1.0], &mut table)
            .expect("valid query");

        assert_eq!(table, vec![4.0, 1.0, 1.0, 4.0, 1.0, 9.0]);
    }

    #[test]
    fn adc_table_specializes_small_multidimensional_subvectors() {
        let pq = ProductQuantizer {
            dimension: 6,
            num_codebooks: 2,
            codebook_size: 2,
            subvector_dim: 3,
            codebooks: vec![
                0.0, 0.0, 0.0, //
                1.0, 1.0, 1.0, //
                2.0, 2.0, 2.0, //
                3.0, 3.0, 3.0,
            ],
        };
        let mut table = Vec::new();

        pq.compute_adc_table_into(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut table)
            .expect("valid query");

        assert_eq!(table, vec![14.0, 5.0, 29.0, 14.0]);
    }

    // ── Properties: ADC identity and quantization round-trip ─────────────────

    /// Deterministic pseudo-random value in `[-1, 1)`.
    fn next_value(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn random_vector(len: usize, state: &mut u64) -> Vec<f32> {
        (0..len).map(|_| next_value(state)).collect()
    }

    /// Train a quantizer on `4 * codebook_size` uniform random vectors, which
    /// is enough for k-means to fill the requested codebooks.
    fn fitted_pq(
        dimension: usize,
        num_codebooks: usize,
        codebook_size: usize,
        seed: u64,
    ) -> ProductQuantizer {
        let training_count = (codebook_size * 4).max(16);
        let mut state = seed | 1;
        let training = random_vector(training_count * dimension, &mut state);
        let mut pq = ProductQuantizer::new(dimension, num_codebooks, codebook_size)
            .expect("valid quantizer shape");
        pq.fit(&training, training_count)
            .expect("training on distinct uniform vectors");
        pq
    }

    fn random_codes(num_codebooks: usize, codebook_size: usize, state: &mut u64) -> Vec<u8> {
        (0..num_codebooks)
            .map(|_| {
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (*state % codebook_size as u64) as u8
            })
            .collect()
    }

    fn squared_l2(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let diff = f64::from(*x) - f64::from(*y);
                diff * diff
            })
            .sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Accumulating the ADC table over random codes must equal the squared
        /// L2 distance to `reconstruct(codes)` — the table is only a
        /// re-association of that sum, so any missing or double-counted
        /// subvector shows up as a large relative error. Covers the
        /// `subvector_dim == 1`, `<= 8` and general table kernels.
        #[test]
        fn adc_table_distance_equals_squared_l2_to_reconstruction(
            num_codebooks in 1usize..=4,
            subvector_dim in 1usize..=10,
            codebook_size in 4usize..=16,
            codebook_seed in any::<u64>(),
            query_seed in any::<u64>(),
            code_seed in any::<u64>(),
        ) {
            let dimension = num_codebooks * subvector_dim;
            let pq = fitted_pq(dimension, num_codebooks, codebook_size, codebook_seed);

            let mut state = query_seed | 1;
            let query = random_vector(dimension, &mut state);
            let mut state = code_seed | 1;
            let codes = random_codes(pq.num_codebooks(), pq.codebook_size(), &mut state);

            let table = pq.compute_adc_table(&query).unwrap();
            let accumulated = pq.distance_with_table(&table, &codes);

            let reconstructed = pq.reconstruct(&codes);
            prop_assert_eq!(reconstructed.len(), dimension);
            let exact = squared_l2(&query, &reconstructed);

            // Accumulation happens in f32, so allow a relative error of 1e-4
            // (with a 1.0 floor for the O(1)-magnitude distances produced by
            // unit-scale inputs).
            let tolerance = 1e-4 * exact.abs().max(1.0);
            prop_assert!(
                (f64::from(accumulated) - exact).abs() <= tolerance,
                "ADC table sum {accumulated} != ||q - reconstruct(codes)||^2 {exact} \
                 (m={}, subvector_dim={}, codebook_size={})",
                num_codebooks,
                subvector_dim,
                pq.codebook_size()
            );

            // The direct (non-table) path must agree with the table path.
            let direct = pq.approximate_distance(&query, &codes);
            prop_assert!(
                (f64::from(direct) - f64::from(accumulated)).abs() <= tolerance,
                "approximate_distance {direct} disagreed with the table sum {accumulated}"
            );
        }

        /// Encoding picks the nearest codeword: for every subvector, the
        /// round-trip error is no larger than the error against *any* codeword
        /// in that codebook. This is the "bounded by codebook resolution"
        /// invariant, and it fails if `quantize` returns a non-argmin code (or
        /// reads the wrong codebook).
        #[test]
        fn round_trip_error_never_exceeds_best_codeword_error(
            num_codebooks in 1usize..=4,
            subvector_dim in 1usize..=10,
            codebook_size in 4usize..=16,
            codebook_seed in any::<u64>(),
            input_seed in any::<u64>(),
        ) {
            let dimension = num_codebooks * subvector_dim;
            let pq = fitted_pq(dimension, num_codebooks, codebook_size, codebook_seed);

            let mut state = input_seed | 1;
            let input = random_vector(dimension, &mut state);
            let codes = pq.quantize(&input);
            let reconstructed = pq.reconstruct(&codes);

            for codebook in 0..pq.num_codebooks() {
                let start = codebook * pq.subvector_dim();
                let end = start + pq.subvector_dim();
                let subvector = &input[start..end];
                let observed = squared_l2(subvector, &reconstructed[start..end]);

                for code in 0..pq.codebook_size() {
                    let mut probe = vec![0u8; pq.num_codebooks()];
                    probe[codebook] = code as u8;
                    let codeword = pq.reconstruct(&probe);
                    let best_available = squared_l2(subvector, &codeword[start..end]);
                    prop_assert!(
                        observed <= best_available * (1.0 + 1e-5) + 1e-6,
                        "codebook {}: round-trip error {} exceeds the error {} of \
                         codeword {} (subvector_dim={})",
                        codebook,
                        observed,
                        best_available,
                        code,
                        pq.subvector_dim()
                    );
                }
            }
        }

        /// Decoding lands exactly on a codebook point, so re-encoding a decoded
        /// vector reproduces the same reconstruction: encode → decode is
        /// idempotent at the codebook resolution.
        #[test]
        fn reconstruction_is_a_fixed_point_of_quantize(
            num_codebooks in 1usize..=4,
            subvector_dim in 1usize..=10,
            codebook_size in 4usize..=16,
            codebook_seed in any::<u64>(),
            code_seed in any::<u64>(),
        ) {
            let dimension = num_codebooks * subvector_dim;
            let pq = fitted_pq(dimension, num_codebooks, codebook_size, codebook_seed);

            let mut state = code_seed | 1;
            let codes = random_codes(pq.num_codebooks(), pq.codebook_size(), &mut state);
            let first = pq.reconstruct(&codes);
            let second = pq.reconstruct(&pq.quantize(&first));
            prop_assert_eq!(
                &second,
                &first,
                "re-encoding a decoded vector changed it (m={}, subvector_dim={})",
                num_codebooks,
                subvector_dim
            );
        }

        /// Finite inputs always produce in-range codes and a finite decode:
        /// no NaN or infinite reconstruction, even for values far outside the
        /// training range.
        #[test]
        fn finite_inputs_encode_to_finite_decodes(
            num_codebooks in 1usize..=4,
            subvector_dim in 1usize..=10,
            codebook_size in 4usize..=16,
            codebook_seed in any::<u64>(),
            components in proptest::collection::vec(-1e30f32..1e30f32, 1..=40),
        ) {
            let dimension = num_codebooks * subvector_dim;
            let pq = fitted_pq(dimension, num_codebooks, codebook_size, codebook_seed);

            // Tile the generated components up to the quantizer's dimension.
            let input: Vec<f32> = (0..dimension)
                .map(|i| components[i % components.len()])
                .collect();
            prop_assume!(input.iter().all(|value| value.is_finite()));

            let codes = pq.quantize(&input);
            prop_assert_eq!(codes.len(), pq.num_codebooks());
            for &code in &codes {
                prop_assert!(
                    (code as usize) < pq.codebook_size(),
                    "code {} is outside the codebook of size {}",
                    code,
                    pq.codebook_size()
                );
            }

            let reconstructed = pq.reconstruct(&codes);
            prop_assert_eq!(reconstructed.len(), dimension);
            prop_assert!(
                reconstructed.iter().all(|value| value.is_finite()),
                "finite input decoded to a non-finite vector: {:?}",
                reconstructed
            );
        }
    }
}
