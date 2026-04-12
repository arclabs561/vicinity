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
            return Err(RetrieveError::InvalidParameter(
                "all parameters must be greater than 0".into(),
            ));
        }

        if !dimension.is_multiple_of(num_codebooks) {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be divisible by num_codebooks".into(),
            ));
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
            let mut kmeans = KMeansEuclidean::new(self.subvector_dim, self.codebook_size)?;
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

        table.clear();
        table.reserve(self.num_codebooks * self.codebook_size);

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
}

/// Get vector from SoA storage.
#[inline]
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}
