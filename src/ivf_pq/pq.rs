//! Product Quantization (PQ) implementation.

use crate::simd::l2_distance_squared;
use crate::partitioning::kmeans::KMeans;
use crate::RetrieveError;

use serde::{Deserialize, Serialize};

/// Product Quantizer.
///
/// Decomposes vectors into subvectors and quantizes each subvector independently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    dimension: usize,
    num_codebooks: usize,
    codebook_size: usize,
    subvector_dim: usize,
    codebooks: Vec<Vec<Vec<f32>>>, // [codebook][codeword][dimension]
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

        if dimension % num_codebooks != 0 {
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
        // Train codebook for each subvector
        self.codebooks = Vec::new();

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;

            // Extract subvectors
            let mut subvectors = Vec::new();
            for i in 0..num_vectors {
                let vec = get_vector(vectors, self.dimension, i);
                subvectors.push(vec[start_dim..end_dim].to_vec());
            }

            // Train k-means on subvectors
            let mut kmeans = KMeans::new(self.subvector_dim, self.codebook_size)?;

            // Flatten subvectors for k-means (SoA format)
            let mut flat = Vec::with_capacity(num_vectors * self.subvector_dim);
            for subvec in &subvectors {
                flat.extend_from_slice(subvec);
            }
            kmeans.fit(&flat, num_vectors)?;

            self.codebooks.push(kmeans.centroids().to_vec());
        }

        // When num_vectors < codebook_size, k-means produces fewer centroids.
        // Update codebook_size to the actual count so indexing stays consistent.
        if let Some(first) = self.codebooks.first() {
            self.codebook_size = first.len();
        }

        Ok(())
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

            for (code, codeword) in self.codebooks[codebook_idx].iter().enumerate() {
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
            let codeword = &self.codebooks[codebook_idx][code as usize];

            total_dist += l2_distance_squared(query_subvector, codeword);
        }

        total_dist
    }

    /// Compute ADC (Asymmetric Distance Computation) lookup table.
    ///
    /// Precomputes distances from query subvectors to all codebook centroids.
    /// Returns a flat table of size `num_codebooks * codebook_size`.
    ///
    /// Table layout: [codebook_0_codeword_0, codebook_0_codeword_1, ..., codebook_1_codeword_0, ...]
    pub fn compute_adc_table(&self, query: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let mut table = Vec::with_capacity(self.num_codebooks * self.codebook_size);

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let query_subvector = &query[start_dim..end_dim];

            for codeword in &self.codebooks[codebook_idx] {
                // Compute distance (typically squared Euclidean or dot product depending on metric)
                // For now, assuming cosine/dot product as used elsewhere
                let dist = l2_distance_squared(query_subvector, codeword);
                table.push(dist);
            }
        }

        Ok(table)
    }

    /// Compute distance using ADC table.
    ///
    /// Very fast: only table lookups and additions.
    #[inline(always)]
    pub fn distance_with_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        let mut total_dist = 0.0;
        for (codebook_idx, &code) in codes.iter().enumerate() {
            // Table index = codebook_offset + code
            let idx = codebook_idx * self.codebook_size + code as usize;
            // Unsafe access for speed? bounds check should be hoisted or optimized
            // For now safe indexing
            total_dist += table[idx];
        }
        total_dist
    }

    /// Get codebooks (for testing/debugging).
    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }

    /// Get mutable codebooks (for online learning).
    pub fn codebooks_mut(&mut self) -> &mut [Vec<Vec<f32>>] {
        &mut self.codebooks
    }
}

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}
