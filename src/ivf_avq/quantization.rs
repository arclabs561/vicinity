//! Anisotropic vector quantization for IVF-AVQ.

use crate::simd;
use crate::RetrieveError;

/// Anisotropic vector quantizer.
///
/// Implements Product Quantization (PQ) on residuals, with support for
/// anisotropic loss scoring during search.
///
/// Codebooks are stored in a flat contiguous buffer for cache-friendly access.
#[derive(Debug)]
pub struct AnisotropicQuantizer {
    dimension: usize,
    num_codebooks: usize,
    codebook_size: usize,
    subvector_dim: usize,
    seed: u64,
    /// Flat codebook storage: `[cb0_cw0..., cb0_cw1..., ..., cb1_cw0..., ...]`
    codebooks: Vec<f32>,
}

impl AnisotropicQuantizer {
    /// Create new quantizer.
    pub fn new(
        dimension: usize,
        num_codebooks: usize,
        codebook_size: usize,
        seed: u64,
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
            seed,
            codebooks: Vec::new(),
        })
    }

    /// Get a codeword slice from the flat codebook storage.
    #[inline]
    fn get_codeword(&self, codebook_idx: usize, code: usize) -> &[f32] {
        let offset = (codebook_idx * self.codebook_size + code) * self.subvector_dim;
        &self.codebooks[offset..offset + self.subvector_dim]
    }

    /// Train quantizer on residuals (x - centroid).
    pub fn fit_residuals(
        &mut self,
        residuals: &[f32],
        num_vectors: usize,
    ) -> Result<(), RetrieveError> {
        if residuals.len() != num_vectors * self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: residuals.len() / num_vectors,
                doc_dim: self.dimension,
            });
        }

        let mut flat_codebooks =
            Vec::with_capacity(self.num_codebooks * self.codebook_size * self.subvector_dim);
        let mut actual_codebook_size = self.codebook_size;

        for m in 0..self.num_codebooks {
            let start_dim = m * self.subvector_dim;

            // Gather all subvectors for subspace m
            let mut subvectors: Vec<f32> = Vec::with_capacity(num_vectors * self.subvector_dim);
            for i in 0..num_vectors {
                let vec_start = i * self.dimension + start_dim;
                subvectors.extend_from_slice(&residuals[vec_start..vec_start + self.subvector_dim]);
            }

            // Train K-Means on this subspace using L2 distance.
            // Residuals live in Euclidean space; cosine k-means would normalize away
            // magnitude, distorting the codebook and mismatching the quantize() L2 assignment.
            let mut kmeans = crate::partitioning::kmeans::KMeansEuclidean::new(
                self.subvector_dim,
                self.codebook_size,
            )?
            .with_seed(self.seed.wrapping_add(m as u64));
            kmeans.fit(&subvectors, num_vectors)?;

            let centroids = kmeans.centroids();
            if m == 0 {
                // k-means may produce fewer centroids than requested
                actual_codebook_size = centroids.len();
            }

            // Flatten centroids into contiguous buffer
            for codeword in centroids {
                flat_codebooks.extend_from_slice(codeword);
            }
        }

        self.codebook_size = actual_codebook_size;
        self.codebooks = flat_codebooks;
        Ok(())
    }

    /// Quantize a single residual vector.
    pub fn quantize(&self, residual: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_codebooks);

        for m in 0..self.num_codebooks {
            let start_dim = m * self.subvector_dim;
            let sub = &residual[start_dim..start_dim + self.subvector_dim];

            let mut best_idx = 0;
            let mut min_dist = f32::MAX;

            for k in 0..self.codebook_size {
                let codeword = self.get_codeword(m, k);
                let dist = squared_euclidean(sub, codeword);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = k;
                }
            }
            codes.push(best_idx as u8);
        }
        codes
    }

    /// Build flat Lookup Table (LUT) for a query.
    ///
    /// Returns a contiguous table of size `num_codebooks * codebook_size`.
    /// Access: `lut[codebook_idx * codebook_size + code]`.
    /// Flat layout eliminates pointer chasing in the inner search loop.
    pub fn build_lut_flat(&self, query: &[f32]) -> Vec<f32> {
        let total = self.num_codebooks * self.codebook_size;
        let mut lut = Vec::with_capacity(total);

        for m in 0..self.num_codebooks {
            let start_dim = m * self.subvector_dim;
            let query_sub = &query[start_dim..start_dim + self.subvector_dim];

            for k in 0..self.codebook_size {
                let codeword = self.get_codeword(m, k);
                lut.push(simd::dot(query_sub, codeword));
            }
        }
        lut
    }

    /// Codebook size (number of codewords per subquantizer).
    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    pub(crate) fn codebooks(&self) -> &[f32] {
        &self.codebooks
    }

    pub(crate) fn from_codebooks(
        dimension: usize,
        num_codebooks: usize,
        codebook_size: usize,
        seed: u64,
        codebooks: Vec<f32>,
    ) -> Result<Self, RetrieveError> {
        let quantizer = Self::new(dimension, num_codebooks, codebook_size, seed)?;
        let expected_len = num_codebooks
            .checked_mul(codebook_size)
            .and_then(|len| len.checked_mul(quantizer.subvector_dim))
            .ok_or_else(|| RetrieveError::FormatError("AVQ codebook length overflow".into()))?;
        if codebooks.len() != expected_len {
            return Err(RetrieveError::FormatError(format!(
                "AVQ codebook length mismatch: expected {}, got {}",
                expected_len,
                codebooks.len()
            )));
        }
        Ok(Self {
            codebooks,
            ..quantizer
        })
    }
}

fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::l2_distance_squared(a, b)
}
