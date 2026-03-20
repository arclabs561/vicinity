//! Anisotropic vector quantization for SCANN.

use crate::simd;
use crate::RetrieveError;

/// Anisotropic vector quantizer.
///
/// Implements Product Quantization (PQ) on residuals, with support for
/// anisotropic loss scoring during search.
///
/// **Theory**:
/// ScaNN minimizes the anisotropic loss:
/// L(x, x̃) = ||x - x̃||² + h * ||<x - x̃, x>||²
///
/// This implementation currently performs training on standard residuals (x - c),
/// which is the first step. Full anisotropic training requires iterating
/// with weighted updates.
#[derive(Debug)]
pub struct AnisotropicQuantizer {
    dimension: usize,
    num_codebooks: usize,
    codebook_size: usize,
    seed: u64,
    // [codebook_idx][codeword_idx][subvector_dim]
    pub(crate) codebooks: Vec<Vec<Vec<f32>>>,
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

        if dimension % num_codebooks != 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be divisible by num_codebooks".into(),
            ));
        }

        Ok(Self {
            dimension,
            num_codebooks,
            codebook_size,
            seed,
            codebooks: Vec::new(),
        })
    }

    /// Train quantizer on residuals (x - centroid).
    ///
    /// The input `residuals` should be pre-computed:
    /// residual[i] = vector[i] - partition_centroid[assignment[i]]
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

        let subvector_dim = self.dimension / self.num_codebooks;
        self.codebooks = Vec::with_capacity(self.num_codebooks);

        for m in 0..self.num_codebooks {
            let start_dim = m * subvector_dim;
            let _end_dim = (m + 1) * subvector_dim;

            // Gather all subvectors for subspace m
            // TODO: In production, downsample if num_vectors is huge
            let mut subvectors: Vec<f32> = Vec::with_capacity(num_vectors * subvector_dim);

            for i in 0..num_vectors {
                let vec_start = i * self.dimension + start_dim;
                subvectors.extend_from_slice(&residuals[vec_start..vec_start + subvector_dim]);
            }

            // Train K-Means on this subspace
            let mut kmeans =
                crate::scann::partitioning::KMeans::new(subvector_dim, self.codebook_size)?
                    .with_seed(self.seed.wrapping_add(m as u64));
            kmeans.fit(&subvectors, num_vectors)?;

            // Store centroids as codewords
            // centroids() returns &[Vec<f32>], one Vec per cluster
            let centers = kmeans.centroids();
            let codewords: Vec<Vec<f32>> = centers.to_vec();
            self.codebooks.push(codewords);
        }

        Ok(())
    }

    /// Quantize a single residual vector.
    pub fn quantize(&self, residual: &[f32]) -> Vec<u8> {
        let subvector_dim = self.dimension / self.num_codebooks;
        let mut codes = Vec::with_capacity(self.num_codebooks);

        for m in 0..self.num_codebooks {
            let start_dim = m * subvector_dim;
            let sub = &residual[start_dim..start_dim + subvector_dim];

            // Find nearest codeword
            let mut best_idx = 0;
            let mut min_dist = f32::MAX;

            for (k, codeword) in self.codebooks[m].iter().enumerate() {
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

    /// Build Lookup Table (LUT) for a query.
    ///
    /// Returns a table of size [num_codebooks][codebook_size] containing distances.
    /// This allows O(M) distance computation per candidate during search.
    pub fn build_lut(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let subvector_dim = self.dimension / self.num_codebooks;
        let mut lut = Vec::with_capacity(self.num_codebooks);

        for m in 0..self.num_codebooks {
            let start_dim = m * subvector_dim;
            let query_sub = &query[start_dim..start_dim + subvector_dim];

            let mut sub_lut = Vec::with_capacity(self.codebook_size);
            for codeword in &self.codebooks[m] {
                // For MIPS: store dot product
                // For L2: store squared distance
                // Here we use dot product as ScaNN is MIPS-optimized
                let score = simd::dot(query_sub, codeword);
                sub_lut.push(score);
            }
            lut.push(sub_lut);
        }
        lut
    }
}

fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
