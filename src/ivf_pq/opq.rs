//! Optimized Product Quantization (OPQ).
//!
//! OPQ improves PQ by learning a rotation matrix R that makes subvectors
//! more independent, reducing quantization error by 10-30%.
//!
//! # Algorithm
//!
//! 1. **Training** (iterative):
//!    - Initialize R as identity matrix
//!    - Repeat for `iterations`:
//!      a. Rotate training vectors: X' = X × R
//!      b. Train PQ codebooks on X'
//!      c. Compute residuals and update R via SVD
//!
//! 2. **Encoding**: Apply R to vector, then use standard PQ
//!
//! 3. **Search**: Apply R to query, compute distance tables
//!
//! # References
//!
//! - Ge et al. (2014). "Optimized Product Quantization"
//! - Norouzi & Fleet (2013). "Cartesian K-Means"

use super::pq::ProductQuantizer;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};

/// Optimized Product Quantizer.
///
/// Wraps a ProductQuantizer with a learned rotation matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedProductQuantizer {
    dimension: usize,
    num_codebooks: usize,
    codebook_size: usize,
    /// Rotation matrix (dimension × dimension), stored row-major.
    /// If None, behaves like standard PQ (identity rotation).
    rotation: Option<Vec<f32>>,
    /// Underlying product quantizer.
    pq: ProductQuantizer,
}

impl OptimizedProductQuantizer {
    /// Create a new OPQ instance.
    pub fn new(
        dimension: usize,
        num_codebooks: usize,
        codebook_size: usize,
    ) -> Result<Self, RetrieveError> {
        let pq = ProductQuantizer::new(dimension, num_codebooks, codebook_size)?;
        Ok(Self {
            dimension,
            num_codebooks,
            codebook_size,
            rotation: None, // Will be learned during fit()
            pq,
        })
    }

    /// Train OPQ on data.
    ///
    /// Uses alternating optimization:
    /// 1. Fix R, optimize codebooks (standard PQ training)
    /// 2. Fix codebooks, optimize R (closed-form via Procrustes)
    pub fn fit(
        &mut self,
        data: &[f32],
        num_vectors: usize,
        iterations: usize,
    ) -> Result<(), RetrieveError> {
        if num_vectors < self.codebook_size {
            return Err(RetrieveError::InvalidParameter(format!(
                "need at least {} training vectors, got {}",
                self.codebook_size, num_vectors
            )));
        }

        // Initialize with identity rotation
        let mut rotation = identity_matrix(self.dimension);
        let mut rotated_data = data.to_vec();

        for iter in 0..iterations {
            // Step 1: Rotate training data
            if iter > 0 {
                apply_rotation_batch(
                    data,
                    num_vectors,
                    self.dimension,
                    &rotation,
                    &mut rotated_data,
                );
            }

            // Step 2: Train PQ on rotated data
            self.pq.fit(&rotated_data, num_vectors)?;

            // Step 3: Compute optimal rotation using Procrustes analysis
            // R* = argmin_R ||XR - Q||_F where Q is quantized reconstruction
            //
            // This is solved via SVD: if X'Q = UΣV', then R = VU'
            rotation =
                compute_optimal_rotation(data, num_vectors, self.dimension, &self.pq, &rotation);
        }

        self.rotation = Some(rotation);
        Ok(())
    }

    /// Quantize a vector.
    ///
    /// Applies rotation before PQ encoding.
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        let rotated = self.rotate_vector(vector);
        self.pq.quantize(&rotated)
    }

    /// Compute ADC (Asymmetric Distance Computation) lookup table.
    ///
    /// Applies rotation to query before computing table.
    pub fn approximate_distance_table(&self, query: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        let rotated = self.rotate_vector(query);
        self.pq.compute_adc_table(&rotated)
    }

    /// Compute distance using precomputed table.
    #[inline(always)]
    pub fn distance_with_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        self.pq.distance_with_table(table, codes)
    }

    /// Apply rotation to a single vector.
    fn rotate_vector(&self, vector: &[f32]) -> Vec<f32> {
        match &self.rotation {
            Some(r) => matrix_vector_multiply(r, vector, self.dimension),
            None => vector.to_vec(),
        }
    }

    /// Get the rotation matrix (for inspection/debugging).
    pub fn rotation_matrix(&self) -> Option<&[f32]> {
        self.rotation.as_deref()
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Create identity matrix (d × d), stored row-major.
fn identity_matrix(d: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// Matrix-vector multiply: result = M × v
/// M is d×d row-major, v is d-dimensional.
fn matrix_vector_multiply(m: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; d];
    for (i, val) in result.iter_mut().enumerate() {
        let mut sum = 0.0;
        let row_start = i * d;
        for j in 0..d {
            sum += m[row_start + j] * v[j];
        }
        *val = sum;
    }
    result
}

/// Apply rotation to a batch of vectors.
fn apply_rotation_batch(
    data: &[f32],
    num_vectors: usize,
    dimension: usize,
    rotation: &[f32],
    output: &mut [f32],
) {
    for i in 0..num_vectors {
        let src_start = i * dimension;
        let src = &data[src_start..src_start + dimension];
        let dst_start = i * dimension;
        let rotated = matrix_vector_multiply(rotation, src, dimension);
        output[dst_start..dst_start + dimension].copy_from_slice(&rotated);
    }
}

/// Compute optimal rotation matrix using SVD-based Procrustes analysis.
///
/// Given original vectors X and their PQ reconstructions Q,
/// find orthogonal R minimizing ||XR - Q||_F.
///
/// Solution (FAISS algorithm):
///   M = Q^T X   (d x d cross-covariance)
///   M = U S V^T  (SVD)
///   R = U V^T    (optimal orthogonal matrix)
fn compute_optimal_rotation(
    data: &[f32],
    num_vectors: usize,
    dimension: usize,
    pq: &ProductQuantizer,
    current_rotation: &[f32],
) -> Vec<f32> {
    if dimension == 0 {
        return Vec::new();
    }

    let sample_size = num_vectors.min(5000);
    if sample_size == 0 {
        return identity_matrix(dimension);
    }

    // Compute cross-covariance matrix M = Q^T X (d x d)
    // M[i,j] = sum_k reconstruction[k,i] * original[k,j]
    let mut m = vec![0.0f32; dimension * dimension];

    for k in 0..sample_size {
        let src_start = k * dimension;
        let original = &data[src_start..src_start + dimension];

        // Apply current rotation
        let rotated = matrix_vector_multiply(current_rotation, original, dimension);

        // Get PQ reconstruction of the rotated vector
        let codes = pq.quantize(&rotated);
        let reconstruction = reconstruct_vector(pq, &codes, dimension);

        // Accumulate: M += reconstruction^T * original  (outer product)
        for i in 0..dimension {
            for j in 0..dimension {
                m[i * dimension + j] += reconstruction[i] * original[j];
            }
        }
    }

    // SVD via nalgebra: M = U S V^T, then R = U V^T
    let na_m = nalgebra::DMatrix::from_row_slice(dimension, dimension, &m);
    let svd = nalgebra::linalg::SVD::new(na_m, true, true);

    let u = match svd.u {
        Some(ref u) => u,
        None => return identity_matrix(dimension),
    };
    let vt = match svd.v_t {
        Some(ref vt) => vt,
        None => return identity_matrix(dimension),
    };

    let rotation = u * vt;

    // Extract to row-major Vec<f32>
    let mut result = vec![0.0f32; dimension * dimension];
    for i in 0..dimension {
        for j in 0..dimension {
            result[i * dimension + j] = rotation[(i, j)];
        }
    }
    result
}

/// Reconstruct vector from PQ codes.
fn reconstruct_vector(pq: &ProductQuantizer, codes: &[u8], dimension: usize) -> Vec<f32> {
    let codebooks = pq.codebooks();
    let num_codebooks = codebooks.len();
    let subvector_dim = dimension / num_codebooks;

    let mut result = vec![0.0f32; dimension];
    for (m, &code) in codes.iter().enumerate() {
        let codeword = &codebooks[m][code as usize];
        let start = m * subvector_dim;
        result[start..start + subvector_dim].copy_from_slice(codeword);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_rotation() {
        let opq = OptimizedProductQuantizer::new(8, 2, 256).unwrap();

        // Without training, should behave like standard PQ
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rotated = opq.rotate_vector(&v);
        assert_eq!(v, rotated);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        // 2x2 identity
        let m = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![3.0, 4.0];
        let result = matrix_vector_multiply(&m, &v, 2);
        assert_eq!(result, vec![3.0, 4.0]);

        // 2x2 rotation by 90 degrees
        let m = vec![0.0, -1.0, 1.0, 0.0];
        let v = vec![1.0, 0.0];
        let result = matrix_vector_multiply(&m, &v, 2);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_opq_training() {
        let dimension = 16;
        let num_codebooks = 4;
        let codebook_size = 8; // Small for testing
        let num_vectors = 100;

        // Generate random training data
        let mut data = Vec::with_capacity(num_vectors * dimension);
        for i in 0..num_vectors {
            for j in 0..dimension {
                data.push(((i * 7 + j * 3) % 100) as f32 / 100.0);
            }
        }

        let mut opq =
            OptimizedProductQuantizer::new(dimension, num_codebooks, codebook_size).unwrap();
        let result = opq.fit(&data, num_vectors, 2);
        assert!(result.is_ok());

        // Should have learned a rotation
        assert!(opq.rotation.is_some());

        // Rotation should be approximately orthonormal
        let r = opq.rotation.as_ref().unwrap();
        for i in 0..dimension {
            let row_start = i * dimension;
            let mut norm_sq = 0.0;
            for j in 0..dimension {
                norm_sq += r[row_start + j] * r[row_start + j];
            }
            assert!(
                (norm_sq - 1.0).abs() < 0.1,
                "Row {} norm = {}",
                i,
                norm_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_opq_quantize() {
        let dimension = 8;
        let num_codebooks = 2;
        let codebook_size = 4;
        let num_vectors = 50;

        // Training data
        let mut data = Vec::with_capacity(num_vectors * dimension);
        for i in 0..num_vectors {
            for j in 0..dimension {
                data.push(((i + j) % 10) as f32 / 10.0);
            }
        }

        let mut opq =
            OptimizedProductQuantizer::new(dimension, num_codebooks, codebook_size).unwrap();
        opq.fit(&data, num_vectors, 2).unwrap();

        // Quantize a vector
        let v = vec![0.5; dimension];
        let codes = opq.quantize(&v);
        assert_eq!(codes.len(), num_codebooks);
        for &c in &codes {
            assert!((c as usize) < codebook_size);
        }
    }
}
